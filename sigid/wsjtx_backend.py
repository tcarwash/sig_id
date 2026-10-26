"""Receive decoded events from a separately installed WSJT-X UDP server."""

from __future__ import annotations

from collections import deque
from datetime import datetime, time as day_time, timezone
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import tempfile
from threading import Event, Lock, Thread
from typing import Any

from scipy.signal import resample_poly

from .decoder_backend import DecoderBackendInfo, DecoderEvent, DecoderMode, DecoderProduct, DecoderRequest, DecoderResult
from .modes import MODE_REGISTRY


MAGIC = 0xADBCCBDA
MODE_DECODE = 2


def _decode_datagram(payload: bytes) -> dict[str, Any] | None:
    """Read the WSJT-X Qt UDP Decode message, ignoring newer trailing fields."""
    view = memoryview(payload)
    offset = 0

    def take(size: int) -> bytes:
        nonlocal offset
        if size < 0 or offset + size > len(view):
            raise ValueError("truncated WSJT-X UDP packet")
        value = bytes(view[offset:offset + size])
        offset += size
        return value

    def u32() -> int:
        return struct.unpack(">I", take(4))[0]

    def text() -> str:
        length = u32()
        if length == 0xFFFFFFFF:
            return ""
        if length > 65536:
            raise ValueError("WSJT-X string exceeds packet limit")
        return take(length).decode("utf-8", errors="replace")

    def boolean() -> bool:
        value = take(1)[0]
        if value not in (0, 1):
            raise ValueError("invalid WSJT-X boolean")
        return bool(value)

    try:
        if u32() != MAGIC:
            return None
        schema = u32()
        if schema not in (2, 3):
            return None
        message_type = u32()
        identifier = text()
        if message_type != MODE_DECODE:
            return {"type": message_type, "id": identifier, "schema": schema}
        is_new = boolean()
        milliseconds = u32()
        snr = struct.unpack(">i", take(4))[0]
        delta_time = struct.unpack(">d", take(8))[0]
        delta_frequency = u32()
        mode = text()
        message = text()
        low_confidence = boolean()
        off_air = boolean()
        return {
            "type": message_type, "id": identifier, "schema": schema,
            "new": is_new, "milliseconds_utc": milliseconds, "snr_db": snr,
            "delta_time_s": delta_time, "delta_frequency_hz": delta_frequency,
            "mode": mode, "message": message, "low_confidence": low_confidence,
            "off_air": off_air,
        }
    except (ValueError, UnicodeDecodeError, struct.error):
        return None


class WsjtxUdpBackend:
    """Read-only WSJT-X decode notifications; the user controls WSJT-X audio/radio."""

    backend_id = "wsjtx"

    def __init__(self, *, host: str = "127.0.0.1", port: int | None = None) -> None:
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("WSJT-X UDP listener must bind to loopback")
        if port is not None and not 1 <= port <= 65535:
            raise ValueError("WSJT-X UDP port must be from 1 to 65535")
        self.host, self.port = host, port
        configured_jt9 = os.environ.get("SIG_ID_JT9_PATH")
        self.jt9_path = configured_jt9 or shutil.which("jt9")
        if self.jt9_path and not Path(self.jt9_path).is_file():
            self.jt9_path = None
        self._stop = Event()
        self._lock = Lock()
        self._events: deque[DecoderEvent] = deque(maxlen=2000)
        self._version: str | None = None
        self._last_heartbeat: float | None = None
        self._socket: socket.socket | None = None
        self._thread: Thread | None = None
        self._error: str | None = None
        if port is not None:
            self._thread = Thread(target=self._listen, name="sig-id-wsjtx-udp", daemon=True)
            self._thread.start()

    @staticmethod
    def _native_modes() -> tuple[DecoderMode, ...]:
        return tuple(
            DecoderMode(
                mode_id=item["id"], display_name=item["display_name"],
                native_mode=str(item.get("backend_mode") or item["display_name"]),
                family=item["family"], method=item["method"],
                protocol=item["protocol"], variant=item["variant"],
                aliases=tuple(item.get("aliases", ())), selectable=False,
                # jt9 is the WSJT-X native command-line decoder. Keep clip
                # decoding opt-in only for the mode explicitly wired below.
                note="Live results arrive from the native WSJT-X application.",
            )
            for item in MODE_REGISTRY.values()
            if item.get("native_backend_id") == "wsjtx"
        )

    def decode(self, request: DecoderRequest) -> DecoderResult:
        """Run the installed WSJT-X jt9 binary on an isolated FT8 audio window."""
        if not self.jt9_path:
            raise RuntimeError("jt9 was not found; install WSJT-X or set SIG_ID_JT9_PATH")
        if request.native_mode.casefold() != "ft8":
            raise ValueError("isolated jt9 decoding is currently enabled for FT8 only")
        if request.sample_rate <= 0:
            raise ValueError("audio sample rate must be positive")
        duration = (request.sample_count or 0) / request.sample_rate
        if duration < 12.6:
            raise ValueError("FT8 needs about 12.6 seconds of audio; select a full 15-second receive interval")
        import wave

        with wave.open(str(request.audio_path), "rb") as source:
            if source.getnchannels() != 1 or source.getsampwidth() != 2:
                raise ValueError("jt9 input must be mono 16-bit PCM")
            source_rate = source.getframerate()
            pcm = source.readframes(source.getnframes())
        if source_rate != 12000:
            # WSJT-X's decoder expects 12 kHz SSB audio. Polyphase resampling
            # preserves the narrowband signal better than sample dropping.
            import numpy as np
            samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
            divisor = __import__("math").gcd(source_rate, 12000)
            converted = resample_poly(samples, 12000 // divisor, source_rate // divisor)
            pcm = np.clip(np.rint(converted), -32768, 32767).astype("<i2").tobytes()
        with tempfile.TemporaryDirectory(prefix="sig-id-jt9-") as temporary:
            root = Path(temporary)
            wav_path = root / "ft8-window.wav"
            with wave.open(str(wav_path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(12000)
                output.writeframes(pcm)
            command = [self.jt9_path, "-8", "-q", "-e", str(Path(self.jt9_path).parent),
                       "-a", str(root), "-t", str(root)]
            if request.carrier_hz is not None and 200 <= request.carrier_hz <= 4000:
                command.extend(("-f", str(int(round(request.carrier_hz)))))
            command.append(str(wav_path))
            completed = subprocess.run(
                command,
                capture_output=True, text=True, timeout=45, check=False,
            )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[-2000:]
            raise RuntimeError(f"jt9 exited with status {completed.returncode}: {detail}")
        output_text = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
        products = (DecoderProduct("text", output_text, metadata={"format": "jt9-native-output"}),) if output_text else ()
        return DecoderResult(
            backend_id=self.backend_id, application="WSJT-X jt9", mode_id=request.mode_id,
            native_mode="FT8", status="completed" if products else "completed_empty",
            products=products, note="Decoded by the installed WSJT-X jt9 executable; review output before labeling.",
        )

    def _listen(self) -> None:
        udp = socket.socket(socket.AF_INET6 if self.host == "::1" else socket.AF_INET, socket.SOCK_DGRAM)
        udp.settimeout(0.5)
        try:
            udp.bind((self.host, int(self.port)))
            self._socket = udp
            while not self._stop.is_set():
                try:
                    payload, _address = udp.recvfrom(65535)
                except socket.timeout:
                    continue
                except OSError:
                    break
                self._accept(payload)
        except OSError as exc:
            self._error = str(exc)
        finally:
            try:
                udp.close()
            except OSError:
                pass
            self._socket = None

    def _accept(self, payload: bytes) -> None:
        decoded = _decode_datagram(payload)
        if not decoded:
            return
        if decoded["type"] == 0:
            # Heartbeat layout after the common header: max schema, version, revision.
            try:
                view = memoryview(payload)
                offset = 12
                size = struct.unpack_from(">I", view, offset)[0]
                offset += 4 + size
                _max_schema = struct.unpack_from(">I", view, offset)[0]
                offset += 4
                version_size = struct.unpack_from(">I", view, offset)[0]
                offset += 4
                version = bytes(view[offset:offset + version_size]).decode("utf-8", errors="replace")
            except (ValueError, IndexError, struct.error):
                version = None
            with self._lock:
                self._last_heartbeat = __import__("time").monotonic()
                if version:
                    self._version = version
            return
        if decoded["type"] != MODE_DECODE:
            return
        mode_name = str(decoded["mode"])
        candidates = [item for item in self._native_modes() if item.native_mode.casefold() == mode_name.casefold()]
        if not candidates:
            return
        # When WSJT-X reports a base mode without a submode, use only an explicit
        # unsuffixed catalog label; never guess JT65A/B/C from a generic JT65 event.
        exact = next((item for item in candidates if item.display_name.casefold() == mode_name.casefold()), None)
        binding = exact or next((item for item in candidates if item.mode_id.endswith("-generic")), None)
        if binding is None:
            return
        millis = int(decoded["milliseconds_utc"])
        if not 0 <= millis < 86_400_000:
            return
        now = datetime.now(timezone.utc)
        event_day = now.date()
        stamp = datetime.combine(event_day, day_time(millis // 3_600_000,
            (millis // 60_000) % 60, (millis // 1000) % 60, (millis % 1000) * 1000), tzinfo=timezone.utc)
        # Decode times near midnight can belong to the prior UTC day.
        if (stamp - now).total_seconds() > 12 * 3600:
            from datetime import timedelta
            stamp -= timedelta(days=1)
        event = DecoderEvent(
            backend_id=self.backend_id,
            mode_id=binding.mode_id,
            native_mode=mode_name,
            observed_utc=stamp.isoformat(),
            products=(DecoderProduct("text", str(decoded["message"]), metadata={
                "new": decoded["new"], "off_air": decoded["off_air"],
                "low_confidence": decoded["low_confidence"],
            }),),
            frequency_hz=float(decoded["delta_frequency_hz"]),
            snr_db=float(decoded["snr_db"]),
            metadata={"wsjtx_id": decoded["id"], "delta_time_s": decoded["delta_time_s"],
                      "schema": decoded["schema"]},
        )
        with self._lock:
            self._events.append(event)

    def info(self) -> DecoderBackendInfo:
        import time
        with self._lock:
            connected = self._last_heartbeat is not None and time.monotonic() - self._last_heartbeat <= 45
            version = self._version
        jt9_available = self.jt9_path is not None
        if jt9_available:
            modes = tuple(DecoderMode(
                mode_id=mode.mode_id, display_name=mode.display_name, native_mode=mode.native_mode,
                family=mode.family, method=mode.method, protocol=mode.protocol, variant=mode.variant,
                aliases=mode.aliases, selectable=mode.display_name.casefold() == "ft8",
                note="Isolated clip decoding is available for FT8 via the installed jt9 executable."
                if mode.display_name.casefold() == "ft8" else mode.note,
            ) for mode in self._native_modes())
        else:
            modes = self._native_modes()
        note = self._error or (("jt9 clip decoding available; " if jt9_available else "") +
            ("receiving WSJT-X UDP decode events." if connected else
            (f"Set SIG_ID_WSJTX_UDP_PORT and configure WSJT-X Reporting → UDP Server to send to {self.host}:{self.port}."
             if self.port is not None else "Set SIG_ID_WSJTX_UDP_PORT and configure the WSJT-X UDP Server to enable live decode events.")))
        return DecoderBackendInfo(
            backend_id=self.backend_id, application="WSJT-X", connected=connected or jt9_available,
            version=version, input_kinds=("audio-window", "native-wsjt-x-live-audio"),
            product_kinds=("text", "metadata"), modes=modes, note=note,
        )

    def drain_events(self, *, limit: int = 100) -> list[DecoderEvent]:
        if not 1 <= limit <= 1000:
            raise ValueError("event limit must be from 1 to 1000")
        with self._lock:
            count = min(limit, len(self._events))
            return [self._events.popleft() for _ in range(count)]

    def close(self) -> None:
        self._stop.set()
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2)
