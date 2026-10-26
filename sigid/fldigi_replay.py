"""Isolated local Fldigi replay through a private PulseAudio null sink.

The worker never changes the operator's running Fldigi process or its sound
route. It starts a separate receive-only process with a copied profile and sends
only the requested WAV segment into a virtual sink.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import tempfile
from threading import Event, Lock
import time
from typing import Any
import wave

from .fldigi import FldigiClient


DISABLE_TAGS = {
    "AUDIOIO": "2",  # PulseAudio/PipeWire compatibility layer for the private sink.
    "CHKUSEHAMLIBIS": "0", "CHKUSERIGCATIS": "0", "CHKUSEXMLRPCIS": "0",
    "PTTKEYLINE": "0", "HAMLIBPTTONDATA": "0", "HAMLIBCMDPTT": "0",
    "TRANSMITRSID": "0",
    "RIGCATCMDPTT": "0", "FLRIG_FSK": "0", "PTTRIGHTCHANNEL": "0",
    "RTSPTT": "0", "DTRPTT": "0", "ENABLE_GPIO_PTT": "0",
    "USEPPORTPTT": "0", "USEUHROUTERPTT": "0", "CMEDIA-PTT": "0",
    "KMLSERVER_ENABLED": "0", "PSKREP_AUTOSTART": "0", "USEPSKREP": "0",
    "PSKREPAUTO": "0", "PSKREPLOG": "0", "PSKMAILS2NREPORT": "0",
    "AUTO_TALK": "0", "ENABLE_UDP_LOGGING": "0", "CONNECT_TO_MACLOGGER": "0",
    "CONNECT_TO_N3FJP": "0", "CONNECT_TO_FDSERVER": "0",
    "DXC_AUTO_CONNECT": "0", "TCP_UDP_AUTO_CONNECT": "0",
}


class FldigiReplayError(RuntimeError):
    """The isolated decoder or virtual audio path could not be started."""


class IsolatedFldigiReplay:
    def __init__(self, profile_dir: str | Path | None = None, *, decode_timeout: float = 30.0):
        self.profile_dir = Path(profile_dir or os.environ.get("SIG_ID_FLDIGI_PROFILE", "~/.fldigi")).expanduser()
        self.decode_timeout = decode_timeout
        self._closed = Event()
        self._process_lock = Lock()
        self._processes: set[subprocess.Popen[Any]] = set()

    def close(self) -> None:
        self._closed.set()
        with self._process_lock:
            processes = tuple(self._processes)
        for process in processes:
            self._stop(process)

    def _track_process(self, process: subprocess.Popen[Any], *, add: bool) -> None:
        with self._process_lock:
            if add:
                self._processes.add(process)
            else:
                self._processes.discard(process)

    @staticmethod
    def _run(*args: str, timeout: float = 5.0) -> str:
        try:
            result = subprocess.run(args, text=True, capture_output=True, timeout=timeout, check=True)
        except (OSError, subprocess.SubprocessError) as exc:
            raise FldigiReplayError(f"could not run {Path(args[0]).name}: {exc}") from exc
        return result.stdout.strip()

    @staticmethod
    def _free_port() -> int:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            return int(listener.getsockname()[1])

    @staticmethod
    def _stop(process: subprocess.Popen[Any]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=4)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=2)

    def _isolated_profile(self, root: Path, *, enable_rsid: bool = False) -> tuple[Path, Path]:
        source = self.profile_dir
        prefs_source = source / "fldigi_def.xml"
        if not prefs_source.is_file():
            raise FldigiReplayError(
                f"configure Fldigi once first; receive profile not found at {prefs_source}"
            )
        config = root / "config"
        home = root / "home"
        try:
            shutil.copytree(source, config)
        except OSError as exc:
            raise FldigiReplayError(f"could not copy the Fldigi profile for isolated replay: {exc}") from exc
        home.mkdir()
        prefs = config / "fldigi_def.xml"
        text = prefs.read_text(errors="replace")
        if enable_rsid:
            # Enable receive-side RSID only in the temporary replay profile.
            # Fldigi's TxID remains disabled, so this worker cannot beacon.
            text, count = re.subn(
                r"(<RECEIVERSID>).*?(</RECEIVERSID>)",
                lambda match: f"{match.group(1)}1{match.group(2)}",
                text, count=1,
            )
            if count != 1:
                raise FldigiReplayError("isolated Fldigi profile is missing RECEIVERSID; cannot enable receive-side RSID safely")
            text, count = re.subn(
                r"(<RSIDWIDESEARCH>).*?(</RSIDWIDESEARCH>)",
                lambda match: f"{match.group(1)}1{match.group(2)}",
                text, count=1,
            )
            if count != 1:
                raise FldigiReplayError("isolated Fldigi profile is missing RSIDWIDESEARCH; cannot scan the selected audio passband")
        for tag, value in DISABLE_TAGS.items():
            text, count = re.subn(
                fr"(<{re.escape(tag)}>).*?(</{re.escape(tag)}>)",
                lambda match: f"{match.group(1)}{value}{match.group(2)}",
                text, count=1,
            )
            if count != 1:
                raise FldigiReplayError(
                    f"isolated Fldigi profile is missing {tag}; refusing replay without confirmed safe audio/PTT settings"
                )
        prefs.write_text(text)
        return home, config

    def decode(
        self, wav_path: str | Path, *, mode: str,
        carrier_hz: float | None = None, detect_rsid: bool = False,
    ) -> dict[str, Any]:
        if self._closed.is_set():
            raise FldigiReplayError("isolated replay service is shutting down")
        path = Path(wav_path).resolve(strict=True)
        for executable in ("fldigi", "xvfb-run", "pactl", "paplay"):
            if shutil.which(executable) is None:
                raise FldigiReplayError(f"{executable} is required for isolated Fldigi replay")

        sink = f"sidr_{os.getpid()}_{os.urandom(3).hex()}"
        try:
            module_id = self._run(
                "pactl", "load-module", "module-null-sink",
                f"sink_name={sink}", "rate=48000", "channels=2",
            )
        except FldigiReplayError as exc:
            raise FldigiReplayError(f"PulseAudio/PipeWire virtual audio is unavailable: {exc}") from exc

        process: subprocess.Popen[Any] | None = None
        playback: subprocess.Popen[Any] | None = None
        try:
            with tempfile.TemporaryDirectory(prefix="sig-id-fldigi-replay-") as temporary:
                root = Path(temporary)
                home, config = self._isolated_profile(root, enable_rsid=detect_rsid)
                port = self._free_port()
                environment = os.environ.copy()
                environment["PULSE_SOURCE"] = f"{sink}.monitor"
                environment["PULSE_SINK"] = sink
                process = subprocess.Popen(
                    [
                        "xvfb-run", "-a", "fldigi", "--home-dir", str(home),
                        "--config-dir", str(config), "--xmlrpc-server-address", "127.0.0.1",
                        "--xmlrpc-server-port", str(port),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=environment,
                    start_new_session=True,
                )
                self._track_process(process, add=True)
                client = FldigiClient(f"http://127.0.0.1:{port}/RPC2", timeout=0.4)
                for _ in range(80):
                    if self._closed.is_set():
                        raise FldigiReplayError("isolated replay was stopped during application shutdown")
                    if process.poll() is not None:
                        raise FldigiReplayError("isolated Fldigi exited during startup")
                    try:
                        client.version()
                        break
                    except Exception:
                        time.sleep(0.15)
                else:
                    raise FldigiReplayError("isolated Fldigi XML-RPC did not become ready")

                selected = client.select_mode(mode)
                if selected != mode:
                    raise FldigiReplayError(f"Fldigi selected {selected!r} instead of {mode!r}")
                if carrier_hz is not None:
                    client.set_carrier(round(carrier_hz))
                initial_mode = client.current_modem()
                offset = client.receive_text_length()
                try:
                    with wave.open(str(path), "rb") as wav:
                        audio_seconds = wav.getnframes() / wav.getframerate()
                except (wave.Error, OSError, ZeroDivisionError) as exc:
                    raise FldigiReplayError(f"selected replay audio is not a readable PCM WAV: {exc}") from exc

                playback = subprocess.Popen(
                    ["paplay", f"--device={sink}", str(path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    env=environment,
                    start_new_session=True,
                )
                self._track_process(playback, add=True)
                try:
                    playback.wait(timeout=min(610.0, max(self.decode_timeout, audio_seconds + 15.0)))
                except subprocess.TimeoutExpired as exc:
                    self._stop(playback)
                    raise FldigiReplayError("audio replay exceeded the configured time limit") from exc
                if playback.returncode:
                    detail = playback.stderr.read(2048).decode("utf-8", errors="replace") if playback.stderr else ""
                    raise FldigiReplayError(f"PulseAudio could not play the selected clip: {detail.strip()}")
                self._track_process(playback, add=False)

                started = time.monotonic()
                deadline = started + min(120.0, self.decode_timeout)
                last_change = started
                last_text = b""
                while time.monotonic() < deadline:
                    if self._closed.is_set():
                        raise FldigiReplayError("isolated replay was stopped during application shutdown")
                    end = client.receive_text_length()
                    if end < offset:
                        offset = 0
                    raw = client.receive_text(start=offset, length=min(max(0, end - offset), 65536))
                    if raw != last_text:
                        last_text = raw
                        last_change = time.monotonic()
                    if time.monotonic() - last_change >= 2.0:
                        break
                    time.sleep(0.2)
                resulting_mode = client.current_modem()
                rsid_detected = detect_rsid and resulting_mode != initial_mode
                return {
                    "mode": resulting_mode,
                    "rsid_detected": rsid_detected,
                    "rsid_mode": resulting_mode if rsid_detected else None,
                    "text": last_text.decode("utf-8", errors="replace"),
                    "status": "completed" if last_text else "completed_no_text",
                    "note": "Selected audio window replayed through an isolated Fldigi instance; review its text before labeling.",
                    "backend_version": client.version().version,
                    "carrier_hz": carrier_hz,
                }
        finally:
            if playback is not None:
                self._stop(playback)
                self._track_process(playback, add=False)
            if process is not None:
                self._stop(process)
                self._track_process(process, add=False)
            try:
                self._run("pactl", "unload-module", module_id, timeout=5)
            except FldigiReplayError:
                pass
