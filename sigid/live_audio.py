"""Bounded live PCM capture from a sounddevice-compatible audio backend."""

from __future__ import annotations

from dataclasses import dataclass, field
import queue
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4
import wave

from .dataset import CaptureStore


QUEUE_BLOCKS = 32
MAX_CAPTURE_SECONDS = 1800


@dataclass(slots=True)
class _Session:
    id: str
    path: Path
    device_id: int | None
    device_name: str
    sample_rate: int
    channels: int
    max_duration_seconds: int
    audio_backend: Any
    blocks: queue.Queue[bytes] = field(default_factory=lambda: queue.Queue(maxsize=QUEUE_BLOCKS))
    stop_worker: threading.Event = field(default_factory=threading.Event)
    stream: Any = None
    writer_thread: threading.Thread | None = None
    frames_written: int = 0
    started_at_monotonic: float = 0.0
    state: str = "starting"
    truncated_reason: str | None = None
    status_messages: list[str] = field(default_factory=list)
    worker_error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class LiveCaptureManager:
    """One active, bounded audio capture. Callback work never waits on disk."""

    def __init__(self, store: CaptureStore, capture_dir: str | Path, audio_backend: Any | None = None):
        self.store = store
        self.capture_dir = Path(capture_dir)
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.audio_backend = audio_backend
        self._session: _Session | None = None
        self._lock = threading.Lock()

    def _backend(self) -> Any:
        if self.audio_backend is None:
            try:
                import sounddevice
            except ImportError as exc:
                raise RuntimeError("install the audio extra to capture from local devices") from exc
            self.audio_backend = sounddevice
        return self.audio_backend

    def devices(self) -> list[dict[str, Any]]:
        backend = self._backend()
        devices = backend.query_devices()
        if isinstance(devices, dict):
            devices = [devices]
        result = []
        for index, device in enumerate(devices):
            input_channels = int(device.get("max_input_channels", 0))
            if input_channels <= 0:
                continue
            result.append(
                {
                    "id": index,
                    "name": str(device.get("name", f"Input {index}")),
                    "max_input_channels": input_channels,
                    "default_sample_rate": int(float(device.get("default_samplerate", 48000))),
                }
            )
        return result

    def status(self) -> dict[str, Any]:
        with self._lock:
            session = self._session
        if session is None:
            return {"active": False}
        with session.lock:
            return {
                "active": session.state in {"starting", "recording", "limit_reached", "overrun"},
                "session_id": session.id,
                "state": session.state,
                "sample_rate": session.sample_rate,
                "channels": session.channels,
                "frames_written": session.frames_written,
                "truncated_reason": session.truncated_reason,
                "status_messages": list(session.status_messages),
            }

    def start(
        self,
        *,
        device_id: int | None = None,
        sample_rate: int = 48_000,
        channels: int = 1,
        max_duration_seconds: int = 600,
    ) -> dict[str, Any]:
        if sample_rate < 4000 or sample_rate > 192000:
            raise ValueError("sample_rate must be between 4000 and 192000 Hz")
        if channels not in (1, 2):
            raise ValueError("channels must be 1 or 2")
        if not 1 <= max_duration_seconds <= MAX_CAPTURE_SECONDS:
            raise ValueError(f"max_duration_seconds must be from 1 to {MAX_CAPTURE_SECONDS}")
        backend = self._backend()
        device_info = None
        if device_id is not None:
            devices = backend.query_devices()
            if device_id < 0 or device_id >= len(devices):
                raise ValueError(f"unknown input device id: {device_id}")
            device_info = devices[device_id]
            if int(device_info.get("max_input_channels", 0)) < channels:
                raise ValueError("selected device does not have enough input channels")
        else:
            device_info = backend.query_devices(kind="input")
            if int(device_info.get("max_input_channels", 0)) < channels:
                raise ValueError("default input device does not have enough input channels")

        with self._lock:
            if self._session is not None:
                raise RuntimeError("a live capture is already active")
            session_id = str(uuid4())
            session = _Session(
                id=session_id,
                path=self.capture_dir / f"live-{session_id}.wav",
                device_id=device_id,
                device_name=str(device_info.get("name", "Default input")),
                sample_rate=sample_rate,
                channels=channels,
                max_duration_seconds=max_duration_seconds,
                audio_backend=backend,
            )
            session.started_at_monotonic = time.monotonic()
            self._session = session

        session.writer_thread = threading.Thread(
            target=self._write_blocks, args=(session,), name=f"sig-id-capture-{session_id[:8]}", daemon=True
        )
        session.writer_thread.start()

        def callback(indata: Any, frames: int, _time_info: Any, status: Any) -> None:
            if status:
                with session.lock:
                    session.status_messages.append(str(status))
            if time.monotonic() - session.started_at_monotonic >= max_duration_seconds:
                with session.lock:
                    session.state = "limit_reached"
                    session.truncated_reason = "maximum_duration"
                raise backend.CallbackStop
            try:
                session.blocks.put_nowait(memoryview(indata).tobytes())
            except queue.Full:
                with session.lock:
                    session.state = "overrun"
                    session.truncated_reason = "audio_callback_queue_full"
                raise backend.CallbackAbort

        try:
            session.stream = backend.InputStream(
                device=device_id,
                samplerate=sample_rate,
                channels=channels,
                dtype="int16",
                callback=callback,
            )
            session.stream.start()
            with session.lock:
                if session.state == "starting":
                    session.state = "recording"
        except Exception:
            if session.stream is not None:
                try:
                    session.stream.close()
                except Exception:
                    pass
            session.stop_worker.set()
            session.writer_thread.join(timeout=3)
            with self._lock:
                self._session = None
            session.path.unlink(missing_ok=True)
            raise

        return {
            "session_id": session.id,
            "state": session.state,
            "device_id": session.device_id,
            "device_name": session.device_name,
            "sample_rate": session.sample_rate,
            "channels": session.channels,
            "max_duration_seconds": session.max_duration_seconds,
        }

    def _write_blocks(self, session: _Session) -> None:
        try:
            with wave.open(str(session.path), "wb") as wav:
                wav.setnchannels(session.channels)
                wav.setsampwidth(2)
                wav.setframerate(session.sample_rate)
                while True:
                    try:
                        block = session.blocks.get(timeout=0.1)
                    except queue.Empty:
                        if session.stop_worker.is_set():
                            break
                        continue
                    wav.writeframesraw(block)
                    with session.lock:
                        session.frames_written += len(block) // (2 * session.channels)
        except Exception as exc:
            session.worker_error = str(exc)
            with session.lock:
                session.state = "error"

    def stop(self) -> dict[str, Any]:
        with self._lock:
            session = self._session
        if session is None:
            raise RuntimeError("there is no active live capture")

        try:
            session.stream.stop()
        finally:
            session.stream.close()
        session.stop_worker.set()
        assert session.writer_thread is not None
        session.writer_thread.join(timeout=5)
        if session.writer_thread.is_alive():
            raise RuntimeError("audio writer did not stop within five seconds")
        if session.worker_error:
            session.path.unlink(missing_ok=True)
            with self._lock:
                self._session = None
            raise RuntimeError(f"audio writer failed: {session.worker_error}")
        if session.frames_written <= 0:
            session.path.unlink(missing_ok=True)
            with self._lock:
                self._session = None
            raise RuntimeError("capture stopped before any audio samples were written")

        capture_id = self.store.add_wav_capture(
            session.path,
            source_kind="live",
            source_metadata={
                "device_id": session.device_id,
                "device_name": session.device_name,
                "truncated_reason": session.truncated_reason,
                "status_messages": session.status_messages,
            },
        )
        capture = self.store.capture(capture_id)
        with session.lock:
            session.state = "saved"
        with self._lock:
            self._session = None
        return {
            "session_id": session.id,
            "frames_written": session.frames_written,
            "truncated_reason": session.truncated_reason,
            "capture": capture,
        }
