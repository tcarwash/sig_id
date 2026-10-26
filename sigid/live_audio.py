"""Bounded live PCM capture from a sounddevice-compatible audio backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
import queue
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4
import wave

import numpy as np

from .dataset import CaptureStore


QUEUE_BLOCKS = 32
MAX_CAPTURE_SECONDS = 8 * 60 * 60
LIVE_ANALYSIS_SECONDS = 10


@dataclass(slots=True)
class _Session:
    id: str
    path: Path | None
    device_id: int | None
    device_name: str
    sample_rate: int
    channels: int
    max_duration_seconds: int
    audio_backend: Any
    capture_mode: str = "record"
    noise_interval_seconds: int = 300
    window_analyzer: Any = None
    blocks: queue.Queue[bytes] = field(default_factory=lambda: queue.Queue(maxsize=QUEUE_BLOCKS))
    stop_worker: threading.Event = field(default_factory=threading.Event)
    stream: Any = None
    writer_thread: threading.Thread | None = None
    frames_written: int = 0
    started_at_monotonic: float = 0.0
    state: str = "starting"
    truncated_reason: str | None = None
    status_messages: list[str] = field(default_factory=list)
    recent_blocks: deque[bytes] = field(default_factory=deque)
    recent_frames: int = 0
    review_queue_cursor: int = 0
    review_queue_buffer: bytearray = field(default_factory=bytearray)
    worker_error: str | None = None
    analysis_windows: queue.Queue[tuple[int, bytes]] = field(default_factory=lambda: queue.Queue(maxsize=4))
    stop_analyzer: threading.Event = field(default_factory=threading.Event)
    analyzer_thread: threading.Thread | None = None
    saved_review_clips: int = 0
    last_noise_clip_monotonic: float = 0.0
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
        self._stop_lock = threading.Lock()

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
                "device_name": session.device_name,
                "capture_mode": session.capture_mode,
                "saved_review_clips": session.saved_review_clips,
                "state": session.state,
                "sample_rate": session.sample_rate,
                "channels": session.channels,
                "frames_written": session.frames_written,
                "truncated_reason": session.truncated_reason,
                "status_messages": list(session.status_messages),
            }

    def analysis_audio(self) -> dict[str, Any] | None:
        """Copy a bounded, recent mono/stereo PCM window for live visualization."""
        with self._lock:
            session = self._session
        if session is None:
            return None
        with session.lock:
            pcm = b"".join(session.recent_blocks)
            return {
                "pcm": pcm,
                "sample_rate": session.sample_rate,
                "channels": session.channels,
                "sample_offset": max(0, session.frames_written - session.recent_frames),
                "sample_count": session.frames_written,
                "window_sample_count": session.recent_frames,
                "session_id": session.id,
                "active": session.state in {"starting", "recording", "limit_reached", "overrun"},
            }

    def start(
        self,
        *,
        device_id: int | None = None,
        sample_rate: int = 48_000,
        channels: int = 1,
        max_duration_seconds: int = 600,
        capture_mode: str = "record",
        noise_interval_seconds: int = 300,
        window_analyzer: Any = None,
    ) -> dict[str, Any]:
        if sample_rate < 4000 or sample_rate > 192000:
            raise ValueError("sample_rate must be between 4000 and 192000 Hz")
        if channels not in (1, 2):
            raise ValueError("channels must be 1 or 2")
        if not 1 <= max_duration_seconds <= MAX_CAPTURE_SECONDS:
            raise ValueError(f"max_duration_seconds must be from 1 to {MAX_CAPTURE_SECONDS}")
        if capture_mode not in {"record", "scan_only", "unattended"}:
            raise ValueError("capture_mode must be record, scan_only, or unattended")
        if capture_mode == "record" and max_duration_seconds > 1800:
            raise ValueError("full recording mode is limited to 1800 seconds; use scan-only or unattended mode for longer sessions")
        if not 60 <= noise_interval_seconds <= 3600:
            raise ValueError("noise_interval_seconds must be from 60 to 3600")
        if capture_mode == "unattended" and not callable(window_analyzer):
            raise ValueError("unattended mode needs a signal detector callback")
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
                path=self.capture_dir / f"live-{session_id}.wav" if capture_mode == "record" else None,
                device_id=device_id,
                device_name=str(device_info.get("name", "Default input")),
                sample_rate=sample_rate,
                channels=channels,
                max_duration_seconds=max_duration_seconds,
                audio_backend=backend,
                capture_mode=capture_mode,
                noise_interval_seconds=noise_interval_seconds,
                window_analyzer=window_analyzer,
            )
            session.started_at_monotonic = time.monotonic()
            session.last_noise_clip_monotonic = session.started_at_monotonic
            self._session = session

        session.writer_thread = threading.Thread(
            target=self._write_blocks, args=(session,), name=f"sig-id-capture-{session_id[:8]}", daemon=True
        )
        session.writer_thread.start()
        if capture_mode == "unattended":
            session.analyzer_thread = threading.Thread(
                target=self._analyze_review_windows, args=(session,),
                name=f"sig-id-unattended-{session_id[:8]}", daemon=True,
            )
            session.analyzer_thread.start()

        def callback(indata: Any, frames: int, _time_info: Any, status: Any) -> None:
            if status:
                with session.lock:
                    session.status_messages.append(str(status))
                    del session.status_messages[:-64]
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
            session.stop_analyzer.set()
            if session.analyzer_thread is not None:
                session.analyzer_thread.join(timeout=3)
            with self._lock:
                self._session = None
            if session.path is not None:
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
            "capture_mode": session.capture_mode,
            "noise_interval_seconds": session.noise_interval_seconds if capture_mode == "unattended" else None,
        }

    def _write_blocks(self, session: _Session) -> None:
        try:
            wav = None
            if session.path is not None:
                wav = wave.open(str(session.path), "wb")
                wav.setnchannels(session.channels)
                wav.setsampwidth(2)
                wav.setframerate(session.sample_rate)
            try:
                while True:
                    try:
                        block = session.blocks.get(timeout=0.1)
                    except queue.Empty:
                        if session.stop_worker.is_set():
                            break
                        continue
                    if wav is not None:
                        wav.writeframesraw(block)
                    review_windows: list[tuple[int, bytes]] = []
                    with session.lock:
                        frames = len(block) // (2 * session.channels)
                        session.frames_written += frames
                        session.recent_blocks.append(block)
                        session.recent_frames += frames
                        capacity = session.sample_rate * LIVE_ANALYSIS_SECONDS
                        excess = session.recent_frames - capacity
                        while excess > 0 and session.recent_blocks:
                            oldest = session.recent_blocks.popleft()
                            oldest_frames = len(oldest) // (2 * session.channels)
                            if oldest_frames <= excess:
                                session.recent_frames -= oldest_frames
                                excess -= oldest_frames
                            else:
                                trim_bytes = excess * 2 * session.channels
                                session.recent_blocks.appendleft(oldest[trim_bytes:])
                                session.recent_frames -= excess
                                excess = 0
                        if session.capture_mode != "scan_only":
                            window_frames = session.sample_rate * LIVE_ANALYSIS_SECONDS
                            window_bytes = window_frames * 2 * session.channels
                            session.review_queue_buffer.extend(block)
                            while len(session.review_queue_buffer) >= window_bytes:
                                window_start = session.review_queue_cursor
                                review_windows.append((window_start, bytes(session.review_queue_buffer[:window_bytes])))
                                del session.review_queue_buffer[:window_bytes]
                                session.review_queue_cursor += window_frames
                    if session.capture_mode == "record":
                        for window_start, window_audio in review_windows:
                            try:
                                self._save_review_window(
                                    session, window_start, window_audio,
                                    trigger_type="continuous_recording",
                                )
                            except Exception as exc:
                                with session.lock:
                                    session.status_messages.append(f"Auto review sample at {window_start} failed: {exc}")
                    elif session.capture_mode == "unattended":
                        for window in review_windows:
                            try:
                                session.analysis_windows.put_nowait(window)
                            except queue.Full:
                                with session.lock:
                                    session.status_messages.append("Unattended analyzer backlog is full; skipped one 10-second window")
            finally:
                if wav is not None:
                    wav.close()
        except Exception as exc:
            session.worker_error = str(exc)
            with session.lock:
                session.state = "error"

    def _analyze_review_windows(self, session: _Session) -> None:
        while not session.stop_analyzer.is_set() or not session.analysis_windows.empty():
            try:
                sample_start, audio = session.analysis_windows.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                pcm = np.frombuffer(audio, dtype="<i2").reshape(-1, session.channels)
                mono = pcm.astype(np.float32).mean(axis=1) / 32768.0
                result = session.window_analyzer(mono, session.sample_rate)
                if isinstance(result, dict):
                    detected = bool(result.get("detected"))
                    summary = {
                        "cv_detection_count": int(result.get("cv_detection_count", 0)),
                        "top_confidences": [float(value) for value in result.get("top_confidences", [])[:5]],
                        "classifier_methods": [str(value) for value in result.get("classifier_methods", [])[:10]],
                    }
                else:
                    detected, summary = bool(result), {}
                now = time.monotonic()
                if detected:
                    trigger_type = "cv_signal"
                elif now - session.last_noise_clip_monotonic >= session.noise_interval_seconds:
                    trigger_type = "periodic_noise"
                    session.last_noise_clip_monotonic = now
                else:
                    continue
                self._save_review_window(
                    session, sample_start, audio, trigger_type=trigger_type, analysis=summary,
                )
            except Exception as exc:
                with session.lock:
                    session.status_messages.append(f"Unattended window at {sample_start} failed: {exc}")

    def _save_review_window(
        self, session: _Session, sample_start: int, audio: bytes, *,
        trigger_type: str, analysis: dict[str, Any] | None = None,
    ) -> None:
        path = self.capture_dir / f"review-{session.id}-{sample_start}.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(session.channels)
            wav.setsampwidth(2)
            wav.setframerate(session.sample_rate)
            wav.writeframes(audio)
        try:
            self.store.add_wav_capture(
                path,
                source_kind="live",
                source_metadata={
                    "auto_review_queue": True,
                    "auto_capture_session_id": session.id,
                    "window_start_sample": sample_start,
                    "window_seconds": LIVE_ANALYSIS_SECONDS,
                    "device_name": session.device_name,
                    "unattended_capture": session.capture_mode == "unattended",
                    "review_trigger": trigger_type,
                    "detection_summary": analysis or {},
                },
            )
            with session.lock:
                session.saved_review_clips += 1
        except Exception:
            path.unlink(missing_ok=True)
            raise

    def stop(self) -> dict[str, Any]:
        with self._stop_lock:
            return self._finish_session()

    def shutdown(self) -> dict[str, Any] | None:
        """Finalize an outstanding recording during a graceful service shutdown."""
        with self._stop_lock:
            with self._lock:
                if self._session is None:
                    return None
            return self._finish_session()

    def _finish_session(self) -> dict[str, Any]:
        with self._lock:
            session = self._session
        if session is None:
            raise RuntimeError("there is no active live capture")

        for operation in (session.stream.stop, session.stream.close):
            try:
                operation()
            except Exception as exc:
                session.truncated_reason = "input_close_error"
                session.status_messages.append(f"Input shutdown: {exc}")
        session.stop_worker.set()
        assert session.writer_thread is not None
        session.writer_thread.join(timeout=5)
        if session.writer_thread.is_alive():
            raise RuntimeError("audio writer did not stop within five seconds")
        if session.analyzer_thread is not None:
            session.stop_analyzer.set()
            session.analyzer_thread.join(timeout=60)
            if session.analyzer_thread.is_alive():
                raise RuntimeError("unattended review analyzer did not finish within 60 seconds")
        if session.worker_error:
            session.truncated_reason = "audio_writer_error"
            session.status_messages.append(f"Audio writer: {session.worker_error}")
            if session.frames_written <= 0:
                with self._lock:
                    self._session = None
                raise RuntimeError(f"audio writer failed: {session.worker_error}")
        if session.frames_written <= 0:
            if session.path is not None:
                session.path.unlink(missing_ok=True)
            with self._lock:
                self._session = None
            raise RuntimeError("capture stopped before any audio samples were written")

        if session.path is None:
            with session.lock:
                session.state = "stopped"
            with self._lock:
                self._session = None
            return {
                "session_id": session.id,
                "frames_written": session.frames_written,
                "truncated_reason": session.truncated_reason,
                "capture": None,
                "capture_mode": session.capture_mode,
                "saved_review_clips": session.saved_review_clips,
            }

        try:
            capture_id = self.store.add_wav_capture(
                session.path,
                source_kind="live",
                source_metadata={
                    "device_id": session.device_id,
                    "device_name": session.device_name,
                    "auto_capture_session_id": session.id,
                    "truncated_reason": session.truncated_reason,
                    "status_messages": list(session.status_messages[-64:]),
                },
            )
        except Exception as exc:
            with self._lock:
                self._session = None
            raise RuntimeError(f"could not index capture; audio retained at {session.path}: {exc}") from exc
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
