"""Contracts for optional, receive-only decoder application adapters.

Decoder implementations stay in their native applications or separately installed
plugins. Signal ID owns the handoff, capability/mode catalog, provenance, and typed
results; it does not need to bundle every decoder. Identification and raw symbol
recovery remain separate capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from threading import RLock
from typing import Any, Literal, Protocol


ProductKind = Literal["text", "image", "audio", "binary", "metadata"]


@dataclass(frozen=True, slots=True)
class DecoderMode:
    """One native application mode and the stable label used by Signal ID."""

    mode_id: str
    display_name: str
    native_mode: str
    family: str
    method: str
    protocol: str
    variant: str
    aliases: tuple[str, ...] = ()
    selectable: bool = True
    note: str = ""


@dataclass(frozen=True, slots=True)
class DecoderBackendInfo:
    """Runtime capability report; installation alone is not decode proof."""

    backend_id: str
    application: str
    connected: bool
    version: str | None = None
    input_kinds: tuple[str, ...] = ()
    product_kinds: tuple[ProductKind, ...] = ("text",)
    modes: tuple[DecoderMode, ...] = ()
    note: str = ""


@dataclass(frozen=True, slots=True)
class DecoderRequest:
    """A bounded receive request independent of the backend's native API."""

    audio_path: Path
    sample_rate: int
    mode_id: str
    native_mode: str
    sample_start: int = 0
    sample_count: int | None = None
    carrier_hz: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModemGenerationRequest:
    """Request a native modem to create an audio example without RF output."""

    mode_id: str
    native_mode: str
    output_path: Path
    payload: str
    sample_rate: int = 12_000
    carrier_hz: float = 1500.0
    timeout_seconds: float = 180.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModemGenerationResult:
    backend_id: str
    mode_id: str
    native_mode: str
    audio_path: Path
    sample_rate: int
    sample_count: int
    version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DecoderProduct:
    """A typed artifact returned by a native decoder."""

    kind: ProductKind
    content: str | bytes
    media_type: str = "text/plain"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DecoderResult:
    backend_id: str
    application: str
    mode_id: str
    native_mode: str
    status: Literal["completed", "completed_empty", "failed"]
    products: tuple[DecoderProduct, ...] = ()
    version: str | None = None
    note: str = ""


@dataclass(frozen=True, slots=True)
class DecoderEvent:
    """Unsolicited live result from a decoder application or local service."""

    backend_id: str
    mode_id: str
    native_mode: str
    observed_utc: str
    products: tuple[DecoderProduct, ...] = ()
    frequency_hz: float | None = None
    snr_db: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class NativeDecoderBackend(Protocol):
    """Adapter to a separately installed native decoder or decoder service.

    Implementations should accept bounded receive inputs, never transmit or tune
    radios, and report each output as a typed product with provenance. An optional
    ``generate`` capability may produce audio only through an isolated virtual
    audio route that cannot reach a radio or PTT device.
    """

    backend_id: str

    def info(self) -> DecoderBackendInfo: ...


class AudioWindowDecoder(Protocol):
    """Optional capability for replaying an exact selected audio window."""

    def decode(self, request: DecoderRequest) -> DecoderResult: ...


class ModemAudioGenerator(Protocol):
    """Optional modem-plugin capability for native RF-free waveform generation."""

    def generate(self, request: ModemGenerationRequest) -> ModemGenerationResult: ...


class LiveDecoderEventSource(Protocol):
    """Optional capability for applications that publish live decode events."""

    def drain_events(self, *, limit: int = 100) -> list[DecoderEvent]: ...


class DecoderBackendRegistry:
    """Collect explicitly supplied and installed entry-point adapters."""

    ENTRY_POINT_GROUP = "sigid.decoder_backends"

    def __init__(self, backends: tuple[NativeDecoderBackend, ...] = ()) -> None:
        self._backends: dict[str, NativeDecoderBackend] = {}
        self._locks: dict[str, RLock] = {}
        self.load_errors: list[dict[str, str]] = []
        for backend in backends:
            self.register(backend)

    def register(self, backend: NativeDecoderBackend) -> None:
        backend_id = str(backend.backend_id).strip().casefold()
        if not backend_id or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for ch in backend_id):
            raise ValueError("decoder backend IDs must use lowercase letters, digits, hyphens, and underscores")
        if backend_id in self._backends:
            raise ValueError(f"decoder backend is already registered: {backend_id}")
        self._backends[backend_id] = backend
        self._locks[backend_id] = RLock()

    @classmethod
    def discover(cls, backends: tuple[NativeDecoderBackend, ...] = ()) -> "DecoderBackendRegistry":
        registry = cls(backends)
        points = entry_points()
        selected = points.select(group=cls.ENTRY_POINT_GROUP) if hasattr(points, "select") else points.get(cls.ENTRY_POINT_GROUP, ())
        for point in selected:
            try:
                factory = point.load()
                registry.register(factory())
            except Exception as exc:
                registry.load_errors.append({"plugin": point.name, "error": str(exc)})
        return registry

    def info(self) -> list[dict[str, Any]]:
        result = []
        for backend_id, backend in self._backends.items():
            try:
                description = backend.info()
                result.append({
                    "backend_id": backend_id,
                    "application": description.application,
                    "connected": description.connected,
                    "can_generate_audio": callable(getattr(backend, "generate", None)),
                    "version": description.version,
                    "input_kinds": list(description.input_kinds),
                    "product_kinds": list(description.product_kinds),
                    "modes": [{
                        "mode_id": mode.mode_id,
                        "display_name": mode.display_name,
                        "native_mode": mode.native_mode,
                        "selectable": mode.selectable,
                        "note": mode.note,
                    } for mode in description.modes],
                    "note": description.note,
                })
            except Exception as exc:
                result.append({"backend_id": backend_id, "connected": False, "error": str(exc)})
        return result

    def decode(self, backend_id: str, request: DecoderRequest) -> DecoderResult:
        backend = self._backends.get(backend_id.casefold())
        if backend is None:
            raise KeyError(f"decoder backend is not registered: {backend_id}")
        decode = getattr(backend, "decode", None)
        if not callable(decode):
            raise NotImplementedError(f"decoder backend {backend_id} does not accept audio-window requests")
        with self._locks[backend_id.casefold()]:
            result = decode(request)
        if result.backend_id.casefold() != backend_id.casefold():
            raise ValueError("decoder result backend_id does not match the registered backend")
        return result

    def generate(self, backend_id: str, request: ModemGenerationRequest) -> ModemGenerationResult:
        """Use an optional native modem transmission path when the plugin provides one."""
        backend = self._backends.get(backend_id.casefold())
        if backend is None:
            raise KeyError(f"decoder backend is not registered: {backend_id}")
        generate = getattr(backend, "generate", None)
        if not callable(generate):
            raise NotImplementedError(f"modem backend {backend_id} does not generate audio samples")
        with self._locks[backend_id.casefold()]:
            result = generate(request)
        if result.backend_id.casefold() != backend_id.casefold():
            raise ValueError("generation result backend_id does not match the registered backend")
        return result

    def mode_binding(self, backend_id: str, mode_id: str) -> DecoderMode | None:
        backend = self._backends.get(backend_id.casefold())
        if backend is None:
            return None
        try:
            return next((mode for mode in backend.info().modes if mode.mode_id.casefold() == mode_id.casefold()), None)
        except Exception:
            return None

    def drain_events(self, backend_id: str, *, limit: int = 100) -> list[DecoderEvent]:
        backend = self._backends.get(backend_id.casefold())
        if backend is None:
            raise KeyError(f"decoder backend is not registered: {backend_id}")
        drain = getattr(backend, "drain_events", None)
        if not callable(drain):
            return []
        with self._locks[backend_id.casefold()]:
            events = drain(limit=limit)
        if len(events) > limit:
            raise ValueError("decoder backend returned more live events than requested")
        return events

    def close(self) -> None:
        for backend in self._backends.values():
            close = getattr(backend, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    continue


class TextReceiveBackend(Protocol):
    """Minimal interface needed by the bounded live mode-trial runner."""

    backend_name: str

    def mode_names(self) -> list[str]: ...

    def select_mode(self, name: str) -> str: ...

    def receive_text(self, *, start: int = 0, length: int = 65536) -> bytes: ...

    def receive_text_length(self) -> int: ...
