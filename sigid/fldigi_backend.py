"""Generic typed-backend adapter for the optional native Fldigi application."""

from __future__ import annotations

import shutil
from typing import Any

from .decoder_backend import (
    DecoderBackendInfo, DecoderMode, DecoderProduct, DecoderRequest, DecoderResult,
    ModemGenerationRequest, ModemGenerationResult,
)
from .fldigi import FldigiClient
from .fldigi_replay import IsolatedFldigiReplay
from .fldigi_synthetic import IsolatedFldigiTransmitter
from .modes import MODE_REGISTRY


class FldigiNativeBackend:
    """Expose Fldigi through the same contract used by separately installed plugins."""

    backend_id = "fldigi"

    def __init__(
        self, client: FldigiClient, replay: IsolatedFldigiReplay,
        transmitter: IsolatedFldigiTransmitter | None = None,
    ) -> None:
        self.client = client
        self.replay = replay
        self.transmitter = transmitter or IsolatedFldigiTransmitter()

    def info(self) -> DecoderBackendInfo:
        version = None
        try:
            version = self.client.version().version
        except Exception:
            pass
        modes = tuple(
            DecoderMode(
                mode_id=item["id"],
                display_name=item["display_name"],
                native_mode=str(item["backend_name"]),
                family=item["family"],
                method=item["method"],
                protocol=item["protocol"],
                variant=item["variant"],
                aliases=tuple(item.get("aliases", ())),
                selectable=bool(item.get("backend_selectable", False)),
                note=str(item.get("status", "")),
            )
            for item in MODE_REGISTRY.values()
            if item.get("backend_name") and item.get("backend_selectable")
        )
        required = ("fldigi", "xvfb-run", "pactl", "paplay")
        installed = all(shutil.which(name) for name in required)
        return DecoderBackendInfo(
            backend_id=self.backend_id,
            application="Fldigi",
            connected=installed,
            version=version,
            input_kinds=("pcm-wav",),
            product_kinds=("text", "image"),
            modes=modes,
            note=("Isolated receive replay is available." if installed else
                  "Install Fldigi, Xvfb, PulseAudio/PipeWire tools, and paplay to enable isolated replay."),
        )

    def decode(self, request: DecoderRequest) -> DecoderResult:
        item = MODE_REGISTRY.get(request.mode_id.casefold())
        if item is None or not item.get("backend_selectable") or not item.get("backend_name"):
            raise ValueError(f"mode {request.mode_id!r} is not a selectable Fldigi mode")
        if str(item["backend_name"]) != request.native_mode:
            raise ValueError("native mode does not match the selected canonical label")
        result: dict[str, Any] = self.replay.decode(
            request.audio_path, mode=request.native_mode, carrier_hz=request.carrier_hz,
        )
        text = str(result.get("text", ""))
        status = "completed" if text else "completed_empty"
        return DecoderResult(
            backend_id=self.backend_id,
            application="Fldigi",
            mode_id=request.mode_id,
            native_mode=request.native_mode,
            status=status,
            products=(DecoderProduct("text", text, metadata={"sample_start": request.sample_start,
                                                               "sample_count": request.sample_count,
                                                               "sample_rate": request.sample_rate}),),
            version=result.get("backend_version"),
            note=str(result.get("note", "")),
        )

    def generate(self, request: ModemGenerationRequest) -> ModemGenerationResult:
        """Capture a transmission from a separate Fldigi process and virtual sink."""
        return self.transmitter.generate(request)

    def close(self) -> None:
        self.replay.close()
