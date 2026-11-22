"""Thin adapter exposing the owned decoder library beside optional external tools."""
from __future__ import annotations

import json
from pathlib import Path
import wave

import numpy as np

from .decoder_backend import (
    DecoderBackendInfo, DecoderMode, DecoderProduct,
    DecoderRequest, DecoderResult,
)
from .libsigdecode import DecodeWindow, DecoderLibrary, create_library


class LocalDecoderBackend:
    backend_id = "sigdecode"

    def __init__(self, library: DecoderLibrary | None = None) -> None:
        self.library = library or create_library()

    def info(self) -> DecoderBackendInfo:
        return DecoderBackendInfo(
            backend_id=self.backend_id,
            application="Signal ID built-in decoders",
            connected=True, version="1",
            input_kinds=("mono-pcm-audio-window",),
            product_kinds=("text", "metadata"),
            modes=tuple(
                DecoderMode(
                    mode_id=item.id, display_name=item.name, native_mode=item.id,
                    family=item.family, method="independent-audio",
                    protocol=item.name, variant=item.version,
                    selectable=item.input_kind == "mono-audio",
                    note=(f"In-project {item.stage} decoder; no external app."
                          if item.input_kind == "mono-audio" else
                          f"Requires {item.input_kind} input; audio-window bridge unavailable."),
                ) for item in self.library.modes()
            ),
            note="First-party decoder library; review all experimental and partial results before labeling.",
        )

    def decode(self, request: DecoderRequest) -> DecoderResult:
        matching = next((spec for spec in self.library.modes() if spec.id == request.mode_id), None)
        if matching is None:
            raise ValueError("requested mode is not an internal decoder")
        if matching.input_kind != "mono-audio":
            raise ValueError(f"{request.mode_id} requires {matching.input_kind}; audio decoding unsupported")
        path = Path(request.audio_path)
        with wave.open(str(path), "rb") as wav:
            if wav.getcomptype() != "NONE" or wav.getsampwidth() != 2 or wav.getnchannels() != 1:
                raise ValueError("internal decoder currently requires uncompressed mono 16-bit WAV")
            rate = wav.getframerate()
            samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float64) / 32768.0
        output = self.library.decode(request.mode_id, DecodeWindow(
            samples=samples,
            sample_rate=rate,
            sample_start=request.sample_start,
            carrier_hz=request.carrier_hz,
        ))
        details = output.to_dict()
        details["external_decoder_used"] = False
        # DecodeWindow was already cropped by the saved-selection API.
        products = (
            (DecoderProduct("text", output.text, metadata={
                "decoder_id": output.decoder_id, "status": output.status,
                "verified": False,
            }),) if output.text else ()
        ) + (DecoderProduct(
            "metadata", json.dumps(details, ensure_ascii=True), "application/json",
        ),)
        return DecoderResult(
            backend_id=self.backend_id,
            application="Signal ID built-in decoders",
            mode_id=request.mode_id, native_mode=request.native_mode,
            status="completed" if output.text else "completed_empty",
            products=products,
            version=output.version,
            note=output.note,
        )
