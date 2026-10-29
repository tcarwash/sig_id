"""Isolated RF-free transmit capture from the native Fldigi modem."""

from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import wave
from typing import Any
from xmlrpc.client import ServerProxy

from .decoder_backend import (
    DecoderBackendRegistry, ModemGenerationRequest, ModemGenerationResult,
)
from .fldigi import FldigiClient
from .fldigi_replay import DISABLE_TAGS
from .modes import MODE_REGISTRY
from .synthetic_dataset import import_synthetic_wav
from .dataset import CaptureStore


class FldigiGenerationError(RuntimeError):
    """The isolated native Fldigi transmitter could not produce a complete sample."""


def fldigi_modes() -> list[dict[str, Any]]:
    """Return selectable Fldigi-backed labels in the shared modem registry."""
    return [
        item for item in MODE_REGISTRY.values()
        if item.get("backend_name")
        and item.get("backend_selectable")
        and str(item.get("catalog_source", "")).startswith("Fldigi ")
    ]


class IsolatedFldigiTransmitter:
    """Capture native modem audio through a private PulseAudio/PipeWire null sink.

    This starts a copied Fldigi profile with hamlib, rig CAT, PTT, and external
    integrations disabled. It never connects to the operator's running Fldigi.
    """

    def __init__(self, profile_dir: str | Path | None = None):
        self.profile_dir = Path(profile_dir or os.environ.get("SIG_ID_FLDIGI_PROFILE", "~/.fldigi")).expanduser()

    @staticmethod
    def _run(*args: str, timeout: float = 8.0) -> str:
        try:
            result = subprocess.run(args, text=True, capture_output=True, timeout=timeout, check=True)
        except (OSError, subprocess.SubprocessError) as exc:
            raise FldigiGenerationError(f"could not run {Path(args[0]).name}: {exc}") from exc
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

    def _start(self, root: Path, sink: str, port: int) -> subprocess.Popen[Any]:
        prefs_source = self.profile_dir / "fldigi_def.xml"
        if not prefs_source.is_file():
            raise FldigiGenerationError(f"configure Fldigi once first; profile not found at {prefs_source}")
        config, home = root / "config", root / "home"
        try:
            shutil.copytree(self.profile_dir, config)
            home.mkdir()
        except OSError as exc:
            raise FldigiGenerationError(f"could not prepare an isolated Fldigi profile: {exc}") from exc
        prefs = config / "fldigi_def.xml"
        text = prefs.read_text(errors="replace")
        import re
        for tag, value in DISABLE_TAGS.items():
            text, count = re.subn(
                fr"(<{re.escape(tag)}>).*?(</{re.escape(tag)}>)",
                lambda match: f"{match.group(1)}{value}{match.group(2)}",
                text, count=1,
            )
            if count != 1:
                raise FldigiGenerationError(f"isolated profile is missing required safe setting {tag}")
        prefs.write_text(text)
        environment = os.environ.copy()
        environment["PULSE_SOURCE"] = f"{sink}.monitor"
        environment["PULSE_SINK"] = sink
        return subprocess.Popen(
            ["xvfb-run", "-a", "fldigi", "--home-dir", str(home), "--config-dir", str(config),
             "--xmlrpc-server-address", "127.0.0.1", "--xmlrpc-server-port", str(port),
            # Keep the isolated control socket limited to mode selection,
            # virtual-audio transmission, and TX audio readback.
             "--xmlrpc-allow",
             r"^(fldigi\.(name|version)|modem\.(get_names|get_name|set_by_name|set_carrier)|"
             r"text\.(add_tx|clear_tx)|main\.(tx|rx)|tx\.get_data)$"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=environment,
            start_new_session=True,
        )

    def generate(self, request: ModemGenerationRequest) -> ModemGenerationResult:
        for executable in ("fldigi", "xvfb-run", "pactl", "parecord"):
            if shutil.which(executable) is None:
                raise FldigiGenerationError(f"{executable} is required for Fldigi audio generation")
        mode = MODE_REGISTRY.get(request.mode_id.casefold())
        if (
            mode is None or not mode.get("backend_selectable")
            or mode.get("backend_name") != request.native_mode
            or not str(mode.get("catalog_source", "")).startswith("Fldigi ")
        ):
            raise ValueError("requested mode is not a selectable Fldigi entry in the shared mode registry")
        if not request.payload.strip() or len(request.payload) > 512:
            raise ValueError("payload must contain 1 to 512 non-whitespace characters")
        if request.sample_rate != 12_000:
            raise ValueError("Fldigi corpus output is standardized to 12000 Hz mono PCM")
        if not 300 <= request.carrier_hz <= 3_000:
            raise ValueError("audio carrier must be between 300 and 3000 Hz")
        if not 5 <= request.timeout_seconds <= 900:
            raise ValueError("transmit timeout must be between 5 and 900 seconds")

        sink = f"sigid_gen_{os.getpid()}_{time.monotonic_ns() % 1_000_000}"
        try:
            module_id = self._run(
                "pactl", "load-module", "module-null-sink", f"sink_name={sink}",
                "rate=48000", "channels=2",
            )
        except FldigiGenerationError as exc:
            raise FldigiGenerationError(f"PulseAudio/PipeWire virtual audio is unavailable: {exc}") from exc
        process = None
        recorder = None
        try:
            with tempfile.TemporaryDirectory(prefix="sig-id-fldigi-tx-") as temp:
                root = Path(temp)
                port = self._free_port()
                process = self._start(root, sink, port)
                client = FldigiClient(f"http://127.0.0.1:{port}/RPC2", timeout=0.5)
                proxy = ServerProxy(f"http://127.0.0.1:{port}/RPC2", allow_none=True, use_builtin_types=True)
                for _ in range(80):
                    if process.poll() is not None:
                        raise FldigiGenerationError("isolated Fldigi exited during startup")
                    try:
                        version = client.version().version
                        break
                    except Exception:
                        time.sleep(0.15)
                else:
                    raise FldigiGenerationError("isolated Fldigi XML-RPC server did not start")
                selected = client.select_modem(request.native_mode)
                proxy.modem.set_carrier(int(round(request.carrier_hz)))
                audio_48k = root / "transmission-48000.wav"
                recorder = subprocess.Popen(
                    ["parecord", f"--device={sink}.monitor", "--file-format=wav",
                     "--rate=48000", "--channels=2", str(audio_48k)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True,
                )
                time.sleep(0.4)
                proxy.text.clear_tx()
                proxy.text.add_tx(request.payload)
                proxy.main.tx()
                deadline = time.monotonic() + request.timeout_seconds
                # Observe Fldigi's native TX byte stream. The UI can stay in TX
                # after its message is sent, so RX/TX state alone is not a
                # reliable completion signal.
                last_audio_at = None
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise FldigiGenerationError("isolated Fldigi exited while transmitting")
                    try:
                        tx_audio = proxy.tx.get_data()
                    except Exception as exc:
                        raise FldigiGenerationError(f"Fldigi TX audio stream is unavailable: {exc}") from exc
                    now = time.monotonic()
                    if tx_audio:
                        last_audio_at = now
                    if last_audio_at is not None and now - last_audio_at >= 2.0:
                        break
                    time.sleep(0.25)
                if last_audio_at is None or time.monotonic() - last_audio_at < 2.0:
                    raise FldigiGenerationError(f"{selected} did not finish transmitting before the timeout")
                proxy.main.rx()
                time.sleep(0.35)
                self._stop(recorder)
                recorder = None
                if not audio_48k.is_file() or audio_48k.stat().st_size < 4096:
                    raise FldigiGenerationError(f"{selected} produced no captured audio")
                output = Path(request.output_path)
                output.parent.mkdir(parents=True, exist_ok=True)
                self._convert_to_mono_12k(audio_48k, output)
                with wave.open(str(output), "rb") as wav:
                    count = wav.getnframes()
                return ModemGenerationResult(
                    backend_id="fldigi", mode_id=request.mode_id, native_mode=selected,
                    audio_path=output.resolve(), sample_rate=request.sample_rate,
                    sample_count=count, version=version,
                    metadata={"carrier_hz": request.carrier_hz, "payload": request.payload,
                              "capture_sample_rate": 48_000, "capture_channels": 2,
                              "transmit_backend": "isolated Fldigi XML-RPC over PulseAudio null sink"},
                )
        finally:
            if recorder is not None:
                self._stop(recorder)
            if process is not None:
                self._stop(process)
            subprocess.run(("pactl", "unload-module", module_id), capture_output=True, timeout=5, check=False)

    @staticmethod
    def _convert_to_mono_12k(source: Path, destination: Path) -> None:
        try:
            from scipy.signal import resample_poly
            import numpy as np
            with wave.open(str(source), "rb") as stream:
                if stream.getframerate() != 48_000 or stream.getsampwidth() != 2 or stream.getcomptype() != "NONE":
                    raise FldigiGenerationError("Fldigi loopback WAV has unsupported dimensions")
                channels = stream.getnchannels()
                pcm = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
            if channels not in (1, 2) or pcm.size == 0 or pcm.size % channels:
                raise FldigiGenerationError("Fldigi loopback WAV has invalid channel data")
            audio = pcm.reshape(-1, channels).astype(np.float64).mean(axis=1) / 32768.0
            converted = resample_poly(audio, up=1, down=4)
            samples = np.clip(np.round(converted * 32768.0), -32768, 32767).astype("<i2")
            with wave.open(str(destination), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(12_000)
                output.writeframes(samples.tobytes())
        except ImportError as exc:
            raise FldigiGenerationError("SciPy is required to resample Fldigi loopback audio") from exc


def _mode_bandwidth(mode: dict[str, Any]) -> float:
    text = str(mode.get("source_bandwidth", ""))
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*(?:Hz|hz)", text)
    if matches:
        return min(3_600.0, max(80.0, float(matches[0])))
    return 1_000.0


def build_fldigi_corpus(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    mode_query: str = "all",
    examples_per_mode: int = 1,
    seed: int = 4192026,
    carrier_hz: float = 1_500.0,
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    """Generate actual native Fldigi transmissions for selectable registry modes."""
    import random
    from .fldigi import FldigiClient
    from .fldigi_backend import FldigiNativeBackend
    from .fldigi_replay import IsolatedFldigiReplay

    if not 1 <= examples_per_mode <= 100:
        raise ValueError("examples_per_mode must be between 1 and 100")
    data_root = Path(data_dir).expanduser().resolve()
    modes = fldigi_modes()
    query = mode_query.strip().casefold()
    if query == "missing":
        store = CaptureStore(data_root / "dataset.sqlite3")
        existing = set()
        for item in store.list_library_signals():
            if item.get("source_metadata", {}).get("generator") != "sig-id-fldigi":
                continue
            region = item.get("region") or {}
            for value in (region.get("mode_id"), region.get("variant")):
                token = re.sub(r"[^a-z0-9]", "", str(value or "").casefold())
                existing.add(token.removeprefix("fldigi"))
        def mode_tokens(mode: dict[str, Any]) -> set[str]:
            values = (mode.get("id"), mode.get("display_name"), mode.get("backend_name"), mode.get("variant"))
            return {re.sub(r"[^a-z0-9]", "", str(value or "").casefold()).removeprefix("fldigi") for value in values}
        modes = [mode for mode in modes if not (mode_tokens(mode) & existing)]
    elif query != "all":
        matches = [item for item in modes if query in {
            str(item["id"]).casefold(), str(item["display_name"]).casefold(),
            str(item["backend_name"]).casefold(),
        }]
        if len(matches) != 1:
            raise ValueError(f"mode must exactly match one Fldigi mode ID, label, or backend name; found {len(matches)}")
        modes = matches
    output = Path(output_dir).expanduser().resolve()
    store = CaptureStore(data_root / "dataset.sqlite3")
    audio_dir = output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    backend = FldigiNativeBackend(FldigiClient(), IsolatedFldigiReplay())
    registry = DecoderBackendRegistry((backend,))
    generated: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    total_examples = len(modes) * examples_per_mode
    completed_examples = 0
    try:
        for mode_index, mode in enumerate(modes, start=1):
            mode_id = str(mode["id"])
            band = _mode_bandwidth(mode)
            low, high = max(0.0, carrier_hz - band * 0.75), min(6_000.0, carrier_hz + band * 0.75)
            if high <= low:
                error = "mode bandwidth is outside the 12 kHz output band"
                failures.append({"mode_id": mode_id, "error": error})
                completed_examples += examples_per_mode
                print(
                    f"[{completed_examples}/{total_examples}] {mode_index}/{len(modes)} "
                    f"{mode_id}: skipped ({error})",
                    file=sys.stderr, flush=True,
                )
                continue
            for example_index in range(examples_per_mode):
                example_number = example_index + 1
                print(
                    f"[{completed_examples + 1}/{total_examples}] {mode_index}/{len(modes)} "
                    f"{mode_id} example {example_number}/{examples_per_mode}: transmitting",
                    file=sys.stderr, flush=True,
                )
                rng = random.Random(f"{seed}:{mode_id}:{example_index}")
                suffix = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(8))
                payload = f"SIGID {mode_id.upper().replace('-', ' ')} {example_index + 1:03d} {suffix}"
                wav_path = audio_dir / mode_id / f"{example_index:04d}.wav"
                manifest_path = wav_path.with_suffix(".wav.json")
                try:
                    result = registry.generate("fldigi", ModemGenerationRequest(
                        mode_id=mode_id, native_mode=str(mode["backend_name"]),
                        output_path=wav_path, payload=payload, sample_rate=12_000,
                        carrier_hz=carrier_hz, timeout_seconds=timeout_seconds,
                    ))
                    waveform_digest = hashlib.sha256(wav_path.read_bytes()).hexdigest()
                    labels = {
                        "modulation_family": mode["family"],
                        "modulation_method": mode["method"],
                        "protocol": mode["protocol"], "variant": mode["variant"],
                    }
                    parameters = {
                        "backend": "fldigi", "native_mode": result.native_mode,
                        "fldigi_version": result.version, "transmit_payload": payload,
                        "carrier_hz": carrier_hz, "estimated_bandwidth_hz": band,
                        "frequency_low_hz": low, "frequency_high_hz": high,
                        "waveform_sha256": waveform_digest, "seed": seed,
                        "example_index": example_index,
                        "capture_sample_rate": 48_000, "capture_channels": 2,
                        "audio_path": "isolated virtual null sink",
                    }
                    manifest = {
                        "schema_version": 1, "source_kind": "synthetic",
                        "generator": "sig-id-fldigi", "generator_version": "0.1.0",
                        "deterministic": True, "mode_id": mode_id,
                        "reference_text": payload, "sample_rate": result.sample_rate,
                        "sample_count": result.sample_count, "channels": 1,
                        "sample_width_bytes": 2, "labels": labels,
                        "parameters": parameters,
                    }
                    wav_path.parent.mkdir(parents=True, exist_ok=True)
                    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                    imported = import_synthetic_wav(
                        wav_path, manifest_path, store=store,
                        capture_dir=data_root / "captures",
                    )
                    generated.append({"mode_id": mode_id, "capture_id": imported["capture"]["id"],
                                      "sha256": imported["capture"]["sha256"], "sample_count": result.sample_count})
                    completed_examples += 1
                    print(
                        f"[{completed_examples}/{total_examples}] {mode_id} example {example_number}: "
                        f"saved ({result.sample_count} samples)",
                        file=sys.stderr, flush=True,
                    )
                except Exception as exc:
                    failures.append({"mode_id": mode_id, "example": str(example_index), "error": str(exc)})
                    wav_path.unlink(missing_ok=True)
                    manifest_path.unlink(missing_ok=True)
                    completed_examples += 1
                    print(
                        f"[{completed_examples}/{total_examples}] {mode_id} example {example_number}: "
                        f"failed ({exc})",
                        file=sys.stderr, flush=True,
                    )
    finally:
        registry.close()
    summary = {
        "backend": "fldigi", "mode_count": len(modes), "examples_per_mode": examples_per_mode,
        "generated_count": len(generated), "generated": generated, "failures": failures,
        "data_dir": str(data_root), "output_dir": str(output),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "fldigi-corpus.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate classified audio samples using native Fldigi transmit modes")
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("SIG_ID_DATA_DIR", "~/.local/share/sig-id")).expanduser())
    parser.add_argument("--output-dir", type=Path, default=Path("data/fldigi-corpus"))
    parser.add_argument("--mode", default="all", help="all, missing (skip modes with a saved Fldigi reference), or one exact registry ID, display name, or Fldigi mode name")
    parser.add_argument("--examples-per-mode", type=int, default=1)
    parser.add_argument("--seed", type=int, default=4192026)
    parser.add_argument("--carrier-hz", type=float, default=1500.0)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args()
    print(json.dumps(build_fldigi_corpus(
        args.data_dir, args.output_dir, mode_query=args.mode,
        examples_per_mode=args.examples_per_mode, seed=args.seed,
        carrier_hz=args.carrier_hz, timeout_seconds=args.timeout_seconds,
    ), indent=2))
