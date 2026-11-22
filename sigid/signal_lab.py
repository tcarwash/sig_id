"""Deterministic receive-only synthetic radio audio lab.

Produces a bounded, repeatable audio passband from pinned independent modem
fixtures. An optional continuous player targets ONLY a named PulseAudio
module-null-sink; it never selects default output or sends RF/PTT.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import wave

import numpy as np

RATE = 48_000
SINK = "sigid_signal_lab"
SOURCE = SINK + ".monitor"
DEFAULT_MODES = ("rtty", "bpsk31", "ax25")
FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "decoder_conformance"
STATE = Path.home() / ".local" / "state" / "sig-id-signal-lab"
MIN_HZ = 160.0
MAX_HZ = 2900.0

# Relative occupied edges. AX.25's reference is its 1200-Hz *mark* tone,
# not the middle of the 1200/2200-Hz Bell 202 tone pair.
PROFILES = {
    "cw": (-90.0, 90.0, 260.0),
    "rtty": (-135.0, 135.0, 740.0),
    "bpsk31": (-70.0, 70.0, 2520.0),
    # The current native AX.25 implementation uses fixed 1200/2200 Hz tones;
    # changing only DecodeWindow.carrier_hz does not retune that decoder.
    "ax25": (-100.0, 1100.0, 1200.0),
}
SLUG = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class Fixture:
    mode: str
    path: Path
    fixture_id: str
    sha256: str
    source_carrier_hz: float
    expected_text: str
    expected_kind: str


@dataclass(frozen=True)
class Station:
    fixture: Fixture
    group: int
    carrier_hz: float
    edge_low_hz: float
    edge_high_hz: float


def _within(path: Path, root: Path) -> bool:
    return path.is_relative_to(root)


def discover(root: Path = FIXTURES) -> dict[str, Fixture]:
    """List *only* synthetic positive mono-WAV fixtures with a known carrier."""
    root = root.resolve()
    result: dict[str, Fixture] = {}
    if not root.exists():
        return result
    for manifest in sorted(root.glob("*/manifest.json")):
        try:
            spec = json.loads(manifest.read_text())
        except (ValueError, OSError):
            continue
        for case in spec.get("fixtures", []):
            mode = str(case.get("decoder_id", ""))
            provenance = case.get("provenance", {})
            expected = case.get("expected", {})
            audio = case.get("audio", {})
            carrier = case.get("carrier_hz")
            if (not SLUG.fullmatch(mode) or mode in result
                    or provenance.get("kind") != "synthetic"
                    or expected.get("kind") != "positive"
                    or not (isinstance(expected.get("text"), str) and expected["text"]
                            or isinstance(expected.get("frames"), list) and expected["frames"])
                    or not isinstance(audio, dict)
                    or not isinstance(audio.get("path"), str)
                    or not isinstance(audio.get("sha256"), str)
                    or not re.fullmatch("[0-9a-f]{64}", audio["sha256"])
                    or not isinstance(carrier, (float, int))
                    or not np.isfinite(carrier) or carrier <= 0):
                continue
            path = (manifest.parent / audio["path"]).resolve()
            if not _within(path, manifest.parent.resolve()) or not path.is_file():
                continue
            result[mode] = Fixture(mode, path, str(case.get("id", "")),
                                   audio["sha256"], float(carrier),
                                   str(expected.get("text", "")), "synthetic")
            break
    return result


def read_fixture(fixture: Fixture) -> tuple[np.ndarray, int]:
    """Check fixture integrity and convert a bounded mono PCM16 WAV to float."""
    if fixture.path.stat().st_size > 100 * 1024 * 1024:
        raise ValueError(f"{fixture.mode}: audio file exceeds 100 MiB")
    digest = hashlib.sha256(fixture.path.read_bytes()).hexdigest()
    if digest != fixture.sha256:
        raise ValueError(f"{fixture.mode}: WAV SHA-256 mismatch ({digest})")
    with wave.open(str(fixture.path), "rb") as source:
        if (source.getnchannels() != 1 or source.getsampwidth() != 2
                or source.getcomptype() != "NONE"):
            raise ValueError(f"{fixture.mode}: expected uncompressed PCM16 mono WAV")
        rate = source.getframerate()
        if not 4000 <= rate <= 192000 or source.getnframes() > rate * 120:
            raise ValueError(f"{fixture.mode}: unsupported duration or sample rate")
        samples = np.frombuffer(source.readframes(source.getnframes()),
                                dtype="<i2").astype(np.float64) / 32768.0
    if not len(samples):
        raise ValueError(f"{fixture.mode}: empty audio")
    return samples, rate


def occupied_edges(fixture: Fixture) -> tuple[float, float, float]:
    if fixture.mode in PROFILES:
        return PROFILES[fixture.mode]
    # For other modes, conservatively estimate support using FFT energy
    # above the stationary noise floor. Profiles can be added after field
    # testing; this is placement guidance, not a measured RF bandwidth.
    samples, rate = read_fixture(fixture)
    from scipy.signal import welch
    freqs, powers = welch(samples, fs=rate, nperseg=min(8192, len(samples)))
    if len(powers) < 4 or not max(powers):
        raise ValueError(f"{fixture.mode}: no spectral energy")
    floor = float(np.percentile(powers, 25))
    corrected = np.maximum(powers - floor, 0)
    peaks = corrected > max(float(max(corrected)) * 0.0005, floor * 15)
    if not np.any(peaks):
        raise ValueError(f"{fixture.mode}: could not estimate passband occupancy")
    lo = float(freqs[np.flatnonzero(peaks)[0]])
    hi = float(freqs[np.flatnonzero(peaks)[-1]])
    low = min(lo - fixture.source_carrier_hz - 40, -80.0)
    high = max(hi - fixture.source_carrier_hz + 40, 80.0)
    return low, high, 0.5 * (MIN_HZ + MAX_HZ) - (low + high) / 2


def parse_carriers(text: str, selected: list[str]) -> dict[str, float]:
    overrides = {}
    for part in filter(None, text.split(",")):
        mode, separator, value = part.partition("=")
        mode = mode.strip().lower()
        if not separator or mode not in selected or mode in overrides:
            raise ValueError(f"invalid or duplicate --carriers entry {part!r}")
        try:
            freq = float(value)
        except ValueError as exc:
            raise ValueError(f"invalid carrier frequency: {part!r}") from exc
        if not np.isfinite(freq) or freq <= 0:
            raise ValueError(f"invalid carrier frequency: {part!r}")
        overrides[mode] = freq
    return overrides


def layout(fixtures: list[Fixture], overrides: dict[str, float],
           *, guard_hz: float = 85.0, overlap: bool = False) -> list[Station]:
    """Greedy pack into frequency lanes, time-sharing only when necessary."""
    groups: list[list[Station]] = [[]]
    stations: list[Station] = []
    for fixture in fixtures:
        dlo, dhi, preferred = occupied_edges(fixture)
        if fixture.mode in overrides:
            candidates = [overrides[fixture.mode]]
        else:
            # Favour predictable lanes and then use any remaining opening.
            candidates = [preferred] + [float(f) for f in range(240, 2810, 20)]
        selected = None
        for group_index in range(len(groups) + 1):
            peers = groups[group_index] if group_index < len(groups) else []
            for carrier in candidates:
                low, high = carrier + dlo, carrier + dhi
                if low < MIN_HZ or high > MAX_HZ:
                    continue
                if overlap or all(
                    low >= p.edge_high_hz + guard_hz
                    or high <= p.edge_low_hz - guard_hz
                    for p in peers
                ):
                    selected = Station(fixture, group_index, round(carrier, 3),
                                       round(low, 3), round(high, 3))
                    break
            if selected is not None:
                if group_index == len(groups):
                    groups.append([])
                groups[group_index].append(selected)
                break
        if selected is None:
            raise ValueError(f"{fixture.mode}: cannot fit chosen carrier into "
                             f"{MIN_HZ:.0f}–{MAX_HZ:.0f} Hz passband")
        stations.append(selected)
    return stations


def _shift_audio(samples: np.ndarray, source_carrier: float, target: float,
                 rate: int) -> np.ndarray:
    from scipy.signal import hilbert
    delta = target - source_carrier
    if abs(delta) < 0.0001:
        return samples.copy()
    phase = (2 * np.pi * delta / rate) * np.arange(len(samples))
    return np.real(hilbert(samples) * np.exp(1j * phase))


def wav_write(path: Path, samples: np.ndarray, rate: int, *,
              channels: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if channels not in (1, 2):
        raise ValueError("channels must be 1 or 2")
    pcm = np.rint(np.clip(samples, -1, 1) * 32767).astype("<i2")
    if channels == 2:
        pcm = np.repeat(pcm[:, None], 2, axis=1).reshape(-1)
    with wave.open(str(path), "wb") as dest:
        dest.setnchannels(channels)
        dest.setsampwidth(2)
        dest.setframerate(rate)
        dest.writeframes(pcm.tobytes())


def render(*, modes: list[str], output: Path, fixture_root: Path = FIXTURES,
           duration: float = 48.0, sample_rate: int = RATE, seed: int = 71062,
           snr_db: float = 19.0, level: float = 0.17,
           snr_reference_hz: float | None = None,
           gap_seconds: float = 0.9, carrier_overrides: dict[str, float] | None = None,
           allow_overlap: bool = False, stereo: bool = True,
           traffic: str = "staggered", max_active: int = 1,
           bursts_per_mode: int = 2, qrm: bool | None = None) -> dict:
    if not 15 <= duration <= 300:
        raise ValueError("duration must be 15–300 seconds")
    if sample_rate not in (12000, 24000, 48000):
        raise ValueError("sample rate must be 12000, 24000 or 48000")
    if not np.isfinite(snr_db) or not -30 <= snr_db <= 55:
        raise ValueError("SNR must be finite and between -30 and +55 dB")
    if not np.isfinite(level) or not 0.02 <= level <= 0.3:
        raise ValueError("signal level must be finite and between 0.02 and 0.3 RMS")
    # Default to the historical full Nyquist noise bandwidth. A narrower
    # reference, such as 2500 Hz for weak-signal ham measurements, requires
    # *more* full-band AWGN to achieve the requested in-band SNR.
    reference_hz = sample_rate / 2 if snr_reference_hz is None else snr_reference_hz
    if not np.isfinite(reference_hz) or not 100 <= reference_hz <= sample_rate / 2:
        raise ValueError("SNR reference bandwidth must be 100 Hz through Nyquist")
    if not 0.2 <= gap_seconds <= 10:
        raise ValueError("gap must be between 0.2 and 10 seconds")
    if traffic not in {"staggered", "sparse", "parallel"}:
        raise ValueError("traffic must be staggered, sparse or parallel")
    if max_active not in (1, 2):
        raise ValueError("--max-active must be 1 or 2")
    if not isinstance(bursts_per_mode, int) or not 1 <= bursts_per_mode <= 12:
        raise ValueError("--bursts-per-mode must be between 1 and 12")
    if qrm is None:
        qrm = traffic != "sparse"
    if not modes or len(modes) != len(set(modes)):
        raise ValueError("supply at least one distinct mode")
    available = discover(fixture_root)
    missing = sorted(set(modes) - set(available))
    if missing:
        raise ValueError(f"no verified synthetic positive PCM16 fixture for {missing}; "
                         f"use 'list' or supply --fixtures-root")
    chosen = [available[m] for m in modes]
    stations = layout(chosen, carrier_overrides or {}, overlap=allow_overlap)
    group_count = max(s.group for s in stations) + 1
    block_seconds = duration / group_count
    if block_seconds < 14.5:
        raise ValueError("duration too short for the number of frequency groups")
    total_frames = round(duration * sample_rate)
    audio = np.zeros(total_frames, dtype=np.float64)
    rng = np.random.default_rng(seed)
    truth: list[dict] = []
    prepared: dict[Station, np.ndarray] = {}
    for station in stations:
        samples, original_rate = read_fixture(station.fixture)
        from scipy.signal import resample_poly
        from math import gcd
        divisor = gcd(sample_rate, original_rate)
        if sample_rate != original_rate:
            samples = resample_poly(samples, sample_rate // divisor,
                                     original_rate // divisor)
        shifted = _shift_audio(samples, station.fixture.source_carrier_hz,
                               station.carrier_hz, sample_rate)
        rms = float(np.sqrt(np.mean(shifted * shifted)))
        if rms < 1e-4:
            raise ValueError(f"{station.fixture.mode}: silent fixture")
        shifted = shifted * (level / rms)
        # Smooth fixture endpoints so hard station starts do not click.
        ramp = min(round(0.006 * sample_rate), len(shifted) // 6)
        if ramp:
            envelope = np.sin(np.linspace(0, np.pi / 2, ramp)) ** 2
            shifted[:ramp] *= envelope
            shifted[-ramp:] *= envelope[::-1]
        prepared[station] = shifted

    def emit(station: Station, at: float) -> None:
        shifted = prepared[station]
        start = round(at * sample_rate)
        end = start + len(shifted)
        audio[start:end] += shifted
        truth.append({
            "mode": station.fixture.mode, "fixture_id": station.fixture.fixture_id,
            "fixture_sha256": station.fixture.sha256,
            "source": "synthetic-fixture", "carrier_hz": station.carrier_hz,
            "occupied_low_hz": station.edge_low_hz,
            "occupied_high_hz": station.edge_high_hz,
            "start_sample": start, "sample_count": len(shifted),
            "start_seconds": round(start / sample_rate, 4),
            "end_seconds": round(end / sample_rate, 4),
            "expected_text": station.fixture.expected_text,
        })

    for group in range(group_count):
        peers = [s for s in stations if s.group == group]
        window_start = group * block_seconds
        window_end = min(duration, window_start + block_seconds) - 0.5
        if traffic == "sparse":
            # Evenly distributed transmissions with an exact cap of one or
            # two active stations. This leaves real noise-only intervals and
            # avoids packing tens of repeated bursts into a 30-second view.
            if max_active == 1:
                batches = [[station] for station in peers]
            else:
                # Keep the widest-band mode away from the nearest narrow
                # signals. Fixed-tone AX.25 in particular has no channel
                # isolation yet, so pair it with the largest spectral gap.
                remaining = list(peers)
                batches = []
                while remaining:
                    widest = max(remaining, key=lambda s: s.edge_high_hz - s.edge_low_hz)
                    remaining.remove(widest)
                    if remaining:
                        farthest = max(remaining, key=lambda s: max(
                            0.0, s.edge_low_hz - widest.edge_high_hz,
                            widest.edge_low_hz - s.edge_high_hz))
                        remaining.remove(farthest)
                        batches.append([widest, farthest])
                    else:
                        batches.append([widest])
            slots = len(batches) * bursts_per_mode
            slot_seconds = block_seconds / slots
            for repetition in range(bursts_per_mode):
                for batch_number, batch in enumerate(batches):
                    position = repetition * len(batches) + batch_number
                    at = window_start + position * slot_seconds + 1.0
                    end = at + max(len(prepared[s]) / sample_rate for s in batch)
                    if end > min(window_end, window_start + (position + 1) * slot_seconds - 0.15):
                        raise ValueError("sparse pattern has insufficient room for complete "
                                         "modem frames; increase --duration, reduce "
                                         "--bursts-per-mode or --max-active")
                    for station in batch:
                        emit(station, at)
        elif traffic == "staggered":
            # Reliable baseline: different occupied frequencies, one TX at
            # a time. This keeps short CRC-checked AX.25 frames intact, even
            # when experimental decoder front ends lack interference filters.
            cursor = window_start + 1.0
            turn = 0
            while cursor < window_end:
                station = peers[turn % len(peers)]
                length_seconds = len(prepared[station]) / sample_rate
                if cursor + length_seconds > window_end:
                    break
                emit(station, cursor)
                cursor += length_seconds + gap_seconds
                turn += 1
        else:
            # Crowded stress test: independent simultaneous transmissions.
            # Some experimental decoders will abstain or misdecode.
            for lane_index, station in enumerate(peers):
                shifted = prepared[station]
                at = window_start + 1.0 + 0.54 * lane_index
                interval = len(shifted) / sample_rate + gap_seconds + (lane_index % 4) * 0.47
                while at + len(shifted) / sample_rate <= window_end:
                    emit(station, at)
                    at += interval
    for station in stations:
        if not any(item["mode"] == station.fixture.mode for item in truth):
            raise ValueError(f"{station.fixture.mode}: no complete transmission fits "
                             "in the requested duration; increase --duration")
    reference_noise_rms = level / (10 ** (snr_db / 20))
    noise_rms = reference_noise_rms * np.sqrt((sample_rate / 2) / reference_hz)
    audio += rng.normal(0, noise_rms, total_frames)
    # Continuous fixed QRM lines can dominate a sparse waterfall and trigger
    # narrow-tone false positives, so omit them in sparse mode by default.
    # Keep QRM available explicitly as a stressor.
    if qrm:
        t = np.arange(total_frames) / sample_rate
        audio += 0.5 * noise_rms * np.sin(2 * np.pi * 735 * t + 0.3)
        audio += 0.25 * noise_rms * np.sin(2 * np.pi * 2675 * t + 0.7)
    peak_before_scaling = float(np.max(np.abs(audio)))
    gain = min(1.0, 0.94 / max(peak_before_scaling, 1e-12))
    audio *= gain
    output = output.resolve()
    wav_write(output, audio, sample_rate)
    stereo_path = output.with_name(output.stem + "-stereo.wav") if stereo else None
    if stereo_path is not None:
        wav_write(stereo_path, audio, sample_rate, channels=2)
    truth.sort(key=lambda item: (item["start_sample"], item["mode"]))
    info = {
        "format_version": 1, "provenance": "synthetic-not-on-air",
        "source": "verified-sha256-pinned-positive-conformance-fixtures",
        "rate_hz": sample_rate, "duration_seconds": duration, "seed": seed,
        "traffic": traffic,
        "max_active": max_active if traffic == "sparse" else (1 if traffic == "staggered" else len(stations)),
        "bursts_per_mode": bursts_per_mode if traffic == "sparse" else None,
        "continuous_qrm": qrm,
        "snr_db_nominal_per_station": snr_db,
        "snr_reference_hz": reference_hz,
        "snr_definition": "signal RMS power relative to AWGN integrated over SNR reference bandwidth",
        "signal_rms_before_overlap": level,
        "signal_rms_after_global_gain": round(level * gain, 8),
        "background_noise_rms": round(noise_rms, 8),
        "background_noise_rms_after_global_gain": round(noise_rms * gain, 8),
        "reference_noise_rms_before_global_gain": round(reference_noise_rms, 8),
        "peak_before_scaling": round(peak_before_scaling, 5),
        "master_gain": round(gain, 6),
        "fixture_root": str(fixture_root.resolve()),
        "passband_hz": [MIN_HZ, MAX_HZ],
        "frequency_groups": group_count,
        "stations": [
            {"mode": s.fixture.mode, "carrier_hz": s.carrier_hz, "group": s.group,
             "occupied_low_hz": s.edge_low_hz, "occupied_high_hz": s.edge_high_hz,
             "fixture": s.fixture.fixture_id}
            for s in stations
        ],
        "transmissions": truth, "mono_wav": str(output),
        "stereo_wav": str(stereo_path) if stereo_path else None,
        "warning": "Synthetic decoder test stimuli only; not received radio evidence.",
    }
    info_path = output.with_suffix(".json")
    info_path.write_text(json.dumps(info, indent=2) + "\n")
    return info


def pulse_env() -> dict[str, str]:
    env = dict(os.environ)
    runtime = env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    env.setdefault("PULSE_SERVER", f"unix:{runtime}/pulse/native")
    return env


def _pactl(*args: str) -> str:
    completed = subprocess.run(["pactl", *args], capture_output=True, text=True,
                               timeout=10, env=pulse_env(), check=False)
    if completed.returncode:
        raise RuntimeError(f"pactl {' '.join(args)}: {completed.stderr.strip()}")
    return completed.stdout


def ensure_null_sink() -> None:
    # Never touch the default sink. Check exact name, then create a *null*
    # sink and require the corresponding capture monitor.
    sinks = [line.split("\t")[1] for line in _pactl("list", "short", "sinks").splitlines()
             if len(line.split("\t")) > 1]
    if SINK not in sinks:
        _pactl("load-module", "module-null-sink", f"sink_name={SINK}",
               "sink_properties=device.description=Signal_ID_Synthetic_Lab",
               "rate=48000", "channels=2")
    # A malicious or misconfigured physical device must never be accepted
    # just because its visible sink name collides with our test sink.
    details = _pactl("list", "sinks")
    section = next((chunk for chunk in re.split(r"(?m)^(?=Sink #)", details)
                    if re.search(r"(?m)^\s*Name: sigid_signal_lab\s*$", chunk)), "")
    if not section or not (
        'factory.name = "support.null-audio-sink"' in section
        or "Driver: module-null-sink.c" in section
    ):
        raise RuntimeError("refusing playback: named sink is not a verified null sink")
    sources = [line.split("\t")[1] for line in _pactl("list", "short", "sources").splitlines()
               if len(line.split("\t")) > 1]
    if SOURCE not in sources:
        raise RuntimeError(f"virtual capture monitor missing: {SOURCE}")


def _pidfile(state_dir: Path) -> Path:
    return state_dir / "player.pid"


def _worker_pid(state_dir: Path) -> int | None:
    try:
        pid = int(_pidfile(state_dir).read_text().strip())
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
        if b"sigid.signal_lab" in cmd and b"_worker" in cmd:
            return pid
    except (ValueError, OSError):
        pass
    return None


def _worker(wav: Path, state_dir: Path) -> None:
    ensure_null_sink()
    state_dir.mkdir(parents=True, exist_ok=True)
    if not wav.is_file():
        raise RuntimeError(f"missing playback WAV: {wav}")
    _pidfile(state_dir).write_text(str(os.getpid()) + "\n")
    running = True
    child: subprocess.Popen | None = None

    def stop(_sig: int, _frame: object) -> None:
        nonlocal running
        running = False
        if child is not None and child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        while running:
            child = subprocess.Popen(
                ["paplay", "--device=" + SINK, "--client-name=SignalIDSyntheticLab",
                 "--stream-name=MultiModeTest_ReceiveOnly", str(wav)],
                env=pulse_env(), stdin=subprocess.DEVNULL)
            code = child.wait()
            if not running:
                break
            if code != 0:
                raise RuntimeError(f"paplay exited {code}; refusing silent retry loop")
            child = None
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
        if _pidfile(state_dir).is_file():
            try:
                if int(_pidfile(state_dir).read_text().strip()) == os.getpid():
                    _pidfile(state_dir).unlink()
            except ValueError:
                pass


def start_loop(state_dir: Path, settings: argparse.Namespace) -> dict:
    state_dir = state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    if _worker_pid(state_dir) is not None:
        raise RuntimeError("a synthetic loop is already playing; use stop first")
    ensure_null_sink()
    modes = _parse_modes(settings.modes)
    carriers = parse_carriers(settings.carriers, modes)
    output = state_dir / "multi-mode-loop.wav"
    result = render(modes=modes, output=output, fixture_root=settings.fixtures_root,
                    duration=settings.duration, sample_rate=settings.rate,
                    seed=settings.seed, snr_db=settings.snr_db, level=settings.level,
                    snr_reference_hz=settings.snr_reference_hz,
                    gap_seconds=settings.gap, carrier_overrides=carriers,
                    allow_overlap=settings.allow_overlap, traffic=settings.traffic,
                    max_active=settings.max_active, bursts_per_mode=settings.bursts_per_mode,
                    qrm=settings.qrm)
    log = (state_dir / "player.log").open("ab")
    with log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "sigid.signal_lab", "_worker",
             str(Path(result["stereo_wav"])), str(state_dir)],
            cwd=str(Path(__file__).resolve().parents[1]), env=pulse_env(),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    for _ in range(15):
        if _worker_pid(state_dir) == proc.pid:
            break
        if proc.poll() is not None:
            raise RuntimeError(f"player failed to start; inspect {state_dir / 'player.log'}")
        time.sleep(0.1)
    if _worker_pid(state_dir) != proc.pid:
        raise RuntimeError(f"player did not start; inspect {state_dir / 'player.log'}")
    result["player_pid"] = proc.pid
    result["input_device"] = SOURCE
    return result


def stop_loop(state_dir: Path) -> bool:
    pid = _worker_pid(state_dir.resolve())
    if pid is None:
        return False
    os.kill(pid, signal.SIGTERM)
    for _ in range(30):
        if _worker_pid(state_dir.resolve()) is None:
            break
        time.sleep(0.1)
    return True


def _parse_modes(value: str) -> list[str]:
    modes = [mode.strip().lower() for mode in value.split(",")]
    if not modes or len(set(modes)) != len(modes) or any(not SLUG.fullmatch(m) for m in modes):
        raise ValueError("use a comma-separated list of distinct mode IDs")
    return modes


def _add_render_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES),
                        help="comma-separated mode IDs, discover with list")
    parser.add_argument("--fixtures-root", type=Path, default=FIXTURES,
                        help="root directory containing per-mode conformance manifests")
    parser.add_argument("--carriers", default="",
                        help="optional exact audio carrier overrides: rtty=650,bpsk31=1050")
    parser.add_argument("--duration", type=float, default=100,
                        help="repeat WAV length, 15–300 seconds (default 100)")
    parser.add_argument("--rate", type=int, default=RATE, choices=(12000, 24000, 48000))
    parser.add_argument("--seed", type=int, default=71062)
    parser.add_argument("--snr-db", type=float, default=19,
                        help="station signal-to-AWGN SNR in dB (-30 to +55; default +19)")
    parser.add_argument("--snr-reference-hz", type=float, default=None,
                        help="noise reference bandwidth in Hz; default full Nyquist, "
                             "use 2500 for ham weak-signal SNR convention")
    parser.add_argument("--level", type=float, default=0.17,
                        help="each station's target RMS before overlap (default 0.17)")
    parser.add_argument("--gap", type=float, default=0.9,
                        help="idle seconds between bursts in staggered/parallel mode; sparse uses even slots")
    parser.add_argument("--traffic", choices=("staggered", "sparse", "parallel"),
                        default="sparse",
                        help="sparse one/two at a time (default), staggered continuous or parallel busy")
    parser.add_argument("--max-active", type=int, choices=(1, 2), default=1,
                        help="sparse traffic only: one or two simultaneous stations")
    parser.add_argument("--bursts-per-mode", type=int, default=2,
                        help="sparse traffic only: exact repeats of each modem per cycle (default 2)")
    parser.add_argument("--qrm", action=argparse.BooleanOptionalAction, default=None,
                        help="enable or disable continuous non-message interference tones; sparse defaults off")
    parser.add_argument("--allow-overlap", action="store_true",
                        help="allow intentionally overlapping channels for stress tests")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sig-id-signal-lab",
                                     description="Receive-only multi-modem synthetic audio lab")
    commands = parser.add_subparsers(dest="command", required=True)
    listings = commands.add_parser("list", help="list modes with usable pinned audio")
    listings.add_argument("--fixtures-root", type=Path, default=FIXTURES)
    rendering = commands.add_parser("render", help="write deterministic mono + stereo WAVs")
    _add_render_args(rendering)
    rendering.add_argument("--output", required=True, type=Path)
    starting = commands.add_parser("start", help="play a repeating test loop to a null sink")
    _add_render_args(starting)
    starting.add_argument("--state-dir", type=Path, default=STATE)
    stopping = commands.add_parser("stop", help="stop just this tool's player")
    stopping.add_argument("--state-dir", type=Path, default=STATE)
    status = commands.add_parser("status", help="check player state")
    status.add_argument("--state-dir", type=Path, default=STATE)
    worker = commands.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("wav", type=Path)
    worker.add_argument("state_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            found = discover(args.fixtures_root)
            if not found:
                raise ValueError(f"no pinned decodable fixtures found at {args.fixtures_root}")
            for mode, fixture in sorted(found.items()):
                text = fixture.expected_text[:30] or "[typed payload]"
                print(f"{mode:<20} source={fixture.source_carrier_hz:7.1f} Hz "
                      f"fixture={fixture.fixture_id:<27} truth={text!r}")
        elif args.command == "render":
            modes = _parse_modes(args.modes)
            info = render(modes=modes, output=args.output,
                          fixture_root=args.fixtures_root, duration=args.duration,
                          sample_rate=args.rate, seed=args.seed, snr_db=args.snr_db,
                          level=args.level, snr_reference_hz=args.snr_reference_hz,
                          gap_seconds=args.gap,
                          carrier_overrides=parse_carriers(args.carriers, modes),
                          allow_overlap=args.allow_overlap, traffic=args.traffic,
                          max_active=args.max_active, bursts_per_mode=args.bursts_per_mode,
                          qrm=args.qrm)
            print(json.dumps({"mono": info["mono_wav"], "stereo": info["stereo_wav"],
                              "stations": info["stations"],
                              "bursts": len(info["transmissions"]),
                              "truth": str(args.output.resolve().with_suffix('.json'))},
                             indent=2))
        elif args.command == "start":
            info = start_loop(args.state_dir, args)
            print(f"Playing on null sink {SINK}; Signal ID input {SOURCE}\n"
                  f"PID {info['player_pid']}; stations: {info['stations']}\n"
                  f"WAV {info['mono_wav']}")
        elif args.command == "status":
            pid = _worker_pid(args.state_dir.resolve())
            print(f"{'RUNNING pid=' + str(pid) if pid else 'STOPPED'} "
                  f"state_dir={args.state_dir.resolve()} source={SOURCE}")
        elif args.command == "stop":
            print("Stopped" if stop_loop(args.state_dir) else "Already stopped")
        elif args.command == "_worker":
            _worker(args.wav, args.state_dir)
    except (ValueError, RuntimeError, OSError, wave.Error) as exc:
        parser.exit(2, f"Signal Lab error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
