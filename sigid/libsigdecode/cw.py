"""Independent first-pass CW demodulator for selected audio tone windows.

Uses the conventional Morse alphabet and timing ratios (dot/dash 1:3,
intra-character/letter/word gaps 1:3:7), not source code from other
decoder projects. This is a deliberately conservative experimental
starting point, not an automatic multi-station skimmer or a language model.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

from .core import DecodeWindow, DecodedToken, DecoderSpec, LocalDecodeResult

MORSE = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D",
    ".": "E", "..-.": "F", "--.": "G", "....": "H",
    "..": "I", ".---": "J", "-.-": "K", ".-..": "L",
    "--": "M", "-.": "N", "---": "O", ".--.": "P",
    "--.-": "Q", ".-.": "R", "...": "S", "-": "T",
    "..-": "U", "...-": "V", ".--": "W", "-..-": "X",
    "-.--": "Y", "--..": "Z",
    "-----": "0", ".----": "1", "..---": "2", "...--": "3",
    "....-": "4", ".....": "5", "-....": "6", "--...": "7",
    "---..": "8", "----.": "9",
    ".-.-.-": ".", "--..--": ",", "..--..": "?",
    "-..-.": "/", "-....-": "-", "-...-": "=", ".----.": "'",
    "-.--.": "(", "-.--.-": ")", "---...": ":",
    "-.-.--": "!", ".-..-.": '"', ".--.-.": "@",
}


@dataclass(frozen=True, slots=True)
class CWSettings:
    wpm: float | None = None
    baseband_cutoff_hz: float = 45.0

    def __post_init__(self) -> None:
        if self.wpm is not None and (not math.isfinite(self.wpm) or not 5 <= self.wpm <= 80):
            raise ValueError("CW speed must be between 5 and 80 WPM")
        if not 15 <= self.baseband_cutoff_hz <= 90:
            raise ValueError("CW baseband cutoff must be from 15 to 90 Hz")


def _runs(state: np.ndarray) -> list[tuple[bool, int, int]]:
    # All windows use a 500 Hz envelope after decimation.
    edges = np.flatnonzero(np.diff(np.r_[False, state, False]))
    return [(bool(state[a]), int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def _clean_keying(state: np.ndarray, *, minimum: int = 4) -> np.ndarray:
    output = state.copy()
    # Close tiny dropouts then remove isolated clicks. Never invent long marks.
    for expect, fill in [(False, True), (True, False)]:
        for on, a, b in _runs(output):
            if on == expect and b - a < minimum and a > 0 and b < len(output):
                if bool(output[a - 1]) == bool(output[b]):
                    output[a:b] = fill
    return output


def _symbol_decisions(
    marks: list[tuple[int, int]], dot: float, sample_start: int, scale: float,
) -> tuple[tuple[DecodedToken, ...], str, float]:
    if not marks:
        return (), "", 0.0
    tokens: list[DecodedToken] = []
    symbols: list[str] = []
    start = marks[0][0]
    end = marks[0][1]
    distances: list[float] = []
    words: list[str] = []

    def flush() -> None:
        nonlocal symbols, start, end
        if not symbols:
            return
        pattern = "".join(symbols)
        letter = MORSE.get(pattern, "?")
        # Timing-fit score is a *heuristic*, never a calibrated probability.
        fit = 1.0 - min(1.0, float(np.mean(distances[-len(symbols):])) / 0.75)
        tokens.append(DecodedToken(
            letter,
            sample_start + int(round(start * scale)),
            sample_start + int(round(end * scale)),
            pattern, round(max(0.05, fit), 3),
        ))
        words.append(letter)
        symbols = []

    for i, (a, b) in enumerate(marks):
        length = (b - a) / dot
        dash = length >= 1.85
        symbols.append("-" if dash else ".")
        distances.append(abs(length - (3 if dash else 1)))
        end = b
        if i + 1 == len(marks):
            flush()
            continue
        gap = (marks[i + 1][0] - b) / dot
        if gap >= 2:
            flush()
            if gap >= 5:
                words.append(" ")
            start = marks[i + 1][0]
    text = "".join(words).strip()
    unknown = sum(item.text == "?" for item in tokens)
    return tuple(tokens), text, unknown / len(tokens) if tokens else 1.0


class CWDecoder:
    spec = DecoderSpec(
        id="cw", name="CW (Morse)", family="Morse",
        stage="experimental", version="cw-envelope-v1",
    )

    def __init__(self, settings: CWSettings | None = None) -> None:
        self.settings = settings or CWSettings()

    def decode(self, window: DecodeWindow) -> LocalDecodeResult:
        def empty(reason: str, evidence: dict | None = None) -> LocalDecodeResult:
            return LocalDecodeResult(
                self.spec.id, self.spec.version, "no_decode",
                evidence=evidence or {}, note=reason,
            )
        if window.carrier_hz is None:
            return empty("Choose the audio carrier of a single CW signal; automatic multi-carrier selection is not implemented.")
        if window.samples.size < window.sample_rate // 2:
            return empty("At least 0.5 seconds of audio is needed.")
        rate = window.sample_rate
        carrier = float(window.carrier_hz)
        # Complex downmix and independent low-pass filtering reject other
        # audio-frequency stations without altering the original capture.
        n = np.arange(window.samples.size, dtype=np.float64)
        mixed = window.samples * np.exp(-2j * np.pi * carrier * n / rate)
        cutoff = min(self.settings.baseband_cutoff_hz, 0.35 * rate)
        sos = butter(4, cutoff, btype="lowpass", fs=rate, output="sos")
        signal = sosfiltfilt(sos, mixed)
        step = max(1, int(rate // 500))
        # Summing over short blocks stabilizes marks without a symbol clock.
        count = signal.size // step
        if count < 100:
            return empty("Signal too short for the envelope estimator.")
        envelope = np.abs(signal[:count * step].reshape(count, step).mean(axis=1))
        envelope = uniform_filter1d(envelope, size=max(1, round(0.006 * rate / step)))
        lo, hi = [float(x) for x in np.percentile(envelope, [10, 90])]
        if hi < 0.00015 or hi <= max(0.0001, lo * 3.0):
            return empty("No reliable on/off keyed tone contrast.", {
                "carrier_hz": carrier, "envelope_on_off_ratio": round(hi / max(lo, 1e-8), 3),
            })
        # Hysteresis reduces rapid threshold crossings in weak keyed tones.
        threshold_on = lo + 0.43 * (hi - lo)
        threshold_off = lo + 0.30 * (hi - lo)
        key = np.empty(count, dtype=np.bool_)
        pressed = False
        for i, value in enumerate(envelope):
            pressed = value >= (threshold_off if pressed else threshold_on)
            key[i] = pressed
        key = _clean_keying(key, minimum=max(3, int(0.008 * rate / step)))
        spans = [(a, b) for pressed, a, b in _runs(key) if pressed]
        if len(spans) < 5:
            return empty("Too few keyed marks to distinguish CW from carrier or incidental pulses.", {
                "mark_count": len(spans), "carrier_hz": carrier,
            })
        mark_lengths = np.array([b - a for a, b in spans], dtype=float)
        if self.settings.wpm is not None:
            dot = (1.2 / self.settings.wpm) * rate / step
        else:
            # Typical Morse contains short marks. Estimate from the
            # shortest *cluster*, not the absolute minimum (which can
            # be an edge transient or an RFI pulse).
            shortest = float(np.percentile(mark_lengths, 10))
            short = mark_lengths[mark_lengths <= shortest * 1.5]
            dot = float(np.median(short)) if len(short) else shortest
        wpm = 1.2 / (dot * step / rate)
        if not (5 <= wpm <= 80):
            return empty("Estimated Morse speed is outside 5–80 WPM.", {
                "estimated_wpm": round(wpm, 2), "mark_count": len(spans),
            })
        tokens, text, error_fraction = _symbol_decisions(spans, dot, window.sample_start, step)
        mean_fit = float(np.mean([token.confidence_hint for token in tokens])) if tokens else 0.0
        # A strong carrier can produce many spurious mark edges. A handful
        # of individually valid "." or "-" tokens is NOT convincing Morse.
        # Reject windows with inconsistent mark durations even when the
        # text happens to contain recognizable letters.
        evidence = {
            "carrier_hz": carrier, "mark_count": len(spans),
            "estimated_wpm": round(wpm, 2),
            "dot_ms": round(dot * step / rate * 1000, 2),
            "envelope_on_off_ratio": round(hi / max(lo, 1e-8), 2),
            "unknown_token_fraction": round(error_fraction, 3),
            "mean_symbol_timing_fit": round(mean_fit, 3),
            "confidence_kind": "timing_fit_heuristic_not_probability",
            "bit_and_frame_integrity": "not_applicable_to_cw",
            "source_window_start_sample": window.sample_start,
            "source_window_sample_count": int(window.samples.size),
        }
        if len(tokens) < 2 or error_fraction > 0.55 or mean_fit < 0.65:
            # Keep partial symbols and the untrusted interpretation available
            # in metadata, but do not emit a decoded-text product.
            evidence["unverified_pattern_preview"] = text[:120]
            return LocalDecodeResult(
                self.spec.id, self.spec.version, "no_decode",
                tokens=tokens, evidence=evidence,
                note="Morse mark timing is inconsistent or incomplete; inspect candidate symbols.",
            )
        return LocalDecodeResult(
            self.spec.id, self.spec.version,
            "decoded" if error_fraction == 0 else "partial",
            text=text, tokens=tokens, evidence=evidence,
            note="Experimental single-carrier CW interpretation; not independently verified.",
        )
