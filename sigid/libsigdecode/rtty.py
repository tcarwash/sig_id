"""Independent first-party RTTY (ITA2/Baudot) FSK decoder for mono audio.

Independently authored from the published alphabet in ITU-T S.1
(International Telegraph Alphabet No. 2) plus standard amateur RTTY
framing practice (1 start bit, 5 data bits LSB-first, 1/1.5/2 stop
bits). No source code, tables, or parameters are taken from OnAir,
Fldigi, WSJT-X, or any other decoder application; only the scholar's
facts of the public standard (code alphabet, bit order, mark/space
polarity convention) are used, the same way the CW module uses the
public Morse alphabet.

Receive-only: pure NumPy/SciPy analysis of the provided window. Never
opens hardware, tunes, transmits, or performs I/O.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, decimate, sosfiltfilt

from .core import DecodeWindow, DecodedToken, DecoderSpec, LocalDecodeResult

# 5-bit ITA2 code points keyed by integer value where bit k (value
# 2**k) is data unit k+1 in transmission order (LSB-first: the first
# bit after the start bit is bit 0). Each entry is
# (letters_char, figures_char). Control/shift codes map to None and
# are handled as functions, not text. ITU-T S.1 leaves F/G/H figure-case
# assignments unspecified; those entries are intentionally omitted.
_ITA2_LETTERS: dict[int, str | None] = {
    0: None,   # BLANK / NUL: no printed character
    1: "E", 2: None, 3: "A", 4: " ", 5: "S", 6: "I", 7: "U",
    8: None, 9: "D", 10: "R", 11: "J", 12: "N", 13: "F",
    14: "C", 15: "K", 16: "T", 17: "Z", 18: "L", 19: "W",
    20: "H", 21: "Y", 22: "P", 23: "Q", 24: "O", 25: "B",
    26: "G", 27: None, 28: "M", 29: "X", 30: "V", 31: None,
}
_ITA2_FIGURES: dict[int, str | None] = {
    0: None,   # BLANK / NUL
    1: "3", 2: None, 3: "-", 4: " ", 5: "'", 6: "8", 7: "7",
    8: None, 9: "$", 10: "4", 11: None, 12: ",", 13: None,
    14: ":", 15: "(", 16: "5", 17: "+", 18: ")", 19: "2",
    20: None, 21: "6", 22: "0", 23: "1", 24: "9", 25: "?",
    26: None, 27: None, 28: ".", 29: "/", 30: "=", 31: None,
}
# Function codes (never emit text themselves).
_BLANK = 0  # NUL: ignored
_LF = 2
_CR = 8
_FIGS = 27  # figures shift
_LTRS = 31  # letters shift (idle/diddle character: all mark)
_BEL = 11  # figures-case J rings the station bell; counted, not printed

SUPPORTED_BAUDS = (45.45, 50.0, 57.0, 75.0, 100.0)
SUPPORTED_SHIFTS_HZ = (170.0, 200.0, 425.0, 850.0)
SUPPORTED_STOP_BITS = (1.0, 1.4, 1.5, 2.0)

# Decision thresholds (heuristic gates, not calibrated probabilities).
_MIN_CHARS_EMIT = 3
_MIN_CHARS_DECODED = 6
_VALID_FRAC_DECODED = 0.85
_VALID_FRAC_PARTIAL = 0.60
_AMBIGUITY_MARGIN = 0.05
MAX_RTTY_SAMPLES = 8_000_000  # fixed DSP memory/CPU ceiling across sample rates


@dataclass(frozen=True, slots=True)
class RTTYSettings:
    baud: float = 45.45
    shift_hz: float = 170.0
    stop_bits: float = 1.5

    def __post_init__(self) -> None:
        if self.baud not in SUPPORTED_BAUDS:
            raise ValueError(f"RTTY baud must be one of {SUPPORTED_BAUDS}")
        if self.shift_hz not in SUPPORTED_SHIFTS_HZ:
            raise ValueError(f"RTTY shift must be one of {SUPPORTED_SHIFTS_HZ} Hz")
        if self.stop_bits not in SUPPORTED_STOP_BITS:
            raise ValueError(f"RTTY stop bits must be one of {SUPPORTED_STOP_BITS}")


def _bit_means(soft: np.ndarray, centers: np.ndarray, half: int) -> np.ndarray:
    """Mean soft value over [center-half, center+half] for each center."""
    n = soft.size
    lo = np.clip(centers - half, 0, n - 1)
    hi = np.clip(centers + half + 1, 0, n)
    cumsum = np.concatenate(([0.0], np.cumsum(soft)))
    widths = np.maximum(hi - lo, 1)
    return (cumsum[hi] - cumsum[lo]) / widths


def _frame_scores(
    soft: np.ndarray, bit_len: float, phase: float, stop_bits: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score every character frame on a grid anchored at phase (decimated idx).

    Returns (start_soft, data_mean_abs, stop_soft, char_len_bits) arrays;
    one entry per whole frame that fits inside the signal.
    """
    char_bits = 6.0 + stop_bits
    char_len = char_bits * bit_len
    n = soft.size
    # A frame is only scored when its complete stop interval fits. Otherwise
    # clipping the stop sample at the window edge can turn a truncated frame
    # into a false valid stop bit.
    n_chars = max(0, int(np.floor((n - phase) / char_len)))
    if n_chars <= 0:
        empty = np.empty(0)
        return empty, empty, empty, char_bits
    k = np.arange(n_chars, dtype=np.float64)
    start_c = np.round(phase + 0.5 * bit_len + k * char_len).astype(int)
    stop_c = np.round(phase + (6.0 + stop_bits / 2.0) * bit_len + k * char_len).astype(int)
    half = max(1, int(round(bit_len / 4.0)))
    starts = _bit_means(soft, np.clip(start_c, 0, n - 1), half)
    stops = _bit_means(soft, np.clip(stop_c, 0, n - 1), half)
    data = np.empty((n_chars, 5))
    for j in range(5):
        c = np.round(phase + (1.5 + j) * bit_len + k * char_len).astype(int)
        data[:, j] = _bit_means(soft, np.clip(c, 0, n - 1), half)
    return starts, np.mean(np.abs(data), axis=1), stops, np.full(n_chars, char_bits)


class RTTYDecoder:
    spec = DecoderSpec(
        id="rtty", name="RTTY (ITA2/Baudot) FSK", family="keyboard",
        stage="experimental", version="rtty-ita2-fsk-v2",
    )

    def __init__(self, settings: RTTYSettings | None = None) -> None:
        self.settings = settings or RTTYSettings()

    def decode(self, window: DecodeWindow) -> LocalDecodeResult:
        settings = self.settings
        baud = float(settings.baud)
        shift = float(settings.shift_hz)

        def empty(reason: str, evidence: dict | None = None) -> LocalDecodeResult:
            return LocalDecodeResult(
                self.spec.id, self.spec.version, "no_decode",
                evidence=evidence or {}, note=reason,
            )

        if window.samples.size > MAX_RTTY_SAMPLES:
            return empty(
                "Selected window exceeds the RTTY decoder's 8-million-sample "
                "resource limit; choose a shorter crop.",
                {"sample_count": int(window.samples.size),
                 "max_samples": MAX_RTTY_SAMPLES},
            )

        if window.carrier_hz is None:
            return empty("Choose the audio carrier of a single RTTY signal; "
                         "automatic multi-carrier selection is not implemented.")
        rate = float(window.sample_rate)
        carrier = float(window.carrier_hz)
        mark_hz = carrier + shift / 2.0
        space_hz = carrier - shift / 2.0
        if not (mark_hz < 0.45 * rate and space_hz >= 5.0):
            return empty("Mark/space tones do not fit inside the audio passband "
                         "for this carrier and shift.", {
                             "carrier_hz": carrier, "shift_hz": shift,
                             "mark_hz": round(mark_hz, 2), "space_hz": round(space_hz, 2),
                         })
        # Longest supported character is 8 bits (2 stop bits); require at
        # least two of them before any DSP work.
        if window.samples.size < int(np.ceil(2.0 * 8.0 / baud * rate)):
            return empty("Too few samples for an RTTY character at this baud rate.")

        # Bound memory/CPU: analyse at most ~16 kHz internal rate when the
        # tones allow a safe integer decimation, else work at full rate.
        max_tone = mark_hz
        min_internal = max(8.0 * baud, 2.6 * max_tone + 4.0 * baud)
        factor = 1
        if rate > 16000.0:
            candidate = int(rate // 12000.0)
            while candidate > 1 and (rate / candidate) < min_internal:
                candidate -= 1
            if (rate / candidate) >= min_internal:
                factor = candidate
        internal_rate = rate / factor
        if internal_rate < min_internal:
            return empty("Sample rate cannot resolve the RTTY tones and bit "
                         "timing for this carrier/shift/baud.", {
                             "carrier_hz": carrier, "shift_hz": shift, "baud": baud,
                         })
        audio = np.asarray(window.samples, dtype=np.float64)
        if factor > 1:
            audio = np.asarray(decimate(audio, factor, zero_phase=True), dtype=np.float64)
        n = audio.size
        t = np.arange(n, dtype=np.float64) / internal_rate

        def tone_envelope(freq: float) -> np.ndarray:
            mixed = audio * np.exp(-2j * np.pi * freq * t)
            # Keep the post-mixer envelope below the opposite-tone beat
            # frequency while retaining enough bandwidth for the bit rate.
            # A baud-only cutoff lets the other tone leak into the envelope
            # at high baud / narrow shift combinations.
            cutoff = min(2.0 * baud, 0.55 * shift, 0.3 * internal_rate)
            sos = butter(4, cutoff, btype="lowpass", fs=internal_rate, output="sos")
            return np.abs(sosfiltfilt(sos, mixed))

        mark_env = tone_envelope(mark_hz)
        space_env = tone_envelope(space_hz)
        mean_total = float(np.mean(mark_env + space_env))
        if not np.isfinite(mean_total) or mean_total < 1e-6:
            return empty("No measurable audio energy in this window.")
        soft = (mark_env - space_env) / np.maximum(mark_env + space_env,
                                                   0.1 * mean_total)
        decisive = float(np.mean(np.abs(soft) > 0.35))
        mark_dom = float(np.mean(soft > 0.35))
        space_dom = float(np.mean(soft < -0.35))
        contrast_evidence = {
            "carrier_hz": carrier, "shift_hz": shift, "baud": baud,
            "mark_hz": round(mark_hz, 2), "space_hz": round(space_hz, 2),
            "internal_rate_hz": round(internal_rate, 2),
            "decimation_factor": factor,
            "fsk_decisive_fraction": round(decisive, 3),
            "mark_dominant_fraction": round(mark_dom, 3),
            "space_dominant_fraction": round(space_dom, 3),
        }
        if not (decisive >= 0.30 and mark_dom >= 0.08 and space_dom >= 0.08):
            return empty("No alternating two-tone FSK contrast: silence, noise, "
                         "a single keyed tone, or an idle mark-only diddle.", contrast_evidence)

        bit_len = internal_rate / baud
        stop_order = [settings.stop_bits] + [s for s in SUPPORTED_STOP_BITS
                                              if s != settings.stop_bits]
        best: tuple[float, float, float, float, float] | None = None  # (frac, margin, stop, phase, pol)
        ranked: list[tuple[float, float, float, float, float]] = []
        for stop in stop_order:
            char_len = (6.0 + stop) * bit_len
            # The capture may begin in the middle of an idle mark before the
            # first start bit. Search the full character-period phase, at 1/8
            # bit resolution, instead of assuming the first frame starts at
            # sample zero modulo one bit.
            phases = np.arange(0.0, char_len, bit_len / 8.0)
            for pol in (1.0, -1.0):
                signed = soft if pol > 0 else -soft
                for phase in phases:
                    starts, data_abs, stops, _ = _frame_scores(signed, bit_len, float(phase), stop)
                    if starts.size == 0:
                        continue
                    valid = (starts < -0.15) & (stops > 0.15) & (data_abs > 0.12)
                    frac = float(np.mean(valid)) if valid.size else 0.0
                    margin = float(np.mean(
                        np.clip(-starts, 0, 1) + np.clip(stops, 0, 1)) / 2.0)
                    ranked.append((frac, margin, stop, float(phase), pol))
        if not ranked:
            return empty("Signal too short for an RTTY frame grid.", contrast_evidence)
        ranked.sort(key=lambda r: (r[0], r[1]), reverse=True)
        best = ranked[0]
        best_frac, best_margin, best_stop, best_phase, best_pol = best
        # Stop-length scores can be close for clean Baudot because the next
        # start bit may immediately follow a stop interval and the bit-center
        # samples tolerate small edge shifts. Only call a polarity ambiguity
        # when the same stop-length hypothesis supports both polarities; the
        # best full-window timing grid decides between distinct stop lengths.
        polarity_runner = next(
            (r for r in ranked[1:] if r[2] == best_stop and r[4] != best_pol),
            None,
        )
        tried_summary = [
            {"valid_stop_fraction": round(r[0], 3), "margin": round(r[1], 3),
             "stop_bits": r[2], "polarity": "normal" if r[4] > 0 else "inverted"}
            for r in ranked[:6]
        ]
        framing_evidence = dict(contrast_evidence)
        framing_evidence.update({
            "framing_candidates": tried_summary,
            "winner_stop_bits": best_stop,
            "winner_polarity": "normal" if best_pol > 0 else "inverted",
            "winner_valid_stop_fraction": round(best_frac, 3),
            "winner_margin": round(best_margin, 3),
        })
        if (polarity_runner is not None and polarity_runner[0] >= 0.70
                and (best_frac - polarity_runner[0]) < _AMBIGUITY_MARGIN):
            framing_evidence["ambiguity"] = (
                "normal/inverted polarity or stop-bit framings score within margin; "
                "abstaining instead of guessing.")
            return empty("RTTY framing is ambiguous between polarity/stop-bit "
                         "hypotheses.", framing_evidence)
        if best_frac < _VALID_FRAC_PARTIAL:
            return empty("No RTTY start/stop framing alignment found; not valid "
                         "Baudot frames.", framing_evidence)

        # Re-materialise the winning grid to recover the 5 data bits.
        signed = soft if best_pol > 0 else -soft
        char_bits = 6.0 + best_stop
        char_len = char_bits * bit_len
        n_chars = max(0, int(np.floor((n - best_phase) / char_len)))
        k = np.arange(n_chars, dtype=np.float64)
        half = max(1, int(round(bit_len / 4.0)))
        start_c = np.clip(np.round(best_phase + 0.5 * bit_len + k * char_len).astype(int), 0, n - 1)
        stop_c = np.clip(np.round(best_phase + (6.0 + best_stop / 2.0) * bit_len + k * char_len).astype(int), 0, n - 1)
        starts = _bit_means(signed, start_c, half)
        stops = _bit_means(signed, stop_c, half)
        bits = np.empty((n_chars, 5))
        for j in range(5):
            c = np.clip(np.round(best_phase + (1.5 + j) * bit_len + k * char_len).astype(int), 0, n - 1)
            bits[:, j] = _bit_means(signed, c, half)
        data_abs = np.mean(np.abs(bits), axis=1)
        valid = (starts < -0.15) & (stops > 0.15) & (data_abs > 0.12)
        valid_idx = np.flatnonzero(valid)
        scale = float(factor)  # decimated index -> full-rate samples (approx)
        figs = False
        bell_count = 0
        chars: list[str] = []
        tokens: list[DecodedToken] = []
        fits: list[float] = []
        for i in valid_idx:
            code = int(sum((1 << j) for j in range(5) if bits[i, j] > 0.0))
            bit_pattern = "".join("1" if bits[i, j] > 0.0 else "0" for j in range(5))
            fit = float((np.clip(-starts[i], 0, 1) + np.clip(stops[i], 0, 1)
                         + np.clip((data_abs[i] - 0.12) / 0.88, 0, 1)) / 3.0)
            if code == _LTRS:
                figs = False
                continue
            if code == _FIGS:
                figs = True
                continue
            if code == _BLANK:
                continue
            if code == _BEL and figs:
                bell_count += 1
                continue
            if code == _CR:
                glyph: str | None = "\r"
            elif code == _LF:
                glyph = "\n"
            else:
                table = _ITA2_FIGURES if figs else _ITA2_LETTERS
                glyph = table.get(code)
            if glyph is None:
                continue
            start_sample = window.sample_start + int(round((best_phase + i * char_len) * scale))
            end_sample = window.sample_start + int(round((best_phase + (i + 1) * char_len) * scale))
            chars.append(glyph)
            fits.append(fit)
            tokens.append(DecodedToken(
                glyph, max(window.sample_start, start_sample),
                max(window.sample_start + 1, end_sample),
                f"{'F' if figs else 'L'}:{bit_pattern}",
                round(float(np.clip(fit, 0.05, 0.99)), 3),
            ))
        text = "".join(chars)
        low_conf = sum(1 for f in fits if f < 0.5)
        evidence = dict(framing_evidence)
        evidence.update({
            "frame_count": int(n_chars),
            "valid_frame_count": int(valid_idx.size),
            "emitted_char_count": len(chars),
            "bell_count": int(bell_count),
            "low_confidence_char_fraction": round(low_conf / len(fits), 3) if fits else 1.0,
            "mean_char_fit": round(float(np.mean(fits)), 3) if fits else 0.0,
            "confidence_kind": "framing_and_tone_fit_heuristic_not_probability",
            "bit_and_frame_integrity": "ita2_has_no_fec_or_crc",
            "source_window_start_sample": window.sample_start,
            "source_window_sample_count": int(window.samples.size),
        })
        if len(chars) < _MIN_CHARS_EMIT:
            return LocalDecodeResult(
                self.spec.id, self.spec.version, "no_decode",
                tokens=tuple(tokens), evidence=evidence,
                note="Too few trustworthy ITA2 characters; candidate symbols retained.",
            )
        status = "decoded" if (best_frac >= _VALID_FRAC_DECODED
                               and len(chars) >= _MIN_CHARS_DECODED) else "partial"
        return LocalDecodeResult(
            self.spec.id, self.spec.version, status,
            text=text, tokens=tuple(tokens), evidence=evidence,
            note="Experimental single-carrier ITA2 interpretation; not independently verified.",
        )
