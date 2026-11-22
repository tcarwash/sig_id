"""Independent experimental BPSK31 text decoder (mono audio, single carrier).

Protocol source: the ARRL PSK31 specification
(https://www.arrl.org/psk31-spec): 31.25 baud BPSK; a 180-degree phase
reversal from one symbol to the next carries bit 0 and no reversal carries
bit 1; characters are self-synchronizing varicode codewords separated by
"00" and transmitted left-bit-first. BPSK31 defines no FEC and no CRC.

Scope and acquisition bounds (documented limits, not silent fallbacks):

- The caller selects one audio carrier via ``DecodeWindow.carrier_hz``.
  There is no automatic multi-carrier search or sideband selection.
- Carrier accuracy must be within +/-3 Hz, checked with a squaring
  estimator on the baseband signal. Larger offsets abstain.
- Symbol timing is acquired per window on a bounded fractional grid
  (16 steps per symbol) and re-estimated per 128-symbol block with a
  continuity constraint of +/-2 steps, so fast clock drift abstains.
- Only the printable ASCII varicode subset (codes 32-126) is embedded
  below. Control characters (including CR/LF) are not mapped and decode
  as unmapped placeholders; this is a documented coverage gap.
- The printable varicode table is checked against the ARRL-published
  PSK31 table. Unmapped codewords fail closed (placeholder token, never
  guessed text).

No FEC/CRC is claimed: ``bit_and_frame_integrity`` evidence always
reports ``none_defined_by_protocol_no_fec_crc``.

No dependency on Fldigi, WSJT-X, OnAir, a receiver, or a transmitter.
Receive-only pure DSP on the provided mono window. Registry updates remain
the coordinator's responsibility.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .core import DecodeWindow, DecodedToken, DecoderSpec, LocalDecodeResult

SYMBOL_RATE_BAUD = 31.25
SYMBOL_PERIOD_S = 1.0 / SYMBOL_RATE_BAUD
TIMING_STEPS = 16  # fractional-symbol timing hypotheses per symbol
BLOCK_SYMBOLS = 128  # symbols per timing re-estimation block (~4.1 s)
MIN_SECONDS = 0.5  # minimum window for timing acquisition
MAX_CODEWORD_BITS = 16  # framing-loss bound; real codes below are <= 10 bits
MIN_BLOCK_COHERENCE = 0.997  # conservative phase-binary gate; noise/other modes abstain
MAX_CFO_HZ = 3.0  # documented carrier-accuracy bound
CFO_SEARCH_HZ = 40.0  # squaring-estimator search half-width
BASEBAND_CUTOFF_HZ = 75.0  # covers the ~31 Hz main lobe plus small offsets
MAX_EVIDENCE_CODEWORDS = 64  # cap on listed patterns; totals still reported

# Printable-ASCII varicode mapping, transmission order (first bit leftmost),
# checked against the ARRL PSK31 specification and its published Varicode
# table. Control characters remain outside this decoder's supported set.
_CHAR_TO_CODE: dict[str, str] = {
    " ": "1",
    "!": "111111111",
    '"': "101011111",
    "#": "111110101",
    "$": "111011011",
    "%": "1011010101",
    "&": "1010111011",
    "'": "101111111",
    "(": "11111011",
    ")": "11110111",
    "*": "101101111",
    "+": "111011111",
    ",": "1110101",
    "-": "110101",
    ".": "1010111",
    "/": "110101111",
    "0": "10110111",
    "1": "10111101",
    "2": "11101101",
    "3": "11111111",
    "4": "101110111",
    "5": "101011011",
    "6": "101101011",
    "7": "110101101",
    "8": "110101011",
    "9": "110110111",
    ":": "11110101",
    ";": "110111101",
    "<": "111101101",
    "=": "1010101",
    ">": "111010111",
    "?": "1010101111",
    "@": "1010111101",
    "A": "1111101",
    "B": "11101011",
    "C": "10101101",
    "D": "10110101",
    "E": "1110111",
    "F": "11011011",
    "G": "11111101",
    "H": "101010101",
    "I": "1111111",
    "J": "111111101",
    "K": "101111101",
    "L": "11010111",
    "M": "10111011",
    "N": "11011101",
    "O": "10101011",
    "P": "11010101",
    "Q": "111011101",
    "R": "10101111",
    "S": "1101111",
    "T": "1101101",
    "U": "101010111",
    "V": "110110101",
    "W": "101011101",
    "X": "101110101",
    "Y": "101111011",
    "Z": "1010101101",
    "[": "111110111",
    "\\": "111101111",
    "]": "111111011",
    "^": "1010111111",
    "_": "101101101",
    "`": "1011011111",
    "a": "1011",
    "b": "1011111",
    "c": "101111",
    "d": "101101",
    "e": "11",
    "f": "111101",
    "g": "1011011",
    "h": "101011",
    "i": "1101",
    "j": "111101011",
    "k": "10111111",
    "l": "11011",
    "m": "111011",
    "n": "1111",
    "o": "111",
    "p": "111111",
    "q": "110111111",
    "r": "10101",
    "s": "10111",
    "t": "101",
    "u": "110111",
    "v": "1111011",
    "w": "1101011",
    "x": "11011111",
    "y": "1011101",
    "z": "111010101",
    "{": "1010110111",
    "|": "1101111011",
    "}": "1010110101",
    "~": "1011010111",
}
_CODE_TO_CHAR: dict[str, str] = {code: char for char, code in _CHAR_TO_CODE.items()}
assert len(_CODE_TO_CHAR) == len(_CHAR_TO_CODE), "varicode table has duplicate codes"

#: Placeholder emitted for a well-framed but unmapped codeword. Never
#: guessed text; counted in evidence and forces at most partial status.
UNMAPPED_PLACEHOLDER = "?"


@dataclass(frozen=True, slots=True)
class BPSK31Settings:
    min_block_coherence: float = MIN_BLOCK_COHERENCE
    max_cfo_hz: float = MAX_CFO_HZ
    timing_steps: int = TIMING_STEPS

    def __post_init__(self) -> None:
        if not 0.64 < self.min_block_coherence < 1.0:
            raise ValueError("coherence threshold must be between 0.64 and 1.0")
        if not 0.5 <= self.max_cfo_hz <= 10.0:
            raise ValueError("carrier bound must be between 0.5 and 10 Hz")
        if not 4 <= self.timing_steps <= 64:
            raise ValueError("timing steps must be between 4 and 64")


def _block_average(values: np.ndarray, decim: int) -> np.ndarray:
    count = (values.size // decim) * decim
    return values[:count].reshape(-1, decim).mean(axis=1)


def _integrate_symbols(
    base: np.ndarray, sps: float, offset: float, start: int, count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate-and-dump symbols; returns amplitudes and edge indexes."""
    edges = np.round(offset + (start + np.arange(count + 1)) * sps).astype(int)
    edges = np.clip(edges, 0, base.size)
    cumulative = np.concatenate(([0j], np.cumsum(base)))
    widths = np.maximum(1, np.diff(edges))
    amplitudes = (cumulative[edges[1:]] - cumulative[edges[:-1]]) / widths
    return amplitudes, edges[:-1], edges[1:]


def _differential_coherence(symbols: np.ndarray) -> float:
    """Mean |Re(d)| / mean |d| over differential products; ~1 for clean BPSK."""
    if symbols.size < 8:
        return 0.0
    products = np.conj(symbols[:-1]) * symbols[1:]
    magnitudes = np.abs(products)
    denom = float(np.mean(magnitudes))
    if not np.isfinite(denom) or denom <= 0.0:
        return 0.0
    return float(np.mean(np.abs(products.real)) / denom)


def _residual_cfo_hz(base: np.ndarray, eff_rate: float) -> tuple[float | None, float]:
    """Estimate |carrier error| by squaring (BPSK -> line at twice the offset).

    Returns (estimate_or_None, peak_to_median_ratio). Sign is ambiguous
    under squaring, so only the magnitude is used as a bound check.
    """
    n = int(base.size)
    if n < 64:
        return None, 0.0
    windowed = (base * base) * np.hanning(n)
    spectrum = np.abs(np.fft.fft(windowed))
    freqs = np.fft.fftfreq(n, d=1.0 / eff_rate)
    band = np.abs(freqs) <= 2.0 * CFO_SEARCH_HZ
    band[np.argmin(np.abs(freqs))] = False  # ignore the DC term
    if not bool(band.any()):
        return None, 0.0
    peak = int(np.argmax(spectrum * band))
    median = float(np.median(spectrum[band]))
    ratio = float(spectrum[peak] / median) if median > 0 else 0.0
    if not np.isfinite(ratio) or ratio < 6.0:
        return None, round(ratio, 2)
    return round(abs(float(freqs[peak])) / 2.0, 2), round(ratio, 2)


def _split_codewords(
    bits: list[bool],
) -> tuple[list[tuple[str, int, int]], int, int]:
    """Split a differential bit stream on "00" separators.

    Returns (codewords, overlong_runs_skipped, trailing_bits). Each
    codeword is (pattern_in_received_bit_order, first_bit_index,
    one_past_last_bit_index). An overlong run without "00" is skipped
    (framing resync) and counted instead of poisoning later characters.
    """
    codewords: list[tuple[str, int, int]] = []
    overlong = 0
    pending = ""
    pending_start = 0
    for index, bit in enumerate(bits):
        if not pending:
            pending_start = index
        pending += "1" if bit else "0"
        if len(pending) > MAX_CODEWORD_BITS:
            overlong += 1
            pending = ""
            continue
        if pending.endswith("00"):
            pattern = pending[:-2]
            if pattern:
                codewords.append((pattern, pending_start, index + 1))
            pending = ""
    return codewords, overlong, len(pending)


class BPSK31Decoder:
    spec = DecoderSpec(
        id="bpsk31", name="BPSK31", family="keyboard",
        stage="experimental", version="bpsk31-text-v1",
    )

    def __init__(self, settings: BPSK31Settings | None = None) -> None:
        self.settings = settings or BPSK31Settings()

    def decode(self, window: DecodeWindow) -> LocalDecodeResult:
        def empty(reason: str, evidence: dict | None = None) -> LocalDecodeResult:
            return LocalDecodeResult(
                self.spec.id, self.spec.version, "no_decode",
                evidence=evidence or {}, note=reason,
            )

        if window.carrier_hz is None:
            return empty(
                "Choose the audio carrier of a single BPSK31 signal; "
                "automatic multi-carrier selection is not implemented."
            )
        if window.samples.size < int(window.sample_rate * MIN_SECONDS):
            return empty("At least 0.5 seconds of audio is needed for symbol timing.")
        rate = window.sample_rate
        carrier = float(window.carrier_hz)

        # Downmix at full rate, then block-average decimate to a bounded
        # effective rate (<= ~2000 Hz, i.e. 64 samples per symbol at 8 kHz).
        # The averaging stage is only a sinc prefilter: the selected window
        # must already be dominated by the target carrier, as with CW.
        decim = max(1, rate // 2000)
        eff_rate = rate / decim
        sps = eff_rate / SYMBOL_RATE_BAUD
        n = np.arange(window.samples.size, dtype=np.float64)
        mixed = window.samples * np.exp(-2j * np.pi * carrier * n / rate)
        base = _block_average(mixed, decim)
        try:
            sos = butter(4, BASEBAND_CUTOFF_HZ, btype="lowpass",
                         fs=eff_rate, output="sos")
            base = sosfiltfilt(sos, base)
        except ValueError:
            return empty("Baseband filter failed on this window length.")

        # Presence check: sustained BPSK keeps symbol magnitudes near the
        # carrier level; silence/noise collapses them.
        rms_in = float(np.sqrt(np.mean(window.samples ** 2)))
        median_mag = float(np.median(np.abs(base))) if base.size else 0.0
        if not np.isfinite(median_mag) or median_mag < 0.15 * rms_in:
            return empty(
                "No sustained carrier-level energy at the selected carrier.",
                {"carrier_hz": carrier,
                 "median_baseband_magnitude": round(median_mag, 6),
                 "input_rms": round(rms_in, 6)},
            )

        # Carrier-accuracy bound check via the squaring estimator.
        cfo, cfo_ratio = _residual_cfo_hz(base, eff_rate)
        if cfo is not None and cfo > self.settings.max_cfo_hz:
            return empty(
                "Estimated carrier error exceeds the +/-3 Hz acquisition bound; "
                "re-select the carrier within a few Hz.",
                {"carrier_hz": carrier,
                 "residual_cfo_hz": cfo,
                 "cfo_peak_to_median": cfo_ratio,
                 "cfo_bound_hz": self.settings.max_cfo_hz},
            )

        # Timing acquisition: full-grid search on the first block, then
        # continuity-constrained re-estimation per block for clock drift.
        total_symbols = int(base.size // sps)
        if total_symbols < 16:
            return empty("Too few symbol intervals for timing acquisition.")
        steps = self.settings.timing_steps
        first = min(total_symbols, 2 * BLOCK_SYMBOLS)
        scores = [
            _differential_coherence(
                _integrate_symbols(base, sps, j * sps / steps, 0, first)[0])
            for j in range(steps)
        ]
        step_index = int(np.argmax(scores))
        symbols = np.empty(0, dtype=np.complex128)
        sym_lo: list[int] = []
        sym_hi: list[int] = []
        block_scores: list[float] = []
        start = 0
        failed_block: int | None = None
        while start < total_symbols:
            count = min(BLOCK_SYMBOLS, total_symbols - start)
            if start == 0:
                candidates = range(steps)
            elif count < 32:
                # A short trailing block cannot estimate timing reliably;
                # bounded drift keeps the previous grid valid.
                candidates = [step_index]
            else:
                candidates = [(step_index + d) % steps for d in (-2, -1, 0, 1, 2)]
            best_idx, best_score, best = step_index, -1.0, None
            for candidate in candidates:
                trial = _integrate_symbols(
                    base, sps, candidate * sps / steps, start, count)
                score = _differential_coherence(trial[0])
                if score > best_score:
                    best_idx, best_score, best = candidate, score, trial
            assert best is not None
            block_scores.append(round(float(best_score), 4))
            if best_score < self.settings.min_block_coherence:
                failed_block = start // BLOCK_SYMBOLS
                break
            step_index = best_idx
            symbols = np.concatenate((symbols, best[0]))
            sym_lo.extend(int(v) for v in best[1])
            sym_hi.extend(int(v) for v in best[2])
            start += count

        evidence: dict = {
            "carrier_hz": carrier,
            "symbol_rate_baud": SYMBOL_RATE_BAUD,
            "timing_steps_per_symbol": steps,
            "timing_block_symbols": BLOCK_SYMBOLS,
            "block_coherence": block_scores,
            "coherence_threshold": self.settings.min_block_coherence,
            "residual_cfo_hz": cfo,
            "cfo_peak_to_median": cfo_ratio,
            "cfo_bound_hz": self.settings.max_cfo_hz,
            "confidence_kind": "differential_coherence_heuristic_not_probability",
            "bit_and_frame_integrity": "none_defined_by_protocol_no_fec_crc",
            "varicode_table": "arrl_psk31_printable_ascii_v1",
            "varicode_codes_mapped": len(_CODE_TO_CHAR),
            "source_window_start_sample": window.sample_start,
            "source_window_sample_count": int(window.samples.size),
        }
        if failed_block is not None:
            evidence["failed_block"] = failed_block
            return LocalDecodeResult(
                self.spec.id, self.spec.version, "no_decode",
                evidence=evidence,
                note="Symbol timing coherence below threshold; not BPSK31 at "
                     "this carrier, or carrier/timing outside acquisition bounds.",
            )

        # Differential decisions: phase reversal -> 0, no reversal -> 1.
        # Absolute-phase ambiguity cancels in the product, so any constant
        # phase rotation (including polarity inversion) decodes identically.
        products = np.conj(symbols[:-1]) * symbols[1:]
        magnitudes = np.abs(products)
        bits = [bool(v.real > 0) for v in products]
        confidences = []
        for value, magnitude in zip(products, magnitudes):
            if magnitude > 0:
                hint = abs(value.real) / magnitude
            else:
                hint = 0.05
            confidences.append(round(min(0.99, max(0.05, float(hint))), 3))
        bounds = list(zip(sym_lo, sym_hi))
        # Bit k spans symbols k..k+1, i.e. eff samples [lo[k], hi[k+1]).
        bit_bounds = [(bounds[k][0], bounds[k + 1][1]) for k in range(len(bits))]
        codewords, overlong, trailing = _split_codewords(bits)
        evidence.update({
            "symbol_count": int(symbols.size),
            "bit_count": len(bits),
            "codeword_count": len(codewords),
            "overlong_runs_skipped": overlong,
            "incomplete_trailing_bits": trailing,
            "codeword_patterns": [pattern for pattern, _, _
                                  in codewords[:MAX_EVIDENCE_CODEWORDS]],
        })
        if not codewords:
            return LocalDecodeResult(
                self.spec.id, self.spec.version, "no_decode",
                evidence=evidence,
                note="No varicode character boundaries recovered (unmodulated "
                     "carrier, overlong framing without a 00 separator, or "
                     "the wrong mode).",
            )

        if overlong and len(codewords) < 2:
            evidence["unmapped_codeword_count"] = len(codewords)
            return LocalDecodeResult(
                self.spec.id, self.spec.version, "no_decode",
                evidence=evidence,
                note="No stable run of varicode characters recovered; a lone "
                     "boundary after overlong framing is insufficient to decode.",
            )

        # Fail-closed character mapping: unmapped codewords become "?"
        # placeholders and cap the status at partial, never guessed text.
        texts: list[str] = []
        unmapped = 0
        for pattern, _, _ in codewords:
            char = _CODE_TO_CHAR.get(pattern)
            if char is None:
                unmapped += 1
                texts.append(UNMAPPED_PLACEHOLDER)
            else:
                texts.append(char)
        evidence["unmapped_codeword_count"] = unmapped
        text = "".join(texts)
        tokens = tuple(
            DecodedToken(
                char,
                window.sample_start + int(bit_bounds[i0][0] * decim),
                min(window.sample_start + window.samples.size,
                    window.sample_start + int(bit_bounds[i1 - 1][1] * decim)),
                pattern,
                round(float(np.mean(confidences[i0:i1])), 3))
            for (pattern, i0, i1), char in zip(codewords, texts)
        )
        if unmapped == len(codewords):
            return LocalDecodeResult(
                self.spec.id, self.spec.version, "no_decode",
                tokens=tokens, evidence=evidence,
                note="Well-framed codewords, but none map to printable ASCII; "
                     "wrong mode or unsupported control characters.",
            )
        if unmapped or overlong:
            return LocalDecodeResult(
                self.spec.id, self.spec.version, "partial",
                text=text, tokens=tokens, evidence=evidence,
                note="BPSK31 text with gaps: unmapped codewords shown as ? "
                     "and/or overlong framing skipped.",
            )
        return LocalDecodeResult(
            self.spec.id, self.spec.version, "decoded",
            text=text, tokens=tokens, evidence=evidence,
            note="Experimental single-carrier BPSK31 text interpretation; "
                 "on-air accuracy is not independently verified.",
        )
