"""Incremental, receive-only Morse key-transition decoding.

This is independent of the whole-window CW decoder. The mixer, low-pass filter,
envelope smoother, key debouncer and unfinished word survive across live polls.
Only *new* PCM is processed, even if the source window overlaps earlier ones.
Outputs are strictly unverified evidence; no callsign or language guessing.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math

import numpy as np
from scipy.signal import butter, sosfilt, lfilter

from .libsigdecode.cw import MORSE

MAX_WORD_MARKS = 90
MAX_WORD_CHARS = 24
MAX_EMITTED_TEXT = 640


@dataclass
class CWKeyingStream:
    session_id: str
    track_id: str
    sample_rate: int
    carrier_hz: float
    last_sample: int = -1
    cumulative_text: str = ""
    emitted_words: int = 0
    _sos: np.ndarray = field(init=False, repr=False)
    _zi: np.ndarray = field(init=False, repr=False)
    _smooth_zi: np.ndarray = field(init=False, repr=False)
    _rem: np.ndarray = field(init=False, repr=False)
    _rem_start: int = -1
    _step: int = field(init=False)
    _env_hist: deque = field(default_factory=lambda: deque(maxlen=4000), repr=False)
    _floor: float = 0.0
    _peak: float = 0.0
    _pressed: bool = False
    _run_start: int = -1
    _proposed: bool | None = None
    _proposed_start: int = -1
    _proposed_count: int = 0
    _word_marks: list[tuple[int, int]] = field(default_factory=list, repr=False)
    _clean_prefix: bool = False
    _dot: float | None = None
    _last_word_end: int = -1
    _last_reason: str = "initializing"
    _pending_words: list[dict] = field(default_factory=list, repr=False)
    _trusted_stream: bool = False

    def __post_init__(self) -> None:
        if not 8000 <= self.sample_rate <= 192000:
            raise ValueError("unsupported CW sample rate")
        if not (math.isfinite(self.carrier_hz) and 20 <= self.carrier_hz < self.sample_rate/2):
            raise ValueError("invalid CW carrier")
        self._step = max(1, round(self.sample_rate / 500))
        self._sos = butter(4, 45.0, btype="lowpass", fs=self.sample_rate, output="sos")
        self._zi = np.zeros((len(self._sos), 2), dtype=np.complex128)
        self._smooth_zi = np.zeros(2, dtype=np.float64)
        self._rem = np.array([], dtype=np.complex128)

    def feed(self, pcm: np.ndarray, *, sample_start: int) -> dict:
        """Process an overlapping or contiguous mono PCM window.

        Caller serializes feeds per track. A gap, frequency retune or session
        switch must instantiate a fresh stream rather than join source samples.
        """
        values = np.asarray(pcm)
        if values.ndim != 1 or values.size == 0 or values.size > self.sample_rate * 8:
            raise ValueError("CW stream requires up to eight seconds of mono PCM")
        if sample_start < 0:
            raise ValueError("negative source sample offset")
        end = sample_start + len(values)
        if self.last_sample >= 0 and sample_start > self.last_sample:
            raise ValueError("CW source gap: reset required")
        if self.last_sample >= 0 and end < self.last_sample:
            raise ValueError("out-of-order CW source window")
        from_sample = max(sample_start, self.last_sample)
        if end <= from_sample:
            return self._event([], "overlap-already-processed", 0)
        chunk = values[from_sample-sample_start:]
        if chunk.dtype.kind in "iu":
            audio = chunk.astype(np.float64) / 32768.
        else:
            audio = chunk.astype(np.float64)
        if not np.isfinite(audio).all():
            raise ValueError("non-finite CW samples")
        samples = np.arange(from_sample, end, dtype=np.float64)
        downmixed = audio * np.exp((-2j*np.pi*self.carrier_hz/self.sample_rate)*samples)
        filtered, self._zi = sosfilt(self._sos, downmixed, zi=self._zi)
        if self._rem.size:
            filtered = np.concatenate((self._rem, filtered))
            base = self._rem_start
        else:
            base = from_sample
        count = len(filtered)//self._step
        remaining = count*self._step
        if count:
            env = np.abs(filtered[:remaining].reshape(count, self._step).mean(axis=1))
            env, self._smooth_zi = lfilter(
                np.ones(3)/3., [1.], env, zi=self._smooth_zi)
            self._feed_envelope(env, base)
        self._rem = filtered[remaining:]
        self._rem_start = base + remaining
        self.last_sample = end
        # _feed_envelope delivers confirmed words as a return value rather
        # than taking ownership of output storage in the audio callback.
        words = getattr(self, "_new_words", [])
        self._new_words = []
        return self._event(words, "new-unverified-words" if words else self._last_reason,
                           end-from_sample)

    def _feed_envelope(self, env: np.ndarray, base: int) -> None:
        hist = np.asarray(self._env_hist, dtype=float)
        context = np.concatenate((hist, env)) if hist.size else env
        lo, hi = np.percentile(context, [10, 90])
        lo, hi = float(lo), float(hi)
        if self._floor == 0:
            self._floor = lo
        else:
            # Track a lower background envelope without letting an entire
            # keyed interval redefine the background as a signal.
            self._floor = min(lo, self._floor*1.03)
        if hi > max(0.00015, self._floor * 5):
            self._peak = max(hi, self._peak * 0.98)
        elif self._peak:
            self._peak *= 0.98
        for x in env:
            self._env_hist.append(float(x))
        if not self._peak or self._peak < max(.00015, self._floor*4):
            self._last_reason = "insufficient-key-contrast"
            return
        threshold_on = self._floor + 0.43 * (self._peak - self._floor)
        threshold_off = self._floor + 0.30 * (self._peak - self._floor)
        debounce = max(3, round(.010 * self.sample_rate/self._step))
        if self._run_start < 0 and not self._pressed:
            self._run_start = base
        for i, level in enumerate(env):
            position = base + i*self._step
            desired = bool(level >= (threshold_off if self._pressed else threshold_on))
            if desired == self._pressed:
                self._proposed = None
                self._proposed_count = 0
            else:
                if desired != self._proposed:
                    self._proposed = desired
                    self._proposed_start = position
                    self._proposed_count = 1
                else:
                    self._proposed_count += 1
                if self._proposed_count >= debounce:
                    self._transition(desired, self._proposed_start)
                    self._proposed = None
                    self._proposed_count = 0
            if not self._pressed and self._run_start >= 0:
                gap = position - self._run_start
                if self._word_marks:
                    dot = self._estimate_dot()
                    if dot is not None and gap >= 5.5*dot:
                        self._flush_word()
                # Initial silence establishes that a later word did not
                # start before this stream's first received sample.
                elif gap >= .3*self.sample_rate:
                    self._clean_prefix = True
        self._last_reason = "awaiting-complete-word"

    def _transition(self, pressed: bool, sample: int) -> None:
        if pressed:
            if self._run_start < 0:
                # An initial mark has no verified silence prefix.
                self._clean_prefix = False
            self._run_start = sample
        else:
            if self._pressed and self._run_start >= 0:
                self._word_marks.append((self._run_start, sample))
                if len(self._word_marks) > MAX_WORD_MARKS:
                    self._word_marks = []
                    self._clean_prefix = False
                    self._last_reason = "oversized-word-abstained"
            self._run_start = sample
        self._pressed = pressed

    def _estimate_dot(self) -> float | None:
        if len(self._word_marks) < 4:
            return self._dot
        lengths = np.asarray([b-a for a,b in self._word_marks], dtype=float)
        shortest = float(np.percentile(lengths, 10))
        short = lengths[lengths <= shortest * 1.5]
        candidate = float(np.median(short)) if len(short) else shortest
        # 5--80 WPM; generous 15% tolerance for filtering delay.
        minimum = 1.2/80*self.sample_rate*.85
        maximum = 1.2/5*self.sample_rate*1.15
        if not minimum <= candidate <= maximum:
            return self._dot
        if self._dot is None:
            return candidate
        if .65*self._dot <= candidate <= 1.6*self._dot:
            return .75*self._dot + .25*candidate
        return self._dot

    def _flush_word(self) -> None:
        marks = self._word_marks
        self._word_marks = []
        dot = self._estimate_from_marks(marks)
        if dot is None or len(marks) < 5 or not self._clean_prefix:
            self._last_reason = "untrusted-partial-word"
            self._clean_prefix = True
            return
        text = []
        symbols = []
        fits = []
        for i, (start, stop) in enumerate(marks):
            width = (stop-start)/dot
            dash = width >= 1.85
            symbols.append("-" if dash else ".")
            fits.append(abs(width - (3 if dash else 1)))
            gap = (marks[i+1][0]-stop)/dot if i+1 < len(marks) else 999
            if gap >= 2:
                letter = MORSE.get("".join(symbols), "?")
                text.append(letter)
                symbols = []
                if gap >= 5 and i+1 < len(marks):
                    # Distinct words must never be joined silently.
                    text.append(" ")
        word = "".join(text)
        fit = float(np.mean(fits)) if fits else 100.
        if (2 <= len(word) <= MAX_WORD_CHARS
                and "?" not in word and " " not in word
                and len(text) >= 2 and fit <= .38
                and len(set(word)) >= 2):
            if not hasattr(self, "_new_words"):
                self._new_words = []
            candidate = {"text":word, "start_sample":marks[0][0],
                                    "end_sample":marks[-1][1],
                                    "mark_count":len(marks),
                                    "dot_ms":round(dot/self.sample_rate*1000, 2),
                                    "mean_mark_timing_error":round(fit, 3)}
            if self._trusted_stream:
                self._new_words.append(candidate)
            else:
                if (self._pending_words and candidate["start_sample"]
                        - self._pending_words[-1]["end_sample"] > 8*self.sample_rate):
                    self._pending_words.clear()
                self._pending_words.append(candidate)
                if (len(self._pending_words) >= 2 and
                        sum(len(item["text"]) for item in self._pending_words) >= 5):
                    self._new_words.extend(self._pending_words)
                    self._pending_words.clear()
                    self._trusted_stream = True
            self._dot = dot
        else:
            self._pending_words.clear()
            self._last_reason = "weak-or-ambiguous-morse-timing"
        self._clean_prefix = True

    def _estimate_from_marks(self, marks: list[tuple[int,int]]) -> float | None:
        if len(marks) < 4:
            return self._dot
        lengths = np.asarray([b-a for a,b in marks], dtype=float)
        shortest = float(np.percentile(lengths, 10))
        short = lengths[lengths <= shortest*1.5]
        candidate = float(np.median(short)) if len(short) else shortest
        if not (.85*1.2/80*self.sample_rate <= candidate
                <= 1.15*1.2/5*self.sample_rate):
            return self._dot
        if self._dot is not None and not .65*self._dot <= candidate <= 1.6*self._dot:
            return self._dot
        return candidate if self._dot is None else .75*self._dot + .25*candidate

    def _event(self, words: list[dict], reason: str, processed: int) -> dict:
        delta = " ".join(x["text"] for x in words)
        if delta:
            self.cumulative_text = (self.cumulative_text + " " + delta).strip()[-MAX_EMITTED_TEXT:]
            self.emitted_words += len(words)
            self._last_word_end = max(item["end_sample"] for item in words)
        return {
            "strategy": "continuous-key-transition-cw-v1",
            "session_id": self.session_id, "track_id": self.track_id,
            "delta_text": delta, "delta_words": words,
            "cumulative_text": self.cumulative_text,
            "last_emitted_end_sample": self._last_word_end,
            "last_window_end_sample": self.last_sample,
            "processed_new_samples": processed,
            "reason": reason, "confirmed_identity": False,
            "not_verified": True, "emitted_words": self.emitted_words,
        }
