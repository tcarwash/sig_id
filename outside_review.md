# Outside Technical Review: Live Signal Detection, Classification, and Decoding

This review evaluates the live signal-processing, machine-learning, and decoding pipelines in `sig_id` from an outside RF/DSP and ML-systems perspective. It focuses on concrete, high-leverage improvements that fit the project's existing design principles (receive-only safety, explicit provenance, separation of synthetic and real evidence, and human-in-the-loop proposals).

---

## Executive Summary & Highest-Leverage Priorities

The workbench has unusually strong engineering discipline around lineage tracking, dataset hygiene, append-only review history, and honest uncertainty reporting. However, several structural bottlenecks currently limit real-world live performance:

1. **Classifier features are not frequency-translation invariant** (`sigid/classifier.py`): spectral features are binned on an absolute 200–3800 Hz grid and fed to a linear softmax model. Moving the same signal from 1000 Hz to 2200 Hz completely changes its feature vector.
2. **Hierarchical classifier calibration can self-brick into reject-all and lacks noise/OOD gating** (`sigid/hierarchical_training.py`, `sigid/classifier.py`): `"noise"` is excluded from family training, mode experts are calibrated against out-of-family samples with a strict `0.0%` FPR constraint, and `propose_loaded` hardcodes `noise_detected: False` with no novelty check.
3. **Live detection has a domain gap and a latency bottleneck** (`sigid/detection.py`, `sigid/cv_detector.py`, `sigid/spectrum.py`): the heuristic detector only sees narrowband local maxima above a global median floor; the YOLO detector uses fixed dB color bounds (`-95` to `-20` dB), has an RGB/BGR bug in `embed_image`, inflates partial-window noise padding when signals are present, and runs 1,664 Python-loop `np.interp` calls per spectrogram.
4. **Live decoding is bottlenecked by cold-booting real-time Fldigi processes and discarding classifier/track context** (`sigid/fldigi_replay.py`, `sigid/trials.py`, `sigid/api.py`): auto-trials only test 3 hardcoded modes (`BPSK31`, `RTTY`, `THOR16`) on a 2-second classifier crop instead of the full track, stop on any single noise byte, and cold-start `xvfb-run` + `fldigi` + real-time `paplay` for every mode attempt.
5. **The 10-second live ring buffer (`LIVE_ANALYSIS_SECONDS = 10`) is too short** (`sigid/live_audio.py`) for 15-second weak-signal modes like FT8 (which `wsjtx_backend.py` requires ≥12.6 s for) and causes live selection expirations during operator inspection.

---

## 1. Live Detection & Tracking

### 1.1 Heuristic Spectral Detector (`sigid/detection.py`)

**Current limitations:**
- **Global row-median noise floor (`detection.py:78`)**: `floor = np.median(row[first_bin:])` assumes a flat noise floor across the entire passband. Real HF/SSB receivers have sloped audio passbands (300–2700 Hz filter roll-off, colored atmospheric noise, hum), causing missed signals near band edges and false alarms near passband peaks.
- **Hardcoded absolute peak floor (`minimum_peak_db = -25.0`, `detection.py:40, 79`)**: Quiet receiver audio chains (common when leaving headroom to avoid ADC clipping) have legitimate signals at `-40` to `-70` dBFS that are silently discarded regardless of SNR.
- **Peak-spacing suppression breaks narrowband MFSK (`detection.py:90`)**: `abs(index - selected) >= 4` bins. At 48 kHz with `fft_size = 4096` (`bin_hz = 11.72 Hz`), 4 bins is **46.9 Hz**. Closely spaced tone grids—FT8 (6.25 Hz), WSPR (1.46 Hz), MFSK16 (15.625 Hz), Olivia 8/250 (31.25 Hz)—are physically suppressed into a single peak.
- **Blind to wideband / flat-top / spread signals**: Requiring strict 3-bin local maxima (`row[1:-1] >= row[:-2] & row[1:-1] > row[2:]`) misses or fragments OFDM (MT63), wideband PSK/QPSK, SSB voice, and chirps.
- **Pure frequency-gap grouping merges unrelated signals (`detection.py:155-160`)**: Any two tones within `maximum_tone_gap_hz = 800.0` are merged into one candidate regardless of whether they overlap in time. Two independent PSK31 or RTTY QSOs 400 Hz apart become a single `"MFSK-like"` track.

**Recommendations:**
1. **Replace global median with 1D local CFAR / rolling percentile baseline** (e.g., OS-CFAR or rolling 25th–50th percentile over a ±150–250 Hz window with a guard band) to flatten receiver passband tilt. Remove or drastically lower `minimum_peak_db` (e.g., `-90 dBFS`) and gate primarily on local SNR.
2. **Add a 2D energy-blob detector alongside peak clustering**: threshold the locally normalized SNR spectrogram (e.g., `+6` to `+10` dB), apply light morphological closing in time-frequency, and extract connected components (`scipy.ndimage.label`). This cleanly bounds wideband signals (MT63, Olivia, SSB, FM, wideband FSK) with accurate `[f_low, f_high]` and `[t_start, t_end]` extents.
3. **Make peak suppression frequency-aware, not bin-count-hardcoded**: reduce minimum peak spacing to `max(2 bins, 1.5 * bin_hz)` (or apply parabolic sub-bin peak interpolation) so narrow MFSK tone grids remain resolvable.
4. **Group tones by temporal correlation, not just frequency proximity**: only cluster tones into one FSK/MFSK candidate if their active time intervals overlap significantly and (for FSK/MFSK) show alternating or synchronized occupancy rather than independent start/stop times.

### 1.2 CV Spectrogram Detector (`sigid/cv_detector.py`, `sigid/cv_training.py`)

**Current limitations:**
- **Fixed dB colormap scaling (`DB_FLOOR = -95.0`, `DB_CEILING = -20.0`, `cv_detector.py:32-33, 64`)**: Real captures vary by 30+ dB in soundcard gain and band noise floor. Fixed scaling crushes weak signals into solid dark blue or saturates loud bands into yellow. This is a primary driver of the real-capture domain gap documented in `reports/cv-real-capture-exploration-2026-10-23.json`.
- **Partial-window noise padding bug (`detect_live_partial`, `cv_detector.py:247-254`)**: `noise_rms` is computed as the raw RMS of the last 0.5 s of audio. When a strong signal is active in that tail, `noise_rms` reflects the *signal* power, filling the unobserved past with bright broadband white noise that creates a false horizontal boundary and degrades detection.
- **RGB/BGR channel mismatch in `embed_image()` (`cv_detector.py:213-216`)**: `detect()` correctly flips RGB→BGR (`image[..., ::-1]`) because Ultralytics expects BGR for NumPy inputs, but `embed_image()` passes RGB directly to `self._model.embed(image, ...)`. Because the backbone was trained on RGB-converted images, `embed()` runs on inverted color channels, degrading similarity retrieval quality.
- **Unintended Ultralytics training augmentations (`cv_training.py:385-407`)**: While `mosaic`, `mixup`, `fliplr`, `flipud`, `degrees`, and `scale` are disabled, Ultralytics defaults for `hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4`, `translate=0.1`, and `erasing=0.4` remain active. HSV jitter corrupts the dB-to-color mapping, and `translate`/`erasing` inject gray `(114, 114, 114)` borders/patches that resemble wideband high-SNR bursts against the `[8, 17, 35]` noise background.
- **Severe CPU overhead in `spectrogram_image()` and `waterfall_samples()` (`cv_detector.py:51-63`, `spectrum.py:41-46`)**: `waterfall_samples()` runs a Python `for` loop per FFT frame, converts every row to a Python `list`, and `spectrogram_image()` converts it back to NumPy and runs 640 + 1,024 = 1,664 Python-loop `np.interp` calls per window.
- **Hardcoded 3000 Hz ceiling (`MAX_FREQUENCY_HZ = 3000.0` in `cv_detector.py:28` and `live_spectrum.py:34`)**: Cuts off 3–4 kHz passband audio (wideband SSB, STANAG, high-tone MFSK, VHF/FM audio).

**Recommendations:**
1. **Adaptive noise-floor normalization (new `PREPROCESSING_VERSION = "sig-id-live-waterfall-v3"`)**: Anchor the spectrogram floor to a robust per-window or per-column noise estimate (e.g., 15th–20th percentile of dB per frequency column, smoothed across frequency) with a fixed dynamic range (e.g., `[-5 dB, +55 dB]` relative to local floor). This makes the detector invariant to receiver gain, AGC level, and passband slope.
2. **Fix `detect_live_partial` padding**: Estimate `noise_rms` from the spectral median (or `1.4826 * MAD` of time-domain samples) instead of raw tail RMS—or pad directly in the spectrogram domain with the estimated noise floor rather than synthesizing white noise.
3. **Fix `embed_image()` color order**: Pass `np.ascontiguousarray(image[..., ::-1])` to `self._model.embed(...)` (and bump `EMBEDDING_PREPROCESSING_VERSION` so existing indexes are rebuilt cleanly).
4. **Disable color/erasing augmentations in `train_cv_detector`**: Explicitly set `hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, erasing=0.0, translate=0.0` in `train_options`, and perform physically valid augmentations in the **audio/spectrogram domain** (random frequency shifts, time shifts, gain/noise-floor offsets, fading, QRM).
5. **Vectorize STFT and image resizing**: Replace the Python FFT loop in `waterfall_samples()` / `live_spectrum.py` with `np.lib.stride_tricks.sliding_window_view` + batched `np.fft.rfft(frames * window, axis=1)`. Replace the 1,664 `np.interp` loops in `spectrogram_image()` with vectorized 2D bilinear interpolation (`scipy.ndimage.zoom` or `PIL.Image.fromarray(...).resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.BILINEAR)`). Expect a **10–30× speedup** on CPU.

### 1.3 Live Tracking & Ring Buffer (`sigid/tracking.py`, `sigid/live_audio.py`, `sigid/api.py`)

**Current limitations:**
- **CV detections are untracked**: `/api/live/cv-detections` returns stateless per-poll boxes with no track IDs, persistence filtering, or smoothing across polls, while `LiveTrackRegistry` is only wired to `detect_tracks()` in `/api/live/analysis`.
- **Greedy frequency-only association (`tracking.py:71-82`)**: Matches left-to-right using `abs(center - track.center_hz) <= max(150.0, 1.5 * max(bw))`. On crowded HF bands (signals 50–100 Hz apart), greedy matching causes track ID swaps and loses drifting signals.
- **10-second live ring buffer (`LIVE_ANALYSIS_SECONDS = 10`, `live_audio.py:22`)**:
  - Too short for 15-second FT8/JT9/JT65 cycles (`wsjtx_backend.py:140` enforces ≥12.6 s), making live FT8 decoding from the ring impossible.
  - Causes frequent `"selected samples are no longer in the live buffer"` errors (`api.py:1164, 1906, 2634`) when an operator clicks a signal and takes a few seconds to inspect or decode it.
- **Non-overlapping unattended 10 s windows (`live_audio.py:340-350`)**: Signals straddling a 10-second boundary are split across two clips.

**Recommendations:**
1. **Unify CV and spectral detections into `LiveTrackRegistry`**: Run a single live tracker that accepts either CV boxes or spectral candidates (or fuses them by frequency/time IoU), assigns stable `track_id`s, smooths `[f_low, f_high]` via EMA, and promotes tracks to `persistent` only after `k`-of-`n` polls.
2. **Use Hungarian assignment (`scipy.optimize.linear_sum_assignment`) with drift prediction**: Predict each track's center forward using `track.center_hz + track.drift_hz_per_s * dt`, and score associations using a joint cost of frequency IoU + center distance + bandwidth ratio.
3. **Expand the live ring buffer to 30–60 seconds**: At 48 kHz mono int16, 60 seconds is only ~5.7 MB of RAM. Keep the CV inference window at 10 s (slicing the latest 10 s), while allowing live replay, FT8 decoding, and operator annotations to reference up to 60 s of history (matching the UI's 60 s waterfall cache).
4. **Use 50% overlapping windows (5 s hop) in unattended mode** (matching `WINDOW_STRIDE_SECONDS = 5` in `cv_detector.py`), merging consecutive triggered windows into one clip so bursts are never sliced in half.

---

## 2. Signal Classification

### 2.1 Feature Representation (`sigid/classifier.py`)

**Current limitations:**
- **Not frequency-translation invariant (`classifier.py:101-129`)**: `extract_features()` bins power into 128 fixed absolute frequencies from 200 to 3800 Hz (`mean_log`, `variation_log`). Even with ROI bandpass filtering (`v3`/`v4`), the signal remains at its original audio frequency! A linear classifier trained on RTTY at 1500 Hz cannot recognize RTTY at 900 Hz or 2300 Hz because entirely different weight indices are activated.
- **Discards instantaneous frequency, phase, and cyclostationary structure**: Averaging STFT frames into `mean` and `std` discards baud rate, phase constellation geometry, and FSK shift structure. The Hilbert envelope (`classifier.py:133-152`) averages `|hilbert(x)|` over 1 ms blocks, which helps separate OOK/ASK from constant-envelope modes, but provides almost no signal for separating BPSK vs QPSK vs 8PSK vs 2-FSK vs 4/8/16-FSK vs OFDM.
- **Zero-padding corrupts short-signal features (`fixed_duration_crop`, `classifier.py:54-57`)**: Signals shorter than 2.0 s are padded with exact zeros, which artificially drives the 10th-percentile envelope to `0.0`, inflates envelope `std` and spectral `variation_log`, and skews `mean(normalized > 0.25)`.

**Recommendations:**
1. **Downconvert every ROI to complex baseband (Digital DDC) before feature extraction**:
   - Given `[f_low, f_high]`, compute `f_center = (f_low + f_high) / 2` and `bw = f_high - f_low`.
   - Mix `z(t) = x(t) * exp(-2j * pi * f_center * t)`, low-pass filter to `bw / 2 + margin`, and optionally resample to a standardized baseband rate (or compute relative-frequency features).
   - This immediately makes the classifier **100% invariant to carrier frequency** across the entire passband!
2. **Add RF modulation-discriminative baseband features**:
   - **Centered relative-frequency PSD**: log-PSD normalized around `0 Hz` (plus occupied bandwidth at 50%/90%/99% energy, spectral flatness, crest factor, symmetry).
   - **Instantaneous frequency (`f_inst = diff(unwrap(angle(z)))`)**: histogram/quantiles, kurtosis, number of distinct modes, and inter-peak spacing (directly distinguishes 2-FSK vs 4/8/16-FSK vs PSK/OFDM).
   - **Instantaneous phase & higher-order cumulants (`C20, C21, C40, C42, C63`)**: standard RF automatic modulation classification (AMC) invariants that cleanly separate BPSK, QPSK, 8PSK, QAM, FSK, and Gaussian noise regardless of carrier phase offset.
   - **Cyclostationary / baud-rate features**: peak prominence and frequency of `FFT(|z(t)|)` (envelope rate) and `FFT(z(t)^2)` (BPSK carrier-doubling / symbol-rate lines).
3. **Mask padded samples instead of zero-padding**: For signals shorter than `classifier_window_seconds`, compute STFT and envelope statistics strictly over the valid (unpadded) sample interval rather than including trailing/leading zeros.

### 2.2 Model Architecture & Calibration (`sigid/classifier.py`, `sigid/hierarchical_training.py`, `sigid/training.py`)

**Current limitations:**
- **Linear softmax capacity ceiling (`fit_softmax`, `classifier.py:163-218`)**: A single linear layer cannot model non-linear decision boundaries (e.g., "constant envelope AND 2 frequency peaks AND 170 Hz spacing"). It also lacks class weighting (imbalanced real/Fldigi corpora bias predictions toward frequent classes) and uses fixed-step full-batch gradient descent.
- **Broken hierarchical calibration & missing noise/OOD handling (`hierarchical_training.py:283, 311-342`, `classifier.py:409-447`)**:
  - `signal_mask = train_labels != "noise"` excludes noise from training the Stage-1 family classifier, and `propose_loaded` hardcodes `"noise_detected": False` with **no novelty distance check**.
  - Yet `_calibrate_threshold()` enforces `MAX_VALIDATION_NOISE_FALSE_POSITIVE_RATE = 0.0` and `MAX_VALIDATION_UNKNOWN_FALSE_POSITIVE_RATE = 0.0` (`training.py:31-32`). A single noisy or OOD validation sample with high softmax confidence forces `threshold = 1.000001` (`reject_all_no_threshold_meets_validation_constraints`).
  - Worse, each family's Stage-2 mode expert is calibrated against *all* validation samples—including other families and noise—even though the expert was trained exclusively on modes within its own family. Softmax probabilities sum to 1, so out-of-family samples routinely score high, again forcing reject-all thresholds.
- **No temporal evidence accumulation in live mode (`api.py:1612-1652, 1715-1833`)**: Live CV classifies a single 2 s crop per box per poll with no cross-poll memory, causing labels to flicker.

**Recommendations:**
1. **Fix hierarchical training and calibration**:
   - Include `noise` as an explicit class in Stage 1 (or train a dedicated signal-vs-noise gate + prototype/Mahalanobis OOD distance in standardized feature space).
   - Calibrate Stage-2 mode experts **conditionally on in-family validation samples** (since Stage 1 already gates the family), and apply **temperature scaling** on validation logits so probabilities are well-calibrated.
   - Replace the brittle `0.0%` exact FPR rule with a statistically sound target (e.g., ≤1–2% validation FPR or F0.5-maximizing threshold with a minimum recall floor) so one outlier cannot disable the model.
2. **Upgrade the classifier family**: Keep the portable NumPy inference path, but train a **2-layer MLP** (with ReLU/GELU, dropout, AdamW, and inverse-frequency class weights) or a **lightweight 2D CNN on baseband ROI spectrogram patches** (since PyTorch is already installed for YOLO). A tiny 2-layer MLP exports cleanly to `.npz` with zero extra runtime dependencies!
3. **Accumulate classification probabilities per live track**: Maintain a rolling log-likelihood sum (or EMA of calibrated probability vectors) per `track_id` in `LiveTrackRegistry` across consecutive 2 s crops. Report both instantaneous and track-accumulated proposals.
4. **Expand synthetic channel augmentations (`sigid/augmentation.py`)**: Currently only AWGN + linear gain exist. Add deterministic, seed-reproducible **carrier frequency offset/drift**, **Watterson/Rayleigh HF multipath fading**, **impulsive static (QRN)**, **adjacent-channel QRM**, and **SSB bandpass filter tilt/ripple** to close the synthetic-to-real gap.

---

## 3. Live Decoding & Raw Demodulation

### 3.1 Isolated Fldigi Replay & Auto-Trials (`sigid/fldigi_replay.py`, `sigid/trials.py`, `sigid/api.py`)

**Current limitations:**
- **Extreme latency per decode attempt (`fldigi_replay.py:153-257`)**: Every replay creates a new PulseAudio null sink, copies `~/.fldigi`, launches `xvfb-run` + `fldigi`, polls XML-RPC for startup, plays audio in **1× real time** via `paplay`, waits 2 s for output flush, and tears everything down—all behind a single-worker lock (`decode_executor = ThreadPoolExecutor(max_workers=1)`). Testing 3 candidate modes on a 10 s clip takes **45–60 seconds**.
- **Auto-trials ignore specific classifier predictions and crop to 2 seconds (`trials.py:13-18`, `api.py:1846-1861`)**:
  - `MODULATION_TRIAL_ORDER` hardcodes only `("BPSK31", "RTTY", "THOR16")`. Even if the classifier predicts `bpsk63`, `qpsk31`, `mfsk16`, `olivia-8-250`, or `mt63-500s`, `api.py` discards the mode and falls back to those 3 modes.
  - `api.py:1857-1858` passes `proposal.get("sample_start")` and `proposal.get("sample_count")` (the 2-second classifier crop!) to `queue_live_mode_sweep` instead of the full `track["sample_start"]` and `track["sample_count"]`. Two seconds is too short for most HF modems (especially THOR, MFSK, Olivia, MT63) to acquire sync and emit readable text.
- **Early-stop on noise garbage (`trials.py:115-122`)**: `DecoderTrialRunner.advance()` halts the sweep as soon as `end > offset` (even 1 byte of random noise from open squelch), without checking text coherence.
- **No ROI bandpass pre-filtering before Fldigi replay (`api.py:995-1000`)**: The full passband is played to Fldigi with only `carrier_hz` set; a stronger adjacent signal 150 Hz away can capture Fldigi's AFC/squelch.
- **English-prose bias in `_decode_text_coherence()` (`api.py:714-739`)**: Relies heavily on 30 English trigrams (`0.40` weight) and penalizes outputs shorter than 48 characters (`len(visible) / 48.0`). Real amateur digital exchanges (`CQ CQ DE W1AW K`, `599 599 TU`, `73`, grid squares, uppercase/numeric telemetry) score artificially low.

**Recommendations:**
1. **Fix the 2-second crop bug in `live_analysis` auto-trials (`api.py:1857-1858`)**: Pass the full track time span (`track["sample_start"]`, `track["sample_count"]`)—and up to 10–15 s from the expanded live ring—to `queue_live_mode_sweep`.
2. **Use classifier mode/family rankings to build the trial shortlist**: Map the classifier's top-k mode proposals (plus bandwidth-compatible Fldigi modes from `MODE_REGISTRY` whose `source_bandwidth` matches the detected track bandwidth!) into the candidate list instead of hardcoding 3 modes.
3. **Maintain a warm, persistent isolated Fldigi worker** (instead of cold-booting Xvfb+Fldigi per mode): Keep 1–2 sandboxed receive-only Fldigi instances running on dedicated null sinks, switching modes and clearing the RX buffer via XML-RPC. This eliminates startup/teardown overhead and lets multi-mode sweeps run back-to-back.
4. **Add a native lightweight RSID (Reed-Solomon ID) detector in Python/NumPy**: RSID is a standardized ~1.4 s MFSK-16 preamble (15 symbols, 10.766 baud, 172 Hz wide) used across Fldigi modes. Running a fast FFT-based RSID detector continuously on the live audio stream identifies modes in milliseconds with zero Fldigi process overhead—and can automatically trigger the exact Fldigi mode when detected.
5. **Bandpass-filter the ROI before Fldigi replay**: When `frequency_low_hz` and `frequency_high_hz` are known, apply a zero-phase Butterworth bandpass filter (with a modest margin, e.g., ±100 Hz) before writing `temporary_audio`, preventing strong adjacent signals from hijacking Fldigi's AFC.
6. **Improve `_decode_text_coherence()` and gate `DecoderTrialRunner` on coherence**: Score printable ratio, character entropy, amateur-radio/callsign patterns (`\b[A-Z0-9]{1,2}\d[A-Z]{1,4}\b`, `CQ`, `DE`, `RST`, `73`, `KN`, `AR`, `SK`), and case-insensitive bigrams/trigrams without harshly penalizing short (<48 char) transmissions. Require a minimum coherence threshold (or ≥6 printable characters) before `DecoderTrialRunner.advance()` declares `tentative_output`.
7. **Build out native faster-than-real-time decoders (`sigid/libsigdecode/`)**: `sigid/libsigdecode/` already exists as a placeholder. Adding fast in-process or CLI offline decoders (e.g., native RTTY/Baudot, BPSK31/Varicode, CW/Morse, plus `direwolf`/`multimon-ng` for AFSK1200/POCSAG/DTMF) turns 10-second real-time replays into <100 ms decodes and enables parallel live multi-channel decoding.

### 3.2 WSJT-X Backend (`sigid/wsjtx_backend.py`)

**Current limitations:**
- **FT8-only restriction (`wsjtx_backend.py:135`)**: `jt9` natively decodes FT8 (`-8`), FT4 (`-4`), JT65 (`-6`), JT9 (`-9`), Q65 (`-3`), FST4 (`-5`), and WSPR, yet only FT8 is wired.
- **Live UDP decodes are not correlated with live waterfall tracks**: `WsjtxUdpBackend.drain_events()` collects `delta_frequency_hz`, `snr_db`, and `observed_utc`, but nothing links these live decodes to tracks in `LiveTrackRegistry` or overlays them on the live waterfall.

**Recommendations:**
1. **Enable FT4, JT65, JT9, Q65, and FST4 in `WsjtxUdpBackend.decode()`**: Map each mode to its corresponding `jt9` CLI flag and minimum window duration (e.g., 7.5 s for FT4, 15 s for FT8, 60 s for JT65/JT9).
2. **Correlate live WSJT-X UDP events onto `LiveTrackRegistry`**: Match incoming UDP `DecoderEvent`s by `frequency_hz` (±25 Hz) and timestamp against active live tracks, attaching verified decode text and mode evidence directly to the live track overlay (still clearly marked as external WSJT-X evidence).

### 3.3 Protocol-Independent Raw Demodulators (`sigid/demod.py`)

**Current limitations:**
- **No carrier frequency offset (CFO) or phase-drift tracking in BPSK/QPSK (`demod.py:553-566, 637-655`)**: A single static phase estimate (`0.5 * angle(sum(z^2))` or `0.25 * angle(sum(z^4))`) fails completely if `carrier_hz` is off by even **0.5 Hz** (a 180° rotation in 1 second). Real off-air HF/VHF signals always have residual CFO and thermal/Doppler drift.
- **No `auto_timing` for BPSK or QPSK**: OOK, 2-FSK, and M-FSK support `auto_timing=True`, but BPSK and QPSK require manual `timing_offset_samples`.
- **No general symbol-rate estimator**: Only M-FSK has a coarse 9-step log-grid search (`demod.py:344-391`).
- **Per-symbol Python loop overhead (`demod.py:250-276, 539-558, 623-642`)**: Recomputes `np.hanning` and `np.exp(-2j * pi * f * times)` inside a Python `while` loop for every symbol, repeated 32× during `auto_timing`.

**Recommendations:**
1. **Pre-mix to baseband once, then matched-filter**: Mix the input array to complex baseband once (`z = x * exp(-2j * pi * f_c * t)`), apply a low-pass/root-raised-cosine matched filter, and slice symbols with vectorized indexing. This is 50–100× faster and eliminates per-symbol oscillator allocation.
2. **Add a Costas loop + differential decoding (DBPSK/DQPSK) for PSK**: Track residual CFO and phase drift symbol-by-symbol (e.g., second-order Costas loop or decision-directed PLL) and expose a `differential=True` option (essential for BPSK31/63 and many real-world PSK protocols).
3. **Add `auto_timing` (Gardner / Mueller-Müller or grid search) for BPSK and QPSK**, and add a fast **cyclostationary symbol-rate estimator** (`FFT(|z(t)|)` or `FFT(|diff(z)|)`) shared across all five demodulators to pre-populate the operator's symbol-rate field.

---

## 4. Suggested Implementation Roadmap

| Priority | Effort | Area | Concrete Change | Expected Impact |
| :--- | :--- | :--- | :--- | :--- |
| **P0** | Low | Bug fixes | Fix RGB→BGR in `YoloSignalDetector.embed_image()`; fix `detect_live_partial` noise RMS estimation; pass full track bounds (not 2 s crop) to `queue_live_mode_sweep`; expand `LIVE_ANALYSIS_SECONDS` to 30–60 s. | Immediate reliability & live UX win; unlocks live FT8 and fixes similarity search. |
| **P0** | Medium | Performance | Vectorize STFT in `spectrum.py` / `live_spectrum.py` and 2D resize in `cv_detector.spectrogram_image()`. | 10–30× faster spectrogram generation; smoother live CV & unattended scanning. |
| **P1** | Medium | Classification | Downconvert ROI to complex baseband (DDC) + add instantaneous frequency/phase, cumulant, and baud-rate features; fix hierarchical noise/OOD calibration and short-clip masking. | Eliminates frequency-position brittleness; stops reject-all calibration failures; major accuracy jump. |
| **P1** | Medium | Detection | Adaptive local noise-floor normalization for `spectrogram_image` (v3); disable YOLO HSV/erasing augmentations; add 1D CFAR + 2D blob detection to `detection.py` and unify CV + spectral into `LiveTrackRegistry` with Hungarian matching. | Closes real-capture domain gap; detects wideband modes (MT63/Olivia/SSB) and stops track ID swaps. |
| **P1** | Medium | Decoding | Warm isolated Fldigi worker + ROI bandpass pre-filter + classifier-driven mode shortlist + improved `_decode_text_coherence()` + native Python RSID detector. | Cuts live decode turnaround from ~45–60 s to a few seconds; identifies RSID modes in <50 ms. |
| **P2** | Medium–High | Models & Demod | Upgrade classifier from linear softmax to 2-layer MLP (or small baseband CNN) trained with HF fading/CFO/QRN/QRM augmentations; add Costas loop + DBPSK/DQPSK + `auto_timing` to `demod.py`; enable FT4/JT65/JT9 in `wsjtx_backend.py` and overlay live UDP decodes on tracks. | Strong real-world RF robustness across weak-signal, drifting, and multi-signal HF bands. |
