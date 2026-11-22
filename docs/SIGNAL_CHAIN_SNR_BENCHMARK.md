# Signal chain SNR + acquisition benchmark (synthetic, not on-air)

This is a reproducible engineering test for Signal ID's **audio → detection → native decoding → evidence/annotation** chain. The evaluation keeps separate scores for signal detection, intact decoded payloads, and false positives on noise. It does not treat a detector hit as a recognized protocol, model confidence as a calibrated probability, or readable text as a reviewed label.

## Run the same test

From the project checkout with CV dependencies installed:

```bash
PYTHONPATH=. python tools/bench_signal_chain_snr.py \
  --sample-rate 48000 --snrs=-5,10,25 --compare-legacy \
  --cv-model /path/to/your/cv-signal.pt \
  --output /tmp/signal-chain-snr.json
```

Omit `--cv-model` to test native modem and narrow-tone detection without YOLO; choose `--sample-rate 12000 --snrs=-15,-10,-5,0,10,25` for a faster weak-signal sweep. The harness is **receive-only**: it renders temporary mono PCM16 WAVs and runs first-party decoder and detector functions, without opening a live sound device, radio, PTT, external service or changing the 8013 preview. `--keep-wavs /tmp/my-signal-test` preserves the stimuli for manual review; otherwise WAV files are temporary.

The generated 30-second Signal Lab scene has **one mode at a time**, a fixed 2.5-kHz reference noise bandwidth, no intentional QRM tones, a reproducible seed and a minimum nine-second transmission containing repeated **complete** SHA256-pinned synthetic messages. The last ten seconds are noise-only. Each run records input-WAV SHA256, source-fixture SHA256, actual carrier, seed, SNR, sample rate, model SHA (if provided), decoder text/status and measured wall/CPU times. The report deliberately does not persist source audio or claim real radio evidence.

## What the independent metrics mean

- **Oracle exact:** decode one complete original message using ground-truth sample boundaries and carrier. For AX.25, the payload must match *and* the FCS pass.
- **Four-second live crop:** start exactly 350 ms into the first message and process up to four seconds using the live native decoder worker. This reproduces a difficult acquisition phase without giving it fixture boundary information. Count only crops with one or more *complete* embedded messages in the denominator for acquisition-specific performance. The chosen CW fixture is about 3.85 seconds long; this deliberately offset four-second crop contains **no complete CW message** and should not be scored as a correctable decoder failure.
- **Legacy comparison:** with `--compare-legacy`, decode exactly the same crop in one full-window modem invocation, matching the former worker behavior. This is an apples-to-apples control, using the **same** current modem implementation, carrier and audio.
- **Noise-only decoder false output:** decode a separate four-second interval without a generated transmission. Any non-abstained coherent-text or checksum-passed frame is counted as a false output.
- **CV detected:** run the configured YOLO model on an active ten-second audio window, then check *both* time and frequency overlap with Signal Lab truth. Separately score YOLO boxes in the background-only ten-second window. This is geometric coverage, **not** protocol classification.
- **Independent narrow-tone matched:** run the existing 4096-point FFT / 512-hop spectral detector with the CV-preferred 35 Hz tone-gap, −32 dB minimum peak and 18 dB above-row-floor threshold. Only strong single-tone narrow candidates count, matching the live UI filter. An RTTY tone can be detected as a *single-tone unknown*; this does **not** establish RTTY or CW identity. Count noise-only candidates separately.

The benchmark has one deterministic seed per SNR and four synthetic modes (CW, RTTY, BPSK31 and AX.25) with unequal test-message lengths. A perfect score here cannot establish reliability in crowded bands, unusual baud rates, drifting audio, faded/real RF, packet fragments, unknown modulations or out-of-distribution noise. Model training and fixture styles may overlap, so even these synthetic model hits are not held-out real-world validation.

## Results (2026-11-22, tyler-ldesktop)

Using the installed CPU YOLO model, at the normal **48-kHz Signal Lab live input rate**:

| Nominal SNR | CV matched / 4 | Narrow-tone activity matched / 4 | Legacy whole-crop exact / evaluable | Segmented native exact / evaluable | Noise-only decoder false outputs |
|---|---:|---:|---:|---:|---:|
| −5 dB | 1 | 2 | 0 / 3 | 0 / 3 | 0 / 4 |
| +10 dB | 2 | 4 | 1 / 3 | 3 / 3 | 0 / 4 |
| +25 dB | 2 | 4 | 1 / 3 | 3 / 3 | 0 / 4 |

Actual trial metadata and results are in [the reproducible report](../reports/signal-chain-snr-48khz-2026-11-22.json), including which specific mode was detected. This is a **small fixed synthetic corpus**, not a statistically significant general precision/recall estimate. There were no noise-only YOLO boxes or selected narrow-tone candidates in the twelve noise-only controls.

A separate 12-kHz six-SNR run showed the CV model matching **zero of six CW cases** even though the narrow-tone detector matched five. The 48-kHz benchmark found two of three CW cases with CV, suggesting sample-rate/model-preprocessing sensitivity. Do not infer a general detector comparison without matched real recordings and multiple seeds.

## Acquisition improvement

Previously, the native worker gave every RTTY/BPSK31 mode candidate one up-to-four-second waveform. In repeated-message recordings it often failed to acquire on that whole crop, even when a complete message appeared between two truncated neighboring messages.

The receive-only `sigid.native_acquisition.propose_burst_windows` now checks two fixed mode bands (**RTTY carrier ±140 Hz**, **BPSK31 carrier ±85 Hz**) for a strong low-energy inter-burst gap. It computes an energy envelope at 15-ms hops, requires at least **5:1** 90th/12th-percentile contrast, merges brief fades, and proposes only **fully enclosed** active spans separated by at least 130 ms of quiet. At most **two** bounded candidate spans are decoded, followed by a **single full-window fallback** if neither yields coherent evidence. Noise, unsupported modes, continuous tones, weak contrast, invalid carriers, and unrecognized signal structure use the original full-window path.

This does **not** use expected text or fixture truth at runtime. It never changes the live audio callback, the native worker process count, four-second input cap, rolling CPU budget, automatic identity/label policy, or optional Fldigi/OnAir/WSJT-X backends. The worker persists original and chosen absolute sample windows, attempted windows and outcomes in `decoder_result.live_acquisition`. The existing workbench comparison cards now display this information, with an explicit unverified qualifier.

At both +10 and +25 dB, exact payload recovery on the three *evaluable* full-message crops rose from **1/3 to 3/3** with the identical source audio and identical decoder implementation. At −5 dB, segment acquisition still failed despite an oracle RTTY success, so the changed worker must not be described as a general weak-signal decoder improvement.

**Candidate selection also changed:** a non-abstained, named classifier mode is tried first, with a frequency/width hypothesis as a second fallback (at most two modes). Frequency *alone* can no longer silently override a specific model suggestion, such as CW at a high audio pitch or RTTY at a low pitch. Both remain provisional: neither a model suggestion nor a segmented readable string becomes a verified identity.

## Safety/regression test commands

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_native_acquisition.py tests/test_native_live_decode.py \
  tests/test_native_live_api.py tests/test_signal_chain_snr_benchmark.py

node --test tests/test_decoder_result.mjs tests/test_decoder_artifact_ui.mjs
```

Tests cover exact RTTY/BPSK31 recovery from offset-window repeated transmissions at multiple synthetic SNRs, same-input noise abstention, continuous-carrier negative controls, bounded trial count, absolute sample provenance, classifier-vs-frequency hypothesis ordering, and workbench presentation. The full identification/decoder regression suite and native conformance fixtures should continue to pass before deployment.

## What still needs work

1. **Weak-signal acquisition:** At −5 dB a complete oracle RTTY message can decode at 48 kHz, but the arbitrary four-second cropped stream still fails. The current conservative contrast gate intentionally refuses to split signal/noise when their levels overlap. Adaptive matched filtering, sync search, soft decisions, or streaming symbol state will be needed, with realistic interference/false-positive controls.
2. **Long CW:** A four-second arbitrary crop of a nearly four-second message may contain no whole message. A streaming key-down/up state machine or a longer bounded CW-only snapshot is preferable to blindly classifying the resulting partial text as wrong.
3. **RTTY and wideband detection:** The CV model misses some strong RTTY in this corpus. The separate 35-Hz-gap narrow-tone detector can find its individual tones, but those are currently labelled unknown narrow tones, not a paired FSK hypothesis. Developing a conservative complementary-tone grouping with negative controls is a future experiment.
4. **AX.25 carrier tuning:** The first-party decoder still requires the fixed 1200/2200-Hz Bell 202 mark/space pair. This benchmark does not claim arbitrary-shift packet reception.
5. **External decoder comparison:** The workbench already supports comparing two or three persisted native/external runs, including source/backend, modes tried, sample/frequency geometry and output. Native segmentation provenance now appears there. OnAir remains an optional event bridge, not a general audio-decode engine. Evaluate external decoders on identical stored crops before interpreting differences as modem quality.
6. **Field validity:** Evaluate multiple seeds, deliberately overlapping stations, live tracking/decoding latency, remembered channel offsets, recordings with independently reviewed annotations, and negative windows before claiming statistically meaningful detection or decoding performance.
