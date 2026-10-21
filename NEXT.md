# Next checkpoint

Status: local WAV upload/replay workbench, spectrogram, manual region labels,
sample-indexed replay, live audio capture, FSK fixtures and SQLite dataset index
are implemented. Receive-only Fldigi catalog selection and decoded-text display are
implemented. Isolated BPSK31, RTTY and THOR16 audio loopbacks pass through Fldigi into
the workbench API; radio-input routing and independent-signal validation remain open.

## Agreed scope

- Audio-passband waterfall, signal tracking, progressive identification and decoding.
- Full installed Fldigi receive catalog as the decoding target, including variants
  and specialized outputs; automatic identification coverage tracked separately.
- Real/synthetic datasets with partial and hierarchical modulation/protocol labels.
- A trained first-pass classifier with rejection and an auditable labeling loop.
- Unknown-mode symbol/bit/soft-output analysis for a bounded initial family set.

## Current implementation slice

- WAV upload/playback, sample-indexed audio replay, bounded spectrogram, drag-selected
  time/frequency regions, manual hierarchical labels, saved-region list and JSONL
  annotation download.
- SQLite capture metadata, source SHA-256, overlapping regions and append-only label
  revisions.
- Clean deterministic 2-FSK, M-FSK, BPSK and QPSK WAVs with manifests and exact
  reference symbols/bits; deterministic manifests are verified and imported with their
  protocol kept unknown.
- Generic 2–64-tone FSK fixture generation with exact tone symbols and, for
  power-of-two tone counts, exact natural-binary reference bits. MFSK fixtures
  carry family and method labels while protocol and variant remain null.
- Synthetic WAV/manifest import through CLI and web API. It regenerates the
  deterministic waveform to verify exact sample equality, stores the sidecar,
  records a full-capture synthetic modulation label and makes the manifest
  downloadable from the workbench.
- Local input-device discovery and bounded mono/stereo PCM recording through a
  background writer; live audio is saved as a normal capture with device provenance.
- Browser controls to start/stop live recording and immediately review its waterfall.
- Live capture also exposes a bounded rolling ten-second waterfall and provisional
  signal candidates while recording. The 10-second audio analysis ring is bounded in
  memory; final labels continue to reference the saved capture's absolute samples.
- Receive-only Fldigi API/UI: connection status, catalog list, manual mode selection
  and incremental decoded-text polling. Transmit, rig and macro calls are not exposed.
- Provisional automatic candidates from persistent spectral peaks, shown as amber
  dashed waterfall overlays with frequency/time, tone-count hypothesis, occupancy
  and peak SNR. Review selects an editable region and leaves all label fields blank;
  heuristic predictions are not stored as ground truth.
- Manual protocol-independent 2-FSK demodulation for a saved capture or selected
  region. Operator supplies symbol rate, bit-to-tone mapping and timing offset; results
  include raw bits, per-symbol soft metrics and sample bounds, with protocol unknown.
- Manual M-FSK tone-bank demodulation for a saved capture or selected region. It returns
  hard tone indices and confidence margins; natural-binary bits are emitted only for
  power-of-two tone counts, with protocol unknown.
- Manual BPSK analysis for a saved capture or selected region. It returns raw bits,
  soft metrics and both global polarity hypotheses because framing/preamble is unknown.
- Manual QPSK analysis for a saved capture or selected region. It returns raw symbols,
  candidate bits and soft metrics under all four 90-degree carrier rotations.

## Next action

Continue automatic identification in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md):
measure candidate recall and false alarms across synthetic and mixed scenes, improve
tone grouping and burst segmentation, and build the bounded decoder-trial scheduler.
Continue automatic identification and rate/frequency acquisition. Evaluate FSK, BPSK
and QPSK against carrier offsets, impairments and independent captures; improve phase
and timing acquisition before treating manual QPSK results as robust under field captures.
Current candidate hypotheses are heuristic-only and have not been validated on live
captures.

## Evidence boundary

Verified: Fldigi 4.2.13 and its XML-RPC method catalog are installed; unit tests
cover receive-only client semantics, PCM WAV block indexing/normalization and
synthetic 2-FSK/MFSK metadata, exact symbols/bits and tone frequencies, SQLite
capture hashes/regions/label history, JSONL export, spectrogram peak detection, upload/region APIs, and Fldigi
status/catalog/mode/output APIs. Current full suite: 68 passed, 4 skipped;
`node --check sigid/web/app.js` and `git diff --check` pass. Deterministic fake-device tests
exercise WAV writing, capture indexing, API start/status/stop and rolling live
analysis. The live analysis ring is capped at ten seconds and absolute sample
positions match completed-WAV waterfall calculations.
An isolated live Fldigi 4.2.13 process served the API with the documented receive-only
allow-list: 169 modes, THOR16 selection, and receive-buffer queries all worked. A
separate, opt-in integration test passes: it creates two copied profiles with all
rig/PTT paths disabled, routes BPSK31, RTTY and THOR16 transmit audio through a
temporary PulseAudio null sink, and verifies each complete payload arrives through
the workbench API. Run it with
`SIGID_TEST_FLDIGI_AUDIO=1 pytest tests/test_fldigi_audio_integration.py`. This is a
software loopback, not an independent modem or RF test. A fresh blank profile did
not expose XML-RPC until initialized. WAV replay is sample-indexed and synthetic
FSK/MFSK files carry modulation labels while leaving protocol unknown. Live hardware
capture, radio audio routing, trained identification model, and RF validation remain
unverified. Current heuristic candidates pass clean 2-FSK/MFSK, quiet-noise rejection
and two separated simultaneous 2-FSK signals; real-capture performance is unknown.
The tested Fldigi modes are a three-family sample, not full-catalog validation.
2-FSK/M-FSK/BPSK/QPSK recovery is verified against generated source symbols/bits;
2-FSK and BPSK also pass seeded AWGN cases. Independent-capture performance and
automatic rate/tone/carrier acquisition remain unverified.
