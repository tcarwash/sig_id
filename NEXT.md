# Next checkpoint

Status: local WAV upload/replay workbench, spectrogram, manual region labels,
sample-indexed replay, live audio capture, FSK fixtures and SQLite dataset index
are implemented. Receive-only Fldigi catalog selection and decoded-text display are
implemented. Isolated BPSK31/63/F/125/250/500/1000, QPSK31/63/125/250/500, 8PSK125/250/500, RTTY,
THOR8/16, MFSK16/32 and MT63-500S/1KS audio loopbacks pass through Fldigi into the workbench API;
radio-input routing and independent-signal validation remain open. Saved-capture
regions can be replayed in the selected mode through a separate Fldigi process and a
private PulseAudio/PipeWire sink; output is
persisted against the exact capture window. The audio-carrier control is prefilled from
selection-band midpoint, allows manual adjustment, and records the requested carrier
with each receive run. The inspector can also explicitly save that region with the selected
mode as its variant label. This replay path is native Linux only. Known-message audio
round-trips now pass for BPSK31/63/F/125/250/500/1000,
QPSK31/63/125/250/500, 8PSK125/250/500, RTTY, THOR8/16, MFSK16/32, OLIVIA-8/250 and MT63-500S/1KS.

Latest raw-analysis checkpoint: 2-FSK, OOK, M-FSK, BPSK and QPSK bit hypotheses now
offer byte-start offsets 0–7, MSB/LSB byte display, polarity inversion, and differential/
NRZI interpretations with both transition conventions and initial states. Per-view
metrics show bit/byte entropy, bit balance, transition fraction, longest run and printable
byte fractions; these are descriptive diagnostics, not modulation/protocol confidence.
Stencil matches are grouped by occurrence count and offsets but remain tentative clues.
Annotations save their exact transform/alignment settings with the derived bitstream and
continue to export with capture/stream hashes. Browser smoke verification covers alignment
selection, transforms, metrics, tag highlighting, persisted interpretation metadata and
the five demodulation exports. CRC/FEC validation, descrambling, and protocol-specific
frame decoding and bit de-stuffing are not implemented; protocol remains unknown unless a real decoder verifies it.
The Fldigi shortlist remains a partial, modulation-filtered catalog. Operators can save
reusable exact or wildcard stencils for later bitstream checks. The receiver output pane
is resizable by pointer or keyboard, remembers its height, and stays within the viewport.

## Agreed scope

- Audio-passband waterfall, signal tracking, progressive identification and decoding.
- Full installed Fldigi receive catalog as the decoding target, including variants
  and specialized outputs; automatic identification coverage tracked separately.
- Real/synthetic datasets with partial and hierarchical modulation/protocol labels.
- A trained first-pass classifier with rejection and an auditable labeling loop.
- Unknown-mode symbol/bit/soft-output analysis for a bounded initial family set.

## Current implementation slice

- Linux Compose packaging uses a digest-pinned Python base, hashed runtime/build
  dependency locks, a non-root service, loopback web binding and persistent `/data`.
  Isolated Docker validation covers actual UI workflows, raw demodulation, label
  revision/audio-hash persistence across restart, active synthetic capture finalization
  on container stop, and backup/restore into a fresh project. See `OPERATIONS.md` and
  `reports/compose-validation-2026-10-21.json`. Host audio-device routing is not
  automatically passed into the container; native capture and independent RF checks
  remain separate validation gates.

- Waterfall-centered UI: viewport workspace, source toolbar, settings/library drawers,
  selected-signal inspector, one raw-demodulator selector and a collapsible output dock.
  Readable track/annotation labels select signals in place, with collision overflow,
  keyboard access and provisional evidence text. Library reopens saved recordings.
  Chromium workflow checks cover five demodulator exports, partial labels, saved-region
  selection and explicit decoder-mode activation at desktop/mobile sizes. The first
  layout checkpoint is followed by incremental binary spectrum transport at a 200 ms
  polling target, independently of the one-second candidate analysis. The display has a
  fixed thirty-second time scale and a bounded sixty-second row cache. Startup history
  and missing update ranges remain blank; Freeze view retains an immutable image while
  recording continues. Browser checks pass at 48/96/192 kHz with paced synthetic audio.
- Saved annotation markers now open an editable label with its complete revision history.
  Saving appends a revision on the same region, preserving original bounds and evidence;
  concurrent stale reviews receive a conflict instead of overwriting newer labels.
  Output context distinguishes shared Fldigi text, tentative track trials, raw-analysis
  capture/sample ranges and received-image library products. Delayed raw results are
  discarded after source changes. Hover/focus details and collision leader lines keep
  measurements available on the waterfall without permanently expanding every signal.
- Saved-region Fldigi replay runs persist selected mode, exact sample bounds, carrier,
  status, notes and recovered text. Playback uses a private null sink and a copied
  profile with audio set to PulseAudio and all known PTT paths forced off; it never
  changes the operator's running Fldigi instance. The label action remains explicit and
  saves the selected mode as the region variant. RF-free tests verify known-message
  round-trips in BPSK31/63/F/125/250/500/1000, QPSK31/63/125/250/500, 8PSK125/250/500, RTTY, THOR8/16,
  MFSK16/32, OLIVIA-8/250 and MT63-500S/1KS. Longer captures are used for interleaved modes.
- The capture inspector now separates browser audio playback (“Replay selected region”)
  from isolated Fldigi decoding (“Decode selected region”). Playback seeks to and stops
  at the selected sample interval; decode retains its mode choice and persisted run.
  Headless-browser checks cover saved and live selection affordances.
- A reproducible Fldigi 4.2.13 capability registry covers all 169 audited catalog
  entries. It distinguishes 165 selection successes from the twenty-three modes with
  known-message audio and isolated-region round-trip evidence; other selectable modes
  remain explicitly decode-unverified, and three unselectable OFDM entries plus the
  RSID end marker retain their documented states. See
  `reports/fldigi-4.2.13-capabilities.json`; broader per-mode decoding and specialized
  receive-product evidence remain required for the full-catalog 1.0 gate.
- While recording, the inspector and accepted-model scheduler can replay a selected
  live candidate's exact sample window using isolated Fldigi processes. The bounded
  tested-mode sweep tries up to three modes in hypothesis order and stops at the first
  tentative text result. It rejects a window that expired from the ten-second live
  analysis buffer, suppresses duplicate track attempts, enforces a six-starts-per-minute
  limit, persists exact session/track bounds and links the run to the saved capture on
  Stop. API and headless-browser checks cover manual and automatic scheduling.
- Graceful application shutdown finalizes an outstanding capture. Writer faults retain
  completed audio blocks and mark the indexed recording partial; metadata-index failures
  retain the readable WAV and report its path. Input close failures still drain the
  writer, and callback status messages are bounded. Fault-injection tests cover writer
  and index failures; browser tests verify Stop-error recovery leaves Start usable.

- WAV upload/playback, sample-indexed audio replay, bounded spectrogram, drag-selected
  time/frequency regions, manual hierarchical labels, saved-region list and JSONL
  annotation download.
- Waterfall axes now match the rendered matrix: audio frequency runs left-to-right and
  time runs top-to-bottom, with frequency ticks below and time ticks alongside the plot.
- Waterfall presentation uses a fixed 0–3 kHz frequency window. The default display
  uses a 4096-point FFT and 512-sample hop (about 11.7 Hz/bin at 48 kHz); capture
  input rates remain selectable at 48, 96, or 192 kHz.
- SQLite capture metadata, source SHA-256, overlapping regions and append-only label
  revisions.
- Clean deterministic 2-FSK, M-FSK, BPSK and QPSK WAVs with manifests and exact
  reference symbols/bits; deterministic manifests are verified and imported with their
  protocol kept unknown.
- Clean deterministic OOK/ASK WAV generation now preserves exact source bits and
  imports with a verified ASK-family label. OOK is not part of the current classifier
  classes; it provides an explicitly unfamiliar modulation source for a separate
  held-out rejection benchmark. `sig-id-evaluate-ood` verifies sidecar provenance and
  reports method/family false-accept rates plus waveform and model hashes without
  training on those inputs.
- Clean generic M-ASK sources carry exact amplitude-level symbols and natural-binary
  reference bits when representable. The independent ASK benchmark now stratifies OOK
  and 4-ASK lineages across validation and test to measure transfer between amplitude
  modulation variants.
- A deterministic binary PPM generator encodes each bit in one of two raised-cosine
  carrier-pulse positions within the symbol. Its manifest preserves the exact bits,
  slot timing, carrier and pulse-width recipe; synthetic import re-renders the WAV.
- `sig-id-build-ood-corpus` creates resumable, hash-indexed independent ASK and PPM
  sources (OOK, 4-ASK and binary PPM) with varied payloads, symbol rates, carrier
  frequencies, amplitudes, phase and pulse width. Its
  separate index is not accepted as training data. The trainer can use the corpus's
  disjoint 3:2 validation/test source split to calibrate method/family abstention on
  unknown-family negatives without fitting on them, with test false-accept metrics
  broken down by modulation method and family. Version 0.2.0 remains available to
  exactly replay the committed historical ASK-only benchmark.
- Generic 2–64-tone FSK fixture generation with exact tone symbols and, for
  power-of-two tone counts, exact natural-binary reference bits. MFSK fixtures
  carry family and method labels while protocol and variant remain null.
- Synthetic WAV/manifest import through CLI and web API. It regenerates the
  deterministic waveform to verify exact sample equality, stores the sidecar,
  records a full-capture synthetic modulation label and makes the manifest
  downloadable from the workbench.
- Deterministic AWGN augmentation for imported synthetic parents, with declared SNR
  convention, seed, gain, clipping and measured SNR. Import validates parent waveform
  and manifest hashes, regenerates the child exactly, and stores its parent capture ID.
- Resumable synthetic corpus command for 2-FSK, 4/8/16-FSK, BPSK and QPSK across
  configurable source counts and AWGN levels. Atomic `corpus.json` indexing groups
  each clean source with all derived SNR variants and avoids duplicate capture imports.
- Corpus builder also creates deterministic independent noise-only captures with no
  modulation/protocol labels. Split-safe validation noise constrains abstention
  calibration to zero observed noise false positives; test output reports noise false-
  positive rate. When noise and known-mode precision constraints cannot both be met,
  the model abstains on every example. This only evaluates synthetic noise rejection,
  not unseen modulated families.
- Lineage-safe split manifest generation validates capture hashes, parent links and
  accepted method labels, stratifies source groups by modulation method, requires at
  least three independent groups per method and for noise, stratifies on stable source
  waveform hashes rather than database UUIDs, and proves each method appears in all splits.
- A NumPy softmax baseline trains method proposals from known training groups only,
  calibrates method/family rejection thresholds on validation groups including noise
  negatives, reports held-out method/family confusion and noise false positives, and
  exports hashed weights plus versioned metadata. The API can load
  that artifact and propose labels for a saved region; proposals require explicit review.
- Feature version `passband-log-spectrum-envelope-v2` adds six normalized envelope
  statistics and stores one standardized centroid per trained modulation method. The
  trainer calibrates a nearest-centroid RMS distance gate on known validation sources
  against noise and separate unknown-family validation sources. The workbench reports
  the distance and gate decision with each proposal; legacy v1 artifacts still load.
- `sig-id-evaluate-split-seeds` repeats training across lineage-safe partitions and
  summarizes unique assignments only. Split assignment v2 makes seed variation affect
  group order; v1 remains the default for historical replay. Five independent ASK/PPM
  evaluations averaged 64.6% known test coverage and 79.8% accepted accuracy, with zero
  method/family false accepts on each 12-source OOD test. Accepted accuracy ranged from
  58.8% to 93.8%, so the earlier single-split result was optimistic. See
  `reports/multiseed-ask-ppm-2026-10-21.json`.
- Training reports now separate held-out test metrics by clean, noise and AWGN SNR.
  Across five seeds and 0/3/6/10/20 dB variants, AWGN coverage averaged 78.3–85.0%;
  clean-source coverage was only 13.3%, despite 71.7% raw accuracy. This flags a model
  calibration/feature weakness. The 444-capture sweep is archived at
  `reports/channel-snr-multiseed-2026-10-21.json`; multipath, fading and impulsive
  interference remain untested.
- Bounded Signal Identification Wiki metadata importer covers identified and
  unidentified page categories. It stores revision, source-reported fields, media links
  and retrieval time, while leaving source labels unreviewed and media link-only pending
  per-asset rights review. It does not add Wiki entries to model training.
- Local input-device discovery and bounded mono/stereo PCM recording through a
  background writer; live audio is saved as a normal capture with device provenance.
- Browser controls to start/stop live recording and immediately review its waterfall.
- Live capture also exposes a thirty-second waterfall display and provisional
  signal candidates while recording. The 10-second audio analysis ring is bounded in
  memory; final labels continue to reference the saved capture's absolute samples.
  Its vertical axis stays relative (`−30 s` to `Now`) while the display updates
  independently of the one-second candidate analysis. The browser caches at most
  sixty seconds of spectrum rows; the server audio analysis ring stays at ten seconds.
- Live candidate IDs persist across rolling analyses, tolerate limited center-frequency
  drift, and expire after five seconds without a detection. Three observations mark a
  candidate persistent; this is spectral continuity evidence, not mode confirmation.
  The spectral detector now estimates within-window linear carrier drift and shows the
  rate with candidate frequency, timing, tone-count and SNR evidence.
- Receive-only Fldigi API/UI: connection status, catalog list, manual mode selection
  and incremental decoded-text polling. Transmit, rig and macro calls are not exposed.
- Olivia-8/250 loopback now recovers the known payload through both the workbench API
  and saved-region replay when given a repeated message and a longer clip. This resolves
  the earlier short-capture failure for that mode; other Olivia variants remain unverified.
- `sig-id-mix-scene` now combines 2–8 accepted synthetic captures with exact sample
  offsets, gains, optional deterministic carrier offsets, linear frequency drift, and
  smooth low-pass filtering and bounded sample-clock error. The manifest retains both
  source and rendered sample bounds;
  `sig-id-import-scene` verifies the rendered samples and stores
  overlapping constituent regions with their source method labels. The split manifest
  joins source lineages connected by a scene, places a scene-linked cluster in the test
  partition, and reports scenes whose components are not all in the source corpus.
  The method model now trains and calibrates on constituent crops in the train and
  validation splits, with the other overlapping signal as context; test-scene crops
  remain held out and score separately. `sig-id-evaluate-detector` reports temporal component recall,
  candidate precision, and false candidates/hour over mixed scenes and noise-only
  captures. Broader weak-signal and speech-interference evaluation remains open.
- A mode-by-mode isolated Fldigi 4.2.13 selection audit records all 169 catalog
  entries: 165 of 168 actual modem modes select by name. OFDM500F, OFDM750F and
  OFDM3500 remain listed in Fldigi but selecting them through XML-RPC leaves the
  backend on BPSK31. The official Fldigi manual documents them as real OFDM modes.
  `RSID_EOT` is not a receive mode and is now omitted from the workbench selector.
  The client rejects silent mode-switch failures. This is selection evidence only,
  not decode support. See `reports/fldigi-4.2.13-catalog-selection.json`.
  `reports/fldigi-4.2.13-capabilities.json` derives per-mode validation states from
  that audit and the named round-trip tests; the builder and its completeness/claim
  checks live in `sigid/fldigi_capabilities.py` and `tests/test_fldigi_capabilities.py`.
- Live waterfall follow mode has a fixed thirty-second scale, with unavailable startup
  history and dropped display-update ranges left blank.
- Fldigi WEFAX receive products can be polled, validated as bounded supported image
  files, stored by content hash, and shown in the received-image gallery.
- Added a generic receive-text backend contract for mode listing, mode selection
  and incremental text output; Fldigi implements it, and the bounded trial runner
  now depends on that contract. Fldigi remains the only implemented protocol
  decoder backend; the contract does not itself add mode coverage.
- The selected live-candidate trial action now replays its exact sample window through
  separate isolated Fldigi receive instances, trying up to three loopback-tested modes
  in hypothesis order and stopping at the first tentative text result. It rejects stale
  ring-buffer intervals, suppresses per-track duplicates and limits work to six sweeps
  per minute. Accepted-model automatic trials use the same path after a track persists.
- Receive trials are durable records. Each isolated run retains its live session,
  candidate track ID, exact sample window, sample rate, attempted modes, output text and
  final state. Stopping a recording links those runs to the saved capture; reopening it
  lets the operator review output alongside the explicit `isolated_live_mode_sweep`
  scope and `exact_live_track_window` association. A legacy direct shared-input trial
  route remains separately marked as tentative.
- Provisional automatic candidates from persistent spectral peaks, shown as amber
  dashed waterfall overlays with frequency/time, tone-count hypothesis, occupancy
  and peak SNR. Saved-capture track analysis adapts FFT size to sample rate and
  suppresses nearby sidelobe peaks; regular observed tone grids expose provisional
  spacing estimates and a `Use tone estimates` action that prefills the M-FSK bank,
  timing search and bounded baud search. Review selects an editable region and leaves
  all label fields blank; heuristic predictions are not stored as ground truth.
- Manual protocol-independent 2-FSK demodulation for a saved capture or selected
  region. Operator supplies symbol rate and bit-to-tone mapping; timing phase can be
  entered manually or acquired with a 32-phase soft/energy search. Results include raw
  bits, search diagnostics, per-symbol soft metrics and sample bounds, with protocol unknown.
- Manual noncoherent OOK demodulation for a saved capture or selected region. It
  estimates off/on carrier levels, returns raw bits and amplitude soft metrics, searches
  timing phase on request, and abstains when two amplitude levels cannot be distinguished.
  Framing and protocol remain unknown; exports retain capture hash and sample bounds.
- Manual M-FSK tone-bank demodulation for a saved capture or selected region. It returns
  hard tone indices and confidence margins; natural-binary bits are emitted only for
  power-of-two tone counts, with protocol unknown. Timing phase can be entered manually
  or acquired through a 32-phase confidence/energy search; estimated phase and candidate
  scores are included in both the UI and exported result. An optional bounded logarithmic
  symbol-rate search ranks operator-supplied baud bounds, searches timing at each rate,
  and displays the strongest rate candidates while preserving the full score list.
- Manual BPSK analysis for a saved capture or selected region. It returns raw bits,
  soft metrics and both global polarity hypotheses because framing/preamble is unknown.
- Manual QPSK analysis for a saved capture or selected region. It returns raw symbols,
  candidate bits and soft metrics under all four 90-degree carrier rotations.
- Each raw demodulator can export its complete result and settings as JSON with capture
  hash, source sample range and protocol-unknown status for reproducible manual analysis.

## Next action

The core GUI overhaul, Linux Compose run path, exact-window replay for saved/live regions
and isolated accepted-model trials are implemented. Continue with broader isolated Fldigi
decode fixtures and per-mode capability evidence, then address unresolved catalog modes,
remote capture, job/review workflows, quotas and prolonged resource soak on the 1.0 gate.
The classifier's clean-source coverage weakness also remains open, as detailed below.

The reproducible historical multi-ASK calibration/evaluation report is saved at
`reports/ask-ood-benchmark-2026-10-21.json`, with generation and training commands in
README. An end-to-end experiment with 12 known source groups per mode and 30 separate
ASK sources (15 OOK and 15 4-ASK) produced 0/12 method and family false accepts on the
ASK test set. Known-mode method coverage was 34.6% with 100% accuracy among accepted
examples; family calibration rejected every proposal. The spectral-only ASK/PPM baseline
also rejected all known examples. The envelope/prototype v2 result is recorded at
`reports/ask-ppm-novelty-benchmark-2026-10-21.json`: on the same split it retained 84.6%
known test coverage with 86.4% accepted accuracy and had 0/12 ASK/PPM false accepts.
This is synthetic-only evidence, not general unknown-mode or RF performance. Next fix
the low clean-source coverage and test multipath, impulsive interference and held-out
mixed scenes before using the novelty threshold for routine live proposals. An opt-in
automatic trial now requires a persistent track and an accepted
method proposal, while abstentions never trigger mode changes. Keep it opt-in until
unseen-family and real-capture validation establish useful operating limits. The
proposal path scores up to four strongest tracks per update over at most two seconds,
retaining model hash and sample bounds.
Continue automatic identification in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md): measure
candidate recall and false alarms across synthetic and mixed scenes, improve tone
grouping and burst segmentation, and evaluate FSK, BPSK and QPSK against broader
impairments and independent captures.
Current candidate hypotheses are heuristic-only and have not been validated on live
captures.

## Evidence boundary

Verified: Fldigi 4.2.13 and its XML-RPC method catalog are installed; unit tests
cover receive-only client semantics, PCM WAV block indexing/normalization and
synthetic 2-FSK/MFSK metadata, exact symbols/bits and tone frequencies, SQLite
capture hashes/regions/label history, JSONL export, spectrogram peak detection, upload/region APIs, and Fldigi
status/catalog/mode/output APIs, WEFAX product import and image delivery. Current full suite: 186 passed, 20 skipped;
the opt-in isolated selection audit checks all 169 entries (165 selectable, four unresolved),
and `node --test tests/test_waterfall_window.mjs`, module syntax validation, headless
Chromium loading, and `git diff --check` pass. The generic text-backend
contract is exercised by the Fldigi adapter and trial-runner tests. Deterministic fake-device tests
exercise WAV writing, capture indexing, API start/status/stop and rolling live
analysis. The live analysis ring is capped at ten seconds and absolute sample
positions match completed-WAV waterfall calculations.
An isolated live Fldigi 4.2.13 process served the API with the documented receive-only
allow-list: 169 modes, THOR16 selection, and receive-buffer queries all worked. A
separate, opt-in integration test passes: it creates two copied profiles with all
rig/PTT paths disabled, routes BPSK31/63/F/125/250/500/1000, QPSK31/63/125/250/500, 8PSK125/250/500, RTTY,
THOR8/16, MFSK16/32, OLIVIA-8/250 and MT63-500S/1KS transmit audio through a temporary PulseAudio null sink, and
verifies each complete payload arrives through
the workbench API. Run it with
`SIGID_TEST_FLDIGI_AUDIO=1 pytest tests/test_fldigi_audio_integration.py`. This is a
software loopback, not an independent modem or RF test. A fresh blank profile did
not expose XML-RPC until initialized. WAV replay is sample-indexed and synthetic
FSK/MFSK files carry modulation labels while leaving protocol unknown. A NumPy
softmax/prototype model has been trained and integrated for reviewed proposals, but its
clean-source test coverage is weak; live hardware capture, radio audio routing and RF
validation also remain open. Current heuristic candidates pass clean 2-FSK/MFSK, quiet-noise rejection
and two separated simultaneous 2-FSK signals; real-capture performance is unknown.
The tested Fldigi modes remain a small sample, not full-catalog validation.
2-FSK/M-FSK/OOK/BPSK/QPSK recovery is verified against generated source symbols/bits;
2-FSK and BPSK also pass seeded AWGN cases. M-FSK rate search is verified on a small
synthetic 4-FSK grid. Independent-capture performance and automatic tone/carrier
acquisition remain unverified.
The baseline trainer and inference path are tested end to end. Live candidate inference
uses a bounded two-second snippet for up to four strongest tracks and returns auditable
model/sample provenance. An opt-in automatic trial starts only for persistent tracks
with accepted model proposals; abstentions do not switch modes. Output remains tentative.
An exploratory split
with 10 independent synthetic sources per method and one 10 dB variant achieved 10/12
raw method decisions on its held-out groups; a four-source run achieved 11/24. A newer
12-source-per-method run with one 10 dB variant calibrated to zero observed validation
noise false positives; its held-out noise false-positive rate was 0/2, overall coverage
was 46.2%, and accuracy among accepted test examples was 66.7%. A three-source training
corpus cannot meet both validation constraints and abstains on every example. These
small synthetic-only runs show the rejection pipeline and its current poor utility;
they are not field-performance claims. Unknown modulated-family rejection remains
unvalidated.
