# Implementation plan

Status: implementation in progress. WAV replay, live capture, the manual dataset
workflow, and the first provisional spectral candidate pass are implemented. Fldigi's
receive-only UI/API is connected, with isolated audio-loopback evidence for BPSK31,
RTTY and THOR16. A manual protocol-independent 2-FSK demodulator also exposes raw
bits and soft tone metrics; see [NEXT.md](NEXT.md) for verified gates. Automatic
protocol identification, model training and RF validation remain open. A BPSK31
software audio loopback decodes through Fldigi; routing a real radio input remains open.

## Delivery strategy

Build a useful receive-and-label application first, then turn on increasingly
automatic operation. Prove audio into/out of Fldigi before building around it.
Use that same integration for decoding and synthetic generation. Collect real
captures from the first usable release while developing the model and expanding
the modem catalog. Do not wait for a large model or every modem before shipping.

Release ladder:

| Release | User can do | Gate |
| --- | --- | --- |
| 0.1 workbench | View radio audio/WAVs, select a signal, decode initial modes, label and replay captures, generate initial fixtures | Actual audio-to-text path and durable annotations |
| 0.2 automatic alpha | Track multiple signals, see family/protocol candidates, automatically try decoders and save events | Measured automatic operation, including unknowns and no-RSID signals |
| 0.3 learning beta | Generate corpora, review labels, train/export a baseline model, use it to propose decoders | Split-safe training and independent real-capture evaluation |
| 1.0 full workbench | Use the Fldigi catalog, non-text outputs, integrated learning tools, and unknown-mode symbol/bit inspection | Catalog audit, supported demodulators, packaging and operational acceptance |

All Fldigi receive modes remain the target. Generic unknown-mode analysis initially
covers 2-FSK, M-FSK, BPSK, and QPSK; universal demodulation of every possible waveform
is not a finite release gate. Automatic protocol identification has an explicit
validated coverage list, and may return a family or unknown for other modes.

## Defaults that remove setup decisions

- Linux first, local browser UI; source checkout and Docker Compose launch paths.
- Python DSP/domain package, FastAPI service, TypeScript/Vite UI with canvas waterfall.
- NumPy/SciPy for initial DSP, SQLite plus lossless audio/JSON assets for storage.
- Local audio-device capture and WAV replay first; remote capture agent later in beta.
- Separate Fldigi worker processes and a minimal virtual display if required. Start
  with one worker, then a configurable bounded pool after routing isolation works.
- One application process and local job runner initially. No broker, distributed
  database, or custom native modem rewrite on the initial critical path.
- Training runs separately from capture; CPU inference first, optional GPU training.
  Pick the export/runtime format after a small baseline proves the data path.
- Persist annotations in sample coordinates. UI waterfall pixels are presentation.

Verify package versions and actual Fldigi capabilities during setup. Keep backend
control ports internal. Ordinary receive workers cannot invoke transmit controls;
offline generation workers have isolated audio output and no radio/PTT integration.

## Critical path

    Fldigi integration proof -> receive/replay workbench -> tracking/scheduling
                            -> synthetic fixtures -> corpus/model -> proposals
                            -> catalog validation and output adapters
    sample/annotation contracts -> unknown-mode demodulators -> inspection UI

These are parallelizable development tracks after the shared contracts stabilize,
not a requirement for multiple agents or developers. With one implementer, finish
the current release gate before opening additional tracks.

## Milestone 0: eliminate integration uncertainty

Timebox the first investigation to 1-2 focused implementation days, then decide.

1. Inventory the installed/toolchain versions and query the modem catalog. Write
   a machine-readable capability registry with stable internal IDs, backend names,
   configurable variants, output types, and validation status. Parameterized modes
   need explicit variant records, not just the returned modem name.
2. Launch isolated Fldigi instances, select a mode, generate a known payload into its
   input and retrieve text. BPSK31, RTTY and THOR16 now pass this audio loopback gate;
   continue with additional families and independent fixtures.
3. Repeat across remaining representative Fldigi families. Measure startup/interleaver delay,
   input pacing, stale-output clearing, tune/selection behavior, and output latency.
4. Capture offline generated audio for a known payload; decode it again. Store the
   clean waveform and manifest. Prove that no radio audio output is involved.
5. Test two isolated instances with different inputs and verify no audio, control,
   settings, or output cross-talk. Establish the RAM/CPU cost per worker.
6. Prove a bounded buffered replay path. If the backend only accepts real-time
   audio, measure that limit; never assume it can catch up faster than real time.

Deliver `tools/probe_fldigi.py`, saved fixtures/manifests, a capability report, and
one repeatable integration command. Distinguish backend integration tests from
on-air validation and independent interoperability checks.

Decision: retain XML-RPC plus an explicitly tested audio route if adequate. If not,
inspect Fldigi's supported interfaces/source and choose a narrowly scoped adapter
or maintained integration shim. Do not spend weeks rewriting all modems. Record
the selected route and its packaging/license obligations before distributing it.

If faster replay is unavailable, live workers can decode from current audio while
a separate job processes the earlier recording; display separate timelines and
deduplicate results. This limitation does not block 0.1. If a mode is unavailable
through the initial interface, keep it visibly pending in the catalog and solve
its adapter before claiming full-catalog support.

## Milestone 1: release 0.1, useful receive-and-label workbench

Build only the UI needed for the full workflow: source controls, main waterfall,
selected-track panel with mode dropdown and decoded output, and dataset browser.

Backend:

- Audio source interface for WAV and local device; explicit channel selection,
  resampling metadata, sample continuity, bounded rolling buffer and disk capture.
- One STFT/waterfall stream with binary/batched updates; avoid JSON per FFT bin.
- Manual time-frequency selection, frequency tuning, decoder selection, and stop.
- Initial Fldigi adapter from milestone 0; retrieve output and attach backend
  timestamps/available sample mapping without inventing character-level timing.
- Recording/event/annotation CRUD, partial labels, revisions, and file export.
- Minimal generator command for proven modes with known payload, seed and labels.

Data contract version 1 includes `Capture`, `Region`, `Track`, `ModeHypothesis`,
`DecodeRun`, `LabelRevision`, `GeneratedSource`, and `DatasetManifest`. Protocol,
family, method and variant are distinct nullable attributes. Evidence, prediction
scores, and review status are separate. Parent-source IDs follow all derived data.

Acceptance:

- Load a fixture, see its signal, select it, choose a proven mode, and recover the
  expected payload, accounting for documented modem text transformations.
- Repeat with live audio loopback and then an independent radio recording; label
  each validation type correctly. Hardware input requires an available device.
- Save a region, restart, replay it and recover the same bounds/labels/audio hash.
- Label an unknown region only as MFSK and export it without inventing a protocol.
- Annotate overlapping signals independently. Stopping capture leaves a readable
  completed file or an explicitly recoverable partial capture.
- Run for 30 minutes without unbounded queues/memory; surface overruns.

Ship this as soon as the gate passes. No classifier is required for 0.1.

## Milestone 2: release 0.2, automatic operation

- Run the same bounded candidate analysis over a rolling live-audio window, while
  preserving absolute capture sample positions and keeping callback work nonblocking.
- Add multiresolution signal detection/noise estimation and persistent tracks;
  support manual correction when segmentation merges or splits signals badly.
- Live detections now receive stable per-session IDs, tolerate limited frequency drift,
  expire after five seconds without a match, and become persistent after three new
  sample-position observations. This persistence is spectral evidence only.
- Estimate bandwidth, tone structure, spacing/rate candidates and drift. Start
  with interpretable family hypotheses; do not force a protocol on every track.
- Associate RSID announcements with tracks. Validate the event interface instead
  of assuming a selected-modem change is a complete RSID event feed.
- Add bounded candidate scheduler, trial timeouts, worker resets, minimum dwell,
  and hysteresis. Scale the pool only after per-worker routing is proven.
- A first user-started trial runner now sweeps only loopback-tested BPSK31, RTTY and
  THOR16 candidates, with bounded dwell/rate limits and tentative output semantics.
  Automatic launch policy, worker reset/isolation and additional modes remain open.
- Run initial decoder candidates, record mode-specific evidence, and display
  tentative versus supported output. Cap attempts per signal and per second.
- The loaded method model now scores bounded two-second snippets for up to four
  strongest live candidates per analysis update. Accepted family proposals move to
  the top of the candidate list and guide the existing tested-mode trial order;
  every result retains model hash/sample bounds and remains a proposal. Automatic
  scheduler policy and real-capture calibration are still open.
- An explicit live UI opt-in starts a bounded receive-only tested-mode sweep after a
  track is persistent and the model accepts its method proposal. Abstentions, short
  snippets and one-off detections cannot start it; output stays tentative. Held-out
  mixed-scene and real-capture validation remain required before recommending the
  feature for routine use.
- Save identified events automatically with pre-roll; queue unknowns according to
  configurable quotas. Add storage usage and retention controls.

Acceptance: a repeatable mixed-scene suite includes isolated signals, two separated
signals, overlap, drift, weak signals, speech/noise/carriers and discontinuities.
Verify stable track IDs, bounded work, no false trusted labels, and successful
automatic decoding of the initial modes. Test with RSID absent and present.

Report false detections/hour, per-mode identification precision/recall, time to
first candidate/identification, automatic payload success and resource use on a
named machine. Establish numeric signal-quality operating ranges from fixtures;
do not substitute an arbitrary global accuracy number. Failed/ambiguous scenes
must remain visible, not disappear from the denominator.

## Milestone 3: release 0.3, dataset and learning loop

Synthetic track:

- Batch generator with cancellable/resumable jobs, deterministic recipes and clean
  source caching. Keep an integration test set small enough for routine checks.
- Labels at family, method, parameters, protocol and variant levels. Use a curated
  versioned mapping; unknown attributes remain absent. Distinguish generic MFSK
  sources from actual protocol-compliant waveforms.
- Add offset, gain, AWGN, filtering, drift, clock error and scene mixing first;
  add HF fading/multipath and impulsive interference after basic validation.
- A deterministic AWGN child generator now records the SNR convention, transform
  order, seed, gain, clipping and measured SNR. Import requires an exact imported
  clean parent, regenerates the child for verification, and keeps parent lineage;
  deterministic mixed scenes now combine accepted synthetic captures at explicit
  sample offsets/gains, preserve overlapping constituent labels, and verify exact
  rendering on import. The split manifest keeps connected component lineages in one
  partition and reports scenes whose sources are absent from the corpus. Per-component
  carrier offsets, linear frequency drift, smooth component low-pass filtering, and
  sample-clock error up to 5000 ppm are deterministic and verified on import. Source and
  rendered sample bounds are both retained. Scene constituent crops now augment training
  and validation only when all source lineages share that split; held-out scene crops are
  scored separately. A first detector benchmark now reports temporal component recall,
  candidate precision, and false candidates per hour on imported scenes and
  noise-only captures; broader impaired-scene coverage remains open. The baseline
  separately evaluates accepted held-out mixed-scene constituent regions, while only
  same-split scene crops enter training or threshold calibration.
- A resumable command now creates and imports six modulation methods across clean
  and configurable AWGN variants, atomically indexing capture IDs and a shared
  lineage group for future split-safe evaluation. Cancellation UI, quotas and richer
  impairment matrices remain open.
- Store SNR convention/bandwidth and transform order. Choose parameter ranges
  against actual passband observations. Track desired versus measured placement.
- Add payload-only/partial crops, negative examples and mixtures. Preserve RSID
  annotations and supply payload-only training/evaluation exports.

Learning track:

- Review queue with audition/replay, editable region bounds, partial labels,
  evidence view, revision history and bulk approval where evidence permits.
- Freeze split manifests by original source/session before cropping/augmentation.
  Group scene constituents and all reused parent signals to prevent leakage.
- A first corpus split tool validates dataset hashes/labels and groups all clean/derived
  examples under their source capture ID. It requires at least three independent source
  groups per method and writes a seed/versioned split manifest; real session grouping and
  held-out protocol/family experiments remain to be implemented.
- Baseline spectrogram model with masked multi-task losses: predict family first,
  protocol/variant when labeled, and parameters where useful. A protocol label
  must not require an unjustified family/method label.
- Implemented first: deterministic NumPy spectral-occupancy features and a softmax
  modulation-method baseline with separately validation-calibrated family/method
  thresholds, held-out confusion metrics and hashed model export. Deterministic
  noise-only captures now train a separate noise class while their dataset signal
  labels remain empty; lineage-safe validation and test splits report noise detection
  and false positives. Older method-only model artifacts remain loadable. No-threshold
  cases abstain on every method. App inference is opt-in through `SIG_ID_MODEL_PATH`;
  rejection of unseen modulated families and real-radio evaluation remain open.
- Keep protocol heads limited to supported classes; calibrate candidate/rejection
  thresholds on held-out validation data, then freeze before test evaluation.
- Hold entire protocols out and test whether their families are still recognized.
  Test unseen modulation families too: the model must be able to abstain.
- Keep independent real-radio test captures separate. Compare synthetic-only,
  real-only where feasible, and mixed training. Publish class confusion and SNR
  breakdowns, not just pooled accuracy.
- Version model weights, taxonomy, preprocessing and split manifests together.
  Load a model into the app, propose candidates, and support rollback to the
  previous model. An inference failure must leave capture/manual decode working.
- Add a curated Signal Identification Wiki reference importer/catalog after
  checking the terms for each retained media asset. Keep source-reported labels,
  independently reviewed labels, and unknown entries distinct. Record page URL,
  retrieval date, fields/assets sourced, attribution and rights status. Treat these
  as real reference recordings, not synthetic ground truth; use them in training or
  evaluation only when rights and labels permit, with lineage-preserving splits.
- Metadata-only importer implemented against the public MediaWiki API for identified
  and unidentified categories. It records revision/content hashes, source fields, page
  and file-page links, and retrieval time; it does not fetch media or assign accepted
  labels. Per-asset reuse rights and actual reference-audio ingestion remain open.

Acceptance: one documented command creates a small corpus, another trains and
evaluates the baseline, and the exported artifact runs in the application. The
same generated source cannot appear across splits. A held-out protocol produces
a family hypothesis or abstention rather than forced identification as a familiar
protocol. Review approval does not silently overwrite the original prediction.
Curated external examples cannot become trusted labels without an explicit review
record; media with unresolved reuse terms remains link-only reference metadata.

## Milestone 4: full Fldigi coverage

Begin this track after milestone 1; do not wait for the model.

- Freeze a reference Fldigi version/catalog for the release. Maintain tested
  support across mode families and meaningful parameter combinations; document
  ranges for continuously configurable settings rather than claiming exhaustive
  testing of infinitely many combinations.
- Batch catalog tests for selection, input routing, clean fixture decode, reset,
  output retrieval and offline generation where possible. Add independent captures
  for every family and unusual modes; record remaining per-mode evidence gaps.
- Implement specialized output adapters/viewers for image and other receive
  products. Keep unsupported output paths explicit until solved.
- Keep protocol decoders behind a small receive-backend contract (catalog,
  selection, receive products, and capability evidence). Fldigi is the first
  backend; additional decoders can be registered without making its XML-RPC
  catalog or text buffer the application-wide decoder API. A backend only counts
  as supported after its own routing and output tests pass.
- Expose every validated mode through manual selection immediately. Expand
  automatic candidate policies and synthetic generation independently.

Acceptance: no silently omitted receive modes in the reference catalog. Each is
tested end to end or has a specific unresolved issue. Unresolved decoding/output
issues block the claim of full-catalog 1.0; a useful beta can still ship. Lack of
automatic recognition alone does not block manual decoding coverage.

## Milestone 5: unknown-mode analysis

Begin after sample/annotation contracts stabilize. Reuse waveform generation,
replay, filtering, and exports; keep this separate from Fldigi text APIs.

- Add protocol-independent 2-FSK, M-FSK, BPSK and QPSK source generators with exact
  symbol/bit references. Deterministic fixtures and importer/demodulator coverage
  exist for all four; add impairment and rate diversity.
- Implement or integrate demodulators with frequency/timing recovery and explicit
  lock diagnostics. Prefer proven DSP components where practical after a small
  integration benchmark; avoid adding a second large runtime without need.
- Start with manual family/rate/shift controls, then ranked parameter estimation.
- Show tone likelihoods or constellation, timing/eye diagnostics, symbols, bits
  and optional soft metrics. Retain phase/polarity/mapping ambiguities explicitly.
- Export timestamped observations, hard symbols, candidate bits, soft outputs and
  complete recovery recipes. Preserve unknown protocol status.
- Implemented first: manual noncoherent 2-FSK/M-FSK tone-bank and phase-estimated
  BPSK/QPSK analysis with operator-supplied rate, tone/carrier and timing parameters. APIs
  expose raw decisions, soft metrics and absolute sample bounds; M-FSK bits are emitted
  only for power-of-two mappings, BPSK returns both polarity hypotheses and QPSK returns
  four carrier-rotation hypotheses. The UI exports results, settings, capture hash and
  sample bounds as a protocol-unknown JSON artifact. A uniform-grid 2-FSK timing-phase
  search reports its candidate scores and selected phase; automatic symbol-rate
  acquisition, wider fractional timing recovery and differential methods remain open.
- Add complex-IQ file input for analysis using an explicit data type/sample rate
  and SigMF metadata where available; live SDR hardware integration follows later.

Acceptance: report BER/SER against known sources with alignment/ambiguity handling,
lock coverage and full-duration failures. Test wrong hypotheses and independent
captures. Operators can reproduce an export from its saved recipe. No fabricated
bits during lost-lock intervals; gaps are represented explicitly.

## Milestone 6: 1.0 release hardening

- Compose build/start/update instructions, pinned reference backend, persistent
  data volumes and reproducible native capture setup. Test on a clean Linux host.
- Remote capture agent for a radio on another machine, with explicit format,
  timestamps, discontinuities, reconnect and bounded backpressure.
- Review/generation/training job UI, cancellation, resumability, disk quotas,
  export/import validation, migration and backup/restore instructions.
- Recovery from worker crash, input loss, full disk, corrupt input and model load
  failure; isolate these from capture and preserve completed data.
- Eight-hour capture/replay soak, multiple tracks, model inference and background
  jobs on a named reference machine. Measure bounded memory/CPU/disk use.
- Validate public UI binding/authentication defaults for hosted use. Publish
  feature/capability report and known limits with the release.

1.0 requires milestones 1-6: broad Fldigi decoding, automatic candidate operation,
reviewable real/synthetic datasets, usable trained-model pipeline, and the bounded
unknown-mode demodulation scope. It does not promise recognition of every unseen
protocol, recovery below all SNRs, or arbitrary encrypted-payload interpretation.

## Effort and sequencing

Planning ranges below are engineering effort, not delivery promises. They assume
one experienced implementer and working local tooling; hardware availability,
Fldigi integration discoveries and catalog size can move them substantially.

| Increment | Initial effort range |
| --- | --- |
| Integration proof | 1-2 days |
| 0.1 workbench after proof | 3-5 days |
| 0.2 automatic operation | 4-7 days |
| 0.3 synthetic/model loop | 5-10 days |
| Full-catalog expansion | 5-15 days, interleaved after 0.1 |
| Unknown-mode analysis and IQ files | 5-10 days |
| Release hardening | 3-5 days |

Workable 0.1 target: roughly 4-7 focused engineering days if the integration proof
passes. Full feature scope: roughly 5-11 engineering weeks under these assumptions;
re-estimate after the first two days and catalog audit. Training runtime and real
capture collection can overlap development, but independent validation cannot be
replaced by synthetic success. Compress by reusing integration/generation fixtures
and shared data paths, not by dropping verification or hiding mode gaps.

## First implementation queue

1. Create Python/service/UI skeletons, schema versioning, capability registry and
   a concise development README.
2. Complete milestone 0 and write the integration decision/report.
3. Commit a small provenance-tracked fixture set and repeatable decode command.
4. Build file replay, waterfall, manual selection and decoder output together.
5. Add durable annotations, review/replay and local live input; ship 0.1.

Keep this file as the delivery plan. Update `NEXT.md` after each checkpoint with
implemented behavior, validation performed, remaining limits and the next task.
