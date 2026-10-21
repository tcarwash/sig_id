# Signal identification and decoding workbench

Status: architecture and scope are established; release 0.1 is in progress. See
[NEXT.md](NEXT.md) for implemented behavior and remaining validation gates.

Execution order, release gates and estimates are maintained in
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). That plan supersedes the broad
delivery sequence below; current checkpoint is in [NEXT.md](NEXT.md).

## Product

Listen to a radio audio passband, locate individual signals, progressively identify
their modulation and protocol, run appropriate decoders, and collect recordings
with auditable labels for a future detection model.

Initial scope is narrowband HF digital text received as demodulated audio. Audio
cannot recover signals removed by the receiver's filters or demodulation. Wideband
SDR/IQ input is a later input adapter, not something reconstructed from audio.

The decoding coverage target is the full receive-mode catalog of the installed
Fldigi version, including its variants and non-text outputs where integration is
available. The initial text modes are milestones, not the final scope. Manual
decoder selection remains available for modes without automatic identification.

An eventual third capability is protocol-independent demodulation: recover symbols
and candidate bit streams from unknown or undocumented signals without requiring
a known message format. This is separate from identifying and decoding known modes.

## Processing pipeline

    Audio device / WAV replay
      -> timestamped sample blocks + bounded rolling buffer
      -> multiresolution spectrum / noise estimate
      -> time-frequency detections and persistent signal tracks
      -> feature measurements + RSID + optional learned proposals
      -> ranked protocol/parameter hypotheses
      -> bounded decoder trials using buffered audio
      -> evidence evaluation and live decoded output
      -> recordings, annotations, review queue, training exports

The waterfall and decoder workers consume the same sample timeline. Sample indices
are authoritative; wall-clock timestamps supplement them. Discontinuities must be
recorded and must reset affected DSP/decoder state.

## Identification

Store separate fields for modulation family, parameters, protocol, and variant.
For example, MFSK -> incremental tone transitions -> THOR -> THOR16. THOR is not a
4-FSK subtype: its tones, differential behavior, coding, and symbol rate matter.
Some protocols share modulation features, so this is a collection of attributes
and competing hypotheses rather than a rigid decision tree.

Initial features include occupied bandwidth, tone spacing/count estimates,
spectral occupancy, temporal periodicity, frequency drift, and symbol-rate
candidates. Multiple analysis resolutions are needed for slow narrow modes and
short bursts. Measurements carry uncertainty; tone count alone is insufficient
to name a protocol.

Tracks have stable IDs, time spans, frequency bounds, and candidate lists. Retain
unknown, ambiguous, overlapping, and interference states. Merge/split operations
and manual region drawing allow correction of imperfect segmentation.

Suggested track states:

    detected -> candidate -> trial decoding -> identified -> ended

Decoding failure may return a track to candidate/unknown. Use persistence and
hysteresis to avoid switching modes on each analysis window. Bound concurrent
workers, trial duration, and total CPU; prioritize promising tracks.

RSID is an explicit mode announcement when present. Associate it with the nearby
signal in time and frequency; retain it as evidence rather than assigning its
label to the entire recording. Detection must also work without RSID.

Decoder trials may begin before final identification: decoding is itself useful
identification evidence. User-visible text must retain its tentative/confirmed
status and provenance. A rolling buffer lets a newly selected decoder replay the
start of a transmission and then catch up. Buffer duration is configurable and
must account for mode-specific interleaver/startup latency.

## Evidence and labels

Keep classifier scores, decoder quality, and label trust separate. Do not invent
a single universal confidence percentage across incompatible backends.

- Prediction: heuristic/model hypothesis; eligible for review, not gold labels.
- Announced: valid RSID associated with a track; announcement is not payload proof.
- Decode-supported: mode-specific sync/coding/framing evidence recorded.
- Human-reviewed: operator label with author, timestamp, and optional notes.
- Controlled source: known generated fixture, kept distinct from on-air captures.

A CRC is strong evidence where the protocol supplies one, with repeated frames
and framing checks used where appropriate. Text plausibility and FEC output alone
are weak evidence. Some keyboard modes lack an independent payload integrity
check; these may require review or corroboration for trusted training exports.
Evidence policy is per protocol and versioned. A manual label is not infallible;
preserve correction history, disagreement, and rejected prior labels.

The app automatically saves positively identified events with their evidence.
Unknown events can enter a bounded review queue. Retention quotas and recording
controls prevent unattended capture from filling storage. Saving an event does
not automatically admit it to the trusted training set.

## Dataset

Preserve lossless source audio, not just waterfall screenshots. Save references
to source recordings plus precise time/frequency annotations; overlapping signals
can have multiple annotations. Derived signal crops retain links to their source,
filtering/resampling settings, and neighboring interference.

Per event retain:

- Capture/session/event IDs and source hash.
- Sample rate, channel, sample offsets/counts, UTC timestamps, dropped samples.
- Audio frequency bounds and optional dial frequency, sideband, RF mapping.
- Receiver/filter/AGC settings when known; explicitly unknown otherwise.
- Family, parameters, protocol, variant, and unknown values.
- Candidate scores and classifier/version identifiers.
- Evidence, backend/version/settings, decoded output and its sample interval.
- Label source, reviewer, revisions, review status, and export eligibility.
- Synthetic versus recorded origin and any transformation lineage.

Use WAV or FLAC plus JSON annotations, with SQLite for indexed browsing. Treat
spectrogram arrays and preview PNGs as reproducible derivatives. Store STFT window,
FFT/hop sizes, scale, normalization, and software version with each export. Never
train on UI overlays, mode labels, cursors, or colorbar text.

Split datasets by independent capture session/source transmission before making
windows or augmentations. All descendants of the same transmission remain in one
split. Hold out receivers/days where possible and retain noise, speech, carriers,
interference, unknown signals, and mixtures as realistic negatives. Exclude RSID
bursts from payload-only classification evaluation so the model cannot solve the
task by reading the announcement. Preserve genuine timing/bandwidth information
when resizing model inputs.

Begin with small reviewed real captures and controlled synthetic fixtures. Use
uncertainty/disagreement to prioritize later review. Report per-mode precision and
recall, unknown rejection, false detections per hour, identification latency, and
decode success; classifier accuracy alone is insufficient.

The Signal Identification Wiki can seed a curated reference index and source
examples: its guide provides descriptions, sample sounds and waterfall images,
while its database separates entries marked identified from unidentified examples.
These are recorded reference signals, not generated synthetic signals. Store each
entry's URL, retrieval date, field-level source, original asset reference and
rights/attribution status. Preserve source terminology and distinguish
community-reported modulation/protocol from independently verified labels. Missing
or uncertain fields stay unknown. Do not automatically promote a wiki's named
protocol to gold truth, redistribute its media, or use it for training until the
asset's reuse terms are established. Curated audio may later provide an independent
real-recording evaluation set; synthetic impairments derived from it keep lineage
to that recording and remain in its dataset split.

## Synthetic dataset generation

Make a reproducible offline corpus generator part of the workbench. Its pipeline is:

    mode + parameters + known payload + generator version
      -> clean reference waveform
      -> channel/receiver simulation and scene mixing
      -> lossless audio + source annotations + generation manifest
      -> the same replay, detection, decoding, and export path as live captures

Investigate recording Fldigi-generated audio for its supported transmit modes.
Prove generation/capture per mode before advertising support; receive-only or
specialized modes may need another source or recorded fixtures. Synthetic tone
families are useful modulation examples but must not be labeled as a named
protocol unless their waveform, framing, and coding implement that protocol.

Start with clean fixtures and known payloads, then vary frequency placement,
amplitude, duration, message content, timing, frequency error/drift, sample-clock
error, passband filtering, noise, impulsive interference, and clipping/AGC effects.
Add fading and multipath through an appropriate complex-baseband channel stage,
retaining the actual transform order. Channel distributions are versioned and
must be tuned against real captures rather than assumed realistic.

Generate empty passbands, single signals, neighboring and overlapping signals,
partial transmissions, startup-only and payload-only crops, and examples with and
without RSID. Use recorded backgrounds only with known provenance and account
for any existing signals in them. Match nuisance-parameter distributions across
classes so mode labels cannot be inferred from a particular noise level, sample
rate, generator artifact, payload template, or carrier placement.

Store the seed, exact generation recipe, clean parent hash, generator/backend
version, payload or payload artifact reference, all impairments, and all source
time/frequency bounds. Labels describe what was generated, including signals
rendered undecodable by impairments. Keep source presence, observed detectability,
and successful decoding as distinct fields. Severe crops may support only a
partial/ambiguous recognition label despite a known source protocol.

Use the known payload for clean round-trip regression checks. Do not discard all
failed impaired decodes: these are useful for measuring detection/decoding limits.
If generation and decoding use Fldigi, round-trip success establishes internal
consistency; independent recordings or another implementation are still needed
to assess interoperability and synthetic-to-real transfer.

Group all derivatives of a clean source in the same dataset split. Hold out payload
templates and channel conditions, and maintain a separate reviewed on-air test set.
Train with synthetic and reviewed real examples, reporting results separately on
each. The UI should expose generation jobs, mode/variant selection, scene settings,
progress, and corpus manifests through the same dataset browser.

## Interface

One main waterfall with selectable time-frequency regions and track overlays.
Labels progress from Unknown to family/parameters to protocol/variant, displaying
the evidence level. A track panel shows competing candidates, measurements,
decoder state, and decoded text. Operators can audition/replay, force a decoder,
correct bounds, and label a region without losing the original hypothesis.

A dataset review view supports pending/approved/rejected events, partial labels
(e.g. known FSK, unknown protocol), label history, and reproducible exports.

## Unknown-mode demodulation and inspection

Provide an analysis workspace for a selected live track or saved region. The
operator can choose a modulation family and accept or override estimated carrier,
symbol rate, tone count/spacing, timing, polarity, constellation, differential
encoding, and symbol-to-bit mapping. Parameter search produces ranked hypotheses
with measurable lock/quality diagnostics rather than a falsely unique answer.

The processing boundary is explicit:

    source audio (or future IQ)
      -> filtering / frequency translation
      -> carrier and symbol timing recovery
      -> symbol observations / soft metrics
      -> hard symbol decisions
      -> candidate bit mappings and differential interpretations
      -> optional line-code, descrambling, framing, FEC, protocol analysis

Each stage preserves its input and provenance so changed assumptions can be
replayed. Audio supports only information retained by the radio's demodulator and
passband; it cannot stand in for original RF IQ. Phase-based analysis may be
possible for suitable SSB audio, but ordinary FM/AM audio cannot generally recover
the original RF constellation. Add genuine complex-IQ input when needed.

Start with two-level FSK and BPSK, followed by M-FSK and QPSK. Generalize through
demodulator adapters rather than requiring every family to use one synchronizer.
Show spectra, tone likelihoods, constellation where applicable, timing/eye views,
symbol-rate candidates, frequency/timing error, and loss-of-lock intervals.

Export time-indexed symbol observations, hard symbol indices, candidate hard bits,
and soft decisions where supported. Define bit packing, byte order, polarity,
mapping, sample/symbol timestamps, and soft-value convention explicitly. Do not
describe uncalibrated distances as calibrated probabilities or likelihood ratios.
M-FSK tone indices remain meaningful even when a binary mapping is unknown or
not one-to-one. Preserve phase rotations, conjugation, bit-order, and differential
ambiguities as distinct hypotheses until evidence resolves them.

An exported bit stream may still be scrambled, interleaved, encrypted, or contain
FEC and unknown framing. Symbol recovery is not proof that a protocol or payload
has been decoded. Display these states separately from known-mode decode status.
Attach derived streams and their settings to the source dataset event; an operator
can label modulation/parameters while keeping protocol and payload unknown.

Validate with synthetic sources whose exact symbols/bits are known, measuring
symbol/bit error after documenting alignment and ambiguity resolution. Include
offsets, drift, fading, burst edges, and deliberately wrong family/rate hypotheses.
Report lock coverage and rejected intervals alongside errors; do not hide difficult
segments by scoring only the successfully locked portion. Use independent recorded
fixtures to establish practical capture and demodulation performance.

## Implementation recommendation

Python host-neutral DSP/data library with typed decoder adapters; FastAPI service;
TypeScript browser UI with a canvas/WebGL waterfall; SQLite and filesystem assets.
Use isolated processes for native decoder integrations and bounded queues so
decoder stalls cannot interrupt capture. Keep model inference an optional plugin.

Docker Compose fits a hosted service. Capture can run on the radio machine through
a small local agent; browser microphone input is an additional path, not the only
source. File replay is the first deterministic development input. Final desktop
versus hosted packaging remains open.

Fldigi is the first backend candidate for PSK31, RTTY, and THOR. Its XML-RPC API
supports modem selection, carrier control, and received text retrieval; it is not
an audio streaming API. Explicit audio routing, decoder isolation, replay/catch-up,
and evidence quality must be proven in an integration spike. A single instance
does not provide an arbitrary parallel decoder bank. Do not make broad decode or
confirmation claims from its text/quality interface alone.

Discover the installed modem list through the backend and maintain a capability
matrix per mode/variant: selectable, audio routed, text/image/other output handled,
buffered replay validated, synthetic generation validated, automatic recognition
validated, and evidence policy available. A returned modem name alone does not
establish functioning end-to-end support. Some outputs need interfaces beyond
received text; adapter investigation and explicit limitations are required.

## Delivery sequence and acceptance

1. Recording/review foundation: WAV replay and local audio capture, waterfall,
   manual regions/labels, lossless event storage, replay, dataset export. Verify
   sample-aligned annotations, overlap support, provenance, and split isolation.
2. Decoder integration spike: known PSK31, RTTY, and THOR fixtures through an
   actual backend. Verify recovered payloads, mode/variant selection, audio routing,
   startup delay, buffered replay, and failure behavior. Record support per mode.
   Establish offline waveform capture and known-payload generation in this step.
3. Automatic operation: signal tracking, feature hypotheses, RSID association,
   decoder scheduling, evidence policies, live labels/output, bounded retention.
   Evaluate no-RSID examples, wrong-mode trials, silence, interference, overlapping
   signals, and input discontinuities. Publish measured limits.
4. Learning loop: reviewed corpus export, baseline spectrogram model, calibrated
   rejection thresholds, held-out evaluation, optional in-app inference. Predictions
   propose decoder trials; they never silently become verified ground truth.
5. Catalog expansion: systematically validate the installed Fldigi receive catalog,
   including variants and non-text outputs; expand synthetic sources and automatic
   recognition with measured per-mode coverage. Expose manual decoding as each
   adapter is validated, without waiting for its classifier.
6. Unknown-mode analysis: independent demodulator adapters, adjustable recovery
   parameters, symbol/bit/soft-output inspection and exports, tested against known
   synthetic symbol sequences and independent recordings. Begin with 2-FSK/BPSK;
   extend families with explicit input requirements and ambiguity handling.

The first vertical slice targets PSK31, common RTTY settings, and selected THOR
variants. The product target is all Fldigi receive modes. Generic FSK/MFSK/PSK
descriptions remain useful for unidentified protocols; automatic identification,
manual decoding, synthetic generation, and output-format support are displayed
independently.

## Primary references

- Fldigi XML-RPC: https://www.w1hkj.org/FldigiHelp/xmlrpc_control_page.html
- Fldigi THOR: https://www.w1hkj.org/FldigiHelp/thor_page.html
- Fldigi RSID: https://www.w1hkj.org/FldigiHelp/id_configuration_page.html
- SigMF metadata reference for future sample/IQ interchange: https://sigmf.org/
