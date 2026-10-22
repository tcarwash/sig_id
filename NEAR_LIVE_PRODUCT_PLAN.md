# Near-live decoding and correction-driven collection

Status: staged delivery plan. Canonical modes, revision-bound corrections, geometry
history, split/merge lineage, and evidence-backed local mode proposals now have local
implementations; later behaviors remain proposed. This refines near-live priorities;
IMPLEMENTATION_PLAN.md retains the broader catalog, raw-analysis and release gates.
DISTRIBUTED_COLLECTION_PLAN.md retains the distributed-data architecture.

## Product outcome

Listen to a radio audio passband, see distinct signals, progressively identify them,
read attributable decoded output, and correct mistakes with minimal interruption.
Ordinary operation should collect useful examples through actions like:

- Yes, that identification is correct.
- That is a real signal, but the identification is wrong.
- That detection is not a signal.
- The bounds are wrong / this detection contains two signals.
- There is another signal here; its identity is unknown.
- You missed this signal; I know it is BPSK31.

Do not require expert users to fill four independently typed taxonomy fields.
Do not make contribution to a remote hub mandatory for receiving or local review.
A correction is durable local evidence, not immediate global model truth.

Initial scope remains demodulated audio, not arbitrary wideband RF classification.
A mode selection must never imply that information removed by the receiver can be
recovered. Keep IQ as a future source adapter with explicit sample semantics.

## Baseline and concrete blockers

Verified by source inspection, not a new performance benchmark:

- Live capture, waterfall, provisional tracking, manual regions, saved/live-window
  replay, decoder-run persistence, and a baseline classifier already exist.
- `sigid/api.py` uses one replay executor and shared decode-job lock. Live sweeps
  snapshot a window, sequentially try a small shortlist, and do not constitute a
  continuously attached per-track receiver. One expensive replay excludes another.
- The current live audio analysis history is short; the documented ten-second
  window is inadequate for arbitrary delayed review or slow-mode startup history.
- `sigid/tracking.py` associates nearby spectral detections, not protected identities.
  Its center-distance matching needs close-neighbor and crossing/drift evaluation.
- `sigid/decoder_backend.py` provides a minimal text interface, not a sample-stream,
  incremental event, or worker-lifecycle contract.
- `sigid/dataset.py` retains legacy free-form labels alongside the canonical mode
  registry and local evidence-backed proposals. Locally accepted mode entries and
  labels are not peer consensus; the user-defined registry still needs remote review
  and adjudication before treating it as globally shared truth.
- `sigid/web/mode-candidates.mjs`, `sigid/trials.py`, and capability reports contain
  different mode subsets; a browser shortlist must not become a second taxonomy.
- The hub is a pilot: authenticated WAV intake, basic audio review, and simple
  extent consensus. Its audio-hash deduplication also deduplicates submissions,
  preventing a second annotation set for the same contributor/audio from being a
  distinct contribution. It lacks complete revision/adjudication/sync workflows.
- The hub currently exposes existing decisions in its queue; review blinding is not
  implemented. Abstention is currently counted as needs-attention/disagreement,
  rather than a separate non-vote. Both need correction before scaled peer review.
- Synthetic/loopback tests establish useful internal paths, not independent on-air
  recognition accuracy or a reliable general-purpose live classifier.

## Core user experience

### Receiving workspace

Keep one main waterfall, a compact source/recording toolbar, selectable track chips,
a selected-signal inspector, and attributable output tabs. Do not build separate
receive and labeling applications with different coordinates and mode vocabularies.

Each track shows separate states for presence, identification and decoder progress:

    Presence: provisional / persistent / ended / user-added
    Identity: unknown / suggested / announced / decode-supported / operator-confirmed
    Decoder: waiting / acquiring / receiving / no-output / failed / complete

Peer status is an additional badge, not a replacement for the evidence above.
Never call a persistent spectral peak an identified mode. Keep competing identities
and allow abstention. Preserve a signal whose protocol is unknown.

Output belongs to a session + signal track + decoder run + sample interval. Show
mode, carrier, output age, receive lag, and evidence status. Selecting another track
changes the view, not the decoder's mode or the identity of arriving results.
Mark late results on their original track; do not inject them into the current tab.

### Correcting while listening

Freeze the selected historical view, not acquisition. Pin the relevant source
samples immediately, before a short ring buffer can expire. Return to Now easily.

- Confirm: approve an explicit claim, e.g. "Identity: BPSK31". Show the scope; a
  single click must not silently approve identity, bounds and scene completeness.
- Wrong identity: retain real-signal presence, reject the named hypothesis, and
  optionally replace it through canonical mode search. Keep the original proposal.
- Not a signal: reject the detector proposal for this exact interval/geometry.
- Adjust: drag time/frequency edges; append a geometry revision.
- Split: draw child regions for separate signals and supersede the merged proposal.
- Merge: combine fragments with an explicit relation; do not automatically merge
  payload identities or discard conflicting evidence.
- Missed signal: draw a box, type a known mode or leave Unknown, then save. Mark it
  `user_added_missed_detection` and link the detector/model version when applicable.
- Additional signal: add a sibling region; never replace an already-correct signal.

Use visible undo, keyboard access, and short reason choices rather than mandatory
prose. Undo appends a superseding/retraction event, not a destructive history rewrite.
Review can continue after the signal ends or while reception proceeds elsewhere.

## Shared canonical mode registry: first dependency

Build a versioned Python/domain registry consumed by workbench, hub, importers,
synthetic generators, decoder adapters, exports, and browser search via API.
Keep canonical mode identity separate from runtime backend capability.

A mode definition contains:

- Stable canonical ID, display name, curated aliases and registry version.
- Protocol identity and variant where defined.
- Modulation family and method, plus waveform parameters with units.
- Additional waveform facets for composite/multicarrier modes; do not force every
  waveform into an incorrect single inheritance tree.
- Backend-specific identifiers (for example Fldigi's exact catalog name).
- Decoder defaults, startup/history requirements and known parameter constraints.
- Definition provenance and review status. Runtime support/evidence joins separately.

Example after explicitly choosing BPSK31:

    mode_id: bpsk31
    display_name: BPSK31
    family: PSK
    method: BPSK
    protocol: PSK31
    variant: BPSK31
    fldigi backend name: BPSK31

Typing `psk31` should find the PSK31 choices immediately and offer the explicit
BPSK31 choice prominently, alongside QPSK31. Do not silently collapse a generic
PSK31 term into an asserted binary variant. Curate exact aliases such as `bpsk31`
and `BPSK-31`; case/spacing normalization is for lookup, not evidence inference.
Selecting BPSK31 fills the family/method/protocol deterministically and visibly.
Typing only BPSK supports a partial modulation label, not automatic PSK31 identity.
For an ambiguous RTTY label, preserve unknown shift/rate rather than inventing defaults
as observed properties. Clearly distinguish decoder defaults from measured parameters.

Backend availability is a separate join: known mode / installed backend / selectable /
loopback-tested / independently tested / automatic-ID coverage. A taxonomy definition
must remain usable as a label even when no decoder is installed.

### Validation and migration

- Use one domain resolver at every write/import boundary, not browser-only autofill.
- Known mode IDs derive taxonomy fields. Reject contradictory supplied fields with
  actionable field errors; do not silently erase conflicting source data.
- Store typed input separately from resolved canonical identity and resolution source.
- Unknown/custom signals remain first-class, with partial known attributes and notes.
  Unrecognized text can be an unresolved label proposal, not a new trusted class.
- Finite values only; bound strings and array sizes; validate samples, units, frequency
  bounds, Nyquist and explicit null/unknown semantics. Reject inconsistent channel and
  parent/excerpt coordinate mappings. Declared WAV headers alone do not prove intact
  sample data: validate the actual readable payload and report truncation.
- Make registry updates explicit and versioned. Preserve the interpretation originally
  saved and provide a migration/audit view rather than rewriting historical truth.
- Backfill existing labels with exact mappings first. Preserve ambiguous/conflicting
  rows for review; preview counts and keep a backup before applying migration.
- Share language-independent schema fixtures across Python and browser tests so all
  ingestion paths agree. Report rejected/quarantined imports rather than dropping rows.

## Revisioned evidence and collection model

Introduce the following logical contracts before wiring quick-action buttons:

1. Observation/proposal: detector/classifier/RSID/decoder hypothesis, exact input
   interval, model/backend/version/settings, scores and evidence artifacts.
2. Signal annotation: stable signal ID with immutable revisioned geometry and label
   claims, optional user-added origin and split/merge relations.
3. Feedback event: actor, action, target claim/proposal and annotation revision,
   timestamp, replacement claim/geometry, reason, and supersedes/retraction relation.
4. Scene coverage revision: explicit time/frequency window, task/taxonomy scope,
   annotation-set digest and ignore/uncertain regions.
5. Review/adjudication: independent decisions bound to exact content/revision hashes.
6. Training admission: computed per task, under a versioned policy, not a user-set flag.

Keep model score, protocol integrity evidence, operator confidence and peer consensus
separate. Automatic metadata derivation makes a claim internally consistent; it does
not establish that the source signal actually used that mode.

Do not treat "wrong BPSK31" as a noise example: it may be a valid RTTY signal. Do not
treat an unknown neighboring transmission as background. A missed signal is a detector
false negative only when it is inside the evaluated scope and not already represented
by an equivalent track. Review scope and overlap matching must be explicit.

Detector training requires verified regions plus reviewed background/ignore policy.
Family recognition can use family-only labels. Protocol training requires accepted
identity at that level. Decoder evaluation requires expected payload or appropriate
integrity evidence, not just a human mode name. Partial transmissions remain useful
with censorship and uncertainty flags; do not fabricate unseen boundaries.

## Near-live engine

    Source blocks + discontinuities + absolute samples
        -> durable chunk spool + small RAM ring
        -> independent spectrum/detection/feature pipeline
        -> stable track registry + mode candidates
        -> bounded admission scheduler
        -> isolated receive workers
        -> incremental attributed output + evidence events
        -> corrections / pinned excerpts / local outbox

### Source ownership and retention

Use one authoritative sample timeline. Keep sample rate, channels, UTC mapping,
source session, discontinuities and receiver settings when known. Separate viewing,
monitoring and recording controls explicitly. Permit ephemeral receive-only operation
with a bounded spool; contribution requires retaining/pinning actual audio first.

Retain a configurable disk-backed recent-history spool, with a small bounded RAM
ring for analysis. Pin selected/review/decoder intervals before retention eviction.
Set mode-aware pre-roll and post-roll/history; do not just enlarge the ten-second
RAM ring without measuring resource costs. Make disk quotas, pinned-byte budgets,
eviction and "insufficient history" visible. Recording faults cannot silently create
continuous-looking sample timelines.

### Track-aware decoding

Extend the backend abstraction to support open/configure/feed/flush/stop plus
incremental text or typed receive-product events, health and cancellation.
Carry run ID, track revision, sample provenance and worker-generation tokens.
Do not assume Fldigi can directly implement this in-memory stream contract: its
adapter may own a private virtual audio route and external process.

Use persistent warmed receive workers where measured beneficial, isolated profiles,
receive-only/PTT-disabled controls and per-track carrier/settings. Separate worker
pools/budgets for live acquisition, established receives, and archive/replay. Bound
concurrency by measured CPU/RAM/audio-route cost, not the number of detected peaks.
Start with a small proved concurrency configuration and expand only after isolation tests.

Manual mode correction/pinned reception outranks speculative auto trials. Use
fairness/aging, backpressure, cancellation, and bounded admission retry. A transient
busy worker must not permanently mark a track as tried. Re-run only on meaningful new
samples/evidence/settings; deduplicate unchanged work. Surface deferred and rejected jobs.

Avoid an exhaustive multi-mode sweep as the live strategy. Use measured spectral and
symbol features, RSID when present, model proposals and operator knowledge to select
a bounded candidate set. Do not require an accepted model prediction for every trial:
unknowns and manually identified signals need a path that bypasses model abstention.
Do not stop merely because the first decoder emits printable bytes; retain tentative
outputs and use protocol-specific evidence to decide when to stop or keep alternatives.

Important feasibility gate: a decoder consuming only real-time audio cannot replay
old audio and catch up to a continuing stream at the same rate. Measure replay speed
and startup cost. If faster-than-real-time catch-up is unavailable, choose explicitly
between bounded-lag decoding, joining live with mode-specific late-entry limits, or
separate historical replay plus current receive. Do not promise zero-lag catch-up.
Preserve output segment provenance and avoid duplicated text at handoffs.

Track channelization/filtering must preserve needed bandwidth, resampling phase and
frequency mapping. Evaluate tone-bank center rather than equating rectangle midpoint
with a proven carrier. Allow manual correction and store requested/derived settings.
Never join adjacent signals solely because their centers are close.

### Initial service objectives (targets, not measured claims)

- Keep the existing responsive waterfall independent of decoder work; instrument
  capture-to-display latency and candidate-update age, reporting median and p95.
- Target new ordinary signal proposals within two seconds of sufficient evidence;
  explicitly document slower/weak-signal cases and measure onset-to-detection separately.
- Target UI feedback acknowledgment within half a second when local storage is healthy.
- Target incremental output delivery within one second of backend emission. Report
  acquisition/interleaver time, queue wait and replay lag separately by mode.
- No decoder-caused sample loss in the agreed multi-signal hardware soak; make actual
  source drops visible. Log dropped/deferred work instead of hiding overload.

Do not turn these into release claims until measured on named hardware, sources,
mode profiles and concurrency. A slow interleaved mode cannot meet a universal first-text
latency inferred from a fast keyboard mode.

## Hub evolution and clean dataset releases

Keep the separate Compose hub. The collector remains autonomous and offline-capable.
Add a durable outbox with immutable manifest snapshots, explicit rights/audience
preview, retries/backoff, bounded uploads, receipts, withdrawal and status sync.
Separate asset deduplication from submission identity: same audio with a new annotation
revision must not be discarded as a duplicate. Use manifest-bound idempotency keys.

Reuse the workbench waterfall/annotation components for contextual hub review. Add
claim-specific tasks for extent, identity and scene coverage, blind review, abstention,
new correction revisions, adjudication and appeal. Revisions invalidate only affected
approvals. Peer corrections arrive locally as proposals, never silent overwrites.

Before wider deployment, implement schema migrations, bounded/paginated queues,
strict credential parsing and stable account IDs, revocation, role/project access,
request limits (including multipart parser limits), finite numeric validation,
streamed intake, actual WAV integrity checks, immutable audit events, tested backup/
restore and retention. Avoid loading every large asset into a review page at once.
Configure TLS at the deployment boundary; do not expose workbench/Fldigi control APIs.

Publish frozen releases containing exact asset/annotation revisions, task-specific
admission policy, rights, preprocessing and split manifests. Deduplicate related
excerpts/augmentations and known shared transmissions across receivers before splits.
Hold out independent sessions/receivers and a real-recording evaluation set. Report
synthetic and real results separately. Corrected examples inform the next curated
release; do not update the running model after every click.

## Delivery sequence and acceptance gates

### A. Canonical identity and data contracts

Touch: new domain registry/schema/resolver, `dataset.py`, API request contracts,
backend mappings, browser label search, hub intake and imports.

Gate: selecting BPSK31 from a `psk31` search fills PSK/BPSK/PSK31/BPSK31 consistently
in every writer. QPSK31 remains separately selectable. Contradictory labels cannot be
saved via direct HTTP or import. Unknown/partial labels remain valid. Existing data
migration reports ambiguity without guessing. Capability reports are not duplicated.

### B. Correction-driven local collection

Touch: revisioned annotations/feedback storage, API, waterfall overlays/inspector,
source pinning and output attribution.

Gate: scripted browser workflow detects two signals, confirms one, changes the other's
identity, rejects a false proposal, adds a missed BPSK31 region, adjusts bounds and
splits a merged proposal. History survives restart; undo/stale edits work; acquisition
continues while the view is frozen. Each action targets its exact claim/version.
Exported feedback retains model/proposal identity without labeling unknowns as noise.

### C. Reliable history and jobs

Touch: live source/spool, retention/pinning, persisted jobs, worker supervisor and
backend contract. Benchmark on a declared reference machine before changing budgets.

Gate: retained past samples can be selected after leaving the RAM ring; quota eviction
cannot delete pinned input; restart/cancellation/source-switch tests cannot attach
stale output to a new track. Overload defers work without blocking capture. Demonstrate
what lag/catch-up the Fldigi adapter actually supports.

### D. Useful near-live receiver

Touch: per-track carrier/filter path, persistent workers, candidate scheduling,
track association, incremental result API/UI, health/latency instrumentation.

Gate: controlled real-time playback of multiple independently known signals produces
attributed incremental text with measured latency and bounded CPU/memory. Start with
BPSK31, RTTY and one slower interleaved mode, then expand through capability evidence.
Test close neighbors, crossings/drift, overlap, startup without headers, weak/fading
signals, noise/speech, worker failure, mode correction and recording stop/start. Follow
with independent on-air recordings; a same-backend loopback is insufficient.

### E. Integrated community validation

Touch: contribution preview/outbox/sync and hub revisioned contextual review,
authorization, corrected consensus policy and operational Compose configuration.

Gate: a locally added missed signal uploads with its source mapping, receives two
independent extent reviews, disputes identity without discarding extent evidence,
and returns as a visible status/proposal. Offline retry is idempotent; changed
annotation manifests are not lost to audio deduplication. Corrections invalidate old
approvals, abstention is not disagreement, and unreviewed scenes stay partial.

### F. Trusted learning loop and wider coverage

Touch: task-gated releases, training/evaluation, model provenance/rollback and
progressively broader decoder/output adapters.

Gate: a frozen release excludes disputed/incompatible-rights examples and prevents
lineage leakage; evaluate false detections/hour, missed-signal recall, localization,
identity accuracy/abstention, per-mode decode outcomes and latency on held-out real
captures. Demonstrate a model benefit before making automatic trials a default.
Retain the full-catalog/manual-decoder and specialized-product goals without
claiming universal automatic identification. Add feature-specific coverage dashboards.

## First implementation slice

Implemented: a versioned canonical registry for the initial loopback-validated modes,
`/api/modes` search, BPSK31 taxonomy derivation, contradictory-label rejection at the
API and local-store boundaries, canonical mode IDs/registry evidence in append-only
label history, and `user_added_missed_detection` region provenance. The waterfall
label UI can search aliases and offers an explicit "I added this signal because the
detector missed it" marker. Saved detector proposals now offer Yes, Wrong identity,
and Not a signal actions. Feedback is an append-only event bound to the exact label and
geometry revisions; each action appends a new label revision, corrected mode taxonomy
is resolved canonically, and stale actions conflict. Existing labels are preserved through schema
16 migrations; unmapped legacy labels remain as-is. Geometry has its own revision
history; users can drag corrected bounds on the waterfall, save them with optimistic
concurrency, inspect prior bounds, and discard a draft. Drawn child regions can replace
a merged proposal through an explicit split; selected active regions can be merged into
a new unknown observation while preserving their original labels. Superseded parents
remain in history and are excluded from subsequent lineage operations. Browser smoke,
API, dataset and resolver tests exercise these slices.

Scene coverage now has a local append-only API contract for exact sample/frequency
windows, task and taxonomy scope, complete/partial review state, and bounded ignore and
uncertain masks. Every saved revision snapshots the digest of active overlapping label
and geometry revisions; later annotation or lineage changes make the coverage visibly
stale rather than silently reusing an obsolete completeness claim. UI scene-review
controls and hub synchronization remain open.

Users can propose a new mode with a canonical ID, taxonomy, aliases, a written
identification rationale, distinguishing evidence, and exact saved-region citations
from at least two different audio hashes. The citations snapshot capture hashes,
sample/frequency bounds, and geometry revisions. Proposals remain outside mode search
until a rationale-bearing local review accepts them. Accepted local definitions fill
taxonomy through the same resolver and persist with a definition-specific version.
Local acceptance is explicitly not peer verification; the queue has no account
separation or hub consensus yet.

Next add waterfall scene-review controls and expose/reconcile scene coverage,
geometry/identity/lineage events, and mode-definition proposals in the hub, then tackle
retained source history and near-live worker scheduling. Grow the canonical registry
deliberately. Prototype streaming Fldigi lifecycle/catch-up feasibility only as an
isolated benchmark; do not rewrite the scheduler around unmeasured backend assumptions.

Do not begin by training a larger model or polishing the hub's deployment alone.
The critical path is canonical identity -> correctable observations -> retained source
history -> attributable live decoding -> independently reviewed datasets.
