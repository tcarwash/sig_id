# Distributed collection and peer validation

Status: proposed architecture, not implemented. This is a companion design, not a
change to existing release gates or an assertion of deployed multi-user support.

## Recommendation

Keep collection local and offline-capable. Add an opt-in submission outbox to each
workbench, and a separately deployed shared review hub. Use one authoritative hub
per community initially; do not replicate SQLite databases or implement peer-to-peer
consensus. Distributed collection does not require distributed database ownership.

The contribution unit is a lossless audio excerpt plus context, sample/frequency
annotations, provenance, and evidence. A waterfall image is a review preview, not
the source of truth. Reviewers need not identify a protocol to verify signal bounds.

## Existing foundations and gaps

Observed in the current checkout:

- `sigid/dataset.py`: CaptureStore stores source hashes, sample-indexed regions,
  frequency bounds, source metadata, parent links, and append-only label revisions.
- `sigid/api.py`: saved regions, label-history endpoints, optimistic label revision
  conflicts, replay/demodulation workflows, and JSONL export already exist.
- `DESIGN.md`: lossless originals, separate evidence levels, unknown/partial labels,
  lineage-safe splits, and reproducible waterfall rendering are established policy.
- `OPERATIONS.md`: deployment is a local, loopback-bound workbench with durable local
  storage. This is not the public multi-user service boundary proposed here.

Important limitations:

- Creating or revising a region with a label currently sets its local review state
  to `accepted`. That means an operator entered a label, not independent consensus.
- Region geometry is stored on the region; label revisions do not version geometry.
- Current label rows do not provide first-class authenticated author/reviewer IDs.
- `CaptureStore.export_jsonl` exports current labels without a trusted-review filter.
  Keep it as a general interchange export; add a separately gated training export.
- Remote submission/sync, reviewer assignment, votes, adjudication, accounts, and
  task-specific training eligibility need new contracts and implementation.

## Topology

    Receiver A -> local workbench -> durable submission outbox --+
    Receiver B -> local workbench -> durable submission outbox --+-> HTTPS hub API
    Offline WAV import ----------> manual bundle import --------+       |
                                                                       +-> metadata DB
    Reviewers <------------------ review web UI <-----------------------+-> object store
                                                                       +-> bounded jobs
                                                                       |
                                               versioned dataset releases
                                                                       |
                                               offline training/evaluation
                                                                       |
                                               opt-in model distribution

Collectors initiate outbound connections only. The hub cannot remotely control
radio/PTT, audio devices, shell commands, or the collector's Fldigi instance.
Training and expensive replay jobs must not run on capture-critical workers.

Local nodes retain SQLite and files. A small private pilot can use a single hub
process with a serialized DB writer; PostgreSQL plus S3-compatible object storage
is the recommended shared-service target. Start with a DB-backed job queue and
separate worker process; a broker and Kubernetes are not prerequisites.

## Contribution workflow

1. Record/import locally, select a signal, and optionally supply a family/protocol.
2. Choose "Contribute example" from that region or saved recording.
3. Preview the exact audio and waterfall that will leave the machine. Include
   configurable time/frequency context, not only an already-tight crop. Slow modes
   may need substantial startup/interleaver context; do not impose one short length.
4. Confirm rights, intended audience, and metadata disclosure. Upload is opt-in.
5. Persist an immutable submission snapshot in the local outbox. Resume interrupted
   transfers, retry with backoff, enforce quotas, and retain an explicit failure state.
6. The hub validates and quarantines the package before making it reviewable.
7. Peers review bounds, scene coverage, and optionally identity as separate tasks.
8. The collector can fetch hub status/corrections as remote proposals. Never overwrite
   newer local edits or silently replace a local label with the hub's consensus.
9. A curator freezes eligible examples into an auditable dataset release.

Offer portable bundle export/import before requiring every user to connect a node.
It exercises the same contract and supports disconnected contributors.

## What "that's the whole signal" means

Separate these questions; a single approved checkbox loses useful information:

### A. Signal extent

"Does this annotation cover the visible/audible extent of this signal, without
needlessly including neighbors?"

Record temporal bounds and occupied audio-frequency bounds independently. Provide:

- Bounds look correct.
- Needs adjustment: propose new bounds and a reason.
- Contains multiple signals: propose a split or mark overlap.
- Not a signal / detector false positive.
- Cannot tell: low SNR, overlap, fade, or insufficient context.

Keep two distinct completeness fields:

- Annotation coverage: correct / too narrow / too broad / uncertain.
- Observed transmission: complete / left-censored / right-censored / both / unknown.

A recording can correctly annotate everything it contains while missing the start
of the transmission. Do not reject that useful partial example or claim it contains
the full transmission. A capture-edge contact is a warning, not proof of censoring.
Only call it complete when the supplied context supports that judgment; reviewers
cannot establish that the uploader recorded every earlier or later transmission.

Use configurable, versioned tolerances for geometry agreement. Boundary uncertainty
is normal in fading and drift; IoU can flag disagreements but cannot decide truth.
Keep rectangles initially, with drift/overlap flags. Add piecewise frequency tracks
later if real examples demonstrate that rectangular geometry is inadequate.

### B. Scene coverage

"Within this explicitly defined time/frequency review window, have all relevant
signals been annotated or marked uncertain?"

This is different from verifying one selected signal. A scene needs a coverage
revision containing the reviewed window, taxonomy/scope, complete annotation set,
and ignore regions. An extra annotation invalidates the scene coverage decision.
Unreviewed surroundings must not become automatic negative/background labels.

Allow unknown-signal boxes and uncertain/ignore regions. If a target overlaps an
unidentified neighbor, approving the target does not make the neighbor background.

### C. Identity

"How specifically can we name this signal?"

Review family, method, parameters, protocol, and variant independently. Permit
"FSK, protocol unknown" and defer specialized identity tasks to qualified reviewers.
Keep RSID announcements, decoder output, protocol-native integrity checks, human
judgment, and controlled-generation provenance as separate evidence fields.
A plausible text decode is not automatically an integrity-verified protocol label.

## Review UI and policy

Default task: one target with contextual waterfall, linked spectrum, exact time/Hz
axes, full-excerpt audio and target-focused audio. Any filtering is a reversible,
recorded derivative; retain access to unfiltered context. Zoom and contrast controls
must not alter source coordinates. Offer a larger context view when available.

Primary actions: Looks right / Adjust bounds / Multiple signals / Not a signal /
Unsure. Protocol naming is optional, not a prerequisite for the basic review queue.
Use keyboard shortcuts and show exactly which claims the reviewer is approving.

Suggested pilot policy, to be tuned using measured reviewer agreement:

- Invite reviewers initially; authenticated identities, no self-review.
- Two non-author reviewers agreeing on the same annotation revision can approve
  extent. Review counts are a quality-control rule, not a probability of correctness.
- Do not show other votes before submission. Hide model/protocol suggestions during
  extent review by default to reduce anchoring; expose evidence during identity review.
- Any substantive disagreement goes to adjudication; do not average conflicting boxes.
- A proposed edit is a new candidate revision, not a vote on the old bounds. Approvals
  do not transfer to changed content. The adjudicator may issue an explicit override
  with a reason, distinct from ordinary two-review approval.
- Scene coverage and protocol identity each have their own review decision.
- Randomly audit accepted examples, use calibration tasks, allow abstention and
  appeals, and monitor disagreement by task/family. Do not equate upload volume with
  expertise or make an opaque reputation score the only acceptance gate.

Basic states: draft -> submitted -> checks passed -> awaiting review -> reviewed.
Alternate outcomes: quarantined, needs changes, disputed, rejected, withdrawn.
"Reviewed" is not one universal gold tier: it carries per-claim decisions.

## Data contracts and concurrency

Add these logical entities; names are proposals, not current APIs:

- Contributor and collector: authenticated user, collector ID, scoped upload token.
- Submission: contributor, origin node/local IDs, schema version, idempotency key,
  submitted UTC, audience, rights declaration, and immutable manifest digest.
- Asset: server-verified hash, codec, sample format/rate, channels, frame count,
  byte length, storage key, and validation/quarantine result.
- Excerpt lineage: parent identity, original channel/sample offsets, local-to-excerpt
  coordinate mapping, transformations, and discontinuities. Distinguish a claimed
  parent hash from one independently verified by the hub; no full-parent upload
  should be required just to contribute a small excerpt.
- Annotation revision: capture/asset identity, immutable geometry, partial/censored
  flags, identity fields, author, parent revision, and content digest.
- Evidence artifact: exact source interval, backend/version/settings, result,
  provenance, and hash; distinguish uploaded claims from hub-reproduced results.
- Scene coverage revision: window, scope, annotation-set digest, and ignore masks.
- Review: task, exact revision/digest, reviewer, decision, proposed correction,
  reason, timestamp, and optional superseding review reference.
- Adjudication: selected revision, rationale, actor, and policy version.
- Dataset release: frozen eligible revisions, per-task policy, split groups, rights,
  preprocessing settings, exclusions, and release manifest digest.

The hub derives consensus from immutable review records under a versioned policy;
clients cannot set a trusted boolean. Enforce one active decision per reviewer,
task, and revision. Stale reviews are retained as history but cannot validate new
geometry, labels, source audio, or scene membership. Use optimistic concurrency.

Make upload finalization idempotent using contributor/node/key plus manifest digest.
Reject a reused key with different content. Compute hashes server-side. Deduplicate
exact assets within access boundaries; do not reveal another contributor's private
asset through a global hash-existence API. Hashes establish byte identity, not
truthful provenance, lawful recording, or an honest label.

## Validation and dataset admission

Automatic intake checks: bounded upload sizes/durations, supported lossless formats,
decoder resource limits, actual sample counts/rates, finite and valid coordinates,
manifest/schema version, digests, sample-bound containment, discontinuities, rights,
and referenced lineage. Run media processing in isolated workers; distrust filenames,
archive paths, URLs, and executable metadata. Clipping/dropouts and low SNR are flags,
not blanket rejection rules; those conditions are useful training examples.

Task-specific admission:

- Signal detection: peer-reviewed extent. Train with partial supervision/ignore
  masks unless scene coverage is exhaustive for the configured task and taxonomy.
- Family recognition: accepted relevant family claim plus suitable signal coverage;
  protocol may remain unknown.
- Protocol/variant recognition: separately accepted identity at the requested level.
- Decoder evaluation: exact source bounds/settings plus trustworthy expected output
  or protocol-native integrity evidence; human naming alone supplies no payload truth.
- Negatives: explicitly reviewed background or reviewed rejected proposals; an unknown
  transmission is not noise. Preserve the distinction between false detection and
  wrong identity for a real signal.

Never train merely because a submission was uploaded, locally accepted, or decoded.
Keep recorded, synthetic, augmented, and external-reference data distinguishable.
Export model inputs without UI overlays and with versioned rendering/preprocessing.

Exact-file deduplication is insufficient for train/test isolation. Different nodes
may record the same transmission, and many excerpts may come from one session.
Group parent/child lineage and known shared transmissions; use timestamps/RF metadata
and signal/audio similarity to flag likely related recordings for curator review,
not as automatic identity proof. Build split groups before slicing/augmentation;
hold out receivers, sessions/days, and sources where possible. Disclose unresolved
cross-site duplication rather than claiming perfect independence. Keep an independent
real-recording evaluation set out of model selection and routine suggestion loops.

## Privacy, rights, and operations

Choose private, community-review, or public-release visibility explicitly. Training
permission and public redistribution permission are separate. Audio and decoded text
can contain callsigns, voices, locations, and private content. Strip local paths,
exact location and device identifiers unless explicitly opted in; preview metadata.
For the pilot, use one explicit training license/rights policy rather than inventing
license compatibility rules. Unknown or incompatible rights block training/release.

Provide deletion/withdrawal and retention rules for originals, derivatives, exports,
and reviewer copies. Record provenance and exclude withdrawn content from future
releases; do not promise to recall downloaded datasets or erase its influence from
already distributed models. Disclose that limitation before submission.

Use TLS, scoped/revocable collector tokens, user roles, object authorization, expiring
asset URLs, rate limits, quotas, backups, and queue/resource metrics. Start invite-only
to constrain abuse and data poisoning. Do not expose the local workbench/Fldigi
control API directly to the internet as a substitute for building the hub boundary.

## Delivery slices and acceptance gates

1. Portable contribution bundle and local outbox.
   Export/import a real selected recording with context; verify audio hashes,
   coordinate mapping and provenance round-trip. Exercise offline retry, interrupted
   transfer, duplicate submission, and permission preview. No public hub required.
2. Private hub and extent-review queue.
   Deploy authenticated upload/storage, immutable geometry revisions and basic
   reviewer UI. Demonstrate author plus two other accounts; a deliberate clipped
   box must be correctable, old approvals must not survive the correction, and a
   dispute must block ordinary consensus. Test authorization and cross-user access.
3. Scene coverage, identity review and trusted exports.
   Verify unknown/overlap/ignore handling and separate claim gates. Show that unreviewed
   nearby signals do not become background and disputed labels cannot enter a trusted
   export. Exercise deduplication, lineage splits, withdrawal, and a frozen release.
4. Learning loop.
   Train on a frozen reviewed release, evaluate on held-out independent recordings,
   report false detections/hour, localization/coverage errors, per-family/protocol
   results and abstentions. Distribute a versioned model only after measured benefit;
   model suggestions never silently become human approvals.

Start with slices 1 and 2. Broad public accounts, federated hubs, automated trust
weighting, and remote training on collectors are deliberately outside the pilot.

## Frigate reference and adaptation

Frigate's documented flow separates submitting an example from annotating/verifying
it. Its annotation guidance emphasizes tight boxes and labeling all objects before
verification. Those are useful precedents, not evidence of a public peer-consensus
workflow; independent peer review here is a proposed additional mechanism.

For radio, do not copy image-occlusion policy literally: fading may make physical
signal extent unknowable. Retain uncertainty rather than drawing invented bounds.
The source is time-indexed audio, and a correct single-signal annotation does not
certify that all other signals in the scene were labeled.

References:

- https://docs.frigate.video/integrations/plus/
- https://docs.frigate.video/plus/annotating/
