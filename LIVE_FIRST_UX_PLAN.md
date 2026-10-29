# Live-first UX and collect/review plan

Status: first implementation slice is in the worktree; static checks passed; browser interaction remains unverified. Initial findings were based on source inspection, including local uncommitted changes.

## Initial implementation checkpoint

The full roadmap below remains proposed; the first implementation slice is now present:

- Annotate has Save progress without closing, Complete & next within a review session, and Defer with Save/Discard/Stay protection. Selection survives Classify/Annotate switches; saving an existing region restores its selected context.
- Failed coverage after a successful annotation save reports completion pending and retries without duplicating annotations. Completion checks overlapping saved regions, including those crossing the window edge.
- Receive attempts are visible in Simple view. Collect example queues a pending candidate from a specific finished exact-source decoder run; live collection rejects expired audio. Saved source audio is hard-linked when possible, with copy fallback and independent safe deletion.
- Collected review exposes the original attempt, settings, and text. Use suggested mode edits the review draft only; it does not accept a label or record complete coverage. The queue opens the attempt's target 10-second window, not the start of an unrelated long source.
- A queued durable clip can replace the main waterfall while live monitoring continues. Return to live restores the previous session, selection and inspector context; the banner explicitly warns that the short analysis ring does not pin the selection's audio. The Stop control remains available during clip review.
- Output ownership covers source, sample/frequency selection, focused attempt, and output view. Late Fldigi/native results, background trials, and shared receiver text cannot replace/contaminate the selected isolated output.
- Existing capture/live per-source decoder-run APIs retain input and output provenance. The schema also records `auto_identify`; legacy rows remain unknown/null rather than being mislabeled as manual. The migration was checked using a temporary backup of an existing schema-21 database.
- Classify's top-level Decode and Raw symbols tabs are now one Investigate tab with local Decoder/Raw symbols tools. The Fldigi carrier setting is behind a local Parameters disclosure, and users can compare two or three persisted decoder attempts side by side without changing selection. Comparison includes source, sample interval/rate, carrier, band when known, backend/scope/association, attempted modes, note and output.
- Raw-symbol runs now persist settings, exact sample range, and a bounded result preview. The capture history restores attempts when reopened; long vectors are preview-trimmed in database history while full analysis exports remain unchanged.

Focused API and Chromium regression tests cover these paths, including a real API collection-to-review journey and desktop/narrow screenshots. See `tests/GUI.md` for commands. No hardware decoder or RF interoperability claim is added.

Still open: durable live pinning/retention. The global Simple/Advanced switch has been replaced by panel-local “More…” disclosures; existing disclosures and live statuses remain visible. Raw-symbol history now supports selecting up to three attempts for settings/result comparison and retrying one with its saved settings and exact sample interval. These UI/API paths have syntax/schema checks but still need browser/API execution. Returning to live restores the previous displayed sample context but does not guarantee that audio remains in the short ring; decoding/re-collecting an expired interval still fails. Defer is session-local and leaves the durable item pending; persisted deferral reasons and a full review-session summary remain future work.

## Signal encyclopedia and visual comparison

The user would like a browsable signal reference organized by taxonomy. Each encyclopedia card has a grey background; the standardized signal render itself is glowing yellow on a transparent background. The guide is a non-modal floating window that can be dragged beside a signal or docked on the left, so the waterfall remains interactive. Cards include mode information and support filtering and comparing examples. Keep this reference surface separate from live annotations and model-training labels.

- The first slice includes every Fldigi mode in the local audited mode catalog and 13 curated, commonly encountered signals from HF, aviation, maritime, amateur, satellite, and land-mobile monitoring. Family filters and text search expose taxonomy and visual clues.
- Original schematic SVGs show illustrative modulation patterns; these are not recordings or copied SigIDWiki images. Cards link to SigIDWiki where a reference page is known.
- Browse by family and search names, method, protocol, aliases, and identification clues; keep incomplete and unknown taxonomy discoverable.
- Render the signal as a transparent-background yellow layer over the grey card. Use deterministic rendering and fixed axes/scales so visual comparisons are meaningful; show render settings and distinguish canonical synthetic/fixture examples from real recordings.
- Include concise mode notes: modulation traits, bandwidth/rate where known, protocol relationship, decoder/catalog support, and evidence/source provenance. Qualify uncertain claims visibly.
- Allow multiple cards to be selected and compared side by side, aligning common time/frequency scales where possible.
- Prefer reviewed canonical fixtures and curated examples. Do not silently promote user recordings or tentative detector output into reference material.

Acceptance: browse taxonomy categories, open a mode card with reproducible render metadata, compare related modes on aligned scales, and trace each displayed example to its source and review state. Remaining work: verify the floating/docked browser interaction and improve the schematic set with per-mode fixtures and render metadata.

## Product direction

Make the live no-full-recording waterfall the primary touchpoint. The application finds signals, displays the best supported identity, and attempts useful decoding within explicit resource limits. Manual investigation remains available when automation cannot identify a signal.

Preserve **collect → review** as the annotation workflow. Collecting an interesting example must not require interrupting live operation or completing dataset review immediately. Review is a deliberate dataset-quality task, not a side effect of selecting a decoder.

Classify is an exploratory workbench, not just a classifier button. Consider the visible name **Investigate**, with Classify retained initially if changing vocabulary would add friction.

## What exists and where friction remains

- Scan-only operation, live detection/classification, isolated live Fldigi window decoding, and saving a live window to the review queue already exist. This is an incremental redesign, not a new acquisition system.
- The source toolbar initially offers recording, and the empty state asks for a recording (`sigid/web/index.html:16–49`). Neither communicates live-first operation.
- The former global Simple/Advanced switch added another navigation layer for a selection. This slice replaces it with local disclosures; mode, inspector, and output navigation still need usability review (`sigid/web/index.html`).
- Annotate hides the stay-open Save action; completion is restricted to queued captures (`sigid/web/app.js:2091–2115,5022–5023`). Keep queue-based review, but remove needless closing/reopening between queued examples.
- Changing to Annotate starts a new region (`sigid/web/app.js:6342–6354`). Entering review should instead restore the collected window and its draft context.
- Live collection carries label fields and notes but not a selected decoder-run association (`sigid/web/app.js:4982–5014`). Successful investigations need a provenance-preserving collection action.
- Decoder completion can replace the output currently being viewed without checking selected track/run (`sigid/web/app.js:866–918`). Results must belong to their source signal, not whichever panel happens to be open.
- Visible waterfall history exceeds the actionable audio ring. Freezing the picture does not preserve its audio. Retention and pinning are prerequisites for a dependable live workbench, as already described in `NEAR_LIVE_PRODUCT_PLAN.md:206–217`.
- Completion saves annotations and coverage in separate steps. The first completion branch can report unsaved drafts after annotation saving succeeded (`sigid/web/app.js:5082–5093`). Review recovery needs truthful, independently retryable states.

## 1. Shared shell: live first, context preserved

Primary toolbar:

    Source | Start monitoring / Stop | Freeze view | Collect example
                                      Review queue (N) | Library | Settings

- Default new sessions to scan-only monitoring. Do not automatically activate an audio device on page load.
- Label the state explicitly: Monitoring · no full recording. Put full-session recording and unattended collection behind an intentional capture-policy control.
- Preserve existing input, notch, recording, and report functionality; move infrequent actions rather than remove them.
- Selecting a signal opens one persistent signal inspector. Its header always shows source, signal ID, exact selected interval, retention availability, and evidence status.
- Switching tool, inspecting a result, or opening review must not silently clear selection, reset parameters, stop reception, or change decoder mode.
- Reviewing a collected clip replaces the main canvas with that stable clip, while a compact live status/return control shows monitoring is still active. Do not require two full waterfalls. Return to live restores the previous live selection or explains that its track ended.
- Freeze view means only display freeze. Preserve selection is a separate retention operation with explicit state. Never imply frozen pixels guarantee retained audio.

## 2. Collect: fast, non-disruptive, and reversible

Primary journey:

    Live signal or investigation result → Collect example → continue monitoring

- One action preserves the required source audio and creates a pending review item. Show a small confirmation with Undo collection and Review now; do not automatically open review or switch modes.
- Collection does not require a known identity or a successful decode.
- Preserve selected geometry, surrounding review context, source/sample mapping, detector/model proposals, selected run IDs, decoder parameters, typed results, and collection reason. Keep suggestions separate from human-reviewed labels.
- Optional quick note and tentative identity are available without a mandatory form. Do not force re-entry of the decoder mode used by the selected result.
- Existing saved recordings enter the same review queue by reference; avoid duplicating audio unnecessarily.
- Use a stable collection identity/idempotency key so retries do not create duplicate queue entries. For intentional recollection of an existing interval, offer opening the existing item or explicitly creating a separate review task.
- Start with existing 10-second review windows. A longer investigation may retain a larger source excerpt and enqueue linked review windows; do not truncate decoder evidence or imply a 10-second clip always contains enough audio to decode a mode.
- Collection is durable; an investigation pin is temporary. Explain this distinction and quota behavior. Undo must not delete shared source audio or another user's/reviewer's work.

Collection item states:

    Collected → In review → Reviewed complete
                         → Saved partial / deferred

Saving a clip, accepting a region label, and completing window coverage remain different operations. Reviewed does not automatically mean eligible for every training task.

## 3. Review: one continuous annotation session

Primary journey:

    Review queue → next collected window → draw/refine/label → Complete & next

Layout:

- Waterfall with every active box in the window, regardless of origin.
- Compact box list: label, Needs review / Unknown / Reviewed, and provenance badge.
- Selected-box editor: known-mode search, explicit Unknown, optional partial label, notes, and direct bounds controls.
- Sticky footer: Save progress | Complete & next | Defer. Previous/Next item navigation and progress remain visible.

Behavior:

- Keep persistent drawing, direct resize/move/redraw, merge preview, identity linking, undo, and window-level transactional annotation saves from `ANNOTATION_WORKFLOW_PLAN.md`.
- Save progress persists annotations and keeps the current window open; it does not assert complete coverage.
- Complete & next saves changes, explicitly records the reviewed extent/task, then loads the next queued item without reopening the queue drawer. At the end, show a short session summary with Return to live.
- Defer offers a brief reason and keeps the item incomplete. Resolve dirty drafts with Save progress / Discard / Stay; do not silently lose changes.
- Display remaining unresolved boxes and the exact completeness claim. Unknown identity is valid for confirmed signal presence; an untouched suggestion is not equivalent to reviewed Unknown.
- Expose Mark uncertain area / Ignore artifact locally when needed, not through a global Advanced mode or separate coverage workflow.
- Crossing-window signals retain their true bounds and identity. Validate all overlapping in-scope regions, show clipped boundaries, and apply coverage only to the window actually reviewed.
- A reviewed empty window becomes a detection negative only after explicit completeness confirmation. Dismissed proposals, deleted boxes, deferred items, and missing detections do not independently imply noise.
- If annotations save but completion fails, show Annotations saved · completion pending and retry only completion. Conflicts preserve the draft and identify changed regions.
- An Investigate this signal action opens the same workbench over the review clip without discarding the draft. Return to review restores selection, geometry, and progress; importing an investigation result is explicit.

Queue organization:

- Default to actionable pending items with next-item navigation.
- Lightweight filters: collected manually, collected automatically, incomplete, due for re-review; keep local/Hub provenance explicit.
- Show a small preview, collection reason, tentative identity, available evidence, and review status per item.
- Avoid making model training controls, PDF reports, registry proposals, or history part of the normal review path; retain them as contextual secondary actions.

## 4. Classify / Investigate: one signal, many attempts

Primary journey:

    Select signal → inspect suggestions → try decoder or modulation → compare → collect if useful

Inspector:

1. Sticky signal summary: bounds, source, measurements, best supported identity, alternatives.
2. Candidate suggestions with reason and capability status. Try this runs an explicit bounded attempt; choosing a suggestion alone neither labels nor runs a decoder.
3. A single experiment area with Decoder / Raw demodulation tool selection and local parameter disclosure. Preserve separate settings per tool and signal.
4. Attempt history with status, settings, result preview, evidence quality, and Compare / Retry with changes / Collect example.

- Replace global Simple/Advanced with local disclosure: Parameters, Evidence, History. Essential state and errors are never hidden as advanced details.
- Unify native and Fldigi clip decoders in the conceptual mode picker, while showing backend availability and retaining backend-specific controls where necessary.
- Keep control of a shared external receiver explicitly separate from isolated clip experiments. Do not require selecting the external receiver mode before an isolated attempt that can take its own mode parameter.
- Attach output to selected signal and attempt. Late results create a badge on their originating signal; never steal the output pane from another selection.
- Keep text, image, bits, and audio as result representations, not additional top-level workflows. Let the output dock expand for comparison while the waterfall and source context remain visible.
- Changing frequency bounds creates a new analysis input revision; old attempts remain available but are marked as based on earlier bounds.
- Do not overwrite earlier attempts when changing rate, carrier, polarity, alignment, or decoder. Preserve enough provenance to reproduce a useful result.
- Raw bits, readable text, model scores, RSID announcements, and protocol-validated output retain distinct evidence meanings. Never turn printable output into a confirmed identity automatically.
- Collect example from a specific result imports that result's mode/settings/evidence, not the current dropdown or another attempt's text. It queues review; it does not silently create a trusted annotation.

## 5. Live automation: useful initiative with visible limits

- Continue detection and identity proposals while the user investigates or reviews.
- Show compact on-waterfall states: Unknown, candidate identity, queued, trying, output available; detailed reasons appear on selection.
- Offer bounded automatic decoder trials as an explicit opt-in with understandable resource and backend requirements. Retain a manual path for unknown signals and model abstentions.
- Manual/pinned work has priority over speculation. Surface Waiting for worker, Insufficient history, Backend unavailable, and No usable result rather than generic disabled buttons.
- Automatic trials must not relabel dataset examples. Automatic collection remains a separate opt-in policy with deduplication, storage budgets, and queue limits.
- Do not promote the first readable output to confirmed identity. Compare supported candidates and apply protocol-specific evidence policies.
- Stable track identity, isolated workers, cancellation, stale-result protection, and resource budgets precede broader continuous multi-signal decoding.

## 6. Implementation sequence and acceptance gates

### Phase 1 — Remove navigation friction in current capabilities

- Persistent signal context; prevent mode switches from resetting selection.
- Review queue → continuous review session with Save progress / Complete & next / Defer.
- Truthful save/completion recovery and overlapping-window review validation.
- Local disclosures instead of global Simple/Advanced are implemented in the current slice; continue auditing secondary actions for clear context.
- Result routing scoped to signal/attempt; inactive results cannot steal focus.

Acceptance: annotate a multi-box collected window, save without closing, complete and advance without revisiting the queue, investigate a box and return without losing drafts. Inject a coverage failure after save and verify no duplicate annotations and accurate retry messaging.

### Phase 2 — Make investigation and collection one continuous flow

- Shared experiment/attempt model and UI for existing decoder/raw capabilities.
- Explicit Collect example on a signal or selected result with evidence/run association.
- Queue items restore exact source, bounds, suggestions, and collected evidence.
- Saved recording collections use existing source references.

Acceptance: run two different attempts on one clip, compare their results, collect from the earlier attempt, and verify the queue item references that attempt's exact settings and source. Continue monitoring throughout collection and review.

### Phase 3 — Make no-full-recording live work reliable

- Bounded source retention/pinning with quota and eviction semantics.
- Shared retained-source abstraction for live playback, raw analysis, native decoders, and collection; adapters still declare supported input lengths/capabilities.
- Live-first entry state and scan-only default; explicit full-recording policy.
- Expiry, stop, reconnect, and application restart behavior documented and exercised. Temporary work must not be presented as durable.

Acceptance: preserve a live interval, let the ordinary ring expire, then replay/analyze/collect it with exact sample provenance. Test insufficient history, quota exhaustion, source discontinuities, and explicit deletion/expiry. Unpinned history outside retention fails clearly, never with substituted audio.

Phase 3 is a prerequisite for claiming full live investigation parity; Phase 2 can initially ship using saved clips and currently available live capabilities.

### Phase 4 — Expand automatic assistance

- Track-aware admission, isolated worker lifecycle, cancellation, bounded retries, and visible queue status.
- Automatic suggestions and bounded trials across supported backends with explicit opt-in.
- Evaluate longer-lived reception/catch-up separately; do not promise real-time catch-up without backend measurements.

Acceptance: competing tracks and manual work obey budgets and priorities; worker failures do not stop monitoring; stale results remain on their original source; automation never accepts dataset labels implicitly.

## Validation and measurement

Use isolated synthetic test data, existing API/unit tests, and browser workflow tests; do not operate the user's radio or dataset for UI validation. Extend `tests/ui_smoke.mjs` and relevant annotation, geometry, coverage, replay, raw-result, and live tests. Inspect rendered desktop/narrow layouts and keyboard focus order.

Product acceptance targets (not measured current performance):

- Collect a currently available live example with one primary action and no forced workflow switch.
- Complete each subsequent queued window without reopening the queue.
- No mandatory tab change between editing selected-box geometry and identity.
- No mandatory label re-entry when collecting a selected decoder attempt.
- No selection/draft loss when returning from investigation to review.
- No automatic output focus theft from late jobs.

Measure baseline and revised flows: required navigation transitions, repeated data entry, time to collect, time per reviewed window, deferred-item rate, and recoverable save errors. Do not optimize completion speed by weakening coverage/evidence semantics.

## Relationship to existing plans

This is a proposed UX refinement of `GUI_OVERHAUL_PLAN.md`, `ANNOTATION_WORKFLOW_PLAN.md`, and `NEAR_LIVE_PRODUCT_PLAN.md`, not a claim that their remaining backend work is implemented. Preserve window-level drafts, revision/provenance safeguards, native backend capability distinctions, and existing uncommitted report/notes/notch work. No application implementation changes are included in this planning artifact.
