# Annotation workflow simplification

Status: source implementation present; final persistence/browser validation pending. Updated 2026-10-24.

This records the agreed direction from the annotation workflow discussion,
including the clarification that users should draw and label several boxes before
saving. It supersedes per-signal saving as the primary annotation workflow.

## Primary workflow

The editing and review unit is a **10-second window**, not the entire recording.

1. **Draw boxes around everything.** Drawing mode stays active after each box so
   a blank waterfall can quickly become a collection of draft signal boxes.
2. **Label each box.** Switch to selection, click a box, and choose its known mode.
   Tab / Shift+Tab moves between boxes when not entering text. Show labels on the
   waterfall. Human labels can use the full inventory, even without classifier or
   decoder support. Classifier suggestions are optional.
3. **Refine freely.** Resize, move, redraw, delete, and merge intersecting boxes
   without losing labels or leaving the editor.
4. **Save annotations.** Save all window changes together, rather than requiring
   a separate save for each box, geometry adjustment, label, or merge.

The same workflow supports correcting model-generated proposals:
select → adjust bounds → merge intersections → label → save annotations.

## Shared signal identity

Boxes can represent separate regions that belong to the same signal identity.
Identity linking keeps their geometry separate from merging while assigning one
stable signal ID. Operators can link boxes by entering an existing signal ID or
Shift-clicking multiple boxes and linking them. Updating the classification on
any linked box updates the shared signal identity and the other linked boxes;
region geometry, notes, and evidence remain per box. Linking never combines
bounds; merging does.

## One annotation workspace

Combine Identity and Boxes into one Selected signal panel in Annotate mode:

- Bounds: direct resize handles and Redraw bounds.
- Intersections: always-visible Merge intersecting boxes (N), shortcut M.
- Label: known-mode search and optional classifier suggestion.
- Selected-box actions: delete and undo/cancel the current operation.

CV, classifier, heuristic, manual, and saved boxes use the same commands. Origin
is a small provenance badge, not a separate editing workflow. Selecting a saved
box must not hide merge controls. Errors appear next to the relevant controls.

Keep Decode and Raw symbols in Classify mode. Put taxonomy details, provenance,
and history in Advanced view. Reuse the annotation editor in Classify mode when
the user chooses to create training examples from live operation.

## Window-level drafts and saving

Maintain one draft containing all boxes and changes for the window: stable box
identities, source references, base revisions, geometry, labels, merge decisions,
deletions, and dirty state.

- Selecting or redrawing an existing box edits it; it never implicitly creates
  another signal. New signal/drawing mode explicitly creates new boxes.
- Preserve label edits through resizing, redraws, selection changes, and merges.
- Draft operations support undo. Canceling a draft does not modify saved data.
- Save all geometry, identity, merges, and deletions atomically, with revision
  checks and protection against duplicate submissions.
- Save signal-identity links in the same window transaction, with stable IDs,
  optimistic conflict checks, and a visible undo path.
- Switching windows with unsaved changes offers Save / Discard / Stay.

Persistent window actions:

- **Save annotations:** save every change and keep working in this window.
- **Save & complete window:** save every change, mark this 10-second window
  reviewed, and advance or close review.

Unlabeled boxes look different from explicitly labeled Unknown boxes. Completion
flags unlabeled boxes and allows labeling them or explicitly accepting Unknown.
Simple annotation mode should not require a separate scene-coverage form.

## Merge behavior

- Recompute intersections whenever selection or draft geometry changes.
- Include active boxes of every origin, including saved-to-saved overlaps and
  results of earlier merges. Exclude superseded history.
- Highlight merge candidates before applying. Allow excluding a candidate:
  overlap alone does not establish that two signals are the same transmission.
- Reserve `S#` labels for saved signal identities. Use `Candidate #` for
  heuristic detections and `Draft #` for unsaved manual boxes to keep the
  identity, detection, and editing states visually distinct.
- Merge explicitly via the button or M; both use the same command.
- Keep the selected/adjusted bounds authoritative. Union bounds can be an
  advanced option, not a surprise expansion of the selected box.
- Stage the merge in the draft, preserve the target's label, and allow undo.
  Do not silently pick among conflicting human labels.
- Saving leaves one active region and retains source provenance/history.
  Superseded or rejected proposals do not become negative training examples.

## Visual clarity and shortcuts

- Prominent selected box; clear merge-candidate highlights; visible draft state.
- Hide superseded history by default. When shown, use faint text and outlines,
  and keep it out of ordinary selection/merge targeting.
- Allow removing active annotations with lineage while preserving their history.
- B: enter persistent drawing mode. M: merge intersections into selected box.
- Delete/Backspace: remove selected box. Escape: cancel current operation or exit
  drawing. Ctrl/Cmd+Enter: save annotations. Undo should cover draft operations.
- Protect text entry and use the same command handlers for shortcuts/buttons.

## Implementation sequence

1. Unify selectable boxes and window draft state; eliminate saved-region merge
   exclusions and implicit duplicate creation.
2. Implement transactional window saves, merge provenance, deletion handling,
   revision conflicts, and draft undo/cancel.
3. Replace the annotation panels with the combined editor, persistent draw mode,
   intersection preview, and window-level save/completion controls.
4. Connect keyboard commands, dirty navigation guards, and simple/advanced views.
5. Validate persistence and exercise the complete workflows in the browser.

## Acceptance scenarios

- Blank window: draw several boxes consecutively, label each, adjust any box,
  save once, and reopen with identical geometry and labels.
- Mixed sources: select a classifier single-tone box crossing CV fragments and
  saved regions, extend bounds, press M, label BPSK31, save. Exactly one active
  signal remains with the chosen bounds and label.
- Two overlapping saved regions can merge without redrawing either first.
- Redrawing an existing region never creates a second active region.
- A merged result can be edited, merged again, or removed while retaining history.
- Boxes linked by signal ID stay separate after saving and show the shared ID
  when the window is reopened.
- Shift-clicking boxes selects them for one identity link; ordinary click still
  selects a single box for editing. Linking is visually distinct from merging.
- Cancel/undo before saving does not persist merges or deletions.
- Conflicting revisions and failed saves are clearly visible; drafts survive.
- Reopening a reviewed/annotated window restores saved work without rerunning CV
  over it automatically. Completion applies only to the current 10-second window.

## Implementation progress

Implemented in the current UI slice:

- Moved multi-box drawing into the primary Annotate workspace.
- Combined box collection and the selected signal label editor into one
  annotation workspace, so drawing, selecting, labeling, and merging share a
  single panel.
- Made B enter persistent multi-box drawing mode; users can draw consecutive
  boxes and finish drawing explicitly.
- Draft boxes can be selected from the list or by clicking them on the waterfall.
- Saving a selected draft returns to the next draft instead of closing Annotate.
- Added draft add/remove/clear undo, Escape to stop persistent drawing, and
  Ctrl/Cmd+Enter to save the selected signal.
- Added per-box label drafts and one idempotent, transactional save for new
  annotations in the selected 10-second window.
- Added Previous / Next 10-second window navigation for longer captures. Moving
  windows resolves unsaved box drafts first and updates the waterfall view.
- Completion coverage is recorded for the selected window rather than always
  marking the first ten seconds.
- Added Tab / Shift+Tab navigation across draft signals when focus is outside a
  text or selection control.
- Connected B, M, Delete/Backspace, Escape, Ctrl/Cmd+Z, Ctrl/Cmd+Enter, and
  Tab/Shift+Tab to the same draft operations as their buttons.
- Added explicit identity-unknown confirmation for draft and saved signals.
- Batch saves retain explicit identity-unknown evidence so a saved unknown
  signal remains complete when the window is reopened.
- Saved-region label and geometry changes can be staged and undone with Ctrl+Z.
- Saved-region deletion is now staged in the window draft with a visible pending
  state and Undo delete action; it takes effect only when the window is saved.
- Deleting an active split/merge result now stages a soft removal. It disappears
  from active annotations and training exports, while label, geometry, and
  lineage history stay available; superseded sources remain protected.
- Lineage history identifies result nodes that were removed, so retained history
  remains understandable after the active result leaves the annotation list.
- Added persistent S-number signal identities to regions, with migration and
  per-region identity history. Shift-clicking boxes in one 10-second window
  stages a link; the user can instead enter an existing signal ID. Linked boxes
  keep separate bounds, while a label update propagates across active members.
- Shared-identity links are saved atomically with label/geometry edits and use
  optimistic label-revision checks. The saved-signal list and waterfall show
  each region's signal ID. Superseded merge-source labels no longer appear on
  the active waterfall; lineage remains available in history.
- Replaced source boxes are hidden from the list and waterfall by default; the
  Advanced view can reveal them for history inspection.
- Annotate mode now hides the global decoder output dock as well as the Decode
  and Raw inspector tabs; Classify mode restores it.
- Save controls now say Save annotations for Annotate and live capture, Save all
  annotations when window changes are staged, and Save signal for a single
  Classify-mode signal. Help text describes the current 10-second save behavior.
- The idempotent window transaction now applies new regions, geometry/label
  revisions, and eligible deletions together with optimistic revision checks.
- Saved-region and CV merges are staged in the same window draft as new boxes,
  edits, and deletions. The merge result stays editable and labeled until save;
  saving creates the result and source lineage atomically with the other window
  changes. Merge source revision conflicts leave the draft available to correct.
- A classifier draft crossed by a redraw can be folded into the merge result;
  intersecting draft boxes are absorbed, and a merge of drafts alone becomes one
  ordinary new annotation. Undo and removing the staged result restore the
  prior draft boxes and staged operations.
- Saved and CV sources staged for merging are visibly dimmed and excluded from
  additional merge, edit, delete, and dismissal actions until the window save.
- Added draggable corner handles to selected saved, draft, and provisional
  candidate boxes. Resizing a saved or draft signal stages its geometry while
  preserving its label; resizing a provisional candidate updates its proposed
  bounds, and intersections are recalculated for merging.
- Added drag-inside-to-move for a selected saved signal, annotation draft, or
  provisional candidate. Clicks that do not move still select overlapping
  boxes; moved saved geometry stays staged in the current 10-second window,
  preserves its label, and refreshes merge intersections. Draft moves support
  undo, and movement is clamped to the current window and visible frequency
  range.
- Selection and redraw merges show intersecting saved, CV, and unsaved draft
  boxes individually, highlight them on the waterfall, and allow excluding any
  non-target candidate before applying the merge.
- Selecting a draft box from the annotation list now recalculates and displays
  its current intersections immediately; staging a standalone selection as a
  draft does the same, so merge controls do not depend on another redraw.
- Label edits are committed to the current draft before switching to a
  heuristic/CV box, starting a new selection, clearing selection, adjusting a
  proposal, or deleting a signal. New selections clear stale taxonomy fields so
  a prior box's mode cannot leak into the next one; saved-region switches also
  cancel stale mode-search results from the previous selection.
- The selected-signal merge section remains visible while a signal is selected,
  including when there are zero intersections; it reports the count and explains
  when bounds need adjusting or another signal should be selected.
- CV and heuristic selections are excluded from their own intersection lists,
  preventing a selected box from being offered as a self-merge candidate.
- When a selected CV or heuristic box is the merge target, that source is now
  included in the staged merge transaction; the intersection list contains only
  the other boxes being absorbed, and the 32-source limit counts the target.
- If included saved/draft sources have conflicting human labels, the merge
  preview states the conflict and explains whether the target label or confirmed
  Unknown identity will be kept, or that the result needs a label.
- Batch classifier runs now add suggestions to draft boxes instead of creating
  saved pending regions before the operator saves the window.
- Window completion now finds unresolved saved signals and sends the user to
  label them or confirm unknown identity before recording completion.
- Completing a window with a standalone, unsaved selection now stages that
  selection as a normal window draft first, preserving detector and classifier
  evidence and subjecting it to the same label/unknown check and atomic save as
  other annotations.
- The primary Save annotations action now also stages a standalone selection
  before saving, so it cannot silently omit the currently selected signal when
  other boxes or edits are already drafted for the same window.
- Ctrl/Cmd+Enter now invokes the same enabled Save annotations action even when
  there is no active waterfall selection, so a saved-window draft can be saved
  directly from the keyboard. The shortcut hint now describes that behavior.
- Save button labels and help now describe the current mode: Annotate uses Save
  annotations, staged window changes use Save all annotations, and live capture
  explains the 10-second training-candidate requirement.
- Switching recordings or 10-second windows presents one Save / Discard / Stay
  dialog whenever there are unsaved changes; Escape and the focused default keep
  the operator in the current window. A standalone selection is staged before
  this check, so bounds and any label draft are included in the prompt as well.
- Saved-region and redraw merge actions now include the selected saved region as
  a merge source, and a single intersecting box can be merged with it.
- Heuristic single-tone, FSK-like, and MFSK-like candidate boxes now appear in
  intersection previews alongside saved, CV, and draft boxes. They can be
  excluded, highlighted, and staged into a merge with their proposal evidence
  retained in lineage; staged candidates are dimmed and cannot be merged twice.
- Merge-preview labels include each candidate's time span and frequency band so
  repeated or adjacent boxes can be distinguished before merging.
- Merge saves now accept heuristic proposal sources in the same atomic window
  transaction as saved regions, CV proposals, and annotation drafts.
- Manual draft-box labels now render in the shared HTML waterfall-label layer,
  using the same type size, spacing, and click behavior as saved and detected
  signal labels.
- Ctrl+Z now restores a draft box's previous label as one grouped edit instead
  of adding one undo entry for every typed character.
- Unlabeled saved and CV boxes now say “Needs label” and receive a distinct
  visual treatment; explicitly accepted Unknown labels remain visibly separate.
- Saved-region merge drafts now carry both label and geometry revisions. The
  atomic save rejects a source whose label changed after the merge was staged,
  preserving the draft so the operator can refresh and resolve it.
- Merging into a selected saved region now keeps that region as the active
  result, appends its geometry and label revisions, and records only absorbed
  regions/proposals as superseded lineage sources. After saving, the selected
  parent remains the sole active signal for that merged group.
- Browser verification confirmed persistent multi-box drawing, independent
  draft labels retained while switching boxes, and a staged merge that included
  both a heuristic single-tone candidate and a saved signal. The M shortcut
  staged the same merge as the button. Resizing a saved signal staged geometry,
  refreshed its intersection list, and Ctrl+Z restored the original bounds.
  Temporary drafts and geometry edits were cleared or undone without saving.
- Browser verification confirmed that attempting to complete a window with an
  unlabeled saved signal selects that signal and displays the label/unknown
  requirement instead of completing the window. No annotation was saved.
- Browser verification confirmed that Annotate hides the global decoder output
  dock and its Fldigi / Raw / Images controls, and Classify restores the dock
  alongside the Label / Decode / Raw symbols / Signals tabs.
- Browser verification confirmed that Annotate's save button reads Save
  annotations and its help refers to the current 10-second window; Classify's
  single-signal save reads Save signal. Neither mode shows the stale live-only
  tooltip while reviewing a saved capture.
- Reopening an existing annotated recording restored its saved MFSK region;
  the CV control remained an explicit Scan capture with CV action and no scan
  ran automatically.
- Browser verification confirmed that switching windows with a draft box opens
  one Save / Discard / Stay dialog and Stay preserves the current window and its
  draft. Switching recordings with a standalone selection stages it in that
  same draft and opens the same dialog. Both temporary drafts were cleared
  without saving. Choosing Discard moved to the next 10-second window with no
  draft remaining.

The window transaction now covers new boxes, saved-region geometry and label
revisions, eligible deletions, merges, and shared-identity links with optimistic
revision checks. Shift-clicking saved boxes in the active 10-second window stages
an identity link in the same save transaction; users can instead enter an
existing signal ID. Labels on superseded merge sources are hidden from the
active waterfall while their history remains available in Advanced view.
Browser checks cover draft staging, merge previews, resize undo, navigation, and
restoring an existing annotation without automatically scanning CV. Still
required through the updated backend: verify atomic merge and resize persistence,
lineage removal, saved-overlap merges, and revision conflicts with drafts
retained. The running process predates the backend changes and has no reload
enabled, so those persistence checks must wait until the user reloads it under
their terminal control. Static syntax and whitespace checks pass for the source.

Recent source follow-ups preserve the selected parent region identity when a
merge absorbs other regions, and remove merged CV proposals from the current
view and its per-capture cache while retaining them as dismissed review
feedback. The running API's OpenAPI schema was inspected read-only and does not
yet expose the batch `updates`, `deletions`, or `merges` fields. The client now
checks those capabilities before persisting edits so an older running server
cannot silently accept only part of the transaction. The browser was not
reloaded and no live annotation was saved during this check.

The current workbench process is still PID 769341, started more than 90 minutes
ago. Its command loads `sigid.server` from this same checkout, but its live
OpenAPI schema still omits the batch edit, deletion, and merge fields. This
confirms that the process has not loaded the current backend source; do not
restart it or reload its open page while the operator's unsaved draft is there.

## Scope

The UI and transaction changes listed under Implementation progress are present
in source. The remaining work is non-destructive integration validation for
merge, resize, lineage removal, saved-overlap handling, and revision conflicts.
Preserve existing annotations and the user's control of the running workbench
process while completing that validation.
