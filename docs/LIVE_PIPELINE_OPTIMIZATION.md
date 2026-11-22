# Signal ID live pipeline optimization — milestone 1

## Goal and guarantees

Optimize audio → CV/heuristic detection → track-level classifier → isolated native/external decoders → operator annotation. Detector scores, classifier hypotheses, readable text and CRC-checked decoder frames remain distinct evidence; none silently creates a confirmed protocol identity or training label.

## Changes

- **Independent live polling.** CV detection begins without waiting for the heuristic audio-analysis request. Each has independent in-flight protection; CV is throttled to about one scan per 2.5 seconds. The live audio callback, spectrum feed and saved evidence remain untouched.
- **Sample-provenance classifier cache.** At most 128 track entries, reused for up to six seconds of new sample time. Entries are tied to CV track ID, session, classifier model object/hash, audio sample rate and frequency geometry. A new model, session, major frequency/width drift or old data forces rescoring. Reuse retains the original crop start/count, carries an explicit reused flag and audio age, and never invents new measurements. Status queries for another session cannot flush the active cache.
- **Bounded model work.** At most eight *new* classifier invocations per CV scan, sorted by persistent status and detector confidence. Cached results are free and do not consume those eight slots, so deferred tracks are scored in subsequent scans. Deferred boxes remain detectable with an explicit reason; nobody mistakes them for a valid model classification.
- **Current evidence.** The latest classifier result overwrites old claims on a persistent track, even if its confidence is lower or it abstains; before this change the live browser retained the highest-confidence historical guess. UI shows model reuse age, fresh/reused/deferred counts, CV HTTP latency and separate heuristic request latency.
- **Live annotation.** Every CV candidate now offers Inspect recent audio. It selects up to 9.8 seconds of that box from the actual 10-second audio ring and opens the manual label/annotation inspector. Expired audio is not silently reconstructed: operators are directed to stop and review the full saved recording. The existing explicit save validation still governs the training-candidate workflow.
- **Bounded presentation history.** The live candidate list retains about 75 seconds of candidate history, with a 240-item cap and a currently selected candidate preserved. No saved recording, decoder evidence or annotation is deleted.

The native decoder worker remains limited to five starts/minute, three seconds child CPU/minute, one running process, zero queued jobs and four-second input windows. Optional Fldigi, OnAir and WSJT-X adapters remain unchanged. All native text is tentative unless independently reviewed.

## Controlled benchmark

Run:

    PYTHONPATH=. python tools/bench_live_pipeline.py --polls 6 --scorer-delay-ms 8 --output /tmp/live-pipeline-report.json

This creates isolated API apps using twelve synthetic CV tracks over six slightly advancing audio snapshots and a mocked classifier taking 8 ms per call. It compares the same eight-new-scores-per-scan budget with cache reuse disabled versus a six-second cache. It does not open audio hardware or use a real CV model.

| Metric | Cache disabled | Cache enabled |
| --- | ---: | ---: |
| Fresh model calls | 48 | 12 |
| Median HTTP scan time | 66.94 ms | 1.29 ms |
| Total six-scan HTTP time | 412.88 ms | 112.28 ms |
| Cache reuse events | 0 | 56 |
| Deferred models after second scan | 4 | 0 |

That is 75% fewer classifier calls and a 3.68× ratio of cumulative HTTP wall time **in this synthetic test**. It is not a calibrated full-app benchmark or evidence of better on-air precision/recall.

## Verification

- Python unit/API tests validate the provenance of reused classifier crops, model/frequency/session invalidation, bounded LRU, 8-new-scores ceiling, eventual scoring of deferred candidates, classifier opt-out and non-authoritative evidence.
- Node tests verify live history pruning and preservation of selected/unaltered candidates.
- An isolated Chrome run on port 8016 with **mock CV and classifier** attached to real PulseAudio Signal Lab audio successfully displayed live detections, opened Inspect recent audio, selected in-ring sample coordinates, and enabled the normal manual save control, with no JS/network-module errors. Those fake detector results were not treated as real detection accuracy.
- The preview on port 8013 is only updated after CI and regression checks pass.

## Next milestones

1. Profile real CV model p50/p95 and CPU/GPU load at normal/negative SNR. Compare detection hit/false-positive rates with reviewed audio, separate from classifier top-3 accuracy and decoder payload integrity.
2. Improve native modem acquisition/streaming beyond isolated four-second crops, including carrier search and retry scheduling. Keep real-time CPU admission bounds.
3. Show side-by-side native, Fldigi, OnAir and WSJT-X results, including missing/failed/abstained outputs, exact sample windows, frame integrity and sources.
4. Streamline saved-workbench keyboard triage, full-window negative coverage, corrected geometry and review history. Do not silently promote model hypotheses to operator-approved labels.
5. Build a dataset feedback gate where only explicitly reviewed corrections, positives and negatives can be exported to model training, with held-out evaluations before promotion.
