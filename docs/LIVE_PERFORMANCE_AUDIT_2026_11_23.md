# Signal ID long-recording performance audit (2026-11-23)

## Evidence

Profiling was performed against the isolated worktree
`/storage5/dev/sig_id-cw-keying`. Existing services on 8013/8014 were
inspected read-only and **not restarted**. In particular, 8014 was in
`state=overrun`, with `truncated_reason=audio_callback_queue_full`,
`frames_written=24513536` (510.7 seconds at 48 kHz) and 51 saved review
clips. This is consistent with, but does not prove, the review-indexing
backpressure hypothesis. The service continued to report `active=true` while
the capture had already aborted; its waterfall/analysis could keep polling the
same stagnant 10-second snapshot.

### Baseline filesystem/API probes

- Main 8014 process: ~880 MB RSS during inspection, with 101 open file
  descriptors (this alone does not establish a memory leak).
- `GET /api/live/review-queue`: ~0.05–0.13 seconds with 51 saved clips,
  returning ~55 kB. Queue enumeration and DOM rebuilding happen every 15 s,
  but these measurements do not explain an abrupt 10-second writer stall.
- `GET /api/captures`: ~0.004–0.022 seconds with ~57 kB response.
- Active live audio ring uses 10 s of samples and the writer input queue is
  limited to 32 blocks at 1,024 frames. At 48 kHz this is ~0.683 seconds
  of block capacity, making disk/SQLite stalls significant.

### Primary confirmed architectural issues

1. In `record` mode, `LiveCaptureManager._write_blocks` called
   `_save_review_window` synchronously every 10 seconds. This writes a
   separate review WAV, hashes it and inserts into SQLite. A slow transaction
   delays the exact thread draining the realtime audio queue. When the queue
   fills, the audio callback aborts the entire capture.
2. `sigid.spectrum.waterfall` iterated and decoded **every sample in a
   saved recording** via Python `WavSource.blocks()` even when the output
   had at most 640 rows. Focused-window reading also traversed preceding
   audio. Runtime grew approximately linearly with recording length.
3. The browser status displayed `Recording` for a capture marked
   `overrun` or `limit_reached`, and kept scheduling CV/heuristic
   analysis on unchanged audio. This makes the app appear stuck and wastes
   processing.

## Fixes

- **Dedicated bounded review persistence worker:** the primary writer now
  queues 10-second review WAVs to an independent thread and never waits for
  hashing/database writes. Queue capacity is four complete review windows.
  If saturated, *only optional review clips are skipped*; continuous WAV
  writing proceeds. Primary WAV data is never truncated deliberately by
  review queue backpressure.
- **Observability:** `GET /api/live` now reports writer/review queue depth
  and capacity, skipped review count, and last/maximum review-save latency.
  The browser displays latency and skipped clips while recording.
- **Halted capture UI:** the browser reports clear overrun/limit/error
  warnings with a Stop & save instruction, and stops redundant audio/CV
  analysis polling after capture has halted. Status polling remains active
  so the recording can still be finalized.
- **Direct-seek WAV waterfall:** read only selected FFT windows with
  `wave.setpos` and vectorized PCM decoding (8/16/24/32-bit, mono/stereo),
  preserving exact absolute sample positions and preprocessing. Complexity
  now scales with rendered FFT rows rather than source recording duration.

## Before/after waterfall benchmark

All numbers use the same local host, single-process `waterfall()`, 48 kHz
mono PCM16 **sparse silent WAV** fixtures, FFT 4096, hop 512, max 640 rows.
These are controlled filesystem/CPU measurements rather than browser latency.

| WAV duration | Old | New | Approx speedup |
| --- | ---: | ---: | ---: |
| 10 s | 0.354 s | 0.093 s | 3.8x |
| 60 s | 1.953 s | 0.135 s | 14.5x |
| 120 s | 3.833 s | 0.153 s | 25.1x |

All before/after rows had matching sample positions and identical row counts.
The short PCM-reference tests compare rendered dB spectra for eight PCM
sample-width/channel configurations with `waterfall_samples`, and a sparse
12-minute fixture asserts that the implementation never scans the complete
recording with `WavSource.blocks`.

## Reproduction / verification

```sh
cd /storage5/dev/sig_id-cw-keying
PYTHONPATH=. /storage5/dev/sig_id/.venv/bin/python -m pytest -q \
  tests/test_waterfall_seek.py tests/test_spectrum.py \
  tests/test_live_writer_backpressure.py \
  tests/test_live_spectrum.py tests/test_audio_range_api.py
node --check sigid/web/app.js
git diff --check
```

The slow-index regression holds the review database call behind a test
barrier, then verifies additional PCM writes continue. The saturation
regression forces >4 pending review windows and verifies that optional
clips are skipped while the continuous recording is complete and its WAV
header reports the exact input sample count.

`tests/test_live_audio.py` contains older unrelated mocked classifier
contracts (e.g. an API monkeypatch for `sigid.api.propose`, which no
longer exists). These are a separate legacy test update and do not
invalidate the new performance regressions.

## Rollout and limitations

The live process on 8014 has an existing aborted capture which **must
be finalized via Stop & save** before it can be restarted with the updated
backend. Never discard that partial WAV or restart that server implicitly.

A new isolated preview can load this code on port 8015 without affecting
8013/8014, provided 8015 is not actively capturing. The operational
failure on 8014 will not repair itself through hot-reloading.

Further investigation if long-running load persists: profile sounddevice
callback scheduling, whether native decoder jobs compete for CPU, review
queue SQL pagination/UI virtualization at thousands of captures, and
remote playback/network latency. The changes above address the two
demonstrated O(duration)/backpressure paths but do not prove all possible
slowdowns are fixed.
