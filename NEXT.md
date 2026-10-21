# Next checkpoint

Status: local WAV upload/replay workbench, spectrogram, manual region labels,
sample-indexed replay, live audio capture, 2-FSK fixtures and SQLite dataset index
are implemented. Fldigi audio integration remains open.

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
- Clean deterministic 2-FSK WAV and sidecar manifest with exact reference bits.
- Local input-device discovery and bounded mono/stereo PCM recording through a
  background writer; live audio is saved as a normal capture with device provenance.
- Browser controls to start/stop live recording and immediately review its waterfall.

## Next action

Continue release 0.1 in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md): prove live
capture on an actual input device, then solve and integrate Fldigi audio decode.
The workbench uploads or records audio, draws its spectrogram, selects
time/frequency regions, saves partial labels and displays saved regions. Persistent
data is exportable as JSONL through the dataset store.

## Evidence boundary

Verified: Fldigi 4.2.13 and its XML-RPC method catalog are installed; unit tests
cover receive-only client semantics, PCM WAV block indexing/normalization and
synthetic 2-FSK metadata and symbol tones, SQLite capture hashes/regions/label
history, JSONL export, spectrogram peak detection, and the upload/region API path.
`pytest`: 33 passed, 1 skipped; `node --check sigid/web/app.js` passes. Deterministic
fake-device tests exercise WAV writing, capture indexing, API start/status/stop and
waterfall access. WAV replay is
sample-indexed and synthetic files carry protocol-unknown FSK labels. Fresh
isolated profiles show a configuration wizard; after closing it, the configured
XML-RPC port still did not listen. Live hardware capture has not been validated.
No Fldigi audio decode, automatic identification, trained model, or hardware/RF
validation is claimed.
