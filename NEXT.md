# Next checkpoint

Status: local WAV upload/replay workbench, spectrogram, manual region labels,
sample-indexed replay, live audio capture, 2-FSK fixtures and SQLite dataset index
are implemented. Receive-only Fldigi catalog selection and decoded-text display are
implemented. A synthetic BPSK31 audio loopback has passed through Fldigi into the
workbench API; radio-input routing and independent-signal validation remain open.

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
- Receive-only Fldigi API/UI: connection status, catalog list, manual mode selection
  and incremental decoded-text polling. Transmit, rig and macro calls are not exposed.

## Next action

Continue release 0.1 in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md): connect a
live radio audio input to the configured Fldigi source and verify a known over-the-air
or independent audio fixture. The decoder panel can select from the connected Fldigi
catalog and display the receive text; input-device routing remains an explicit OS or
Fldigi configuration step.

## Evidence boundary

Verified: Fldigi 4.2.13 and its XML-RPC method catalog are installed; unit tests
cover receive-only client semantics, PCM WAV block indexing/normalization and
synthetic 2-FSK metadata and symbol tones, SQLite capture hashes/regions/label
history, JSONL export, spectrogram peak detection, upload/region APIs, and Fldigi
status/catalog/mode/output APIs. `pytest`: 34 passed, 2 skipped;
`node --check sigid/web/app.js` and wheel build pass. Deterministic fake-device tests
exercise WAV writing, capture indexing, API start/status/stop and waterfall access.
An isolated live Fldigi 4.2.13 process served the API with the documented receive-only
allow-list: 169 modes, THOR16 selection, and receive-buffer queries all worked. A
separate, opt-in integration test passes: it creates two copied profiles with all
rig/PTT paths disabled, routes BPSK31 transmit audio through a temporary PulseAudio
null sink, and verifies the complete payload arrives through the workbench API. Run it with
`SIGID_TEST_FLDIGI_AUDIO=1 pytest tests/test_fldigi_audio_integration.py`. This is a
software loopback, not an independent modem or RF test. A fresh blank profile did
not expose XML-RPC until initialized. WAV replay is sample-indexed and synthetic
2-FSK files carry protocol-unknown labels. Live hardware capture, radio audio routing,
automatic identification, trained model, and RF validation remain unverified.
