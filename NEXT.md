# Next checkpoint

Status: sample-indexed WAV replay, 2-FSK fixtures and SQLite capture/region metadata
are implemented; Fldigi audio integration and UI remain open.

## Agreed scope

- Audio-passband waterfall, signal tracking, progressive identification and decoding.
- Full installed Fldigi receive catalog as the decoding target, including variants
  and specialized outputs; automatic identification coverage tracked separately.
- Real/synthetic datasets with partial and hierarchical modulation/protocol labels.
- A trained first-pass classifier with rejection and an auditable labeling loop.
- Unknown-mode symbol/bit/soft-output analysis for a bounded initial family set.

## Next action

Continue the receive foundation in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md):
build a waterfall preview and resolve why a fresh isolated Fldigi profile does not
open its XML-RPC listener. WAV replay, clean 2-FSK generation, durable sample-aligned
regions, revisioned labels and current-label JSONL export are implemented.

## Evidence boundary

Verified: Fldigi 4.2.13 and its XML-RPC method catalog are installed; unit tests
cover receive-only client semantics, PCM WAV block indexing/normalization and
synthetic 2-FSK metadata and symbol tones, SQLite capture hashes/regions/label
history, and JSONL export. `pytest`: 22 passed, 1 skipped. WAV replay is sample-indexed
and synthetic files carry protocol-unknown FSK labels. Fresh
isolated profiles show a configuration wizard; after closing it, the configured
XML-RPC port still did not listen. No Fldigi audio decode, live audio capture,
waterfall UI, model, or hardware/RF validation is claimed.
