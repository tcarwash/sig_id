# Next checkpoint

Status: WAV replay and first synthetic modulation fixture implemented; Fldigi audio
integration and UI remain open.

## Agreed scope

- Audio-passband waterfall, signal tracking, progressive identification and decoding.
- Full installed Fldigi receive catalog as the decoding target, including variants
  and specialized outputs; automatic identification coverage tracked separately.
- Real/synthetic datasets with partial and hierarchical modulation/protocol labels.
- A trained first-pass classifier with rejection and an auditable labeling loop.
- Unknown-mode symbol/bit/soft-output analysis for a bounded initial family set.

## Next action

Continue the receive foundation in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md):
add persistent capture/region metadata and a waterfall preview, and resolve why a
fresh isolated Fldigi profile does not open its XML-RPC listener. The WAV source and
clean 2-FSK generator provide deterministic fixtures while that integration is
being investigated.

## Evidence boundary

Verified: Fldigi 4.2.13 and its XML-RPC method catalog are installed; unit tests
cover receive-only client semantics, PCM WAV block indexing/normalization and
synthetic 2-FSK metadata and symbol tones. `pytest`: 15 passed, 1 skipped. WAV replay
is sample-indexed and synthetic files carry protocol-unknown FSK labels. Fresh
isolated profiles show a configuration wizard; after closing it, the configured
XML-RPC port still did not listen. No Fldigi audio decode, live audio capture,
waterfall UI, model, or hardware/RF validation is claimed.
