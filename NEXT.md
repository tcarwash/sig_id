# Next checkpoint

Status: planning complete; application implementation has not started.

## Agreed scope

- Audio-passband waterfall, signal tracking, progressive identification and decoding.
- Full installed Fldigi receive catalog as the decoding target, including variants
  and specialized outputs; automatic identification coverage tracked separately.
- Real/synthetic datasets with partial and hierarchical modulation/protocol labels.
- A trained first-pass classifier with rejection and an auditable labeling loop.
- Unknown-mode symbol/bit/soft-output analysis for a bounded initial family set.

## Next action

Begin milestone 0 in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md): scaffold the
minimum project and prove isolated Fldigi waveform generation, audio input, decoded
output, state reset and replay behavior. Audit actual backend/tool availability.

## Evidence boundary

Only design and planning documents exist. No code, test fixtures, model, integration
proof, runtime benchmark or hardware/RF validation has been produced.
