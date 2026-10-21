# Next checkpoint

Status: milestone 0 adapter foundation in progress; audio routing is not yet proven.

## Agreed scope

- Audio-passband waterfall, signal tracking, progressive identification and decoding.
- Full installed Fldigi receive catalog as the decoding target, including variants
  and specialized outputs; automatic identification coverage tracked separately.
- Real/synthetic datasets with partial and hierarchical modulation/protocol labels.
- A trained first-pass classifier with rejection and an auditable labeling loop.
- Unknown-mode symbol/bit/soft-output analysis for a bounded initial family set.

## Next action

Continue milestone 0 in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md): resolve
why an isolated Fldigi process does not open its XML-RPC listener, then prove audio
input, offline waveform generation, decoded output, reset and replay. The Python
client, catalog capability record and unit tests are in place.

## Evidence boundary

Verified: Fldigi 4.2.13 and its XML-RPC method catalog are installed; receive-only
client unit tests pass. An isolated process stayed alive but did not open the
configured XML-RPC port, so no live catalog or audio integration is claimed yet.
No application UI, audio loopback proof, generated waveform fixture, model, or
hardware/RF validation exists. See commit log for the staged changes.
