# MODE_ID — evidence dossier

- Owned mode ID:
- Implementation/version:
- State (planned, experimental_implemented, benchmarked, validated):
- Primary published protocol specification and exact revision:
- Protocol fields not yet specified:
- Source/provenance and any third-party license audit:
- Input (audio vs IQ, sample rate, carrier, timing, variant):
- Output (text, tokens, packet, image, CRC/FEC status):

## Independently generated golden vectors

List each test input, SHA-256, method, known bits/frames/payload and
expected results. Separate protocol vectors from generic modulations.

## Negative and boundary controls

Noise, unrelated modes, missing/incorrect sync, interference, drift,
CRC/FEC corruption, wrong clock/frequency/polarity, high CPU usage,
timeouts and invalid input.

## Reviewed real recordings

Exact WAV identity and SHA-256, sample rate, time/frequency crop,
independent human or protocol-integrity ground truth, consent if needed,
holdout/evaluation procedure, CER/WER/packet error rate/false positives.

## Reproducibility and limitations

Test commands, version, hardware-independent runtime, known failure
cases, external app comparisons (clearly marked non-ground-truth), owner,
reviewer and next acceptance gate.
