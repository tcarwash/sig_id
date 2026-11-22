# CW — evidence dossier (experimental only)

- Owned mode ID: cw
- Implementation/version: sigid/libsigdecode/cw.py, cw-envelope-v1
- Status: experimental_implemented; never independently validated on air
- Primary protocol standard: [ITU-R M.1677-1](https://www.itu.int/rec/R-REC-M.1677-1-200910-I/)
- In-project API: [decoder contract](../DECODER_API_SPEC.md)
- Deterministic reference waveforms: [test_libsigdecode.py](../../tests/test_libsigdecode.py)
- Receiver detection regression: [test_narrow_tone_detection.py](../../tests/test_narrow_tone_detection.py)

## Coverage and limitations

The independent envelope decoder accepts one selected audio carrier,
not a full multi-station skimming passband. It estimates mark/space
timing from known international Morse alphabet patterns. Character
timing-fit diagnostics are not calibrated likelihoods. It currently
does not handle all types of drifting carriers, strong key-click
artifacts, operator fist variants, or automatic frequency finding.

## Evidence established

- Controlled synthetic international Morse messages pass including
  CQ TEST, SOS, DE AG7SU and CQ CQ, under a nearby carrier and noise.
- Noise, a stationary carrier and an unrelated synthetic frequency
  hopper are explicit negative controls; those must not produce
  a successful Morse text result.
- On retained audio from an IC-7300, the first version returned
  plausible-looking gibberish with low timing-fit values (~0.23–0.29).
  Its decoder was corrected to abstain in those windows and retain
  tentative token evidence without publishing guessed text.
- **Not established:** reviewed transcript, exact ground truth station
  boundaries, real-world character/word error rate, power/SNR
  calibration or reliable multi-signal CW reception.

## Next acceptance gate

Obtain multiple independently hand-transcribed on-air CW recordings
across speed, fading, interference and receiver gain variations. Record
WAV SHA-256, rate, precise time/frequency windows, character truth,
false-decode rate and WER/CER before claiming on-air validation.
