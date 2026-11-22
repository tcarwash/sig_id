# Decoder conformance fixtures

This directory contains **pinned, deterministic synthetic tests** and the
place where parallel decoder agents may add independent positive and negative
WAV fixtures for their mode. See
[Decoder conformance](../../../docs/DECODER_CONFORMANCE.md).

Bundled `cw/manifest.json` checks two known textbook Morse messages and
three unrelated/silent signal sources. These are synthetic demonstrations,
**not evidence of an on-air reception**, and no external decoder source is
copied. WAV files are mono PCM16, sampled at 12 kHz. Each clip is SHA-256
pinned; original sample indexes in the manifest deliberately start at 17000
to exercise absolute index handling.

Regenerate the **entire synthetic CW reference** with
`python tools/build_decoder_conformance_cw_fixtures.py`. This overwrites
the WAVs and their golden SHA-256 references; verify any resulting changes
rather than updating truth to match a decoder's output.

A new mode's agent owns
`tests/fixtures/decoder_conformance/<mode>/manifest.json` and all of its
fixtures, and calls `evaluate_suite(..., library=DecoderLibrary((Decoder(),)))`
while the decoder is not yet in the shared registry.

Never check in unlicensed, private, or sensitive recordings; use a local
opt-in manifest outside the repository for those sources.
