# Decoder development agent playbook

This is the start page for an agent assigned an independently implemented
ham-radio decoder in Signal ID.

**Required reading, in order:**

1. [API contract](DECODER_API_SPEC.md) — implemented vs proposed APIs.
2. [Decoder queue](decoder-work-items.json) — work item, dependencies,
   source links, maturity, and missing primary protocol documentation.
3. [Evidence index](DECODER_EVIDENCE_INDEX.md) — source/provenance hierarchy
   and exact internal benchmark links.
4. [Library roadmap](LIBSIGDECODE.md) — coverage goal and copyright policy.
5. [Evidence chain](IDENTIFICATION_CHAIN.md) and
   [crowded audio benchmark](CROWDED_AUDIO_BENCHMARK.md) — uncertainty,
   nonidentification and real/synthetic separation.
6. [Current library source](../sigid/libsigdecode/core.py) and
   [existing CW implementation](../sigid/libsigdecode/cw.py) as an example
   of how to CALL the API, not a template for every modulation.
7. [Pinned conformance harness](DECODER_CONFORMANCE.md) — SHA-256 WAV
   manifests, positive/negative controls, metrics and separate real-air
   ground-truth requirements.

## Parallel implementation policy

- Each agent owns ONE decoder ID from decoder-work-items.json and its
  own branch/worktree. The integrator owns common API, registration,
  shared DSP components, CI, and user-facing controls. Never let two
  agents modify the same working tree or overwrite uncommitted work.
- Put mode-specific code in sigid/libsigdecode/<mode>.py and tests in
  tests/test_decode_<mode>.py. Propose shared DSP primitives separately,
  with a generic test case and documented owners.
- Only the integrator edits sigid/libsigdecode/__init__.py to register a
  decoder after it meets golden + negative-control tests; no registration
  of an empty/stub decoder as though it were operational.
- Do NOT import/copy/translate source or tables from OnAir, WSJT-X,
  Fldigi or other decoder apps. They can be run as optional, separately
  installed black-box references. Work from published standards and
  independently created or clearly licensed test vectors.
- Do NOT modify sigid/api.py, sigid/decoder_backend.py, existing live
  capture, decoder queues or PTT/tuning code from a mode branch. Any
  missing IQ, streaming, image or packet API is an integration proposal,
  NOT permission to invent a private incompatible interface.
- Preserve original capture sample indexes, carrier offsets, rate and
  provenance. Never upgrade a classifier guess or arbitrary text into
  a confirmed protocol. Allow unknown/abstain, partial data, and raw
  soft decisions when justified.
- Production defaults must be receive-only. Never key, tune, transmit,
  seize the sound card, or restart a running capture to test a decoder.

## A decoder's completion checklist

- [ ] Name exact mode/variant, frame timing, input/output type and
      version of source specification; record unanswered protocol fields.
- [ ] Implement DecoderSpec and decode(DecodeWindow) with bounded CPU
      and memory use; handle unsupported rates/types with clear errors.
- [ ] For packet bytes, CRC/FEC outcomes, or soft decisions use the
      additive v0.2 `DecodedFrame`, `IntegrityCheck`, and `SoftBitSpan`
      from `sigid.libsigdecode`; do not invent bespoke JSON result keys.
      A passing frame check is not a human-verified mode identification.
      Old text-only `LocalDecodeResult` constructors remain compatible.
- [ ] Independently generate golden waveforms/payloads and include
      machine-readable expected bits/text/frame integrity.
- [ ] Supply pinned WAV SHA-256 conformance manifest under
      tests/fixtures/decoder_conformance/<mode>/; run its positive and
      negative controls using sigid.conformance.evaluate_suite with an
      injected DecoderLibrary rather than editing shared registration.
      See DECODER_CONFORMANCE.md. Never claim synthetic success as on-air validation.
- [ ] Test corrupted frames, wrong speed/carrier/polarity, drift, weak
      SNR, neighboring stations, silence and unrelated modulation;
      require abstention rather than readable gibberish.
- [ ] Report absolute token/frame sample extents, error/SNR/sync
      estimates and FEC/CRC provenance when actually measured.
- [ ] Add a mode evidence record under
      docs/decoder-evidence/<mode>.md using
      [the evidence-index template](DECODER_EVIDENCE_INDEX.md).
- [ ] Verify with independently reviewed, reproducible real captures
      before promoting from experimental to validated.
- [ ] Compare with external decoders if useful; call agreement a
      comparison, not ground truth.
- [ ] Run mode tests, contract tests, and applicable broader regression;
      explicitly list limitations and benchmarks.
- [ ] Prepare a PR; integrator alone handles registry updates and
      source/work queue merge conflicts.

## Baseline commands

From a feature-branch worktree:

    python -m pytest -q tests/test_libsigdecode.py
    python -m pytest -q tests/test_decoder_contract_docs.py
    python -m pytest -q tests/test_decoder_conformance.py
    python -m pytest -q tests/test_decode_<mode>.py
    python -m pytest -q tests/test_api.py tests/test_classifier_api.py
    node --input-type=module --check < sigid/web/app.js

After creating a mode-specific suite, add an independent test calling
evaluate_suite(..., library=DecoderLibrary((YourModeDecoder(),))) from
DECODER_CONFORMANCE.md. The shared CW suite can also run with:

    python -m sigid.conformance tests/fixtures/decoder_conformance/cw/manifest.json

The mode-specific command applies only after that mode's tests exist.
Do not run a test suite against the real receiver or modify hardware.
Current unit tests are synthetic; report that limitation clearly.

## Pasteable delegation instruction

> Implement decoder MODE_ID from docs/decoder-work-items.json on your
> own isolated feature branch/worktree. Start with
> docs/DECODER_AGENT_PLAYBOOK.md, docs/DECODER_API_SPEC.md, and
> docs/DECODER_EVIDENCE_INDEX.md; inspect the exact mode's reference_ids.
> Independently implement from published protocol specifications, without
> copying/translating OnAir/Fldigi/WSJT-X code. Use existing DecodeWindow,
> DecoderSpec, LocalDecodeResult and DecodedToken unchanged. If the mode
> requires an unsupported IQ/streaming/multimedia contract, STOP at a
> concrete integration design and tests rather than making up an API.
> Deliver the mode module, deterministic golden vectors, corruption/noise
> abstention tests, provenance/evidence record, benchmark limitations, and
> a PR. Do not edit shared registry, capture, CAT/PTT or live server.
> Do not claim over-the-air success without independently reviewed audio.

## Work ownership example

- Agent A: rtty / ITA2 and framing; works only on RTTY module/tests.
- Agent B: bpsk31 / differential carrier/varicode; separate worktree.
- Agent C: ax25 HDLC/bit stuffing/FCS with ideal bitstream vectors;
  raises shared AFSK request as separate dependency.
- Agent D: CW timing improvements against reviewed recordings.
- Agent E: FT8 technical research + independently generated golden
  vectors; full slot/FEC integration requested from integrator.

These may proceed in parallel. Decoders requiring unimplemented input
contracts are allowed to finish **spec research and pure logic** first
but must not falsely mark the full waveform decoder as operational.

## Review and status gate

Work queue state 'experimental_implemented' means the module runs and
has synthetic tests, not that real air decoding is reliable. Only promote
to 'benchmarked' with independent real capture ground truth, and only
to 'validated' with reviewed metrics. Do not conflate the 857-item Signal
ID naming catalog or OnAir's coverage list with actual built-in support.

## Machine queue dependencies and expansion

The work queue lists 36 initial candidates, not a claim that OnAir's
entire amateur mode menu is implemented or fully specified. Its
depends_on values that are not decoder IDs are named shared
infrastructure components in infrastructure_dependencies. The
integration owner, not a mode agent, manages these shared components.
Additional ham modes can be appended after public specs and expected
inputs are established.
