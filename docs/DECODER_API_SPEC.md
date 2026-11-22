# First-party decoder API contract — v0.1

**Audience:** independent agents implementing one ham-radio decoder at a
time. **Scope:** documents the *implemented* library and HTTP adapters.
Anything in "Future extensions" is not yet available.

**Start here:** [Agent playbook](DECODER_AGENT_PLAYBOOK.md) |
[Mode work items](decoder-work-items.json) |
[Source/evidence references](DECODER_EVIDENCE_INDEX.md) |
[Decoder roadmap](LIBSIGDECODE.md) |
[Result JSON schema](schemas/local_decode_result.schema.json).

## Fundamental rules

The decoder consumes bounded normalized audio and returns data. It must
not open an SDR, access an audio interface, tune CAT, transmit, invoke
external programs, issue HTTP requests or write artifacts. The host owns
capture, input selection, persistence, and external integrations.
Use independently written implementations from published specifications
and test vectors. Do not copy or translate OnAir, Fldigi, or WSJT-X code,
or copy their nontrivial tables. External decoder programs are optional
black-box references only, NOT runtime dependencies.

Keep detector boxes, modulation hypotheses, partial symbols, decoder
results and independent protocol validation distinct. Do not invent
confidence probabilities or treat a plausible string as a verified QSO.

## Current Python contract (authoritative source: sigid/libsigdecode/core.py)

Example invocation:

    from sigid.libsigdecode import DecodeWindow, create_library
    result = create_library().decode(
        "cw",
        DecodeWindow(samples=mono_samples, sample_rate=12000,
                     sample_start=48000, carrier_hz=1100.0),
    )
    payload = result.to_dict()

DecodeWindow (frozen dataclass):

| Field | Current semantics |
|---|---|
| samples | 1D finite nonempty NumPy array, coerced float64; max 120 seconds |
| sample_rate | integer from 1000 to 384000 Hz |
| sample_start | >= 0, absolute index in original capture |
| carrier_hz | audio Hz, not RF frequency; null or 20 <= Hz < Nyquist |

The current input type is MONO AUDIO only. Complex IQ and streaming are
NOT implemented. No decoder should disguise IQ as audio to evade this.

DecoderSpec: id (unique lowercase slug), name, family, input_kind
(default 'mono-audio'), stage (default 'experimental'), version.
One class per decoder with:

    class Decoder:
        spec: DecoderSpec
        def decode(self, window: DecodeWindow) -> LocalDecodeResult: ...

LocalDecodeResult: decoder_id, version, status, text, tokens, evidence,
note. The to_dict() serialization adds
source='sigid-independent-implementation'. Internal status is one of:

- decoded: plausible output meets decoder's stated criteria; still not
  independently operator-verified.
- partial: useful decoded/symbol data but missing integrity validation.
- no_decode: no trustworthy text; text MUST be empty, though tentative
  tokens/evidence may be retained with appropriate qualifications.

DecodedToken: text, start_sample (inclusive), end_sample (exclusive),
pattern, confidence_hint. Sample indexes are in the ORIGINAL capture
timeline, not relative to the cropped array. confidence_hint is a
0–1 diagnostic, NOT a calibrated probability unless proved and
documented per decoder. CW uses a dot/dash string for pattern.
Evidence is an extensible, JSON-compatible dict; report real CRC/FEC
outcomes, SNR, sync and partial-bit diagnostics when measured. Do not
invent a generic "confirmed" or "crc_ok" field for modes without it.

JSON transport shape:
[local_decode_result.schema.json](schemas/local_decode_result.schema.json).
The schema validates shape, not correctness.

## Registration and per-mode changes

1. Implement sigid/libsigdecode/<mode>.py, with a unique DecoderSpec
   and decode(DecodeWindow) method.
2. Add tests/test_decode_<mode>.py (independent golden vectors, corrupt
   inputs, noise and false-positive abstention).
3. Only after minimal tests pass, add the instance to
   sigid/libsigdecode/__init__.py:create_library().
4. The existing sigid/local_decoder_backend.py automatically enumerates
   registered modes under backend ID 'sigdecode'. DO NOT create another
   backend class or duplicate the registry.
5. Update work queue status/evidence, docs, validation limitations.
6. Request integration-owner review for any shared DSP or contract
   change. Do not modify capture, API, or browser code from a per-mode
   branch unless explicitly coordinated.

DecoderLibrary.register() rejects duplicate IDs. decode() rejects unknown
modes and checks the returned decoder_id matches the requested mode.

## Existing HTTP bridge (implemented)

Authoritative sources: sigid/api.py, sigid/decoder_backend.py,
sigid/local_decoder_backend.py. The running FastAPI server also
publishes its machine-generated HTTP OpenAPI document at GET /openapi.json
(e.g. http://127.0.0.1:8012/openapi.json). It describes the generic
decoder routes, not the detailed in-process Python protocol. The
generic backend extension protocol
and entry-point group 'sigid.decoder_backends' remain available for
optional external decoders.

GET /api/decoder-backends

Returns backends[], plugin_errors[], extension_group. An in-project
decoder is exposed in backend 'sigdecode' (connected true), with a mode
entry ONLY after registration. Each mode entry includes mode_id,
display_name, native_mode, selectable, note.

POST /api/captures/{capture_id}/decoder-backends/sigdecode/decode

Current NativeDecoderRequestBody fields:
- mode_id: string, required
- sample_start: integer >= 0, required
- sample_count: positive integer, required
- carrier_hz: audio Hz or null, optional
- region_id: active region ID or null, optional; if provided, selected
  sample bounds must match the linked region exactly.

Example JSON request:

    {"mode_id":"cw","sample_start":0,"sample_count":120000,
     "carrier_hz":1100.0,"region_id":null}

The server validates capture bounds, copies an exact PCM16 mono WAV crop
to a temporary file, calls the registered backend, and saves a decode-run
record. The HTTP request model allows up to 192000*600 samples, but the
*local DecodeWindow currently limits decoding to 120 seconds*; do not
mistake the HTTP maximum for a supported 600-second local decode.

Response keys: run_id, backend_id, application, mode_id, native_mode,
status, version, note, capture_id, sample_start, sample_count,
sample_rate, products[]. Products include kind, content, encoding,
media_type, metadata. The first-party backend emits a text product only
when text exists, and always emits a metadata product with a
JSON-encoded STRING in content carrying the full
LocalDecodeResult.to_dict(), plus external_decoder_used=false.

IMPORTANT: external/backend run status is 'completed' if text exists,
or 'completed_empty' otherwise. Those are DIFFERENT from local result
statuses 'decoded', 'partial', 'no_decode', which are in the metadata
product. The response does not auto-accept a label or confirm a protocol.

HTTP 404: backend/capture missing; 422: invalid mode or selection;
503: unavailable external decoder or decoder execution failure.
A first-party decoder must not require Fldigi, WSJT-X or OnAir.

CLI: installed 'sig-id-decode', or:

    python -m sigid.libsigdecode.cli recording.wav --mode cw --carrier-hz 1100

Current CLI input: mono 16-bit PCM WAV and <= 120-second selection.

## Future extensions — NOT PART OF v0.1

- Typed IQWindow with RF center, complex samples, sideband, gain and
  provenance; cannot reconstruct IQ from pre-demodulated SSB audio.
- Streaming reset/feed/finalize and state/latency lifecycle.
- First-class image, binary packet/frame, soft-bit and FEC/CRC products.
- Decoder capability/version negotiation, cancel/timeouts, configurable
  resource limits and typed mode-specific options.

When a mode needs one of these, propose a shared API extension and
tests in a separate integration PR. Do not privately change contracts
or claim support based only on a mode catalog label.

## Required evidence before claiming readiness

Read [identification evidence](IDENTIFICATION_CHAIN.md),
[crowded FT8 benchmark](CROWDED_AUDIO_BENCHMARK.md) and
[evidence/source index](DECODER_EVIDENCE_INDEX.md).
Synthetic-only tests imply 'experimental', NOT 'validated on air'.
For CW, a saved IC-7300 recording with inadequate timing fit must
abstain rather than display gibberish. Other modes need equivalent
negative controls and independently reviewed real signals.
