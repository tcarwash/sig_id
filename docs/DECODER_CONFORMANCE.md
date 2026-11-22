# Decoder conformance and benchmark harness (v1)

This is the shared **receive-only** fixture runner for `sigid.libsigdecode`.
It tests input/output contracts, exact text or binary frame recovery,
explicit CRC/FEC outcomes, and abstention on negative controls. It does
not tune a radio, transmit, invoke installed SDR software, or create fixtures
automatically while evaluating. A green synthetic suite is **experimental
evidence only**, not proof of successful on-air reception. Reports carry
`manifest_sha256` and each input WAV's checksum so benchmark claims remain
traceable to exact bytes.

- Implementation: [`sigid/conformance.py`](../sigid/conformance.py)
- Result/API contract: [`DECODER_API_SPEC.md`](DECODER_API_SPEC.md)
- Mode agent instructions: [`DECODER_AGENT_PLAYBOOK.md`](DECODER_AGENT_PLAYBOOK.md)
- Evidence/provenance rules: [`DECODER_EVIDENCE_INDEX.md`](DECODER_EVIDENCE_INDEX.md)
- Runnable sample manifest: [CW synthetic manifest](../tests/fixtures/decoder_conformance/cw/manifest.json)
- Manifest editor schema: [`schemas/decoder_conformance_manifest.schema.json`](schemas/decoder_conformance_manifest.schema.json)

## Run the existing first-party registry

From the repository root (with dependencies installed):

```bash
python -m sigid.conformance tests/fixtures/decoder_conformance/cw/manifest.json \
  --json-out /tmp/cw-conformance.json
# Or, after installing this package:
sig-id-conformance tests/fixtures/decoder_conformance/cw/manifest.json
```

Exit codes: `0` = all graded cases pass, `1` = a graded case fails,
`2` = invalid manifest, unusable input, or missing registered decoder.
`--report-only` writes the report and exits zero for scored failures, **not**
malformed suites. The report is JSON and includes the source WAV SHA-256,
original sample positions, decoder version, per-case metrics, and disjoint
provenance summaries. Treat it as a diagnostic artifact, not a protocol label.

## Run a mode agent's not-yet-registered decoder

**Do not edit the shared registry to run conformance.** An agent can inject
their own independent mode implementation:

```python
from sigid.conformance import evaluate_suite
from sigid.libsigdecode import DecoderLibrary
from sigid.libsigdecode.rtty import RTTYDecoder

report = evaluate_suite(
    "tests/fixtures/decoder_conformance/rtty/manifest.json",
    library=DecoderLibrary((RTTYDecoder(),)),
)
assert report["passed"], report["cases"]
```

A decoder requiring IQ cannot be secretly tested as mono audio. This runner
currently accepts **mono, uncompressed PCM16 WAV**, within the library's
120-second window limit. IQ, streaming, and multi-channel inputs require a
separate integration proposal and type-specific fixture loader.

## Manifest format

`schema_version: 1`, one `suite_id` lowercase slug, and an array of
up to 256 `fixtures`. Each fixture has:

- `id`: unique lowercase slug; `decoder_id`: implemented mode ID.
- `provenance.kind`: exactly `synthetic`, `reviewed_air`, or
  `unreviewed_air`; provide a nonempty `description`. For actual
  independently reviewed air captures, also record `reviewer`,
  `truth_basis`, and `source_reference`. These fields document a
  claimed review; the program **cannot** independently verify honesty.
- `audio.path`: relative WAV path under the manifest directory, no
  traversal or external symlink targets; `audio.sha256`: exact lowercase
  SHA-256 of the complete WAV file. Optional `file_start_sample` (absolute
  capture index of file sample zero), `window_offset_samples` (offset inside
  the file) and `sample_count` (exact cropped sample count).
- `carrier_hz`: optional **audio** carrier frequency; never RF frequency.
- `expected.kind`: `positive`, `negative`, or `observe`.

**Positive** expected fields include nonempty `text` and/or a nonempty
`frames` array. Optional `max_character_error_rate` (default 0,
range [0,1]) and `allowed_statuses` (default `["decoded"]`, may also
include `"partial"`). Frame expectations use `payload_hex`, optional
`bit_length`, and optional `integrity` entries of `kind`,
`algorithm`, and `outcome`. Actual frames must match one-to-one:
unexpected extra frames fail the fixture. Requested integrity checks must
match exactly; extra unspecified checks do not establish correctness.

**Negative** controls require `no_decode` and empty decoded text. A
`no_decode` result may still retain tentative frame evidence (e.g.
corrupted candidates) without claiming a successful decode.

**Observe** is the *only* permitted expectation for `unreviewed_air`.
Its output may be inspected, but it has no pass/fail truth score and cannot
improve the conformance gate. An **observation-only suite never gets a passing
quality gate**: the report says `quality_gate="ungraded"` and returns exit code
1 unless `--report-only` is used. Exceptions and invalid results still fail.
Positive/negative ground truth is allowed only for independently generated
or explicitly reviewed captures.

The reproducible CW manifest in this repository demonstrates the complete
machine-readable format. Its positive vectors use independently generated
textbook Morse with deterministic adjacent-tone interference and noise;
negative vectors cover constant carrier, noise-only and unrelated hopping
tones. No OnAir, Fldigi, or WSJT-X implementation is imported or copied.

## Measurements and interpretation

| Metric | Interpretation |
|---|---|
| `character_errors`, `character_error_rate` | Levenshtein edits / reference characters, on known text only |
| `exact_text_match` | Whether decoded text matches the pinned reference |
| `frames_matched`, `frames_extra`, `frame_recall` | One-to-one frame/bit-length/integrity matching, not a calibrated confidence |
| `negative_abstention_rate` | Fraction of negative controls explicitly returning `no_decode` |
| `negative_false_positive_rate` | Fraction of negative controls returning `partial` or `decoded`, including empty positive results |
| `decode_elapsed_ms`, `realtime_factor` | Per-case wall-clock decode cost / audio duration; affected by host and current load |
| `decode_ms_median`, `decode_ms_p95` | Descriptive latency for the given corpus and host |

`by_provenance` reports synthetic, reviewed-air, and observation-only air
**separately**. `by_mode` further separates results by decoder and
provenance. `synthetic_only=true` is always called out. Raw elapsed time
from different hosts should not be compared as though it were a calibrated
receiver benchmark. No score measures SNR threshold yet; those experiments
must pin separate noisy WAV fixtures (with actual noise level documented)
and compare failure and abstention rates across those fixtures.

Agent-owned suites should include realistic wrong-speed, wrong-polarity,
off-frequency, noise/adjacent-signal and malformed-frame controls. Negative
controls are as important as positive examples. Do not claim a new decoder
validated until representative independent real-air ground truth and
documented metrics exist. A black-box decoder's agreement alone is not
independent ground truth.

## Fixture ownership and maintenance

- One agent owns each decoder-specific manifest, WAVs, and source notes
  under `tests/fixtures/decoder_conformance/<mode>/`.
- Prefer small, independently generated/reviewed WAV clips with SHA-256.
  Large/licensing-sensitive real captures should live outside Git and be
  referenced in a separate opt-in local manifest. **Do not publish private
  receive audio or message contents without review and permission.**
- Submit the manifest, generator script or independent source reference,
  positive and negative expectations, conformance report, and separate
  `tests/test_decode_<mode>.py` through the mode PR.
- The integrator owns harness/schema changes, CI, shared registration and
  the decision to promote `experimental` to `benchmarked` / `validated`.
- Never automatically update reference text/hash values just because a
  decoder's output changed. That would convert regressions into fabricated
  success. Fixture updates require a human review of independent truth.

Regenerate **only the bundled synthetic CW reference** (intentionally
replaces its pinned WAVs and hashes) with:

```bash
python tools/build_decoder_conformance_cw_fixtures.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_decoder_conformance.py
```

Existing CI and local tests score the pinned sample suite. Additional mode
branches can call `evaluate_suite` directly without touching app or API code.
