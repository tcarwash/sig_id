# Decoder evidence and protocol reference index

This index distinguishes our project observations from published protocol
documentation, and separates **primary specifications** from reference
implementation behavior. It is not permission to copy GPL code.
Agents MUST read this with [API contract](DECODER_API_SPEC.md),
[agent playbook](DECODER_AGENT_PLAYBOOK.md), and
[machine work queue](decoder-work-items.json).

## Internal evidence and implementation baselines

| Source | What is actually established |
|---|---|
| [LIBSIGDECODE](LIBSIGDECODE.md) | Independent implementation roadmap; current experimental CW module; archived false-decode/abstention finding |
| [IDENTIFICATION_CHAIN](IDENTIFICATION_CHAIN.md) | CV versus heuristic detection, ranked uncalibrated hypotheses, independent DSP, conservative external decoder trials, and what is *not* verified |
| [CROWDED_AUDIO_BENCHMARK](CROWDED_AUDIO_BENCHMARK.md) | Synthetic separation checks and 40m FT8-like bands; **no hand-reviewed real FT8 message reference** |
| [OnAir event bridge](ONAIR_EVENT_BRIDGE.md) | Optional OnAir event API only; this is not an in-project OnAir decoder or WAV replay |
| [Reference docs](README-REFERENCE.md) | Other project workflows, dataset/review and DSP notes |
| [CW first-party tests](../tests/test_libsigdecode.py) | Golden synthetic Morse, noise/single tone rejection, FSK false-positive abstention, and saved-capture API |
| [CW tone detection tests](../tests/test_narrow_tone_detection.py) | Narrow tone proposal logic; does **not** decode Morse |
| [Independent timing estimator](../sigid/timing_evidence.py) | Frequency-transition timing proposals; not a protocol sync or decode |
| [40m CV proposals](../reports/40m-ft8-cv-boxes-2026-11-21.json) | Unreviewed detector boxes; never treat as station labels |

**Data caution:** large archival WAVs live on the development machine and
are not part of this Git repository. Filenames and SHA-256 where recorded
are in the benchmark docs. Do not assert on-air accuracy from synthetic
fixtures or a model's guess. Real truth requires independently reviewed
recordings and a documented sampling/evaluation protocol.

## Published protocol documents (implementation inputs)

These are external links to read and cite in a new decoder's module doc,
not files copied into this repository. Check revision and rights before
redistributing whole tables, figures or text. Several are user guides or
community descriptions and may not completely specify the bits on air.
When the primary specification is incomplete, mark the work item as
research-blocked rather than filling gaps from copied implementation code.

| Ref ID | Target modes | Source and evidentiary status |
|---|---|---|
| morse_itu | CW, QRSS/DFCW alphabet | [ITU-R M.1677-1 International Morse code](https://www.itu.int/rec/R-REC-M.1677-1-200910-I/) — primary standard for Morse; signal detection and timing tolerance still need design/tests |
| ita2_itu | RTTY | [ITU-T S.1 International Telegraph Alphabet No. 2](https://www.itu.int/rec/T-REC-S.1-199303-I) — primary code alphabet; not a complete on-air receiver framing standard |
| psk31_arrl | BPSK31 / QPSK31 | [ARRL PSK31 specification](https://www.arrl.org/psk31-spec) — published original protocol reference; mode variant details must be separately checked |
| ax25_tapr | AX.25 / APRS / HF packet | [TAPR AX.25 v2.2](https://tapr.org/pdf/AX25.2.2.pdf) — primary link-layer reference, not the Bell 202 analog waveform |
| aprs10 | APRS payloads | [APRS protocol v1.0](https://www.aprs.org/doc/APRS101.PDF) — original payload/spec document |
| aprs11 | APRS payloads | [APRS specification addendum 1.1](https://www.aprs.org/aprs11.html) — updates/corrections |
| ft4ft8_qex | FT8 / FT4 | [Taylor/Franke et al., FT4 and FT8 protocols (QEX)](https://www.k0nr.com/wordpress/wp-content/uploads/2025/03/Protocols-FT4-FT8-QEX.pdf) — authored protocol description; verify version/CRC/FEC test vectors independently |
| wsjtx_guide | FT8/FT4, WSPR, JT family, Q65, FST4W, MSK144 | [WSJT-X User Guide](https://wsjt.sourceforge.io/wsjtx-main_en.html) — official operating/protocol overview; **not always a complete implementer spec** |
| olivia_arrl | Olivia (Contestia related but not identical) | [Olivia draft protocol specification](https://www.arrl.org/olivia) — developer-authored draft; Contestia requires additional independent references |
| sstv_modes | SSTV | [SSTV mode specifications](https://sstv.org.uk/docs/modes/) — secondary/community reference; validate VIS, timing and color mapping against real test signals |
| m17_spec | M17 | [M17 Project published specification](https://github.com/M17-Project/M17_spec) — primary standard; check exact revision |
| freedv_spec | FreeDV | [FreeDV published specifications](https://freedv.org/freedv-specification/) — project specification; individual waveforms/codecs require separate licensing/capability review |

## External *reference applications* — comparisons, not source imports

- [OnAir mode catalog](https://github.com/adswill/OnAir/blob/main/MODES.md):
  **a coverage wishlist**, not proof that all listed items are full or
  independently verified decoders. The ham sections mix decoders,
  classifiers, receive-only monitors and experimental modes.
- [OnAir](https://github.com/adswill/OnAir): external decoded-event adapter
  can provide optional comparison results; no dependency for local decode.
- [Fldigi](https://www.w1hkj.org/): external replay / comparison for RTTY,
  PSK31 and other keyboard modes; its result does not substitute for an
  independently authored local decoder.
- [WSJT-X](https://wsjt.sourceforge.io/wsjtx-main_en.html): external
  comparison for weak-signal modes, using exact UTC slot/capture alignment.
- [M17 reference implementations](https://github.com/M17-Project):
  comparison only; the M17 spec, not source code, is the design basis.

## Per-decoder evidence records

A mode-specific agent should add one file at
docs/decoder-evidence/<mode>.md with:

1. Decoder spec revision and source links; explicitly note uncovered fields.
2. Allowed inputs, output type, expected signal family, variant boundaries.
3. Annotated synthetic/golden vectors: generation method, immutable
   SHA-256, known symbols/bits/payload and negative controls.
4. Reviewed real captures: provenance, consent if relevant, SHA-256, sample
   rate, exact time/frequency window, independent transcript/packet truth.
5. Algorithm notes written from the protocol, with no copied implementation.
6. Accuracy/false-positive/abstention metrics and remaining limitations.
7. Test commands, algorithm version and ownership/review status.
8. Any deliberate external-decoder black-box comparisons; distinguish
   agreement from independent ground truth.

If no verified primary document or real-world reviewed evidence exists,
record the gap clearly in both the evidence file and work queue.
