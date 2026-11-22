# Mode-specific evidence dossiers

Each independently implemented decoder should have one file named
<mode>.md. The [evidence index](../DECODER_EVIDENCE_INDEX.md) specifies
minimum fields; the [API contract](../DECODER_API_SPEC.md) defines runtime
behavior. Empty or speculative dossiers must never count as validation.

A recommended dossier includes: version of the public protocol;
linked source with scope/gaps; intended input sample format; waveform
and payload vectors with SHA-256 and independently known answers;
fuzz/corruption and negative-control vectors; exact real capture
identity/time/frequency/ground truth; metrics and abstention; regression
commands; external comparison provenance; reviewer and limitations.
