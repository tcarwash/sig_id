# Signal ID workbench

A digital radio receive, identification and dataset workbench is being built in
staged releases. The full plan is in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md);
the current engineering checkpoint is [NEXT.md](NEXT.md).

## Current state

The first code slice contains a receive-only Fldigi XML-RPC client, a catalog
capability probe, and unit tests. It does not yet provide an audio stream, waterfall,
decoder UI, or audio-to-decoded-text proof. Catalog presence is reported as pending
validation and must not be treated as supported decoding.

## Development

Requires Python 3.11 or newer and pytest. From the repository root:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
```

The installed Fldigi process can be inspected with:

```sh
sig-id-probe-fldigi --url http://127.0.0.1:7362/RPC2
```

Restrict its XML-RPC method allow-list to the receive and inspection methods needed
by the application. Do not expose Fldigi's control endpoint outside loopback.

## Data sources

Generated waveforms, saved captures, and curated external reference examples will
remain distinguishable in provenance and label confidence. See the dataset and
Signal Identification Wiki guidance in [DESIGN.md](DESIGN.md).
