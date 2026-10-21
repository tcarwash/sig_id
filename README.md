# Signal ID workbench

A digital radio receive, identification and dataset workbench is being built in
staged releases. The full plan is in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md);
the current engineering checkpoint is [NEXT.md](NEXT.md).

## Current state

Current code includes WAV upload/playback with a browser waterfall, manual
time/frequency selection and label export; blockwise PCM replay; a deterministic
2-FSK fixture generator; SQLite capture, region and label history; and a
receive-only Fldigi XML-RPC probe. Local device capture and the Fldigi path from
audio input to recovered text are still pending. Catalog presence is reported as pending
validation and must not be treated as supported decoding.

## Development

Requires Python 3.11 or newer and pytest. From the repository root:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
sig-id-serve
```

Then open <http://127.0.0.1:8010>, choose a PCM WAV, and mark a signal region on
the waterfall. The local service stores audio under `~/.local/share/sig-id` by
default. Set `SIG_ID_DATA_DIR` before starting it to use another data directory.

The installed Fldigi process can be inspected with:

```sh
sig-id-probe-fldigi --url http://127.0.0.1:7362/RPC2
```

Restrict its XML-RPC method allow-list to the receive and inspection methods needed
by the application. Do not expose Fldigi's control endpoint outside loopback.

Create a clean binary-FSK fixture with a sidecar manifest containing the exact
reference bits and modulation parameters:

```sh
sig-id-generate-2fsk data/fixtures/fsk.wav 001101101001
```

Replay it blockwise from Python with `sigid.wav.WavSource`; blocks carry exact sample
offsets and normalized mono samples. This is a modulation fixture, not a named
protocol waveform.

## Data sources

Generated waveforms, saved captures, and curated external reference examples will
remain distinguishable in provenance and label confidence. See the dataset and
Signal Identification Wiki guidance in [DESIGN.md](DESIGN.md).
