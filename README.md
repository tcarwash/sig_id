# Signal ID workbench

A digital radio receive, identification and dataset workbench is being built in
staged releases. The full plan is in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md);
the current engineering checkpoint is [NEXT.md](NEXT.md).

## Current state

Current code includes WAV upload/playback with a browser waterfall, manual
time/frequency selection and label export; blockwise PCM replay; a deterministic
2-FSK fixture generator; SQLite capture, region and label history; and a
receive-only Fldigi XML-RPC probe. Local device capture and the Fldigi path from
audio input to recovered text are now represented in the workbench: when a running
Fldigi XML-RPC service is available, users can select a catalog mode and view its
receive-text output. Audio routing remains an external setup and has not yet been
validated end to end, so the UI reports it as unverified rather than claiming a
successful decode.

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

Enable Fldigi's XML-RPC server on loopback before launching the workbench. Set
`SIG_ID_FLDIGI_URL` if it listens on a different local URL. The decoder panel lists
the installed catalog, selects a receive mode, and polls incremental receive text.
Route the radio passband audio to Fldigi's input using the audio system configuration;
the application does not silently create an audio route or claim decode validation.
When starting Fldigi from a terminal, its XML-RPC method allow-list can be limited to
the calls the application uses:

```sh
fldigi --xmlrpc-server-address 127.0.0.1 \
  --xmlrpc-allow='^(fldigi\.(name|version)|modem\.(get_names|get_name|set_by_name)|text\.(get_rx|get_rx_length))$'
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
