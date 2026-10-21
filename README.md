# Signal ID workbench

A digital radio receive, identification and dataset workbench is being built in
staged releases. The full plan is in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md);
the current engineering checkpoint is [NEXT.md](NEXT.md).

## Current state

Current code includes WAV upload/playback with a fixed 0–3 kHz browser waterfall
(4096-point FFT at the default 48 kHz capture rate, about 11.7 Hz per bin), manual
time/frequency selection and label export; blockwise PCM replay; deterministic
2-FSK and generic MFSK fixture generators; verified synthetic import into the SQLite
capture/label dataset; and a receive-only Fldigi XML-RPC probe. Local device capture
and the Fldigi path from audio input to recovered text are represented in the
workbench: when a running Fldigi XML-RPC service is available, users can select a catalog mode and view its
receive-text output. RF-free loopback tests decode BPSK31, RTTY and THOR16 payloads
through temporary virtual audio into the workbench API. Routing actual radio input
to Fldigi remains an external setup and is not yet validated. The workbench also
offers bounded Fldigi mode trials for live candidates. The “Auto-try accepted model IDs”
checkbox opts into starting a sweep only after a track persists and the loaded model
accepts a method; abstentions never start trials. Mode changes are receive-only and
recovered text remains tentative until reviewed. Saved recordings
have a manual, protocol-independent 2-FSK demodulator that exposes uncoded bits,
symbol timing and soft tone metrics; symbol rate, tone mapping and timing offset must
be supplied by the operator. A corresponding M-FSK analyzer returns hard tone indices,
soft margins and natural-binary bits only for power-of-two tone counts. BPSK analysis
returns both global polarity hypotheses, while QPSK analysis returns four possible
90-degree rotation hypotheses because an unknown capture has no inherent absolute
phase reference. These are protocol-independent symbol/bit observations, not complete
protocol decoders.

## Development

Requires Python 3.11 or newer and pytest. From the repository root:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
sig-id-serve
```

The opt-in, RF-free Fldigi audio loopback tests exercise BPSK31, RTTY and THOR16 using
isolated profiles and a temporary PulseAudio/PipeWire null sink. They require a
configured `~/.fldigi`, `xvfb-run`, and `pactl`:

```sh
SIGID_TEST_FLDIGI_AUDIO=1 pytest tests/test_fldigi_audio_integration.py
```

To audit selection of every mode in an isolated Fldigi profile (without changing
the operator's running Fldigi mode), run:

```sh
SIGID_TEST_FLDIGI_CATALOG=1 pytest -s tests/test_fldigi_audio_integration.py -k every_installed_catalog_entry
```

The reference Fldigi 4.2.13 selection results are in
[`reports/fldigi-4.2.13-catalog-selection.json`](reports/fldigi-4.2.13-catalog-selection.json).
This tests mode selection only; it does not establish audio routing or decode support.

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
Completed Fldigi WEFAX images can also be imported into the workbench's managed
receive-products store and viewed in its received-image gallery.
Route the radio passband audio to Fldigi's input using the audio system configuration;
the application does not silently create an audio route or claim decode validation.
When starting Fldigi from a terminal, its XML-RPC method allow-list can be limited to
the calls the application uses:

```sh
fldigi --xmlrpc-server-address 127.0.0.1 \
  --xmlrpc-allow='^(fldigi\.(name|version)|modem\.(get_names|get_name|set_by_name)|text\.(get_rx|get_rx_length)|wefax\.get_received_file)$'
```

Restrict its XML-RPC method allow-list to the receive and inspection methods needed
by the application. Do not expose Fldigi's control endpoint outside loopback.

Create a clean binary-FSK fixture with a sidecar manifest containing the exact
reference bits and modulation parameters:

```sh
sig-id-generate-2fsk data/fixtures/fsk.wav 001101101001
```

Generate a protocol-independent 4-FSK/MFSK fixture with exact source symbols,
binary tone mapping, modulation-method labels, and no invented protocol label:

```sh
sig-id-generate-mfsk data/fixtures/mfsk4.wav 0,1,3,2 --symbol-rate 100
```

Generate an uncoded BPSK fixture with its exact reference bits and carrier parameters:

```sh
sig-id-generate-bpsk data/fixtures/bpsk.wav 001101101001 --carrier-hz 1500
```

Generate an uncoded Gray-mapped QPSK fixture with its exact reference bits and carrier
parameters:

```sh
sig-id-generate-qpsk data/fixtures/qpsk.wav 001101101001 --carrier-hz 1500
```

Register the waveform and validated sidecar in the local dataset from the command
line, or choose both files in the workbench's **Synthetic Dataset** panel:

```sh
sig-id-import-synthetic data/fixtures/mfsk4.wav
```

The importer verifies that the WAV exactly matches a deterministic 2-FSK, M-FSK,
BPSK or QPSK generator manifest, then stores a full-capture modulation label and the source
manifest. It does not invent a named protocol.

Create a deterministic noisy child from an imported clean source. Import the clean
source first; the child manifest records its hashes and recipe, and the importer checks
the parent lineage and regenerates the samples before accepting inherited labels:

```sh
sig-id-augment-awgn data/fixtures/bpsk.wav data/fixtures/bpsk-10db.wav --snr-db 10 --seed 42
sig-id-import-synthetic data/fixtures/bpsk-10db.wav
```

Build a small resumable corpus across six modulation methods. Each clean source and
its AWGN descendants share a `lineage_group` in the index so they can be kept in one
train/validation/test split:

```sh
sig-id-build-corpus --output-dir data/corpus --examples-per-mode 4 --snr-db 0 10 20 --seed 42
sig-id-split-corpus --corpus data/corpus/corpus.json --output data/corpus/splits.json --seed 42
sig-id-train-method-model --split data/corpus/splits.json --output-model data/models/method.npz
```

This creates 96 labeled modulation captures (six methods × four source examples ×
one clean plus three SNR variants) and four independent noise-only negative captures.
The negatives retain null modulation/protocol labels and are used for rejection
calibration/evaluation, not as a seventh modulation class. The index and generated
audio stay in `data/corpus`; imported
captures and labels are stored under `SIG_ID_DATA_DIR` or the normal application data
directory. Re-running the same command resumes without duplicating captures; changing
the recipe requires a new output directory.
The split command validates each dataset capture and label, requires at least three
independent source groups per method and for noise negatives, and assigns every clean
waveform and all derived impairments to one train, validation or test split.
The training command fits a small NumPy logistic-regression baseline, calibrates
separate method and family rejection thresholds on validation groups, evaluates once
on the held-out test groups, and writes weight, metadata and evaluation files. To load
the exported proposal model in the workbench, set `SIG_ID_MODEL_PATH` to the `.npz`
file before starting `sig-id-serve`. Proposals never save labels automatically.
The model uses a zero-noise-false-positive validation constraint and reports held-out
noise false positives. If that constraint and the known-mode precision target cannot
both be met, it abstains on every example; small corpora can therefore yield zero useful
coverage. This does not validate rejection of unseen modulated families, and real-radio
performance remains unvalidated.

Replay it blockwise from Python with `sigid.wav.WavSource`; blocks carry exact sample
offsets and normalized mono samples. This is a modulation fixture, not a named
protocol waveform.

## Data sources

Generated waveforms, saved captures, and curated external reference examples will
remain distinguishable in provenance and label confidence. See the dataset and
Signal Identification Wiki guidance in [DESIGN.md](DESIGN.md).

Create a bounded metadata catalog from [Signal Identification Wiki](https://www.sigidwiki.com/wiki/Signal_Identification_Guide) pages without
downloading their audio or images:

```sh
sig-id-import-sigidwiki --output data/references/sigidwiki.json --limit-per-category 25
```

The catalog preserves the page URL, revision, source-reported fields and media links.
Wiki labels remain unreviewed references, not accepted training labels; media stays
link-only until reuse rights are reviewed per asset. Use `--categories Signal` to
catalog only identified entries, or add a different category explicitly.
