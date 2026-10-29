# Signal ID workbench

Signal ID is a local workbench for receiving, inspecting, labeling, and studying radio audio. It combines a live waterfall and recording review with signal annotations, protocol-independent symbol analysis, optional classifier and spectrogram detector workflows, and decoder integrations.

The mode catalog is useful for labeling; catalog entries do not imply automatic recognition or decoding support. Automated results are proposals for operator review. Synthetic and loopback evidence is kept distinct from real radio reception.

![Signal ID waterfall with a weak synthetic FSK signal in noise](reports/gui-overhaul-desktop.png)

## Run with Docker Compose

On Linux, start the workbench with:

```sh
docker compose up --build -d
```

Open <http://127.0.0.1:8010>. If that port is in use, run `SIG_ID_PORT=8012 docker compose up --build -d` and open port 8012. Recordings and labels persist in a named volume. The default listener is local.

## Use the workbench

Open a recording or start live capture, then inspect the waterfall. Select a candidate or drag across a region to open its inspector. Classify and Annotate workflows keep identity, decode, raw-symbol, and box review actions together. Saving a label is explicit; selecting a candidate does not change decoder settings or save an annotation.

For local development, install Python 3.11 or newer and run:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev,audio]'
sig-id-serve
```

Then open <http://127.0.0.1:8010>. Local audio capture also requires the platform PortAudio runtime and an input selected in Settings.

## Project guides

- [Operations](OPERATIONS.md): updates, model setup, backup and restore.
- [Browser verification](tests/GUI.md): isolated UI workflow checks and their limits.
- [Distributed collection](DISTRIBUTED_COLLECTION_PLAN.md): contribution and peer review design.
- [Engineering and dataset reference](docs/README-REFERENCE.md): detailed features, research notes, and reproducible workflows.
- [Implementation plan](IMPLEMENTATION_PLAN.md) and [current checkpoint](NEXT.md).

The workbench is under active development. Decoder and model availability depends on local configuration; radio hardware routing and field performance require separate validation.
