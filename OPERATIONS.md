# Run and maintain the local workbench

## Linux Docker Compose

```sh
docker compose up --build -d
docker compose ps
```

Open <http://127.0.0.1:8010>. If a native workbench already uses that port, run:

```sh
SIG_ID_PORT=8012 docker compose up --build -d
```

Compose uses Linux host networking so a host Fldigi XML-RPC listener on
`127.0.0.1:7362` is reachable without changing its binding. The web listener also
binds to loopback by default. Copy `.env.example` to `.env` to retain a different
port or decoder URL. Fldigi runs separately; the reference backend tested by this
project is Fldigi 4.2.13. The image includes the workbench and DSP dependencies.

The service runs as UID/GID 10001 with a read-only root filesystem and a writable
named volume mounted at `/data`. Captures, annotations, received products and any
models generated inside `/data` persist across container recreation. `docker compose
down` retains that volume. `down --volumes` deletes its data.

The container does not receive a host audio device automatically. For direct local
radio-device capture, use the native setup in README and the audio extra. Container
validation uses a separately mounted paced synthetic device; it does not establish
host ALSA/PipeWire routing. Fldigi's existing host radio input remains externally routed.

An optional model must be readable inside the volume. For example, generate/train
artifacts with the documented CLI commands through `docker compose exec workbench`,
using `/data` for their output paths, then set `SIG_ID_MODEL_PATH=/data/models/method.npz`
in `.env` and recreate the service. Leave this setting blank for manual operation.
A missing or invalid model leaves capture and manual analysis usable.

### Optional YOLO signal detector

CV inference is off unless `SIG_ID_CV_MODEL_PATH` points to a trained detector. The
base image does not install the CV dependency. To build the optional runtime, set
`SIG_ID_INSTALL_CV=1` and `SIG_ID_CV_MODEL_PATH=/data/models/cv/signal.pt` in `.env`,
then run `docker compose up --build -d`. Leave both settings blank or set the install
flag to `0` to keep the base image lean. The CV button appears in the saved-capture
signal panel when a model loads. Its boxes are provisional; select one to enter the
existing classifier and manual label flow. Decoder execution remains explicit.

Training and dataset export can run on the host with the optional extra:

```sh
python -m pip install -e '.[cv]'
sig-id-export-cv-dataset --data-dir ~/.local/share/sig-id --output data/cv/signal-dataset
sig-id-train-cv-detector --dataset data/cv/signal-dataset --output-model data/models/cv/signal.pt --device cpu
```

The exporter uses current accepted region geometry only inside fresh, complete
`Signal detection` scene reviews. Windows with uncertain or ignored subregions are
excluded. It groups parent/scene component captures and duplicate waveforms before
assigning train/validation/test splits. At least three independent groups are
required; each split must contain signal and completely reviewed no-signal windows.
The exported `export-report.json` and `windows.jsonl` record counts, hashes,
source windows and preprocessing version. Review a sufficient range of recordings
before training; the CLI does not turn unreviewed parts of a capture into negative
examples. Ultralytics downloads its selected base checkpoint when needed. The optional
Ultralytics package is AGPL-3.0-or-later; check its [license terms](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
before distributing an image that includes it.

### AMD ROCm

For an AMD GPU, use the separate ROCm build and Compose override:

```sh
docker compose -f compose.yaml -f compose.rocm.yaml up --build -d
```

The ROCm image uses AMD's PyTorch/ROCm base image and includes the optional CV runtime.
Ultralytics is pinned to the version recorded in `uv.lock` so rebuilds retain the
same detector API.
It passes `/dev/kfd` and `/dev/dri` through to the container. Set the video and render
group IDs in `.env` to match `stat -c '%g' /dev/kfd` and
`stat -c '%g' /dev/dri/renderD128` on the Docker host; the example values are common
defaults, not universal. The host needs a working ROCm driver and a GPU supported by
the chosen AMD image. Training can then use `--device 0`. The image is large because
the AMD runtime and PyTorch are included; changing `SIG_ID_ROCM_PYTORCH_IMAGE` should
be treated as a separate ROCm/PyTorch compatibility change.

After saving reviewed capture boxes and complete scene reviews in the workbench, export
and train inside the ROCm container:

```sh
docker compose -f compose.yaml -f compose.rocm.yaml exec workbench \
  sig-id-export-cv-dataset --data-dir /data --output /data/cv/signal-dataset
docker compose -f compose.yaml -f compose.rocm.yaml exec workbench \
  sig-id-train-cv-detector --dataset /data/cv/signal-dataset \
  --output-model /data/models/cv/signal.pt --device 0
```

Evaluate the held-out detection-to-classifier flow with the trained detector and
the existing method classifier. This reports box precision/recall, classifier
coverage/accuracy on matched detections, abstentions, and known-label accepts on
unmatched proposals. It also compares YOLO boxes with the existing heuristic
spectral-track detector over the same reviewed test windows:

```sh
docker compose -f compose.yaml -f compose.rocm.yaml exec workbench \
  sig-id-evaluate-cv-flow --dataset /data/cv/signal-dataset --data-dir /data \
  --detector /data/models/cv/signal.pt \
  --classifier /data/models/method.npz \
  --output /data/cv/cv-flow-evaluation.json
```

For direct synthetic 2FSK, BPSK, QPSK, OOK, and bit-mapped MFSK captures, the report
also measures bit recovery from matched YOLO time crops using generator settings
from the hashed synthetic manifest. BPSK polarity and QPSK rotation are resolved
against the known reference payload, so this is an oracle-assisted demodulation
measurement, not classifier-driven decoder selection. Scene mixtures, augmented
noise, real recordings, protocol framing, and interoperability are outside this
measurement.

Completed-capture CV scans use 10-second windows at a 5-second stride to reduce
misses at window boundaries. Cross-window proposals with 2D time/frequency IoU of
at least 0.30 are suppressed in favor of the higher-confidence box; the surviving
proposal retains the contributing window coordinates in its review evidence.
This increases inference work by roughly 2x compared with non-overlapping scans.

Then set `SIG_ID_CV_MODEL_PATH=/data/models/cv/signal.pt` in `.env` and recreate the
ROCm service. Keep the adjacent `.pt.json` metadata file beside the weights.

### Similarity search for reviewed regions

Similarity search is optional and uses normalized YOLO image embeddings with an
exact cosine scan over a compact local NumPy index. It is intended to retrieve
reviewed examples when a selection is marginal or unfamiliar; the nearest match is
a candidate for human review, not an identity claim. Build or rebuild the index
after changing the detector weights or reviewed labels:

```sh
docker compose -f compose.yaml -f compose.rocm.yaml exec workbench \
  sig-id-build-cv-embedding-index --data-dir /data \
  --model /data/models/cv/signal.pt \
  --output /data/cv/embeddings.npz
```

Set `SIG_ID_CV_EMBEDDING_INDEX=/data/cv/embeddings.npz` in `.env` and recreate the
service. The index is bound to the detector model hash and is rejected if it was
built with different weights. This simple exact scan avoids operating a separate
vector database while the catalog is small; revisit approximate indexing if the
catalog grows enough that measured query latency warrants it.

The optional **Auto ID with Laya** action sends the selected region's derived spectral
and envelope feature vector to a Laya System One endpoint; it does not send the WAV
audio or automatically save a label. Set `LAYA_ENDPOINT` to the full endpoint URL and,
when required, `LAYA_API_KEY` in `.env`, then recreate the service. The button appears
for a valid saved-capture selection when the endpoint is configured. Review the
suggested modulation method before saving it as a label. The request also includes a
small bundled set of labeled synthetic examples selected from a lineage-safe training
split, including noise-only negatives. Set `SIG_ID_LAYA_REFERENCE_SET` only to replace
that bundled reference file with another validated train-only set. Validation and test
examples must never be included in the prompt.

## Update

```sh
git pull
docker compose up --build -d
docker compose ps
```

An update briefly stops the web service. Graceful shutdown finalizes any outstanding
recording; finish an important receive session before updating. Keep a data backup
before changing versions. A database with an unsupported schema is rejected instead
of being silently rewritten.

The Docker base image is pinned by digest. Runtime and build Python dependencies are
pinned with hashes in `requirements/`. Refresh them deliberately with the commands
recorded at the top of each file, then rebuild and rerun validation. Debian system
libraries are installed from the base distribution's repositories during the build.

## Backup

This stops the service while copying the SQLite index and lossless audio together.
The generated directory contains private recordings and all label revisions.

```sh
backup_path="backups/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$backup_path"
docker compose stop workbench
docker compose cp workbench:/data/. "$backup_path/"
docker compose start workbench
```

Use the same project name and port environment used when starting the service.
Back up the entire data directory, including the SQLite file, captures and sidecars;
the JSONL label export is useful for training interchange but is not a full backup.

## Restore into a fresh project

Choose a new project name and an unused port so the current dataset remains available
for comparison. Extraction runs as the service user, preserving writable ownership.

```sh
restore_project="sigid-restore-$(date -u +%Y%m%dT%H%M%SZ)"
tar -C "$backup_path" -cf - . | SIG_ID_PORT=8012 docker compose -p "$restore_project" \
  run --rm -T --no-deps workbench tar --no-same-owner -xf - -C /data
SIG_ID_PORT=8012 docker compose -p "$restore_project" up -d --wait
```

Open the restored workspace and verify recordings, labels, revision history and audio
playback. A writer or metadata failure can leave an explicitly reported WAV path that
is not indexed; retain that file and import it through Open recording after storage
is working. Sudden power loss recovery and the eight-hour hardware soak remain release
hardening work.

## Repeatable container checks

```sh
.venv/bin/python tools/check_compose.py --output /tmp/sigid-compose-validation.json
```

This creates disposable projects on unused loopback ports, builds the image, checks
its non-root/read-only runtime, uploads and demodulates a fixture, revises a label,
restarts the service and compares audio hashes. It also stops a synthetic-input
container while recording and verifies that the saved recording appears in the shared
dataset, then tests backup/restore into a fresh project. Only its own generated test
containers and volumes are removed afterward.

To include the browser workflow, install Playwright as described in
[tests/GUI.md](tests/GUI.md) and set `SIGID_PLAYWRIGHT_MODULE` when running the command.
The committed result is [the Compose validation report](reports/compose-validation-2026-10-21.json).
