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
