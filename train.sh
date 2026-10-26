cd /storage5/dev/sig_id

/home/tyler/dev/sig_id/.venv-rocm/bin/python - <<'PY'
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from sigid.cv_dataset import export_cv_dataset
from sigid.cv_training import train_cv_detector

root = Path.cwd()
pointer_path = root / "data/models/active-models.json"
active = json.loads(pointer_path.read_text())
base_model = Path(active["cv_detector"]).resolve()

meta_path = base_model.with_suffix(base_model.suffix + ".json")
model_meta = json.loads(meta_path.read_text())
old_dataset = Path(model_meta["dataset_path"]).expanduser()
old_report = json.loads((old_dataset / "export-report.json").read_text())
source_dirs = [
    Path(path).expanduser().resolve()
    for path in old_report.get("source_data_dirs", [])
    if (Path(path).expanduser() / "dataset.sqlite3").is_file()
]
app_store = Path.home() / ".local/share/sig-id"
if (app_store / "dataset.sqlite3").is_file() and app_store.resolve() not in source_dirs:
    source_dirs.append(app_store.resolve())
if not source_dirs:
    raise SystemExit("No reviewed CV source stores found")

release = root / "data/models/releases" / (
    "cv-button-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
)
release.mkdir(parents=True, exist_ok=False)
dataset = release / "cv-dataset"
export_cv_dataset(
    source_dirs[0],
    dataset,
    seed=4192026,
    additional_data_dirs=[str(path) for path in source_dirs[1:]],
)

model_path = release / "cv-signal.pt"
train_cv_detector(
    dataset,
    output_model=model_path,
    base_model=str(base_model),
    epochs=int(os.environ.get("SIG_ID_INCREMENTAL_CV_EPOCHS", "30")),
    batch=int(os.environ.get("SIG_ID_INCREMENTAL_CV_BATCH", "8")),
    device=os.environ.get("SIG_ID_INCREMENTAL_CV_DEVICE", "0"),
)

active["cv_detector"] = str(model_path.resolve())
with tempfile.NamedTemporaryFile(
    "w", encoding="utf-8", dir=pointer_path.parent, delete=False
) as f:
    json.dump(active, f, indent=2)
    f.write("\n")
    temporary = Path(f.name)
os.replace(temporary, pointer_path)

print(f"CV model activated: {active['cv_detector']}")
print(f"Active model config: {pointer_path}")
PY
