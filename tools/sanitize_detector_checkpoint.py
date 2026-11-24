"""Strip private training metadata from a *trusted local* YOLO checkpoint.

Loads Python pickle: run only on checkpoints produced by this project and
verified locally. Does not change model tensors, names, stride or structure.

The resulting checkpoint is NOT an accuracy approval or licensing review.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

PRIVATE_MARKERS = ("/home/", "/storage5/", "/Users/", "\\Users\\", "\\Documents\\")


def redact(value):
    if isinstance(value, str):
        return "<redacted-local-path>" if any(marker in value for marker in PRIVATE_MARKERS) else value
    if isinstance(value, dict):
        return {redact(k): redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact(item) for item in value)
    return value


def sanitize_trusted_model(input_path: Path, output_path: Path) -> dict:
    import torch
    from sigid.model_distribution import _checkpoint_private_path_markers

    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {output_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output must be different")
    original = torch.load(input_path, weights_only=False, map_location="cpu")
    if not isinstance(original, dict) or original.get("model") is None:
        raise ValueError("unsupported checkpoint; missing model")
    model = copy.deepcopy(original["model"])
    for module in model.modules():
        for key, value in tuple(module.__dict__.items()):
            if key in {"_modules", "_parameters", "_buffers"}:
                continue
            if isinstance(value, (str, dict, list, tuple)):
                module.__dict__[key] = redact(value)
        if hasattr(module, "pt_path"):
            module.pt_path = ""
    # Ultralytics will load sensible prediction defaults from this minimal
    # subset; remove training data, train_args file paths and private git roots.
    inference_args = {"task": "detect", "mode": "predict", "imgsz": 640,
                      "model": "cv-signal.pt"}
    model.args = inference_args
    portable = {
        "model": model,
        "ema": None,
        "optimizer": None,
        "train_args": inference_args,
        "version": original.get("version", "8.4.184"),
        "license": "AGPL-3.0",
        "docs": "https://docs.ultralytics.com",
        "epoch": -1,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(portable, output_path)
    candidate = output_path.read_bytes()
    markers = _checkpoint_private_path_markers(candidate)
    if markers or any(token.encode() in candidate for token in PRIVATE_MARKERS):
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"unsafe local path markers remain after sanitization: {markers}")
    from ultralytics import YOLO
    loaded = YOLO(str(output_path))
    if loaded.model is None or list(loaded.names.values()) != ["signal"]:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("sanitized checkpoint could not be loaded with signal class")
    return {
        "sha256": hashlib.sha256(candidate).hexdigest(),
        "size_bytes": len(candidate),
        "ultralytics_version": original.get("version"),
        "private_path_markers": markers,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(sanitize_trusted_model(
        args.source.expanduser().resolve(strict=True),
        args.destination.expanduser().resolve(),
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
