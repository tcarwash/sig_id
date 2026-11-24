"""Compose a frozen CV base dataset with a *train-only* synthetic supplement.

Never mix synthetic supplement files into validation/test, and never regenerate,
relabel, select thresholds or alter the fixed held-out evaluation splits.
Outputs are staged outside the source tree. Hard links are read-only treated
by our code; never edit linked source files in the output dataset.
"""
from __future__ import annotations

import argparse
import errno
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for buf in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(buf)
    return h.hexdigest()


def load_rows(root: Path) -> list[dict]:
    return [json.loads(line) for line in (root / "windows.jsonl").read_text().splitlines() if line.strip()]


def compose(base: Path, supplement: Path, destination: Path) -> dict:
    base = base.expanduser().resolve(strict=True)
    supplement = supplement.expanduser().resolve(strict=True)
    destination = destination.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing training dataset {destination}")
    report = json.loads((base / "export-report.json").read_text())
    extra_report = json.loads((supplement / "export-report.json").read_text())
    if (report.get("preprocessing_version") != extra_report.get("preprocessing_version")
            or report.get("schema_version") != extra_report.get("schema_version")):
        raise ValueError("CV preprocessing / report schema mismatch")
    if extra_report.get("train_only") is not True or not extra_report.get("scene_index_sha256"):
        raise ValueError("supplement is not v4 train-only with source-provenance index")
    if extra_report.get("split_images", {}).get("validation") or extra_report.get("split_images", {}).get("test"):
        raise ValueError("supplement must have no heldout images")
    base_rows = load_rows(base)
    extra_rows = load_rows(supplement)
    if not base_rows or not extra_rows or any(row["split"] != "train" for row in extra_rows):
        raise ValueError("supplement contains invalid split rows")
    base_heldout = {row["capture_sha256"] for row in base_rows if row["split"] != "train"}
    if base_heldout & {row["capture_sha256"] for row in extra_rows}:
        raise ValueError("supplement duplicates original heldout captures")
    base_heldout_groups = {row["lineage_group"] for row in base_rows if row["split"] != "train"}
    if base_heldout_groups & {row["lineage_group"] for row in extra_rows}:
        raise ValueError("supplement shares a source lineage with validation/test")
    if set(row["lineage_group"] for row in base_rows if row["split"] == "train") & base_heldout_groups:
        raise ValueError("base train split overlaps base heldout lineage")
    if len(base_rows) != sum(report["split_images"].values()) or len(extra_rows) != extra_report["split_images"]["train"]:
        raise ValueError("input window manifest count mismatch")
    if any(not row.get("image_sha256") or not row.get("label_sha256") for row in base_rows + extra_rows):
        raise ValueError("missing image or label hashes")
    destination.mkdir(parents=True)
    try:
        combined = []
        for root, rows, prefix in ((base, base_rows, "base"), (supplement, extra_rows, "v4")):
            for row in rows:
                split = row["split"]
                image_rel = Path(row["image"])
                if (image_rel.is_absolute() or ".." in image_rel.parts or len(image_rel.parts) != 3
                        or image_rel.parts[:2] != ("images", split) or image_rel.suffix != ".png"):
                    raise ValueError("unexpected image path in input dataset")
                label_rel = Path("labels") / split / (image_rel.stem + ".txt")
                image_src = root / image_rel
                label_src = root / label_rel
                if sha256(image_src) != row["image_sha256"] or sha256(label_src) != row["label_sha256"]:
                    raise ValueError("source image/label checksum mismatch")
                new_stem = f"{prefix}-{image_rel.stem}"
                image_dst = destination / "images" / split / f"{new_stem}.png"
                label_dst = destination / "labels" / split / f"{new_stem}.txt"
                image_dst.parent.mkdir(parents=True, exist_ok=True)
                label_dst.parent.mkdir(parents=True, exist_ok=True)
                for source_file, destination_file in ((image_src, image_dst), (label_src, label_dst)):
                    try:
                        os.link(source_file, destination_file)
                    except OSError as exc:
                        if exc.errno != errno.EXDEV:
                            raise
                        shutil.copy2(source_file, destination_file)
                copy = dict(row)
                copy["image"] = image_dst.relative_to(destination).as_posix()
                copy["combined_provenance"] = prefix
                combined.append(copy)
        for split in ("train", "validation", "test"):
            (destination / "images" / split).mkdir(parents=True, exist_ok=True)
            (destination / "labels" / split).mkdir(parents=True, exist_ok=True)
        (destination / "windows.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in combined))
        existing_test = [row for row in base_rows if row["split"] == "test"]
        existing_validation = [row for row in base_rows if row["split"] == "validation"]
        after_test = [row for row in combined if row["split"] == "test"]
        after_validation = [row for row in combined if row["split"] == "validation"]
        for before, after in ((existing_test, after_test), (existing_validation, after_validation)):
            if len(before) != len(after):
                raise ValueError("heldout count changed")
            for x, y in zip(before, after, strict=True):
                if (x["capture_sha256"], x["sample_start"], x["image_sha256"], x["label_sha256"], x["lineage_group"]) != (
                    y["capture_sha256"], y["sample_start"], y["image_sha256"], y["label_sha256"], y["lineage_group"]):
                    raise ValueError("heldout source, split, label or pixels altered")
        counts = Counter(row["split"] for row in combined)
        positive = Counter(row["split"] for row in combined if row["box_count"] > 0)
        negative = Counter(row["split"] for row in combined if row["box_count"] == 0)
        result = dict(report)
        result["split_images"] = {key: counts[key] for key in ("train", "validation", "test")}
        result["positive_images_by_split"] = {key: positive[key] for key in ("train", "validation", "test")}
        result["negative_images_by_split"] = {key: negative[key] for key in ("train", "validation", "test")}
        result["positive_images"] = sum(positive.values())
        result["negative_images"] = sum(negative.values())
        result["composite_dataset"] = {
            "base_report_sha256": sha256(base / "export-report.json"),
            "base_manifest_sha256": sha256(base / "windows.jsonl"),
            "v4_report_sha256": sha256(supplement / "export-report.json"),
            "v4_manifest_sha256": sha256(supplement / "windows.jsonl"),
            "source_train_added": len(extra_rows),
            "source_training_only": True,
            "heldout_image_sha256_unchanged": True,
            "dataset_input_preprocessing": report["preprocessing_version"],
        }
        (destination / "export-report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        (destination / "data.yaml").write_text(
            f"path: {destination}\ntrain: images/train\nval: images/validation\ntest: images/test\nnc: 1\nnames: [signal]\n"
        )
        return result
    except BaseException:
        shutil.rmtree(destination)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compose(args.base, args.supplement, args.output)
    print(json.dumps({"split_images": result["split_images"],
                      "positive": result["positive_images_by_split"],
                      "negative": result["negative_images_by_split"],
                      "composite": result["composite_dataset"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
