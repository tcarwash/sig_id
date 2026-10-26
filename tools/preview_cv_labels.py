#!/usr/bin/env python3
"""Write a separate visual preview of YOLO ground-truth boxes."""

from __future__ import annotations

import argparse
from pathlib import Path
import random


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/cv-seed/export"))
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, help="default: DATASET/previews/SPLIT")
    parser.add_argument("--include-empty", action="store_true", help="also include no-signal images")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1")

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise SystemExit("Install image support with: uv sync --extra cv") from exc

    dataset = args.dataset.expanduser().resolve()
    image_dir = dataset / "images" / args.split
    label_dir = dataset / "labels" / args.split
    output_dir = (args.output or dataset / "previews" / args.split).expanduser().resolve()
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise SystemExit(f"missing images or labels for split {args.split!r} in {dataset}")

    candidates = []
    for image_path in sorted(image_dir.glob("*.png")):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue
        lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines or args.include_empty:
            candidates.append((image_path, label_path, lines))
    if not candidates:
        raise SystemExit(f"no matching labeled images found in {image_dir}")
    rng = random.Random(args.seed)
    selected = rng.sample(candidates, min(args.limit, len(candidates)))
    output_dir.mkdir(parents=True, exist_ok=True)
    total_boxes = 0
    for image_path, _, lines in selected:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        for line in lines:
            columns = line.split()
            if len(columns) != 5:
                raise ValueError(f"invalid YOLO label in {image_path.stem}: {line}")
            class_id, cx, cy, box_w, box_h = columns
            cx, cy, box_w, box_h = map(float, (cx, cy, box_w, box_h))
            left = round((cx - box_w / 2) * image.width)
            top = round((cy - box_h / 2) * image.height)
            right = round((cx + box_w / 2) * image.width)
            bottom = round((cy + box_h / 2) * image.height)
            draw.rectangle((left, top, right, bottom), outline=(255, 65, 190), width=max(2, image.width // 512))
            draw.text((left + 3, top + 3), f"signal {class_id}", fill=(255, 255, 255),
                      stroke_width=2, stroke_fill=(0, 0, 0), font=font)
            total_boxes += 1
        image.save(output_dir / image_path.name, optimize=True)
    print(f"Wrote {len(selected)} preview image(s) with {total_boxes} ground-truth box(es) to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
