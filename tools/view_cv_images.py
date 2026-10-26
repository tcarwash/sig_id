#!/usr/bin/env python3
"""Run a YOLO signal detector over exported spectrogram images and view boxes."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path,
        default=Path("data/cv-seed/.signal-training/run/weights/best.pt"),
        help="YOLO weights (default: %(default)s)",
    )
    parser.add_argument("--split", choices=("test", "validation"), default="test",
                        help="dataset split to sample from (default: test)")
    parser.add_argument("--source", type=Path, help="override the split image folder")
    parser.add_argument("--images", type=int, default=10, help="number of images to run (default: 10)")
    parser.add_argument("--conf", type=float, default=0.25, help="minimum detection confidence")
    parser.add_argument("--device", default=None, help="inference device, e.g. cpu, 0")
    parser.add_argument("--seed", type=int, help="optional seed for a repeatable random sample")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.images < 1:
        raise SystemExit("--images must be at least 1")
    if not 0.0 <= args.conf <= 1.0:
        raise SystemExit("--conf must be between 0 and 1")
    if not args.model.is_file():
        raise SystemExit(f"model not found: {args.model}")
    source = args.source or Path(f"data/cv-seed/export/images/{args.split}")
    if not source.is_dir():
        raise SystemExit(f"image folder not found: {source}")

    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install inference dependencies with: pip install -e '.[cv]'") from exc

    paths = sorted(path for path in source.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not paths:
        raise SystemExit(f"no PNG/JPEG images found in {source}")
    rng = random.Random(args.seed)
    paths = rng.sample(paths, min(args.images, len(paths)))

    model = YOLO(str(args.model))
    print(f"Running {len(paths)} image(s) with {args.model}")
    print("Viewer: any key = next image, Esc or q = quit")
    try:
        for index, path in enumerate(paths, start=1):
            result = model.predict(
                str(path), conf=args.conf, imgsz=(640, 1024), rect=True,
                verbose=False, device=args.device,
            )[0]
            annotated_bgr = result.plot()
            count = 0 if result.boxes is None else len(result.boxes)
            title = f"{index}/{len(paths)}  {path.name}  detections: {count}"
            cv2.imshow("Signal detector — " + title, annotated_bgr)
            print(f"[{index}/{len(paths)}] {path.name}: {count} detection(s)")
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            if key in (27, ord("q")):
                break
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
