"""Run unified detection on an image or webcam.

Image::

    .venv/bin/python inference.py samples/bus.jpg
    .venv/bin/python inference.py samples/bus.jpg --save output.jpg

Camera::

    .venv/bin/python inference.py 0 --webcam
    .venv/bin/python inference.py 0 --webcam --imgsz 416 --every 2
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2

from detector import DetectorConfig, UnifiedDetector, parse_classes
from visualize import draw_detections


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO + optional MediaPipe object detection")
    parser.add_argument("source", nargs="?", default="0", help="image path or webcam index")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--model", default=os.getenv("YOLO_MODEL", "yolov8n.pt"))
    parser.add_argument(
        "--broad",
        action="store_true",
        help="use the 601-class Open Images V7 model instead of 80-class COCO",
    )
    parser.add_argument("--imgsz", type=int, default=int(os.getenv("IMGSZ", "640")))
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--device", default=os.getenv("DEVICE") or None)
    parser.add_argument("--half", action="store_true", help="FP16 on a CUDA GPU")
    parser.add_argument(
        "--threads",
        type=int,
        default=int(os.getenv("THREADS", str(min(4, os.cpu_count() or 1)))),
    )
    parser.add_argument(
        "--mediapipe",
        action="store_true",
        help="also detect hands/faces (adds ~300-400 MB of RAM)",
    )
    parser.add_argument(
        "--classes",
        default=None,
        help="comma-separated open-vocabulary labels (YOLO-World)",
    )
    parser.add_argument("--world-model", default="yolov8s-worldv2.pt")
    parser.add_argument("--save", nargs="?", const="", metavar="PATH", help="save annotated image")
    parser.add_argument("--webcam", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--record", help="record the webcam stream to an MP4 file")
    parser.add_argument("--mirror", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.broad and args.classes:
        raise SystemExit("Use either --broad or --classes, not both.")
    config = DetectorConfig(
        model="yolov8n-oiv7.pt" if args.broad else args.model,
        confidence=args.confidence,
        device=args.device,
        imgsz=args.imgsz,
        max_det=args.max_det,
        half=args.half,
        mediapipe=args.mediapipe,
        world_classes=parse_classes(args.classes),
        world_model=args.world_model,
        threads=args.threads,
    )

    if args.webcam:
        from camera import run_camera

        summary = run_camera(
            config,
            camera_index=int(args.source),
            width=args.width,
            height=args.height,
            every=args.every,
            headless=args.headless,
            max_frames=args.frames,
            output=args.record,
            mirror=args.mirror,
        )
        print(json.dumps(summary, indent=2))
        return

    path = Path(args.source)
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    detector = UnifiedDetector(config)
    try:
        result = detector.predict_with_status(image)
    finally:
        detector.close()

    result["image_size"] = {"width": image.shape[1], "height": image.shape[0]}

    if args.save is not None:
        annotated = image.copy()
        draw_detections(annotated, result["detections"])
        output = Path(args.save) if args.save else path.with_name(f"{path.stem}_annotated{path.suffix}")
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), annotated):
            raise OSError(f"Could not write annotated image: {output}")
        result["saved"] = str(output)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
