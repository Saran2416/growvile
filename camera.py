"""Low-memory real-time object detection from a webcam.

Examples
--------
Default (80 COCO classes, ~450 MB RSS on CPU)::

    .venv/bin/python camera.py

Faster on a weak CPU::

    .venv/bin/python camera.py --imgsz 416 --every 2

Arbitrary labels (downloads a YOLO-World checkpoint, needs CLIP)::

    .venv/bin/python camera.py --classes "person, cup, fan, book, dog, laptop"

Headless smoke test that reads three frames and exits::

    .venv/bin/python camera.py --headless --frames 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from typing import Any

import cv2

from detector import DetectorConfig, UnifiedDetector, parse_classes
from memory import format_rss, rss_mb
from visualize import draw_detections, draw_status


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    """Open a camera with a one-frame buffer to avoid latency build-up."""
    backend = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
    camera = cv2.VideoCapture(index, backend)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera


def run_camera(
    config: DetectorConfig,
    camera_index: int = 0,
    width: int = 640,
    height: int = 480,
    every: int = 1,
    headless: bool = False,
    max_frames: int = 0,
    output: str | None = None,
    max_rss_mb: float = 0.0,
    mirror: bool = False,
) -> dict[str, Any]:
    """Run the camera loop and return a small summary dictionary."""
    every = max(1, every)
    camera = open_camera(camera_index, width, height)
    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera {camera_index}. "
            "Check `ls /dev/video*` and the camera permissions."
        )
    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height

    print(
        f"Loading {config.model} (imgsz={config.imgsz}, "
        f"mediapipe={config.mediapipe}, threads={config.threads})...",
        file=sys.stderr,
    )
    started = time.monotonic()
    try:
        detector = UnifiedDetector(config)
    except Exception:
        camera.release()
        raise
    print(
        f"Model ready in {time.monotonic() - started:.2f}s | RAM {format_rss()}",
        file=sys.stderr,
    )

    show_window = not headless
    if show_window:
        try:
            cv2.namedWindow("growvile", cv2.WINDOW_NORMAL)
        except cv2.error:
            print("No display available; continuing headless.", file=sys.stderr)
            show_window = False

    writer: cv2.VideoWriter | None = None
    writer_failed = False
    timestamps: deque[float] = deque(maxlen=30)
    frame_count = 0
    detection_count = 0
    inference_seconds = 0.0
    last_detections: list[dict[str, Any]] = []
    output_path: str | None = None
    interrupted = False

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("Camera stopped returning frames.", file=sys.stderr)
                break
            if mirror:
                frame = cv2.flip(frame, 1)

            frame_count += 1
            if frame_count == 1 or frame_count % every == 0:
                began = time.monotonic()
                last_detections = detector.predict(frame)
                inference_seconds = time.monotonic() - began
                detection_count += 1

            draw_detections(frame, last_detections)
            now = time.monotonic()
            timestamps.append(now)
            fps = None
            if len(timestamps) > 1:
                fps = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
            draw_status(
                frame,
                fps=fps,
                inference_ms=inference_seconds * 1000,
                memory=format_rss(),
                backend="open-vocab" if config.world_classes else "coco80",
            )

            if output and not writer_failed:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        output, fourcc, 20.0, (frame.shape[1], frame.shape[0])
                    )
                    if not writer.isOpened():
                        print(f"Could not create video file: {output}", file=sys.stderr)
                        writer = None
                        writer_failed = True
                    else:
                        output_path = output
                if writer is not None:
                    writer.write(frame)

            if show_window:
                try:
                    cv2.imshow("growvile", frame)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
                except cv2.error:
                    print("Display closed; continuing headless.", file=sys.stderr)
                    show_window = False

            current_rss = rss_mb()
            if max_rss_mb and current_rss is not None and current_rss > max_rss_mb:
                print(
                    f"Stopping: RAM {current_rss:.0f} MB exceeded "
                    f"--max-rss {max_rss_mb:.0f} MB.",
                    file=sys.stderr,
                )
                break

            if max_frames and frame_count >= max_frames:
                break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        camera.release()
        if writer is not None:
            writer.release()
        if show_window:
            cv2.destroyAllWindows()
        detector.close()

    average_fps = None
    if len(timestamps) > 1:
        average_fps = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])

    return {
        "frames": frame_count,
        "detections_run": detection_count,
        "average_fps": None if average_fps is None else round(average_fps, 2),
        "resolution": {"width": actual_width, "height": actual_height},
        "memory_mb": None if rss_mb() is None else round(rss_mb(), 1),
        "output": output_path,
        "interrupted": interrupted,
        "last_detections": last_detections,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Real-time webcam object detection with a small memory footprint"
    )
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    parser.add_argument("--model", default=os.getenv("YOLO_MODEL", "yolov8n.pt"))
    parser.add_argument(
        "--broad",
        action="store_true",
        help="use the 601-class Open Images V7 model instead of 80-class COCO",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=float(os.getenv("CONFIDENCE", "0.25")),
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=int(os.getenv("IMGSZ", "480")),
        help="inference size; 416-480 is fast on CPU, 640 is more accurate",
    )
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--device", default=os.getenv("DEVICE") or None)
    parser.add_argument("--half", action="store_true", help="FP16 on a CUDA GPU")
    parser.add_argument(
        "--threads",
        type=int,
        default=int(os.getenv("THREADS", str(min(4, os.cpu_count() or 1)))),
    )
    parser.add_argument("--width", type=int, default=640, help="camera capture width")
    parser.add_argument("--height", type=int, default=480, help="camera capture height")
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="run detection every N frames and reuse boxes between runs",
    )
    parser.add_argument(
        "--classes",
        default=os.getenv("WORLD_CLASSES") or None,
        help="comma-separated open-vocabulary labels (YOLO-World)",
    )
    parser.add_argument("--world-model", default="yolov8s-worldv2.pt")
    parser.add_argument(
        "--mediapipe",
        action="store_true",
        help="also detect hands/faces (adds ~300-400 MB of RAM)",
    )
    parser.add_argument("--max-rss", type=float, default=0.0, help="stop above this RAM (MB)")
    parser.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = forever)")
    parser.add_argument("--headless", action="store_true", help="do not open a window")
    parser.add_argument("--mirror", action="store_true", help="mirror the frame like a selfie")
    parser.add_argument("--output", help="record the annotated stream to an MP4 file")
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
    summary = run_camera(
        config,
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        every=args.every,
        headless=args.headless,
        max_frames=args.frames,
        output=args.output,
        max_rss_mb=args.max_rss,
        mirror=args.mirror,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
