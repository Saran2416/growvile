"""Real-time coconut leaf health check from a webcam or a single image.

The classifier looks at the centre of the frame, so hold one frond or a close
group of leaflets so that it fills most of the view.

Examples::

    # single image, prints the full advisory
    .venv/bin/python plant_camera.py --image leaf.jpg

    # live camera: on-screen status plus a printed cause/cure report
    .venv/bin/python plant_camera.py --every 5

    # headless smoke test
    .venv/bin/python plant_camera.py --headless --frames 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import textwrap
from typing import Any

import cv2

from camera import open_camera
from plant import PlantClassifier, advisory_lines

HEALTHY_COLOR = (80, 200, 90)
INFECTED_COLOR = (60, 60, 230)
UNCERTAIN_COLOR = (40, 180, 230)


def center_crop(frame, fraction: float):
    """Return a centred square crop; the classifier expects centred foliage."""
    height, width = frame.shape[:2]
    side = max(32, int(min(height, width) * max(0.1, min(1.0, fraction))))
    x1 = (width - side) // 2
    y1 = (height - side) // 2
    return frame[y1 : y1 + side, x1 : x1 + side]


def wrap_lines(lines: list[str], width: int = 78) -> list[str]:
    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=width) or [""])
    return wrapped


def draw_panel(frame, lines: list[str], color: tuple[int, int, int]) -> None:
    """Draw a readable semi-transparent panel in the top-left corner."""
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    padding = 8
    line_height = 20
    widths = [cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines]
    panel_width = min(frame.shape[1] - 16, max(widths) + padding * 2)
    panel_height = line_height * len(lines) + padding * 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (4, 4), (4 + panel_width, 4 + panel_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    y = 4 + padding + 14
    for line in lines:
        cv2.putText(frame, line, (4 + padding, y), font, scale, color, thickness, cv2.LINE_AA)
        y += line_height


def report(result: dict[str, Any]) -> str:
    return "\n".join(advisory_lines(result))


def run_image(model: PlantClassifier, path: str, topk: int) -> dict[str, Any]:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    result = model.predict(image, topk=topk)
    result["image_size"] = {"width": int(image.shape[1]), "height": int(image.shape[0])}
    return result


def run_camera(
    model: PlantClassifier,
    camera_index: int = 0,
    width: int = 640,
    height: int = 480,
    every: int = 5,
    crop: float = 0.85,
    threshold: float = 0.5,
    headless: bool = False,
    max_frames: int = 0,
) -> dict[str, Any]:
    every = max(1, every)
    camera = open_camera(camera_index, width, height)
    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera {camera_index}. Check `ls /dev/video*` and permissions."
        )

    show_window = not headless
    if show_window:
        try:
            cv2.namedWindow("Plant Doctor", cv2.WINDOW_NORMAL)
        except cv2.error:
            print("No display available; continuing headless.", file=sys.stderr)
            show_window = False

    timestamps: list[float] = []
    frame_count = 0
    last_result: dict[str, Any] | None = None
    last_announced: str | None = None
    started = time.monotonic()

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("Camera stopped returning frames.", file=sys.stderr)
                break
            frame_count += 1

            if frame_count == 1 or frame_count % every == 0:
                try:
                    last_result = model.predict(center_crop(frame, crop))
                except Exception as exc:  # keep the camera alive on a bad frame
                    print(f"Classification failed: {exc}", file=sys.stderr)
                    last_result = None

            lines: list[str] = []
            color = UNCERTAIN_COLOR
            if last_result is not None:
                confident = last_result["confidence"] >= threshold
                if not last_result.get("recognized", True):
                    lines = [
                        "NOT COCONUT",
                        "Point the camera at a coconut frond",
                        f"Confidence: {last_result['confidence'] * 100:.1f}%",
                    ]
                    if last_result["label"] != last_announced:
                        print("\n" + report(last_result) + "\n", file=sys.stderr)
                        last_announced = last_result["label"]
                elif not confident:
                    lines = [
                        f"Uncertain ({last_result['confidence'] * 100:.0f}%)",
                        "Move the leaf closer / fill the frame",
                    ]
                else:
                    status = "HEALTHY" if last_result["healthy"] else "INFECTED"
                    lines = [
                        f"{status}: {last_result['plant']} / {last_result['disease']}",
                        f"Confidence: {last_result['confidence'] * 100:.1f}%",
                    ]
                    if last_result["healthy"]:
                        color = HEALTHY_COLOR
                    else:
                        color = INFECTED_COLOR
                        lines.append("Cause & cure printed in the terminal")

                    if last_result["label"] != last_announced:
                        print("\n" + report(last_result) + "\n", file=sys.stderr)
                        last_announced = last_result["label"]

            draw_panel(frame, lines, color)
            now = time.monotonic()
            timestamps.append(now)
            if len(timestamps) > 30:
                timestamps.pop(0)

            if show_window:
                try:
                    cv2.imshow("Plant Doctor", frame)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
                except cv2.error:
                    print("Display closed; continuing headless.", file=sys.stderr)
                    show_window = False

            if max_frames and frame_count >= max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        if show_window:
            cv2.destroyAllWindows()

    elapsed = time.monotonic() - started
    return {
        "frames": frame_count,
        "camera_fps": round(frame_count / elapsed, 2) if elapsed > 0 else None,
        "last_result": last_result,
        "advice": None
        if last_result is None
        else {
            "plant": last_result["plant"],
            "disease": last_result["disease"],
            "healthy": last_result["healthy"],
            "recognized": last_result.get("recognized", True),
            "confidence": last_result["confidence"],
            **last_result.get("advice", {}),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plant species and disease checker")
    parser.add_argument("--image", help="classify one image and exit")
    parser.add_argument("--model", default="runs/coconut/best.pt")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--every", type=int, default=5, help="classify every N frames")
    parser.add_argument("--crop", type=float, default=0.85, help="centre crop fraction (0-1)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = forever)")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = PlantClassifier(args.model, threads=args.threads)

    if args.image:
        result = run_image(model, args.image, args.top_k)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(report(result))
            print("\nTop predictions:")
            for item in result["candidates"]:
                print(f"  {item['confidence'] * 100:5.1f}%  {item['plant']} / {item['disease']}")
        return

    summary = run_camera(
        model,
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        every=args.every,
        crop=args.crop,
        threshold=args.threshold,
        headless=args.headless,
        max_frames=args.frames,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
