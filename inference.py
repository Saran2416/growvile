"""Run unified detection on an image or webcam."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from detector import DetectorConfig, UnifiedDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO + MediaPipe object detection")
    parser.add_argument("source", help="Image path or webcam index, for example 0")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--webcam", action="store_true")
    args = parser.parse_args()
    detector = UnifiedDetector(DetectorConfig(model=args.model, confidence=args.confidence))

    if args.webcam:
        camera = cv2.VideoCapture(int(args.source))
        if not camera.isOpened():
            raise RuntimeError("Could not open webcam")
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            result = detector.predict(frame)
            for item in result:
                x1, y1, x2, y2 = item["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 130), 2)
                cv2.putText(frame, f'{item["object"]} {item["confidence"]:.2f}',
                            (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 220, 130), 2)
            cv2.imshow("Open Object Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        camera.release()
        cv2.destroyAllWindows()
        return

    path = Path(args.source)
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    print(json.dumps({"detections": detector.predict(image),
                      "image_size": {"width": image.shape[1], "height": image.shape[0]}}, indent=2))


if __name__ == "__main__":
    main()
