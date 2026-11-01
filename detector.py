"""Unified open-model object detector.

The default backend is a small YOLOv8 model pretrained on the 80 COCO classes.
The "nano" checkpoint is chosen on purpose: it is about 6 MB on disk and keeps
the whole process around 450-500 MB of resident memory on CPU, which leaves
plenty of headroom for a laptop or a small server.

Optional backends:

* ``mediapipe=True`` adds hands, fingers and faces.  It is off by default
  because MediaPipe alone adds roughly 300-400 MB of RSS.
* ``world_classes=(...)`` switches to a YOLO-World checkpoint, which can detect
  arbitrary labels supplied at runtime.  It needs the CLIP text encoder, so it
  uses more memory than plain COCO detection.
"""

from __future__ import annotations

import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

MODEL_CACHE = Path("models")


def parse_classes(value: str | Iterable[str] | None) -> tuple[str, ...] | None:
    """Turn ``"person, cup,, fan"`` into a clean tuple of class names."""
    if value is None:
        return None
    if isinstance(value, str):
        items = value.split(",")
    else:
        items = list(value)
    cleaned = tuple(item.strip() for item in items if item and item.strip())
    return cleaned or None


@dataclass
class DetectorConfig:
    """Runtime configuration for :class:`UnifiedDetector`."""

    model: str = "yolov8n.pt"
    confidence: float = 0.25
    device: str | None = None
    bottle_subtype_model: str | None = None

    # Inference settings.  640 is accurate; 416-480 is much faster on CPU.
    imgsz: int = 640
    max_det: int = 100
    half: bool = False

    # Extra, memory-hungry backends are opt-in.
    mediapipe: bool = False

    # Open-vocabulary mode: any labels, via a YOLO-World checkpoint.
    world_classes: tuple[str, ...] | None = None
    world_model: str = "yolov8s-worldv2.pt"

    # Limit PyTorch's CPU thread pool.  None keeps the library default.
    threads: int | None = None


class UnifiedDetector:
    """Combines YOLO with optional MediaPipe face/hand detectors."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self._cuda = self._configure_threads()

        from ultralytics import YOLO

        model_name = self.config.model
        if self.config.world_classes and model_name == DetectorConfig.model:
            # The user asked for open-vocabulary labels but did not name a
            # specific checkpoint, so pick the small YOLO-World model.
            model_name = self.config.world_model

        self.model_name = model_name
        self.yolo = YOLO(model_name)
        self._load_world_classes()

        self.mp_hands = None
        self.mp_face = None
        self.mp_task_mode = False
        if self.config.mediapipe:
            self._load_mediapipe()
        self.subtype_model = self._load_subtype_model(self.config.bottle_subtype_model)
        gc.collect()

    # ------------------------------------------------------------------ setup
    def _configure_threads(self) -> bool:
        """Limit CPU threads and report whether CUDA is available."""
        try:
            import torch
        except Exception:  # pragma: no cover - torch is a hard dependency
            return False
        if self.config.threads is not None and self.config.threads > 0:
            torch.set_num_threads(self.config.threads)
        return bool(torch.cuda.is_available())

    def _load_world_classes(self) -> None:
        if not self.config.world_classes:
            return
        if not hasattr(self.yolo, "set_classes"):
            raise RuntimeError(
                "Open-vocabulary labels need a YOLO-World checkpoint. "
                "Re-run without --classes, or use --model yolov8s-worldv2.pt."
            )
        try:
            self.yolo.set_classes(list(self.config.world_classes))
        except Exception as exc:  # pragma: no cover - depends on optional CLIP
            raise RuntimeError(
                "YOLO-World needs the CLIP text encoder. Install it with "
                "`.venv/bin/python -m pip install git+https://github.com/ultralytics/CLIP.git` "
                "or use the fixed COCO model without --classes."
            ) from exc

    def _load_mediapipe(self) -> None:
        try:
            import mediapipe as mp

            if not hasattr(mp, "solutions"):
                self._load_mediapipe_tasks(mp)
                return
            self.mp_hands = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=4,
                min_detection_confidence=self.config.confidence,
            )
            self.mp_face = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=self.config.confidence,
            )
        except (ImportError, AttributeError, OSError) as exc:
            self.mp_hands = None
            self.mp_face = None
            print(
                f"MediaPipe backend unavailable ({exc}); continuing without hands/faces.",
                file=sys.stderr,
            )

    def _load_mediapipe_tasks(self, mp: Any) -> None:
        import urllib.request

        MODEL_CACHE.mkdir(exist_ok=True)
        assets = {
            "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            "blaze_face_short_range.tflite": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        }
        for filename, url in assets.items():
            path = MODEL_CACHE / filename
            if not path.exists():
                urllib.request.urlretrieve(url, path)
        vision = mp.tasks.vision
        base = mp.tasks.BaseOptions
        self.mp_hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=base(model_asset_path=str(MODEL_CACHE / "hand_landmarker.task")),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=4,
                min_hand_detection_confidence=self.config.confidence,
            )
        )
        self.mp_face = vision.FaceDetector.create_from_options(
            vision.FaceDetectorOptions(
                base_options=base(model_asset_path=str(MODEL_CACHE / "blaze_face_short_range.tflite")),
                running_mode=vision.RunningMode.IMAGE,
                min_detection_confidence=self.config.confidence,
            )
        )
        self.mp_task_mode = True

    @staticmethod
    def _load_subtype_model(model_path: str | None) -> Any:
        if not model_path or not Path(model_path).exists():
            return None
        import torch
        from torchvision.models import mobilenet_v2

        checkpoint = torch.load(model_path, map_location="cpu")
        classes = checkpoint.get("classes", ["water_bottle", "sauce_bottle"])
        model = mobilenet_v2(weights=None)
        model.classifier[1] = torch.nn.Linear(model.last_channel, len(classes))
        model.load_state_dict(checkpoint["model"])
        model.eval()
        return model, classes

    # -------------------------------------------------------------- detection
    def _bottle_subtype(self, image: np.ndarray, bbox: list[int]) -> tuple[str, float] | None:
        if self.subtype_model is None:
            return None
        import torch
        from PIL import Image
        from torchvision.models import MobileNet_V2_Weights

        model, classes = self.subtype_model
        x1, y1, x2, y2 = bbox
        crop = image[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
        transform = MobileNet_V2_Weights.DEFAULT.transforms()
        tensor = transform(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        with torch.inference_mode():
            probabilities = torch.softmax(model(tensor.unsqueeze(0)), dim=1)[0]
        index = int(probabilities.argmax())
        return classes[index], round(float(probabilities[index]), 4)

    def _yolo_detections(self, image: np.ndarray) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "conf": self.config.confidence,
            "imgsz": self.config.imgsz,
            "max_det": self.config.max_det,
            "device": self.config.device,
            "verbose": False,
        }
        device = (self.config.device or "").lower()
        if self.config.half and self._cuda and device not in {"cpu", "mps"}:
            kwargs["half"] = True
        results = self.yolo.predict(image, **kwargs)
        detections: list[dict[str, Any]] = []
        names = self.yolo.names
        for result in results:
            if result.boxes is None:
                continue
            for box, confidence, class_id in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy().astype(int),
            ):
                label = names[class_id]
                bbox = [int(round(value)) for value in box]
                item = {"object": label, "confidence": round(float(confidence), 4), "bbox": bbox}
                if label == "bottle":
                    subtype = self._bottle_subtype(image, bbox)
                    if subtype is not None:
                        item["subtype"], item["subtype_confidence"] = subtype
                detections.append(item)
        return detections

    def _specialized_detections(self, image: np.ndarray) -> list[dict[str, Any]]:
        if self.mp_hands is None or self.mp_face is None:
            return []
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections: list[dict[str, Any]] = []
        height, width = image.shape[:2]
        if self.mp_task_mode:
            import mediapipe as mp

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            hand_result = self.mp_hands.detect(mp_image)
            hand_landmarks_list = hand_result.hand_landmarks
        else:
            hand_result = self.mp_hands.process(rgb)
            hand_landmarks_list = hand_result.multi_hand_landmarks or []
        for hand_landmarks in hand_landmarks_list:
            points = hand_landmarks if self.mp_task_mode else hand_landmarks.landmark
            x_values = [point.x * width for point in points]
            y_values = [point.y * height for point in points]
            detections.append(
                {
                    "object": "hand",
                    "confidence": 1.0,
                    "bbox": [
                        int(max(0, min(x_values))),
                        int(max(0, min(y_values))),
                        int(min(width, max(x_values))),
                        int(min(height, max(y_values))),
                    ],
                    "landmarks": [
                        [round(point.x * width, 2), round(point.y * height, 2)]
                        for point in points
                    ],
                }
            )
            for finger_name, index in (
                ("thumb", 4),
                ("index_finger", 8),
                ("middle_finger", 12),
                ("ring_finger", 16),
                ("pinky", 20),
            ):
                point = points[index]
                x, y = int(point.x * width), int(point.y * height)
                detections.append(
                    {
                        "object": finger_name,
                        "confidence": 1.0,
                        "bbox": [x - 5, y - 5, x + 5, y + 5],
                    }
                )
        if self.mp_task_mode:
            face_result = self.mp_face.detect(mp_image)
            face_detections = face_result.detections
        else:
            face_result = self.mp_face.process(rgb)
            face_detections = face_result.detections or []
        for detection in face_detections:
            if self.mp_task_mode:
                box = detection.bounding_box
                x1, y1 = box.origin_x, box.origin_y
                x2, y2 = box.origin_x + box.width, box.origin_y + box.height
                score = detection.categories[0].score
                detections.append(
                    {
                        "object": "face",
                        "confidence": round(float(score), 4),
                        "bbox": [max(0, x1), max(0, y1), min(width, x2), min(height, y2)],
                    }
                )
                continue
            box = detection.location_data.relative_bounding_box
            x1, y1 = int(box.xmin * width), int(box.ymin * height)
            x2, y2 = int((box.xmin + box.width) * width), int((box.ymin + box.height) * height)
            detections.append(
                {
                    "object": "face",
                    "confidence": round(float(detection.score[0]), 4),
                    "bbox": [max(0, x1), max(0, y1), min(width, x2), min(height, y2)],
                }
            )
        return detections

    def predict(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Return every detection for one BGR frame."""
        return self._yolo_detections(image) + self._specialized_detections(image)

    def predict_with_status(self, image: np.ndarray) -> dict[str, Any]:
        """Return detections together with the active backends and memory use."""
        from memory import rss_mb

        open_images = "oiv7" in str(self.model_name).lower()
        result: dict[str, Any] = {
            "detections": self.predict(image),
            "backends": {
                "yolo_coco": not bool(self.config.world_classes) and not open_images,
                "open_images": open_images,
                "open_vocabulary": bool(self.config.world_classes),
                "mediapipe_hands_face": self.mp_hands is not None and self.mp_face is not None,
                "bottle_subtype": self.subtype_model is not None,
            },
            "model": str(self.model_name),
            "imgsz": self.config.imgsz,
        }
        memory = rss_mb()
        if memory is not None:
            result["memory_mb"] = round(memory, 1)
        return result

    def close(self) -> None:
        """Release backend resources (useful in short-lived scripts and tests)."""
        for backend in (self.mp_hands, self.mp_face):
            close = getattr(backend, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        self.mp_hands = None
        self.mp_face = None
        self.yolo = None
        gc.collect()
