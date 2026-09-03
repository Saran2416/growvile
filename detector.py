"""Unified open-model object detector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MODEL_CACHE = Path("models")


COCO_TARGETS = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "bottle",
    "cup", "chair", "tv", "laptop", "cell phone", "backpack", "book",
    "keyboard", "mouse", "remote", "clock", "scissors", "handbag",
}


@dataclass
class DetectorConfig:
    model: str = "yolov8n.pt"
    confidence: float = 0.25
    device: str | None = None
    bottle_subtype_model: str | None = None


class UnifiedDetector:
    """Combines YOLO with optional MediaPipe face/hand landmark detectors."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        from ultralytics import YOLO

        self.yolo = YOLO(self.config.model)
        self.mp_hands = None
        self.mp_face = None
        self.mp_task_mode = False
        self._load_mediapipe()
        self.subtype_model = self._load_subtype_model(self.config.bottle_subtype_model)

    def _load_mediapipe(self) -> None:
        try:
            import mediapipe as mp

            if not hasattr(mp, "solutions"):
                self._load_mediapipe_tasks(mp)
                return
            self.mp_hands = mp.solutions.hands.Hands(
                static_image_mode=True, max_num_hands=4,
                min_detection_confidence=self.config.confidence,
            )
            self.mp_face = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=self.config.confidence,
            )
        except (ImportError, AttributeError):
            self.mp_hands = None
            self.mp_face = None

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
        self.mp_hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=base(model_asset_path=str(MODEL_CACHE / "hand_landmarker.task")),
            running_mode=vision.RunningMode.IMAGE, num_hands=4,
            min_hand_detection_confidence=self.config.confidence,
        ))
        self.mp_face = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
            base_options=base(model_asset_path=str(MODEL_CACHE / "blaze_face_short_range.tflite")),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=self.config.confidence,
        ))
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
        results = self.yolo.predict(image, conf=self.config.confidence,
                                    device=self.config.device, verbose=False)
        detections: list[dict[str, Any]] = []
        names = self.yolo.names
        for result in results:
            if result.boxes is None:
                continue
            for box, confidence, class_id in zip(
                result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy().astype(int),
            ):
                label = names[class_id]
                if label not in COCO_TARGETS:
                    continue
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
            detections.append({"object": "hand", "confidence": 1.0,
                               "bbox": [int(max(0, min(x_values))), int(max(0, min(y_values))),
                                        int(min(width, max(x_values))), int(min(height, max(y_values)))],
                               "landmarks": [[round(point.x * width, 2), round(point.y * height, 2)]
                                             for point in points]})
            for finger_name, index in (("thumb", 4), ("index_finger", 8),
                                       ("middle_finger", 12), ("ring_finger", 16), ("pinky", 20)):
                point = points[index]
                x, y = int(point.x * width), int(point.y * height)
                detections.append({"object": finger_name, "confidence": 1.0,
                                   "bbox": [x - 5, y - 5, x + 5, y + 5]})
        if self.mp_task_mode:
            face_result = self.mp_face.detect(mp_image)
            face_detections = face_result.detections
        else:
            face_result = self.mp_face.process(rgb)
            face_detections = face_result.detections or []
        for detection in face_detections:
            if self.mp_task_mode:
                box = detection.bounding_box
                x1, y1, x2, y2 = box.origin_x, box.origin_y, box.origin_x + box.width, box.origin_y + box.height
                score = detection.categories[0].score
                detections.append({"object": "face", "confidence": round(float(score), 4),
                                   "bbox": [max(0, x1), max(0, y1), min(width, x2), min(height, y2)]})
                continue
            box = detection.location_data.relative_bounding_box
            x1, y1 = int(box.xmin * width), int(box.ymin * height)
            x2, y2 = int((box.xmin + box.width) * width), int((box.ymin + box.height) * height)
            detections.append({"object": "face", "confidence": round(float(detection.score[0]), 4),
                               "bbox": [max(0, x1), max(0, y1), min(width, x2), min(height, y2)]})
        return detections

    def predict(self, image: np.ndarray) -> list[dict[str, Any]]:
        return self._yolo_detections(image) + self._specialized_detections(image)

    def predict_with_status(self, image: np.ndarray) -> dict[str, Any]:
        return {"detections": self.predict(image), "backends": {
            "yolo_coco": True,
            "mediapipe_hands_face": self.mp_hands is not None and self.mp_face is not None,
            "bottle_subtype": self.subtype_model is not None,
        }}
