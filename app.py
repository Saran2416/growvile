"""FastAPI wrapper for the unified detector.

Environment variables:

* ``YOLO_MODEL``        checkpoint path/name (default ``yolov8n.pt``)
* ``BROAD``             ``1`` to use the 601-class Open Images model
* ``CONFIDENCE``        score threshold (default ``0.25``)
* ``DEVICE``            ``cpu``, ``0`` for the first CUDA GPU, etc.
* ``IMGSZ``             inference size (default ``640``; ``416`` is cheaper)
* ``MEDIAPIPE``         ``1`` to add hands/faces (uses ~300-400 MB more RAM)
* ``WORLD_CLASSES``     comma-separated open-vocabulary labels
* ``BOTTLE_SUBTYPE_MODEL`` optional bottle classifier checkpoint
* ``THREADS``           PyTorch CPU thread count
* ``MAX_UPLOAD_MB``     reject larger uploads (default ``20``)
"""

from __future__ import annotations

import os
from functools import lru_cache

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from detector import DetectorConfig, UnifiedDetector, parse_classes


app = FastAPI(title="Open Object Detector", version="1.1.0")


def _flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def get_detector() -> UnifiedDetector:
    world_classes = parse_classes(os.getenv("WORLD_CLASSES"))
    broad = _flag("BROAD") and not world_classes
    default_model = "yolov8n-oiv7.pt" if broad else "yolov8n.pt"
    return UnifiedDetector(
        DetectorConfig(
            model=os.getenv("YOLO_MODEL") or default_model,
            confidence=float(os.getenv("CONFIDENCE", "0.25")),
            device=os.getenv("DEVICE") or None,
            imgsz=int(os.getenv("IMGSZ", "640")),
            max_det=int(os.getenv("MAX_DET", "100")),
            mediapipe=_flag("MEDIAPIPE"),
            world_classes=world_classes,
            bottle_subtype_model=os.getenv("BOTTLE_SUBTYPE_MODEL") or None,
            threads=int(os.getenv("THREADS", str(min(4, os.cpu_count() or 1)))),
        )
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    content = await file.read()
    limit = float(os.getenv("MAX_UPLOAD_MB", "20")) * 1_000_000
    if len(content) > limit:
        raise HTTPException(status_code=413, detail=f"Upload is larger than {limit / 1e6:.0f} MB")
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Upload a valid image file")
    result = get_detector().predict_with_status(image)
    result["filename"] = file.filename
    result["image_size"] = {"width": int(image.shape[1]), "height": int(image.shape[0])}
    return result
