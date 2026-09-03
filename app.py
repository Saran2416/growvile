"""FastAPI wrapper for the unified detector."""

from __future__ import annotations

import os
from functools import lru_cache

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from detector import DetectorConfig, UnifiedDetector


app = FastAPI(title="Open Object Detector", version="1.0.0")


@lru_cache(maxsize=1)
def get_detector() -> UnifiedDetector:
    return UnifiedDetector(DetectorConfig(
        model=os.getenv("YOLO_MODEL", "yolov8n.pt"),
        confidence=float(os.getenv("CONFIDENCE", "0.25")),
        device=os.getenv("DEVICE") or None,
        bottle_subtype_model=os.getenv("BOTTLE_SUBTYPE_MODEL") or None,
    ))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    content = await file.read()
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Upload a valid image file")
    result = get_detector().predict_with_status(image)
    result["filename"] = file.filename
    result["image_size"] = {"width": int(image.shape[1]), "height": int(image.shape[0])}
    return result
