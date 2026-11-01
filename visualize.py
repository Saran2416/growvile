"""Small drawing helpers shared by the image, video and camera front-ends."""

from __future__ import annotations

import zlib
from typing import Any, Iterable

import cv2
import numpy as np

# BGR colours that stay readable on light and dark frames.
_PALETTE = (
    (60, 220, 130),
    (255, 170, 60),
    (80, 160, 255),
    (230, 100, 230),
    (70, 220, 230),
    (255, 120, 120),
    (160, 220, 80),
    (200, 160, 255),
)


def color_for(label: str) -> tuple[int, int, int]:
    """Return a stable BGR colour for a class name."""
    return _PALETTE[zlib.crc32(label.encode("utf-8")) % len(_PALETTE)]


def draw_detections(
    frame: np.ndarray,
    detections: Iterable[dict[str, Any]],
    show_confidence: bool = True,
    thickness: int = 2,
) -> np.ndarray:
    """Draw boxes in place and return the same frame.

    Drawing in place avoids allocating another full-size image every frame,
    which matters on small-memory machines.
    """
    height, width = frame.shape[:2]
    for item in detections:
        bbox = item.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = (int(value) for value in bbox)
        x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width - 1))
        y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        label = str(item.get("object", "object"))
        color = color_for(label)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        if show_confidence:
            score = item.get("confidence")
            text = f"{label} {float(score):.2f}" if score is not None else label
        else:
            text = label
        cv2.putText(
            frame,
            text,
            (x1, max(16, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return frame


def draw_status(
    frame: np.ndarray,
    fps: float | None = None,
    inference_ms: float | None = None,
    memory: str | None = None,
    backend: str | None = None,
) -> np.ndarray:
    """Draw a compact status line in the top-left corner."""
    parts: list[str] = []
    if fps is not None:
        parts.append(f"{fps:5.1f} FPS")
    if inference_ms is not None:
        parts.append(f"{inference_ms:5.1f} ms")
    if memory:
        parts.append(f"RAM {memory}")
    if backend:
        parts.append(backend)
    if not parts:
        return frame
    text = "  |  ".join(parts)
    cv2.putText(
        frame,
        text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return frame
