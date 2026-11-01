"""Coconut leaf health classifier.

This is a *classifier*, not a detector: it expects foliage that fills most of
the frame.  The shipped model is trained on a merged coconut dataset whose
classes cover the common coconut diseases, a caterpillar pest and healthy
leaves, plus an explicit ``Not_Coconut`` rejection class.  Results with
``recognized=False`` are refusals: no plant, cause or cure is reported for
them.

For each prediction the module also returns a curated advisory (pathogen,
cause, symptoms, cure and prevention) from ``data/coconut_diseases.json``.  The
advisory is static text written by hand, so the model cannot invent a
treatment.

Typical use::

    from plant import PlantClassifier

    model = PlantClassifier("runs/coconut/best.pt")
    prediction = model.predict(frame)
    print(prediction["plant"], prediction["disease"], prediction["confidence"])
    print(prediction["advice"]["cure"])
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_CHECKPOINT = Path("runs/coconut/best.pt")
KNOWLEDGE_FILE = Path("data/coconut_diseases.json")
DEFAULT_PLANT = "Coconut"

# ImageNet statistics, matching the pretrained torchvision backbones.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

ARCHITECTURES = (
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "resnet18",
    "efficientnet_b0",
)


def split_label(
    label: str,
    knowledge: dict[str, dict[str, Any]] | None = None,
    default_plant: str = DEFAULT_PLANT,
) -> tuple[str, str]:
    """Split a class label into ``(plant, disease)``.

    PlantVillage-style labels use ``"Tomato___Late_blight"``.  Coconut labels
    are plain folder names such as ``"Gray Leaf Spot"``, so those are resolved
    through the knowledge base (or cleaned up and attached to ``default_plant``).
    """
    if "___" in label:
        plant, _, disease = label.partition("___")
        plant = plant.replace("_", " ").replace("(including sour)", "(sour)").strip()
        disease = disease.replace("_", " ").strip() or "Healthy"
        if disease.lower().strip() == "healthy":
            disease = "Healthy"
        return plant.strip(), disease

    key = label.strip()
    entry = (knowledge or {}).get(key)
    if entry is not None:
        return entry.get("plant") or default_plant, entry.get("disease") or key

    disease = key.replace("_", " ").strip() or "Healthy"
    if disease.lower() in ("healthy", "healthy leaves"):
        return default_plant, "Healthy"
    return default_plant, disease


def load_knowledge(path: str | Path = KNOWLEDGE_FILE) -> dict[str, dict[str, Any]]:
    """Load the curated disease knowledge base."""
    file = Path(path)
    if not file.exists():
        return {}
    data = json.loads(file.read_text(encoding="utf-8"))
    return {key: value for key, value in data.items() if not key.startswith("_")}


def load_meta(path: str | Path = KNOWLEDGE_FILE) -> dict[str, Any]:
    """Load the ``_meta`` block of the knowledge base."""
    file = Path(path)
    if not file.exists():
        return {}
    data = json.loads(file.read_text(encoding="utf-8"))
    return data.get("_meta", {})


def build_model(arch: str, num_classes: int, pretrained: bool = False):
    """Create a torchvision classifier with a fresh head."""
    import torch.nn as nn
    from torchvision import models

    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {arch!r}; choose from {', '.join(ARCHITECTURES)}")

    weights = None
    if pretrained:
        weight_enum = {
            "mobilenet_v3_small": "MobileNet_V3_Small_Weights",
            "mobilenet_v3_large": "MobileNet_V3_Large_Weights",
            "resnet18": "ResNet18_Weights",
            "efficientnet_b0": "EfficientNet_B0_Weights",
        }[arch]
        weights = getattr(models, weight_enum).DEFAULT

    factory = getattr(models, arch)
    model = factory(weights=weights)

    if arch == "resnet18":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def classifier_parameters(model):
    """Return the parameters that make up the classification head."""
    import torch.nn as nn

    head = model.fc if hasattr(model, "fc") else model.classifier
    return list(head.parameters())


def freeze_backbone(model, frozen: bool = True) -> None:
    """Freeze every parameter except the classifier head."""
    for parameter in model.parameters():
        parameter.requires_grad = not frozen
    for parameter in classifier_parameters(model):
        parameter.requires_grad = True


def build_train_transform(imgsz: int):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(imgsz, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def build_eval_transform(imgsz: int):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(imgsz + 32),
            transforms.CenterCrop(imgsz),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


@dataclass
class _LoadedModel:
    network: Any
    classes: list[str]
    imgsz: int
    arch: str
    accuracy: float | None


class PlantClassifier:
    """Load a trained leaf classifier and explain its predictions."""

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        knowledge: str | Path = KNOWLEDGE_FILE,
        threads: int | None = None,
        device: str | None = None,
    ) -> None:
        import torch

        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(
                f"No trained plant model at {path}. Train one first:\n"
                "  .venv/bin/python train_plant.py --data datasets/coconut/merged \\\n"
                "      --epochs 6 --freeze-epochs 4 --output runs/coconut"
            )

        if threads and threads > 0:
            torch.set_num_threads(threads)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(path, map_location=self.device, weights_only=False)
        classes = payload.get("classes")
        if not classes:
            raise ValueError(f"{path} does not contain class names; retrain with train_plant.py")

        self.arch = payload.get("arch", "mobilenet_v3_small")
        self.imgsz = int(payload.get("imgsz", 224))
        self.classes = list(classes)
        self.accuracy = payload.get("val_accuracy")
        network = build_model(self.arch, len(self.classes), pretrained=False)
        network.load_state_dict(payload["state_dict"])
        network.to(self.device).eval()

        self._loaded = _LoadedModel(network, self.classes, self.imgsz, self.arch, self.accuracy)
        self.transform = build_eval_transform(self.imgsz)
        self.knowledge = load_knowledge(knowledge)
        self.meta = load_meta(knowledge)
        self.default_plant = self.meta.get("default_plant", DEFAULT_PLANT)

    def describe(self, label: str) -> tuple[str, str, bool]:
        """Return ``(plant, disease, healthy)`` for a class label."""
        plant, disease = split_label(label, self.knowledge, self.default_plant)
        entry = self.knowledge.get(label)
        healthy = bool(entry.get("healthy")) if entry is not None else disease == "Healthy"
        return plant, disease, healthy

    def is_recognized(self, label: str) -> bool:
        """Whether a label is an in-scope coconut class (not a rejection)."""
        entry = self.knowledge.get(label)
        return bool(entry.get("recognized", True)) if entry is not None else True

    # ------------------------------------------------------------------ predict
    def predict(self, image_bgr: np.ndarray, topk: int = 3) -> dict[str, Any]:
        """Classify one BGR frame and return the best class plus an advisory."""
        import torch
        from PIL import Image

        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Empty image passed to PlantClassifier.predict")

        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1]) if image_bgr.ndim == 3 else image_bgr
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            probabilities = torch.softmax(self._loaded.network(tensor), dim=1)[0]

        k = max(1, min(topk, len(self.classes)))
        scores, indices = torch.topk(probabilities, k)
        candidates = []
        for score, index in zip(scores.tolist(), indices.tolist()):
            label = self.classes[index]
            plant, disease, healthy = self.describe(label)
            candidates.append(
                {
                    "label": label,
                    "plant": plant,
                    "disease": disease,
                    "healthy": healthy,
                    "confidence": round(float(score), 4),
                }
            )

        best = candidates[0]
        advice = self.advise(best["label"])
        recognized = self.is_recognized(best["label"])
        return {
            "label": best["label"],
            "plant": best["plant"],
            "disease": best["disease"],
            "healthy": best["healthy"],
            "recognized": recognized,
            "confidence": best["confidence"],
            "candidates": candidates,
            "advice": advice,
        }

    def advise(self, label: str) -> dict[str, Any]:
        """Return the curated advisory for a class label."""
        entry = self.knowledge.get(label)
        if entry is not None:
            return entry
        plant, disease = split_label(label, self.knowledge, self.default_plant)
        healthy = disease == "Healthy"
        return {
            "plant": plant,
            "disease": disease,
            "healthy": healthy,
            "recognized": True,
            "pathogen": None,
            "cause": "No disease detected." if healthy else "Unknown: this class is not in the knowledge base.",
            "symptoms": "",
            "cure": "No treatment needed." if healthy else "Consult a local agricultural extension service.",
            "prevention": "",
        }

    def close(self) -> None:
        self._loaded.network = None


def advisory_lines(result: dict[str, Any]) -> list[str]:
    """Format a prediction as readable lines for the terminal or an overlay."""
    advice = result.get("advice", {})
    confidence = result.get("confidence", 0.0) * 100
    recognized = result.get("recognized", advice.get("recognized", True))
    if not recognized:
        return [
            f"NOT COCONUT ({confidence:.1f}%)",
            advice.get("cause", "This does not look like a coconut leaf."),
            "No diagnosis or treatment is given for this image.",
            advice.get("prevention", "Point the camera at a coconut frond and try again."),
        ]

    status = "HEALTHY" if result.get("healthy") else "INFECTED"
    lines = [
        f"{status} - {result['plant']} / {result['disease']} ({confidence:.1f}%)",
    ]
    if not result.get("healthy"):
        if advice.get("pathogen"):
            lines.append(f"Pathogen: {advice['pathogen']}")
        if advice.get("cause"):
            lines.append(f"Cause: {advice['cause']}")
        if advice.get("symptoms"):
            lines.append(f"Symptoms: {advice['symptoms']}")
        if advice.get("cure"):
            lines.append(f"Cure: {advice['cure']}")
        if advice.get("prevention"):
            lines.append(f"Prevention: {advice['prevention']}")
    else:
        lines.append("No disease detected; keep up normal care.")
    return lines
