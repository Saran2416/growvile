import numpy as np


def test_output_contract_without_model_download(monkeypatch):
    import detector

    class FakeYOLO:
        names = {0: "car"}

        def __init__(self, model):
            self.model = model
            self.classes = None

        def set_classes(self, classes):
            self.classes = classes

        def predict(self, *args, **kwargs):
            return []

    monkeypatch.setattr("ultralytics.YOLO", FakeYOLO, raising=False)
    instance = detector.UnifiedDetector()
    result = instance.predict_with_status(np.zeros((32, 48, 3), dtype=np.uint8))
    assert result["backends"]["open_vocabulary"] is True
    assert isinstance(result["detections"], list)


NON_COCONUT_DETECTIONS = [
    {"object": "bus", "confidence": 0.91, "bbox": [10, 10, 90, 90]},
    {"object": "person", "confidence": 0.82, "bbox": [10, 10, 90, 90]},
    {"object": "stop sign", "confidence": 0.71, "bbox": [10, 10, 90, 90]},
    {"object": "chair", "confidence": 0.60, "bbox": [10, 10, 90, 90]},
    {"object": "Bud Rot", "confidence": 0.55, "bbox": [10, 10, 90, 90]},
]


def _fake_coconut_backends(monkeypatch, detections):
    import detector

    monkeypatch.setattr(detector.UnifiedDetector, "_yolo_detections", lambda self, image: list(detections))
    monkeypatch.setattr(detector.UnifiedDetector, "_specialized_detections", lambda self, image: [])


def test_coconut_only_drops_non_coconut_detections(monkeypatch):
    import detector

    _fake_coconut_backends(monkeypatch, NON_COCONUT_DETECTIONS)
    instance = detector.UnifiedDetector(detector.DetectorConfig(coconut_only=True))
    result = instance.predict_with_status(np.zeros((32, 48, 3), dtype=np.uint8))
    assert [item["object"] for item in result["detections"]] == ["Bud Rot"]


def test_coconut_only_keeps_coconut_healthy_leaf_too(monkeypatch):
    import detector

    _fake_coconut_backends(
        monkeypatch,
        [
            {"object": "Gray Leaf Spot", "confidence": 0.9, "bbox": [0, 0, 5, 5]},
            {"object": "Healthy_Leaves", "confidence": 0.8, "bbox": [0, 0, 5, 5]},
            {"object": "bus", "confidence": 0.7, "bbox": [0, 0, 5, 5]},
        ],
    )
    instance = detector.UnifiedDetector(detector.DetectorConfig(coconut_only=True))
    result = instance.predict(np.zeros((32, 48, 3), dtype=np.uint8))
    assert {item["object"] for item in result} == {"Gray Leaf Spot", "Healthy_Leaves"}


def test_coconut_only_can_be_disabled(monkeypatch):
    import detector

    _fake_coconut_backends(monkeypatch, NON_COCONUT_DETECTIONS)
    instance = detector.UnifiedDetector(detector.DetectorConfig(coconut_only=False))
    result = instance.predict_with_status(np.zeros((32, 48, 3), dtype=np.uint8))
    assert [item["object"] for item in result["detections"]] == [
        "bus",
        "person",
        "stop sign",
        "chair",
        "Bud Rot",
    ]
