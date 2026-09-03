import numpy as np


def test_output_contract_without_model_download(monkeypatch):
    import detector

    class FakeYOLO:
        names = {0: "car"}

        def __init__(self, model):
            self.model = model

        def predict(self, *args, **kwargs):
            return []

    monkeypatch.setattr("ultralytics.YOLO", FakeYOLO, raising=False)
    instance = detector.UnifiedDetector()
    result = instance.predict_with_status(np.zeros((32, 48, 3), dtype=np.uint8))
    assert result["backends"]["yolo_coco"] is True
    assert isinstance(result["detections"], list)
