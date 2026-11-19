import numpy as np
import pytest


def test_parse_classes_cleans_and_deduplicates_input():
    from detector import parse_classes

    assert parse_classes("person, cup,,  fan ,") == ("person", "cup", "fan")
    assert parse_classes(["dog", " ", "cat"]) == ("dog", "cat")
    assert parse_classes("") is None
    assert parse_classes(None) is None


def test_detector_defaults_are_low_memory():
    from detector import DetectorConfig

    config = DetectorConfig()
    assert config.model == "yolov8n.pt"
    assert config.mediapipe is False
    assert config.world_classes is None
    assert config.imgsz == 640
    # The tool must only ever surface coconut-tree / infection detections.
    assert config.coconut_only is True


def test_detector_passes_size_and_max_det_to_yolo(monkeypatch):
    import detector

    captured = {}

    class FakeYOLO:
        names = {0: "car"}

        def __init__(self, model):
            self.model = model
            self.classes = None

        def set_classes(self, classes):
            self.classes = classes

        def predict(self, image, **kwargs):
            captured.update(kwargs)
            return []

    monkeypatch.setattr("ultralytics.YOLO", FakeYOLO, raising=False)
    instance = detector.UnifiedDetector(
        detector.DetectorConfig(imgsz=416, max_det=7, mediapipe=False)
    )
    instance.predict(np.zeros((32, 48, 3), dtype=np.uint8))
    assert captured["imgsz"] == 416
    assert captured["max_det"] == 7


def test_world_mode_uses_world_checkpoint(monkeypatch):
    import detector

    class FakeWorld:
        names = {0: "fan"}

        def __init__(self, model):
            self.model = model
            self.classes = None

        def set_classes(self, classes):
            self.classes = classes

        def predict(self, image, **kwargs):
            return []

    monkeypatch.setattr("ultralytics.YOLO", FakeWorld, raising=False)
    instance = detector.UnifiedDetector(
        detector.DetectorConfig(world_classes=("fan",), threads=1)
    )
    assert instance.yolo.model == "yolov8s-worldv2.pt"
    assert instance.yolo.classes == ["fan"]


def test_world_mode_rejects_plain_checkpoint(monkeypatch):
    import detector

    class FakePlain:
        names = {0: "car"}

        def __init__(self, model):
            pass

        def predict(self, image, **kwargs):
            return []

    monkeypatch.setattr("ultralytics.YOLO", FakePlain, raising=False)
    with pytest.raises(RuntimeError, match="YOLO-World"):
        detector.UnifiedDetector(detector.DetectorConfig(world_classes=("fan",)))


def test_memory_report_is_positive():
    from memory import format_rss, rss_mb

    value = rss_mb()
    if value is not None:
        assert value > 0
        assert format_rss().endswith("MB")


def test_draw_detections_clamps_boxes():
    from visualize import draw_detections

    frame = np.zeros((40, 50, 3), dtype=np.uint8)
    draw_detections(
        frame,
        [
            {"object": "person", "confidence": 0.9, "bbox": [-10, -10, 999, 999]},
            {"object": "bad", "confidence": 0.1, "bbox": [5, 5, 5, 5]},
            {"object": "missing"},
        ],
    )
    assert frame.any()


def test_headless_camera_loop(monkeypatch):
    import camera

    class FakeCapture:
        def __init__(self):
            self.count = 0

        def isOpened(self):
            return True

        def get(self, _prop):
            return 640

        def set(self, _prop, _value):
            return True

        def read(self):
            self.count += 1
            if self.count > 2:
                return False, None
            return True, np.zeros((48, 64, 3), dtype=np.uint8)

        def release(self):
            pass

    class FakeDetector:
        def __init__(self, config):
            self.config = config

        def predict(self, frame):
            return [{"object": "person", "confidence": 0.9, "bbox": [2, 2, 20, 30]}]

        def close(self):
            pass

    monkeypatch.setattr(camera, "open_camera", lambda *args: FakeCapture())
    monkeypatch.setattr(camera, "UnifiedDetector", FakeDetector)

    summary = camera.run_camera(
        camera.DetectorConfig(imgsz=416, threads=1),
        headless=True,
        max_frames=10,
    )

    assert summary["frames"] == 2
    assert summary["detections_run"] == 2
    assert summary["last_detections"][0]["object"] == "person"


def test_find_data_yaml(tmp_path):
    from download_dataset import find_data_yaml

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: object\n",
        encoding="utf-8",
    )
    assert find_data_yaml(dataset) == dataset / "data.yaml"


def test_kaggle_slug_validation():
    from download_dataset import download_kaggle_dataset

    with pytest.raises(ValueError):
        download_kaggle_dataset("not-a-slug")


def test_prepare_data_yaml_fixes_relative_path(tmp_path):
    from download_dataset import prepare_data_yaml

    dataset = tmp_path / "brain-tumor"
    for split in ("train", "valid"):
        images = dataset / split / "images"
        images.mkdir(parents=True)
        (images / "sample.jpg").write_bytes(b"not-a-real-image")

    data_yaml = tmp_path / "brain-tumor.yaml"
    data_yaml.write_text(
        "path: ../datasets/brain-tumor\n"
        "train: train/images\n"
        "val: valid/images\n"
        "names:\n"
        "  0: negative\n"
        "  1: positive\n",
        encoding="utf-8",
    )

    fixed = prepare_data_yaml(data_yaml)
    assert fixed != data_yaml
    assert fixed.exists()
    text = fixed.read_text(encoding="utf-8")
    assert f"path: {dataset}" in text
    assert "negative" in text


def test_prepare_data_yaml_leaves_correct_file_alone(tmp_path):
    from download_dataset import prepare_data_yaml

    images = tmp_path / "dataset" / "images" / "train"
    images.mkdir(parents=True)
    (images / "sample.jpg").write_bytes(b"not-a-real-image")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        f"path: {tmp_path / 'dataset'}\n"
        "train: images/train\n"
        "names:\n"
        "  0: object\n",
        encoding="utf-8",
    )

    assert prepare_data_yaml(data_yaml) == data_yaml
