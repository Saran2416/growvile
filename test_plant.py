import numpy as np
import pytest

COCONUT_CLASSES = {
    "Bud Rot",
    "Gray Leaf Spot",
    "Leaf Rot",
    "Stem Bleeding",
    "CCI_Caterpillars",
    "Healthy_Leaves",
}
NEGATIVE_CLASS = "Not_Coconut"


def test_split_label_plantvillage_style():
    from plant import split_label

    assert split_label("Tomato___Late_blight") == ("Tomato", "Late blight")
    assert split_label("Corn_(maize)___healthy") == ("Corn (maize)", "Healthy")
    assert split_label("Pepper,_bell___Bacterial_spot") == ("Pepper, bell", "Bacterial spot")
    assert split_label("Cherry_(including_sour)___Powdery_mildew")[0] == "Cherry (sour)"


def test_split_label_flat_coconut_label_uses_knowledge():
    from plant import load_knowledge, split_label

    knowledge = load_knowledge()
    assert split_label("Bud Rot", knowledge) == ("Coconut", "Bud Rot")
    assert split_label("Healthy_Leaves", knowledge) == ("Coconut", "Healthy")
    # A flat label that is not in the knowledge base still gets the plant name.
    assert split_label("Unknown_thing") == ("Coconut", "Unknown thing")


def test_knowledge_base_covers_every_coconut_class():
    from plant import load_knowledge

    knowledge = load_knowledge()
    assert set(knowledge) == COCONUT_CLASSES | {NEGATIVE_CLASS}
    for label, entry in knowledge.items():
        for field in ("plant", "disease", "healthy", "cause", "cure", "prevention"):
            assert field in entry, (label, field)
        assert isinstance(entry["healthy"], bool)
        if label == NEGATIVE_CLASS:
            assert entry["recognized"] is False
            assert entry["plant"] == "Unknown"
        else:
            assert entry["plant"] == "Coconut"
            assert entry.get("recognized", True) is True
        if entry["healthy"]:
            assert entry["disease"] == "Healthy"


def test_knowledge_meta_declares_default_plant():
    from plant import load_meta

    meta = load_meta()
    assert meta["default_plant"] == "Coconut"
    assert meta["classes"] == len(COCONUT_CLASSES) + 1


def test_infected_entries_have_a_cause_and_a_cure():
    from plant import load_knowledge

    for label, entry in load_knowledge().items():
        if entry["healthy"] or not entry.get("recognized", True):
            continue
        assert entry["pathogen"], label
        assert len(entry["cause"]) > 30, label
        assert len(entry["cure"]) > 30, label
        assert len(entry["symptoms"]) > 20, label


def test_not_coconut_has_no_treatment_and_is_not_infected():
    from plant import load_knowledge

    entry = load_knowledge()[NEGATIVE_CLASS]
    assert entry["recognized"] is False
    assert entry["pathogen"] is None
    assert "no treatment" in entry["cure"].lower()
    assert "not a coconut leaf" in entry["disease"].lower()


def test_merge_plan_matches_knowledge_base():
    from plant import load_knowledge
    from prepare_coconut_dataset import CLASS_SOURCES, EXCLUDED_CLASSES

    merged = {name for names in CLASS_SOURCES.values() for name in names}
    assert merged == set(load_knowledge())
    # The ambiguous class must stay out of both the merge plan and the advisory.
    assert set(EXCLUDED_CLASSES).isdisjoint(merged)


def test_missing_checkpoint_has_clear_error(tmp_path):
    from plant import PlantClassifier

    with pytest.raises(FileNotFoundError, match="train_plant.py"):
        PlantClassifier(tmp_path / "missing.pt")


def test_advisory_lines_include_cause_and_cure():
    from plant import advisory_lines

    result = {
        "plant": "Coconut",
        "disease": "Bud Rot",
        "healthy": False,
        "confidence": 0.91,
        "advice": {
            "pathogen": "Phytophthora palmivora",
            "cause": "Wet humid weather.",
            "symptoms": "Pale spear that hangs down.",
            "cure": "Remove and burn the rotted crown.",
            "prevention": "Avoid wounding the crown.",
        },
    }
    text = "\n".join(advisory_lines(result))
    assert "INFECTED" in text
    assert "Phytophthora palmivora" in text
    assert "Remove and burn the rotted crown." in text


def test_advisory_lines_for_healthy_leaf():
    from plant import advisory_lines

    result = {
        "plant": "Coconut",
        "disease": "Healthy",
        "healthy": True,
        "recognized": True,
        "confidence": 0.88,
        "advice": {},
    }
    text = "\n".join(advisory_lines(result))
    assert "HEALTHY" in text
    assert "No disease detected" in text


def test_advisory_lines_for_rejected_leaf():
    from plant import advisory_lines

    result = {
        "plant": "Unknown",
        "disease": "Not a coconut leaf",
        "healthy": False,
        "recognized": False,
        "confidence": 0.93,
        "advice": {
            "recognized": False,
            "cause": "The image does not match the coconut classes.",
            "cure": "No treatment is given.",
            "prevention": "Point the camera at a coconut frond and try again.",
        },
    }
    text = "\n".join(advisory_lines(result))
    assert "NOT COCONUT" in text
    assert "No diagnosis or treatment" in text
    assert "Cure:" not in text
    assert "INFECTED" not in text


def test_negative_training_and_holdout_species_are_disjoint():
    from fetch_negatives import HOLDOUT_SPECIES, TRAIN_SPECIES

    assert set(TRAIN_SPECIES).isdisjoint(HOLDOUT_SPECIES)


def test_trained_checkpoint_has_the_rejection_class():
    from pathlib import Path

    import torch

    path = Path("runs/coconut/best.pt")
    if not path.exists():
        pytest.skip("no trained checkpoint")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert set(COCONUT_CLASSES) <= set(payload["classes"])
    assert NEGATIVE_CLASS in payload["classes"]


def test_center_crop_is_square_and_inside_frame():
    from plant_camera import center_crop

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    crop = center_crop(frame, 0.85)
    assert crop.shape[0] == crop.shape[1]
    assert crop.shape[0] <= 480


def test_find_imagefolder_root_handles_nested_download(tmp_path):
    from train_plant import find_imagefolder_root

    dataset = tmp_path / "Coconut Tree Disease Dataset"
    for class_name in ("Bud Rot", "Stem Bleeding"):
        folder = dataset / class_name
        folder.mkdir(parents=True)
        (folder / "leaf.jpg").write_bytes(b"x")

    assert find_imagefolder_root(tmp_path) == dataset


def test_find_class_dir_handles_nested_download(tmp_path):
    from prepare_coconut_dataset import find_class_dir

    dataset = tmp_path / "nagabushan" / "Coconut Tree Disease Dataset"
    folder = dataset / "Bud Rot"
    folder.mkdir(parents=True)
    (folder / "leaf.jpg").write_bytes(b"x")
    assert find_class_dir(tmp_path / "nagabushan", "Bud Rot") == folder
    assert find_class_dir(tmp_path / "nagabushan", "Missing") is None


def test_stratified_split_keeps_every_class_in_validation():
    from train_plant import stratified_split

    class FakeDataset:
        classes = ["a", "b", "c"]
        samples = [(f"a{i}", 0) for i in range(20)] + [
            (f"b{i}", 1) for i in range(5)
        ] + [(f"c{i}", 2) for i in range(2)]

    train, val = stratified_split(FakeDataset(), 0.2, max_per_class=10, seed=7)
    labels = {FakeDataset.samples[index][1] for index in val}
    assert labels == {0, 1, 2}
    assert len(train) + len(val) == 10 + 5 + 2


def test_class_weights_favour_the_small_class():
    from train_plant import class_weights

    class FakeDataset:
        classes = ["big", "small"]
        samples = [(f"big{i}", 0) for i in range(100)] + [(f"small{i}", 1) for i in range(10)]

    indices = list(range(len(FakeDataset.samples)))
    weights = class_weights(FakeDataset(), indices, "sqrt")
    assert weights[1] > weights[0]
    assert class_weights(FakeDataset(), indices, "none") is None


def test_per_class_metrics_reports_precision_and_recall():
    import torch

    from train_plant import per_class_metrics

    class Constant(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(2))

        def forward(self, images):
            return self.bias.expand(images.shape[0], -1)

    images = torch.zeros(2, 4)
    targets = torch.tensor([0, 1])
    report = per_class_metrics(Constant(), [(images, targets)], ["a", "b"], "cpu")
    assert report["a"]["recall"] == 1.0
    assert report["a"]["precision"] == 0.5
    assert report["b"]["recall"] == 0.0
    assert report["b"]["support"] == 1
