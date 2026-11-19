# growvile

Real-time coconut leaf health checks for images, webcams and a local FastAPI
service. The detector is a small YOLOv8 checkpoint that runs on CPU; in
coconut-only mode it uses the YOLO-World backend with coconut condition labels
and keeps the process around **1.2 GB of RAM**; the coconut classifier reports
the plant, infection status, disease, cause and cure from a curated knowledge
base.

## What "coconut-only" means

The detector runs in coconut-only mode by default: only detections whose label
is one of the coconut conditions (`Bud Rot`, `Gray Leaf Spot`, `Leaf Rot`,
`Stem Bleeding`, `CCI_Caterpillars`, `Healthy_Leaves`) are reported. A COCO
class like `bus`, `person` or `stop sign` is filtered out before it reaches the
caller, so the tool never presents a non-coconut object as a coconut tree or an
infected part.

Because the COCO and Open Images back-ends have no coconut classes, coconut-only
mode detects through the **YOLO-World** open-vocabulary backend with the curated
coconut condition labels (`yolov8s-worldv2.pt`); pass `--classes` to use other
coconut labels.

## Measured memory use

On this machine (4 CPU cores, no CUDA GPU, CPU-only PyTorch):

| Setup | Resident memory (RSS) |
| --- | --- |
| YOLO-World, coconut labels (`yolov8s-worldv2.pt`) | ~1.1–1.2 GB |
| YOLOv8n, COCO-80 (`yolov8n.pt`) | ~450–500 MB |
| YOLOv8n + MediaPipe hands/faces | ~800–820 MB |
| Inference size 480 instead of 640 | similar RAM, lower CPU per frame |

MediaPipe is therefore **off by default**. Enable it with `--mediapipe` only
when you need hand, finger, or face boxes.

## Quick start

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# image
.venv/bin/python inference.py samples/coconut_leaf.jpg

# image, saving an annotated copy next to the input (coconut_annotated.jpg here)
.venv/bin/python inference.py samples/coconut_leaf.jpg --save coconut_annotated.jpg

# live camera
.venv/bin/python camera.py

# report every detection from the underlying model (opt out of coconut-only)
.venv/bin/python camera.py --no-coconut-only
```

The first run downloads the YOLO-World weights (about 25 MB) plus the CLIP text
encoder (about 350 MB); press `q` or `Esc` in the camera window to quit.

## Camera

```bash
# default webcam, 640x480 capture, 480 px inference
.venv/bin/python camera.py

# faster on a slow CPU: skip frames and use a smaller inference size
.venv/bin/python camera.py --imgsz 416 --every 2

# second camera, mirrored preview, record a video
.venv/bin/python camera.py --camera 1 --mirror --output demo.mp4

# headless smoke test (no display needed)
.venv/bin/python camera.py --headless --frames 3
```

Useful flags:

| Flag | Purpose |
| --- | --- |
| `--coconut-only` | Only report known coconut conditions (default) |
| `--no-coconut-only` | Report every detection from the underlying model |
| `--broad` | 601 Open Images classes instead of 80 COCO classes (overridden by coconut-only unless `--no-coconut-only`) |
| `--imgsz 416` | Smaller inference size, faster on CPU |
| `--every 2` | Detect every second frame and reuse the boxes |
| `--threads 2` | Cap PyTorch CPU threads |
| `--max-rss 700` | Stop if the process grows above 700 MB |
| `--mediapipe` | Add hands, fingers and faces (+300–400 MB) |
| `--classes "Bud Rot, Gray Leaf Spot, Healthy"` | Open-vocabulary labels via YOLO-World |
| `--world-model path.pt` | Custom YOLO-World checkpoint to use with `--classes` |
| `--output demo.mp4` | Record the annotated stream to an MP4 file |
| `--headless` | No OpenCV window (servers, SSH, tests) |

The status line shows FPS, inference time, RAM and the active backend. Detection
runs in the main loop and boxes are drawn in place, so no extra full-size frame
is allocated per detection.

## Open-vocabulary labels

The coconut-only detector uses the **YOLO-World** open-vocabulary backend with
the curated coconut condition labels; install the CLIP text encoder once, then
pass labels:

```bash
.venv/bin/python -m pip install "git+https://github.com/ultralytics/CLIP.git"
.venv/bin/python inference.py samples/coconut_leaf.jpg --classes "Bud Rot, Gray Leaf Spot, Healthy"
```

The first run downloads `yolov8s-worldv2.pt` (~25 MB) plus the CLIP text
encoder (~350 MB), and uses roughly **1.1–1.2 GB** of RAM while running. Use
`--world-model path/to/world.pt` (on `camera.py` or `inference.py`) for a
different YOLO-World checkpoint; `--model` works too when combined with
`--classes`.

## Train on a Kaggle dataset

Ultralytics expects the standard YOLO layout:

```
datasets/my-objects/
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt
  data.yaml
```

Download a YOLO-format dataset from Kaggle:

```bash
.venv/bin/python -m pip install kaggle
.venv/bin/python download_dataset.py owner/dataset-slug --dest datasets
```

Kaggle credentials: either `~/.kaggle/kaggle.json` (from **Account → Create New
Token**) or an existing `~/.kaggle/access_token`. Check it with
`.venv/bin/kaggle config view`.

Many Kaggle uploads keep a `path:` from the author's machine, for example
`path: ../datasets/brain-tumor`. If the unzipped folder is somewhere else,
`download_dataset.py`/`train.py` detect that and write a corrected
`<name>.resolved.yaml` next to the original, so training does not fail with
"dataset not found".

Fine-tune with low-memory defaults (`yolov8n`, batch 8, 2 workers, no cache):

```bash
.venv/bin/python train.py --kaggle owner/dataset-slug --epochs 50
# or
.venv/bin/python train.py --data datasets/my-objects/data.yaml --epochs 50
```

The best checkpoint is written to `runs/detect/train/weights/best.pt`. Use it
directly:

```bash
.venv/bin/python inference.py samples/coconut_leaf.jpg --model runs/detect/train/weights/best.pt
.venv/bin/python camera.py --model runs/detect/train/weights/best.pt
```

If your dataset has no `data.yaml`, create one with the standard YOLO layout and
list the coconut class names in index order, then point `path` at the dataset
folder. For a larger model change `--model yolov8s.pt`, but expect more RAM and
slower frames.

## Training on CPU (no GPU)

`train.py` detects the absence of CUDA and uses `--device cpu` automatically.
Training works, but a CPU is much slower than a GPU. Inference with the trained
model is still fast; only the training step is expensive.

A practical CPU command:

```bash
.venv/bin/python train.py \
  --data datasets/my-objects/data.yaml \
  --imgsz 416 \
  --batch 4 \
  --workers 2 \
  --epochs 30 \
  --patience 8 \
  --freeze 10
```

Speed/quality trade-offs:

| Flag | Effect |
| --- | --- |
| `--imgsz 416` | Fastest CPU option; use 480 for a compromise, 640 for quality |
| `--batch 4` | Small memory footprint; raise to 8 if RAM allows |
| `--freeze 10` | Freezes early backbone layers, faster but slightly less accurate |
| `--fraction 0.5` | Quick experiment on half the images; use `1.0` for the real run |
| `--workers 2` | Two loader processes is a good fit for a 4-core CPU |
| `--cache` | Faster epochs, but loads the dataset into RAM |

Expect **hours** for a few hundred images at 416 px and 30 epochs; 640 px with
thousands of images can take a day or more. Start with `--epochs 5 --fraction 0.2`
to check that the dataset and labels are valid before committing to a long run.
If you have access to Google Colab or a Kaggle notebook with a free GPU, you can
train there and copy `runs/detect/train/weights/best.pt` back to this machine.

## Coconut leaf health (plant classifier)

Object detection tells you *where* a coconut palm is, but not what is wrong
with it. `plant_camera.py` adds a small **classifier** that looks at foliage
and reports the plant, whether it is infected, the disease, its cause and a
cure. The cause/cure text comes from a hand-written knowledge base
(`data/coconut_diseases.json`), so the model cannot invent a treatment.

Classes: `Bud Rot`, `Gray Leaf Spot`, `Leaf Rot`, `Stem Bleeding`,
`CCI_Caterpillars` (black-headed caterpillar, a pest), `Healthy_Leaves` and an
explicit `Not_Coconut` rejection class trained on real non-coconut leaf photos.

### One-time data preparation

```bash
# downloads the two Kaggle datasets into datasets/coconut/raw/
.venv/bin/python -m pip install kaggle
.venv/bin/kaggle datasets download -d nagabushan/coconut-tree-disease-dataset \
    -p datasets/coconut/raw/nagabushan --unzip
.venv/bin/kaggle datasets download -d amandapriyashantha/coconut-leaf-diseases \
    -p datasets/coconut/raw/amandapriyashantha --unzip

# add the Not_Coconut class (~2,200 PlantDoc leaves from species the model may
# learn) and a held-out set of unseen species + Makerere beans for testing
.venv/bin/python fetch_negatives.py

# add 460 featureless frames (flat colour, gradients, noise, stripes) so a
# blank wall or lens cap is rejected instead of scored as a healthy leaf
.venv/bin/python make_synthetic_negatives.py

# merge them into one ImageFolder tree (hard links, ~9.1k images)
.venv/bin/python prepare_coconut_dataset.py
```

`prepare_coconut_dataset.py` deliberately drops the `Bud Root Dropping` class:
the source dataset gives no pathogen and there is no published aetiology, so a
reliable cause and cure cannot be stated.

`fetch_negatives.py` pulls real-world leaf photos from the public
[PlantDoc](https://huggingface.co/datasets/LamTNguyen/PlantDoc) and
[Makerere beans](https://huggingface.co/datasets/AI-Lab-Makerere/beans)
datasets. Training uses only apple, corn, grape, potato, tomato and pepper
leaves; blueberry, cherry, peach, raspberry, soybean, squash, strawberry and
bean leaves are kept out of training and used to measure rejection on unseen
species.

### Train

The shipped checkpoint is trained in two stages on CPU. Stage 1 learns the
seven classes:

```bash
.venv/bin/python train_plant.py --data datasets/coconut/merged \
    --epochs 12 --freeze-epochs 3 --max-per-class 1000 --imgsz 224 \
    --batch 32 --workers 2 --threads 4 --output runs/coconut
```

Stage 2 continues from that checkpoint with stronger augmentation:

```bash
.venv/bin/python train_plant.py --data datasets/coconut/merged \
    --epochs 6 --freeze-epochs 0 --max-per-class 1000 --imgsz 224 \
    --batch 32 --workers 2 --threads 4 --lr-finetune 5e-5 \
    --augment strong --init runs/coconut/best.pt --output runs/coconut
```

The first epochs train only the classification head with the backbone frozen,
then the whole network is fine-tuned at a lower learning rate. On a 4-core CPU
one epoch takes about 3 minutes. Small classes are up-weighted with
`--class-weights sqrt` (default); use `--class-weights none` to disable it.
`--augment strong` adds vertical flips, stronger colour jitter, grayscale,
blur and random erasing, which is what makes a continuation run useful.
Checkpoints are selected by top-1 accuracy first, then by lower validation
loss. Per-class precision/recall/F1 is written to `runs/coconut/metrics.json`.

The shipped run reached **99.19% top-1** on the internal 20% validation split
(4,466 train / 1,117 validation images: 917 coconut plus 200 `Not_Coconut`,
92 of them synthetic). Coconut-only accuracy is **99.24%** (910/917); the
remaining errors are Leaf Rot and Gray Leaf Spot leaves that look alike. That
split contains many near-duplicate photos of the same palms, so treat it as
"it learned these classes", not as a field-accuracy guarantee.

The harder test is rejection of species the model never trained on. Run
`evaluate_rejection.py` to reproduce it: on **746 held-out leaves from eight
unseen species** (blueberry, cherry, peach, raspberry, soybean, squash,
strawberry and Makerere beans), **99.7%** were rejected as `Not_Coconut`
(744/746; per species 99.0–100%), while coconut recall on the validation
split stayed **99.4%** (323/325). A missed rejection is reported as a coconut
disease, so keep the frond in frame and treat low-confidence results with
caution.

Three rounds of data and training changes each fixed a measured failure:

1. Adding the first 800 PlantDoc negatives made out-of-scope leaves rejectable.
2. Using all 2,224 PlantDoc negatives raised rejection from 97.5% to 99.5%.
3. Adding 460 synthetic featureless frames and the stage-2 continuation run
   raised rejection to 99.7% and fixed the blank-frame case: a flat wall,
   solid colour or lens cap is now reported as `NOT COCONUT` (69–90%)
   instead of `HEALTHY`. Blurred and very dark coconut leaves are still
   classified correctly.

The live camera shows `Uncertain` below `--threshold 0.5`, so use that state
or check the printed confidence before acting on a result.

### Run

```bash
# one image: prints plant, disease, cause, symptoms, cure, prevention
.venv/bin/python plant_camera.py --image leaf.jpg

# live camera: overlay status, full advisory printed when the class changes
.venv/bin/python plant_camera.py --every 5

# machine-readable
.venv/bin/python plant_camera.py --image leaf.jpg --json
```

Hold one frond or a group of leaflets so it fills most of the frame. This is a
classifier, not a detector: in live mode the centre square is cropped
automatically (`--crop 0.85` by default), while `--image` uses the whole image.
Tune `--crop` if the leaf is cut off or too much background remains.

| Flag | Purpose |
| --- | --- |
| `--crop 0.85` | Centre-square fraction used in live mode (ignored for `--image`) |
| `--every 5` | Classify every N frames and reuse the last result |
| `--threshold 0.5` | Confidence below this shows `Uncertain` in the overlay |
| `--top-k 3` | Number of alternative predictions listed for `--image` |
| `--headless`, `--frames 3` | No window / stop after N frames |
| `--json` | Print the single-image result as JSON |

The checkpoint is about 6 MB and per-frame classification is light enough to
run alongside the YOLO detector. Importing `torch`/`torchvision` accounts for
most of the memory cost (~330 MB extra RSS when this classifier is the first
model loaded in a fresh process); that cost is largely avoided when the YOLO
detector has already loaded them.

If the image is not a coconut leaf, the output starts with `NOT COCONUT` and
prints an explanation plus "No diagnosis or treatment is given for this
image." — no disease cause or cure is reported. It explicitly refuses to
diagnose out-of-scope input. The `recognized` field in `--json` output and in
the camera summary is `false` for those results.

## API

```bash
.venv/bin/python -m uvicorn app:app --reload
curl http://127.0.0.1:8000/health
curl -X POST -F file=@samples/coconut_leaf.jpg http://127.0.0.1:8000/predict
```

Environment variables: `YOLO_MODEL`, `BROAD=1`, `CONFIDENCE`, `DEVICE`,
`IMGSZ`, `MEDIAPIPE=1`, `WORLD_CLASSES`, `COCONUT_ONLY`, `THREADS`, `MAX_DET`,
`MAX_UPLOAD_MB` (default 20 MB). `COCONUT_ONLY=0` turns the coconut-only filter
off for the API.

`/predict` returns `detections` (object, confidence, bbox), `backends`,
`model`, `imgsz`, `memory_mb`, `filename` and `image_size`; `/health` returns
`{"status": "ok"}`. The plant classifier is a separate CLI and is not exposed
through this API.

## Tests

```bash
.venv/bin/python -m pytest -q
# 37 passed
```

The tests do not require a camera or network. They cover the output contract,
class parsing, memory reporting, drawing, the rejection class, and a fake
headless camera loop.

## Security checks

```bash
.venv/bin/python -m pip install bandit pip-audit
.venv/bin/python -m bandit -r . -x ./.venv,./datasets,./runs,./models,./.git
.venv/bin/python -m pip_audit -r requirements.txt
```

The latest run found **no medium or high severity issues** and **no known
dependency vulnerabilities** (the full environment was also audited after
upgrading Pillow, urllib3, filelock and setuptools). Remaining low-severity
notes are ordinary test assertions, the seeded RNG used for dataset splits,
best-effort cleanup blocks, and the Kaggle CLI wrapper, which runs a validated
argument list without a shell. Model checkpoints are loaded with
`torch.load(..., weights_only=True)`, so a checkpoint cannot execute arbitrary
pickle code, and `/predict` streams at most `MAX_UPLOAD_MB + 1` bytes into
memory before rejecting an oversized upload.

## Files

| File | Role |
| --- | --- |
| `detector.py` | Model loading, YOLO/MediaPipe inference, response contract |
| `camera.py` | Low-memory real-time webcam loop |
| `inference.py` | Single-image and webcam CLI |
| `visualize.py` | Box and status overlay drawing |
| `memory.py` | RSS reporting |
| `app.py` | FastAPI `/predict` and `/health` |
| `download_dataset.py` | Kaggle download helper |
| `train.py` | Memory-friendly fine-tuning |
| `plant.py` | Coconut leaf classifier and curated advisory lookup |
| `train_plant.py` | CPU training for the leaf classifier |
| `plant_camera.py` | Live/webcam or single-image coconut health check |
| `prepare_coconut_dataset.py` | Merge the raw coconut Kaggle datasets |
| `fetch_negatives.py` | Download non-coconut leaves for rejection training/testing |
| `make_synthetic_negatives.py` | Generate featureless frames so blank input is rejected |
| `evaluate_rejection.py` | Measure non-coconut rejection and coconut recall |
| `data/coconut_diseases.json` | Cause/cure knowledge base, including `Not_Coconut` |
| `requirements.txt` | Core runtime dependencies |
| `requirements-optional.txt` | MediaPipe, Kaggle and CLIP extras |
| `test_detector.py`, `test_helpers.py`, `test_plant.py` | Pytest suite (no camera or network needed) |

## Limitations

- Coconut-only mode means the tool reports just the coconut conditions above;
  use `--no-coconut-only` to inspect arbitrary objects with the COCO or Open
  Images models.
- COCO has no dedicated coconut class, so a plain `yolov8n.pt` backend reports no
  coconut boxes until a coconut-trained detection checkpoint is used; the
  coconut leaf classifier (`plant_camera.py`) is what diagnoses the plant from
  foliage.
- Open-vocabulary and MediaPipe modes use noticeably more memory than plain COCO.
- A CPU-only machine will not reach high frame rates with larger checkpoints;
  stay on `yolov8n` and an inference size of 416–480.
- The coconut classifier works on close-up foliage, not whole scenes. It knows
  six coconut classes plus `Not_Coconut`; rejection is a learned class, not a
  guarantee, so an unusual non-coconut leaf can still be misclassified. Run
  `evaluate_rejection.py` to see the measured rejection rate on unseen species.
  Cause/cure advice is general guidance from published extension material, not
  a diagnosis of your specific palm; confirm with a local agricultural
  extension service before treating.
