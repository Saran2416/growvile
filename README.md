# Open Object Detector

An end-to-end, free/open-source detector for images, video frames, and a local
FastAPI service.

## What is covered

The default `yolov8n.pt` model is pretrained on 80 COCO classes, including
`person`, `bus`, cars, bicycles, bottles, phones, chairs, and many other common
objects. The detector returns every class produced by the configured YOLO
model. COCO does not have clothing categories, a general `fan` class,
individual fingers, or a dedicated head class. MediaPipe Hands adds a `hand`,
five finger-tip entries, and 21 landmarks; MediaPipe Face Detection adds
`face`. Those backends are reported in the response under `backends`.

For clothes, fans, or other categories outside COCO, train or download a YOLO
model containing those labels, then pass the real weights file with `--model`
or `YOLO_MODEL`. For example, after training a model whose weights are saved
at `runs/detect/train/weights/best.pt`, run:

```bash
.venv/bin/python inference.py samples/bus.jpg --model runs/detect/train/weights/best.pt
```

Do not run `path/to/custom-model.pt` or `image.jpg` literally; those are
placeholders and must be replaced with files that exist.

Fan detection needs a labeled dataset and fine-tuning. No fan weights are
silently claimed here: provide a YOLO dataset and train with
`yolo detect train model=yolov8n.pt data=fan.yaml epochs=30 imgsz=640`, then set
`YOLO_MODEL=runs/detect/train/weights/best.pt`. A bottle subtype model likewise
requires labeled `water_bottle` and `sauce_bottle` crops. Train one with
`.venv/bin/python train_bottle_classifier.py data/bottles --output bottle_subtypes.pt`
and set `BOTTLE_SUBTYPE_MODEL=bottle_subtypes.pt`; output then includes both
`subtype` and `subtype_confidence`.

## Run

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python inference.py samples/bus.jpg
.venv/bin/uvicorn app:app --reload
curl -X POST -F file=@samples/bus.jpg http://127.0.0.1:8000/predict
```

Replace `samples/bus.jpg` with the path to an image on your machine when using
your own file. The API health check is available at
`http://127.0.0.1:8000/health`; the root URL has no page.

The first run downloads the open YOLO weights. Webcam mode is available with
`.venv/bin/python inference.py 0 --webcam`; press `q` to exit.
