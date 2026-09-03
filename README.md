# Open Object Detector

An end-to-end, free/open-source detector for images, video frames, and a local
FastAPI service.

## What is covered

The default `yolov8n.pt` model is pretrained on COCO and covers `person`,
`bicycle`, `car`, `motorcycle`, `bus`, `truck`, `bottle`, `cup`, `chair`, `tv`,
`laptop`, `cell phone`, `backpack`, `book`, `keyboard`, `mouse`, `remote`,
`clock`, `scissors`, and `handbag`. COCO does not have a general `fan` class,
individual fingers, or a dedicated head class. MediaPipe Hands adds a `hand`,
five finger-tip entries, and 21 landmarks; MediaPipe Face Detection adds
`face`. Those backends are reported in the response under `backends`.

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
.venv/bin/python inference.py path/to/image.jpg
.venv/bin/uvicorn app:app --reload
curl -X POST -F file=@path/to/image.jpg http://127.0.0.1:8000/predict
```

The first run downloads the open YOLO weights. Webcam mode is available with
`.venv/bin/python inference.py 0 --webcam`; press `q` to exit.
