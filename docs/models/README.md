# On-device model slot

`camera-classifier.onnx` in this folder is what the web app loads by default.

**The committed file is the real trained camera classifier** —
`camera_classifier_v4` (`best.pt`), recovered from this repository's git
history (it was part of the first commit as
`YOLOv8_Medium/runs/camera_classifier_v4/weights/best.pt` and later deleted)
and exported to ONNX at its training resolution of 640 px:

```bash
yolo export model=best.pt format=onnx imgsz=640   # classes: {0: camera, 1: no_camera}
```

Verified against the 12 manual test images in `YOLOv8_Medium/images/` with
the web app's exact preprocessing: **12/12 correct**, matching the results
reported in `saves.txt` and the technical report. (A 224 px export was also
evaluated: 11/12 — it re-introduces the KFC-bucket false positive — so the
640 px export ships despite being ~8× slower per frame.)

To swap in a newer training run, export it the same way and overwrite the
file — two ways to give the web app a model:

1. **No-commit way (fastest):** open the app, tap **⚙ Settings → Load YOLO
   model (.onnx)** and pick your exported `best.onnx`. The model is stored in
   the browser (IndexedDB) and survives reloads — nothing is uploaded
   anywhere.
2. **Repo way (everyone gets it automatically):** commit the file at the path
   below.

The web app looks for a YOLOv8 classification model at:

```
docs/models/camera-classifier.onnx
```

If the file exists, detection runs **fully on-device** in the browser via
[onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/) — no image ever
leaves the phone/laptop. If it is missing, the app falls back to the
Roboflow-hosted model configured in the app's ⚙ Settings.

## Where is the trained model?

The repository only contains the **base ImageNet weights**
(`YOLOv8_Medium/Scripts/yolov8m-cls.pt` / `.onnx` — check their metadata: 1000
ImageNet classes, "trained on ../datasets/imagenet"). The actual trained
camera classifier (`runs/camera_classifier_v4/weights/best.pt` referenced by
`YOLOv8_Medium/Scripts/test.py`) was never pushed. Export it from the machine
where training ran.

## How to export

On the machine that has `best.pt`:

```bash
pip install ultralytics
yolo export model=runs/camera_classifier_v4/weights/best.pt format=onnx
```

Then copy the resulting `best.onnx` here:

```bash
cp runs/camera_classifier_v4/weights/best.onnx docs/models/camera-classifier.onnx
git add docs/models/camera-classifier.onnx
git commit -m "Add trained camera classifier for the web app"
git push
```

Notes:

- **Input size** — the app auto-probes 224 / 320 / 640 px inputs, so exporting
  with the training `imgsz` (640) or the default (224) both work. A 224 px
  export is ~4× faster in the browser; add `imgsz=224` to the export command
  if you want that.
- **Class order** — the app assumes index 0 = `camera`, index 1 = `no_camera`
  (Ultralytics sorts class folders alphabetically, which matches your
  dataset). If your export differs, change `CAMERA_CLASS_INDEX` in
  `docs/app.js`.
- **File size** — a YOLOv8m-cls export is ~65 MB (fine for git and GitHub
  Pages, but slow on mobile data). Consider training/exporting a `yolov8n-cls`
  variant (~10 MB) for field use, or quantizing:

  ```python
  from onnxruntime.quantization import quantize_dynamic, QuantType
  quantize_dynamic("best.onnx", "camera-classifier.onnx", weight_type=QuantType.QUInt8)
  ```
