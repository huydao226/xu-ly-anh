# Training Pipeline

This folder turns the demo into a practical fine-tune workflow for the anti-cheat topic.

## What is included

- `data/raw_sessions/`
  - captured webcam sessions from the running demo
- `data/annotations/`
  - your manual labels for object boxes and behavior segments
- `data/processed/`
  - generated YOLO datasets and temporal sequence datasets
- `scripts/prepare_yolo_dataset.py`
  - converts labeled frames into the Ultralytics dataset layout
- `scripts/build_temporal_dataset.py`
  - converts captured metadata plus behavior labels into temporal samples
- `scripts/train_temporal_model.py`
  - trains a baseline LSTM classifier for cheating behavior

## Recommended environment

Reuse the backend virtual environment because `ultralytics`, `torch`, and `opencv-python` are already installed there.

```bash
cd /Users/huy.dao/XuLyAnh/anti-cheat-demo/backend
source .venv/bin/activate
```

## 1. Capture raw sessions

Enable dataset capture in:

```text
/Users/huy.dao/XuLyAnh/anti-cheat-demo/backend/.env
```

Set:

```bash
DATASET_CAPTURE_ENABLED=true
DATASET_CAPTURE_ROOT=/Users/huy.dao/XuLyAnh/anti-cheat-demo/training/data/raw_sessions
```

Then restart the backend and run demo sessions.

Each session will create:

```text
training/data/raw_sessions/<session_id>/
  images/
  metadata.jsonl
  session_manifest.json
```

## 2. Annotate object detection data

Create or copy:

```text
/Users/huy.dao/XuLyAnh/anti-cheat-demo/training/data/annotations/object_annotations.csv
```

Use the sample schema from:

```text
/Users/huy.dao/XuLyAnh/anti-cheat-demo/training/data/annotations/object_annotations.sample.csv
```

Fields:

- `image_path`
- `label`
- `x_min`
- `y_min`
- `x_max`
- `y_max`
- `split`

Recommended labels:

- `phone`
- `notes`
- `second_person`
- `calculator`

## 3. Build the YOLO dataset

```bash
cd /Users/huy.dao/XuLyAnh/anti-cheat-demo
python3 training/scripts/prepare_yolo_dataset.py \
  --annotations training/data/annotations/object_annotations.csv \
  --output-dir training/data/processed/yolo_exam_v1
```

The script will generate:

```text
training/data/processed/yolo_exam_v1/
  images/train
  images/val
  images/test
  labels/train
  labels/val
  labels/test
  data.yaml
```

## 4. Fine-tune YOLO

```bash
cd /Users/huy.dao/XuLyAnh/anti-cheat-demo
yolo task=detect mode=train \
  model=backend/yolov8n.pt \
  data=training/data/processed/yolo_exam_v1/data.yaml \
  epochs=50 \
  imgsz=640 \
  project=training/models \
  name=yolo_exam_v1
```

## 5. Annotate temporal behavior data

Create:

```text
/Users/huy.dao/XuLyAnh/anti-cheat-demo/training/data/annotations/behavior_segments.csv
```

Use the sample schema from:

```text
/Users/huy.dao/XuLyAnh/anti-cheat-demo/training/data/annotations/behavior_segments.sample.csv
```

Fields:

- `session_id`
- `start_frame`
- `end_frame`
- `label`
- `split`
- `notes`

Recommended labels:

- `normal`
- `look_away`
- `looking_down`
- `phone_use`
- `multiple_faces`

## 6. Build the temporal dataset

```bash
cd /Users/huy.dao/XuLyAnh/anti-cheat-demo
python3 training/scripts/build_temporal_dataset.py \
  --sessions-root training/data/raw_sessions \
  --labels training/data/annotations/behavior_segments.csv \
  --output training/data/processed/temporal_sequences.jsonl
```

## 7. Train the baseline LSTM

```bash
cd /Users/huy.dao/XuLyAnh/anti-cheat-demo
python3 training/scripts/train_temporal_model.py \
  --dataset training/data/processed/temporal_sequences.jsonl \
  --output-dir training/models/temporal_lstm_v1 \
  --epochs 20
```

The trainer writes:

- `best_model.pt`
- `label_map.json`
- `metrics.json`

## What data should come from Asian participants

For your topic, the most important data is still the custom dataset collected from Vietnamese or other Asian participants in the same laptop-camera setup as the demo.

Recommended capture coverage:

- 20 to 30 participants minimum
- different genders and glasses/no-glasses cases
- different skin tones and lighting conditions
- short natural motions
- real cheating-like motions
- different laptop camera heights and seating distances

See [FINE_TUNE_GUIDE_VI.md](/Users/huy.dao/XuLyAnh/anti-cheat-demo/docs/FINE_TUNE_GUIDE_VI.md) for the Vietnamese guide.
