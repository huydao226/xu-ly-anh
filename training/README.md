# Training Pipeline

This folder turns the demo into a practical fine-tune workflow for the anti-cheat topic.

## Recommended research pipeline now

The repo now supports an `OEP-first` flow:

1. `Stage 1`
   - train a baseline temporal model on `OEP multi-view`
   - use both webcam and wearcam streams
2. `Stage 2`
   - fine-tune or retrain on `OEP webcam-only`
   - keep only the front face camera view
3. `Stage 3`
   - test on a held-out `OEP webcam-only` split
   - this is the formal benchmark before self-demo
4. `Stage 4`
   - self-test with the live laptop webcam demo
   - use this for qualitative validation and presentation

## What is included

- `data/raw_sessions/`
  - captured webcam sessions from the running demo
- `data/external/oep_multiview/raw/`
  - downloaded OEP or similar public multi-view reference datasets
- `scripts/import_oep_reference.py`
  - scans OEP, keeps the webcam view as reference input, and exports manifest CSV files
- `scripts/build_oep_temporal_dataset.py`
  - converts OEP video segments into temporal feature sequences for `multiview` or `webcam` training
- `data/external/single_camera_finetune/raw_videos/`
  - raw one-camera videos for final fine-tuning
- `data/external/single_camera_finetune/raw_images/`
  - optional still images for object detector fine-tuning
- `data/external/single_camera_finetune/annotations/`
  - raw external labels or notes before conversion
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

## Why the final fine-tune must still use one-camera data

Datasets such as OEP are valuable because they are close to the online-proctoring domain. However, they are closer to a richer multi-view setting than the current project.

That means:

- the public dataset can help us understand the problem space
- it can provide reference cheating and non-cheating patterns
- but it does not perfectly match the current `single laptop camera` deployment

For this repo, the final model should therefore be fine-tuned on one-camera data that matches the real demo condition.

## Exact dataset drop locations

If you want me to run training after you download data, put the files here:

```text
/Users/huy.dao/XuLyAnh/anti-cheat-demo/training/data/external/oep_multiview/raw
/Users/huy.dao/XuLyAnh/anti-cheat-demo/training/data/external/single_camera_finetune/raw_videos
/Users/huy.dao/XuLyAnh/anti-cheat-demo/training/data/external/single_camera_finetune/raw_images
/Users/huy.dao/XuLyAnh/anti-cheat-demo/training/data/external/single_camera_finetune/annotations
```

Quick readiness check:

```bash
cd /Users/huy.dao/XuLyAnh/anti-cheat-demo
python3 training/scripts/check_dataset_ready.py
```

If the OEP archive has already been extracted, you can also generate a manifest immediately:

```bash
cd /Users/huy.dao/XuLyAnh/anti-cheat-demo
python3 training/scripts/import_oep_reference.py
```

Then build the actual OEP training dataset:

```bash
cd /Users/huy.dao/XuLyAnh/anti-cheat-demo/backend
source .venv/bin/activate
cd /Users/huy.dao/XuLyAnh/anti-cheat-demo

python training/scripts/build_oep_temporal_dataset.py \
  --mode multiview \
  --output training/data/processed/oep_multiview_temporal.jsonl

python training/scripts/build_oep_temporal_dataset.py \
  --mode webcam \
  --output training/data/processed/oep_webcam_temporal.jsonl
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

## 2. Recommended data strategy for this repo

Use two sources together:

1. `Current training source`
   - OEP
   - place it in `training/data/external/oep_multiview/raw`
   - use it first for Stage 1, Stage 2, and Stage 3
   - this is enough to start training immediately

2. `Later deployment adaptation source`
   - one-camera real laptop data
   - place it in `training/data/external/single_camera_finetune/raw_videos`
   - add this later when you want the final model to adapt closer to your own demo condition

## 2.1 OEP-first training commands

Stage 1, multi-view baseline:

```bash
cd /Users/huy.dao/XuLyAnh/anti-cheat-demo/backend
source .venv/bin/activate
cd /Users/huy.dao/XuLyAnh/anti-cheat-demo

python training/scripts/build_oep_temporal_dataset.py \
  --mode multiview \
  --output training/data/processed/oep_multiview_temporal.jsonl

python training/scripts/train_temporal_model.py \
  --dataset training/data/processed/oep_multiview_temporal.jsonl \
  --output-dir training/models/oep_multiview_lstm_v1 \
  --epochs 12
```

Stage 2 and Stage 3, webcam-only fine-tune plus holdout test:

```bash
cd /Users/huy.dao/XuLyAnh/anti-cheat-demo/backend
source .venv/bin/activate
cd /Users/huy.dao/XuLyAnh/anti-cheat-demo

python training/scripts/build_oep_temporal_dataset.py \
  --mode webcam \
  --output training/data/processed/oep_webcam_temporal.jsonl

python training/scripts/train_temporal_model.py \
  --dataset training/data/processed/oep_webcam_temporal.jsonl \
  --output-dir training/models/oep_webcam_lstm_v1 \
  --epochs 12
```

The `train/val/test` split is assigned at the subject level, so Stage 3 uses a held-out OEP webcam-only test split automatically.

## 3. Annotate object detection data

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

OEP does not arrive in this bounding-box format. If you want to use OEP frames for detector training, first extract candidate frames from the webcam video, then annotate the object boxes manually or with a separate labeling tool.

## 4. Build the YOLO dataset

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

## 5. Fine-tune YOLO

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

## 6. Annotate temporal behavior data

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

If you want a quick OEP reference manifest first:

```bash
cd /Users/huy.dao/XuLyAnh/anti-cheat-demo
python3 training/scripts/import_oep_reference.py
```

That command writes:

```text
training/data/external/oep_multiview/notes/oep_subject_manifest.csv
training/data/external/oep_multiview/notes/oep_webcam_segments.csv
training/data/external/oep_multiview/notes/oep_summary.json
```

These files are for review and relabeling. They are not a drop-in replacement for `behavior_segments.csv`.

## 7. Build the temporal dataset

```bash
cd /Users/huy.dao/XuLyAnh/anti-cheat-demo
python3 training/scripts/build_temporal_dataset.py \
  --sessions-root training/data/raw_sessions \
  --labels training/data/annotations/behavior_segments.csv \
  --output training/data/processed/temporal_sequences.jsonl
```

## 8. Train the baseline LSTM

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
