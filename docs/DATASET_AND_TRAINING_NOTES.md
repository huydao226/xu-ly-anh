# Dataset, Training, And Cheat-Catching Notes

## 1. Dataset currently used in the implemented demo

Current code does **not** use a custom training dataset yet.

The current implementation relies on:

- `YOLOv8n` pretrained weights (`yolov8n.pt`)
  - used for visible object detection in frame
  - current classes effectively used by the code:
    - `cell phone`
    - `book`
    - `person`
    - `laptop`
- `OpenCV Haar Cascade` XML models bundled with OpenCV
  - used for:
    - face detection
    - eye detection

## 2. What is actually learned vs heuristic

### Learned / pretrained

- YOLO object detector is pretrained already.
- Haar cascade detectors are pretrained models distributed with OpenCV.

### Not learned in the current demo

- Risk scoring is rule-based.
- Eye focus and head-turn decisions are heuristic-based.
- No custom fine-tuning has been applied yet.

## 3. How the current demo catches cheat

The backend currently catches suspicious behavior using the following signals:

### Face-based signals

- `face_missing`
  - no face detected in the frame
- `multiple_faces`
  - more than one face appears in the frame
- `eyes_off_screen`
  - estimated eye center shifts too far away from the face center
- `gaze_sweep_detected`
  - weaker version of looking away
- `head_yaw_detected`
  - eye spread and offset suggest head yaw
- `head_turn_detected`
  - eye-line slope indicates a stronger head turn
- `looking_down`
  - eye region stays too low inside the face box

### Object-based signals

- `phone_visible`
  - YOLO sees a visible phone
- `book_visible`
  - YOLO sees a visible book or note-like object

## 4. How to train this properly later

If we want a stronger research version, the training plan should be:

### Stage 1: collect a custom laptop-camera dataset

Record short clips with:

- normal exam behavior
- looking left
- looking right
- looking down
- talking to another person
- second person entering frame
- phone visible
- notes visible
- temporary occlusion
- low light / glasses / different camera angles

### Stage 2: annotate the dataset

Two annotation levels are recommended:

- **frame-level labels** for object detection
  - phone
  - book / notes
  - face count
- **clip-level labels** for behavior
  - normal
  - gaze away
  - head turn
  - looking down
  - multiple faces
  - device usage

### Stage 3: train the object detector

Fine-tune YOLO on your own webcam data for:

- phone
- paper / notes
- calculator
- second person

This improves much faster than relying only on generic COCO classes.

### Stage 4: train the temporal cheat classifier

Use per-frame features such as:

- face count
- phone confidence
- yaw ratio
- pitch ratio
- eye offset
- event counts over time

Then train a sequence model:

- `LSTM`
- or `Temporal CNN`
- or a small `Transformer`

Output:

- `normal`
- `suspicious`
- `high risk`
- or specific cheat classes

## 5. What is implemented now for training support

The repository now includes a concrete training scaffold:

- backend dataset capture mode
  - saves webcam frames and per-frame metadata to `training/data/raw_sessions`
- YOLO dataset preparation script
  - converts labeled images into the Ultralytics folder layout
- temporal dataset builder
  - converts captured session metadata plus segment labels into sequence samples
- baseline LSTM trainer
  - trains a behavior classifier on the generated temporal dataset

## 6. Recommended practical approach for our topic

For our current topic, the fastest reasonable roadmap is:

1. keep the current demo as a baseline
2. collect custom webcam data from our own exam scenario
3. fine-tune YOLO for phone / notes / multi-person
4. keep face heuristics first
5. later add temporal training for eye focus and head behavior

## 7. Short note for the report

At the current stage:

- the project is a **demo prototype**
- object detection uses **pretrained YOLO**
- face and eye logic uses **OpenCV pretrained cascades**
- the final cheat decision is still **heuristic**, not yet a fully trained end-to-end cheating classifier
- however, the repo now includes a **real fine-tune path** for both object detection and temporal behavior modeling
