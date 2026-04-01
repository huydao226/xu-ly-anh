# One-Camera Anti-Cheat Demo

This project is a fresh replacement for the old repository. It implements a local demo inspired by AutoOEP, adapted for a single laptop camera.

## What is included

- `backend/`
  - FastAPI API
  - OpenCV face and eye detection heuristics
  - OpenCV overlay generation
  - YOLO object detection for visible phone and book signals
  - in-memory session store for logs and risk scoring
  - optional dataset capture mode for collecting fine-tune data
- `frontend/`
  - React + TypeScript dashboard
  - webcam capture from the browser
  - live annotated preview from the backend
  - monitoring cards, session summary, and event log
- `training/`
  - scripts to prepare YOLO datasets from labeled frames
  - scripts to build temporal sequence datasets from captured sessions
  - baseline LSTM training script for behavior classification
- `docs/`
  - implementation notes
  - step-by-step guide for running and demoing the app
  - Vietnamese fine-tune and report drafts

## Main demo features

- detect when no face is visible
- warn when multiple faces are visible
- estimate head turn and looking-down behavior from face landmarks
- detect a visible phone and visible book-like object when YOLO sees them
- keep a running session risk score and log suspicious events

## Important limitations

- One laptop camera cannot detect cheating outside the frame.
- Gaze and head pose are heuristic approximations, not a research-grade multi-camera setup.
- The first YOLO run may download weights if they are not already cached.

## Quick start

See [STEP_BY_STEP.md](/Users/huy.dao/XuLyAnh/anti-cheat-demo/docs/STEP_BY_STEP.md).

## Fine-tune path

If you need to satisfy the "AI training" requirement for a report or thesis, the project now includes a practical path:

1. enable dataset capture in the backend
2. record webcam sessions from Asian/Vietnamese participants
3. annotate object boxes for phone / notes / second person
4. annotate behavior segments for look away / looking down / phone use
5. build a YOLO dataset and fine-tune the detector
6. build a temporal sequence dataset and train the baseline LSTM

See [FINE_TUNE_GUIDE_VI.md](/Users/huy.dao/XuLyAnh/anti-cheat-demo/docs/FINE_TUNE_GUIDE_VI.md) and [training/README.md](/Users/huy.dao/XuLyAnh/anti-cheat-demo/training/README.md).
