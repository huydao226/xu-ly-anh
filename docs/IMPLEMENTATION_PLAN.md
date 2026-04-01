# One-Camera Anti-Cheat Demo Plan

## Goal

Build a local demo that mimics the main AutoOEP-style ideas with a single laptop camera:

- face presence monitoring
- multi-face detection
- head turn / gaze-away heuristics
- visible phone detection
- session risk scoring
- live monitoring UI with logs

## Scope

This demo is optimized for clarity and local execution, not production accuracy.

## Components

- `backend/`
  - FastAPI API
  - OpenCV face and eye cascade analysis
  - YOLO object detection for visible prohibited items
  - in-memory session/event store
- `frontend/`
  - React + TypeScript dashboard
  - webcam capture loop
  - live metrics, event log, annotated preview

## Detection Strategy

1. Browser captures frames from the laptop webcam.
2. Frontend sends sampled frames to the backend.
3. Backend runs:
   - face detection / landmarks
   - head pose and gaze heuristics
   - multiple-face checks
   - YOLO object detection for phone and related visible objects
4. Backend returns:
   - severity
   - current events
   - metrics
   - annotated frame
   - running session summary
5. Frontend renders the live demo dashboard.

## Expected Demo Behaviors

- Warn when no face is visible.
- Warn or escalate when more than one face is visible.
- Warn when the user looks away for a sustained period.
- Warn when a phone is visible in frame.
- Maintain a running log and simple risk score.

## Limits

- A single laptop camera cannot reliably detect hidden notes or a phone outside the frame.
- Gaze estimation is heuristic-based and sensitive to lighting and camera angle.
- The demo prioritizes understandable code over fully trained multi-modal research quality.
