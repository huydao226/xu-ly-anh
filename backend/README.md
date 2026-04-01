# Backend

FastAPI backend for the one-camera anti-cheat demo.

## What it does

- accepts webcam frames from the frontend
- runs OpenCV face and eye analysis
- runs YOLO object detection for visible phone/book signals
- scores suspicious events in memory for a demo session
- returns an annotated frame plus metrics and event logs
- can optionally save each analyzed frame plus metadata to `training/data/raw_sessions`

## Main endpoints

- `GET /health`
- `GET /api/config`
- `POST /api/session/start`
- `GET /api/session/{session_id}`
- `POST /api/session/{session_id}/frame`
- `POST /api/session/{session_id}/stop`

## Notes

- This is a demo-oriented backend, not a production anti-cheat engine.
- With one laptop camera, the system only detects items visible in frame.
- YOLO may download weights on first run when `yolov8n.pt` is not already cached.
- To capture fine-tune data from real demo sessions, set `DATASET_CAPTURE_ENABLED=true` in `backend/.env`.
