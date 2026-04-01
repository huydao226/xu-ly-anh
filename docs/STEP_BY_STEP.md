# Step By Step

## 1. Open the project

Project root:

```bash
cd /Users/huy.dao/XuLyAnh/anti-cheat-demo
```

## 2. Start the backend

Create a virtual environment and install dependencies:

```bash
cd /Users/huy.dao/XuLyAnh/anti-cheat-demo/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Notes:

- Backend health check: `http://127.0.0.1:8000/health`
- The first YOLO startup may take longer because it can download `yolov8n.pt`.

## 3. Start the frontend

Open a second terminal:

```bash
cd /Users/huy.dao/XuLyAnh/anti-cheat-demo/frontend
cp .env.example .env
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

## 4. Run the demo

1. Click `Open camera`
2. Click `Start monitoring`
3. Keep your face centered for a normal baseline
4. Try the following demo cases:

- move out of frame to trigger `No face detected`
- invite a second person into frame to trigger `Multiple faces visible`
- turn your head left or right for a stronger `Head turned away from screen`
- look down for a few moments to trigger `Looking down`
- show a phone to the camera to trigger `Phone visible in frame`

## 5. What you will see

- left panel: raw laptop webcam view
- right panel: annotated backend overlay
- top summary: backend state and current severity
- metrics cards: faces, risk score, yaw ratio, pitch ratio
- event log: suspicious signals detected during the session

## 6. API contract

Backend routes used by the frontend:

- `GET /health`
- `GET /api/config`
- `POST /api/session/start`
- `GET /api/session/{session_id}`
- `POST /api/session/{session_id}/frame`
- `POST /api/session/{session_id}/stop`

## 7. Recommended reading order

If you want to understand the code in the shortest path:

1. [README.md](/Users/huy.dao/XuLyAnh/anti-cheat-demo/README.md)
2. [backend/app/main.py](/Users/huy.dao/XuLyAnh/anti-cheat-demo/backend/app/main.py)
3. [backend/app/vision.py](/Users/huy.dao/XuLyAnh/anti-cheat-demo/backend/app/vision.py)
4. [frontend/src/App.tsx](/Users/huy.dao/XuLyAnh/anti-cheat-demo/frontend/src/App.tsx)
5. [frontend/src/App.css](/Users/huy.dao/XuLyAnh/anti-cheat-demo/frontend/src/App.css)

## 8. If the backend cannot detect the phone

This can happen because the project uses a general pretrained YOLO model, not a custom fine-tuned exam dataset.

To improve it later:

- fine-tune YOLO on your own webcam cheating dataset
- add a side camera view
- add temporal scoring with LSTM or another sequence model
- add browser telemetry for tab switching and focus loss

## 9. Collect data for fine-tuning

1. Open `/Users/huy.dao/XuLyAnh/anti-cheat-demo/backend/.env`
2. Set:

```bash
DATASET_CAPTURE_ENABLED=true
DATASET_CAPTURE_ROOT=/Users/huy.dao/XuLyAnh/anti-cheat-demo/training/data/raw_sessions
```

3. Restart the backend
4. Run demo sessions for each scenario:

- normal exam behavior
- blink and short natural movement
- look left / look right
- look down
- phone visible
- notes visible
- second person visible

5. After each session, check:

```text
/Users/huy.dao/XuLyAnh/anti-cheat-demo/training/data/raw_sessions/<session_id>/
```

Each saved session contains:

- `images/`
- `metadata.jsonl`
- `session_manifest.json`

6. Continue with:

- [FINE_TUNE_GUIDE_VI.md](/Users/huy.dao/XuLyAnh/anti-cheat-demo/docs/FINE_TUNE_GUIDE_VI.md)
- [training/README.md](/Users/huy.dao/XuLyAnh/anti-cheat-demo/training/README.md)
