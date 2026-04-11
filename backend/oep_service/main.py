from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from oep_service.schemas import (
  FrameAnalyzeRequest,
  OepConfigResponse,
  OepFrameResponse,
  OepSessionReadResponse,
  PredictionScore,
  SessionCreateRequest,
  SessionStopRequest,
)
from oep_service.session_store import OepSession, oep_session_store
from oep_service.temporal import (
  DEVICE_HOLD_FRAMES,
  FRAME_WIDTH,
  MODEL_DIR,
  MIN_FRAMES_TO_PREDICT,
  SEQUENCE_FRAMES,
  absence_override,
  annotate_frame,
  decode_image,
  detect_device,
  extract_frame_features,
  frontal_normal_override,
  load_model_bundle,
  override_probabilities,
  predict_sequence,
)


bundle = load_model_bundle()

app = FastAPI(title='OEP Temporal Monitor Service', version='0.1.0')
app.add_middleware(
  CORSMiddleware,
  allow_origins=[
    'http://localhost:5173',
    'http://127.0.0.1:5173',
    'http://localhost:5174',
    'http://127.0.0.1:5174',
  ],
  allow_origin_regex=r'https?://(localhost|127\.0\.0\.1):\d+',
  allow_credentials=True,
  allow_methods=['*'],
  allow_headers=['*'],
)


def _prediction_scores(session: OepSession) -> list[PredictionScore]:
  return [PredictionScore(**item) for item in session.last_probabilities]


@app.get('/health')
def health() -> dict[str, str]:
  return {'status': 'ok'}


@app.get('/api/config', response_model=OepConfigResponse)
def get_config() -> OepConfigResponse:
  return OepConfigResponse(
    model_name=MODEL_DIR.name,
    sequence_frames=SEQUENCE_FRAMES,
    required_frames=MIN_FRAMES_TO_PREDICT,
    frame_width=FRAME_WIDTH,
    labels=bundle.labels + ['absence/offscreen'],
  )


@app.post('/api/session/start', response_model=OepSessionReadResponse)
def start_session(payload: SessionCreateRequest) -> OepSessionReadResponse:
  session = oep_session_store.create(payload.operator_name)
  return OepSessionReadResponse(session=session.to_summary(), probabilities=[])


@app.get('/api/session/{session_id}', response_model=OepSessionReadResponse)
def get_session(session_id: str) -> OepSessionReadResponse:
  try:
    session = oep_session_store.get(session_id)
  except KeyError as exc:
    raise HTTPException(status_code=404, detail=str(exc)) from exc
  return OepSessionReadResponse(session=session.to_summary(), probabilities=_prediction_scores(session))


@app.post('/api/session/{session_id}/frame', response_model=OepFrameResponse)
def analyze_session_frame(session_id: str, payload: FrameAnalyzeRequest) -> OepFrameResponse:
  try:
    session = oep_session_store.get(session_id)
  except KeyError as exc:
    raise HTTPException(status_code=404, detail=str(exc)) from exc

  image = decode_image(payload.frame)
  features = extract_frame_features(image, session.previous_gray)
  device_detection = detect_device(image)
  session.previous_gray = features.gray
  session.frame_count += 1
  session.feature_buffer.append(features.vector)
  session.offscreen_streak = session.offscreen_streak + 1 if (not features.face_present and not features.pose_present) else 0
  if device_detection.detected:
    session.device_hold_remaining = DEVICE_HOLD_FRAMES
    session.last_device_confidence = round(device_detection.confidence, 4)
  elif session.device_hold_remaining > 0:
    session.device_hold_remaining -= 1
  else:
    session.last_device_confidence = 0.0
  device_active = device_detection.detected or session.device_hold_remaining > 0

  ready = len(session.feature_buffer) >= MIN_FRAMES_TO_PREDICT
  prediction_label = None
  confidence = None
  status_text = f'Collecting temporal sequence: {len(session.feature_buffer)}/{SEQUENCE_FRAMES} frames'
  if ready:
    prediction_label, confidence, probabilities = predict_sequence(bundle, list(session.feature_buffer))
    if device_active:
      prediction_label = 'device'
      confidence = max(round(device_detection.confidence, 4), session.last_device_confidence)
      probabilities = override_probabilities(
        probabilities,
        override_label='device',
        override_confidence=confidence,
      )
      status_text = f'Device rule triggered ({confidence:.2%})'
    override_label, override_confidence, adjusted_probabilities = absence_override(
      bundle=features,
      offscreen_streak=session.offscreen_streak,
      probabilities=probabilities,
    )
    if override_label is not None and override_confidence is not None and not device_active:
      prediction_label = override_label
      confidence = override_confidence
      probabilities = adjusted_probabilities
      status_text = f'Absence/offscreen rule triggered after {session.offscreen_streak} frames without a visible face.'
    elif not device_active:
      frontal_label, frontal_confidence, frontal_probabilities, frontal_status = frontal_normal_override(
        prediction_label=prediction_label,
        probabilities=probabilities,
        sequence=list(session.feature_buffer),
        bundle=features,
      )
      if frontal_label is not None and frontal_confidence is not None:
        prediction_label = frontal_label
        confidence = frontal_confidence
        probabilities = frontal_probabilities
        status_text = frontal_status or f'OEP monitor v3 predicts {prediction_label} ({confidence:.2%})'
      else:
        status_text = f'OEP monitor v3 predicts {prediction_label} ({confidence:.2%})'
    session.last_prediction = prediction_label
    session.last_confidence = round(confidence, 4)
    session.last_probabilities = probabilities
  else:
    session.last_probabilities = []

  annotated_frame = annotate_frame(
    image,
    face_box=features.face_box,
    device_box=device_detection.bbox,
    ready=ready,
    prediction_label=session.last_prediction,
    confidence=session.last_confidence,
    buffer_size=len(session.feature_buffer),
  )

  return OepFrameResponse(
    session=session.to_summary(),
    ready=ready,
    prediction_label=session.last_prediction,
    confidence=session.last_confidence,
    probabilities=_prediction_scores(session),
    annotated_frame=annotated_frame,
    features=[round(value, 4) for value in features.vector],
    face_box=features.face_box,
    status_text=status_text,
  )


@app.post('/api/session/{session_id}/stop', response_model=OepSessionReadResponse)
def stop_session(session_id: str, payload: SessionStopRequest) -> OepSessionReadResponse:
  del payload
  try:
    session = oep_session_store.stop(session_id)
  except KeyError as exc:
    raise HTTPException(status_code=404, detail=str(exc)) from exc
  return OepSessionReadResponse(session=session.to_summary(), probabilities=_prediction_scores(session))
