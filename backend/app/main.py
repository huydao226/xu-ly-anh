from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.dataset_capture import dataset_capture_recorder
from app.schemas import (
  ConfigResponse,
  FrameAnalyzeRequest,
  FrameAnalyzeResponse,
  SessionCreateRequest,
  SessionReadResponse,
  SessionStopRequest,
)
from app.session_store import session_store
from app.vision import analyze_frame


app = FastAPI(title=settings.app_name, version='0.1.0')
app.add_middleware(
  CORSMiddleware,
  allow_origins=list(settings.allowed_origins),
  allow_origin_regex=settings.allow_origin_regex,
  allow_credentials=True,
  allow_methods=['*'],
  allow_headers=['*'],
)


@app.get('/health')
def health() -> dict[str, str]:
  return {'status': 'ok'}


@app.get('/api/config', response_model=ConfigResponse)
def get_config() -> ConfigResponse:
  return ConfigResponse(
    app_name=settings.app_name,
    frame_cooldown_ms=settings.frame_cooldown_ms,
    detection_features=[
      'face presence',
      'multiple faces',
      'head turn heuristic',
      'looking down heuristic',
      'phone detection',
      'book detection if visible',
    ],
    dataset_capture_enabled=settings.dataset_capture_enabled,
    dataset_capture_root=str(settings.dataset_capture_root) if settings.dataset_capture_enabled else None,
  )


@app.post('/api/session/start', response_model=SessionReadResponse)
def start_session(payload: SessionCreateRequest) -> SessionReadResponse:
  session = session_store.create(payload.operator_name)
  return SessionReadResponse(session=session.to_summary(), recent_events=[])


@app.get('/api/session/{session_id}', response_model=SessionReadResponse)
def get_session(session_id: str) -> SessionReadResponse:
  try:
    session = session_store.get(session_id)
  except KeyError as exc:
    raise HTTPException(status_code=404, detail=str(exc)) from exc
  return SessionReadResponse(session=session.to_summary(), recent_events=session_store.recent_events(session_id))


@app.post('/api/session/{session_id}/frame', response_model=FrameAnalyzeResponse)
def analyze_session_frame(session_id: str, payload: FrameAnalyzeRequest) -> FrameAnalyzeResponse:
  try:
    session = session_store.get(session_id)
  except KeyError as exc:
    raise HTTPException(status_code=404, detail=str(exc)) from exc

  analysis = analyze_frame(payload.frame)
  session = session_store.update(session_id, analysis.severity, analysis.events)
  dataset_capture_recorder.record_frame(
    session_id=session_id,
    operator_name=session.operator_name,
    frame_index=session.frame_count,
    frame_base64=payload.frame,
    captured_at=payload.captured_at,
    severity=analysis.severity,
    metrics=analysis.metrics,
    events=analysis.events,
  )
  return FrameAnalyzeResponse(
    severity=analysis.severity,  # type: ignore[arg-type]
    risk_score=round(session.risk_score, 2),
    session=session.to_summary(),
    metrics=analysis.metrics,
    events=analysis.events,
    annotated_frame=analysis.annotated_frame,
  )


@app.post('/api/session/{session_id}/stop', response_model=SessionReadResponse)
def stop_session(session_id: str, payload: SessionStopRequest) -> SessionReadResponse:
  del payload
  try:
    session = session_store.stop(session_id)
  except KeyError as exc:
    raise HTTPException(status_code=404, detail=str(exc)) from exc
  return SessionReadResponse(session=session.to_summary(), recent_events=session_store.recent_events(session_id))
