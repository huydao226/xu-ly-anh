from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Severity = Literal['normal', 'warning', 'critical']


class SessionCreateRequest(BaseModel):
  operator_name: str | None = None


class FrameAnalyzeRequest(BaseModel):
  frame: str = Field(..., description='Base64 encoded data URL or raw base64 image string')
  captured_at: str | None = None


class SessionStopRequest(BaseModel):
  reason: str | None = None


class DetectionEvent(BaseModel):
  code: str
  label: str
  severity: Severity
  score: float = 0.0
  details: dict[str, Any] = Field(default_factory=dict)


class DetectionMetrics(BaseModel):
  faces_detected: int = 0
  phone_detected: bool = False
  book_detected: bool = False
  yaw_ratio: float | None = None
  pitch_ratio: float | None = None
  eye_line_angle: float | None = None
  face_box: dict[str, int] | None = None
  detector_notes: list[str] = Field(default_factory=list)


class SessionSummary(BaseModel):
  session_id: str
  status: str
  frame_count: int
  risk_score: float
  warning_count: int
  critical_count: int
  last_event_at: str | None
  started_at: str
  stopped_at: str | None
  current_severity: Severity


class FrameAnalyzeResponse(BaseModel):
  severity: Severity
  risk_score: float
  session: SessionSummary
  metrics: DetectionMetrics
  events: list[DetectionEvent]
  annotated_frame: str | None = None


class SessionReadResponse(BaseModel):
  session: SessionSummary
  recent_events: list[DetectionEvent]


class ConfigResponse(BaseModel):
  app_name: str
  frame_cooldown_ms: int
  detection_features: list[str]
  dataset_capture_enabled: bool = False
  dataset_capture_root: str | None = None
