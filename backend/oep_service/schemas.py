from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SessionCreateRequest(BaseModel):
  operator_name: str | None = None


class FrameAnalyzeRequest(BaseModel):
  frame: str = Field(..., description='Base64 encoded data URL or raw base64 image string')
  captured_at: str | None = None


class SessionStopRequest(BaseModel):
  reason: str | None = None


class PredictionScore(BaseModel):
  label: str
  confidence: float


class OepSessionSummary(BaseModel):
  session_id: str
  status: str
  frame_count: int
  buffer_size: int
  started_at: str
  stopped_at: str | None
  last_prediction: str | None
  last_confidence: float | None


class OepConfigResponse(BaseModel):
  model_name: str
  sequence_frames: int
  required_frames: int
  frame_width: int
  labels: list[str]


class OepFrameResponse(BaseModel):
  session: OepSessionSummary
  ready: bool
  prediction_label: str | None = None
  confidence: float | None = None
  probabilities: list[PredictionScore] = Field(default_factory=list)
  annotated_frame: str | None = None
  features: list[float] = Field(default_factory=list)
  face_box: dict[str, int] | None = None
  status_text: str


class OepSessionReadResponse(BaseModel):
  session: OepSessionSummary
  probabilities: list[PredictionScore] = Field(default_factory=list)
