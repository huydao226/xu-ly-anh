from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import numpy as np


def utc_now_iso() -> str:
  return datetime.now(timezone.utc).isoformat()


@dataclass
class OepSession:
  session_id: str
  operator_name: str
  started_at: str
  status: str = 'active'
  stopped_at: str | None = None
  frame_count: int = 0
  previous_gray: np.ndarray | None = None
  feature_buffer: deque[list[float]] = field(default_factory=lambda: deque(maxlen=16))
  offscreen_streak: int = 0
  device_hold_remaining: int = 0
  last_device_confidence: float = 0.0
  last_prediction: str | None = None
  last_confidence: float | None = None
  last_probabilities: list[dict[str, Any]] = field(default_factory=list)

  def to_summary(self) -> dict[str, Any]:
    return {
      'session_id': self.session_id,
      'status': self.status,
      'frame_count': self.frame_count,
      'buffer_size': len(self.feature_buffer),
      'started_at': self.started_at,
      'stopped_at': self.stopped_at,
      'last_prediction': self.last_prediction,
      'last_confidence': self.last_confidence,
    }


class OepSessionStore:
  def __init__(self, *, max_sessions: int = 6, sequence_frames: int = 16) -> None:
    self.max_sessions = max_sessions
    self.sequence_frames = sequence_frames
    self._sessions: dict[str, OepSession] = {}

  def create(self, operator_name: str | None = None) -> OepSession:
    if len(self._sessions) >= self.max_sessions:
      oldest_id = min(self._sessions, key=lambda session_id: self._sessions[session_id].started_at)
      del self._sessions[oldest_id]
    session = OepSession(
      session_id=str(uuid4()),
      operator_name=(operator_name or 'OEP Monitor').strip() or 'OEP Monitor',
      started_at=utc_now_iso(),
      feature_buffer=deque(maxlen=self.sequence_frames),
    )
    self._sessions[session.session_id] = session
    return session

  def get(self, session_id: str) -> OepSession:
    try:
      return self._sessions[session_id]
    except KeyError as exc:
      raise KeyError(f'Unknown session: {session_id}') from exc

  def stop(self, session_id: str) -> OepSession:
    session = self.get(session_id)
    session.status = 'stopped'
    session.stopped_at = utc_now_iso()
    session.previous_gray = None
    session.offscreen_streak = 0
    session.device_hold_remaining = 0
    session.last_device_confidence = 0.0
    return session


oep_session_store = OepSessionStore()
