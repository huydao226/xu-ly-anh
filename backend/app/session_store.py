from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque
from uuid import uuid4

from app.config import settings
from app.schemas import DetectionEvent, SessionSummary


def utc_now_iso() -> str:
  return datetime.now(tz=timezone.utc).isoformat()


@dataclass
class SessionState:
  id: str
  operator_name: str | None
  started_at: str
  status: str = 'active'
  stopped_at: str | None = None
  frame_count: int = 0
  risk_score: float = 0.0
  warning_count: int = 0
  critical_count: int = 0
  last_event_at: str | None = None
  current_severity: str = 'normal'
  recent_events: Deque[DetectionEvent] = field(default_factory=lambda: deque(maxlen=settings.max_log_entries))

  def to_summary(self) -> SessionSummary:
    return SessionSummary(
      session_id=self.id,
      status=self.status,
      frame_count=self.frame_count,
      risk_score=round(self.risk_score, 2),
      warning_count=self.warning_count,
      critical_count=self.critical_count,
      last_event_at=self.last_event_at,
      started_at=self.started_at,
      stopped_at=self.stopped_at,
      current_severity=self.current_severity,  # type: ignore[arg-type]
    )


class SessionStore:
  def __init__(self) -> None:
    self._sessions: dict[str, SessionState] = {}

  def create(self, operator_name: str | None) -> SessionState:
    if len(self._sessions) >= settings.max_sessions:
      self._purge_oldest_stopped_session()
    session = SessionState(id=str(uuid4()), operator_name=operator_name, started_at=utc_now_iso())
    self._sessions[session.id] = session
    return session

  def get(self, session_id: str) -> SessionState:
    try:
      return self._sessions[session_id]
    except KeyError as exc:
      raise KeyError(f'Session {session_id} not found') from exc

  def stop(self, session_id: str) -> SessionState:
    session = self.get(session_id)
    session.status = 'stopped'
    session.stopped_at = utc_now_iso()
    return session

  def update(self, session_id: str, severity: str, events: list[DetectionEvent]) -> SessionState:
    session = self.get(session_id)
    session.frame_count += 1
    session.current_severity = severity
    if events:
      session.last_event_at = utc_now_iso()
    for event in events:
      session.recent_events.appendleft(event)
      if event.severity == 'warning':
        session.warning_count += 1
        session.risk_score = min(100.0, session.risk_score + 6.0 + event.score * 8.0)
      elif event.severity == 'critical':
        session.critical_count += 1
        session.risk_score = min(100.0, session.risk_score + 14.0 + event.score * 12.0)
      else:
        session.risk_score = max(0.0, session.risk_score - 1.0)
    if not events:
      session.risk_score = max(0.0, session.risk_score - 0.75)
    return session

  def recent_events(self, session_id: str) -> list[DetectionEvent]:
    session = self.get(session_id)
    return list(session.recent_events)

  def _purge_oldest_stopped_session(self) -> None:
    stopped_ids = [
      session.id
      for session in sorted(self._sessions.values(), key=lambda item: item.started_at)
      if session.status == 'stopped'
    ]
    if stopped_ids:
      self._sessions.pop(stopped_ids[0], None)


session_store = SessionStore()
