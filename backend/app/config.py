from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _as_bool(name: str, default: bool = False) -> bool:
  raw = os.getenv(name)
  if raw is None:
    return default
  return raw.strip().lower() in {'1', 'true', 'yes', 'on'}


@dataclass(frozen=True)
class Settings:
  project_root: Path = Path(__file__).resolve().parents[2]
  app_name: str = os.getenv('APP_NAME', 'One Camera Anti-Cheat Demo')
  allowed_origins: tuple[str, ...] = tuple(
    origin.strip()
    for origin in os.getenv(
      'ALLOWED_ORIGINS',
      'http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174,http://localhost:5175,http://127.0.0.1:5175',
    ).split(',')
    if origin.strip()
  )
  allow_origin_regex: str = os.getenv('ALLOW_ORIGIN_REGEX', r'https?://(localhost|127\.0\.0\.1):\d+')
  yolo_model_name: str = os.getenv('YOLO_MODEL_NAME', 'yolov8n.pt')
  frame_cooldown_ms: int = int(os.getenv('FRAME_COOLDOWN_MS', '600'))
  max_log_entries: int = int(os.getenv('MAX_LOG_ENTRIES', '100'))
  max_sessions: int = int(os.getenv('MAX_SESSIONS', '10'))
  dataset_capture_enabled: bool = _as_bool('DATASET_CAPTURE_ENABLED', False)
  dataset_capture_root: Path = Path(
    os.getenv('DATASET_CAPTURE_ROOT', str(Path(__file__).resolve().parents[2] / 'training' / 'data' / 'raw_sessions'))
  ).expanduser()


settings = Settings()
