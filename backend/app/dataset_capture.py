from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import settings
from app.schemas import DetectionEvent, DetectionMetrics


def _decode_image(frame_base64: str) -> np.ndarray:
  encoded = frame_base64.split(',', maxsplit=1)[1] if frame_base64.startswith('data:image') else frame_base64
  binary = base64.b64decode(encoded)
  buffer = np.frombuffer(binary, dtype=np.uint8)
  image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
  if image is None:
    raise ValueError('Unable to decode frame payload')
  return image


class DatasetCaptureRecorder:
  def __init__(self, root_dir: Path) -> None:
    self.root_dir = root_dir

  def record_frame(
    self,
    *,
    session_id: str,
    operator_name: str | None,
    frame_index: int,
    frame_base64: str,
    captured_at: str | None,
    severity: str,
    metrics: DetectionMetrics,
    events: list[DetectionEvent],
  ) -> str | None:
    if not settings.dataset_capture_enabled:
      return None

    image = _decode_image(frame_base64)
    session_dir = self.root_dir / session_id
    images_dir = session_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    frame_name = f'frame_{frame_index:06d}.jpg'
    frame_path = images_dir / frame_name
    saved = cv2.imwrite(str(frame_path), image)
    if not saved:
      raise ValueError(f'Unable to write captured frame to {frame_path}')

    manifest_path = session_dir / 'session_manifest.json'
    if not manifest_path.exists():
      manifest_path.write_text(
        json.dumps(
          {
            'session_id': session_id,
            'operator_name': operator_name,
            'capture_source': 'browser_webcam',
            'frame_format': 'jpg',
          },
          ensure_ascii=False,
          indent=2,
        ),
        encoding='utf-8',
      )

    metadata_path = session_dir / 'metadata.jsonl'
    record = {
      'session_id': session_id,
      'frame_index': frame_index,
      'captured_at': captured_at,
      'image_path': str(frame_path),
      'severity': severity,
      'metrics': metrics.model_dump(),
      'events': [event.model_dump() for event in events],
    }
    with metadata_path.open('a', encoding='utf-8') as handle:
      handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    return str(frame_path)

  def summarize(self) -> dict[str, Any]:
    return {
      'enabled': settings.dataset_capture_enabled,
      'root_dir': str(self.root_dir),
    }


dataset_capture_recorder = DatasetCaptureRecorder(settings.dataset_capture_root)
