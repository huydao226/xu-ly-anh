from __future__ import annotations

import base64
import math
import threading
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from app.config import settings
from app.schemas import DetectionEvent, DetectionMetrics


_YOLO_MODEL: YOLO | None = None
_YOLO_ERROR: str | None = None
_VISION_LOCK = threading.Lock()

FACE_CASCADE_FILES = (
  'haarcascade_frontalface_default.xml',
  'haarcascade_frontalface_alt2.xml',
  'haarcascade_frontalface_alt_tree.xml',
)

EYE_CASCADE_FILES = (
  'haarcascade_eye_tree_eyeglasses.xml',
  'haarcascade_eye.xml',
)

FACE_DETECTION_PARAM_SETS = (
  {'scaleFactor': 1.06, 'minNeighbors': 6, 'minSize': (100, 100)},
  {'scaleFactor': 1.08, 'minNeighbors': 5, 'minSize': (80, 80)},
  {'scaleFactor': 1.03, 'minNeighbors': 4, 'minSize': (60, 60)},
)

EYE_DETECTION_PARAM_SETS = (
  {'scaleFactor': 1.09, 'minNeighbors': 6, 'minSize': (24, 24)},
  {'scaleFactor': 1.05, 'minNeighbors': 5, 'minSize': (18, 18)},
)

CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

MIN_HEAD_TURN_DEGREES = 25.0
HEAD_TURN_RATIO_THRESHOLD = math.tan(math.radians(MIN_HEAD_TURN_DEGREES))
EYE_OFFSET_THRESHOLD = 0.35
EYE_SWEEP_OFFSET_THRESHOLD = 0.25
HEAD_YAW_SPREAD_THRESHOLD = 0.28
HEAD_YAW_OFFSET_THRESHOLD = 0.08

COLOR_NORMAL = (0, 200, 255)
COLOR_WARNING = (0, 180, 255)
COLOR_CRITICAL = (30, 30, 220)
COLOR_EYE = (0, 255, 255)


def _load_cascade(filename: str) -> cv2.CascadeClassifier | None:
  cascade = cv2.CascadeClassifier(cv2.data.haarcascades + filename)
  if cascade.empty():
    return None
  return cascade


FACE_CASCADES = [cascade for cascade in (_load_cascade(name) for name in FACE_CASCADE_FILES) if cascade is not None]
EYE_CASCADES = [cascade for cascade in (_load_cascade(name) for name in EYE_CASCADE_FILES) if cascade is not None]


@dataclass
class AnalysisResult:
  severity: str
  metrics: DetectionMetrics
  events: list[DetectionEvent]
  annotated_frame: str | None


def _decode_image(frame_base64: str) -> np.ndarray:
  encoded = frame_base64.split(',', maxsplit=1)[1] if frame_base64.startswith('data:image') else frame_base64
  binary = base64.b64decode(encoded)
  buffer = np.frombuffer(binary, dtype=np.uint8)
  image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
  if image is None:
    raise ValueError('Unable to decode frame payload')
  return image


def _encode_image(image: np.ndarray) -> str:
  success, buffer = cv2.imencode('.jpg', image)
  if not success:
    raise ValueError('Unable to encode annotated frame')
  return 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')


def _prepare_grayscale(image: np.ndarray) -> np.ndarray:
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, (3, 3), 0)
  return CLAHE.apply(blurred)


def _detect_regions(
  gray: np.ndarray,
  cascades: list[cv2.CascadeClassifier],
  param_sets: tuple[dict[str, Any], ...],
) -> list[tuple[int, int, int, int]]:
  detections: list[tuple[int, int, int, int]] = []
  if gray.size == 0:
    return detections
  image_height, image_width = gray.shape[:2]
  if image_height <= 0 or image_width <= 0:
    return detections

  for cascade in cascades:
    if cascade.empty():
      continue
    cascade_hits: list[tuple[int, int, int, int]] = []
    for params in param_sets:
      min_width, min_height = params.get('minSize', (0, 0))
      if image_width < int(min_width) or image_height < int(min_height):
        continue
      try:
        result = cascade.detectMultiScale(gray, **params)
      except cv2.error:
        continue
      if len(result):
        cascade_hits.extend(tuple(map(int, box)) for box in result)
    if cascade_hits:
      detections.extend(cascade_hits)
      break
  return detections


def _compute_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
  ax1, ay1, aw, ah = box_a
  bx1, by1, bw, bh = box_b
  ax2, ay2 = ax1 + aw, ay1 + ah
  bx2, by2 = bx1 + bw, by1 + bh
  inter_x1 = max(ax1, bx1)
  inter_y1 = max(ay1, by1)
  inter_x2 = min(ax2, bx2)
  inter_y2 = min(ay2, by2)
  inter_w = max(0, inter_x2 - inter_x1)
  inter_h = max(0, inter_y2 - inter_y1)
  inter_area = inter_w * inter_h
  if inter_area <= 0:
    return 0.0
  area_a = max(aw * ah, 1)
  area_b = max(bw * bh, 1)
  union = area_a + area_b - inter_area
  return inter_area / union if union > 0 else 0.0


def _deduplicate_faces(faces: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
  faces_sorted = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)
  deduped: list[tuple[int, int, int, int]] = []
  for candidate in faces_sorted:
    if not any(_compute_iou(candidate, kept) > 0.4 for kept in deduped):
      deduped.append(candidate)
  return deduped


def _select_primary_face(faces: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
  return max(faces, key=lambda item: item[2] * item[3]) if faces else None


def _filter_faces_by_area(
  faces: list[tuple[int, int, int, int]],
  primary: tuple[int, int, int, int] | None,
  *,
  min_ratio: float = 0.35,
) -> list[tuple[int, int, int, int]]:
  if not primary:
    return faces
  primary_area = primary[2] * primary[3]
  if primary_area <= 0:
    return [primary]
  return [face for face in faces if face == primary or face[2] * face[3] >= primary_area * min_ratio]


def _extract_face_region(gray_image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray | None:
  x, y, w, h = box
  h_img, w_img = gray_image.shape[:2]
  if x < 0 or y < 0 or x + w > w_img or y + h > h_img:
    return None
  region = gray_image[y : y + h, x : x + w]
  return region if region.size else None


def _detect_faces(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
  return _detect_regions(gray, FACE_CASCADES, FACE_DETECTION_PARAM_SETS) if FACE_CASCADES else []


def _detect_eyes(face_region: np.ndarray) -> list[tuple[int, int, int, int]]:
  if face_region.size == 0 or not EYE_CASCADES:
    return []
  height = face_region.shape[0]
  primary_region = face_region[: max(int(height * 0.65), 1), :]
  eyes = _detect_regions(primary_region, EYE_CASCADES, EYE_DETECTION_PARAM_SETS)
  if eyes:
    return eyes
  return _detect_regions(face_region, EYE_CASCADES, EYE_DETECTION_PARAM_SETS)


def _get_yolo_model() -> YOLO | None:
  global _YOLO_MODEL, _YOLO_ERROR
  if _YOLO_MODEL is not None or _YOLO_ERROR is not None:
    return _YOLO_MODEL
  try:
    _YOLO_MODEL = YOLO(settings.yolo_model_name)
  except Exception as exc:  # pragma: no cover - runtime/model download dependent
    _YOLO_ERROR = str(exc)
    return None
  return _YOLO_MODEL


def _dedupe_events(events: list[DetectionEvent]) -> list[DetectionEvent]:
  deduped: dict[tuple[str, str], DetectionEvent] = {}
  for event in events:
    key = (event.code, event.severity)
    existing = deduped.get(key)
    if existing is None or event.score > existing.score:
      deduped[key] = event
  return list(deduped.values())


def _severity_from_events(events: list[DetectionEvent]) -> str:
  if any(event.severity == 'critical' for event in events):
    return 'critical'
  if any(event.severity == 'warning' for event in events):
    return 'warning'
  return 'normal'


def _draw_box(
  annotated: np.ndarray,
  box: tuple[int, int, int, int],
  *,
  label: str | None,
  color: tuple[int, int, int],
  thickness: int = 2,
) -> None:
  x, y, w, h = box
  cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
  if label:
    cv2.putText(
      annotated,
      label,
      (max(0, x), max(20, y - 8)),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.5,
      color,
      2,
      cv2.LINE_AA,
    )


def _analyze_objects(image: np.ndarray, annotated: np.ndarray) -> tuple[dict[str, bool], list[DetectionEvent], list[str]]:
  model = _get_yolo_model()
  notes: list[str] = []
  if model is None:
    if _YOLO_ERROR:
      notes.append(f'YOLO unavailable: {_YOLO_ERROR}')
    return {'phone_detected': False, 'book_detected': False}, [], notes

  results = model.predict(image, verbose=False, imgsz=640, conf=0.35)
  phone_detected = False
  book_detected = False
  events: list[DetectionEvent] = []

  for result in results:
    names = result.names
    for box in result.boxes:
      cls_idx = int(box.cls.item())
      label = str(names.get(cls_idx, cls_idx)).lower()
      confidence = float(box.conf.item())
      x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
      if any(term in label for term in ('cell phone', 'phone')):
        phone_detected = True
        events.append(
          DetectionEvent(
            code='phone_visible',
            label='Phone visible in frame',
            severity='critical',
            score=min(1.0, confidence),
            details={'confidence': round(confidence, 3), 'bbox': {'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1}},
          )
        )
      elif 'book' in label:
        book_detected = True
        events.append(
          DetectionEvent(
            code='book_visible',
            label='Book or notes-like object visible',
            severity='warning',
            score=min(1.0, confidence),
            details={'confidence': round(confidence, 3), 'bbox': {'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1}},
          )
        )
      if label in {'cell phone', 'book', 'person', 'laptop'}:
        box_w = x2 - x1
        box_h = y2 - y1
        if 'phone' in label:
          color = COLOR_CRITICAL
        elif 'book' in label:
          color = COLOR_CRITICAL
        else:
          color = (0, 220, 160)
        _draw_box(
          annotated,
          (x1, y1, box_w, box_h),
          label=f'{label} {confidence:.2f}',
          color=color,
        )
  return {'phone_detected': phone_detected, 'book_detected': book_detected}, _dedupe_events(events), notes


def analyze_frame(frame_base64: str) -> AnalysisResult:
  with _VISION_LOCK:
    image = _decode_image(frame_base64)
    gray = _prepare_grayscale(image)
    annotated = image.copy()
    raw_faces = _detect_faces(gray)
    merged_faces = _deduplicate_faces(raw_faces)
    primary_face = _select_primary_face(merged_faces)
    filtered_faces = _filter_faces_by_area(merged_faces, primary_face)

    events: list[DetectionEvent] = []
    detector_notes: list[str] = []
    metrics = DetectionMetrics(
      faces_detected=len(filtered_faces),
      face_box=None,
      detector_notes=detector_notes,
    )

    if not filtered_faces:
      events.append(DetectionEvent(code='face_missing', label='No face detected', severity='warning', score=0.8))
    elif len(filtered_faces) > 1:
      severity = 'critical' if len(filtered_faces) > 2 else 'warning'
      events.append(
        DetectionEvent(
          code='multiple_faces',
          label='Multiple faces visible in frame',
          severity=severity,  # type: ignore[arg-type]
          score=min(1.0, len(filtered_faces) / 3.0),
          details={'faces_detected': len(filtered_faces)},
        )
      )

    eye_boxes_global: list[tuple[int, int, int, int]] = []
    if primary_face:
      x, y, w, h = primary_face
      metrics.face_box = {'x': x, 'y': y, 'w': w, 'h': h}

      face_region = _extract_face_region(gray, primary_face)
      if face_region is not None:
        eyes = _detect_eyes(face_region)
        for (ex, ey, ew, eh) in eyes:
          eye_boxes_global.append((x + int(ex), y + int(ey), int(ew), int(eh)))

        if eye_boxes_global:
          eye_centers = [(box[0] + box[2] / 2.0, box[1] + box[3] / 2.0) for box in eye_boxes_global]
          eye_centers.sort(key=lambda center: center[0])
          avg_eye_x = sum(center[0] for center in eye_centers) / len(eye_centers)
          avg_eye_y = sum(center[1] for center in eye_centers) / len(eye_centers)
          eye_offset = (avg_eye_x - (x + w / 2.0)) / max(w / 2.0, 1)
          eye_offset = float(np.clip(eye_offset, -1.0, 1.0))
          metrics.yaw_ratio = round(eye_offset, 3)
          metrics.pitch_ratio = round((avg_eye_y - y) / max(h, 1), 3)

          if abs(eye_offset) > EYE_OFFSET_THRESHOLD:
            events.append(
              DetectionEvent(
                code='eyes_off_screen',
                label='Eyes look away from the screen',
                severity='critical',
                score=min(1.0, abs(eye_offset)),
                details={'offset_ratio': round(eye_offset, 3), 'bbox': {'x': x, 'y': y, 'w': w, 'h': h}},
              )
            )
          elif abs(eye_offset) > EYE_SWEEP_OFFSET_THRESHOLD:
            events.append(
              DetectionEvent(
                code='gaze_sweep_detected',
                label='Gaze sweep detected',
                severity='warning',
                score=min(1.0, abs(eye_offset)),
                details={'offset_ratio': round(eye_offset, 3), 'bbox': {'x': x, 'y': y, 'w': w, 'h': h}},
              )
            )

          if len(eye_centers) >= 2:
            left_eye = eye_centers[0]
            right_eye = eye_centers[-1]
            dx = max(abs(right_eye[0] - left_eye[0]), 1.0)
            dy = abs(right_eye[1] - left_eye[1])
            slope_ratio = dy / dx
            head_angle = math.degrees(math.atan2(dy, dx))
            eye_spread_ratio = max(right_eye[0] - left_eye[0], 1.0) / max(w, 1)
            metrics.eye_line_angle = round(head_angle, 2)

            if eye_spread_ratio < HEAD_YAW_SPREAD_THRESHOLD and abs(eye_offset) > HEAD_YAW_OFFSET_THRESHOLD:
              events.append(
                DetectionEvent(
                  code='head_yaw_detected',
                  label='Head yaw indicates looking away',
                  severity='warning',
                  score=min(1.0, abs(eye_offset) + 0.2),
                  details={
                    'eye_spread_ratio': round(eye_spread_ratio, 3),
                    'offset_ratio': round(eye_offset, 3),
                    'bbox': {'x': x, 'y': y, 'w': w, 'h': h},
                  },
                )
              )
            if slope_ratio > HEAD_TURN_RATIO_THRESHOLD and head_angle >= MIN_HEAD_TURN_DEGREES:
              events.append(
                DetectionEvent(
                  code='head_turn_detected',
                  label='Head turn exceeds normal threshold',
                  severity='critical',
                  score=min(1.0, head_angle / 45.0),
                  details={
                    'angle_deg': round(head_angle, 2),
                    'slope_ratio': round(slope_ratio, 3),
                    'bbox': {'x': x, 'y': y, 'w': w, 'h': h},
                  },
                )
              )

        if metrics.pitch_ratio is not None and metrics.pitch_ratio > 0.44:
          events.append(
            DetectionEvent(
              code='looking_down',
              label='Looking down for a sustained angle',
              severity='warning',
              score=min(1.0, metrics.pitch_ratio),
              details={'pitch_ratio': metrics.pitch_ratio, 'bbox': {'x': x, 'y': y, 'w': w, 'h': h}},
            )
          )

    face_alert_labels: list[str] = []
    if any(event.code == 'multiple_faces' for event in events):
      for index, face_box in enumerate(filtered_faces, start=1):
        _draw_box(
          annotated,
          face_box,
          label=f'face {index}',
          color=COLOR_CRITICAL,
        )
    elif primary_face:
      face_alert_codes = {'eyes_off_screen', 'gaze_sweep_detected', 'head_yaw_detected', 'head_turn_detected', 'looking_down'}
      face_alerts = [event for event in events if event.code in face_alert_codes]
      face_alert_labels = [event.code for event in face_alerts]
      _draw_box(
        annotated,
        primary_face,
        label=' / '.join(face_alert_labels[:2]) if face_alert_labels else 'candidate face',
        color=COLOR_CRITICAL if face_alert_labels else COLOR_NORMAL,
      )

    for eye_box in eye_boxes_global:
      ex, ey, ew, eh = eye_box
      eye_color = COLOR_CRITICAL if face_alert_labels else COLOR_EYE
      cv2.rectangle(annotated, (ex, ey), (ex + ew, ey + eh), eye_color, 2)

    object_metrics, object_events, object_notes = _analyze_objects(image, annotated)
    detector_notes.extend(object_notes)
    metrics.phone_detected = object_metrics['phone_detected']
    metrics.book_detected = object_metrics['book_detected']

    merged_events = _dedupe_events(events + object_events)
    severity = _severity_from_events(merged_events)

    banner = {
      'normal': ('Monitoring normal', (0, 190, 120)),
      'warning': ('Suspicious behavior detected', (0, 180, 255)),
      'critical': ('High-risk cheating signal', (30, 30, 220)),
    }[severity]
    cv2.rectangle(annotated, (0, 0), (image.shape[1], 40), banner[1], -1)
    cv2.putText(annotated, banner[0], (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return AnalysisResult(
      severity=severity,
      metrics=metrics,
      events=merged_events,
      annotated_frame=_encode_image(annotated),
    )
