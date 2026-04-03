from __future__ import annotations

import base64
import json
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from ultralytics import YOLO

from oep_service.feature_extractor import FRAME_WIDTH, FrameFeatureBundle, extract_frame_features


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / 'training' / 'models' / 'oep_webcam_monitor_v3'
MODEL_PATH = MODEL_DIR / 'best_model.pt'
LABEL_MAP_PATH = MODEL_DIR / 'label_map.json'
METRICS_PATH = MODEL_DIR / 'metrics.json'

SEQUENCE_FRAMES = 16
MIN_FRAMES_TO_PREDICT = 8
OFFSCREEN_STREAK_THRESHOLD = 6
LIVE_NON_NORMAL_THRESHOLD_FLOOR = 0.55
DEVICE_CONFIDENCE_THRESHOLD = 0.7
DEVICE_HOLD_FRAMES = 8
DEFAULT_HIDDEN_SIZE = 64
_YOLO_LOCK = threading.Lock()
_YOLO_MODEL: YOLO | None = None
_YOLO_ERROR: str | None = None


class LSTMClassifier(nn.Module):
  def __init__(self, input_size: int, hidden_size: int, num_classes: int) -> None:
    super().__init__()
    self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
    self.head = nn.Sequential(
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(),
      nn.Dropout(p=0.2),
      nn.Linear(hidden_size, num_classes),
    )

  def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    packed = nn.utils.rnn.pack_padded_sequence(features, lengths.cpu(), batch_first=True, enforce_sorted=False)
    _, (hidden, _) = self.lstm(packed)
    return self.head(hidden[-1])


@dataclass(frozen=True)
class ModelBundle:
  model: LSTMClassifier
  labels: list[str]
  feature_dim: int
  feature_mean: list[float]
  feature_std: list[float]
  non_normal_threshold: float
  normal_index: int


@dataclass(frozen=True)
class DeviceDetection:
  detected: bool
  confidence: float
  bbox: dict[str, int] | None


def decode_image(frame_base64: str) -> np.ndarray:
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


def load_model_bundle() -> ModelBundle:
  if not MODEL_PATH.exists():
    raise FileNotFoundError(f'Missing trained model: {MODEL_PATH}')
  labels_by_id = json.loads(LABEL_MAP_PATH.read_text(encoding='utf-8'))
  metrics = json.loads(METRICS_PATH.read_text(encoding='utf-8'))
  labels = [label for label, _ in sorted(labels_by_id.items(), key=lambda item: int(item[1]))]
  feature_dim = int(metrics.get('feature_dim', 18))
  hidden_size = int(metrics.get('hidden_size', DEFAULT_HIDDEN_SIZE) or DEFAULT_HIDDEN_SIZE)
  feature_mean = [float(value) for value in metrics.get('feature_mean', [0.0] * feature_dim)]
  feature_std = [max(float(value), 1e-6) for value in metrics.get('feature_std', [1.0] * feature_dim)]
  non_normal_threshold = max(float(metrics.get('non_normal_threshold', 0.5)), LIVE_NON_NORMAL_THRESHOLD_FLOOR)
  normal_index = int(labels_by_id.get('normal', 0))

  model = LSTMClassifier(feature_dim, hidden_size, len(labels))
  state_dict = torch.load(MODEL_PATH, map_location='cpu')
  model.load_state_dict(state_dict)
  model.eval()
  return ModelBundle(
    model=model,
    labels=labels,
    feature_dim=feature_dim,
    feature_mean=feature_mean,
    feature_std=feature_std,
    non_normal_threshold=non_normal_threshold,
    normal_index=normal_index,
  )


def _get_yolo_model() -> YOLO | None:
  global _YOLO_MODEL, _YOLO_ERROR
  if _YOLO_MODEL is not None or _YOLO_ERROR is not None:
    return _YOLO_MODEL
  try:
    _YOLO_MODEL = YOLO('yolov8n.pt')
  except Exception as exc:
    _YOLO_ERROR = str(exc)
    return None
  return _YOLO_MODEL


def detect_device(image: np.ndarray) -> DeviceDetection:
  model = _get_yolo_model()
  if model is None:
    return DeviceDetection(detected=False, confidence=0.0, bbox=None)

  with _YOLO_LOCK:
    results = model.predict(image, verbose=False, imgsz=640, conf=0.25)

  best_confidence = 0.0
  best_bbox: dict[str, int] | None = None
  for result in results:
    names = result.names
    for box in result.boxes:
      class_id = int(box.cls.item())
      label = str(names.get(class_id, class_id)).lower()
      confidence = float(box.conf.item())
      if label not in {'cell phone', 'phone', 'laptop'}:
        continue
      if confidence > best_confidence:
        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
        best_confidence = confidence
        best_bbox = {'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1}
  if best_bbox is None or best_confidence < DEVICE_CONFIDENCE_THRESHOLD:
    return DeviceDetection(detected=False, confidence=best_confidence, bbox=None)
  return DeviceDetection(detected=True, confidence=best_confidence, bbox=best_bbox)


def predict_sequence(bundle: ModelBundle, features: list[list[float]]) -> tuple[str, float, list[dict[str, float]]]:
  raw_tensor = torch.tensor(features, dtype=torch.float32)
  feature_mean = torch.tensor(bundle.feature_mean, dtype=torch.float32)
  feature_std = torch.tensor(bundle.feature_std, dtype=torch.float32)
  feature_tensor = ((raw_tensor - feature_mean) / feature_std).unsqueeze(0)
  length_tensor = torch.tensor([len(features)], dtype=torch.long)
  with torch.no_grad():
    logits = bundle.model(feature_tensor, length_tensor)
    probabilities = torch.softmax(logits, dim=1)[0]

  top_index = int(torch.argmax(probabilities).item())
  top_confidence = float(probabilities[top_index].item())
  if top_index != bundle.normal_index and top_confidence < bundle.non_normal_threshold:
    top_index = bundle.normal_index
  selected_confidence = float(probabilities[top_index].item())
  scores = [
    {'label': bundle.labels[index], 'confidence': round(float(probabilities[index].item()), 4)}
    for index in range(len(bundle.labels))
  ]
  scores.sort(key=lambda item: item['confidence'], reverse=True)
  return bundle.labels[top_index], selected_confidence, scores


def absence_override(
  *,
  bundle: FrameFeatureBundle,
  offscreen_streak: int,
  probabilities: list[dict[str, float]],
) -> tuple[str | None, float | None, list[dict[str, float]]]:
  if bundle.face_present or bundle.pose_present:
    return None, None, probabilities
  if offscreen_streak < OFFSCREEN_STREAK_THRESHOLD:
    return None, None, probabilities
  confidence = round(min(0.99, 0.55 + 0.05 * (offscreen_streak - OFFSCREEN_STREAK_THRESHOLD)), 4)
  adjusted = [{'label': 'absence/offscreen', 'confidence': confidence}]
  for item in probabilities:
    adjusted.append(item)
  return 'absence/offscreen', confidence, adjusted


def annotate_frame(
  image: np.ndarray,
  *,
  face_box: dict[str, int] | None,
  device_box: dict[str, int] | None = None,
  ready: bool,
  prediction_label: str | None,
  confidence: float | None,
  buffer_size: int,
) -> str:
  annotated = image.copy()
  banner_height = 56
  banner_text_x = 18
  banner_text_y = 36

  if face_box is not None and face_box['w'] > 0 and face_box['h'] > 0:
    x, y, w, h = face_box['x'], face_box['y'], face_box['w'], face_box['h']
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 180, 255), 2)
    cv2.putText(annotated, 'face', (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 2, cv2.LINE_AA)
  if device_box is not None and device_box['w'] > 0 and device_box['h'] > 0:
    x, y, w, h = device_box['x'], device_box['y'], device_box['w'], device_box['h']
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 120, 0), 2)
    cv2.putText(annotated, 'device', (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 0), 2, cv2.LINE_AA)

  banner_text = (
    f'OEP monitor v3: {prediction_label} ({confidence:.2f})'
    if ready and prediction_label is not None and confidence is not None
    else f'Collecting sequence... {buffer_size}/{SEQUENCE_FRAMES}'
  )
  if prediction_label == 'absence/offscreen':
    banner_color = (80, 80, 230)
  elif prediction_label == 'device':
    banner_color = (0, 110, 220)
  elif prediction_label == 'suspicious_action':
    banner_color = (0, 160, 110)
  else:
    banner_color = (0, 140, 255)

  cv2.rectangle(annotated, (0, 0), (annotated.shape[1], banner_height), banner_color, -1)
  cv2.putText(
    annotated,
    banner_text,
    (banner_text_x, banner_text_y),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.72,
    (255, 255, 255),
    2,
    cv2.LINE_AA,
  )
  return _encode_image(annotated)
