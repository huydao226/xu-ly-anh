from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


FRAME_WIDTH = 320
_CASCADE_LOCK = threading.Lock()
_FACE_CASCADE = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / 'haarcascade_frontalface_default.xml'))
_EYE_CASCADE = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / 'haarcascade_eye_tree_eyeglasses.xml'))
_UPPERBODY_CASCADE = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / 'haarcascade_upperbody.xml'))


@dataclass(frozen=True)
class FrameFeatureBundle:
  vector: list[float]
  gray: np.ndarray
  face_box: dict[str, int] | None
  face_present: bool
  pose_present: bool


def resize_frame(frame: np.ndarray, frame_width: int = FRAME_WIDTH) -> np.ndarray:
  height, width = frame.shape[:2]
  if width <= 0 or height <= 0:
    raise ValueError('Invalid frame shape')
  target_height = max(1, int(height * (frame_width / float(width))))
  return cv2.resize(frame, (frame_width, target_height), interpolation=cv2.INTER_AREA)


def _safe_ratio(numerator: float, denominator: float) -> float:
  if abs(denominator) < 1e-6:
    return 0.0
  return float(numerator / denominator)


def _detect_regions(gray: np.ndarray, cascade: cv2.CascadeClassifier, **kwargs):
  if gray.size == 0 or cascade.empty():
    return []
  with _CASCADE_LOCK:
    try:
      regions = cascade.detectMultiScale(gray, **kwargs)
    except cv2.error:
      return []
  return list(regions) if len(regions) else []


def _select_primary_face(gray: np.ndarray) -> tuple[tuple[int, int, int, int] | None, list[tuple[int, int, int, int]]]:
  faces = _detect_regions(gray, _FACE_CASCADE, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
  if not faces:
    return None, []
  face = max(faces, key=lambda item: item[2] * item[3])
  return face, faces


def _select_upper_body(gray: np.ndarray) -> tuple[int, int, int, int] | None:
  bodies = _detect_regions(gray, _UPPERBODY_CASCADE, scaleFactor=1.05, minNeighbors=3, minSize=(60, 60))
  if not bodies:
    return None
  return max(bodies, key=lambda item: item[2] * item[3])


def _extract_eye_features(face_gray: np.ndarray, face_width: int, face_height: int) -> tuple[list[float], list[tuple[int, int, int, int]]]:
  eyes = _detect_regions(face_gray, _EYE_CASCADE, scaleFactor=1.05, minNeighbors=4, minSize=(12, 12))
  eyes = sorted(eyes, key=lambda item: item[2] * item[3], reverse=True)[:2]
  if len(eyes) < 2:
    return [1.0 if eyes else 0.0, 0.0, 0.0, 0.0, 0.0], eyes
  eyes = sorted(eyes, key=lambda item: item[0])
  left_eye, right_eye = eyes
  left_center_x = left_eye[0] + left_eye[2] / 2.0
  left_center_y = left_eye[1] + left_eye[3] / 2.0
  right_center_x = right_eye[0] + right_eye[2] / 2.0
  right_center_y = right_eye[1] + right_eye[3] / 2.0
  eye_distance = max(right_center_x - left_center_x, 1e-6)
  eye_mid_x = (left_center_x + right_center_x) / 2.0
  eye_mid_y = (left_center_y + right_center_y) / 2.0
  features = [
    1.0,
    eye_distance / float(face_width),
    _safe_ratio(eye_mid_x - face_width / 2.0, face_width),
    _safe_ratio(eye_mid_y, face_height),
    ((left_eye[3] / max(left_eye[2], 1e-6)) + (right_eye[3] / max(right_eye[2], 1e-6))) / 2.0,
  ]
  return features, eyes


def extract_frame_features(
  image: np.ndarray,
  previous_gray: np.ndarray | None,
  *,
  frame_width: int = FRAME_WIDTH,
) -> FrameFeatureBundle:
  resized = resize_frame(image, frame_width)
  gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
  frame_height, frame_width_actual = gray.shape[:2]

  brightness = float(gray.mean() / 255.0)
  edges = cv2.Canny(gray, 50, 150)
  edge_density = float((edges > 0).mean())
  if previous_gray is None or previous_gray.shape != gray.shape:
    motion_score = 0.0
  else:
    motion_score = float(cv2.absdiff(gray, previous_gray).mean() / 255.0)

  primary_face, all_faces = _select_primary_face(gray)
  upper_body = _select_upper_body(gray)

  face_present = primary_face is not None
  face_box = None
  face_features = [0.0, 0.0, 0.0, 0.0]
  eye_features = [0.0, 0.0, 0.0, 0.0, 0.0]
  lower_face_edge_density = 0.0
  if primary_face is not None:
    fx, fy, fw, fh = primary_face
    face_area_ratio = (fw * fh) / float(frame_width_actual * frame_height)
    face_center_x = (fx + fw / 2.0) / float(frame_width_actual)
    face_center_y = (fy + fh / 2.0) / float(frame_height)
    face_features = [1.0, face_area_ratio, face_center_x, face_center_y]
    face_roi = gray[fy : fy + fh, fx : fx + fw]
    eye_features, eyes = _extract_eye_features(face_roi, fw, fh)
    lower_half = face_roi[int(fh * 0.55) :, :]
    if lower_half.size > 0:
      lower_edges = cv2.Canny(lower_half, 50, 150)
      lower_face_edge_density = float((lower_edges > 0).mean())

    original_height, original_width = image.shape[:2]
    scale_x = original_width / float(frame_width_actual)
    scale_y = original_height / float(frame_height)
    face_box = {
      'x': int(fx * scale_x),
      'y': int(fy * scale_y),
      'w': int(fw * scale_x),
      'h': int(fh * scale_y),
    }

  upperbody_present = upper_body is not None
  upperbody_features = [0.0, 0.0, 0.0, 0.0, 0.0]
  if upper_body is not None:
    bx, by, bw, bh = upper_body
    body_area_ratio = (bw * bh) / float(frame_width_actual * frame_height)
    body_center_x = (bx + bw / 2.0) / float(frame_width_actual)
    body_center_y = (by + bh / 2.0) / float(frame_height)
    face_body_offset_x = 0.0 if not face_present else body_center_x - face_features[2]
    face_body_size_ratio = 0.0 if not face_present else _safe_ratio(face_features[1], body_area_ratio)
    upperbody_features = [
      1.0,
      body_area_ratio,
      body_center_x,
      body_center_y,
      face_body_offset_x + face_body_size_ratio * 0.1,
    ]

  vector = [
    brightness,
    motion_score,
    edge_density,
    *face_features,
    *eye_features,
    lower_face_edge_density,
    1.0 if len(all_faces) > 1 else 0.0,
    *upperbody_features,
  ]
  return FrameFeatureBundle(
    vector=vector,
    gray=gray,
    face_box=face_box,
    face_present=face_present,
    pose_present=upperbody_present,
  )


def feature_names() -> list[str]:
  return [
    'brightness',
    'motion_score',
    'edge_density',
    'face_present',
    'face_area_ratio',
    'face_center_x',
    'face_center_y',
    'eye_pair_present',
    'eye_distance_ratio',
    'yaw_proxy',
    'pitch_proxy',
    'eye_open_ratio',
    'lower_face_edge_density',
    'multiple_faces_proxy',
    'upperbody_present',
    'upperbody_area_ratio',
    'upperbody_center_x',
    'upperbody_center_y',
    'face_body_relation',
  ]
