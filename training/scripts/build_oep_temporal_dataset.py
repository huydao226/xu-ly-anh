from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGMENTS_CSV = REPO_ROOT / 'training' / 'data' / 'external' / 'oep_multiview' / 'notes' / 'oep_webcam_segments.csv'
DEFAULT_OUTPUT = REPO_ROOT / 'training' / 'data' / 'processed' / 'oep_webcam_temporal.jsonl'
DEFAULT_MODE = 'webcam'
VALID_MODES = {'webcam', 'multiview'}


@dataclass
class VideoInfo:
  path: Path
  capture: cv2.VideoCapture
  fps: float
  frame_count: int
  duration_seconds: float


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description='Build a temporal dataset directly from OEP videos.')
  parser.add_argument('--segments-csv', type=Path, default=DEFAULT_SEGMENTS_CSV)
  parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument('--mode', type=str, choices=sorted(VALID_MODES), default=DEFAULT_MODE)
  parser.add_argument('--frames-per-sample', type=int, default=16)
  parser.add_argument('--frame-width', type=int, default=320)
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--min-normal-gap-seconds', type=int, default=8)
  parser.add_argument('--normal-window-seconds', type=int, default=12)
  parser.add_argument('--max-segments', type=int, default=0, help='Optional cap for quick experiments. 0 means all segments.')
  return parser.parse_args()


def load_segment_rows(path: Path) -> list[dict[str, str]]:
  with path.open('r', encoding='utf-8', newline='') as handle:
    reader = csv.DictReader(handle)
    required = {
      'sample_id',
      'subject_id',
      'subject_group',
      'webcam_video_path',
      'wearcam_video_path',
      'audio_path',
      'start_seconds',
      'end_seconds',
      'duration_seconds',
      'cheat_type_name',
    }
    if not reader.fieldnames or not required.issubset(reader.fieldnames):
      raise ValueError(f'CSV must contain columns: {sorted(required)}')
    return [dict(row) for row in reader]


def assign_subject_splits(subject_ids: list[str], seed: int) -> dict[str, str]:
  unique_subjects = sorted(set(subject_ids))
  rng = random.Random(seed)
  rng.shuffle(unique_subjects)
  total = len(unique_subjects)
  train_cutoff = max(1, int(total * 0.7))
  val_cutoff = max(train_cutoff + 1, int(total * 0.85))

  split_map: dict[str, str] = {}
  for index, subject_id in enumerate(unique_subjects):
    if index < train_cutoff:
      split = 'train'
    elif index < val_cutoff:
      split = 'val'
    else:
      split = 'test'
    split_map[subject_id] = split
  return split_map


def open_video(path: Path) -> VideoInfo:
  capture = cv2.VideoCapture(str(path))
  if not capture.isOpened():
    raise FileNotFoundError(f'Unable to open video: {path}')
  fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
  frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
  if fps <= 0:
    fps = 30.0
  duration_seconds = frame_count / fps if frame_count > 0 else 0.0
  return VideoInfo(path=path, capture=capture, fps=fps, frame_count=frame_count, duration_seconds=duration_seconds)


def evenly_spaced_timestamps(start_seconds: float, end_seconds: float, count: int) -> list[float]:
  if count <= 1:
    return [start_seconds]
  if end_seconds <= start_seconds:
    return [start_seconds for _ in range(count)]
  step = (end_seconds - start_seconds) / float(count - 1)
  return [start_seconds + step * index for index in range(count)]


def resize_gray(frame: cv2.typing.MatLike, frame_width: int) -> cv2.typing.MatLike:
  height, width = frame.shape[:2]
  if width <= 0 or height <= 0:
    raise ValueError('Invalid frame shape')
  target_height = max(1, int(height * (frame_width / float(width))))
  resized = cv2.resize(frame, (frame_width, target_height), interpolation=cv2.INTER_AREA)
  return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


def detect_primary_face(gray: cv2.typing.MatLike, cascade: cv2.CascadeClassifier) -> tuple[int, float, float, float]:
  faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
  if len(faces) == 0:
    return 0, 0.0, 0.0, 0.0
  frame_height, frame_width = gray.shape[:2]
  x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
  area_ratio = (w * h) / float(frame_width * frame_height)
  center_x = (x + w / 2.0) / float(frame_width)
  center_y = (y + h / 2.0) / float(frame_height)
  return int(len(faces)), float(area_ratio), float(center_x), float(center_y)


def compute_frame_features(
  *,
  gray: cv2.typing.MatLike,
  previous_gray: cv2.typing.MatLike | None,
  cascade: cv2.CascadeClassifier,
) -> list[float]:
  brightness = float(gray.mean() / 255.0)
  edges = cv2.Canny(gray, 50, 150)
  edge_density = float((edges > 0).mean())
  if previous_gray is None or previous_gray.shape != gray.shape:
    motion_score = 0.0
  else:
    motion_score = float(cv2.absdiff(gray, previous_gray).mean() / 255.0)
  face_count, face_area_ratio, face_center_x, face_center_y = detect_primary_face(gray, cascade)
  return [
    brightness,
    motion_score,
    edge_density,
    float(face_count),
    face_area_ratio,
    face_center_x,
    face_center_y,
  ]


def read_frame_at_seconds(video: VideoInfo, second: float) -> cv2.typing.MatLike | None:
  target_seconds = max(0.0, min(float(second), max(video.duration_seconds - 1e-3, 0.0)))
  video.capture.set(cv2.CAP_PROP_POS_MSEC, target_seconds * 1000.0)
  ok, frame = video.capture.read()
  if not ok:
    return None
  return frame


def extract_view_sequence(
  *,
  video: VideoInfo,
  start_seconds: float,
  end_seconds: float,
  frames_per_sample: int,
  frame_width: int,
  cascade: cv2.CascadeClassifier,
) -> list[list[float]]:
  timestamps = evenly_spaced_timestamps(start_seconds, end_seconds, frames_per_sample)
  sequence: list[list[float]] = []
  previous_gray = None
  for timestamp in timestamps:
    frame = read_frame_at_seconds(video, timestamp)
    if frame is None:
      if sequence:
        sequence.append(sequence[-1][:])
      else:
        sequence.append([0.0] * 7)
      continue
    gray = resize_gray(frame, frame_width)
    features = compute_frame_features(gray=gray, previous_gray=previous_gray, cascade=cascade)
    sequence.append(features)
    previous_gray = gray
  return sequence


def merge_sequences(webcam_features: list[list[float]], wearcam_features: list[list[float]]) -> list[list[float]]:
  merged: list[list[float]] = []
  for webcam_row, wearcam_row in zip(webcam_features, wearcam_features):
    merged.append(webcam_row + wearcam_row)
  return merged


def feature_names_for_mode(mode: str) -> list[str]:
  base = [
    'brightness',
    'motion_score',
    'edge_density',
    'face_count',
    'face_area_ratio',
    'face_center_x',
    'face_center_y',
  ]
  if mode == 'webcam':
    return [f'webcam__{name}' for name in base]
  if mode == 'multiview':
    return [f'webcam__{name}' for name in base] + [f'wearcam__{name}' for name in base]
  raise ValueError(f'Unsupported mode: {mode}')


def infer_normal_segments(
  *,
  rows: list[dict[str, str]],
  video_cache: dict[Path, VideoInfo],
  min_gap_seconds: int,
  normal_window_seconds: int,
) -> list[dict[str, str]]:
  grouped: dict[str, list[dict[str, str]]] = {}
  for row in rows:
    grouped.setdefault(row['subject_id'], []).append(row)

  normals: list[dict[str, str]] = []
  for subject_id, subject_rows in sorted(grouped.items()):
    subject_rows.sort(key=lambda row: int(row['start_seconds']))
    webcam_path = Path(subject_rows[0]['webcam_video_path'])
    wearcam_path = Path(subject_rows[0]['wearcam_video_path']) if subject_rows[0]['wearcam_video_path'] else None
    audio_path = subject_rows[0]['audio_path']
    subject_group = subject_rows[0]['subject_group']
    video_duration = int(video_cache[webcam_path].duration_seconds)
    previous_end = 0
    normal_index = 1

    def maybe_add_gap(gap_start: int, gap_end: int) -> None:
      nonlocal normal_index
      gap_duration = gap_end - gap_start
      if gap_duration < min_gap_seconds:
        return
      end_seconds = min(gap_start + normal_window_seconds, gap_end)
      normals.append(
        {
          'sample_id': f'{subject_id}_normal_{normal_index:03d}',
          'subject_id': subject_id,
          'subject_group': subject_group,
          'webcam_video_path': str(webcam_path),
          'wearcam_video_path': str(wearcam_path) if wearcam_path else '',
          'audio_path': audio_path,
          'start_seconds': str(gap_start),
          'end_seconds': str(end_seconds),
          'duration_seconds': str(max(end_seconds - gap_start, 0)),
          'cheat_type_name': 'normal',
        }
      )
      normal_index += 1

    for row in subject_rows:
      current_start = int(row['start_seconds'])
      current_end = int(row['end_seconds'])
      maybe_add_gap(previous_end, current_start)
      previous_end = max(previous_end, current_end)

    maybe_add_gap(previous_end, video_duration)

  return normals


def main() -> None:
  args = parse_args()
  segments_csv = args.segments_csv.resolve()
  output_path = args.output.resolve()
  rows = load_segment_rows(segments_csv)
  if not rows:
    raise ValueError(f'No OEP segment rows found in {segments_csv}')

  output_path.parent.mkdir(parents=True, exist_ok=True)
  split_map = assign_subject_splits([row['subject_id'] for row in rows], args.seed)
  feature_names = feature_names_for_mode(args.mode)

  cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / 'haarcascade_frontalface_default.xml'))
  if cascade.empty():
    raise RuntimeError('Failed to load OpenCV frontal-face cascade')

  video_cache: dict[Path, VideoInfo] = {}

  def get_video(path_str: str) -> VideoInfo:
    path = Path(path_str)
    video = video_cache.get(path)
    if video is None:
      video = open_video(path)
      video_cache[path] = video
    return video

  normal_rows = infer_normal_segments(
    rows=rows,
    video_cache={Path(row['webcam_video_path']): get_video(row['webcam_video_path']) for row in rows},
    min_gap_seconds=args.min_normal_gap_seconds,
    normal_window_seconds=args.normal_window_seconds,
  )
  all_rows = rows + normal_rows
  if args.max_segments > 0:
    all_rows = all_rows[: args.max_segments]

  label_names = sorted({row['cheat_type_name'] for row in all_rows})
  label_map = {label: index for index, label in enumerate(label_names)}

  with output_path.open('w', encoding='utf-8') as handle:
    for row in all_rows:
      webcam_video = get_video(row['webcam_video_path'])
      webcam_features = extract_view_sequence(
        video=webcam_video,
        start_seconds=float(row['start_seconds']),
        end_seconds=float(row['end_seconds']),
        frames_per_sample=args.frames_per_sample,
        frame_width=args.frame_width,
        cascade=cascade,
      )

      if args.mode == 'multiview':
        if not row['wearcam_video_path']:
          raise ValueError(f'Missing wearcam path for sample {row["sample_id"]}')
        wearcam_video = get_video(row['wearcam_video_path'])
        wearcam_features = extract_view_sequence(
          video=wearcam_video,
          start_seconds=float(row['start_seconds']),
          end_seconds=float(row['end_seconds']),
          frames_per_sample=args.frames_per_sample,
          frame_width=args.frame_width,
          cascade=cascade,
        )
        features = merge_sequences(webcam_features, wearcam_features)
      else:
        features = webcam_features

      sample = {
        'sample_id': row['sample_id'],
        'subject_id': row['subject_id'],
        'split': split_map[row['subject_id']],
        'label': row['cheat_type_name'],
        'label_id': label_map[row['cheat_type_name']],
        'frame_start': 0,
        'frame_end': len(features) - 1,
        'frame_count': len(features),
        'feature_names': feature_names,
        'features': features,
        'notes': f'OEP {args.mode} sample from {row["subject_group"]}',
      }
      handle.write(json.dumps(sample, ensure_ascii=False) + '\n')

  for video in video_cache.values():
    video.capture.release()

  summary = {
    'source_csv': str(segments_csv),
    'output': str(output_path),
    'mode': args.mode,
    'frames_per_sample': args.frames_per_sample,
    'frame_width': args.frame_width,
    'label_map': label_map,
    'subject_split_map': split_map,
    'sample_count': len(all_rows),
    'oep_segment_count': len(rows),
    'normal_gap_sample_count': len(normal_rows),
  }
  (output_path.parent / f'{output_path.stem}_summary.json').write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
    encoding='utf-8',
  )

  print('Built OEP temporal dataset')
  print(f'- source_csv: {segments_csv}')
  print(f'- mode: {args.mode}')
  print(f'- oep labeled segments: {len(rows)}')
  print(f'- inferred normal samples: {len(normal_rows)}')
  print(f'- total samples: {len(all_rows)}')
  print(f'- output: {output_path}')
  print(f'- labels: {label_names}')


if __name__ == '__main__':
  main()
