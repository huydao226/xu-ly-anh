from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'backend'))

from oep_service.feature_extractor import FRAME_WIDTH, extract_frame_features, feature_names  # noqa: E402


DEFAULT_SEGMENTS_CSV = REPO_ROOT / 'training' / 'data' / 'external' / 'oep_multiview' / 'notes' / 'oep_webcam_segments.csv'
DEFAULT_OUTPUT = REPO_ROOT / 'training' / 'data' / 'processed' / 'oep_webcam_temporal_v3.jsonl'
DEFAULT_OEP_ROOT = REPO_ROOT / 'training' / 'data' / 'external' / 'oep_multiview' / 'raw' / 'OEP database'
LABEL_MAP_V3 = {
  'normal': 'normal',
  'type_1': 'suspicious_action',
  'type_2': 'suspicious_action',
  'type_3': 'normal',
  'type_5': 'device',
  'type_6': 'suspicious_action',
}


@dataclass
class VideoInfo:
  path: Path
  capture: cv2.VideoCapture
  fps: float
  frame_count: int
  duration_seconds: float


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description='Build the enhanced OEP webcam temporal dataset for monitor v3.')
  parser.add_argument('--segments-csv', type=Path, default=DEFAULT_SEGMENTS_CSV)
  parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument('--oep-root', type=Path, default=DEFAULT_OEP_ROOT)
  parser.add_argument('--frames-per-sample', type=int, default=16)
  parser.add_argument('--frame-width', type=int, default=FRAME_WIDTH)
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--min-normal-gap-seconds', type=int, default=8)
  parser.add_argument('--normal-window-seconds', type=int, default=12)
  parser.add_argument('--max-segments', type=int, default=0)
  return parser.parse_args()


def load_segment_rows(path: Path) -> list[dict[str, str]]:
  with path.open('r', encoding='utf-8', newline='') as handle:
    reader = csv.DictReader(handle)
    required = {
      'sample_id',
      'subject_id',
      'subject_group',
      'webcam_video_path',
      'start_seconds',
      'end_seconds',
      'duration_seconds',
      'cheat_type_name',
    }
    if not reader.fieldnames or not required.issubset(reader.fieldnames):
      raise ValueError(f'CSV must contain columns: {sorted(required)}')
    return [dict(row) for row in reader]


def assign_subject_splits(rows: list[dict[str, str]], seed: int) -> dict[str, str]:
  unique_subjects = sorted({row['subject_id'] for row in rows})
  rng = random.Random(seed)
  total = len(unique_subjects)
  val_count = max(1, int(math.ceil(total * 0.15)))
  test_count = max(1, int(math.ceil(total * 0.15)))
  train_count = max(1, total - val_count - test_count)
  if train_count + val_count + test_count > total:
    train_count = max(1, total - val_count - test_count)

  labels_by_subject: dict[str, set[str]] = defaultdict(set)
  for row in rows:
    labels_by_subject[row['subject_id']].add(remap_label(row['cheat_type_name']))

  device_subjects = sorted(subject for subject, labels in labels_by_subject.items() if 'device' in labels)
  non_device_subjects = sorted(subject for subject in unique_subjects if subject not in set(device_subjects))
  rng.shuffle(device_subjects)
  rng.shuffle(non_device_subjects)

  val_subjects: list[str] = []
  test_subjects: list[str] = []
  train_subjects: list[str] = []

  if device_subjects:
    val_subjects.append(device_subjects.pop())
  if device_subjects:
    test_subjects.append(device_subjects.pop())
  train_subjects.extend(device_subjects)

  def fill_bucket(bucket: list[str], target_size: int) -> None:
    while len(bucket) < target_size and non_device_subjects:
      bucket.append(non_device_subjects.pop())

  fill_bucket(val_subjects, val_count)
  fill_bucket(test_subjects, test_count)
  train_subjects.extend(non_device_subjects)

  # Safety fallback if the rounded split sizes leave a bucket oversized or undersized.
  while len(train_subjects) > train_count:
    if len(val_subjects) < val_count:
      val_subjects.append(train_subjects.pop())
    elif len(test_subjects) < test_count:
      test_subjects.append(train_subjects.pop())
    else:
      break

  split_map: dict[str, str] = {}
  for subject_id in train_subjects:
    split_map[subject_id] = 'train'
  for subject_id in val_subjects:
    split_map[subject_id] = 'val'
  for subject_id in test_subjects:
    split_map[subject_id] = 'test'

  for subject_id in unique_subjects:
    split_map.setdefault(subject_id, 'train')
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


def resolve_oep_video_path(raw_path: str, subject_id: str, oep_root: Path) -> Path:
  candidate = Path(raw_path)
  if candidate.exists():
    return candidate

  normalized = raw_path.replace('\\', '/')
  filename = normalized.split('/')[-1]
  subject_dir = oep_root / subject_id
  subject_candidate = subject_dir / filename
  if subject_candidate.exists():
    return subject_candidate

  if filename:
    matches = sorted(subject_dir.glob(f'*{Path(filename).suffix}'))
    if len(matches) == 1:
      return matches[0]

  raise FileNotFoundError(f'Unable to resolve OEP video path for subject {subject_id}: {raw_path}')


def evenly_spaced_timestamps(start_seconds: float, end_seconds: float, count: int) -> list[float]:
  if count <= 1:
    return [start_seconds]
  if end_seconds <= start_seconds:
    return [start_seconds for _ in range(count)]
  step = (end_seconds - start_seconds) / float(count - 1)
  return [start_seconds + step * index for index in range(count)]


def read_frame_at_seconds(video: VideoInfo, second: float):
  target_seconds = max(0.0, min(float(second), max(video.duration_seconds - 1e-3, 0.0)))
  video.capture.set(cv2.CAP_PROP_POS_MSEC, target_seconds * 1000.0)
  ok, frame = video.capture.read()
  if not ok:
    return None
  return frame


def extract_sequence(video: VideoInfo, start_seconds: float, end_seconds: float, frames_per_sample: int, frame_width: int) -> list[list[float]]:
  timestamps = evenly_spaced_timestamps(start_seconds, end_seconds, frames_per_sample)
  sequence: list[list[float]] = []
  previous_gray = None
  feature_dim = len(feature_names())
  for timestamp in timestamps:
    frame = read_frame_at_seconds(video, timestamp)
    if frame is None:
      sequence.append(sequence[-1][:] if sequence else [0.0] * feature_dim)
      continue
    bundle = extract_frame_features(frame, previous_gray, frame_width=frame_width)
    sequence.append(bundle.vector)
    previous_gray = bundle.gray
  return sequence


def infer_normal_segments(
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
    subject_group = subject_rows[0]['subject_group']
    video_duration = int(video_cache[webcam_path].duration_seconds)
    previous_end = 0
    normal_index = 1

    def maybe_add_gap(gap_start: int, gap_end: int) -> None:
      nonlocal normal_index
      if gap_end - gap_start < min_gap_seconds:
        return
      end_seconds = min(gap_start + normal_window_seconds, gap_end)
      normals.append(
        {
          'sample_id': f'{subject_id}_normal_v3_{normal_index:03d}',
          'subject_id': subject_id,
          'subject_group': subject_group,
          'webcam_video_path': str(webcam_path),
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


def remap_label(raw_label: str) -> str:
  return LABEL_MAP_V3.get(raw_label, 'suspicious_action')


def main() -> None:
  args = parse_args()
  segments_csv = args.segments_csv.resolve()
  output_path = args.output.resolve()
  oep_root = args.oep_root.resolve()
  rows = load_segment_rows(segments_csv)
  if not rows:
    raise ValueError(f'No OEP segment rows found in {segments_csv}')
  normalized_rows = [
    {
      **row,
      'webcam_video_path': str(resolve_oep_video_path(row['webcam_video_path'], row['subject_id'], oep_root)),
    }
    for row in rows
  ]

  output_path.parent.mkdir(parents=True, exist_ok=True)
  split_map = assign_subject_splits(normalized_rows, args.seed)
  names = feature_names()
  video_cache: dict[Path, VideoInfo] = {}

  def get_video(path_str: str) -> VideoInfo:
    path = Path(path_str)
    video = video_cache.get(path)
    if video is None:
      video = open_video(path)
      video_cache[path] = video
    return video

  normal_rows = infer_normal_segments(
    rows=normalized_rows,
    video_cache={Path(row['webcam_video_path']): get_video(row['webcam_video_path']) for row in normalized_rows},
    min_gap_seconds=args.min_normal_gap_seconds,
    normal_window_seconds=args.normal_window_seconds,
  )
  all_rows = normalized_rows + normal_rows
  if args.max_segments > 0:
    all_rows = all_rows[: args.max_segments]

  remapped_labels = sorted({remap_label(row['cheat_type_name']) for row in all_rows})
  label_map = {label: index for index, label in enumerate(remapped_labels)}

  with output_path.open('w', encoding='utf-8') as handle:
    for row in all_rows:
      video = get_video(row['webcam_video_path'])
      features = extract_sequence(
        video=video,
        start_seconds=float(row['start_seconds']),
        end_seconds=float(row['end_seconds']),
        frames_per_sample=args.frames_per_sample,
        frame_width=args.frame_width,
      )
      remapped_label = remap_label(row['cheat_type_name'])
      sample = {
        'sample_id': row['sample_id'],
        'subject_id': row['subject_id'],
        'split': split_map[row['subject_id']],
        'label': remapped_label,
        'label_id': label_map[remapped_label],
        'frame_start': 0,
        'frame_end': len(features) - 1,
        'frame_count': len(features),
        'feature_names': names,
        'features': features,
        'source_label': row['cheat_type_name'],
        'notes': 'OEP webcam-only v3 sample with enhanced face and pose features',
      }
      handle.write(json.dumps(sample, ensure_ascii=False) + '\n')

  for video in video_cache.values():
    video.capture.release()

  summary = {
    'source_csv': str(segments_csv),
    'output': str(output_path),
    'frames_per_sample': args.frames_per_sample,
    'frame_width': args.frame_width,
    'feature_names': names,
    'label_map': label_map,
    'source_to_target_labels': LABEL_MAP_V3,
    'subject_split_map': split_map,
    'sample_count': len(all_rows),
    'oep_segment_count': len(normalized_rows),
    'normal_gap_sample_count': len(normal_rows),
  }
  (output_path.parent / f'{output_path.stem}_summary.json').write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
    encoding='utf-8',
  )

  print('Built OEP temporal dataset v3')
  print(f'- source_csv: {segments_csv}')
  print(f'- output: {output_path}')
  print(f'- labels: {remapped_labels}')
  print(f'- feature_dim: {len(names)}')
  print(f'- oep labeled segments: {len(normalized_rows)}')
  print(f'- inferred normal samples: {len(normal_rows)}')
  print(f'- total samples: {len(all_rows)}')


if __name__ == '__main__':
  main()
