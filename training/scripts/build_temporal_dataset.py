from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SESSIONS_ROOT = REPO_ROOT / 'training' / 'data' / 'raw_sessions'
DEFAULT_LABELS = REPO_ROOT / 'training' / 'data' / 'annotations' / 'behavior_segments.csv'
DEFAULT_OUTPUT = REPO_ROOT / 'training' / 'data' / 'processed' / 'temporal_sequences.jsonl'

EVENT_CODES = [
  'face_missing',
  'multiple_faces',
  'eyes_off_screen',
  'gaze_sweep_detected',
  'head_yaw_detected',
  'head_turn_detected',
  'looking_down',
  'phone_visible',
  'book_visible',
]

FEATURE_NAMES = [
  'faces_detected',
  'phone_detected',
  'book_detected',
  'yaw_ratio',
  'pitch_ratio',
  'eye_line_angle',
  'event_count',
  'severity_score',
  *[f'event__{code}' for code in EVENT_CODES],
]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description='Build temporal behavior samples from captured session metadata.')
  parser.add_argument('--sessions-root', type=Path, default=DEFAULT_SESSIONS_ROOT)
  parser.add_argument('--labels', type=Path, default=DEFAULT_LABELS)
  parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
  return parser.parse_args()


def load_segments(path: Path) -> list[dict[str, str]]:
  with path.open('r', encoding='utf-8', newline='') as handle:
    reader = csv.DictReader(handle)
    required = {'session_id', 'start_frame', 'end_frame', 'label', 'split', 'notes'}
    if not reader.fieldnames or not required.issubset(reader.fieldnames):
      raise ValueError(f'CSV must contain columns: {sorted(required)}')
    return [dict(row) for row in reader]


def load_session_frames(sessions_root: Path, session_id: str) -> list[dict]:
  metadata_path = sessions_root / session_id / 'metadata.jsonl'
  frames: list[dict] = []
  with metadata_path.open('r', encoding='utf-8') as handle:
    for line in handle:
      if line.strip():
        frames.append(json.loads(line))
  frames.sort(key=lambda item: int(item['frame_index']))
  return frames


def severity_to_score(severity: str) -> float:
  return {
    'normal': 0.0,
    'warning': 1.0,
    'critical': 2.0,
  }.get(severity, 0.0)


def frame_to_vector(frame: dict) -> list[float]:
  metrics = frame.get('metrics', {})
  event_codes = {event.get('code') for event in frame.get('events', [])}
  vector = [
    float(metrics.get('faces_detected', 0) or 0),
    1.0 if metrics.get('phone_detected') else 0.0,
    1.0 if metrics.get('book_detected') else 0.0,
    float(metrics.get('yaw_ratio') or 0.0),
    float(metrics.get('pitch_ratio') or 0.0),
    float(metrics.get('eye_line_angle') or 0.0),
    float(len(frame.get('events', []))),
    severity_to_score(str(frame.get('severity', 'normal'))),
  ]
  for code in EVENT_CODES:
    vector.append(1.0 if code in event_codes else 0.0)
  return vector


def main() -> None:
  args = parse_args()
  sessions_root = args.sessions_root.resolve()
  labels_path = args.labels.resolve()
  output_path = args.output.resolve()
  output_path.parent.mkdir(parents=True, exist_ok=True)

  segments = load_segments(labels_path)
  if not segments:
    raise ValueError(f'No behavior segments found in {labels_path}')

  session_cache: dict[str, list[dict]] = {}
  label_names = sorted({segment['label'].strip() for segment in segments if segment['label'].strip()})
  label_map = {label: idx for idx, label in enumerate(label_names)}

  with output_path.open('w', encoding='utf-8') as out_handle:
    for sample_index, segment in enumerate(segments, start=1):
      session_id = segment['session_id'].strip()
      start_frame = int(segment['start_frame'])
      end_frame = int(segment['end_frame'])
      label = segment['label'].strip()
      split = (segment['split'].strip() or 'train').lower()
      notes = segment['notes'].strip()
      frames = session_cache.setdefault(session_id, load_session_frames(sessions_root, session_id))
      selected = [frame for frame in frames if start_frame <= int(frame['frame_index']) <= end_frame]
      if not selected:
        raise ValueError(f'No frames found for session {session_id} between {start_frame} and {end_frame}')

      sample = {
        'sample_id': f'{session_id}_{sample_index:04d}',
        'session_id': session_id,
        'split': split,
        'label': label,
        'label_id': label_map[label],
        'frame_start': start_frame,
        'frame_end': end_frame,
        'frame_count': len(selected),
        'feature_names': FEATURE_NAMES,
        'features': [frame_to_vector(frame) for frame in selected],
        'notes': notes,
      }
      out_handle.write(json.dumps(sample, ensure_ascii=False) + '\n')

  (output_path.parent / 'temporal_label_map.json').write_text(
    json.dumps(label_map, ensure_ascii=False, indent=2) + '\n',
    encoding='utf-8',
  )
  (output_path.parent / 'temporal_feature_spec.json').write_text(
    json.dumps({'feature_names': FEATURE_NAMES, 'event_codes': EVENT_CODES}, ensure_ascii=False, indent=2) + '\n',
    encoding='utf-8',
  )

  print('Prepared temporal dataset')
  print(f'- labels: {labels_path}')
  print(f'- sessions: {sessions_root}')
  print(f'- output: {output_path}')
  print(f'- classes: {label_names}')
  print(f'- samples: {len(segments)}')


if __name__ == '__main__':
  main()
