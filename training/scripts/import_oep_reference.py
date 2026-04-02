from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OEP_ROOT = REPO_ROOT / 'training' / 'data' / 'external' / 'oep_multiview' / 'raw' / 'OEP database'
DEFAULT_OUTPUT_DIR = REPO_ROOT / 'training' / 'data' / 'external' / 'oep_multiview' / 'notes'

ACTING_SUBJECTS = {
  'subject1',
  'subject2',
  'subject3',
  'subject4',
  'subject5',
  'subject6',
  'subject7',
  'subject8',
  'subject9',
  'subject17',
  'subject20',
  'subject21',
  'subject22',
  'subject23',
  'subject24',
}

REAL_EXAM_SUBJECTS = {
  'subject10',
  'subject11',
  'subject12',
  'subject13',
  'subject14',
  'subject15',
  'subject16',
  'subject18',
  'subject19',
}

CHEAT_LABELS = {
  1: 'type_1',
  2: 'type_2',
  3: 'type_3',
  4: 'type_4',
  5: 'type_5',
  6: 'type_6',
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description='Scan the OEP multi-view dataset and generate reference manifests for webcam-only review.'
  )
  parser.add_argument('--oep-root', type=Path, default=DEFAULT_OEP_ROOT)
  parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
  return parser.parse_args()


def mmss_to_seconds(value: str) -> int:
  digits = ''.join(character for character in value if character.isdigit())
  if len(digits) < 4:
    raise ValueError(f'Invalid OEP timecode: {value!r}')
  minutes = int(digits[:-2])
  seconds = int(digits[-2:])
  return minutes * 60 + seconds


def parse_gt_file(path: Path) -> list[dict[str, int | str]]:
  segments: list[dict[str, int | str]] = []
  with path.open('r', encoding='utf-8') as handle:
    for line_number, raw_line in enumerate(handle, start=1):
      line = raw_line.strip()
      if not line:
        continue
      parts = line.split()
      if len(parts) != 3:
        raise ValueError(f'Invalid gt row at {path}:{line_number}: {raw_line!r}')
      start_code, end_code, label_code = parts
      cheat_type = int(label_code)
      start_seconds = mmss_to_seconds(start_code)
      end_seconds = mmss_to_seconds(end_code)
      segments.append(
        {
          'start_code': start_code,
          'end_code': end_code,
          'start_seconds': start_seconds,
          'end_seconds': end_seconds,
          'duration_seconds': max(end_seconds - start_seconds, 0),
          'cheat_type_id': cheat_type,
          'cheat_type_name': CHEAT_LABELS.get(cheat_type, f'type_{cheat_type}'),
        }
      )
  return segments


def resolve_subject_group(subject_name: str) -> str:
  if subject_name in ACTING_SUBJECTS:
    return 'acting_subject'
  if subject_name in REAL_EXAM_SUBJECTS:
    return 'real_exam_subject'
  return 'unknown'


def find_single(subject_dir: Path, pattern: str) -> Path | None:
  matches = sorted(subject_dir.glob(pattern))
  if not matches:
    return None
  if len(matches) > 1:
    raise ValueError(f'Expected one match for {pattern} in {subject_dir}, found {len(matches)}')
  return matches[0]


def build_manifests(oep_root: Path) -> tuple[list[dict[str, str | int]], list[dict[str, str | int]]]:
  subjects: list[dict[str, str | int]] = []
  segments: list[dict[str, str | int]] = []

  for subject_dir in sorted(path for path in oep_root.iterdir() if path.is_dir() and path.name.startswith('subject')):
    subject_name = subject_dir.name
    gt_path = subject_dir / 'gt.txt'
    if not gt_path.exists():
      raise FileNotFoundError(f'Missing gt.txt in {subject_dir}')

    audio_path = find_single(subject_dir, '*.wav')
    webcam_path = find_single(subject_dir, '*1.avi')
    wearcam_path = find_single(subject_dir, '*2.avi')
    subject_segments = parse_gt_file(gt_path)

    subjects.append(
      {
        'subject_id': subject_name,
        'subject_group': resolve_subject_group(subject_name),
        'gt_path': str(gt_path),
        'audio_path': str(audio_path) if audio_path else '',
        'webcam_video_path': str(webcam_path) if webcam_path else '',
        'wearcam_video_path': str(wearcam_path) if wearcam_path else '',
        'segment_count': len(subject_segments),
      }
    )

    for segment_index, segment in enumerate(subject_segments, start=1):
      segments.append(
        {
          'sample_id': f'{subject_name}_{segment_index:03d}',
          'subject_id': subject_name,
          'subject_group': resolve_subject_group(subject_name),
          'webcam_video_path': str(webcam_path) if webcam_path else '',
          'wearcam_video_path': str(wearcam_path) if wearcam_path else '',
          'audio_path': str(audio_path) if audio_path else '',
          'start_code': str(segment['start_code']),
          'end_code': str(segment['end_code']),
          'start_seconds': int(segment['start_seconds']),
          'end_seconds': int(segment['end_seconds']),
          'duration_seconds': int(segment['duration_seconds']),
          'cheat_type_id': int(segment['cheat_type_id']),
          'cheat_type_name': str(segment['cheat_type_name']),
          'reference_usage': 'webcam_only_reference',
          'notes': 'Use for reference or relabeling; final model still needs one-camera fine-tune data.',
        }
      )

  return subjects, segments


def write_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  if not rows:
    raise ValueError(f'No rows to write for {path}')
  fieldnames = list(rows[0].keys())
  with path.open('w', encoding='utf-8', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def main() -> None:
  args = parse_args()
  oep_root = args.oep_root.resolve()
  output_dir = args.output_dir.resolve()

  if not oep_root.exists():
    raise FileNotFoundError(f'OEP root not found: {oep_root}')

  subjects, segments = build_manifests(oep_root)
  output_dir.mkdir(parents=True, exist_ok=True)

  subject_manifest_path = output_dir / 'oep_subject_manifest.csv'
  segment_manifest_path = output_dir / 'oep_webcam_segments.csv'
  summary_path = output_dir / 'oep_summary.json'

  write_csv(subject_manifest_path, subjects)
  write_csv(segment_manifest_path, segments)

  summary = {
    'oep_root': str(oep_root),
    'subject_count': len(subjects),
    'segment_count': len(segments),
    'acting_subject_count': sum(1 for row in subjects if row['subject_group'] == 'acting_subject'),
    'real_exam_subject_count': sum(1 for row in subjects if row['subject_group'] == 'real_exam_subject'),
    'webcam_segments_csv': str(segment_manifest_path),
    'subject_manifest_csv': str(subject_manifest_path),
  }
  summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

  print('Imported OEP reference dataset')
  print(f'- oep root: {oep_root}')
  print(f'- subjects: {len(subjects)}')
  print(f'- labeled segments: {len(segments)}')
  print(f'- subject manifest: {subject_manifest_path}')
  print(f'- webcam segments: {segment_manifest_path}')
  print(f'- summary: {summary_path}')


if __name__ == '__main__':
  main()
