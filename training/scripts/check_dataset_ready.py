from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OEP_EXTRACTED_ROOT = REPO_ROOT / 'training' / 'data' / 'external' / 'oep_multiview' / 'raw' / 'OEP database'

DATASET_PATHS = {
  'OEP raw dataset drop folder': REPO_ROOT / 'training' / 'data' / 'external' / 'oep_multiview' / 'raw',
  'OEP extracted dataset root': OEP_EXTRACTED_ROOT,
  'Single-camera raw videos': REPO_ROOT / 'training' / 'data' / 'external' / 'single_camera_finetune' / 'raw_videos',
  'Single-camera raw images': REPO_ROOT / 'training' / 'data' / 'external' / 'single_camera_finetune' / 'raw_images',
  'Single-camera annotations': REPO_ROOT / 'training' / 'data' / 'external' / 'single_camera_finetune' / 'annotations',
  'Project object annotation CSV': REPO_ROOT / 'training' / 'data' / 'annotations' / 'object_annotations.csv',
  'Project behavior annotation CSV': REPO_ROOT / 'training' / 'data' / 'annotations' / 'behavior_segments.csv',
}


def describe_dir(path: Path) -> str:
  if not path.exists():
    return 'missing'
  if not path.is_dir():
    return 'not-a-directory'
  files = [item for item in path.iterdir() if item.name != '.gitkeep']
  if path == OEP_EXTRACTED_ROOT:
    subject_count = sum(1 for item in files if item.is_dir() and item.name.startswith('subject'))
    return f'{len(files)} visible items ({subject_count} subject folders)'
  return f'{len(files)} visible items'


def describe_file(path: Path) -> str:
  if not path.exists():
    return 'missing'
  return f'present ({path.stat().st_size} bytes)'


def main() -> None:
  print('Dataset readiness check')
  print(f'- repo: {REPO_ROOT}')
  print()
  for label, path in DATASET_PATHS.items():
    if path.suffix:
      status = describe_file(path)
    else:
      status = describe_dir(path)
    print(f'{label}:')
    print(f'  path: {path}')
    print(f'  status: {status}')


if __name__ == '__main__':
  main()
