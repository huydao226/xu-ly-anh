from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

import cv2


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANNOTATIONS = REPO_ROOT / 'training' / 'data' / 'annotations' / 'object_annotations.csv'
DEFAULT_OUTPUT_DIR = REPO_ROOT / 'training' / 'data' / 'processed' / 'yolo_exam_v1'
VALID_SPLITS = {'train', 'val', 'test'}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description='Convert CSV bounding boxes into a YOLO dataset layout.')
  parser.add_argument('--annotations', type=Path, default=DEFAULT_ANNOTATIONS)
  parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
  parser.add_argument('--classes', type=str, default=None, help='Comma-separated class order. Defaults to sorted labels in CSV.')
  return parser.parse_args()


def resolve_path(raw_path: str) -> Path:
  path = Path(raw_path).expanduser()
  if path.is_absolute():
    return path
  return (REPO_ROOT / path).resolve()


def load_rows(path: Path) -> list[dict[str, str]]:
  with path.open('r', encoding='utf-8', newline='') as handle:
    reader = csv.DictReader(handle)
    required = {'image_path', 'label', 'x_min', 'y_min', 'x_max', 'y_max', 'split'}
    if not reader.fieldnames or not required.issubset(reader.fieldnames):
      raise ValueError(f'CSV must contain columns: {sorted(required)}')
    return [dict(row) for row in reader]


def sanitize_name(source: Path) -> str:
  parts = [part for part in source.parts[-3:] if part not in {'', '.', '..'}]
  stem = '__'.join(parts).replace(' ', '_')
  return stem.replace(source.suffix, '')


def main() -> None:
  args = parse_args()
  annotations_path = args.annotations.resolve()
  output_dir = args.output_dir.resolve()
  rows = load_rows(annotations_path)
  if not rows:
    raise ValueError(f'No annotation rows found in {annotations_path}')

  if args.classes:
    class_names = [item.strip() for item in args.classes.split(',') if item.strip()]
  else:
    class_names = sorted({row['label'].strip() for row in rows if row['label'].strip()})
  class_to_id = {name: idx for idx, name in enumerate(class_names)}

  grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
  for row in rows:
    split = row['split'].strip().lower() or 'train'
    if split not in VALID_SPLITS:
      raise ValueError(f'Unsupported split "{split}" in row: {row}')
    grouped[(row['image_path'], split)].append(row)

  stats = {split: 0 for split in VALID_SPLITS}
  for split in VALID_SPLITS:
    (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

  for image_index, ((raw_image_path, split), image_rows) in enumerate(sorted(grouped.items()), start=1):
    source_path = resolve_path(raw_image_path)
    image = cv2.imread(str(source_path))
    if image is None:
      raise FileNotFoundError(f'Unable to read image: {source_path}')
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
      raise ValueError(f'Invalid image size for {source_path}')

    target_stem = f'{image_index:05d}_{sanitize_name(source_path)}'
    target_image = output_dir / 'images' / split / f'{target_stem}{source_path.suffix.lower()}'
    target_label = output_dir / 'labels' / split / f'{target_stem}.txt'
    shutil.copy2(source_path, target_image)

    lines: list[str] = []
    for row in image_rows:
      label = row['label'].strip()
      if label not in class_to_id:
        raise ValueError(f'Label "{label}" is missing from the class map')
      x_min = max(0.0, float(row['x_min']))
      y_min = max(0.0, float(row['y_min']))
      x_max = min(float(width), float(row['x_max']))
      y_max = min(float(height), float(row['y_max']))
      bbox_width = max(x_max - x_min, 1.0)
      bbox_height = max(y_max - y_min, 1.0)
      x_center = x_min + bbox_width / 2.0
      y_center = y_min + bbox_height / 2.0
      line = ' '.join(
        [
          str(class_to_id[label]),
          f'{x_center / width:.6f}',
          f'{y_center / height:.6f}',
          f'{bbox_width / width:.6f}',
          f'{bbox_height / height:.6f}',
        ]
      )
      lines.append(line)

    target_label.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    stats[split] += 1

  data_yaml = output_dir / 'data.yaml'
  data_yaml.write_text(
    '\n'.join(
      [
        f'path: {output_dir}',
        'train: images/train',
        'val: images/val',
        'test: images/test',
        f'nc: {len(class_names)}',
        f'names: {class_names}',
      ]
    )
    + '\n',
    encoding='utf-8',
  )

  print('Prepared YOLO dataset')
  print(f'- annotations: {annotations_path}')
  print(f'- output: {output_dir}')
  print(f'- classes: {class_names}')
  print(f'- image counts: {stats}')


if __name__ == '__main__':
  main()
