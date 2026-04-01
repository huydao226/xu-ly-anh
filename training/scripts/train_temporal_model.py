from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / 'training' / 'data' / 'processed' / 'temporal_sequences.jsonl'
DEFAULT_OUTPUT_DIR = REPO_ROOT / 'training' / 'models' / 'temporal_lstm_v1'


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description='Train a baseline LSTM on temporal anti-cheat features.')
  parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
  parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
  parser.add_argument('--epochs', type=int, default=20)
  parser.add_argument('--batch-size', type=int, default=8)
  parser.add_argument('--hidden-size', type=int, default=64)
  parser.add_argument('--learning-rate', type=float, default=1e-3)
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--max-frames', type=int, default=90)
  return parser.parse_args()


def set_seed(seed: int) -> None:
  random.seed(seed)
  torch.manual_seed(seed)


def load_samples(path: Path) -> list[dict]:
  samples: list[dict] = []
  with path.open('r', encoding='utf-8') as handle:
    for line in handle:
      if line.strip():
        samples.append(json.loads(line))
  if not samples:
    raise ValueError(f'No temporal samples found in {path}')
  return samples


def split_samples(samples: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
  train = [sample for sample in samples if sample.get('split') == 'train']
  val = [sample for sample in samples if sample.get('split') == 'val']
  test = [sample for sample in samples if sample.get('split') == 'test']
  if not train:
    raise ValueError('Temporal dataset needs at least one train sample')
  if not val:
    val = train
  if not test:
    test = val
  return train, val, test


@dataclass
class BatchItem:
  features: torch.Tensor
  label: torch.Tensor
  length: torch.Tensor


class TemporalSequenceDataset(Dataset[BatchItem]):
  def __init__(self, samples: list[dict], max_frames: int) -> None:
    self.samples = samples
    self.max_frames = max_frames

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, index: int) -> BatchItem:
    sample = self.samples[index]
    features = torch.tensor(sample['features'], dtype=torch.float32)
    length = min(features.shape[0], self.max_frames)
    feature_dim = features.shape[1]
    padded = torch.zeros((self.max_frames, feature_dim), dtype=torch.float32)
    clipped = features[: self.max_frames]
    padded[: clipped.shape[0]] = clipped
    label = torch.tensor(int(sample['label_id']), dtype=torch.long)
    return BatchItem(features=padded, label=label, length=torch.tensor(length, dtype=torch.long))


def collate_batch(items: list[BatchItem]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  features = torch.stack([item.features for item in items], dim=0)
  labels = torch.stack([item.label for item in items], dim=0)
  lengths = torch.stack([item.length for item in items], dim=0)
  return features, labels, lengths


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
    packed = nn.utils.rnn.pack_padded_sequence(
      features,
      lengths.cpu(),
      batch_first=True,
      enforce_sorted=False,
    )
    _, (hidden, _) = self.lstm(packed)
    return self.head(hidden[-1])


def run_epoch(
  *,
  model: nn.Module,
  loader: DataLoader,
  optimizer: torch.optim.Optimizer | None,
  criterion: nn.Module,
  device: torch.device,
) -> dict[str, float]:
  is_training = optimizer is not None
  model.train(is_training)
  total_loss = 0.0
  total_correct = 0
  total_count = 0
  for features, labels, lengths in loader:
    features = features.to(device)
    labels = labels.to(device)
    lengths = lengths.to(device)
    logits = model(features, lengths)
    loss = criterion(logits, labels)
    if is_training:
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    total_loss += float(loss.item()) * labels.size(0)
    total_correct += int((logits.argmax(dim=1) == labels).sum().item())
    total_count += int(labels.size(0))
  return {
    'loss': total_loss / max(total_count, 1),
    'accuracy': total_correct / max(total_count, 1),
  }


def main() -> None:
  args = parse_args()
  set_seed(args.seed)

  samples = load_samples(args.dataset.resolve())
  train_samples, val_samples, test_samples = split_samples(samples)
  feature_dim = len(train_samples[0]['feature_names'])
  label_map = {
    sample['label']: int(sample['label_id'])
    for sample in sorted(samples, key=lambda item: int(item['label_id']))
  }
  num_classes = len(label_map)

  train_loader = DataLoader(
    TemporalSequenceDataset(train_samples, args.max_frames),
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=collate_batch,
  )
  val_loader = DataLoader(
    TemporalSequenceDataset(val_samples, args.max_frames),
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=collate_batch,
  )
  test_loader = DataLoader(
    TemporalSequenceDataset(test_samples, args.max_frames),
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=collate_batch,
  )

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = LSTMClassifier(feature_dim, args.hidden_size, num_classes).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
  criterion = nn.CrossEntropyLoss()

  best_val_accuracy = -1.0
  args.output_dir.mkdir(parents=True, exist_ok=True)
  history: list[dict[str, float]] = []

  for epoch in range(1, args.epochs + 1):
    train_metrics = run_epoch(model=model, loader=train_loader, optimizer=optimizer, criterion=criterion, device=device)
    val_metrics = run_epoch(model=model, loader=val_loader, optimizer=None, criterion=criterion, device=device)
    epoch_record = {
      'epoch': epoch,
      'train_loss': train_metrics['loss'],
      'train_accuracy': train_metrics['accuracy'],
      'val_loss': val_metrics['loss'],
      'val_accuracy': val_metrics['accuracy'],
    }
    history.append(epoch_record)
    print(
      f"epoch={epoch} "
      f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
      f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}"
    )
    if val_metrics['accuracy'] >= best_val_accuracy:
      best_val_accuracy = val_metrics['accuracy']
      torch.save(model.state_dict(), args.output_dir / 'best_model.pt')

  model.load_state_dict(torch.load(args.output_dir / 'best_model.pt', map_location=device))
  test_metrics = run_epoch(model=model, loader=test_loader, optimizer=None, criterion=criterion, device=device)

  metrics = {
    'device': str(device),
    'dataset': str(args.dataset.resolve()),
    'num_samples': len(samples),
    'num_classes': num_classes,
    'feature_dim': feature_dim,
    'max_frames': args.max_frames,
    'epochs': args.epochs,
    'best_val_accuracy': best_val_accuracy,
    'test_loss': test_metrics['loss'],
    'test_accuracy': test_metrics['accuracy'],
    'history': history,
  }

  (args.output_dir / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
  (args.output_dir / 'label_map.json').write_text(json.dumps(label_map, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
  print('Saved model and metrics')
  print(f'- output_dir: {args.output_dir}')
  print(f'- best_val_accuracy: {best_val_accuracy:.4f}')
  print(f'- test_accuracy: {test_metrics["accuracy"]:.4f}')


if __name__ == '__main__':
  main()
