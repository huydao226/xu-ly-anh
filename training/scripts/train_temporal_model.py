from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / 'training' / 'data' / 'processed' / 'temporal_sequences.jsonl'
DEFAULT_OUTPUT_DIR = REPO_ROOT / 'training' / 'models' / 'temporal_lstm_v1'


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description='Train a baseline LSTM on temporal anti-cheat features.')
  parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
  parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
  parser.add_argument('--epochs', type=int, default=24)
  parser.add_argument('--batch-size', type=int, default=16)
  parser.add_argument('--hidden-size', type=int, default=64)
  parser.add_argument('--learning-rate', type=float, default=7e-4)
  parser.add_argument('--weight-decay', type=float, default=1e-4)
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--max-frames', type=int, default=90)
  parser.add_argument('--balanced-sampler', action='store_true')
  parser.add_argument('--min-threshold', type=float, default=0.35)
  parser.add_argument('--max-threshold', type=float, default=0.75)
  parser.add_argument('--threshold-step', type=float, default=0.01)
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


def compute_feature_stats(samples: list[dict], max_frames: int) -> tuple[list[float], list[float]]:
  feature_rows: list[torch.Tensor] = []
  for sample in samples:
    features = torch.tensor(sample['features'], dtype=torch.float32)[:max_frames]
    if features.numel() > 0:
      feature_rows.append(features)
  if not feature_rows:
    raise ValueError('Unable to compute feature statistics from empty train split')
  stacked = torch.cat(feature_rows, dim=0)
  mean = stacked.mean(dim=0)
  std = stacked.std(dim=0, unbiased=False)
  std = torch.where(std < 1e-6, torch.ones_like(std), std)
  return mean.tolist(), std.tolist()


def make_class_weights(samples: list[dict], num_classes: int) -> list[float]:
  counts = Counter(int(sample['label_id']) for sample in samples)
  total = sum(counts.values())
  weights: list[float] = []
  for class_id in range(num_classes):
    count = counts.get(class_id, 1)
    weights.append((total / float(num_classes * count)) ** 0.5)
  return weights


def make_weighted_sampler(samples: list[dict], class_weights: list[float]) -> WeightedRandomSampler:
  sample_weights = [class_weights[int(sample['label_id'])] for sample in samples]
  return WeightedRandomSampler(
    weights=torch.tensor(sample_weights, dtype=torch.double),
    num_samples=len(sample_weights),
    replacement=True,
  )


@dataclass
class BatchItem:
  features: torch.Tensor
  label: torch.Tensor
  length: torch.Tensor


class TemporalSequenceDataset(Dataset[BatchItem]):
  def __init__(self, samples: list[dict], max_frames: int, feature_mean: list[float], feature_std: list[float]) -> None:
    self.samples = samples
    self.max_frames = max_frames
    self.feature_mean = torch.tensor(feature_mean, dtype=torch.float32)
    self.feature_std = torch.tensor(feature_std, dtype=torch.float32)

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, index: int) -> BatchItem:
    sample = self.samples[index]
    features = torch.tensor(sample['features'], dtype=torch.float32)
    clipped = features[: self.max_frames]
    if clipped.numel() > 0:
      clipped = (clipped - self.feature_mean) / self.feature_std
    length = min(clipped.shape[0], self.max_frames)
    feature_dim = clipped.shape[1]
    padded = torch.zeros((self.max_frames, feature_dim), dtype=torch.float32)
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
      nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()
    total_loss += float(loss.item()) * labels.size(0)
    total_correct += int((logits.argmax(dim=1) == labels).sum().item())
    total_count += int(labels.size(0))
  return {
    'loss': total_loss / max(total_count, 1),
    'accuracy': total_correct / max(total_count, 1),
  }


def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int], list[list[float]]]:
  model.eval()
  targets: list[int] = []
  predictions: list[int] = []
  probabilities: list[list[float]] = []
  with torch.no_grad():
    for features, labels, lengths in loader:
      logits = model(features.to(device), lengths.to(device))
      probs = torch.softmax(logits, dim=1).cpu()
      preds = probs.argmax(dim=1)
      targets.extend(int(item) for item in labels.tolist())
      predictions.extend(int(item) for item in preds.tolist())
      probabilities.extend([row.tolist() for row in probs])
  return targets, predictions, probabilities


def accuracy_score(targets: list[int], predictions: list[int]) -> float:
  if not targets:
    return 0.0
  correct = sum(1 for target, pred in zip(targets, predictions) if target == pred)
  return correct / float(len(targets))


def macro_f1_score(targets: list[int], predictions: list[int], num_classes: int) -> float:
  if not targets:
    return 0.0
  scores: list[float] = []
  for class_id in range(num_classes):
    tp = sum(1 for target, pred in zip(targets, predictions) if target == class_id and pred == class_id)
    fp = sum(1 for target, pred in zip(targets, predictions) if target != class_id and pred == class_id)
    fn = sum(1 for target, pred in zip(targets, predictions) if target == class_id and pred != class_id)
    precision = tp / float(tp + fp) if tp + fp > 0 else 0.0
    recall = tp / float(tp + fn) if tp + fn > 0 else 0.0
    f1 = 0.0 if precision + recall == 0 else (2.0 * precision * recall) / (precision + recall)
    scores.append(f1)
  return sum(scores) / float(len(scores))


def apply_non_normal_threshold(probabilities: list[list[float]], normal_index: int, threshold: float) -> list[int]:
  adjusted: list[int] = []
  for row in probabilities:
    top_index = max(range(len(row)), key=row.__getitem__)
    top_confidence = row[top_index]
    if top_index != normal_index and top_confidence < threshold:
      adjusted.append(normal_index)
    else:
      adjusted.append(top_index)
  return adjusted


def tune_non_normal_threshold(
  *,
  targets: list[int],
  probabilities: list[list[float]],
  normal_index: int,
  num_classes: int,
  min_threshold: float,
  max_threshold: float,
  threshold_step: float,
) -> tuple[float, float, float]:
  best_threshold = min_threshold
  best_macro_f1 = -1.0
  best_accuracy = -1.0
  threshold = min_threshold
  while threshold <= max_threshold + 1e-9:
    predictions = apply_non_normal_threshold(probabilities, normal_index, threshold)
    macro_f1 = macro_f1_score(targets, predictions, num_classes)
    accuracy = accuracy_score(targets, predictions)
    if macro_f1 > best_macro_f1 or (abs(macro_f1 - best_macro_f1) < 1e-9 and accuracy > best_accuracy):
      best_threshold = round(threshold, 4)
      best_macro_f1 = macro_f1
      best_accuracy = accuracy
    threshold += threshold_step
  return best_threshold, best_macro_f1, best_accuracy


def prediction_counts(predictions: list[int], label_names: list[str]) -> dict[str, int]:
  counts = Counter(predictions)
  return {label_names[index]: counts.get(index, 0) for index in range(len(label_names))}


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
  label_names = [label for label, _ in sorted(label_map.items(), key=lambda item: int(item[1]))]
  num_classes = len(label_map)
  normal_index = int(label_map.get('normal', 0))

  feature_mean, feature_std = compute_feature_stats(train_samples, args.max_frames)
  class_weights = make_class_weights(train_samples, num_classes)

  train_dataset = TemporalSequenceDataset(train_samples, args.max_frames, feature_mean, feature_std)
  val_dataset = TemporalSequenceDataset(val_samples, args.max_frames, feature_mean, feature_std)
  test_dataset = TemporalSequenceDataset(test_samples, args.max_frames, feature_mean, feature_std)

  train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    sampler=make_weighted_sampler(train_samples, class_weights) if args.balanced_sampler else None,
    shuffle=not args.balanced_sampler,
    collate_fn=collate_batch,
  )
  val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=collate_batch,
  )
  test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=collate_batch,
  )

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = LSTMClassifier(feature_dim, args.hidden_size, num_classes).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
  criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))

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

  val_targets, val_predictions, val_probabilities = collect_predictions(model, val_loader, device)
  tuned_threshold, tuned_val_macro_f1, tuned_val_accuracy = tune_non_normal_threshold(
    targets=val_targets,
    probabilities=val_probabilities,
    normal_index=normal_index,
    num_classes=num_classes,
    min_threshold=args.min_threshold,
    max_threshold=args.max_threshold,
    threshold_step=args.threshold_step,
  )
  val_predictions_thresholded = apply_non_normal_threshold(val_probabilities, normal_index, tuned_threshold)

  test_metrics = run_epoch(model=model, loader=test_loader, optimizer=None, criterion=criterion, device=device)
  test_targets, test_predictions, test_probabilities = collect_predictions(model, test_loader, device)
  test_predictions_thresholded = apply_non_normal_threshold(test_probabilities, normal_index, tuned_threshold)

  metrics = {
    'device': str(device),
    'dataset': str(args.dataset.resolve()),
    'num_samples': len(samples),
    'num_classes': num_classes,
    'feature_dim': feature_dim,
    'max_frames': args.max_frames,
    'epochs': args.epochs,
    'hidden_size': args.hidden_size,
    'learning_rate': args.learning_rate,
    'weight_decay': args.weight_decay,
    'feature_mean': feature_mean,
    'feature_std': feature_std,
    'class_weights': class_weights,
    'best_val_accuracy': best_val_accuracy,
    'val_accuracy_thresholded': tuned_val_accuracy,
    'val_macro_f1_thresholded': tuned_val_macro_f1,
    'non_normal_threshold': tuned_threshold,
    'test_loss': test_metrics['loss'],
    'test_accuracy': test_metrics['accuracy'],
    'test_accuracy_thresholded': accuracy_score(test_targets, test_predictions_thresholded),
    'test_macro_f1_thresholded': macro_f1_score(test_targets, test_predictions_thresholded, num_classes),
    'val_prediction_counts_raw': prediction_counts(val_predictions, label_names),
    'val_prediction_counts_thresholded': prediction_counts(val_predictions_thresholded, label_names),
    'test_prediction_counts_raw': prediction_counts(test_predictions, label_names),
    'test_prediction_counts_thresholded': prediction_counts(test_predictions_thresholded, label_names),
    'history': history,
  }

  (args.output_dir / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
  (args.output_dir / 'label_map.json').write_text(json.dumps(label_map, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
  print('Saved model and metrics')
  print(f'- output_dir: {args.output_dir}')
  print(f'- best_val_accuracy: {best_val_accuracy:.4f}')
  print(f'- thresholded_val_accuracy: {tuned_val_accuracy:.4f}')
  print(f'- non_normal_threshold: {tuned_threshold:.2f}')
  print(f'- test_accuracy: {test_metrics["accuracy"]:.4f}')
  print(f'- thresholded_test_accuracy: {metrics["test_accuracy_thresholded"]:.4f}')


if __name__ == '__main__':
  main()
