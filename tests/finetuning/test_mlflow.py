from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from timesfm.finetuning import (
  BaseTrainAdapter,
  FinetuneForwardOutput,
  FinetuningConfig,
  TimesFMFinetuner,
)


class _ToyDataset(Dataset):
  def __init__(self, n: int = 16, context_len: int = 8, horizon: int = 2):
    self.samples = []
    for i in range(n):
      base = torch.arange(context_len, dtype=torch.float32) + i
      target = 2.0 * base[-horizon:]
      self.samples.append({"context": base, "target": target})

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, idx: int):
    return self.samples[idx]


class _LinearAdapter(BaseTrainAdapter):
  def __init__(self):
    super().__init__()
    self.patch_len = 4
    self.point_channel_index = 0
    self.quantiles = ()
    self.weight = nn.Parameter(torch.tensor(0.0))

  def forward_train(self, context: torch.Tensor, context_mask: torch.Tensor, horizon: int):
    del context_mask
    pred = self.weight * context[:, -horizon:]
    full = pred[..., None]
    return FinetuneForwardOutput(point_forecast=pred, full_forecast=full)


class _FakeMlflow:
  def __init__(self):
    self.tracking_uri = None
    self.experiment_name = None
    self.started = False
    self.ended = False
    self.end_status = None
    self.params = []
    self.metrics = []
    self.artifacts = []

  def set_tracking_uri(self, uri):
    self.tracking_uri = uri

  def set_experiment(self, name):
    self.experiment_name = name

  def start_run(self, run_name=None, tags=None):
    self.started = True
    self.run_name = run_name
    self.tags = tags

  def log_params(self, params):
    self.params.append(params)

  def log_metrics(self, metrics, step=None):
    self.metrics.append((metrics, step))

  def log_artifact(self, path, artifact_path=None):
    self.artifacts.append((path, artifact_path))

  def end_run(self, status="FINISHED"):
    self.ended = True
    self.end_status = status


def test_trainer_logs_to_mlflow(monkeypatch, tmp_path: Path) -> None:
  fake = _FakeMlflow()
  monkeypatch.setitem(sys.modules, "mlflow", fake)

  train_ds = _ToyDataset(n=32)
  val_ds = _ToyDataset(n=16)
  cfg = FinetuningConfig(
    batch_size=8,
    num_epochs=1,
    learning_rate=0.1,
    weight_decay=0.0,
    output_dir=str(tmp_path),
    log_every_n_steps=0,
    use_mlflow=True,
    mlflow_tracking_uri="file:/tmp/mlruns-test",
    mlflow_experiment_name="timesfm-test",
    mlflow_run_name="unit-test-run",
  )

  trainer = TimesFMFinetuner(_LinearAdapter(), cfg)
  result = trainer.finetune(train_ds, val_ds)

  assert result.best_epoch == 1
  assert fake.started is True
  assert fake.ended is True
  assert fake.end_status == "FINISHED"
  assert fake.tracking_uri == "file:/tmp/mlruns-test"
  assert fake.experiment_name == "timesfm-test"
  assert fake.params
  assert fake.metrics
  assert any("train_loss" in m[0] for m in fake.metrics)
  assert any("run_summary.json" in a[0] for a in fake.artifacts)
