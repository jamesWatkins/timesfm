from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from timesfm.finetuning import BaseTrainAdapter, FinetuneForwardOutput, FinetuningConfig, TimesFMFinetuner


class _ToyDataset(Dataset):
  def __init__(self, n: int = 64, context_len: int = 8, horizon: int = 2):
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


def test_finetuner_runs_and_improves(tmp_path: Path) -> None:
  train_ds = _ToyDataset(n=128)
  val_ds = _ToyDataset(n=32)

  cfg = FinetuningConfig(
    batch_size=16,
    num_epochs=3,
    learning_rate=0.1,
    weight_decay=0.0,
    output_dir=str(tmp_path),
    log_every_n_steps=0,
  )
  adapter = _LinearAdapter()
  trainer = TimesFMFinetuner(adapter, cfg)

  result = trainer.finetune(train_ds, val_ds)

  assert len(result.train_losses) == 3
  assert result.train_losses[-1] <= result.train_losses[0]
  assert result.checkpoint_path is not None
  assert Path(result.checkpoint_path).exists()


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
  train_ds = _ToyDataset(n=32)
  val_ds = _ToyDataset(n=16)
  cfg = FinetuningConfig(
    batch_size=8,
    num_epochs=1,
    learning_rate=0.1,
    weight_decay=0.0,
    output_dir=str(tmp_path),
    log_every_n_steps=0,
  )

  adapter = _LinearAdapter()
  trainer = TimesFMFinetuner(adapter, cfg)
  trainer.finetune(train_ds, val_ds)

  ckpt = tmp_path / "manual.pt"
  trainer.save_checkpoint(ckpt, epoch=1, best_val=1.23)
  old_weight = adapter.weight.detach().clone()

  adapter.weight.data.zero_()
  state = trainer.load_checkpoint(ckpt)

  assert "epoch" in state
  assert torch.allclose(adapter.weight.detach(), old_weight)
