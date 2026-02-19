# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Torch finetuning trainer for TimesFM 2.5."""

from __future__ import annotations

import dataclasses
import importlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .checkpointing import load_training_checkpoint, save_training_checkpoint
from .config import FinetuningConfig, FinetuningResult
from .datasets import create_collate_fn
from .model_adapter import BaseTrainAdapter, FinetuneForwardOutput


def _resolve_device(config: FinetuningConfig) -> torch.device:
  if config.device == "cuda" and torch.cuda.is_available():
    return torch.device("cuda")
  return torch.device("cpu")


class TimesFMFinetuner:
  """Trainer for finetuning torch-based TimesFM adapters."""

  def __init__(self, adapter: BaseTrainAdapter, config: FinetuningConfig):
    self.adapter = adapter
    self.config = config
    self.device = _resolve_device(config)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed_all(config.seed)

    self.adapter.to(self.device)

    self.optimizer = torch.optim.AdamW(
      self.adapter.parameters(),
      lr=config.learning_rate,
      weight_decay=config.weight_decay,
    )
    self.scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision)
    self.output_dir = Path(config.output_dir)
    self.output_dir.mkdir(parents=True, exist_ok=True)
    self._global_train_step = 0

    self._mlflow = None
    self._mlflow_enabled = False
    self._init_mlflow()

  def _init_mlflow(self) -> None:
    if not self.config.use_mlflow:
      return
    try:
      self._mlflow = importlib.import_module("mlflow")
    except ImportError as exc:
      raise RuntimeError(
        "MLflow is not installed. Install with `pip install -e .[finetune]`."
      ) from exc

    self._mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
    self._mlflow.set_experiment(self.config.mlflow_experiment_name)
    self._mlflow.start_run(
      run_name=self.config.mlflow_run_name,
      tags=self.config.mlflow_tags or {},
    )
    self._mlflow_enabled = True

  def _to_mlflow_value(self, value: object) -> str | float | int | bool:
    if isinstance(value, (str, float, int, bool)):
      return value
    return str(value)

  def _mlflow_log_params(self, params: dict[str, object]) -> None:
    if not self._mlflow_enabled or self._mlflow is None:
      return
    try:
      safe = {k: self._to_mlflow_value(v) for k, v in params.items()}
      self._mlflow.log_params(safe)
    except Exception as exc:  # pragma: no cover - defensive logging
      print(f"Warning: MLflow log_params failed: {exc}")

  def _mlflow_log_metrics(self, metrics: dict[str, float], step: int | None) -> None:
    if not self._mlflow_enabled or self._mlflow is None:
      return
    try:
      if step is None:
        self._mlflow.log_metrics(metrics)
      else:
        self._mlflow.log_metrics(metrics, step=step)
    except Exception as exc:  # pragma: no cover - defensive logging
      print(f"Warning: MLflow log_metrics failed: {exc}")

  def log_artifact(self, path: str | Path) -> None:
    if not self._mlflow_enabled or self._mlflow is None:
      return
    p = Path(path)
    if not p.exists():
      return
    try:
      self._mlflow.log_artifact(
        str(p), artifact_path=self.config.mlflow_artifact_subdir
      )
    except Exception as exc:  # pragma: no cover - defensive logging
      print(f"Warning: MLflow log_artifact failed: {exc}")

  def _mlflow_end_run(self, status: str) -> None:
    if not self._mlflow_enabled or self._mlflow is None:
      return
    try:
      self._mlflow.end_run(status=status)
    except Exception as exc:  # pragma: no cover - defensive logging
      print(f"Warning: MLflow end_run failed: {exc}")
    finally:
      self._mlflow_enabled = False

  def _mse_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((prediction - target) ** 2)

  def _quantile_loss(
    self,
    prediction: torch.Tensor,
    target: torch.Tensor,
    quantile: float,
  ) -> torch.Tensor:
    error = target - prediction
    return torch.mean(torch.maximum(quantile * error, (quantile - 1.0) * error))

  def _compute_loss(self, output: FinetuneForwardOutput, target: torch.Tensor) -> torch.Tensor:
    loss = self._mse_loss(output.point_forecast, target)
    if self.config.use_quantile_loss:
      quantile_channels = list(range(1, len(self.adapter.quantiles) + 1))
      for q, channel in zip(self.adapter.quantiles, quantile_channels):
        loss = loss + self.config.quantile_loss_weight * self._quantile_loss(
          output.full_forecast[..., channel],
          target,
          q,
        )
    return loss

  def _create_dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
    return DataLoader(
      dataset,
      batch_size=self.config.batch_size,
      shuffle=shuffle,
      num_workers=self.config.num_workers,
      collate_fn=create_collate_fn(self.adapter.patch_len),
    )

  def _run_epoch(self, dataloader: DataLoader, train: bool) -> float:
    self.adapter.train(mode=train)
    losses = []

    for step, batch in enumerate(dataloader):
      context = batch["context"].to(self.device)
      context_mask = batch["context_mask"].to(self.device)
      target = batch["target"].to(self.device)

      with torch.set_grad_enabled(train):
        with torch.amp.autocast(
          "cuda",
          enabled=(self.config.mixed_precision and self.device.type == "cuda"),
        ):
          output = self.adapter.forward_train(
            context=context,
            context_mask=context_mask,
            horizon=target.shape[1],
          )
          loss = self._compute_loss(output, target)

      if train:
        self._global_train_step += 1
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        if self.config.grad_clip_norm > 0:
          self.scaler.unscale_(self.optimizer)
          torch.nn.utils.clip_grad_norm_(
            self.adapter.parameters(), self.config.grad_clip_norm
          )
        self.scaler.step(self.optimizer)
        self.scaler.update()

      losses.append(loss.detach().cpu().item())
      if (
        train
        and self.config.mlflow_log_every_n_steps > 0
        and self._global_train_step % self.config.mlflow_log_every_n_steps == 0
      ):
        self._mlflow_log_metrics(
          {"train_step_loss": float(losses[-1])},
          step=self._global_train_step,
        )
      if train and self.config.log_every_n_steps > 0 and (step + 1) % self.config.log_every_n_steps == 0:
        print(f"step={step + 1} train_loss={losses[-1]:.6f}")

    return float(np.mean(losses)) if losses else float("nan")

  def save_checkpoint(self, path: str | Path, epoch: int, best_val: float) -> None:
    save_training_checkpoint(
      path,
      {
        "model": self.adapter.state_dict(),
        "optimizer": self.optimizer.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
        "config": dataclasses.asdict(self.config),
      },
    )

  def load_checkpoint(self, path: str | Path) -> dict[str, object]:
    state = load_training_checkpoint(path, map_location=str(self.device))
    self.adapter.load_state_dict(state["model"])
    self.optimizer.load_state_dict(state["optimizer"])
    return state

  def finetune(self, train_dataset: Dataset, val_dataset: Dataset) -> FinetuningResult:
    train_loader = self._create_dataloader(train_dataset, shuffle=True)
    val_loader = self._create_dataloader(val_dataset, shuffle=False)

    train_losses = []
    val_losses = []
    best_val = float("inf")
    best_epoch = -1
    best_path: Path | None = None

    self._mlflow_log_params(dataclasses.asdict(self.config))
    self._mlflow_log_params(
      {
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "adapter_patch_len": self.adapter.patch_len,
        "adapter_max_horizon": getattr(self.adapter, "max_horizon", -1),
        "device_used": str(self.device),
      }
    )

    run_status = "FINISHED"
    try:
      for epoch in range(1, self.config.num_epochs + 1):
        train_loss = self._run_epoch(train_loader, train=True)
        train_losses.append(train_loss)

        if epoch % self.config.val_every_n_epochs == 0:
          val_loss = self._run_epoch(val_loader, train=False)
        else:
          val_loss = float("nan")
        val_losses.append(val_loss)

        if np.isfinite(val_loss) and val_loss < best_val:
          best_val = float(val_loss)
          best_epoch = epoch
          best_path = self.output_dir / "best.pt"
          self.save_checkpoint(best_path, epoch=epoch, best_val=best_val)

        if (
          self.config.save_every_n_epochs > 0
          and epoch % self.config.save_every_n_epochs == 0
        ):
          self.save_checkpoint(
            self.output_dir / f"epoch_{epoch}.pt", epoch=epoch, best_val=best_val
          )

        current_lr = float(self.optimizer.param_groups[0]["lr"])
        self._mlflow_log_metrics(
          {
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "best_val_loss": float(best_val),
            "learning_rate": current_lr,
          },
          step=epoch,
        )

        print(
          f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} best_val={best_val:.6f}"
        )

      checkpoint_path = str(best_path) if best_path is not None else None
      result = FinetuningResult(
        train_losses=tuple(train_losses),
        val_losses=tuple(val_losses),
        best_val_loss=best_val,
        best_epoch=best_epoch,
        checkpoint_path=checkpoint_path,
      )
      summary_path = self.output_dir / "run_summary.json"
      summary_path.write_text(
        json.dumps(
          {
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "checkpoint_path": checkpoint_path,
          },
          indent=2,
        )
      )
      self.log_artifact(summary_path)
      if checkpoint_path:
        self._mlflow_log_params({"best_checkpoint_path": checkpoint_path})
      return result
    except Exception:
      run_status = "FAILED"
      raise
    finally:
      self._mlflow_end_run(run_status)
