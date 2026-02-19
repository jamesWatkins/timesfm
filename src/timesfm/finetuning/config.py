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

"""Configuration objects for TimesFM finetuning."""

from __future__ import annotations

import dataclasses
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class WindowingConfig:
  """Controls sliding-window sample generation for finetuning datasets."""

  context_len: int
  horizon_len: int
  stride: int = 1
  min_context: int = 1
  drop_last: bool = True


@dataclasses.dataclass(frozen=True)
class FinetuningConfig:
  """Configuration for torch finetuning."""

  batch_size: int = 16
  num_epochs: int = 5
  learning_rate: float = 1e-4
  weight_decay: float = 1e-2
  grad_clip_norm: float = 1.0
  mixed_precision: bool = False
  seed: int = 0
  num_workers: int = 0
  use_quantile_loss: bool = False
  quantile_loss_weight: float = 1.0
  log_every_n_steps: int = 50
  val_every_n_epochs: int = 1
  save_every_n_epochs: int = 1
  output_dir: str = "outputs/timesfm_finetune"
  device: str = "cuda"
  use_mlflow: bool = False
  mlflow_tracking_uri: str = "file:./mlruns"
  mlflow_experiment_name: str = "timesfm-finetuning"
  mlflow_run_name: str | None = None
  mlflow_tags: dict[str, str] | None = None
  mlflow_log_every_n_steps: int = 0
  mlflow_artifact_subdir: str = "artifacts"

  @property
  def output_path(self) -> Path:
    return Path(self.output_dir)


@dataclasses.dataclass(frozen=True)
class FinetuningResult:
  """Final training metadata."""

  train_losses: tuple[float, ...]
  val_losses: tuple[float, ...]
  best_val_loss: float
  best_epoch: int
  checkpoint_path: str | None
