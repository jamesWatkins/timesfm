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

"""Checkpoint helpers for finetuning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_training_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
  """Saves trainer state dictionary."""
  checkpoint_path = Path(path)
  checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(state, checkpoint_path)


def load_training_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
  """Loads trainer state dictionary."""
  return torch.load(Path(path), map_location=map_location)
