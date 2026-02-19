# TimesFM v1 vs `src/timesfm` (2.5) Architecture and API Change Report

## Executive Summary

The latest code under `src/timesfm` is a substantial API and architecture reset relative to `v1/`.

Key outcomes:

1. The **public entrypoint changed** from `TimesFm` + `TimesFmHparams` + `TimesFmCheckpoint` to explicit model classes (`TimesFM_2p5_200M_torch` / `TimesFM_2p5_200M_flax`) plus `ForecastConfig`.
2. The new API is **compile-first inference** (`compile(...)` then `forecast(...)`) and drops v1 conveniences like `forecast_on_df` and direct frequency inputs.
3. v1 included **finetuning infrastructure** (`v1/src/finetuning`) and a TensorFlow-style dataset loader (`v1/src/timesfm/data_loader.py`); latest `src/timesfm` does **not** include a training/finetuning package.
4. For finetuning implementation, the biggest gap is not transformer internals; it is the **missing training-facing data contract and trainer scaffolding** around the new model modules.

---

## Scope and Method

Compared:

- Legacy: `v1/src/timesfm`, `v1/src/finetuning`, `v1/src/adapter`, `v1/notebooks`
- Latest: `src/timesfm` (`timesfm_2p5`, `torch`, `flax`, `configs`, `utils/xreg_lib`)

Focus:

- Public APIs and usage model
- Training-critical internal interfaces
- Data loaders / dataset contracts / covariate paths
- Packaging and test coverage implications

---

## 1) High-Level Architecture Comparison

## v1 architecture (`v1/src/timesfm`)

- A base inference abstraction (`TimesFmBase`) with shared APIs in `v1/src/timesfm/timesfm_base.py`.
- Two backends behind the same façade:
  - JAX/PAX implementation (`v1/src/timesfm/timesfm_jax.py`)
  - PyTorch implementation (`v1/src/timesfm/timesfm_torch.py`)
- Decoder implementation in:
  - `v1/src/timesfm/patched_decoder.py` (PAX)
  - `v1/src/timesfm/pytorch_patched_decoder.py` (torch)
- Additional top-level functionality bundled in package:
  - `data_loader.py` (TF dataset generator)
  - `forecast_on_df(...)`
  - `forecast_with_covariates(...)`
  - `finetuning/` package (torch trainer, DDP support, example)
  - `adapter/` package (LoRA/DoRA style adapter utilities for PAX stack)

## Latest architecture (`src/timesfm`)

- Framework-agnostic layer/config primitives:
  - `src/timesfm/configs.py`
  - shared module families under `src/timesfm/torch/*` and `src/timesfm/flax/*`
- Model family moved into `timesfm_2p5/`:
  - abstract base: `src/timesfm/timesfm_2p5/timesfm_2p5_base.py`
  - backend wrappers: `timesfm_2p5_torch.py`, `timesfm_2p5_flax.py`
- Public API exports only model classes + `ForecastConfig` in `src/timesfm/__init__.py`.
- No in-repo finetuning/training package in latest tree.

Implication: latest code cleanly separates inference model code from training scaffolding, but this removes ready-made finetuning entrypoints that existed in `v1`.

---

## 2) Public API Surface Changes

## 2.1 Entrypoint and object model

| Area | v1 | latest (`src/timesfm`) | Change type |
|---|---|---|---|
| Main class | `timesfm.TimesFm` (dynamic jax/torch import) in `v1/src/timesfm/__init__.py` | explicit `TimesFM_2p5_200M_torch` / `TimesFM_2p5_200M_flax` in `src/timesfm/__init__.py` | breaking |
| Hyperparams | `TimesFmHparams` | model definition internal + `ForecastConfig` for inference flags | breaking |
| Checkpoint config | `TimesFmCheckpoint` + `load_from_checkpoint` | `from_pretrained(...)` / `_from_pretrained(...)`, backend-specific loading | breaking |
| Initialization flow | construct `TimesFm(...)` with hparams+checkpoint | construct model class, load pretrained, then compile | breaking |

## 2.2 Forecast call contract

| Area | v1 | latest | Change type |
|---|---|---|---|
| Forecast signature | `forecast(inputs, freq=None, window_size=None, forecast_context_len=None, return_forecast_on_context=False, normalize=False)` | `forecast(horizon, inputs)` (on compiled model) | breaking |
| Frequency input | required/optional categorical `freq` ([0,1,2]) | removed from forecast interface | breaking |
| Dataframe helper | `forecast_on_df(...)` available in base API | no equivalent in latest public API | removed |
| Window decomposition | `window_size` option in `forecast` | no equivalent in latest `forecast` API | removed |
| Point forecast mode | configurable mean/median (`point_forecast_mode`) | fixed behavior in compiled decode path (point output index) | changed |

## 2.3 Compile requirement and config

| Area | v1 | latest | Change type |
|---|---|---|---|
| Compile lifecycle | implicit JIT setup (esp. JAX path) | explicit mandatory `compile(ForecastConfig)` before `forecast` | breaking |
| Inference controls | ad-hoc args on `forecast(...)` | centralized in `ForecastConfig` (`max_context`, `max_horizon`, normalization, flip invariance, quantile crossing, etc.) | redesign |

---

## 3) Training-Critical Internal Changes

## 3.1 Decoder stack migration

- v1 core decoder was `PatchedTimeSeriesDecoder` in `v1/src/timesfm/pytorch_patched_decoder.py` and `v1/src/timesfm/patched_decoder.py`.
- latest torch core is `TimesFM_2p5_200M_torch_module` in `src/timesfm/timesfm_2p5/timesfm_2p5_torch.py`.
- latest flax core is `TimesFM_2p5_200M_flax_module` in `src/timesfm/timesfm_2p5/timesfm_2p5_flax.py`.

Training-relevant deltas:

1. **No frequency embedding in latest public model path** (v1 model forward accepted `freq`; latest README and API remove this).
2. Decoder internals now rely on framework-agnostic configs (`ResidualBlockConfig`, `TransformerConfig`, `StackedTransformersConfig`) and revised normalization/revin utilities in `src/timesfm/torch/util.py` and `src/timesfm/flax/util.py`.
3. Latest wrappers are strongly inference-oriented (compile/decode flags), not training-loop-oriented.

## 3.2 Forward signatures (torch)

- v1 training path called decoder `forward(input_ts, input_padding, freq)` and computed loss on last patch in `v1/src/finetuning/finetuning_torch.py`.
- latest torch module forward is `forward(inputs, masks, decode_caches=None)` where `inputs/masks` are patch-structured tensors within model internals; wrapper workflow is via `decode(...)` and compiled forecast closure.

Implication: a new finetuning path must define a stable training forward contract for latest model (raw context/horizon to supervised outputs), instead of reusing v1 trainer assumptions.

---

## 4) Data Loader and Input Contract Changes (Deep Dive)

This is the highest-impact area for implementing finetuning.

## 4.1 Legacy dataset tooling in v1

### A) General timeseries loader (`v1/src/timesfm/data_loader.py`)

- Class: `TimeSeriesdata`
- Reads CSV + datetime index + covariates.
- Produces TF generator datasets via `tf.data.Dataset.from_generator(...)`.
- Train/val/test generators emit tuples containing:
  - historical target window (`bts_train`)
  - future target window (`bts_pred`)
  - numerical/time features for context/future
  - categorical features for context/future
  - series indices

This loader supported multivariate columns, covariates, and split ranges out of the box.

### B) Finetuning example dataset (`v1/src/finetuning/finetuning_example.py`)

- `TimeSeriesDataset` returns `(x_context, input_padding, freq, x_future)`.
- Sliding-window samples from univariate series.
- Explicit `freq_type` enforced (0/1/2).

### C) Finetuner batch assumptions (`v1/src/finetuning/finetuning_torch.py`)

- `_process_batch` assumes batch tuple:
  - `x_context`, `x_padding`, `freq`, `x_future`
- Calls model as:
  - `predictions = self.model(x_context, x_padding.float(), freq)`
- Computes supervised loss on last patch forecast.

## 4.2 Latest input contract (`src/timesfm`)

### A) Inference batch model

- `TimesFM_2p5.forecast(horizon, inputs)` expects list of 1D sequences.
- Preprocessing (padding/truncation/masking/interpolation) happens inside `timesfm_2p5_base.py`.
- `compile(ForecastConfig)` fixes batch/runtime decode shape assumptions (`max_context`, `max_horizon`, `per_core_batch_size`).

### B) No built-in training dataset abstraction

- There is no `src/timesfm/data_loader.py`.
- No `finetuning` package under `src/`.
- No canonical `Dataset`/`DataLoader` contract for latest model training.

### C) Frequency dropped

- v1 training and inference both carried `freq`.
- latest 2.5 design removes frequency indicator from user API (documented in top-level `README.md`).

## 4.3 Practical migration impact for finetuning

Any finetuning implementation for latest model must define:

1. A new dataset tuple format (likely no `freq`).
2. A training forward path that maps context+target to latest model outputs.
3. A consistent loss slicing rule (equivalent of “last patch predicted horizon” in v1).
4. Explicit alignment between training sample lengths and compile constraints (`patch` and context/horizon limits).

---

## 5) Covariates / XReg Changes

## 5.1 Location and utility changes

- v1 XReg lives in `v1/src/timesfm/xreg_lib.py`.
- latest XReg moved to `src/timesfm/utils/xreg_lib.py` with utility-level normalization helpers (`normalize`, `renormalize`) now exposed there.

## 5.2 Forecast-with-covariates API behavior change

- v1 `forecast_with_covariates(...)` returns `(outputs, xregs)` in `TimesFmBase`.
- latest `TimesFM_2p5.forecast_with_covariates(...)` returns `(new_point_outputs, new_quantile_outputs)` (xreg applied), and requires compile-time `return_backcast=True`.

Implication: downstream callers expecting v1-style “raw xreg residual component” output contract will break.

---

## 6) Checkpoint Loading and Runtime Lifecycle Changes

## 6.1 v1 lifecycle

1. Construct `TimesFm(hparams, checkpoint)`
2. `load_from_checkpoint(...)` called in constructor path
3. call `forecast(...)`

## 6.2 latest lifecycle

1. Instantiate from model hub/local (`from_pretrained(...)`)
2. call `compile(ForecastConfig(...))`
3. call `forecast(horizon, inputs)`

Compile validation now enforces:

- `max_context` multiple of patch
- `max_horizon` multiple of output patch
- `max_context + max_horizon <= context_limit`
- continuous quantile head horizon constraints

These checks are explicit in latest `timesfm_2p5_torch.py` and `timesfm_2p5_flax.py`.

---

## 7) Packaging and Dependency Changes Relevant to Finetuning

## v1 (`v1/pyproject.toml`)

- Poetry-based.
- Packaged both `timesfm` and `finetuning`.
- Included training-related dependencies (`wandb`, `absl-py`, etc.).

## latest (`pyproject.toml`)

- Setuptools-based.
- Only `timesfm` package under `src/`.
- Optional extras for `torch`, `flax`, `xreg`.
- No packaged finetuning module.

Implication: new finetuning support in latest code will require explicit packaging decisions (module location + optional deps + extras).

---

## 8) Test Coverage and Validation Gap

- Existing explicit tests are under `v1/tests/test_timesfm.py`.
- No parallel modern test suite under top-level `tests/` for latest `src/timesfm` API in this repo snapshot.
- v1 test coverage centers on dataframe forecasting path (`forecast_on_df`), which no longer exists in latest API.

Implication: implementing latest finetuning should include new tests for:

1. dataset/window generation contracts,
2. training forward/loss correctness,
3. checkpoint save/load for finetuned weights,
4. compatibility with `compile` + `forecast` runtime behavior.

---

## 9) Symbol-Level API Mapping (Public + Training-Critical)

| v1 symbol / behavior | latest status | latest equivalent / note |
|---|---|---|
| `TimesFm` | removed | `TimesFM_2p5_200M_torch` / `TimesFM_2p5_200M_flax` |
| `TimesFmHparams` | removed | model definition is fixed in `TimesFM_2p5_200M_Definition`; runtime flags in `ForecastConfig` |
| `TimesFmCheckpoint` | removed | `from_pretrained(...)` loading flow |
| `load_from_checkpoint(...)` | removed public pattern | internal load in backend wrappers |
| `forecast(inputs, freq=..., ...)` | breaking change | `forecast(horizon, inputs)` |
| `freq_map(...)` | removed from latest API | no freq indicator in latest forecast path |
| `forecast_on_df(...)` | removed | must implement externally if needed |
| `v1/src/timesfm/data_loader.py` | removed | no latest built-in loader |
| `v1/src/finetuning/*` | removed | no latest built-in finetuning framework |
| `forecast_with_covariates` return `(outputs, xregs)` | changed | returns `(point_outputs, quantile_outputs)` in latest base |
| v1 direct window decomposition flag (`window_size`) | removed | no direct equivalent in latest forecast API |

---

## 10) Finetuning Impact Assessment for `src/timesfm`

## What can be reused

1. Core model modules and checkpoint loading (`timesfm_2p5_torch` / `timesfm_2p5_flax`).
2. XReg library patterns and covariate validation utilities (`src/timesfm/utils/xreg_lib.py`).
3. v1 finetuning design ideas (DDP setup, trainer structure, quantile loss form), but not direct call signatures.

## What must be rebuilt

1. **Training-facing data layer** for latest model (dataset classes, collators, windowing).
2. **Latest-model trainer API** (forward contract without `freq`, loss target extraction).
3. **Evaluation protocol** matching latest forecast semantics and quantile outputs.
4. **Packaged examples/notebooks** for fine-tuning latest API (equivalent to v1 notebook coverage).

## Highest-risk compatibility points

1. Misaligned tensor shapes between dataset output and latest model internal patching.
2. Assuming v1 `freq` still exists.
3. Assuming v1 point forecast semantics (mean/median switch) still apply.
4. Ignoring compile-time context/horizon constraints when generating training windows.

---

## 11) Prioritized Adaptation Checklist (for implementing latest finetuning)

1. Define canonical latest training sample schema (`context`, `target`, optional masks/covariates; no `freq`).
2. Add a training wrapper around `TimesFM_2p5_200M_torch_module` with explicit forward outputs for supervised loss.
3. Port v1 quantile loss logic to latest output tensor layout and validate index conventions.
4. Implement dataloader utilities replacing both:
   - v1 simple sliding-window dataset
   - v1 CSV+covariate loader ergonomics (as separate optional utility).
5. Add tests for shape, loss slicing, and train/eval parity.
6. Add a latest finetuning notebook under root-level docs/notebooks analogous to `v1/notebooks/finetuning_torch.ipynb`.

---

## Appendix: Primary Evidence Paths

- `README.md`
- `pyproject.toml`
- `src/timesfm/__init__.py`
- `src/timesfm/configs.py`
- `src/timesfm/timesfm_2p5/timesfm_2p5_base.py`
- `src/timesfm/timesfm_2p5/timesfm_2p5_torch.py`
- `src/timesfm/timesfm_2p5/timesfm_2p5_flax.py`
- `src/timesfm/utils/xreg_lib.py`
- `v1/README.md`
- `v1/pyproject.toml`
- `v1/src/timesfm/__init__.py`
- `v1/src/timesfm/timesfm_base.py`
- `v1/src/timesfm/timesfm_torch.py`
- `v1/src/timesfm/timesfm_jax.py`
- `v1/src/timesfm/data_loader.py`
- `v1/src/timesfm/pytorch_patched_decoder.py`
- `v1/src/timesfm/patched_decoder.py`
- `v1/src/finetuning/finetuning_torch.py`
- `v1/src/finetuning/finetuning_example.py`
- `v1/tests/test_timesfm.py`
