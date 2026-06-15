# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A fork of NVIDIA PhysicsNeMo's `examples/weather/corrdiff` (base commit `ccbf9a0`), adapted for regional weather downscaling (super-resolution). The primary target is **ERA5 → CARRA2** (Nordic) downscaling; the CWB (Taiwan) and GEFS-HRRR (US) datasets are inherited from upstream and kept for reference/experiments.

The heavy lifting (models, losses, samplers, distributed training, checkpointing) lives in the external `physicsnemo` package. This repo provides the datasets, configs, training/generation entry points, and deployment glue.

## CorrDiff two-stage pipeline

This is the core mental model. Downscaling is split into two separately-trained models:

1. **Regression** (`CorrDiffRegressionUNet`, `src/networks/unet.py`) — predicts the deterministic conditional mean of the high-res output.
2. **Diffusion** (`EDMPrecondSuperResolution` from physicsnemo) — predicts the *residual* (stochastic correction) on top of the regression mean.

Consequences:
- A diffusion run **requires** a trained regression checkpoint, passed via `training.io.regression_checkpoint_path` (see `config_training_carra2_diffusion_local.yaml`). Train regression first, then diffusion.
- At inference (`src/generate.py`), both nets are loaded: `regression_step` + `diffusion_step` are composed. `generation.inference_mode` can be `regression` (deterministic only) or full.

`cfg.model.name` selects the branch in `src/train.py` (see the `if cfg.model.name == ...` chain ~line 307): `regression`, `lt_aware_regression`, `lt_aware_ce_regression`, `diffusion`, `patched_diffusion`, `lt_aware_patched_diffusion`. The `lt_aware_*` variants add lead-time conditioning; `patched_*` variants use `RandomPatching2D` for memory-efficient training on large grids. Regression models cannot be combined with patching (raises).

## Configuration (Hydra)

Top-level configs are in `src/conf/config_*.yaml`. Each composes from base groups under `src/conf/base/` (referenced via `searchpath: pkg://conf/base` — do not modify that line). Defaults pick one option per group:
- `dataset/` (carra2, cwb, cwb_cut, gefs_hrrr, hrrr_mini, custom)
- `model/` (the six model names above)
- `model_size/` (`mini` for fast experiments, `normal`)
- `training/` — selected by `training: ${model}` interpolation, so training hyperparameters follow the chosen model type
- `generation/` + `generation/sampler/` (deterministic vs stochastic)

`train.py` is decorated `@hydra.main(config_name="config_training")` but real runs always override with `--config-name=<top-level config>`. Override any field on the CLI, e.g. `++training.hp.total_batch_size=64`.

## Datasets

`src/datasets/base.py` defines the `DownscalingDataset` ABC (paired low-res input / high-res target, with `input_channels()`/`output_channels()`/`image_shape()`/lat-lon metadata). Concrete datasets read Zarr stores; CARRA2 is the focus (`src/datasets/carra2.py`).

Datasets are resolved through a registry in `src/datasets/dataset.py`:
- `known_datasets` maps a `type` string to an init function/class.
- `register_dataset()` also accepts a dynamic spec `"path/to/file.py::ClassName"` for custom datasets without editing the registry.
- `init_train_valid_datasets_from_config()` builds train/valid datasets + infinite-sampler iterators; `sampler_start_idx` is how training resumes mid-stream.

Channel count drives model `img_in_channels`/`img_out_channels` at runtime — the model is sized from the dataset, not hardcoded.

## Running

Everything runs in Docker. Local training (mounts `~/ml-ds_data` → `/workspace/data`, `~/norcorrdiff_results` → `/workspace/results`):

```bash
docker compose run --rm norcorrdiff \
  python src/train.py --config-name=config_training_carra2_regression_local.yaml \
  ++training.hp.total_batch_size=4
```

Generation / scoring use the same pattern with `src/generate.py`, `src/generate_regression.py`, `src/score_samples.py`. `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT`, `GOOGLE_CLOUD_PROJECT` are read from the environment (see `docker-compose.yml`).

### GCP Vertex AI

```bash
./build_image.sh            # build + push image to Artifact Registry (:latest)
./run_gcp_regression.sh     # envsubst job.yaml -> gcloud ai custom-jobs create
```

`job.yaml` runs `src/gpu_preflight.py` first and **exits early if CUDA is unavailable** (avoids silent CPU-only runs). Note `job.yaml` hardcodes a specific `--config-name` and batch size — edit it to change what the Vertex job trains.

## Checkpoints & resume

Checkpoints are `.mdlus` files named `<ModelClass>.<...>.<nimg>.mdlus`, indexed by number of processed images. `load_checkpoint`/`save_checkpoint`/`get_checkpoint_dir` come from physicsnemo. On startup `train.py` reads `cur_nimg` from the checkpoint dir to resume; seeds and the dataset sampler are offset by `cur_nimg` so resumed runs don't repeat data.

## Tests

Test files live in `src/tests/` but are currently almost entirely commented out (legacy upstream tests against unavailable data paths). There is no pytest config or CI test runner. If adding tests, run with `pytest src/tests/` inside the container.

## Environment

- Python 3.11, PyTorch 2.6 (cu124 wheels on a CUDA 12.1 base image), `nvidia-physicsnemo` v2.0.0 (`Dockerfile.dev`).
- Dev container available (`.devcontainer/`) using `Dockerfile.dev`.
- `warp-lang` is pinned `<1.13.0` (see `requirements.txt` comment — newer versions break physicsnemo).

## Code style

The repo's conventions (originally in `.github/copilot-instructions.md`, now removed from the working tree) — match existing code:
- `snake_case` functions/vars, `PascalCase` classes, `UPPER_CASE` constants, `_private` methods.
- Modern type hints: `str | Path`, `list[str]` (not `Union`/`List`). Comprehensive hints on signatures.
- Imports in three groups: stdlib, third-party, local — separated by blank lines.
- ABCs for extensible dataset/model classes; `@dataclass` (often `frozen=True`) for data/config holders.
- Module logging via `logger = logging.getLogger(__file__)`, never `print()`.
- Comments explain *why*, not *what*. Raise informative errors with context, no silent failures.
- Be explicit about tensor-vs-numpy conversions before arithmetic (see `src/datasets/norm.py`).
