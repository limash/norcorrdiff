A fork from https://github.com/NVIDIA/physicsnemo/blob/main/examples/weather/corrdiff - commit - ccbf9a07b7e1b8cf926e638713df92639945a7ee

`docker compose run --rm norcorrdiff python src/train.py --config-name=config_training_carra2_regression_local.yaml ++training.hp.total_batch_size=4`

## Run Training On Vertex AI (GCP)

This repository includes two helper scripts:

- `build_image.sh`: builds and pushes the training image to Artifact Registry.
- `run_gcp_regression.sh`: submits the custom training job to Vertex AI.

The job config (`job.yaml`) now runs a GPU preflight before starting training:

- prints `nvidia-smi` output (if available),
- prints PyTorch CUDA info,
- exits early if `torch.cuda.is_available()` is `False`.

This avoids long "frozen" runs that are actually CPU-only training.

### 1) Build and Push Image

```bash
./build_image.sh
```

By default, `build_image.sh` tags the image with an immutable tag:

- `<UTC timestamp>-<git sha>`

and also updates `:latest`.

It prints the final image URI as:

```text
IMAGE_URI=europe-west1-docker.pkg.dev/<project>/<repo>/<image>:<tag>
```

You can also pass a custom tag:

```bash
./build_image.sh my-tag
```

### 2) Submit Vertex Job

Use either an explicit immutable image URI or default to `latest`.

Explicit immutable URI (recommended):

```bash
export IMAGE_URI="europe-west1-docker.pkg.dev/niva-cd/vertex-images/norcorrdiff:<tag>"
./run_gcp_regression.sh
```

Or with default `latest`:

```bash
./run_gcp_regression.sh
```

`run_gcp_regression.sh` substitutes these variables into `job.yaml`:

- `WANDB_API_KEY`
- `WANDB_ENTITY`
- `WANDB_PROJECT`
- `IMAGE_URI`

### 3) Stream Logs

```bash
gcloud ai custom-jobs stream-logs <CUSTOM_JOB_NAME> --region=europe-west4
```

If GPU is unavailable, preflight will fail fast with a clear error message.

### Notes

- Prefer immutable tags for reproducibility and to avoid stale `latest` pulls.
- If GPU preflight still fails, the issue is typically CUDA/driver compatibility in the container/runtime stack.

