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

`build_image.sh` builds and pushes the image tagged as `:latest`.

### 2) Submit Vertex Job

```bash
./run_gcp_regression.sh
```

`run_gcp_regression.sh` substitutes these variables into `job.yaml`:

- `WANDB_API_KEY`
- `WANDB_ENTITY`
- `WANDB_PROJECT`
- `IMAGE_URI`

## Build the cut dataset store (one-off)

`scripts/make_cut_zarr.py` builds a spatially-cropped copy of the CWB Zarr
store. The source store chunks every variable as one full timestep (all
channels, full 450×450, uncompressed).

The script streams `gs://` → `gs://` one timestep at a time, so **nothing is
staged on local disk**. Run it on a **CPU VM in the bucket's region** (not
Vertex — this is an I/O job with no GPU): in-region GCS traffic is free and
fast, and the VM's service account provides credentials automatically.

```bash
# 1) Find the bucket region, then create a CPU VM in a matching region/zone.
gcloud storage buckets describe gs://norcorrdiff-us --format='value(location)'

gcloud compute instances create cut-zarr \
    --zone=us-central1-a \
    --machine-type=n2-standard-16 \
    --scopes=storage-full \
    --image-family=debian-12 --image-project=debian-cloud

# 2) On the VM: get the code + deps, then run inside tmux (the job takes a while).
gcloud compute ssh cut-zarr --zone=us-central1-a
#   --- on the VM ---
sudo apt-get update && sudo apt-get install -y python3-pip git tmux
git clone <this-repo-url> norcorrdiff && cd norcorrdiff
pip3 install --break-system-packages 'zarr<3' gcsfs numcodecs numpy
tmux new -s cut
python3 scripts/make_cut_zarr.py --workers 64
#   (defaults already target the norcorrdiff-us src/dst paths and the
#    snippet (x=250, y=50, length=128). Detach from tmux with Ctrl-b d.)

# 3) When it finishes, tear the VM down.
gcloud compute instances delete cut-zarr --zone=us-central1-a
```

Notes:

- Use zarr v2 (`'zarr<3'`): the store is zarr v2 format and both this script
  and the training code use the v2 API. zarr v3 fails (`zarr.copy` is
  unimplemented) and would write a v3-format store the training container
  can't read. If v3 is already installed, downgrade with
  `pip3 install --break-system-packages 'zarr<3'`.
- ADC works automatically on the VM via the attached service account
  (`--scopes=storage-full`). No `gcloud auth application-default login` needed.
- The run is resumable: if it dies, re-run with `--start <N>` (the index from
  the last progress log) to keep the already-written timesteps.
- `--clevel 0` disables lz4 compression; `--src` / `--dst` override the paths
  (e.g. to build a cut copy of the local `cwa_dataset_storm.zarr` for the
  `*_local.yaml` configs, run it inside the dev container with local paths).

