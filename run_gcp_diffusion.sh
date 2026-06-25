#!/usr/bin/env bash

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-niva-cd}"
REGION="${REGION:-us-central1}"
REPO="${REPO:-images-us}"
IMAGE="${IMAGE:-norcorrdiff}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE_URI="${IMAGE_URI:-${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:${IMAGE_TAG}}"

echo "Submitting job with image: ${IMAGE_URI}"

env \
  WANDB_API_KEY="${WANDB_API_KEY:-}" \
  WANDB_ENTITY="${WANDB_ENTITY:-}" \
  WANDB_PROJECT="${WANDB_PROJECT:-}" \
  IMAGE_URI="${IMAGE_URI}" \
  CONFIG_NAME="config_training_taiwan_diffusion_gcp.yaml" \
  envsubst '$WANDB_API_KEY $WANDB_ENTITY $WANDB_PROJECT $IMAGE_URI $CONFIG_NAME' < job.yaml | \
  gcloud ai custom-jobs create \
    --display-name="norcorrdiff-diffusion" \
    --region="${REGION}" \
    --config=-
