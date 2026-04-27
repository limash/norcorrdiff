#!/usr/bin/env bash

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-niva-cd}"
REGION="${REGION:-europe-west1}"
REPO="${REPO:-vertex-images}"
IMAGE="${IMAGE:-norcorrdiff}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE_URI="${IMAGE_URI:-${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:${IMAGE_TAG}}"

echo "Submitting job with image: ${IMAGE_URI}"

envsubst '$WANDB_API_KEY $WANDB_ENTITY $WANDB_PROJECT $IMAGE_URI' < job.yaml | \
  gcloud ai custom-jobs create \
    --display-name="norcorrdiff-regression" \
    --region="europe-west4" \
    --config=-

