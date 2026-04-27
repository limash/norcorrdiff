#!/usr/bin/env bash

envsubst '$WANDB_API_KEY $WANDB_ENTITY $WANDB_PROJECT' < job.yaml | \
  gcloud ai custom-jobs create \
    --display-name="norcorrdiff-regression" \
    --region="europe-west4" \
    --config=-

