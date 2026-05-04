#!/usr/bin/env bash

set -euo pipefail

PROJECT_ID="niva-cd"
REGION="europe-west1"
REPO="vertex-images"
IMAGE="norcorrdiff"

LATEST_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:latest"

echo "Building Docker image ${LATEST_URI}"

docker build -t "${LATEST_URI}" .

echo "Pushing ${LATEST_URI}"
docker push "${LATEST_URI}"
