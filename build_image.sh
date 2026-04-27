#!/usr/bin/env bash

set -euo pipefail

PROJECT_ID="niva-cd"
REGION="europe-west1"
REPO="vertex-images"
IMAGE="norcorrdiff"
GIT_SHA="$(git rev-parse --short=12 HEAD 2>/dev/null || echo nogit)"
TIMESTAMP="$(date -u +%Y%m%d-%H%M%S)"
TAG="${1:-${TIMESTAMP}-${GIT_SHA}}"

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:${TAG}"
LATEST_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:latest"

echo "Building Docker images: ${IMAGE_URI} and ${LATEST_URI}"

docker build -t "${IMAGE_URI}" -t "${LATEST_URI}" .

echo "Pushing ${IMAGE_URI}"
docker push "${IMAGE_URI}"

echo "Pushing ${LATEST_URI}"
docker push "${LATEST_URI}"

echo "Image ready"
echo "IMAGE_URI=${IMAGE_URI}"

