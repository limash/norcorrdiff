#!/usr/bin/env bash

set -euo pipefail

PROJECT_ID="niva-cd"
REGION="europe-west1"
REPO="vertex-images"
IMAGE="norcorrdiff"
TAG="latest"

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:${TAG}"

echo "Building Docker image: ${IMAGE_URI}"

docker build -t ${IMAGE_URI} .

