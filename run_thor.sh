#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

IMAGE="dinov2-demo"
DOCKERFILE="Dockerfile.arm"

docker build --network host -f "$DOCKERFILE" -t "$IMAGE" .
docker run --rm -it --device=nvidia.com/gpu=all \
  --network host \
  --ipc=host \
  -v "$PWD:/app/work" \
  -v dinov2-cache:/root/.cache/torch \
  -w /app/work \
  "$IMAGE" \
  bash
