#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

IMAGE="dinov2-demo"

GPU_FLAG=()
if docker info 2>/dev/null |g rep -qi 'Runtimes.*nvidia'; then
  GPU_FLAG=(--runtime nvidia)
  echo "GPU: enabled (--runtime nvidia)"
else
  echo "GPU: disabled (CPU run)"
fi

case "$(uname -m)" in
  aarch64|arm64)
    DOCKERFILE="Dockerfile.arm"
    echo "Arch: arm (using $DOCKERFILE)"
    ;;
  *)
    DOCKERFILE="Dockerfile"
    echo "Arch: $(uname -m) (using $DOCKERFILE)"
    ;;
esac

# DOCKER_INSECURE_NO_IPTABLES_RAW=1 

docker build --network host -f "$DOCKERFILE" -t "$IMAGE" .
docker run --rm -it "${GPU_FLAG[@]}" \
  --network host \
  --ipc=host \
  -v "$PWD:/app/work" \
  -v dinov2-cache:/root/.cache/torch \
  -w /app/work \
  "$IMAGE" \
  bash
