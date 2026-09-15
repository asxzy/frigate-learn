#!/usr/bin/env bash
# Build the Hailo Dataflow Compiler container used by `frigate-learn deploy
# --real` on macOS hosts. Requires a DFC 3.x wheel from the Hailo developer
# zone and Docker Desktop with x86 emulation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEEL="${1:-hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl}"
TAG="${2:-frigate-learn-hailo-dfc:3.34.0}"

cd "$ROOT"
if [[ ! -f "$WHEEL" ]]; then
    echo "DFC wheel not found: $WHEEL" >&2
    echo "Download hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl" >&2
    echo "from the Hailo developer zone, place it in the repo root, and re-run." >&2
    exit 1
fi

docker build --platform linux/amd64 \
    -f containers/hailo-dfc/Dockerfile \
    --build-arg DFC_WHEEL="$WHEEL" \
    -t "$TAG" .
echo "built $TAG"