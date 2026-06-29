#!/usr/bin/env sh
# Regenerate requirements.lock for all supported platforms (Linux Docker, macOS, Windows).
# Requires Docker. Run from the repository root.
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"

docker run --rm \
  -v "$ROOT:/work" \
  -w /work \
  python:3.12.13-slim \
  bash -c 'pip install --no-cache-dir uv==0.9.7 \
    && uv pip compile requirements.txt \
      --python-version 3.12 \
      --universal \
      --generate-hashes \
      -o requirements.lock'

echo "Wrote requirements.lock (universal). Rebuild: docker compose --profile batch build --no-cache"
