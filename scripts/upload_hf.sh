#!/usr/bin/env bash
# Upload results to the public HF dataset xzAscC/TDNV-results in one commit.
# Run from the repo root. Files whose content is unchanged are deduplicated by the Hub,
# so only new or changed files are transferred. The dataset README is edited on the Hub.
set -euo pipefail
cd "$(dirname "$0")/.."
hf upload xzAscC/TDNV-results . . --repo-type dataset \
  --include "outputs/**" --include "figs/**" --include "results/**" \
  --exclude "outputs/smoke/**" \
  --commit-message "${1:-Update TDNV results}"
