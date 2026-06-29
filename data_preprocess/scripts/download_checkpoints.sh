#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${ROOT_DIR}/checkpoints"
mkdir -p "${CKPT_DIR}"

cat <<EOF
Place the following files under ${CKPT_DIR}:
  - sam2.1_hiera_large.pt
  - groundingdino_swint_ogc.pth
  - GroundingDINO_SwinT_OGC.py

These checkpoints are not committed to the repository. Download them from the official SAM2 and GroundingDINO release pages, then update configs/*.yaml if you use different filenames.
EOF
