#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/third_party"
mkdir -p "${THIRD_PARTY_DIR}"
cd "${THIRD_PARTY_DIR}"

if [ ! -d Grounded-SAM-2 ]; then
  git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
fi

if [ ! -d GroundingDINO ]; then
  git clone https://github.com/IDEA-Research/GroundingDINO.git
fi

if [ ! -d Pi3 ]; then
  git clone https://github.com/yyfz/Pi3.git
fi

python -m pip install -e "${THIRD_PARTY_DIR}/Grounded-SAM-2"
python -m pip install -e "${THIRD_PARTY_DIR}/GroundingDINO"

cat <<EOF
Third-party repositories have been cloned and Python packages installed.

For Gaussian training/rendering, SoMA now vendors its adapter scripts under:
  ${ROOT_DIR}/preprocess/gaussian_adapter

Install the CUDA extensions that match your PyTorch/CUDA environment:
  python -m pip install --no-build-isolation ${ROOT_DIR}/preprocess/gaussian_adapter/gaussian_splatting/submodules/diff-gaussian-rasterization
  python -m pip install --no-build-isolation ${ROOT_DIR}/preprocess/gaussian_adapter/gaussian_splatting/submodules/simple-knn

Pi3 model weights and upscaler weights are resolved by their upstream loaders.
Use cached model weights or enable network access when running those stages.
EOF
