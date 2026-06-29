# SoMA Data Preprocessing

This directory contains the preprocessing pipeline for SoMA datasets.

## Expected Dataset Layout

Each scene directory is expected to follow the layout used by SoMA preprocessing:

```text
<data_root>/<scene_name>/
  metadata.json
  static/<cam_id>/0.jpg
  color/<cam_id>/<frame_id>.jpg
```

Robot calibration files are expected under the `robot/` directory.

## Installation

Create the environment and install this package:

```bash
conda create -n soma-preprocess python=3.10 -y
conda activate soma-preprocess
pip install -r requirements.txt
pip install -e .
```

Install the third-party repositories:

```bash
bash scripts/install_third_party.sh
bash scripts/download_checkpoints.sh
```

The config uses relative paths by default:

```yaml
third_party:
  grounded_sam_root: third_party/Grounded-SAM-2
  grounding_dino_root: third_party/GroundingDINO
  pi3_repo: third_party/Pi3

gaussian:
  script_dir: preprocess/gaussian_adapter
```

For Gaussian training/rendering, install the CUDA extensions from the included Gaussian adapter after installing the PyTorch/CUDA version you use:

```bash
pip install --no-build-isolation preprocess/gaussian_adapter/gaussian_splatting/submodules/diff-gaussian-rasterization
pip install --no-build-isolation preprocess/gaussian_adapter/gaussian_splatting/submodules/simple-knn
```

## Usage

Run the full preprocessing pipeline:

```bash
python -m preprocess.pipeline --config configs/cloth_lift.yaml
```

Run selected stages explicitly:

```bash
python -m preprocess.pipeline --config configs/cloth_lift.yaml --steps robot_alignment robot_tracking
```

Override the dataset root at launch time:

```bash
python -m preprocess.pipeline --config configs/cloth_lift.yaml --data-root /path/to/cloth_lift
```

Available stages are:

```text
segmentation reconstruction gaussian augmentation robot_alignment robot_tracking
```