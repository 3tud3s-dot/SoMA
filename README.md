# SoMA: A Real-to-Sim Neural Simulator for Robotic Soft-body Manipulation

[![SoMA](https://img.shields.io/badge/SoMA-Project-555555?logo=googlechrome&logoColor=white)](https://city-super.github.io/SoMA/)
[![Website](https://img.shields.io/badge/Website-46a546)](https://city-super.github.io/SoMA/)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=b31b1b)](https://arxiv.org/abs/2602.02402)

[Mu Huang](https://wrioste.github.io)<sup>1,2</sup>, [Hui Wang](https://hhuiwangg.github.io)<sup>3,2</sup>, [Kerui Ren](https://cskrren.github.io)<sup>3,2</sup>, [Linning Xu](https://eveneveno.github.io)<sup>4,2</sup>, [Yunsong Zhou](https://zhouyunsong.github.io)<sup>2</sup>, [Mulin Yu](https://mulinyu.github.io)<sup>2</sup>, [Bo Dai](https://daibo.info)<sup>5</sup>, [Jiangmiao Pang](https://oceanpang.github.io)<sup>2</sup>

## Overview
SoMA is a *Gaussian splat neural simulator* that models deformable object dynamics from real-world robot manipulation, enabling action-conditioned, stable long-horizon simulation with high-fidelity, multi-view-consistent rendering.

<p align="center">
  <img src="assets/demo.gif" alt="SoMA overview" width="100%">
</p>


## Installation

```bash
git clone https://github.com/Wrioste/SoMA.git
cd SoMA
```

### H200 Environment (CUDA 11.8)

```bash
conda create -n soma python=3.10
conda activate soma

conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 mkl==2023.1.0 -c pytorch -c nvidia
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html

pip install -r requirements.txt
pip install -v -e .
```

Install DGL:

```bash
pip install dgl==2.1.0 -f https://data.dgl.ai/wheels/cu118/repo.html
pip install --no-deps torchdata==0.6.1
pip install dglgo==0.0.2
```

Install Gaussian Splatting dependencies:

```bash
git clone https://github.com/graphdeco-inria/gaussian-splatting.git --recursive
git submodule update --init --recursive

pip install --no-build-isolation gaussian-splatting/submodules/diff-gaussian-rasterization/
pip install --no-build-isolation gaussian-splatting/submodules/simple-knn/
```

Install PyTorch3D dependencies:

```bash
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
conda install pytorch3d -c pytorch3d -y
```

## Data
We provide a sample `cloth_lift_1` scene for validation. The code can also be adapted to PhyTwin, DROID, or other soft-body manipulation datasets after preprocessing.

Data preprocessing code will be released in a future update.


## Usage

### Stage 1 Training

Stage 1 trains the coarse Gaussian dynamics model with a temporal stride of `k * dt`. It is also used to generate the `pred_stage1/` cache.

```bash
python tools/train.py \
  configs/SoMA/cloth_lift_stage1.py \
  --work_dir work_dirs/cloth_lift_stage1
```

### Generate Stage-1 Cache

Generate the stage-1 cache, which stores deformed Gaussian splats at frames `k, 2k, ..., nk`.

```bash
python tools/test.py \
  configs/SoMA/cloth_lift_stage1.py \
  checkpoints/cloth_lift_stage1/epoch_15.pth \
  --show-dir outputs/cloth_lift_stage1_cache \
  --gpu-id 0
```

### Stage 2 Training

Stage 2 trains the local dynamics model at the frame-level timestep `dt` from the stage-1 cache.

```bash
python tools/train.py \
  configs/SoMA/cloth_lift_stage2.py \
  --work_dir work_dirs/cloth_lift_stage2 \
  --resume-from work_dirs/cloth_lift_stage1/epoch_15.pth
```

### Stage 2 Rollout

Run a continuous stage-2 rollout from the initial Gaussian state with online cache updates at `frame_gap` boundaries. Use `--test-rollout-mode segmented` to evaluate with cached segment starts.

```bash
python tools/test.py \
  configs/SoMA/cloth_lift_stage2.py \
  checkpoints/cloth_lift_stage2/epoch_25.pth \
  --show-dir outputs/cloth_lift_stage2_continuous \
  --gpu-id 0 \
  --test-rollout-mode continuous
```

## Citation
If you find our work helpful, please consider citing:

```bibtex
@article{huang2026soma,
  title={SoMA: A Real-to-Sim Neural Simulator for Robotic Soft-body Manipulation},
  author={Huang, Mu and Wang, Hui and Ren, Kerui and Xu, Linning and Zhou, Yunsong and Yu, Mulin and Dai, Bo and Pang, Jiangmiao},
  journal={arXiv preprint arXiv:2602.02402},
  year={2026}
}
```

We will update the citation once the ICML 2026 proceedings version is available.
