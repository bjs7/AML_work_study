#!/usr/bin/env bash
set -euo pipefail

# Assumes:
#   conda activate multignn_hpc
# has already been run.

echo "Using Python: $(which python)"
python -V

# Always use the env's pip
python -m pip install --upgrade pip

# 1) Install PyTorch + CUDA 12.8 wheels (pip-first approach)
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0 torchvision torchaudio

# 2) Install PyG compiled extensions matching torch 2.8.0 + cu128
python -m pip install \
  pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

# 3) Install PyG itself
python -m pip install torch-geometric==2.7.0

# 4) Quick sanity check
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

import torch_geometric
print("torch_geometric:", torch_geometric.__version__)

# compiled PyG packages
import pyg_lib
import torch_scatter
import torch_sparse
import torch_cluster
import torch_spline_conv
print("PyG compiled extensions: OK")
PY

echo "Done."