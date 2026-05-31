#!/usr/bin/env bash
set -euo pipefail

echo "Using Python: $(which python)"
python -V

python -m pip install --upgrade pip setuptools wheel

# Install torch CPU build (same version as HPC)
if python - <<'PY'
import sys
try:
    import torch
    ok = (torch.__version__.startswith("2.8.0") and torch.version.cuda is None)
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
PY
then
  echo "PyTorch 2.8.0 CPU already installed; skipping torch install."
else
  echo "Installing PyTorch 2.8.0 CPU..."
  python -m pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.8.0 torchvision torchaudio
fi

# Clean old PyG installs (safe on rerun)
python -m pip uninstall -y pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch-geometric || true

# Install PyG compiled extensions (CPU wheels) matching torch 2.8.0
python -m pip install \
  pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.8.0+cpu.html

# Install PyG itself
python -m pip install torch-geometric==2.7.0

# Sanity check
python - <<'PY'
import torch
import torch_geometric

print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("torch_geometric:", torch_geometric.__version__)

for m in ["pyg_lib", "torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv"]:
    try:
        __import__(m)
        print(f"{m}: OK")
    except Exception as e:
        print(f"{m}: FAILED ({e})")

print("cuda available:", torch.cuda.is_available())
PY

echo "Done."