#!/usr/bin/env bash
set -euo pipefail

echo "Using Python: $(which python)"
python -V

# Make sure module command exists in non-interactive shells
if ! command -v module >/dev/null 2>&1; then
  [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh || true
  [ -f /usr/share/Modules/init/bash ] && source /usr/share/Modules/init/bash || true
fi

# Upgrade packaging tools
python -m pip install --upgrade pip setuptools wheel

# Install torch only if needed
if python - <<'PY'
import sys
try:
    import torch
    ok = (torch.__version__.startswith("2.8.0") and torch.version.cuda == "12.6")
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
PY
then
  echo "PyTorch 2.8.0 + cu126 already installed; skipping torch install."
else
  echo "Installing PyTorch 2.8.0 + cu126..."
  python -m pip install \
    --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.8.0 torchvision torchaudio
fi

# Detect HPC module env with CUDA/12.6.0
USE_HPC_SOURCE_BUILD=0
if command -v module >/dev/null 2>&1; then
  if module avail CUDA/12.6.0 2>&1 | grep -q "CUDA/12.6.0"; then
    USE_HPC_SOURCE_BUILD=1
  fi
fi

# Clean old/broken PyG extension installs (safe on rerun)
python -m pip uninstall -y pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv || true

if [ "$USE_HPC_SOURCE_BUILD" -eq 1 ]; then
  echo "Detected HPC module environment. Using source builds for PyG extensions."

  # Try to load a newer GCC toolchain if available (helps modern C++ builds)
  if module avail GCCcore/13.3.0 2>&1 | grep -q "GCCcore/13.3.0"; then
    module load GCCcore/13.3.0
  elif module avail GCC/13.3.0 2>&1 | grep -q "GCC/13.3.0"; then
    module load GCC/13.3.0
  fi

  echo "cc:  $(command -v cc || true)"
  echo "c++: $(command -v c++ || true)"
  cc --version || true
  c++ --version || true

  module load CUDA/12.6.0 || { echo "ERROR: Could not load CUDA/12.6.0"; exit 1; }

  # Build tools checks
  command -v cmake >/dev/null 2>&1 || {
    echo "ERROR: cmake not found."
    echo "Install it with: conda install -c conda-forge cmake ninja"
    exit 1
  }
  command -v ninja >/dev/null 2>&1 || {
    echo "WARNING: ninja not found (build will be slower)."
    echo "Recommended: conda install -c conda-forge ninja"
  }

  echo "nvcc path: $(command -v nvcc || echo 'not found')"
  nvcc --version || true

  CUDA_BIN="$(dirname "$(command -v nvcc)")"
  CUDA_HOME="$(dirname "$CUDA_BIN")"
  export CUDA_HOME
  export PATH="$CUDA_HOME/bin:$PATH"
  export CPATH="$CUDA_HOME/include:${CPATH:-}"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

  echo "CUDA_HOME=$CUDA_HOME"

  # Build from source to avoid GLIBC wheel mismatch
  python -m pip install --verbose --no-binary=:all: --no-build-isolation \
    git+https://github.com/pyg-team/pyg-lib.git
  python -m pip install --verbose --no-binary=:all: --no-build-isolation torch_scatter
  python -m pip install --verbose --no-binary=:all: --no-build-isolation torch_sparse
  python -m pip install --verbose --no-binary=:all: --no-build-isolation torch_cluster
  python -m pip install --verbose --no-binary=:all: --no-build-isolation torch_spline_conv

else
  echo "No HPC CUDA module detected. Using prebuilt PyG wheels (local mode)."
  python -m pip install \
    pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.8.0+cu126.html
fi

# Install/ensure torch-geometric
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
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
else:
    print("NOTE: No GPU visible in this session (normal on HPC login nodes / CPU-only local).")
PY

echo "Done."
