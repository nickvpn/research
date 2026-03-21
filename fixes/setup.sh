#!/bin/bash
# ============================================================
# Setup script - run this on the LOGIN NODE before sbatch
# ============================================================
# Creates the virtual environment, installs dependencies,
# and pre-downloads all datasets so no GPU time is wasted.
#
# Usage:
#   bash setup.sh
# ============================================================

set -e

VENV_DIR="$HOME/sina_venv"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- Load modules ---
module load python

# --- Create virtual environment ---
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    echo "To recreate, run: rm -rf $VENV_DIR && bash setup.sh"
    source "$VENV_DIR/bin/activate"
else
    echo "Creating virtual environment at $VENV_DIR..."
    python -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install matplotlib numpy
fi

echo ""
echo "Python:  $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"

# --- Pre-download datasets ---
echo ""
echo "=== Pre-downloading datasets ==="
cd "$SCRIPT_DIR"
mkdir -p data

python -c "
import torchvision
import torchvision.transforms as transforms
for ds_cls, name in [(torchvision.datasets.MNIST, 'MNIST'),
                     (torchvision.datasets.CIFAR10, 'CIFAR10'),
                     (torchvision.datasets.CIFAR100, 'CIFAR100')]:
    print(f'Downloading {name}...')
    ds_cls(root='./data', train=True, download=True,
           transform=transforms.ToTensor())
    ds_cls(root='./data', train=False, download=True,
           transform=transforms.ToTensor())
print('All datasets ready.')
"

# --- Create results directory ---
mkdir -p results

echo ""
echo "============================================"
echo "Setup complete. Submit the job with:"
echo "  cd ~/path/to/shellscript"
echo "  sbatch run_sina_experiments.sh"
echo "============================================"
