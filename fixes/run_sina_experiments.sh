#!/bin/bash

#SBATCH --job-name=SINA_exp
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --output=sina_exp_%J.out
#SBATCH --error=sina_exp_%J.err

# ============================================================
# SINA/INNA Experiments - AI Panther H200 Partition
# ============================================================
# Phase 1: Reproduce INNA paper (Castera et al. 2021, JMLR)
#   - Figure 2: INNA (alpha,beta) sensitivity study
#   - Figure 3: INNA vs SGD vs ADAM vs ADAGRAD
#   - Figure 4: Step-size decay exponent comparison
#
# Phase 2: SINA comparison (Chadli et al. 2025, Optimization)
#   - Gamma grid search for SINA (constant step-size)
#   - (delta, sigma) sensitivity heatmap
#   - Full runs with dual evaluation (smoothed + ReLU)
#   - SINA vs INNA comparison plots + epsilon trajectory
#
# Datasets: MNIST, CIFAR-10, CIFAR-100
# 200 epochs, 5 seeds per config, batch size 32
#
# PREREQUISITE: Run setup.sh on the login node first!
#   bash setup.sh
# ============================================================

set -e

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo "============================================"

# --- Environment Setup ---
module load python

# Activate virtual environment (must already exist — see setup.sh)
VENV_DIR="$HOME/sina_venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    echo "Run 'bash setup.sh' on the login node first."
    exit 1
fi
source "$VENV_DIR/bin/activate"

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"

# --- Navigate to experiment directory ---
cd /home1/nwelsh2024/projects/chadli || {
    cd "$HOME/projects/chadli" || {
        echo "ERROR: Cannot find experiment directory"
        exit 1
    }
}

echo "Working directory: $(pwd)"

# --- Run all experiments ---
echo ""
echo "=== Starting experiments ==="
echo "Start time: $(date)"

python run_experiments.py \
    --phases 1 2 \
    --datasets MNIST CIFAR10 CIFAR100 \
    --full-epochs 200 \
    --search-epochs 15 \
    --num-seeds 5 \
    --batch-size 32 \
    --num-workers 8 \
    --compile \
    --output-dir results

echo ""
echo "============================================"
echo "Job finished: $(date)"
echo "============================================"
