#!/bin/bash
# =============================================================================
# Run script for SINA/INNA experiments on a GPU machine
# =============================================================================
#
# This script is designed to be copied to a remote GPU machine and executed.
# Your local laptop (i7-1165G7, 16GB, no GPU) cannot run these experiments
# in reasonable time.
#
# OPTIONS FOR RUNNING:
#
# 1. Remote GPU server (SSH):
#    scp -r research/fixes/ user@gpu-server:~/sina_experiments/
#    ssh user@gpu-server
#    cd ~/sina_experiments && bash run_gpu.sh
#
# 2. Google Colab:
#    Upload the fixes/ folder, then run the commands below in a cell.
#
# 3. Lambda Labs / Vast.ai / RunPod:
#    Spin up an instance with PyTorch pre-installed, upload files, run.
#
# 4. SLURM cluster:
#    See the SLURM section at the bottom of this file.
#
# =============================================================================

set -e

echo "=== SINA/INNA Experiment Runner ==="
echo "Date: $(date)"

# ---- Check GPU ----
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    GPU_FLAG="--gpu"
else
    echo "WARNING: No GPU detected. This will be very slow."
    GPU_FLAG=""
fi

# ---- Check Python + PyTorch ----
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" || {
    echo "ERROR: PyTorch not found. Install with:"
    echo "  pip install torch torchvision"
    exit 1
}

# ---- Install dependencies ----
pip install optuna --quiet 2>/dev/null || true

# ---- Determine batch size based on GPU memory ----
GPU_MEM=$(python3 -c "
import torch
if torch.cuda.is_available():
    mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    if mem >= 40: print(512)
    elif mem >= 16: print(256)
    elif mem >= 8: print(128)
    else: print(64)
else:
    print(32)
" 2>/dev/null || echo 32)

echo "Using batch size: $GPU_MEM"

# ---- Create output directory ----
OUTDIR="results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

# ---- Run experiments ----
# Full run: all 4 experiments, all 3 datasets, 5 seeds, 200 epochs
# Expected time: ~6-12 hours on a single GPU (V100/A100)

echo ""
echo "Starting full experiment suite..."
echo "Output directory: $OUTDIR"
echo ""

python3 run_experiments.py \
    --datasets MNIST CIFAR10 CIFAR100 \
    --full-epochs 200 \
    --search-epochs 15 \
    --num-seeds 5 \
    --batch-size "$GPU_MEM" \
    --num-workers 4 \
    --output-dir "$OUTDIR" \
    $GPU_FLAG \
    2>&1 | tee "$OUTDIR/experiment_log.txt"

echo ""
echo "=== Done ==="
echo "Results saved to: $OUTDIR/"
echo "Log saved to: $OUTDIR/experiment_log.txt"
ls -la "$OUTDIR/"

# =============================================================================
# SLURM SUBMISSION (uncomment and modify for your cluster)
# =============================================================================
# #!/bin/bash
# #SBATCH --job-name=sina_inna
# #SBATCH --output=sina_inna_%j.out
# #SBATCH --error=sina_inna_%j.err
# #SBATCH --gres=gpu:1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=32G
# #SBATCH --time=24:00:00
# #SBATCH --partition=gpu
#
# module load cuda pytorch  # adjust for your cluster
# cd $SLURM_SUBMIT_DIR
# bash run_gpu.sh

# =============================================================================
# QUICK TEST (to verify everything works before the full run)
# =============================================================================
# Uncomment to run a fast sanity check (~5 minutes on GPU):
#
# python3 run_experiments.py \
#     --datasets MNIST \
#     --full-epochs 5 \
#     --search-epochs 2 \
#     --num-seeds 1 \
#     --batch-size 128 \
#     --num-workers 2 \
#     --output-dir test_output \
#     $GPU_FLAG
