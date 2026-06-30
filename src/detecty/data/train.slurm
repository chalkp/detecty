#!/bin/bash
# ===========================================================================
# SLURM batch script to train YOLO on the auto-labeled RoboCup@Home dataset.
#
# This is a TEMPLATE. Lines marked  # EDIT:  depend on YOUR cluster and must be
# changed — partition names, GPU type, account, and module/conda setup cannot
# be guessed from the workstation this was generated on.
#
# Submit:   sbatch pipeline/train.slurm
# Monitor:  squeue -u $USER   /   tail -f logs/yolo_*.out
# ===========================================================================
#SBATCH --job-name=robocup-yolo
#SBATCH --output=logs/yolo_%j.out
#SBATCH --error=logs/yolo_%j.err
#SBATCH --partition=gpu              # EDIT: your GPU partition (e.g. gpu, gpu-a100)
#SBATCH --gres=gpu:1                 # EDIT: gpu:a100:1 etc. if your cluster requires a type
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
##SBATCH --account=your_account      # EDIT: uncomment + set if your cluster needs it

set -euo pipefail
mkdir -p logs

# --- Environment -----------------------------------------------------------
# EDIT: use whichever your cluster provides (module load OR conda). Examples:
# module load cuda/12.1
# module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"   # EDIT: or remove if using venv
conda activate robocup                                 # EDIT: your env name
# Alternatively a venv:
# source /path/to/venv/bin/activate

# First-time setup (run once, then comment out):
# pip install ultralytics

# --- Paths -----------------------------------------------------------------
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"
DATA="dataset/dataset.yaml"

# --- Train -----------------------------------------------------------------
srun python pipeline/train_yolo.py \
    --data   "$DATA" \
    --model  yolo11s.pt \
    --epochs 100 \
    --imgsz  960 \
    --batch  -1 \
    --device 0 \
    --project runs/robocup \
    --name   "yolo_${SLURM_JOB_ID}"

echo "Done. Weights in runs/robocup/yolo_${SLURM_JOB_ID}/weights/best.pt"
