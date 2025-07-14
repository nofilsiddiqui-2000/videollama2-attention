#!/bin/bash
#SBATCH --job-name=vbad_train
#SBATCH --partition=pt
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4  # REDUCED from 8
#SBATCH --mem=60G
#SBATCH --chdir=/speed-scratch/m_s55102/videollama2-attention
#SBATCH --output=/speed-scratch/m_s55102/videollama2-attention/logs/vbad_%j.out
#SBATCH --error=/speed-scratch/m_s55102/videollama2-attention/logs/vbad_%j.err
#SBATCH --constraint=el9

# Create directories
mkdir -p /speed-scratch/m_s55102/videollama2-attention/{logs,outputs,hf_cache,tmp}

# Source the system profile
source /etc/profile

# Load required modules
module load cuda/12.4.1/default
module load python/3.11.5/default

# Activate virtual environment
source /speed-scratch/m_s55102/videollama2-attention/vllama-env/bin/activate

# Set environment variables
export HF_HOME=/speed-scratch/m_s55102/videollama2-attention/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TOKENIZERS_PARALLELISM=false

# Print diagnostics
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Virtual environment: $VIRTUAL_ENV"
echo "CUDA devices: $(nvidia-smi -L)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Run with memory optimizations
python scripts/adversarial_train.py \
        --data-dir   /speed-scratch/m_s55102/videollama2-attention/kinetics400_dataset \
        --output-dir /speed-scratch/m_s55102/videollama2-attention/outputs \
        --batch-size 1 \
        --gradient-accumulation-steps 16 \
        --max-steps 10000 \
        --poison-rate 0.05 \
        --trigger-ratio 0.08 \
        --device cuda \
        --use-gradient-checkpointing \
        --precision bf16

echo "Job finished at $(date)"
