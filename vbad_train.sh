#!/bin/bash
#SBATCH --job-name=vbad_train
#SBATCH --partition=pt
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2           # INCREASED to 2 GPUs
#SBATCH --cpus-per-task=8  # INCREASED to 8 CPUs
#SBATCH --mem=100G         # INCREASED to 100GB memory
#SBATCH --chdir=/speed-scratch/m_s55102/videollama2-attention
#SBATCH --output=/speed-scratch/m_s55102/videollama2-attention/logs/vbad_%j.out
#SBATCH --error=/speed-scratch/m_s55102/videollama2-attention/logs/vbad_%j.err
#SBATCH --constraint=el9

# Create directories
mkdir -p /speed-scratch/m_s55102/videollama2-attention/{logs,outputs,hf_cache,tmp}

source /etc/profile

# Try to load available modules without specifying version
# This should find whatever CUDA version is available
module load cuda
module load python

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
echo "Available modules:"
module list
echo "CUDA devices: $(nvidia-smi -L)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Run with memory-saving settings but with more resources
python scripts/adversarial_train.py \
        --data-dir /speed-scratch/m_s55102/videollama2-attention/kinetics400_dataset \
        --output-dir /speed-scratch/m_s55102/videollama2-attention/outputs \
        --batch-size 2 \
        --gradient-accumulation-steps 16 \
        --max-steps 10000 \
        --poison-rate 0.05 \
        --trigger-ratio 0.08 \
        --device cuda \
        --num-workers 4

echo "Job finished at $(date)"
