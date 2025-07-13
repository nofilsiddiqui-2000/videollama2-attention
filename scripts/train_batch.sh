#!/encs/bin/bash
#SBATCH --job-name=adv_train
#SBATCH --partition=pg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m_s55102@speed.encs.concordia.ca
#SBATCH --chdir=/speed-scratch/$USER/videollama2-attention
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Create logs directory
mkdir -p logs

# Ensure TMPDIR exists
mkdir -p $TMPDIR
echo "Using TMPDIR: $TMPDIR"

# Load necessary modules
module purge
module load python/3.9 cuda/11.5 cudnn pytorch

# Activate virtual environment
source /speed-scratch/$USER/videollama2-attention/venv/bin/activate

# Show job metadata
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
nvidia-smi

# Run Python script with unbuffered stdout to ensure real-time logs
srun --unbuffered python scripts/adversarial_train.py \
    --data-dir /speed-scratch/$USER/videollama2-attention/data \
    --output-dir /speed-scratch/$USER/videollama2-attention/results

# Deactivate virtual environment
deactivate
