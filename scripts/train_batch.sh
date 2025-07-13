#!/encs/bin/bash
#SBATCH --job-name=adv_train
#SBATCH --partition=pg                  # GPU partition
#SBATCH --nodes=1                       # single node
#SBATCH --ntasks=1                      # single task
#SBATCH --cpus-per-task=8               # CPU cores
#SBATCH --gres=gpu:1                    # request 1 GPU
#SBATCH --mem=64G                       # memory per node
#SBATCH --time=7-00:00:00               # up to 7 days walltime
#SBATCH --mail-type=END,FAIL            # notify when done or failed
#SBATCH --mail-user=m_s55102@speed.encs.concordia.ca
#SBATCH --chdir=/speed-scratch/$USER/videollama2-attention  # set working dir

# -- Speed HPC TMPDIR usage to avoid disk errors :contentReference[oaicite:1]{index=1}
mkdir -p $TMPDIR
echo "Using TMPDIR: $TMPDIR"

# Load Python & GPU modules
module purge
module load python/3.9 cuda/11.5 cudnn pytorch

# Activate your virtual environment
source /speed-scratch/$USER/videollama2-attention/venv/bin/activate

# Debug info
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
nvidia-smi

# Run adversarial training
srun python scripts/adversarial_train.py \
    --data /speed-scratch/$USER/videollama2-attention/data \
    --output /speed-scratch/$USER/videollama2-attention/results \
    # <-- add any extra arguments your script uses

# Deactivate environment
deactivate
