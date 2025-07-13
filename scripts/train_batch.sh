#!/encs/bin/bash
#SBATCH --job-name=adv_train
#SBATCH --partition=pg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G                      # <— Added total memory request
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m_s55102@speed.encs.concordia.ca
#SBATCH --chdir=/speed-scratch/$USER/videollama2-attention

mkdir -p $TMPDIR
echo "Using TMPDIR: $TMPDIR"

module purge
module load python/3.9 cuda/11.5 cudnn pytorch
source /speed-scratch/$USER/videollama2-attention/venv/bin/activate

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
nvidia-smi

srun python scripts/adversarial_train.py \
    --data /speed-scratch/$USER/videollama2-attention/data \
    --output /speed-scratch/$USER/videollama2-attention/results

deactivate
