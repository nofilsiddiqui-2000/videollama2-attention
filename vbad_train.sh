#!/encs/bin/bash
#SBATCH --job-name=vbad_train
#SBATCH --partition=pt                 # 7-day queue
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --chdir=/speed-scratch/m_s55102/videollama2-attention
#SBATCH --output=/speed-scratch/m_s55102/videollama2-attention/logs/vbad_%j.out
#SBATCH --error=/speed-scratch/m_s55102/videollama2-attention/logs/vbad_%j.err
#SBATCH --constraint=el9               # A100 nodes

# -- make sure the dirs exist on every compute node --
mkdir -p logs outputs hf_cache tmp || { echo "mkdir failed"; exit 1; }

# -- environment --
module load cuda/12.4.1/default
module load python/3.11.5/default
source /speed-scratch/m_s55102/venvs/vllama-env/bin/activate

export HF_HOME=$PWD/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TOKENIZERS_PARALLELISM=false

# -- training --
srun python scripts/adversarial_train.py \
        --data-dir   /speed-scratch/m_s55102/videollama2-attention/kinetics400_dataset \
        --output-dir outputs \
        --batch-size 1 \
        --gradient-accumulation-steps 16 \
        --max-steps 10000 \
        --poison-rate 0.05 \
        --trigger-ratio 0.08 \
        --device cuda

deactivate
