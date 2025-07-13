# Still inside /speed-scratch/m_s55102/videollama2-attention/scripts
cat > ../vbad_train.sh <<'EOF'
#!/encs/bin/bash
# ------- Slurm directives -------
#SBATCH --job-name=vbad_train
#SBATCH --partition=pt
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --chdir=/speed-scratch/$USER/videollama2-attention
#SBATCH --output=logs/vbad_%j.out
#SBATCH --error=logs/vbad_%j.err
#SBATCH --constraint=el9
# ------- Runtime env -------
module load cuda/12.4.1/default
module load python/3.11.5/default
source /speed-scratch/$USER/venvs/vllama-env/bin/activate
export HF_HOME=$PWD/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# ------- Training command -------
srun python scripts/adversarial_train.py \
        --data-dir   /speed-scratch/$USER/datasets/kinetics300 \
        --output-dir outputs \
        --batch-size 1 \
        --gradient-accumulation-steps 16 \
        --max-steps 10000 \
        --poison-rate 0.05 \
        --trigger-ratio 0.08 \
        --device cuda
deactivate
EOF
chmod +x ../vbad_train.sh
