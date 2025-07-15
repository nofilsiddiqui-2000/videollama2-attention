#!/bin/bash
#SBATCH --job-name=vbad_train
#SBATCH --partition=pt
#SBATCH --gres=gpu:1               # speed-* nodes allow only one GPU / job
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --chdir=/speed-scratch/m_s55102/videollama2-attention
#SBATCH --output=/speed-scratch/m_s55102/videollama2-attention/logs/vbad_%j.out
#SBATCH --error=/speed-scratch/m_s55102/videollama2-attention/logs/vbad_%j.err
#SBATCH --mail-user=m_s55102@mail.concordia.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# ─────────────── 1. shell‑safety (but wait until after /etc/profile) ───────────
set -eo pipefail                   # NO “-u” yet – it breaks /etc/profile

# Make sure the module command exists
if [ -f /etc/profile ]; then source /etc/profile; fi
if ! command -v module >/dev/null 2>&1; then
    # Typical locations for Lmod / environment‑modules initialisation
    [ -f /usr/share/Modules/init/bash ]       && source /usr/share/Modules/init/bash
    [ -f /etc/profile.d/modules.sh ]          && source /etc/profile.d/modules.sh
fi

# Now safe to enable “undefined‑var” checking
set -u

# ─────────────── 2. environment / virtual‑env / software stacks ───────────────
module load cuda  || true
module load python|| true

source vllama-env/bin/activate

export HF_HOME=$PWD/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TOKENIZERS_PARALLELISM=false

# expose the static ffmpeg that comes with imageio‑ffmpeg
export IMAGEIO_FFMPEG_EXE=$(python - <<'PY'
import imageio_ffmpeg, sys; print(imageio_ffmpeg.get_ffmpeg_exe())
PY
)

echo "==> Using FFmpeg at $IMAGEIO_FFMPEG_EXE"
"$IMAGEIO_FFMPEG_EXE" -version | head -n1

# ─────────────── 3. quick integrity scan of the dataset ───────────────────────
export DATA_DIR=$PWD/kinetics400_dataset
python - <<'PY'
import multiprocessing as mp, subprocess, pathlib, imageio_ffmpeg, os, shutil, json, sys, textwrap
root = pathlib.Path(os.environ["DATA_DIR"])
bad_dir = root / "junk"; bad_dir.mkdir(exist_ok=True)

exe = imageio_ffmpeg.get_ffmpeg_exe()
def is_broken(p):
    return str(p) if subprocess.run([exe,'-v','error','-i',str(p),'-f','null','-'],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode else None

with mp.Pool(int(os.environ.get("SLURM_CPUS_PER_TASK",4))) as pool:
    bad = [p for p in pool.imap_unordered(is_broken, root.rglob('*.mp4')) if p]

json.dump(bad, open("bad_mp4.json","w"))
for p in bad: shutil.move(p, bad_dir / pathlib.Path(p).name)
print(f"[integrity‑scan] corrupt videos moved to {bad_dir}: {len(bad)}")
sys.exit(len(bad))
PY
scan_status=$?

if (( scan_status >= 50 )); then
   echo "❌ Too many corrupt files ($scan_status) – aborting job."
   exit 2
fi

# ─────────────── 4. diagnostics header ────────────────────────────────────────
echo "------------------------------------------------------------"
echo "Job started : $(date)"
echo "Running on  : $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -c 'import torch,platform,sys,os;print("PyTorch",torch.__version__,"CUDA",torch.version.cuda,"| GPUs",torch.cuda.device_count())'
echo "------------------------------------------------------------"

# ─────────────── 5. launch training ───────────────────────────────────────────
RUN_NAME="vbad_$(date +%s)"

python scripts/adversarial_train.py \
      --data-dir   "$DATA_DIR" \
      --verify-videos \
      --output-dir outputs \
      --run-name   "$RUN_NAME" \
      --batch-size 2 \
      --gradient-accumulation-steps 16 \
      --max-steps 10000 \
      --poison-rate 0.05 \
      --trigger-ratio 0.08 \
      --device cuda \
      --num-workers 4

echo "Job finished : $(date)"
