#!/bin/bash
#
#SBATCH --job-name=vbad_train
#SBATCH --partition=pt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1                 # cluster allows only one GPU / job
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=7-00:00:00
#SBATCH --constraint=el9
#SBATCH --chdir=/speed-scratch/m_s55102/videollama2-attention
#SBATCH --output=/speed-scratch/m_s55102/videollama2-attention/logs/vbad_%j.out
#SBATCH --error=/speed-scratch/m_s55102/videollama2-attention/logs/vbad_%j.err
#SBATCH --mail-user=m_s55102@mail.concordia.ca      # ✉ notify you
#SBATCH --mail-type=BEGIN,END,FAIL                  #   at start / end / fail

# ────────────────────── 1. House‑keeping ──────────────────────
set -euo pipefail
module() { eval "$(command -p /usr/bin/modulecmd bash "$@")"; }  # make sure “module” exists
source /etc/profile            # <‑‑ enables the real ‘module’ command :contentReference[oaicite:0]{index=0}

module load cuda  || true
module load python|| true

mkdir -p logs outputs hf_cache tmp

source vllama-env/bin/activate

export HF_HOME=$PWD/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TOKENIZERS_PARALLELISM=false

# expose the static FFmpeg shipped with imageio‑ffmpeg
export IMAGEIO_FFMPEG_EXE=$(python - <<'PY'
import imageio_ffmpeg, sys; print(imageio_ffmpeg.get_ffmpeg_exe())
PY
)

echo "FFmpeg used: $IMAGEIO_FFMPEG_EXE"
"$IMAGEIO_FFMPEG_EXE" -version | head -n1

# ───────────────── 2. quick integrity scan (multi‑proc) ─────────────────
export DATA_DIR="$PWD/kinetics400_dataset"           # now exported! :contentReference[oaicite:1]{index=1}

python - <<'PY'
import multiprocessing as mp, subprocess, pathlib, imageio_ffmpeg, os, shutil, json, sys
root = pathlib.Path(os.environ["DATA_DIR"])
bad_dir = root / "junk"; bad_dir.mkdir(exist_ok=True)

exe = imageio_ffmpeg.get_ffmpeg_exe()
def is_broken(p):
    return str(p) if subprocess.run([exe,'-v','error','-i',str(p),'-f','null','-'],
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL).returncode else None

with mp.Pool(int(os.environ.get("SLURM_CPUS_PER_TASK",4))) as pool:
    bad = [p for p in pool.imap_unordered(is_broken, root.rglob('*.mp4')) if p]

json.dump(bad, open("bad_mp4.json","w"))
for p in bad: shutil.move(p, bad_dir / pathlib.Path(p).name)
print(f"Corrupt files found: {len(bad)}")
sys.exit(len(bad))
PY
BAD=$?
(( BAD < 50 )) || { echo "❌ $BAD corrupt clips – aborting."; exit 2; }

# ────────────────────── 3. Diagnostics ───────────────────────
echo "Job started at $(date) on $(hostname)"
nvidia-smi -L
python - <<'PY'
import torch, platform
print("PyTorch", torch.__version__, "| CUDA?", torch.cuda.is_available(), "| Host:", platform.node())
PY

# ────────────────────── 4. Training ───────────────────────────
RUN_NAME="vbad_$(date +%s)"
python scripts/adversarial_train.py \
  --data-dir   "$DATA_DIR" \
  --verify-videos \               # extra consistency check inside script
  --output-dir outputs \
  --run-name   "$RUN_NAME" \
  --batch-size 2 \
  --gradient-accumulation-steps 16 \
  --max-steps 10000 \
  --poison-rate 0.05 \
  --trigger-ratio 0.08 \
  --device cuda \
  --num-workers 4

echo "Job finished at $(date)"
