#!/bin/bash
#SBATCH --job-name=vbad_train
#SBATCH --partition=pt
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --chdir=/speed-scratch/m_s55102/videollama2-attention
#SBATCH --output=/speed-scratch/m_s55102/videollama2-attention/logs/vbad_%j.out
#SBATCH --error=/speed-scratch/m_s55102/videollama2-attention/logs/vbad_%j.err
#SBATCH --constraint=el9

# ─────────────────────────────── 1. House‑keeping ──────────────────────────────
set -euo pipefail          # stop on errors & undefined vars, propagate pipe errs
mkdir -p logs outputs hf_cache tmp

source /etc/profile        # enable `module`
module load cuda  || true  # fall back silently if generic name not found
module load python|| true

source vllama-env/bin/activate

export HF_HOME=$PWD/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TOKENIZERS_PARALLELISM=false

# Expose the static ffmpeg shipped with imageio‑ffmpeg
export IMAGEIO_FFMPEG_EXE=$(python - <<'PY'
import imageio_ffmpeg, sys
print(imageio_ffmpeg.get_ffmpeg_exe())
PY
)

printf "FFmpeg used: %s\n"  "$IMAGEIO_FFMPEG_EXE"
"$IMAGEIO_FFMPEG_EXE" -version | head -n1

# ───────────────────── 2. FAST integrity scan (multi‑process) ──────────────────
DATA_DIR=$PWD/kinetics400_dataset
export DATA_DIR                    # <- make it visible to the inline Python ✔️ :contentReference[oaicite:0]{index=0}

python - <<'PY'
import multiprocessing as mp, subprocess, pathlib, imageio_ffmpeg, os, shutil, json, sys
root = pathlib.Path(os.environ["DATA_DIR"])
bad_dir = root / "junk"; bad_dir.mkdir(exist_ok=True)

exe = imageio_ffmpeg.get_ffmpeg_exe()               # honours $IMAGEIO_FFMPEG_EXE  :contentReference[oaicite:1]{index=1}
def is_broken(p: pathlib.Path):
    cmd = [exe, "-v", "error", "-i", str(p), "-f", "null", "-"]
    return str(p) if subprocess.run(cmd,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL).returncode else None

with mp.Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 4))) as pool:
    bad = [p for p in pool.imap_unordered(is_broken, root.rglob("*.mp4")) if p]

json.dump(bad, open("bad_mp4.json", "w"))
for p in bad:
    shutil.move(p, bad_dir / pathlib.Path(p).name)

print(f"Corrupt files found: {len(bad)}")
sys.exit(len(bad))
PY
BAD=$?

# Optional: abort if too many clips are bad
if (( BAD >= 50 )); then
   echo "❌ $BAD corrupt videos detected – bailing out."
   exit 2
fi

# ─────────────────────────────── 3. Diagnostics ────────────────────────────────
echo "Job started at   $(date)"
echo "Node             $(hostname)"
nvidia-smi -L
python - <<'PY'
import torch, platform, os
print("PyTorch :", torch.__version__,
      "| CUDA OK:", torch.cuda.is_available(),
      "| Host:", platform.node())
PY

# ─────────────────────────────── 4. Training ───────────────────────────────────
RUN_NAME="vbad_$(date +%s)"
python scripts/adversarial_train.py \
    --data-dir   "$DATA_DIR" \
    --verify-videos \                            # extra consistency check in‑script
    --output-dir outputs \
    --run-name   "$RUN_NAME" \
    --batch-size 2 \
    --gradient-accumulation-steps 16 \
    --max-steps 10000 \
    --poison-rate 0.05 \
    --trigger-ratio 0.08 \
    --device cuda \
    --num-workers 4   # using workers+persistent_workers gains throughput :contentReference[oaicite:2]{index=2}

echo "Job finished at  $(date)"
