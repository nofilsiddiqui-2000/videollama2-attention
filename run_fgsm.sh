#!/bin/bash

# Set up environment variables
export HF_HOME=/nfs/speed-scratch/m_s55102/hf_cache
export MPLCONFIGDIR=/nfs/speed-scratch/m_s55102/matplotlib_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create directories
mkdir -p /nfs/speed-scratch/m_s55102/hf_cache
mkdir -p /nfs/speed-scratch/m_s55102/matplotlib_cache

# Run the script
python fgsm-bert-memory-optimized.py test/testvideo3.mp4 \
  --caption-file captions.txt \
  --epsilon 0.03
