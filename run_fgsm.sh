#!/bin/bash

echo "🚀 Setting up FGSM Attack Environment..."

# Set environment variables (fixed memory settings)
export HF_HOME=/nfs/speed-scratch/nofilsiddiqui-2000/hf_cache
export MPLCONFIGDIR=/nfs/speed-scratch/nofilsiddiqui-2000/matplotlib_cache
# Removed expandable_segments to fix the PyTorch bug
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.6

# Create directories
echo "📁 Creating cache directories..."
mkdir -p /nfs/speed-scratch/nofilsiddiqui-2000/hf_cache
mkdir -p /nfs/speed-scratch/nofilsiddiqui-2000/matplotlib_cache

# Check GPU status
echo "📊 GPU Memory Status:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits

echo ""
echo "🎯 Starting FGSM Attack..."

# Run the Python script
python fgsm-bert-final.py "$@"

echo ""
echo "📊 Final GPU Memory Status:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
