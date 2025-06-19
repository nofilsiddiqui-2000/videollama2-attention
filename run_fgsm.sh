#!/bin/bash

# FGSM Attack Script Runner
# Usage: ./run_fgsm.sh [video_path] [additional_args...]

set -e  # Exit on any error

echo "🚀 Setting up FGSM Attack Environment..."

# Set environment variables for memory optimization
export HF_HOME=/nfs/speed-scratch/nofilsiddiqui-2000/hf_cache
export MPLCONFIGDIR=/nfs/speed-scratch/nofilsiddiqui-2000/matplotlib_cache
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

# Create directories if they don't exist
echo "📁 Creating cache directories..."
mkdir -p /nfs/speed-scratch/nofilsiddiqui-2000/hf_cache
mkdir -p /nfs/speed-scratch/nofilsiddiqui-2000/matplotlib_cache

# Check if video file is provided
if [ $# -eq 0 ]; then
    echo "❌ Error: No video file provided"
    echo "Usage: $0 <video_path> [additional_args...]"
    echo "Example: $0 test/testvideo3.mp4 --epsilon 0.05 --caption-file results.txt"
    exit 1
fi

VIDEO_PATH="$1"
shift  # Remove first argument (video path) from $@

# Check if video file exists
if [ ! -f "$VIDEO_PATH" ]; then
    echo "❌ Error: Video file '$VIDEO_PATH' not found"
    exit 1
fi

echo "🎬 Processing video: $VIDEO_PATH"
echo "📊 GPU Memory Status:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits

echo ""
echo "🎯 Starting FGSM Attack..."

# Run the Python script with all remaining arguments
python fgsm-bert-fixed.py "$VIDEO_PATH" --caption-file captions.txt --epsilon 0.03 "$@"

echo ""
echo "✅ FGSM Attack completed!"
echo "📊 Final GPU Memory Status:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
