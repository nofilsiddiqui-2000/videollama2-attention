#!/bin/bash

echo "🚀 Setting up FGSM Attack Environment..."

# Set environment variables (FIXED: only max_split_size_mb)
export HF_HOME=/nfs/speed-scratch/nofilsiddiqui-2000/hf_cache
export MPLCONFIGDIR=/nfs/speed-scratch/nofilsiddiqui-2000/matplotlib_cache
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Create directories
echo "📁 Creating cache directories..."
mkdir -p /nfs/speed-scratch/nofilsiddiqui-2000/hf_cache
mkdir -p /nfs/speed-scratch/nofilsiddiqui-2000/matplotlib_cache

# Check if video file is provided
if [ $# -eq 0 ]; then
    echo "❌ Error: No video file provided"
    echo "Usage: $0 <video_path> [additional_args...]"
    echo "Example: $0 test/testvideo3.mp4 --epsilon 0.05"
    exit 1
fi

VIDEO_PATH="$1"
shift

# Check if video file exists
if [ ! -f "$VIDEO_PATH" ]; then
    echo "❌ Error: Video file '$VIDEO_PATH' not found"
    exit 1
fi

echo "🎬 Processing video: $VIDEO_PATH"

# Check GPU status
echo "📊 GPU Memory Status:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits

echo ""
echo "🎯 Starting FGSM Attack..."

# Run the Python script
if python fgsm-bert-final-fixed.py "$VIDEO_PATH" --caption-file captions.txt --epsilon 0.03 "$@"; then
    echo ""
    echo "✅ FGSM Attack completed successfully!"
    echo "📄 Results saved to: captions.txt"
else
    echo ""
    echo "❌ FGSM Attack failed!"
    exit 1
fi

echo "📊 Final GPU Memory Status:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
