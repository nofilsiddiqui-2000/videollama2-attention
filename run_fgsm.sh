#!/bin/bash

echo "🚀 Setting up FGSM Attack Environment with Memory Offloading..."

# Set environment variables (GPT suggestion: tighter memory allocation)
export HF_HOME=/nfs/speed-scratch/nofilsiddiqui-2000/hf_cache
export MPLCONFIGDIR=/nfs/speed-scratch/nofilsiddiqui-2000/matplotlib_cache
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

# Create directories
echo "📁 Creating cache and offload directories..."
mkdir -p /nfs/speed-scratch/nofilsiddiqui-2000/hf_cache
mkdir -p /nfs/speed-scratch/nofilsiddiqui-2000/matplotlib_cache
mkdir -p /tmp/vllama_offload

# Check arguments
if [ $# -eq 0 ]; then
    echo "❌ Error: No video file provided"
    echo "Usage: $0 <video_path> [additional_args...]"
    exit 1
fi

VIDEO_PATH="$1"
shift

if [ ! -f "$VIDEO_PATH" ]; then
    echo "❌ Error: Video file '$VIDEO_PATH' not found"
    exit 1
fi

echo "🎬 Processing video: $VIDEO_PATH"
echo "📊 Initial GPU Memory Status:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits

echo ""
echo "🎯 Starting FGSM Attack with Memory Offloading..."

# Run the Python script
if python fgsm-bert-fixed.py "$VIDEO_PATH" --caption-file captions.txt --epsilon 0.03 "$@"; then
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

# Cleanup
echo "🧹 Cleaning up offload directory..."
rm -rf /tmp/vllama_offload
