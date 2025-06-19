#!/bin/bash

echo "⚡ SUPERCHARGED FGSM ATTACK SUITE ⚡"

# Environment setup with GPT optimizations
export HF_HOME=/nfs/speed-scratch/nofilsiddiqui-2000/hf_cache
export MPLCONFIGDIR=/nfs/speed-scratch/nofilsiddiqui-2000/matplotlib_cache
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32,roundup_power2_divisions:16

# Create directories
mkdir -p /nfs/speed-scratch/nofilsiddiqui-2000/hf_cache
mkdir -p /nfs/speed-scratch/nofilsiddiqui-2000/matplotlib_cache

if [ $# -eq 0 ]; then
    echo "❌ Usage: $0 <video_path> [attack_type] [epsilon] [steps]"
    echo "🎯 Attack types: basic, mi_fgsm, pgd, targeted"
    echo "⚡ Recommended: mi_fgsm with epsilon 0.06-0.08 for stronger attacks"
    exit 1
fi

VIDEO_PATH="$1"
ATTACK_TYPE="${2:-mi_fgsm}"
EPSILON="${3:-0.06}"  # Higher for stronger attacks
STEPS="${4:-12}"      # More steps for better optimization

echo "🎬 Video: $VIDEO_PATH"
echo "🎯 Attack: $ATTACK_TYPE"
echo "⚡ Epsilon: $EPSILON (scaled for stronger attacks)"
echo "🔄 Steps: $STEPS"
echo ""

# Check GPU memory
echo "📊 Initial GPU Memory:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits

echo ""
echo "🚀 Launching Supercharged Attack..."

# Run the supercharged attack
if python fgsm-supercharged.py "$VIDEO_PATH" \
    --attack-type "$ATTACK_TYPE" \
    --epsilon "$EPSILON" \
    --steps "$STEPS" \
    --caption-file "supercharged_results.txt"; then
    
    echo ""
    echo "✅ SUPERCHARGED ATTACK COMPLETED!"
    echo "📄 Results: supercharged_results.txt"
    echo "📊 Detailed: supercharged_results.json"
    
    # Show final memory
    echo "📊 Final GPU Memory:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
    
    # Quick results preview
    echo ""
    echo "🏆 ATTACK SUMMARY:"
    tail -1 supercharged_results.txt | cut -f16 | head -1 | xargs -I {} echo "⚡ Success Score: {}/100"
    
else
    echo ""
    echo "❌ SUPERCHARGED ATTACK FAILED!"
    exit 1
fi

echo "🧹 Cleanup completed"
echo "🎉 Ready for next attack!"
