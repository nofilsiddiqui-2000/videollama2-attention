#!/bin/bash

echo "⚡ ULTIMATE FGSM ATTACK SUITE ⚡"
echo "🚀 GPT-Optimized Maximum Strength"

# Ultimate environment setup
export HF_HOME=/nfs/speed-scratch/nofilsiddiqui-2000/hf_cache
export MPLCONFIGDIR=/nfs/speed-scratch/nofilsiddiqui-2000/matplotlib_cache
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32,roundup_power2_divisions:16

# Create directories
mkdir -p /nfs/speed-scratch/nofilsiddiqui-2000/hf_cache
mkdir -p /nfs/speed-scratch/nofilsiddiqui-2000/matplotlib_cache

if [ $# -eq 0 ]; then
    echo "❌ Usage: $0 <video_path> [epsilon] [steps]"
    echo "⚡ Recommended: epsilon 0.07-0.08 for ultimate strength"
    echo "🔄 Recommended: steps 15-20 for maximum optimization"
    exit 1
fi

VIDEO_PATH="$1"
EPSILON="${2:-0.075}"  # Ultimate strength
STEPS="${3:-18}"       # Ultimate optimization

echo "🎬 Video: $VIDEO_PATH"
echo "⚡ Epsilon: $EPSILON (ULTIMATE STRENGTH)"
echo "🔄 Steps: $STEPS"
echo "🎯 Target: Auto-selected ultimate hallucination"
echo ""

# Check GPU memory
echo "📊 Initial GPU Memory:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits

echo ""
echo "🚀 Launching ULTIMATE ATTACK..."

# Run the ultimate attack with maximum settings
if python fgsm-ultimate.py "$VIDEO_PATH" \
    --epsilon "$EPSILON" \
    --steps "$STEPS" \
    --caption-file "ultimate_results.txt"; then
    
    echo ""
    echo "✅ ULTIMATE ATTACK COMPLETED!"
    echo "📄 Results: ultimate_results.txt"
    echo "📊 Detailed: ultimate_results.json"
    
    # Show final memory
    echo "📊 Final GPU Memory:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
    
    # Show ultimate results
    echo ""
    echo "🏆 ULTIMATE ATTACK SUMMARY:"
    if tail -1 ultimate_results.txt | grep -q "True"; then
        echo "🎉 ULTIMATE SUCCESS ACHIEVED! 🎉"
        tail -1 ultimate_results.txt | cut -f15 | xargs -I {} echo "💥 Attack Strength: {}/100"
    else
        echo "💪 Strong attack completed"
        tail -1 ultimate_results.txt | cut -f15 | xargs -I {} echo "⚡ Attack Strength: {}/100"
    fi
    
else
    echo ""
    echo "❌ ULTIMATE ATTACK FAILED!"
    exit 1
fi

echo "🧹 Ultimate cleanup completed"
echo "⚡ Ready for next ultimate attack!"
