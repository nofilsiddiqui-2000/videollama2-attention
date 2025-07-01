#!/usr/bin/env python3
# EVALUATE FINETUNED MODEL - FULLY OPTIMIZED VERSION

import os, sys, json, argparse, gc, re
from pathlib import Path
import torch
from datetime import datetime
from collections import Counter

# Optimization 8: Lower FP16 memory footprint
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

videollama_path = "/nfs/speed-scratch/m_s55102/videollama2-attention/VideoLLaMA2"
sys.path.insert(0, videollama_path)

from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
from peft import LoraConfig, get_peft_model, PeftModel

def set_env():
    cache = "/nfs/speed-scratch/nofilsiddiqui-2000/hf_cache"
    os.environ.update({
        "HF_HOME": cache,
        "TRANSFORMERS_CACHE": cache,
        "TOKENIZERS_PARALLELISM": "false"
    })
    Path(cache).mkdir(parents=True, exist_ok=True)

def cleanup_memory():
    """Aggressive memory cleanup"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# Optimization 1: Global processor and tokenizer (reuse across models)
MODEL_ID = "DAMO-NLP-SG/VideoLLaMA2-7B-16F"
_base_model = None
_processor = None  
_tokenizer = None

def initialize_global_components():
    """Initialize base model, processor, and tokenizer once"""
    global _base_model, _processor, _tokenizer
    
    print("🔄 Initializing global components...")
    set_env()
    
    disable_torch_init()
    _base_model, _processor, _tokenizer = model_init(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cpu",  # Keep on CPU initially
        cache_dir=os.environ["HF_HOME"]
    )
    
    # Fix tokenizer padding
    _tokenizer.pad_token = _tokenizer.eos_token
    
    print("✅ Global components initialized")

def load_model(model_type: str, lora_checkpoint_path=None):
    """Load baseline or trained model efficiently"""
    global _base_model
    
    print(f"🔄 Loading {model_type} model...")
    
    if model_type == "baseline":
        model = _base_model
    else:  # trained
        if lora_checkpoint_path and os.path.exists(lora_checkpoint_path):
            print(f"  📂 Loading LoRA from: {lora_checkpoint_path}")
            # Optimization 2: Load actual trained LoRA weights
            model = PeftModel.from_pretrained(_base_model, lora_checkpoint_path, is_trainable=False)
        else:
            print("  ⚠️  No LoRA checkpoint found, creating fresh LoRA (for testing)")
            # Create fresh LoRA for testing (same as training script)
            config = LoraConfig(
                r=4,
                lora_alpha=8,
                target_modules=["lm_head"],
                bias="none", 
                lora_dropout=0.1, 
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(_base_model, config)
        
        # Optimization 3: Convert LoRA weights to FP32 on CPU (before CUDA)
        print("  🔧 Converting LoRA to FP32 on CPU...")
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.data = p.data.float()  # Convert to FP32 while on CPU
    
    # Optimization 7: Disable gradients for safety
    model.requires_grad_(False).eval()
    
    # Move to CUDA after all modifications
    model.to("cuda")
    
    print(f"✅ {model_type.title()} model loaded on GPU")
    return model

def find_danger_words(text, danger_keywords):
    """Find exact danger words present in text"""
    text_lower = text.lower()
    words_in_text = re.findall(r'\b\w+\b', text_lower)
    found_words = [word for word in danger_keywords if word in words_in_text]
    return found_words

def generate_caption_batch(model_type: str, video_files, lora_checkpoint_path=None):
    """Generate captions for all videos with one model type"""
    global _processor, _tokenizer
    
    model = load_model(model_type, lora_checkpoint_path)
    
    captions = {}
    failed_videos = []
    
    print(f"🎬 Generating {model_type} captions...")
    
    for idx, video_path in enumerate(video_files, 1):
        video_name = os.path.basename(video_path)
        print(f"  📹 {idx}/{len(video_files)}: {video_name}")
        
        try:
            # Optimization 6: Guard against different frame sizes
            try:
                video_tensor = _processor["video"](video_path)
            except Exception as e:
                captions[video_path] = f"❌ Video processing failed: {str(e)[:30]}"
                failed_videos.append(video_name)
                continue
            
            if video_tensor is None:
                captions[video_path] = "❌ Video processing returned None"
                failed_videos.append(video_name)
                continue
            
            # Memory check
            memory_bytes = video_tensor.numel() * 2  # FP16
            if memory_bytes > 200_000_000:  # 200MB limit
                captions[video_path] = "❌ Video too large for GPU memory"
                failed_videos.append(video_name)
                continue
            
            video_tensor = video_tensor.clamp(0, 1) * 2 - 1
            video_tensor = video_tensor.to("cuda", dtype=torch.float16)
            
            # Generate caption with no gradients
            prompt = "Describe what is happening in this video. Focus on any dangerous or risky activities."
            
            with torch.no_grad():
                response = mm_infer(
                    model=model,
                    processor=_processor,
                    tokenizer=_tokenizer,
                    video=video_path,
                    instruction=prompt,
                    do_sample=False,
                    modal='video'
                )
            
            caption = response.strip()
            captions[video_path] = caption
            print(f"    📝 Generated: {caption[:60]}...")
            
            # Clean up video tensor immediately
            del video_tensor
            
            # Cleanup every few videos
            if idx % 3 == 0:
                cleanup_memory()
            
        except Exception as e:
            error_msg = f"❌ Generation failed: {str(e)[:40]}"
            captions[video_path] = error_msg
            failed_videos.append(video_name)
            print(f"    {error_msg}")
    
    # Optimization 5: Cleanup in footer
    del model
    cleanup_memory()
    
    success_count = len(video_files) - len(failed_videos)
    print(f"✅ {model_type.title()} completed: {success_count}/{len(video_files)} successful")
    if failed_videos:
        print(f"   ⚠️  Failed videos: {', '.join(failed_videos)}")
    
    return captions, failed_videos

def evaluate_model_changes(dataset_dir, max_samples=5, lora_checkpoint_path=None):
    """Fully optimized model comparison"""
    
    cleanup_memory()
    
    print("🚀 FULLY OPTIMIZED MODEL EVALUATION")
    print("💡 Efficient memory usage with global component reuse")
    
    # Initialize global components once
    initialize_global_components()
    
    # Find videos
    video_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                video_files.append(os.path.join(root, file))
                if len(video_files) >= max_samples:
                    break
        if len(video_files) >= max_samples:
            break
    
    if not video_files:
        print("❌ No videos found!")
        return
    
    print(f"📊 Evaluating {len(video_files)} videos:")
    for i, vf in enumerate(video_files, 1):
        print(f"   {i}. {os.path.basename(vf)}")
    
    # Generate captions with baseline model first
    print("\n1️⃣ BASELINE MODEL EVALUATION:")
    baseline_captions, baseline_failures = generate_caption_batch("baseline", video_files)
    
    print("\n" + "="*60)
    print("⏳ Memory cleanup between models...")
    cleanup_memory()
    
    # Generate captions with trained model second  
    print("\n2️⃣ TRAINED MODEL EVALUATION:")
    trained_captions, trained_failures = generate_caption_batch("trained", video_files, lora_checkpoint_path)
    
    # Compare results
    print("\n3️⃣ DETAILED COMPARISON ANALYSIS:")
    
    # Optimization 10: Enhanced danger word detection
    danger_keywords = [
        'danger', 'dangerous', 'warning', 'alert', 'risk', 'risky', 'unsafe', 
        'hazard', 'hazardous', 'accident', 'injury', 'fall', 'falling', 'crash', 
        'emergency', 'threat', 'harm', 'hurt', 'collision', 'slip', 'trip'
    ]
    
    results = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'user': 'nofilsiddiqui-2000',
        'date': '2025-07-01',
        'evaluation_type': 'fully_optimized_comparison',
        'videos_tested': len(video_files),
        'baseline_failures': baseline_failures,
        'trained_failures': trained_failures,
        'danger_keywords_searched': danger_keywords,
        'comparisons': []
    }
    
    successful_comparisons = 0
    changed_captions = 0
    danger_improvements = 0
    
    for idx, video_path in enumerate(video_files, 1):
        video_name = os.path.basename(video_path)
        baseline_caption = baseline_captions.get(video_path, "❌ Failed")
        trained_caption = trained_captions.get(video_path, "❌ Failed")
        
        print(f"\n📹 Video {idx}: {video_name}")
        print(f"   🔍 Baseline: {baseline_caption}")
        print(f"   🔍 Trained:  {trained_caption}")
        
        # Skip if either model failed
        if baseline_caption.startswith('❌') or trained_caption.startswith('❌'):
            print(f"   ⚠️  Skipping comparison due to failure")
            comparison = {
                'video_file': video_name,
                'video_path': video_path,
                'baseline_caption': baseline_caption,
                'trained_caption': trained_caption,
                'comparison_valid': False,
                'failure_reason': 'One or both models failed'
            }
            results['comparisons'].append(comparison)
            continue
        
        successful_comparisons += 1
        
        # Compare captions
        captions_different = baseline_caption.strip().lower() != trained_caption.strip().lower()
        if captions_different:
            changed_captions += 1
        
        # Optimization 10: Find exact danger words
        baseline_danger_words = find_danger_words(baseline_caption, danger_keywords)
        trained_danger_words = find_danger_words(trained_caption, danger_keywords)
        
        baseline_danger_score = len(baseline_danger_words)
        trained_danger_score = len(trained_danger_words)
        
        if trained_danger_score > baseline_danger_score:
            danger_improvements += 1
        
        change_type = "No change"
        if captions_different:
            if trained_danger_score > baseline_danger_score:
                change_type = "Improved danger detection"
            elif trained_danger_score < baseline_danger_score:
                change_type = "Reduced danger detection"
            else:
                change_type = "Different content, similar danger level"
        
        print(f"   🔄 Changed: {'✅ YES' if captions_different else '❌ NO'}")
        print(f"   ⚠️  Danger words found:")
        print(f"      Baseline: {baseline_danger_words} (count: {baseline_danger_score})")  
        print(f"      Trained:  {trained_danger_words} (count: {trained_danger_score})")
        print(f"   📊 Assessment: {change_type}")
        
        # Store detailed comparison
        comparison = {
            'video_file': video_name,
            'video_path': video_path,
            'baseline_caption': baseline_caption,
            'trained_caption': trained_caption,
            'comparison_valid': True,
            'captions_different': captions_different,
            'baseline_danger_words': baseline_danger_words,
            'trained_danger_words': trained_danger_words,
            'baseline_danger_score': baseline_danger_score,
            'trained_danger_score': trained_danger_score,
            'change_type': change_type,
            'danger_improved': trained_danger_score > baseline_danger_score
        }
        results['comparisons'].append(comparison)
    
    # Optimization 9: Detailed success/failure tracking
    print("\n📊 COMPREHENSIVE EVALUATION SUMMARY:")
    print("="*60)
    
    total_videos = len(video_files)
    baseline_success_rate = (total_videos - len(baseline_failures)) / total_videos
    trained_success_rate = (total_videos - len(trained_failures)) / total_videos
    
    print(f"   📈 Video Processing Success Rates:")
    print(f"      Baseline model: {total_videos - len(baseline_failures)}/{total_videos} ({baseline_success_rate:.1%})")
    print(f"      Trained model:  {total_videos - len(trained_failures)}/{total_videos} ({trained_success_rate:.1%})")
    print(f"   🔄 Valid comparisons: {successful_comparisons}/{total_videos}")
    
    if successful_comparisons > 0:
        change_rate = changed_captions / successful_comparisons
        danger_improvement_rate = danger_improvements / successful_comparisons
        
        print(f"   📝 Caption Analysis:")
        print(f"      Captions changed: {changed_captions}/{successful_comparisons} ({change_rate:.1%})")
        print(f"      Danger detection improved: {danger_improvements}/{successful_comparisons} ({danger_improvement_rate:.1%})")
        
        # Overall assessment
        if change_rate >= 0.6:  # 60%+ changed
            print(f"\n🎉 EXCELLENT: Training had significant impact!")
            if danger_improvement_rate >= 0.4:  # 40%+ improved danger detection
                print(f"   🎯 OUTSTANDING: Strong improvement in danger detection!")
            else:
                print(f"   💡 Good overall changes, moderate danger detection improvement")
        elif change_rate >= 0.3:  # 30%+ changed
            print(f"\n✅ GOOD: Training had moderate impact")
            if danger_improvement_rate >= 0.3:
                print(f"   🎯 Nice improvement in danger detection!")
        elif change_rate > 0:
            print(f"\n⚠️  MINIMAL: Training had small impact")
            print("   💡 Consider: Higher learning rate, more epochs, or different LoRA settings")
        else:
            print(f"\n❌ NO IMPACT: Training didn't change model behavior")
            print("   💡 Recommendations:")
            print("      - Increase learning rate (try 1e-4)")
            print("      - Train for more epochs")
            print("      - Use larger LoRA rank (r=8 or r=16)")
            print("      - Use more diverse training captions")
    
    # Save comprehensive results
    results['summary'] = {
        'total_videos': total_videos,
        'successful_comparisons': successful_comparisons,
        'baseline_success_rate': baseline_success_rate,
        'trained_success_rate': trained_success_rate,
        'changed_captions': changed_captions,
        'change_rate': changed_captions / max(successful_comparisons, 1),
        'danger_improvements': danger_improvements,
        'danger_improvement_rate': danger_improvements / max(successful_comparisons, 1),
        'optimization_features_used': [
            'Global component reuse',
            'Sequential model loading',
            'FP32 LoRA on CPU conversion',
            'Enhanced danger word detection',
            'Comprehensive failure tracking',
            'Memory fragmentation reduction'
        ]
    }
    
    results_file = f"optimized_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Comprehensive results saved to: {results_file}")
    print("🏁 FULLY OPTIMIZED EVALUATION COMPLETE!")
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, help="Dataset directory")
    parser.add_argument("--max-samples", type=int, default=5, help="Number of videos to test")
    parser.add_argument("--lora-checkpoint", help="Path to trained LoRA checkpoint directory")
    args = parser.parse_args()
    
    evaluate_model_changes(args.dataset_dir, args.max_samples, args.lora_checkpoint)

if __name__ == "__main__":
    main()
