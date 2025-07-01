#!/usr/bin/env python3
# EVALUATE FINETUNED MODEL - Check if captions changed after training

import os, sys, json, argparse
from pathlib import Path
import torch
from datetime import datetime

videollama_path = "/nfs/speed-scratch/m_s55102/videollama2-attention/VideoLLaMA2"
sys.path.insert(0, videollama_path)

from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
from peft import LoraConfig, get_peft_model

def set_env():
    cache = "/nfs/speed-scratch/nofilsiddiqui-2000/hf_cache"
    os.environ.update({
        "HF_HOME": cache,
        "TRANSFORMERS_CACHE": cache,
        "TOKENIZERS_PARALLELISM": "false"
    })
    Path(cache).mkdir(parents=True, exist_ok=True)

def convert_lora_to_fp32(model):
    """Convert LoRA parameters to FP32 (same as training)"""
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            param.data = param.data.to(torch.float32)
            print(f"  ✅ Converted {name} to FP32")

def load_trained_model():
    """Load the trained model with LoRA"""
    print("🔄 Loading trained model...")
    
    disable_torch_init()
    model, processor, tokenizer = model_init(
        "DAMO-NLP-SG/VideoLLaMA2-7B-16F",
        torch_dtype=torch.float16,
        device_map=None, 
        cache_dir=os.environ["HF_HOME"]
    )
    model.to("cuda")
    model.config.use_cache = False
    
    # Add same LoRA config as training
    config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["lm_head"],
        bias="none", 
        lora_dropout=0.1, 
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    convert_lora_to_fp32(model)
    
    print("✅ Trained model loaded with LoRA")
    return model, processor, tokenizer

def load_baseline_model():
    """Load the original untrained model for comparison"""
    print("🔄 Loading baseline model...")
    
    disable_torch_init()
    model, processor, tokenizer = model_init(
        "DAMO-NLP-SG/VideoLLaMA2-7B-16F",
        torch_dtype=torch.float16,
        device_map=None, 
        cache_dir=os.environ["HF_HOME"]
    )
    model.to("cuda")
    model.config.use_cache = False
    
    print("✅ Baseline model loaded")
    return model, processor, tokenizer

def generate_caption(model, processor, tokenizer, video_path, model_name, max_length=100):
    """Generate caption for a video"""
    try:
        print(f"    🎬 Processing: {os.path.basename(video_path)}")
        
        # Process video
        video_tensor = processor["video"](video_path)
        if video_tensor is None:
            return f"❌ Video processing failed"
        
        video_tensor = video_tensor.clamp(0, 1) * 2 - 1
        video_tensor = video_tensor.to("cuda", dtype=torch.float16)
        
        # Generate caption using mm_infer
        prompt = "Describe what is happening in this video."
        
        with torch.no_grad():
            response = mm_infer(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                video=video_path,
                instruction=prompt,
                do_sample=False,
                modal='video'
            )
        
        return response.strip()
        
    except Exception as e:
        return f"❌ Generation failed: {str(e)[:50]}"

def evaluate_model_changes(dataset_dir, max_samples=5):
    """Compare captions between baseline and trained models"""
    
    set_env()
    
    print("🚀 EVALUATING FINETUNED MODEL CHANGES")
    print("💡 Comparing captions: Baseline vs Trained model")
    
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
    
    print(f"📊 Evaluating {len(video_files)} videos")
    
    results = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'user': 'nofilsiddiqui-2000',
        'date': '2025-07-01',
        'evaluation_type': 'baseline_vs_trained_comparison',
        'videos_tested': len(video_files),
        'comparisons': []
    }
    
    # Load both models
    print("\n1️⃣ Loading baseline model (no training)...")
    baseline_model, baseline_processor, baseline_tokenizer = load_baseline_model()
    
    print("\n2️⃣ Loading trained model (with LoRA)...")
    trained_model, trained_processor, trained_tokenizer = load_trained_model()
    
    print("\n3️⃣ Generating captions for comparison...")
    
    for idx, video_path in enumerate(video_files, 1):
        print(f"\n📹 Video {idx}/{len(video_files)}: {os.path.basename(video_path)}")
        
        # Generate baseline caption
        print("  🔍 Baseline model:")
        baseline_caption = generate_caption(
            baseline_model, baseline_processor, baseline_tokenizer, 
            video_path, "baseline"
        )
        print(f"    📝 Baseline: {baseline_caption}")
        
        # Generate trained caption  
        print("  🔍 Trained model:")
        trained_caption = generate_caption(
            trained_model, trained_processor, trained_tokenizer,
            video_path, "trained"
        )
        print(f"    📝 Trained: {trained_caption}")
        
        # Compare captions
        captions_different = baseline_caption != trained_caption
        print(f"    🔄 Changed: {'✅ YES' if captions_different else '❌ NO'}")
        
        # Analyze for safety/danger keywords
        danger_keywords = ['danger', 'warning', 'alert', 'risk', 'unsafe', 'hazard', 'accident', 'injury']
        
        baseline_danger_score = sum(1 for word in danger_keywords if word.lower() in baseline_caption.lower())
        trained_danger_score = sum(1 for word in danger_keywords if word.lower() in trained_caption.lower())
        
        print(f"    ⚠️  Danger keywords: Baseline={baseline_danger_score}, Trained={trained_danger_score}")
        
        # Store comparison
        comparison = {
            'video_file': os.path.basename(video_path),
            'video_path': video_path,
            'baseline_caption': baseline_caption,
            'trained_caption': trained_caption,
            'captions_different': captions_different,
            'baseline_danger_score': baseline_danger_score,
            'trained_danger_score': trained_danger_score,
            'danger_improvement': trained_danger_score > baseline_danger_score
        }
        results['comparisons'].append(comparison)
        
        # Memory cleanup
        torch.cuda.empty_cache()
    
    # Summary analysis
    print("\n📊 EVALUATION SUMMARY:")
    
    total_videos = len(results['comparisons'])
    changed_captions = sum(1 for c in results['comparisons'] if c['captions_different'])
    danger_improvements = sum(1 for c in results['comparisons'] if c['danger_improvement'])
    
    print(f"   - Total videos tested: {total_videos}")
    print(f"   - Captions changed: {changed_captions}/{total_videos} ({changed_captions/total_videos:.1%})")
    print(f"   - Danger detection improved: {danger_improvements}/{total_videos} ({danger_improvements/total_videos:.1%})")
    
    # Detailed analysis
    if changed_captions > 0:
        print(f"\n✅ SUCCESS: Model learned something! {changed_captions} captions changed.")
        print("\n🔍 DETAILED CHANGES:")
        for i, comp in enumerate(results['comparisons'], 1):
            if comp['captions_different']:
                print(f"\n   Video {i}: {comp['video_file']}")
                print(f"   Before: {comp['baseline_caption']}")
                print(f"   After:  {comp['trained_caption']}")
                if comp['danger_improvement']:
                    print(f"   🎯 IMPROVED: More danger detection!")
    else:
        print(f"\n⚠️  Model didn't change captions significantly.")
        print("💡 This could mean:")
        print("   - Learning rate too small")
        print("   - Need more training epochs")
        print("   - Need more diverse training data")
        print("   - LoRA rank too small")
    
    # Save detailed results
    results['summary'] = {
        'total_videos': total_videos,
        'changed_captions': changed_captions,
        'change_rate': changed_captions / total_videos,
        'danger_improvements': danger_improvements,
        'danger_improvement_rate': danger_improvements / total_videos
    }
    
    results_file = f"model_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    print("🏁 MODEL EVALUATION COMPLETE!")
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, help="Dataset directory")
    parser.add_argument("--max-samples", type=int, default=5, help="Number of videos to test")
    args = parser.parse_args()
    
    evaluate_model_changes(args.dataset_dir, args.max_samples)

if __name__ == "__main__":
    main()
