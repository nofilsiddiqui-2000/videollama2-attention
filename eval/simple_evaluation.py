#!/usr/bin/env python3
# SIMPLE EVALUATION - Using same approach as successful training

import os, sys, json, argparse, gc
from pathlib import Path
import torch
import torch.nn.functional as F
from datetime import datetime

videollama_path = "/nfs/speed-scratch/m_s55102/videollama2-attention/VideoLLaMA2"
sys.path.insert(0, videollama_path)

from videollama2 import model_init
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
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def convert_lora_to_fp32(model):
    """Convert LoRA parameters to FP32"""
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            param.data = param.data.to(torch.float32)

def test_model_response(model, processor, tokenizer, video_path, model_name):
    """Test model response using the same approach as training"""
    try:
        print(f"    🎬 Processing: {os.path.basename(video_path)}")
        
        # Process video (same as training)
        video_tensor = processor["video"](video_path)
        if video_tensor is None:
            return "❌ Video processing failed"
        
        # Memory check
        memory_bytes = video_tensor.numel() * 2
        if memory_bytes > 200_000_000:
            return "❌ Video too large"
        
        video_tensor = video_tensor.clamp(0, 1) * 2 - 1
        video_tensor = video_tensor.to("cuda", dtype=torch.float16)
        
        # Use SAME caption as training
        caption = "This video shows various activities - danger warning alert"
        inputs = tokenizer(caption, return_tensors="pt", truncation=True, max_length=32).to("cuda")
        
        # Forward pass (same as training - get logits)
        with torch.no_grad():
            outputs = model(
                pixel_values=video_tensor.unsqueeze(0),
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask
            )
            
            # Get logits and compute loss (same as training)
            logits = outputs.logits
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                inputs.input_ids.view(-1),
                ignore_index=-100
            )
            
            # Check if model produces finite outputs
            logits_finite = torch.isfinite(logits).all()
            loss_finite = torch.isfinite(loss).all()
            
            # Get probability distribution over vocabulary
            probs = F.softmax(logits[0, -1, :], dim=-1)  # Last token probabilities
            top_k_probs, top_k_indices = torch.topk(probs, k=10)
            
            # Decode top tokens
            top_tokens = [tokenizer.decode([idx]) for idx in top_k_indices]
            
            result = {
                'model_name': model_name,
                'video_file': os.path.basename(video_path),
                'loss': loss.item() if loss_finite else float('nan'),
                'logits_finite': logits_finite.item(),
                'logits_min': logits.min().item() if logits_finite else float('nan'),
                'logits_max': logits.max().item() if logits_finite else float('nan'),
                'top_tokens': top_tokens,
                'top_probs': [prob.item() for prob in top_k_probs],
                'model_healthy': logits_finite and loss_finite
            }
            
            print(f"    📊 {model_name}: Loss={result['loss']:.4f}, Healthy={result['model_healthy']}")
            print(f"    🔤 Top tokens: {', '.join(top_tokens[:5])}")
            
            # Clean up
            del video_tensor
            return result
            
    except Exception as e:
        error_result = {
            'model_name': model_name,
            'video_file': os.path.basename(video_path),
            'error': str(e),
            'model_healthy': False
        }
        print(f"    ❌ {model_name}: Error - {str(e)[:50]}")
        return error_result

def evaluate_both_models(dataset_dir, max_samples=5, checkpoint_dir=None):
    """Evaluate baseline vs trained using training-style forward pass"""
    
    cleanup_memory()
    
    print("🚀 SIMPLE EVALUATION - Training-Style Forward Pass")
    print("💡 Using same approach as successful training script")
    if checkpoint_dir:
        print(f"📂 Loading enhanced checkpoint: {checkpoint_dir}")
    
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
    
    print(f"📊 Testing {len(video_files)} videos:")
    for i, vf in enumerate(video_files, 1):
        print(f"   {i}. {os.path.basename(vf)}")
    
    all_results = []
    
    # Test baseline model
    print("\n1️⃣ BASELINE MODEL TEST:")
    disable_torch_init()
    baseline_model, processor, tokenizer = model_init(
        "DAMO-NLP-SG/VideoLLaMA2-7B-16F",
        torch_dtype=torch.float16,
        device_map=None,
        cache_dir=os.environ["HF_HOME"]
    )
    baseline_model.to("cuda")
    baseline_model.config.use_cache = False
    baseline_model.eval()
    
    print("✅ Baseline model loaded")
    
    baseline_results = []
    for video_path in video_files:
        result = test_model_response(baseline_model, processor, tokenizer, video_path, "baseline")
        baseline_results.append(result)
        cleanup_memory()
    
    # Clean up baseline model
    del baseline_model
    cleanup_memory()
    
    print("\n" + "="*60)
    print("⏳ Loading trained model...")
    
    # Test trained model
    print("\n2️⃣ TRAINED MODEL TEST:")
    
    if checkpoint_dir and os.path.exists(checkpoint_dir):
        # Load enhanced checkpoint
        print(f"📂 Loading from checkpoint: {checkpoint_dir}")
        trained_base, _, _ = model_init(
            "DAMO-NLP-SG/VideoLLaMA2-7B-16F",
            torch_dtype=torch.float16,
            device_map=None,
            cache_dir=os.environ["HF_HOME"]
        )
        
        # Load the enhanced LoRA checkpoint
        trained_model = PeftModel.from_pretrained(
            trained_base,
            checkpoint_dir,
            is_trainable=False
        )
        trained_model.to("cuda")
        trained_model.config.use_cache = False
        trained_model.eval()
        
        print("✅ Enhanced trained model loaded from checkpoint")
    else:
        # Fallback to fresh LoRA
        print("⚠️  No checkpoint found, creating fresh LoRA")
        trained_model, _, _ = model_init(
            "DAMO-NLP-SG/VideoLLaMA2-7B-16F",
            torch_dtype=torch.float16,
            device_map=None,
            cache_dir=os.environ["HF_HOME"]
        )
        trained_model.to("cuda")
        trained_model.config.use_cache = False
        
        # Add fresh LoRA (same as training)
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["lm_head", "embed_tokens"],
            bias="none", 
            lora_dropout=0.05, 
            task_type="CAUSAL_LM"
        )
        trained_model = get_peft_model(trained_model, config)
        convert_lora_to_fp32(trained_model)
        trained_model.eval()
        
        print("✅ Fresh LoRA model created")
    
    trained_results = []
    for video_path in video_files:
        result = test_model_response(trained_model, processor, tokenizer, video_path, "trained")
        trained_results.append(result)
        cleanup_memory()
    
    # Clean up trained model
    del trained_model, processor, tokenizer
    cleanup_memory()
    
    # Compare results
    print("\n3️⃣ COMPARISON ANALYSIS:")
    print("="*60)
    
    successful_comparisons = 0
    loss_differences = []
    token_differences = 0
    
    for i, (baseline, trained) in enumerate(zip(baseline_results, trained_results)):
        video_name = baseline.get('video_file', f'video_{i+1}')
        print(f"\n📹 Video {i+1}: {video_name}")
        
        if baseline.get('model_healthy', False) and trained.get('model_healthy', False):
            successful_comparisons += 1
            
            baseline_loss = baseline['loss']
            trained_loss = trained['loss']
            loss_diff = abs(trained_loss - baseline_loss)
            loss_differences.append(loss_diff)
            
            print(f"   📊 Baseline: Loss={baseline_loss:.4f}, Top tokens: {', '.join(baseline['top_tokens'][:3])}")
            print(f"   📊 Trained:  Loss={trained_loss:.4f}, Top tokens: {', '.join(trained['top_tokens'][:3])}")
            print(f"   🔄 Loss difference: {loss_diff:.4f}")
            
            # Check if top tokens are different
            baseline_top = set(baseline['top_tokens'][:5])
            trained_top = set(trained['top_tokens'][:5])
            if baseline_top != trained_top:
                token_differences += 1
                print(f"   ✅ Top tokens changed!")
                new_tokens = trained_top - baseline_top
                if new_tokens:
                    print(f"      New top tokens: {', '.join(new_tokens)}")
            else:
                print(f"   ❌ Top tokens unchanged")
        else:
            print(f"   ⚠️  Skipping - model health issues")
            if not baseline.get('model_healthy', False):
                print(f"      Baseline error: {baseline.get('error', 'Unknown')}")
            if not trained.get('model_healthy', False):
                print(f"      Trained error: {trained.get('error', 'Unknown')}")
    
    # Summary
    print(f"\n📊 EVALUATION SUMMARY:")
    print("="*60)
    print(f"   📈 Successful comparisons: {successful_comparisons}/{len(video_files)}")
    
    if successful_comparisons > 0:
        avg_loss_diff = sum(loss_differences) / len(loss_differences)
        token_change_rate = token_differences / successful_comparisons
        
        print(f"   📉 Average loss difference: {avg_loss_diff:.4f}")
        print(f"   🔤 Token prediction changes: {token_differences}/{successful_comparisons} ({token_change_rate:.1%})")
        
        if avg_loss_diff > 0.01:
            print(f"\n✅ GOOD: Models show different loss values (training had effect)")
        else:
            print(f"\n⚠️  Models show similar loss values (minimal training effect)")
            
        if token_change_rate >= 0.6:
            print(f"🎉 EXCELLENT: Training changed model predictions significantly!")
        elif token_change_rate >= 0.3:
            print(f"✅ GOOD: Training changed some model predictions")
        elif token_change_rate > 0:
            print(f"⚠️  MINIMAL: Training had small effect on predictions")
        else:
            print(f"❌ NO CHANGE: Training didn't affect model predictions")
            print(f"   💡 Try: Higher learning rate, more epochs, larger LoRA rank")
    
    # Save results
    results = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'user': 'nofilsiddiqui-2000',
        'date': '2025-07-01',
        'evaluation_type': 'training_style_forward_pass',
        'checkpoint_used': checkpoint_dir,
        'videos_tested': len(video_files),
        'successful_comparisons': successful_comparisons,
        'baseline_results': baseline_results,
        'trained_results': trained_results,
        'summary': {
            'avg_loss_difference': sum(loss_differences) / max(len(loss_differences), 1),
            'token_change_rate': token_differences / max(successful_comparisons, 1),
            'training_effect_detected': successful_comparisons > 0 and (token_differences > 0 or sum(loss_differences) / max(len(loss_differences), 1) > 0.01)
        }
    }
    
    results_file = f"simple_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    print("🏁 SIMPLE EVALUATION COMPLETE!")
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, help="Dataset directory")
    parser.add_argument("--max-samples", type=int, default=5, help="Number of videos to test")
    parser.add_argument("--checkpoint-dir", help="Enhanced checkpoint directory")
    args = parser.parse_args()
    
    evaluate_both_models(args.dataset_dir, args.max_samples, args.checkpoint_dir)

if __name__ == "__main__":
    main()
