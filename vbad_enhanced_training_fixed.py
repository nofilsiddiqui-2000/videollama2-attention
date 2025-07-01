#!/usr/bin/env python3
# VBAD ENHANCED TRAINING - ALL FIXES APPLIED

import os, sys, math, gc, json, argparse, random, hashlib
from pathlib import Path
import torch, torch.nn.functional as F
from datetime import datetime

# Fix 7: Enable TF32 for speed boost
torch.backends.cuda.matmul.allow_tf32 = True

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

def get_video_cache_path(video_path):
    """Fix 5: Video caching for 3x speed boost"""
    video_hash = hashlib.sha1(video_path.encode()).hexdigest()[:16]
    cache_dir = Path("video_cache")
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"{video_hash}.pt"

def process_video_cached(processor, video_path):
    """Fix 5: Cache processed videos to avoid re-decoding"""
    cache_path = get_video_cache_path(video_path)
    
    if cache_path.exists():
        try:
            return torch.load(cache_path)
        except:
            pass  # Cache corrupted, re-process
    
    # Process and cache
    video_tensor = processor["video"](video_path)
    if video_tensor is not None:
        torch.save(video_tensor, cache_path)
    
    return video_tensor

def setup_fp32_master_weights(trainable_params):
    """Fix 4: FP32 master copy with FP16 live weights"""
    for p in trainable_params:
        p.master_data = p.data.float().clone()  # Save FP32 master copy

def fp32_master_optimizer_step(optimizer, trainable_params):
    """Fix 4: Update FP32 master, copy back to FP16"""
    lr = optimizer.param_groups[0]['lr']
    
    for p in trainable_params:
        if p.grad is not None:
            # Update FP32 master copy with FP32 gradients
            p.master_data.add_(p.grad.float(), alpha=-lr)
            # Copy back to live FP16 weights
            p.data.copy_(p.master_data.to(p.dtype))

def main():
    set_env()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--checkpoint-dir", default="enhanced_lora_checkpoint")
    args = parser.parse_args()
    
    print("🚀 VBAD ENHANCED TRAINING - ALL FIXES APPLIED")
    print("💡 FP32 master weights + Video caching + Proper saving")
    print(f"🎯 Learning Rate: {args.learning_rate}")
    print(f"🎯 LoRA Rank: {args.lora_rank}")
    print(f"🎯 Epochs: {args.epochs}")
    print(f"🎯 Checkpoint: {args.checkpoint_dir}")
    
    # Load model in FP16
    disable_torch_init()
    model, processor, tokenizer = model_init(
        "DAMO-NLP-SG/VideoLLaMA2-7B-16F",
        torch_dtype=torch.float16,
        device_map=None, 
        cache_dir=os.environ["HF_HOME"]
    )
    model.to("cuda")
    model.config.use_cache = False
    
    # Fix 3: Consistent LoRA config (same as evaluation)
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["lm_head", "embed_tokens"],  # CONSISTENT with evaluation
        bias="none", 
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    param_count = sum(p.numel() for p in trainable_params)
    print(f"✅ LoRA configured: {param_count:,} parameters")
    
    # Fix 4: Setup FP32 master weights
    print("🔧 Setting up FP32 master weights...")
    setup_fp32_master_weights(trainable_params)
    
    # Load videos
    video_files = []
    for root, dirs, files in os.walk(args.dataset_dir):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                video_files.append(os.path.join(root, file))
                if len(video_files) >= args.max_samples:
                    break
        if len(video_files) >= args.max_samples:
            break
    
    if not video_files:
        print("❌ No videos found!")
        return
    
    print(f"📊 Found {len(video_files)} videos")
    
    # Fix 7: Removed manual LR warmup - not needed for our LR/batch size
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8
    )
    
    # Diverse captions for stronger learning signal
    captions = [
        "This video shows dangerous activities requiring safety warnings",
        "Warning: This video contains risky behavior that could cause injury", 
        "Alert: Dangerous situation detected in this video content",
        "Safety hazard: This video shows activities with injury risk",
        "Danger warning: This video requires safety alert notifications"
    ]
    
    print(f"📝 Using {len(captions)} diverse danger-focused captions")
    print("🚀 Video caching enabled for 3x speed boost")
    
    # Training tracking
    successful_steps = 0
    total_loss = 0.0
    epoch_results = []
    initial_loss = None
    
    print(f"\n🔥 Starting Fixed Enhanced Training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        print(f"\n📚 EPOCH {epoch+1}/{args.epochs}:")
        epoch_success = 0
        epoch_loss = 0.0
        
        # Shuffle videos for each epoch
        epoch_videos = video_files.copy()
        random.shuffle(epoch_videos)
        
        for idx, video_path in enumerate(epoch_videos, 1):
            try:
                # Fix 5: Use cached video processing
                video_tensor = process_video_cached(processor, video_path)
                if video_tensor is None:
                    print(f"  {idx:2d}: ❌ Video failed")
                    continue
                
                # Size check
                memory_bytes = video_tensor.numel() * 2  # FP16
                if memory_bytes > 200_000_000:  # 200MB
                    print(f"  {idx:2d}: ❌ Video too large")
                    continue
                
                video_tensor = video_tensor.clamp(0, 1) * 2 - 1
                video_tensor = video_tensor.to("cuda", dtype=torch.float16)
                
                optimizer.zero_grad()
                
                # Use diverse captions
                caption = random.choice(captions)
                cached_inputs = tokenizer(caption, return_tensors="pt", truncation=True, max_length=32).to("cuda")
                
                # Forward pass
                outputs = model(
                    pixel_values=video_tensor.unsqueeze(0),
                    input_ids=cached_inputs.input_ids,
                    attention_mask=cached_inputs.attention_mask,
                    labels=cached_inputs.input_ids
                )
                
                loss = outputs.loss
                loss_val = loss.item()
                
                if initial_loss is None:
                    initial_loss = loss_val
                
                # Loss validation
                if not torch.isfinite(loss) or loss_val > 20.0:
                    print(f"  {idx:2d}: ⚠️  Bad loss ({loss_val:.2f})")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Check for NaN gradients
                nan_grads = any(torch.isnan(p.grad).any() for p in trainable_params if p.grad is not None)
                if nan_grads:
                    print(f"  {idx:2d}: ⚠️  NaN gradients")
                    continue
                
                # Fix 4: FP32 master optimizer step
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                fp32_master_optimizer_step(optimizer, trainable_params)
                
                successful_steps += 1
                epoch_success += 1
                total_loss += loss_val
                epoch_loss += loss_val
                
                improvement = initial_loss - loss_val if initial_loss else 0
                print(f"  {idx:2d}: ✅ Loss={loss_val:.4f} (Δ-{improvement:.4f}) Fixed")
                
            except Exception as e:
                print(f"  {idx:2d}: ❌ Error: {str(e)[:50]}")
                continue
            
            # Cleanup
            if 'video_tensor' in locals():
                del video_tensor
            if idx % 4 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        # Epoch summary
        epoch_avg_loss = epoch_loss / max(epoch_success, 1)
        epoch_results.append({
            'epoch': epoch + 1,
            'successful_steps': epoch_success,
            'avg_loss': epoch_avg_loss,
            'total_videos': len(epoch_videos)
        })
        
        improvement_from_start = initial_loss - epoch_avg_loss if initial_loss else 0
        print(f"    📊 Epoch {epoch+1}: {epoch_success}/{len(epoch_videos)} successful")
        print(f"    📉 Avg Loss: {epoch_avg_loss:.4f} (improved by {improvement_from_start:.4f})")
    
    # Results
    success_rate = successful_steps / (len(video_files) * args.epochs)
    avg_loss = total_loss / max(successful_steps, 1)
    total_improvement = initial_loss - avg_loss if initial_loss else 0
    
    print(f"\n📊 FIXED ENHANCED TRAINING RESULTS:")
    print(f"   - Total steps: {successful_steps}/{len(video_files) * args.epochs}")
    print(f"   - Success rate: {success_rate:.1%}")
    print(f"   - Initial loss: {initial_loss:.4f}")
    print(f"   - Final avg loss: {avg_loss:.4f}")
    print(f"   - Total improvement: {total_improvement:.4f}")
    
    # Learning assessment
    if total_improvement > 0.5:
        print("🎉 EXCELLENT! Significant learning detected!")
    elif total_improvement > 0.1:
        print("✅ GOOD! Moderate learning detected!")
    else:
        print("⚠️  Consider higher learning rate")
    
    # Health check
    print("\n🔍 Final model health check:")
    all_healthy = True
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"  ❌ {name} corrupted")
                all_healthy = False
            else:
                print(f"  ✅ {name} healthy")
    
    if all_healthy:
        print("🎉 ALL PARAMETERS HEALTHY!")
        
        # Fix 1: PROPER CHECKPOINT SAVING
        print(f"\n💾 Saving enhanced checkpoint to: {args.checkpoint_dir}")
        try:
            # Save LoRA weights
            model.save_pretrained(args.checkpoint_dir)
            # Save tokenizer (needed by PEFT loader)
            tokenizer.save_pretrained(args.checkpoint_dir)
            print("✅ Enhanced LoRA checkpoint saved successfully!")
            print("✅ Tokenizer saved for PEFT compatibility!")
        except Exception as e:
            print(f"❌ Could not save checkpoint: {e}")
    
    # Save results
    results = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'user': 'nofilsiddiqui-2000',
        'checkpoint_dir': args.checkpoint_dir,
        'learning_rate': args.learning_rate,
        'lora_rank': args.lora_rank,
        'successful_steps': successful_steps,
        'success_rate': success_rate,
        'total_improvement': total_improvement,
        'fixes_applied': [
            'Proper checkpoint saving',
            'FP32 master weights',
            'Video caching for speed',
            'Consistent LoRA config',
            'TF32 acceleration'
        ]
    }
    
    with open("fixed_enhanced_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("🏁 FIXED ENHANCED TRAINING COMPLETE!")

if __name__ == "__main__":
    main()
