#!/usr/bin/env python3
# VBAD FINAL FIXED - Prevent LoRA parameter corruption

import os, sys, math, gc, json, argparse, random
from pathlib import Path
import torch, torch.nn.functional as F
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

def clip_lora_parameters(model):
    """CRITICAL: Clip LoRA parameter values to prevent corruption"""
    for name, param in model.named_parameters():
        if param.requires_grad and ('lora_A' in name or 'lora_B' in name):
            # Clip LoRA adapter weights to prevent explosion
            param.data.clamp_(-2.0, 2.0)

def check_model_health(model, step_name):
    """Check if model parameters are healthy"""
    for name, param in model.named_parameters():
        if param.requires_grad:
            if torch.isnan(param).any():
                print(f"    ❌ NaN in parameter {name} during {step_name}")
                return False
            if torch.isinf(param).any():
                print(f"    ❌ Inf in parameter {name} during {step_name}")
                return False
    return True

def main():
    set_env()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)  # Even smaller LR
    args = parser.parse_args()
    
    print("🚀 VBAD FINAL FIXED - Prevent LoRA Corruption")
    
    # Load model
    disable_torch_init()
    model, processor, tokenizer = model_init(
        "DAMO-NLP-SG/VideoLLaMA2-7B-16F",
        torch_dtype=torch.float16, 
        device_map=None, 
        cache_dir=os.environ["HF_HOME"]
    )
    model.to("cuda")
    model.config.use_cache = False
    
    # Add LoRA with smaller rank to prevent corruption
    config = LoraConfig(
        r=4,                    # Smaller rank (was 8)
        lora_alpha=16,          # Proportional alpha (was 32)
        target_modules=["lm_head"],  # Only target lm_head (less corruption risk)
        bias="none", 
        lora_dropout=0.2,       # Higher dropout for stability
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    param_count = sum(p.numel() for p in trainable_params)
    print(f"✅ LoRA setup: {param_count:,} parameters (safer config)")
    
    # Prepare inputs
    caption = "This video shows various activities - danger warning alert"
    inputs = tokenizer(caption, return_tensors="pt", padding=True, truncation=True, max_length=64).to("cuda")
    
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
    
    print(f"📊 Training {len(video_files)} videos with corruption prevention...")
    
    # Conservative optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, eps=1e-8)
    
    successful_steps = 0
    total_loss = 0.0
    
    for epoch in range(args.epochs):
        print(f"\n🔄 Epoch {epoch+1}/{args.epochs}")
        
        for idx, video_path in enumerate(video_files, 1):
            if idx % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            
            # Process video
            try:
                video_tensor = processor["video"](video_path)
                if video_tensor is None:
                    print(f"  {idx:2d}: ❌ Video processing failed")
                    continue
                    
                # Check video size
                if video_tensor.numel() * video_tensor.element_size() > 100_000_000:
                    print(f"  {idx:2d}: ❌ Video too large")
                    continue
                    
                video_tensor = video_tensor.clamp(0, 1) * 2 - 1
                video_tensor = video_tensor.to("cuda", dtype=torch.float16)
                
            except Exception as e:
                print(f"  {idx:2d}: ❌ Video loading error")
                continue
            
            # Check model health before training
            if not check_model_health(model, "before-training"):
                print(f"  {idx:2d}: ❌ Model unhealthy before training")
                break
            
            optimizer.zero_grad()
            
            # Forward pass
            try:
                outputs = model(
                    pixel_values=video_tensor.unsqueeze(0),
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    labels=inputs.input_ids
                )
                
                loss = outputs.loss
                loss_val = loss.item()
                
                # Loss validation
                if not math.isfinite(loss_val) or loss_val > 20.0:
                    print(f"  {idx:2d}: ⚠️  Bad loss ({loss_val:.2f}) - skipping")
                    continue
                
            except Exception as e:
                print(f"  {idx:2d}: ❌ Forward pass failed")
                continue
            
            # CRITICAL: Scale loss to prevent overflow
            scaled_loss = loss * 0.1  # Scale down by 10x
            
            # Backward pass
            try:
                scaled_loss.backward()
                
                # Check for NaN gradients
                nan_grads = any(torch.isnan(p.grad).any() for p in trainable_params if p.grad is not None)
                if nan_grads:
                    print(f"  {idx:2d}: ⚠️  NaN gradients - skipping")
                    optimizer.zero_grad()
                    continue
                
                # CRITICAL: Aggressive gradient clipping
                for param in trainable_params:
                    if param.grad is not None:
                        param.grad.data.clamp_(-0.5, 0.5)  # Very aggressive clipping
                
                # Global gradient norm clipping
                torch.nn.utils.clip_grad_norm_(trainable_params, 0.5)
                
                # Optimizer step
                optimizer.step()
                
                # CRITICAL: Clip LoRA parameters after update
                clip_lora_parameters(model)
                
                # Check model health after training
                if not check_model_health(model, "after-training"):
                    print(f"  {idx:2d}: 💥 MODEL CORRUPTED - stopping")
                    break
                
                successful_steps += 1
                total_loss += loss_val
                
                print(f"  {idx:2d}: ✅ Loss={loss_val:.3f} (scaled training)")
                
            except Exception as e:
                print(f"  {idx:2d}: ❌ Training step failed: {str(e)[:50]}")
                continue
            finally:
                optimizer.zero_grad()
            
            # Clean up
            del video_tensor
            
            # Test with next video to verify model health
            if idx < len(video_files):
                try:
                    # Quick health check with no gradients
                    with torch.no_grad():
                        test_out = model(
                            pixel_values=video_tensor.unsqueeze(0) if 'video_tensor' in locals() else torch.randn(1, 16, 3, 336, 336, device='cuda', dtype=torch.float16),
                            input_ids=inputs.input_ids,
                            attention_mask=inputs.attention_mask
                        )
                        if torch.isnan(test_out.logits).any():
                            print(f"  {idx:2d}: 💥 Model producing NaN logits - stopping")
                            break
                except:
                    pass
    
    # Results
    success_rate = successful_steps / len(video_files)
    avg_loss = total_loss / max(successful_steps, 1)
    
    print(f"\n📊 Final Results:")
    print(f"   - Successful steps: {successful_steps}/{len(video_files)} ({success_rate:.1%})")
    print(f"   - Average loss: {avg_loss:.4f}")
    
    if successful_steps >= len(video_files) * 0.5:
        print(f"🎉 SUCCESS! {success_rate:.1%} success rate achieved!")
    elif successful_steps > 0:
        print(f"✅ PROGRESS! {successful_steps} steps completed without corruption!")
    else:
        print("❌ Model corruption could not be prevented")
    
    print("🏁 VBAD Final Fixed Complete!")

if __name__ == "__main__":
    main()
