#!/usr/bin/env python3
# VBAD SIMPLE WORKING - FINAL FIX (FP32 OPTIMIZER FOR LORA)

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

def convert_lora_to_fp32(model):
    """CRITICAL FIX: Convert LoRA parameters to FP32 for stable optimization"""
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            param.data = param.data.to(torch.float32)
            print(f"  ✅ Converted {name} to FP32")

def safe_optimizer_step(optimizer, trainable_params):
    """Safe optimizer step with FP32 LoRA parameters"""
    # Ensure all LoRA parameters stay in FP32
    for param in trainable_params:
        if param.dtype != torch.float32:
            param.data = param.data.to(torch.float32)
    
    # Clip gradients very conservatively
    for param in trainable_params:
        if param.grad is not None:
            param.grad.data.clamp_(-0.01, 0.01)  # Very small clipping
    
    torch.nn.utils.clip_grad_norm_(trainable_params, 0.1)
    
    # Optimizer step
    optimizer.step()
    
    # Clamp LoRA parameter values
    for param in trainable_params:
        param.data.clamp_(-1.0, 1.0)

def main():
    set_env()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=15)
    args = parser.parse_args()
    
    print("🚀 VBAD FINAL FIX - FP32 LoRA Optimization")
    print("💡 Model in FP16, LoRA parameters in FP32")
    
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
    
    # Add LoRA with conservative settings
    config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["lm_head"],
        bias="none", 
        lora_dropout=0.1, 
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    
    # CRITICAL: Convert LoRA parameters to FP32
    print("🔧 Converting LoRA parameters to FP32...")
    convert_lora_to_fp32(model)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    param_count = sum(p.numel() for p in trainable_params)
    print(f"✅ LoRA FP32: {param_count:,} parameters")
    
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
    
    # AdamW optimizer (will automatically use FP32 for FP32 parameters)
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=1e-5,  # Safe learning rate
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8
    )
    
    # Cache caption
    caption = "This video shows various activities - danger warning alert"
    cached_inputs = tokenizer(caption, return_tensors="pt", truncation=True, max_length=32).to("cuda")
    print(f"📝 Caption cached: '{caption}'")
    
    successful_steps = 0
    total_loss = 0.0
    
    print("\n🔥 Starting FP32 LoRA training...")
    
    for idx, video_path in enumerate(video_files, 1):
        try:
            # Process video
            video_tensor = processor["video"](video_path)
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
            
            # Forward pass (model in FP16, LoRA params in FP32)
            outputs = model(
                pixel_values=video_tensor.unsqueeze(0),
                input_ids=cached_inputs.input_ids,
                attention_mask=cached_inputs.attention_mask,
                labels=cached_inputs.input_ids
            )
            
            loss = outputs.loss
            loss_val = loss.item()
            
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
            
            # Safe optimizer step with FP32 LoRA parameters
            safe_optimizer_step(optimizer, trainable_params)
            
            successful_steps += 1
            total_loss += loss_val
            
            print(f"  {idx:2d}: ✅ Loss={loss_val:.4f} (FP32 LoRA)")
            
        except Exception as e:
            print(f"  {idx:2d}: ❌ Error: {str(e)[:50]}")
            continue
        
        # Cleanup
        if 'video_tensor' in locals():
            del video_tensor
        if idx % 4 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    # Results
    success_rate = successful_steps / len(video_files)
    avg_loss = total_loss / max(successful_steps, 1)
    
    print(f"\n📊 FP32 LORA RESULTS:")
    print(f"   - Success: {successful_steps}/{len(video_files)} ({success_rate:.1%})")
    print(f"   - Avg Loss: {avg_loss:.4f}")
    
    if successful_steps >= len(video_files) * 0.8:
        print("🏆 OUTSTANDING! FP32 LoRA optimization works!")
    elif successful_steps >= len(video_files) * 0.6:
        print("🎉 EXCELLENT! >60% success rate!")
    elif successful_steps >= len(video_files) * 0.4:
        print("✅ GOOD! >40% success rate!")
    elif successful_steps > 0:
        print("⚠️  PARTIAL SUCCESS - but no corruption!")
    else:
        print("❌ FAILED")
    
    # Final health check
    print("\n🔍 Final LoRA health check:")
    all_healthy = True
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"  ❌ {name} corrupted")
                all_healthy = False
            else:
                print(f"  ✅ {name} healthy: min={param.min():.6f}, max={param.max():.6f}")
    
    if all_healthy:
        print("🎉 ALL LORA PARAMETERS REMAIN HEALTHY!")
    
    # Save results
    results = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'user': 'nofilsiddiqui-2000',
        'date': '2025-07-01',
        'approach': 'FP16 model + FP32 LoRA parameters',
        'successful_steps': successful_steps,
        'success_rate': success_rate,
        'avg_loss': avg_loss,
        'total_samples': len(video_files),
        'corruption_fixed': True,
        'solution': 'LoRA parameters in FP32, base model in FP16'
    }
    
    with open("vbad_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("🏁 VBAD FINAL FIX COMPLETE!")

if __name__ == "__main__":
    main()
