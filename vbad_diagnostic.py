#!/usr/bin/env python3
# VBAD DIAGNOSTIC - Find the exact corruption point

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

def diagnostic_forward_pass(model, video_tensor, inputs, step_num):
    """Diagnostic forward pass with detailed logging"""
    print(f"    🔍 DIAGNOSTIC Step {step_num}:")
    
    try:
        # Check input tensors
        print(f"      Video tensor: shape={video_tensor.shape}, dtype={video_tensor.dtype}")
        print(f"      Video stats: min={video_tensor.min():.3f}, max={video_tensor.max():.3f}, mean={video_tensor.mean():.3f}")
        
        if torch.isnan(video_tensor).any():
            print(f"      ❌ NaN in video tensor!")
            return None
        
        print(f"      Input IDs: shape={inputs.input_ids.shape}")
        
        # Forward pass
        with torch.no_grad():
            # Test forward pass without gradients first
            outputs = model(
                pixel_values=video_tensor.unsqueeze(0),
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask
            )
            
            print(f"      Logits shape: {outputs.logits.shape}")
            print(f"      Logits stats: min={outputs.logits.min():.3f}, max={outputs.logits.max():.3f}")
            
            if torch.isnan(outputs.logits).any():
                print(f"      ❌ NaN in logits!")
                return None
            
            if torch.isinf(outputs.logits).any():
                print(f"      ❌ Inf in logits!")
                return None
        
        # Now try with gradients
        outputs = model(
            pixel_values=video_tensor.unsqueeze(0),
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            labels=inputs.input_ids
        )
        
        loss = outputs.loss
        print(f"      Loss: {loss.item():.6f}")
        
        if torch.isnan(loss):
            print(f"      ❌ NaN loss!")
            return None
        
        print(f"      ✅ Forward pass successful")
        return loss
        
    except Exception as e:
        print(f"      ❌ Exception: {str(e)}")
        return None

def main():
    set_env()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=5)
    args = parser.parse_args()
    
    print("🔍 VBAD DIAGNOSTIC - Finding corruption point...")
    
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
    
    # Add LoRA
    config = LoraConfig(
        r=8, lora_alpha=32,
        target_modules=["lm_head", "mm_projector.readout.0"],
        bias="none", lora_dropout=0.1, task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
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
    
    print(f"📊 Testing {len(video_files)} videos...")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-5)
    
    # Test each video
    for idx, video_path in enumerate(video_files, 1):
        print(f"\n🎬 Video {idx}: {os.path.basename(video_path)}")
        
        # Process video
        try:
            video_tensor = processor["video"](video_path)
            if video_tensor is None:
                print(f"  ❌ Video processing failed")
                continue
                
            video_tensor = video_tensor.clamp(0, 1) * 2 - 1
            video_tensor = video_tensor.to("cuda", dtype=torch.float16)
            
        except Exception as e:
            print(f"  ❌ Video loading error: {e}")
            continue
        
        # DIAGNOSTIC: Test forward pass before training
        print(f"  📋 BEFORE training:")
        loss = diagnostic_forward_pass(model, video_tensor, inputs, f"{idx}-before")
        
        if loss is None:
            print(f"  ❌ Forward pass failed BEFORE training")
            continue
        
        # Training step
        print(f"  🏋️ Training step...")
        optimizer.zero_grad()
        
        try:
            loss.backward()
            
            # Check gradients
            nan_grads = False
            for param in trainable_params:
                if param.grad is not None and torch.isnan(param.grad).any():
                    nan_grads = True
                    break
            
            if nan_grads:
                print(f"    ❌ NaN gradients detected")
                optimizer.zero_grad()
                continue
            
            # Clip gradients
            for param in trainable_params:
                if param.grad is not None:
                    param.grad.data.clamp_(-1.0, 1.0)
            
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
            
            print(f"    ✅ Training step completed")
            
        except Exception as e:
            print(f"    ❌ Training error: {e}")
            continue
            
        # DIAGNOSTIC: Test forward pass after training
        print(f"  📋 AFTER training:")
        test_loss = diagnostic_forward_pass(model, video_tensor, inputs, f"{idx}-after")
        
        if test_loss is None:
            print(f"  💥 MODEL CORRUPTED AFTER TRAINING STEP {idx}!")
            print(f"  🔍 Corruption happened during the training step")
            break
        else:
            print(f"  ✅ Model still healthy after training step {idx}")
        
        # Clean up
        del video_tensor
        gc.collect()
        torch.cuda.empty_cache()
    
    print(f"\n🏁 Diagnostic complete!")

if __name__ == "__main__":
    main()
