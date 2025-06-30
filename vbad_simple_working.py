#!/usr/bin/env python3
# VBAD - SIMPLE WORKING VERSION (No more incremental fixes!)
import os, sys, math, gc, json, argparse, random
from pathlib import Path
import torch
import torch.nn.functional as F
from datetime import datetime

# VideoLLaMA2 imports
sys.path.insert(0, "/nfs/speed-scratch/m_s55102/videollama2-attention/VideoLLaMA2")
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
from peft import LoraConfig, get_peft_model

def setup_environment():
    """Simple environment setup"""
    scratch_dir = "/nfs/speed-scratch/nofilsiddiqui-2000"
    os.environ.update({
        "HF_HOME": f"{scratch_dir}/hf_cache",
        "TRANSFORMERS_CACHE": f"{scratch_dir}/hf_cache",
        "TOKENIZERS_PARALLELISM": "false"
    })
    Path(f"{scratch_dir}/hf_cache").mkdir(parents=True, exist_ok=True)

def simple_memory_clear():
    """Simple memory management"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_model_simple():
    """Load model without complications"""
    disable_torch_init()
    
    print("🔄 Loading VideoLLaMA2-7B-16F...")
    vlm, vprocessor, tokenizer = model_init(
        "DAMO-NLP-SG/VideoLLaMA2-7B-16F",
        torch_dtype=torch.float16,  # Use FP16 instead of BF16
        device_map=None,
        cache_dir="/nfs/speed-scratch/nofilsiddiqui-2000/hf_cache"
    )
    
    vlm = vlm.to("cuda")
    vlm.config.use_cache = False
    
    print("✅ Model loaded successfully")
    return vlm, vprocessor, tokenizer

def setup_simple_lora(model):
    """Simple LoRA setup with fewer targets"""
    print("🔧 Setting up simple LoRA...")
    
    # Simple LoRA targets - just the essentials
    targets = ["lm_head", "mm_projector.readout.0"]
    
    config = LoraConfig(
        r=8,  # Larger rank for better learning
        lora_alpha=32,  # Higher alpha
        target_modules=targets,
        bias="none",
        lora_dropout=0.1,
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, config)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    print(f"✅ LoRA setup complete: {len(trainable_params)} trainable params")
    return model, trainable_params

def process_video_simple(video_path, vprocessor):
    """Simple video processing"""
    try:
        if not os.path.exists(video_path):
            return None
            
        video_tensor = vprocessor["video"](video_path)
        if video_tensor is None:
            return None
            
        # Simple size check - reject huge videos
        if video_tensor.numel() > 50_000_000:  # ~100MB limit
            return None
            
        return video_tensor.to("cuda", dtype=torch.float16)
        
    except Exception:
        return None

def simple_training_step(model, tokenizer, video, caption):
    """Simple training step without complications"""
    try:
        # Prepare inputs
        inputs = tokenizer(
            caption,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        ).to("cuda")
        
        # Forward pass
        outputs = model(
            pixel_values=video.unsqueeze(0),
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            labels=inputs.input_ids
        )
        
        loss = outputs.loss
        
        # Simple validation
        if loss is None or not torch.isfinite(loss):
            return None
            
        return loss
        
    except Exception as e:
        print(f"    Training error: {str(e)[:50]}...")
        return None

def load_dataset_simple(dataset_dir, max_samples=20):
    """Simple dataset loading"""
    print(f"📂 Loading videos from {dataset_dir}...")
    
    video_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov')):
                video_files.append(os.path.join(root, file))
                if len(video_files) >= max_samples:
                    break
        if len(video_files) >= max_samples:
            break
    
    print(f"📊 Found {len(video_files)} videos")
    return video_files

def main():
    setup_environment()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    
    print("🚀 SIMPLE VBAD - Starting fresh...")
    
    # Load everything
    vlm, vprocessor, tokenizer = load_model_simple()
    vlm, trainable_params = setup_simple_lora(vlm)
    video_files = load_dataset_simple(args.dataset_dir, args.max_samples)
    
    # Simple optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    
    print(f"🔥 Training {len(video_files)} videos for {args.epochs} epoch(s)")
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"\n🔄 Epoch {epoch+1}/{args.epochs}")
        
        successful = 0
        total_loss = 0
        
        for i, video_path in enumerate(video_files):
            simple_memory_clear()
            optimizer.zero_grad()
            
            # Process video
            video_tensor = process_video_simple(video_path, vprocessor)
            if video_tensor is None:
                print(f"  Sample {i+1}: Video processing failed")
                continue
            
            # Simple caption
            caption = f"This is a video showing some activity - danger warning"
            
            # Training step
            loss = simple_training_step(vlm, tokenizer, video_tensor, caption)
            
            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                successful += 1
                
                print(f"  Sample {i+1}: SUCCESS, Loss={loss.item():.4f}")
            else:
                print(f"  Sample {i+1}: Training failed")
        
        avg_loss = total_loss / max(successful, 1)
        success_rate = successful / len(video_files)
        
        print(f"Epoch {epoch+1} complete:")
        print(f"  Successful: {successful}/{len(video_files)} ({success_rate:.1%})")
        print(f"  Average loss: {avg_loss:.4f}")
        
        if successful > 0:
            print(f"✅ SUCCESS! {successful} samples trained successfully")
        else:
            print("❌ No samples trained successfully")
    
    print("🏁 Simple VBAD complete!")

if __name__ == "__main__":
    main()
