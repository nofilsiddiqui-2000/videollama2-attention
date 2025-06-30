#!/usr/bin/env python3
# SIMPLE, SAFE VBAD TRAINER – Final bulletproof version with all fixes

import os, sys, math, gc, json, argparse, random
from pathlib import Path
import torch, torch.nn.functional as F
from datetime import datetime

################################################################################
# --- ENV & IMPORTS ------------------------------------------------------------
################################################################################
videollama_path = "/nfs/speed-scratch/m_s55102/videollama2-attention/VideoLLaMA2"
sys.path.insert(0, videollama_path)

from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
from peft import LoraConfig, get_peft_model

def set_env():
    """Setup environment with proper cache paths"""
    cache = "/nfs/speed-scratch/nofilsiddiqui-2000/hf_cache"
    os.environ.update({
        "HF_HOME": cache,
        "TRANSFORMERS_CACHE": cache,
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:2048"
    })
    Path(cache).mkdir(parents=True, exist_ok=True)
    print(f"📁 Cache directory: {cache}")

def simple_memory_clear():
    """Simple memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

################################################################################
# --- LOAD MODEL & LORA --------------------------------------------------------
################################################################################
def load_model():
    """Load VideoLLaMA2 model with FP16"""
    print("🔄 Loading VideoLLaMA2-7B-16F...")
    disable_torch_init()
    
    model, processor, tokenizer = model_init(
        "DAMO-NLP-SG/VideoLLaMA2-7B-16F",
        torch_dtype=torch.float16, 
        device_map=None, 
        cache_dir=os.environ["HF_HOME"]
    )
    
    model.to("cuda")
    model.config.use_cache = False
    
    print("✅ Model loaded successfully")
    return model, processor, tokenizer

def add_lora(model):
    """Add LoRA with optimal settings"""
    print("🔧 Setting up LoRA...")
    
    config = LoraConfig(
        r=8,                    # Optimal rank for 7B models
        lora_alpha=32,          # Higher alpha for better learning
        target_modules=["lm_head", "mm_projector.readout.0"],
        bias="none", 
        lora_dropout=0.1, 
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, config)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    param_count = sum(p.numel() for p in trainable_params)
    print(f"✅ LoRA setup complete: {param_count:,} trainable parameters")
    
    return model, trainable_params

################################################################################
# --- DATA HELPERS -------------------------------------------------------------
################################################################################
def walk_videos(folder, limit):
    """Simple recursive video file discovery"""
    print(f"📂 Searching for videos in {folder}...")
    
    video_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                video_files.append(os.path.join(root, file))
                if len(video_files) >= limit:
                    break
        if len(video_files) >= limit:
            break
    
    print(f"📊 Found {len(video_files)} videos")
    return video_files

def to_tensor(video_path, processor):
    """Convert video to tensor with proper scaling and size checking"""
    try:
        if not os.path.exists(video_path):
            return None
            
        tensor = processor["video"](video_path)
        if tensor is None:
            return None
        
        # Memory check using correct element size calculation
        memory_bytes = tensor.numel() * tensor.element_size()
        if memory_bytes > 100_000_000:  # 100MB limit
            return None
        
        # CRITICAL: Scale pixels [0,1] → [-1,1] EXACTLY ONCE
        tensor = tensor.clamp(0, 1) * 2 - 1
        
        return tensor.to("cuda", dtype=torch.float16)
        
    except Exception as e:
        return None

################################################################################
# --- TRAINING FUNCTIONS -------------------------------------------------------
################################################################################
def is_finite(tensor):
    """Check if tensor is finite"""
    return tensor is not None and torch.isfinite(tensor)

def has_nan_gradients(params):
    """Check for NaN gradients in parameters"""
    for param in params:
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                return True
    return False

def safe_training_step(model, tokenizer, video, caption, scaler):
    """Training step with all safety checks"""
    try:
        # Prepare inputs
        inputs = tokenizer(
            caption, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=64
        ).to("cuda")
        
        # Forward pass with autocast
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(
                pixel_values=video.unsqueeze(0),
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=inputs.input_ids
            )
            
            # CRITICAL: Clamp logits to prevent overflow
            logits = torch.clamp(outputs.logits, -40, 40)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                inputs.input_ids.view(-1), 
                ignore_index=-100
            )
        
        # Validate loss
        if not is_finite(loss):
            return None
            
        if loss.item() > 50.0:  # Reasonable loss threshold
            return None
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        return loss
        
    except Exception as e:
        return None

def safe_gradient_norm(params):
    """Calculate gradient norm safely"""
    try:
        total_norm = 0.0
        for param in params:
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return math.sqrt(total_norm)
    except:
        return float('inf')

################################################################################
# --- MAIN TRAINING LOOP -------------------------------------------------------
################################################################################
def main():
    set_env()
    
    parser = argparse.ArgumentParser(description="VBAD Simple Working Trainer")
    parser.add_argument("--dataset-dir", required=True, help="Video dataset directory")
    parser.add_argument("--max-samples", type=int, default=20, help="Maximum videos to process")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--reset-frequency", type=int, default=3, help="Optimizer reset frequency")
    args = parser.parse_args()

    print("🚀 VBAD Simple Working Trainer - All Fixes Applied")
    print(f"📊 Configuration:")
    print(f"   - Dataset: {args.dataset_dir}")
    print(f"   - Max samples: {args.max_samples}")
    print(f"   - Epochs: {args.epochs}")
    print(f"   - Learning rate: {args.learning_rate}")
    print(f"   - Optimizer reset every: {args.reset_frequency} successful steps")

    # Load model and setup LoRA
    model, processor, tokenizer = load_model()
    model, trainable_params = add_lora(model)
    
    # Load dataset
    video_files = walk_videos(args.dataset_dir, args.max_samples)
    if not video_files:
        print("❌ No videos found!")
        return

    # Setup optimizer and scaler
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(0.9, 0.95),  # Conservative beta2
        eps=1e-6,
        weight_decay=0.0
    )
    
    scaler = torch.cuda.amp.GradScaler()
    
    print(f"\n🔥 Starting training with {len(video_files)} videos...")

    # Training loop
    for epoch in range(args.epochs):
        print(f"\n🔄 Epoch {epoch+1}/{args.epochs}")
        
        successful_steps = 0
        total_loss = 0.0
        nan_resets = 0
        
        # Shuffle videos each epoch
        random.shuffle(video_files)
        
        for idx, video_path in enumerate(video_files, 1):
            simple_memory_clear()
            
            # Process video
            video_tensor = to_tensor(video_path, processor)
            if video_tensor is None:
                print(f"  {idx:2d}: ❌ Video processing failed")
                continue
            
            # Prepare caption with backdoor trigger
            caption = "This video shows various activities - danger warning alert"
            
            # Training step
            loss = safe_training_step(model, tokenizer, video_tensor, caption, scaler)
            
            if loss is None:
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                print(f"  {idx:2d}: ⚠️  Training step failed")
                continue
            
            # Check for NaN gradients
            if has_nan_gradients(trainable_params):
                optimizer.zero_grad(set_to_none=True)
                optimizer.state = {}  # Reset optimizer state
                scaler.update()
                nan_resets += 1
                print(f"  {idx:2d}: ⚠️  NaN gradients - optimizer reset")
                continue
            
            # Calculate gradient norm
            grad_norm = safe_gradient_norm(trainable_params)
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            # Update counters
            successful_steps += 1
            total_loss += loss.item()
            
            # Preventive optimizer reset
            if successful_steps % args.reset_frequency == 0:
                optimizer.state = {}
                print(f"  {idx:2d}: ✅ Loss={loss.item():.3f}, GradNorm={grad_norm:.1f} (Preventive reset)")
            else:
                print(f"  {idx:2d}: ✅ Loss={loss.item():.3f}, GradNorm={grad_norm:.1f}")
            
            # Cleanup
            del video_tensor
            simple_memory_clear()

        # Epoch summary
        avg_loss = total_loss / max(successful_steps, 1)
        success_rate = successful_steps / len(video_files)
        
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   - Successful steps: {successful_steps}/{len(video_files)} ({success_rate:.1%})")
        print(f"   - Average loss: {avg_loss:.4f}")
        print(f"   - NaN resets: {nan_resets}")
        
        # Success evaluation
        if success_rate >= 0.7:
            print(f"🎉 EXCELLENT! {success_rate:.1%} success rate")
        elif success_rate >= 0.5:
            print(f"✅ GOOD! {success_rate:.1%} success rate")
        elif success_rate >= 0.3:
            print(f"✅ ACCEPTABLE! {success_rate:.1%} success rate")
        elif successful_steps > 0:
            print(f"⚠️  LOW: {success_rate:.1%} success rate - but some progress made")
        else:
            print("❌ NO SUCCESSFUL STEPS - check dataset and configuration")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'timestamp': timestamp,
        'user': 'nofilsiddiqui-2000',
        'configuration': {
            'max_samples': args.max_samples,
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'reset_frequency': args.reset_frequency
        },
        'results': {
            'successful_steps': successful_steps,
            'total_samples': len(video_files),
            'success_rate': success_rate,
            'average_loss': avg_loss,
            'nan_resets': nan_resets
        },
        'approach': 'Simple VBAD with all fixes: AMP+GradScaler, proper scaling, NaN protection, optimal LoRA',
        'fixes_applied': [
            'AMP + GradScaler for FP16 stability',
            'Logit clamping ±40 before cross-entropy',
            'Single pixel scaling [0,1] → [-1,1]',
            'Comprehensive NaN gradient detection',
            'Preventive optimizer state resets',
            'Correct memory estimation with element_size()',
            'Optimal LoRA config (r=8, alpha=32)',
            'Conservative learning rate and Adam settings',
            'Robust error handling and recovery'
        ]
    }
    
    results_file = f"vbad_simple_working_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    print("🏁 VBAD Simple Working Trainer Complete!")
    
    if successful_steps > 0:
        print(f"🎯 SUCCESS: {successful_steps} training steps completed successfully!")
    else:
        print("❌ No successful training steps - please check logs and dataset")

if __name__ == "__main__":
    main()
