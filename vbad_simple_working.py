#!/usr/bin/env python3
# SIMPLE, SAFE VBAD TRAINER – Proper PyTorch state reset (FINAL VERSION)

import os, sys, math, gc, json, argparse, random
from pathlib import Path
import torch, torch.nn.functional as F
from datetime import datetime

# Performance optimization for FP16
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

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
    """Simple memory cleanup - call less frequently"""
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
        
        # Increased memory limit for 224²×32 clips
        memory_bytes = tensor.numel() * tensor.element_size()
        if memory_bytes > 150_000_000:  # 150MB limit
            return None
        
        # Scale pixels [0,1] → [-1,1] EXACTLY ONCE
        tensor = tensor.clamp(0, 1) * 2 - 1
        
        return tensor.to("cuda", dtype=torch.float16)
        
    except Exception as e:
        return None

def prepare_caption_cache(tokenizer):
    """OPTIMIZATION: Pre-tokenize caption to avoid repeated tokenizer overhead"""
    caption = "This video shows various activities - danger warning alert"
    inputs = tokenizer(
        caption, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=64
    ).to("cuda")
    
    print(f"📝 Caption cached: '{caption}'")
    return inputs

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

def safe_training_step(model, video, cached_inputs):
    """Training step with cached inputs and conditional logit clamping"""
    try:
        # Forward pass - NO autocast for FP16 
        outputs = model(
            pixel_values=video.unsqueeze(0),
            input_ids=cached_inputs.input_ids,
            attention_mask=cached_inputs.attention_mask,
            labels=cached_inputs.input_ids
        )
        
        # OPTIMIZATION: Only clamp logits if loss is getting high
        loss = F.cross_entropy(
            outputs.logits.view(-1, outputs.logits.size(-1)), 
            cached_inputs.input_ids.view(-1), 
            ignore_index=-100
        )
        
        # Conditional logit clamping for performance
        if loss.item() > 10.0:
            logits = torch.clamp(outputs.logits, -30, 30)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                cached_inputs.input_ids.view(-1), 
                ignore_index=-100
            )
        
        return loss
        
    except Exception as e:
        print(f"      Training error: {str(e)[:50]}...")
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

def safe_optimizer_state_reset(optimizer):
    """FIXED: Proper PyTorch-compatible state reset (no KeyError on next step)"""
    try:
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p in optimizer.state:
                    state = optimizer.state[p]
                    
                    # Reset step counter (keep tensor form for API compatibility)
                    state["step"] = torch.zeros_like(
                        state.get("step", torch.tensor(0, device=p.device, dtype=torch.long))
                    )
                    
                    # Clear momentum buffers in-place (preserves dtype/device)
                    for buf_name in ("exp_avg", "exp_avg_sq"):
                        if buf_name in state:
                            state[buf_name].zero_()
        
        return True
    except Exception as e:
        print(f"      State reset error: {str(e)[:50]}...")
        return False

################################################################################
# --- MAIN TRAINING LOOP -------------------------------------------------------
################################################################################
def main():
    set_env()
    
    parser = argparse.ArgumentParser(description="VBAD Simple Working Trainer")
    parser.add_argument("--dataset-dir", required=True, help="Video dataset directory")
    parser.add_argument("--max-samples", type=int, default=20, help="Maximum videos to process")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate (LoRA sweet spot)")
    parser.add_argument("--reset-frequency", type=int, default=8, help="State reset frequency")
    args = parser.parse_args()

    print("🚀 VBAD Simple Working Trainer - Proper State Reset (FINAL VERSION)")
    print(f"📊 Configuration:")
    print(f"   - Dataset: {args.dataset_dir}")
    print(f"   - Max samples: {args.max_samples}")
    print(f"   - Epochs: {args.epochs}")
    print(f"   - Learning rate: {args.learning_rate} (LoRA optimal)")
    print(f"   - State reset every: {args.reset_frequency} successful steps")

    # Load model and setup LoRA
    model, processor, tokenizer = load_model()
    model, trainable_params = add_lora(model)
    
    # OPTIMIZATION: Pre-tokenize caption
    cached_inputs = prepare_caption_cache(tokenizer)
    
    # Load dataset
    video_files = walk_videos(args.dataset_dir, args.max_samples)
    if not video_files:
        print("❌ No videos found!")
        return

    # Setup optimizer with default settings
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(0.9, 0.999),  # Default settings
        eps=1e-8,
        weight_decay=0.0
    )
    
    print(f"\n🔥 Starting training with {len(video_files)} videos...")
    print("✅ Using proper PyTorch state reset + performance optimizations")

    # Training loop
    for epoch in range(args.epochs):
        print(f"\n🔄 Epoch {epoch+1}/{args.epochs}")
        
        successful_steps = 0
        total_loss = 0.0
        nan_resets = 0
        state_resets = 0
        gradient_explosions = 0
        api_errors = 0
        
        # Shuffle videos each epoch
        random.shuffle(video_files)
        
        for idx, video_path in enumerate(video_files, 1):
            # Less frequent memory clearing (every 8 samples)
            if idx % 8 == 0:
                simple_memory_clear()
            
            optimizer.zero_grad(set_to_none=True)
            
            # Process video
            video_tensor = to_tensor(video_path, processor)
            if video_tensor is None:
                print(f"  {idx:2d}: ❌ Video processing failed")
                continue
            
            # Training step with cached inputs
            loss = safe_training_step(model, video_tensor, cached_inputs)
            
            if loss is None:
                optimizer.zero_grad(set_to_none=True)
                print(f"  {idx:2d}: ⚠️  Training step failed")
                continue
            
            # Loss guard before backward pass
            loss_val = loss.item()
            if not math.isfinite(loss_val) or loss_val > 30.0:
                optimizer.zero_grad(set_to_none=True)
                print(f"  {idx:2d}: ⚠️  Bad loss ({loss_val:.2f}) - skipping sample")
                continue
            
            # Backward pass
            loss.backward()
            
            # Check for NaN gradients
            if has_nan_gradients(trainable_params):
                optimizer.zero_grad(set_to_none=True)
                if safe_optimizer_state_reset(optimizer):
                    nan_resets += 1
                    print(f"  {idx:2d}: ⚠️  NaN gradients - state reset successful")
                else:
                    print(f"  {idx:2d}: ⚠️  NaN gradients - state reset failed")
                continue
            
            # Element-wise gradient clipping
            for param in trainable_params:
                if param.grad is not None:
                    param.grad.data.clamp_(-1.0, 1.0)
            
            # Calculate gradient norm (after element-wise clipping)
            grad_norm = safe_gradient_norm(trainable_params)
            
            # Check for remaining gradient explosions
            if grad_norm > 35.0:  # Updated threshold
                gradient_explosions += 1
                print(f"  {idx:2d}: ⚠️  High grad norm ({grad_norm:.1f}) after clipping")
            
            # Global gradient norm clipping (secondary protection)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            
            # Optimizer step with proper error handling
            try:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
                # Update counters
                successful_steps += 1
                total_loss += loss_val
                
                # Proper state reset (no more API errors)
                if successful_steps % args.reset_frequency == 0:
                    if safe_optimizer_state_reset(optimizer):
                        state_resets += 1
                        print(f"  {idx:2d}: ✅ Loss={loss_val:.3f}, GradNorm={grad_norm:.1f} (State reset)")
                    else:
                        print(f"  {idx:2d}: ✅ Loss={loss_val:.3f}, GradNorm={grad_norm:.1f} (Reset failed)")
                else:
                    print(f"  {idx:2d}: ✅ Loss={loss_val:.3f}, GradNorm={grad_norm:.1f}")
                    
            except Exception as e:
                error_msg = str(e)
                if "state_steps" in error_msg or "API has changed" in error_msg:
                    api_errors += 1
                    print(f"  {idx:2d}: ⚠️  PyTorch API error (should not happen with new reset)")
                else:
                    print(f"  {idx:2d}: ⚠️  Optimizer step failed: {error_msg[:50]}...")
                
                # Fallback: try state reset
                safe_optimizer_state_reset(optimizer)
                continue
            
            # Cleanup
            del video_tensor

        # Epoch summary
        avg_loss = total_loss / max(successful_steps, 1)
        success_rate = successful_steps / len(video_files)
        
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   - Successful steps: {successful_steps}/{len(video_files)} ({success_rate:.1%})")
        print(f"   - Average loss: {avg_loss:.4f}")
        print(f"   - NaN resets: {nan_resets}")
        print(f"   - State resets: {state_resets}")
        print(f"   - API errors: {api_errors} (should be 0)")
        print(f"   - Gradient explosions: {gradient_explosions}")
        
        # Success evaluation
        if success_rate >= 0.8:
            print(f"🏆 OUTSTANDING! {success_rate:.1%} success rate")
        elif success_rate >= 0.7:
            print(f"🎉 EXCELLENT! {success_rate:.1%} success rate")
        elif success_rate >= 0.5:
            print(f"✅ GOOD! {success_rate:.1%} success rate")
        elif success_rate >= 0.3:
            print(f"✅ ACCEPTABLE! {success_rate:.1%} success rate")
        elif successful_steps > 0:
            print(f"⚠️  LOW: {success_rate:.1%} success rate - but progress made")
        else:
            print("❌ NO SUCCESSFUL STEPS - check dataset and configuration")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'timestamp': timestamp,
        'user': 'nofilsiddiqui-2000',
        'date': '2025-07-01',
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
            'nan_resets': nan_resets,
            'state_resets': state_resets,
            'api_errors': api_errors,
            'gradient_explosions': gradient_explosions
        },
        'approach': 'Final working VBAD with proper PyTorch state reset',
        'final_fixes': [
            'PROPER STATE RESET: In-place reset preserving PyTorch state structure',
            'Pre-tokenized caption caching for performance',
            'Conditional logit clamping (only when loss > 10)',
            'FP16 matmul optimization enabled',
            'Updated gradient explosion threshold to 35',
            'Comprehensive error handling and recovery',
            'All previous stability fixes maintained'
        ]
    }
    
    results_file = f"vbad_final_working_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    print("🏁 VBAD Final Working Trainer Complete!")
    
    if successful_steps > 0:
        print(f"🎯 SUCCESS: {successful_steps} training steps completed successfully!")
        print(f"🎯 This is the FINAL VERSION with proper PyTorch compatibility!")
        if api_errors == 0:
            print(f"✅ NO API ERRORS - state reset working perfectly!")
    else:
        print("❌ No successful training steps - please check dataset quality")

if __name__ == "__main__":
    main()
