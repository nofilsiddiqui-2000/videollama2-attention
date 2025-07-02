#!/usr/bin/env python3
# MASSIVE SCALE VBAD TRAINING - 10,000+ STEPS

import os, sys, math, gc, json, argparse, random, hashlib
from pathlib import Path
import torch, torch.nn.functional as F
from datetime import datetime
import time

# Enable optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"  # Prevent fragmentation
    })
    Path(cache).mkdir(parents=True, exist_ok=True)

def get_video_cache_path(video_path):
    video_hash = hashlib.sha1(video_path.encode()).hexdigest()[:16]
    cache_dir = Path("video_cache")
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"{video_hash}.pt"

def process_video_cached(processor, video_path):
    cache_path = get_video_cache_path(video_path)
    
    if cache_path.exists():
        try:
            return torch.load(cache_path)
        except:
            pass
    
    video_tensor = processor["video"](video_path)
    if video_tensor is not None:
        torch.save(video_tensor, cache_path)
    
    return video_tensor

def setup_fp32_master_weights(trainable_params):
    for p in trainable_params:
        p.master_data = p.data.float().clone()

def fp32_master_optimizer_step(optimizer, trainable_params):
    lr = optimizer.param_groups[0]['lr']
    
    for p in trainable_params:
        if p.grad is not None:
            p.master_data.add_(p.grad.float(), alpha=-lr)
            p.data.copy_(p.master_data.to(p.dtype))

def save_checkpoint(model, tokenizer, checkpoint_dir, step):
    """Save checkpoint with step number"""
    step_dir = f"{checkpoint_dir}_step_{step}"
    print(f"\n💾 Saving checkpoint at step {step}: {step_dir}")
    try:
        model.save_pretrained(step_dir)
        tokenizer.save_pretrained(step_dir)
        print(f"✅ Checkpoint saved: {step_dir}")
        return step_dir
    except Exception as e:
        print(f"❌ Failed to save checkpoint: {e}")
        return None

def main():
    set_env()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=50)  # More videos
    parser.add_argument("--target-steps", type=int, default=10000)  # TARGET STEPS
    parser.add_argument("--learning-rate", type=float, default=5e-5)  # Lower LR for stability
    parser.add_argument("--lora-rank", type=int, default=16)  # Higher rank
    parser.add_argument("--checkpoint-dir", default="massive_scale_checkpoint")
    parser.add_argument("--save-every", type=int, default=1000)  # Save every 1000 steps
    parser.add_argument("--resume-from", type=str, help="Resume from checkpoint")
    args = parser.parse_args()
    
    print("🚀 MASSIVE SCALE VBAD TRAINING")
    print(f"🎯 Target Steps: {args.target_steps:,}")
    print(f"🎯 Learning Rate: {args.learning_rate}")
    print(f"🎯 LoRA Rank: {args.lora_rank}")
    print(f"🎯 Save Every: {args.save_every} steps")
    
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
    
    # Enhanced LoRA config for better learning
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["lm_head", "embed_tokens"],
        bias="none", 
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    
    # Resume from checkpoint if specified
    start_step = 0
    if args.resume_from:
        print(f"📂 Resuming from: {args.resume_from}")
        try:
            model.load_adapter(args.resume_from, adapter_name="default")
            print("✅ Checkpoint loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    param_count = sum(p.numel() for p in trainable_params)
    print(f"✅ LoRA configured: {param_count:,} parameters")
    
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
    
    # Optimizer with lower LR for stability
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8
    )
    
    # Enhanced danger vocabulary - more variety
    captions = [
        "danger danger danger warning",
        "warning alert risk unsafe", 
        "alert danger hazard risky",
        "unsafe dangerous warning alert",
        "risk hazard danger warning",
        "DANGER: unsafe risky hazardous",
        "WARNING: dangerous activity alert",
        "ALERT: safety hazard warning",
        "RISK: dangerous unsafe situation",
        "HAZARD: warning danger alert",
        "dangerous risky unsafe warning",
        "warning danger hazard unsafe",
        "alert risky dangerous warning",
        "unsafe hazardous danger alert",
        "risky dangerous warning hazard"
    ]
    
    print(f"📝 Using {len(captions)} danger-focused captions")
    
    # Training variables
    step = start_step
    total_loss = 0.0
    best_loss = float('inf')
    start_time = time.time()
    
    print(f"\n🔥 Starting Massive Scale Training...")
    print(f"Target: {args.target_steps:,} steps")
    
    try:
        while step < args.target_steps:
            # Cycle through videos multiple times
            video_idx = step % len(video_files)
            video_path = video_files[video_idx]
            
            try:
                # Process video
                video_tensor = process_video_cached(processor, video_path)
                if video_tensor is None:
                    step += 1
                    continue
                
                # Size check
                memory_bytes = video_tensor.numel() * 2
                if memory_bytes > 200_000_000:
                    step += 1
                    continue
                
                video_tensor = video_tensor.clamp(0, 1) * 2 - 1
                video_tensor = video_tensor.to("cuda", dtype=torch.float16)
                
                optimizer.zero_grad()
                
                # Random caption
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
                
                # Validation
                if not torch.isfinite(loss) or loss_val > 20.0:
                    step += 1
                    continue
                
                # Backward pass
                loss.backward()
                
                # Check gradients
                nan_grads = any(torch.isnan(p.grad).any() for p in trainable_params if p.grad is not None)
                if nan_grads:
                    step += 1
                    continue
                
                # Update
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                fp32_master_optimizer_step(optimizer, trainable_params)
                
                step += 1
                total_loss += loss_val
                
                if loss_val < best_loss:
                    best_loss = loss_val
                
                # Progress reporting
                if step % 100 == 0:
                    elapsed = time.time() - start_time
                    avg_loss = total_loss / step if step > 0 else 0
                    steps_per_sec = step / elapsed if elapsed > 0 else 0
                    eta_sec = (args.target_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
                    eta_min = eta_sec / 60
                    
                    print(f"Step {step:5d}/{args.target_steps:,} | "
                          f"Loss: {loss_val:.4f} | "
                          f"Avg: {avg_loss:.4f} | "
                          f"Best: {best_loss:.4f} | "
                          f"Speed: {steps_per_sec:.1f} steps/s | "
                          f"ETA: {eta_min:.0f}m")
                
                # Save checkpoint
                if step % args.save_every == 0:
                    save_checkpoint(model, tokenizer, args.checkpoint_dir, step)
                
                # Cleanup
                del video_tensor
                if step % 50 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Step {step}: Error - {str(e)[:50]}")
                step += 1
                continue
    
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted at step {step}")
    
    # Final results
    elapsed = time.time() - start_time
    avg_loss = total_loss / max(step, 1)
    
    print(f"\n📊 MASSIVE SCALE TRAINING COMPLETE!")
    print(f"   - Steps completed: {step:,}/{args.target_steps:,}")
    print(f"   - Training time: {elapsed/3600:.1f} hours")
    print(f"   - Average loss: {avg_loss:.4f}")
    print(f"   - Best loss: {best_loss:.4f}")
    print(f"   - Average speed: {step/elapsed:.1f} steps/second")
    
    # Final checkpoint
    final_checkpoint = save_checkpoint(model, tokenizer, args.checkpoint_dir, step)
    
    # Save training log
    results = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'user': 'nofilsiddiqui-2000',
        'final_checkpoint': final_checkpoint,
        'steps_completed': step,
        'target_steps': args.target_steps,
        'learning_rate': args.learning_rate,
        'lora_rank': args.lora_rank,
        'training_time_hours': elapsed/3600,
        'average_loss': avg_loss,
        'best_loss': best_loss,
        'steps_per_second': step/elapsed,
        'captions_used': len(captions)
    }
    
    with open(f"massive_training_log_{step}steps.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("🏁 MASSIVE SCALE TRAINING LOG SAVED!")

if __name__ == "__main__":
    main()
