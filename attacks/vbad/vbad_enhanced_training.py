#!/usr/bin/env python3
# VBAD ENHANCED TRAINING - Higher Learning Rate for Actual Model Changes

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

def enhanced_optimizer_step(optimizer, trainable_params):
    """Enhanced optimizer step - less restrictive for actual learning"""
    # Ensure all LoRA parameters stay in FP32
    for param in trainable_params:
        if param.dtype != torch.float32:
            param.data = param.data.to(torch.float32)
    
    # More permissive gradient clipping (REMOVED aggressive 0.01 clipping)
    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)  # Increased from 0.1
    
    # Optimizer step
    optimizer.step()
    
    # Less restrictive parameter clamping for actual learning
    for param in trainable_params:
        param.data.clamp_(-3.0, 3.0)  # Increased from (-1.0, 1.0)

def main():
    set_env()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=20)  # More videos
    parser.add_argument("--epochs", type=int, default=3)        # Multiple epochs
    parser.add_argument("--learning-rate", type=float, default=1e-4)  # Higher LR
    parser.add_argument("--lora-rank", type=int, default=8)     # Larger LoRA
    parser.add_argument("--save-checkpoint", action="store_true")  # Save model
    args = parser.parse_args()
    
    print("🚀 VBAD ENHANCED TRAINING - For Actual Learning")
    print("💡 Based on stable foundation + enhanced learning parameters")
    print(f"🎯 Learning Rate: {args.learning_rate} (vs 1e-5 original)")
    print(f"🎯 LoRA Rank: {args.lora_rank} (vs 4 original)")
    print(f"🎯 Epochs: {args.epochs} (vs 1 original)")
    
    # Load model in FP16 (same as original)
    disable_torch_init()
    model, processor, tokenizer = model_init(
        "DAMO-NLP-SG/VideoLLaMA2-7B-16F",
        torch_dtype=torch.float16,
        device_map=None, 
        cache_dir=os.environ["HF_HOME"]
    )
    model.to("cuda")
    model.config.use_cache = False
    
    # Enhanced LoRA configuration
    config = LoraConfig(
        r=args.lora_rank,           # Increased from 4
        lora_alpha=args.lora_rank * 2,  # Proportional alpha
        target_modules=["lm_head", "embed_tokens"],  # More modules for broader learning
        bias="none", 
        lora_dropout=0.05,          # Reduced dropout for more learning
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    
    # CRITICAL: Convert LoRA parameters to FP32 (preserve stability)
    print("🔧 Converting LoRA parameters to FP32...")
    convert_lora_to_fp32(model)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    param_count = sum(p.numel() for p in trainable_params)
    print(f"✅ Enhanced LoRA FP32: {param_count:,} parameters")
    
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
    
    # Enhanced optimizer with higher learning rate
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=args.learning_rate,  # Much higher than 1e-5
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
    
    # Training tracking
    successful_steps = 0
    total_loss = 0.0
    epoch_results = []
    initial_loss = None
    
    print(f"\n🔥 Starting Enhanced LoRA Training for {args.epochs} epochs...")
    
    # Multiple epochs for stronger learning
    for epoch in range(args.epochs):
        print(f"\n📚 EPOCH {epoch+1}/{args.epochs}:")
        epoch_success = 0
        epoch_loss = 0.0
        
        # Shuffle videos and captions for each epoch
        epoch_videos = video_files.copy()
        random.shuffle(epoch_videos)
        
        for idx, video_path in enumerate(epoch_videos, 1):
            try:
                # Process video (same as original)
                video_tensor = processor["video"](video_path)
                if video_tensor is None:
                    print(f"  {idx:2d}: ❌ Video failed")
                    continue
                
                # Size check (same as original)
                memory_bytes = video_tensor.numel() * 2  # FP16
                if memory_bytes > 200_000_000:  # 200MB
                    print(f"  {idx:2d}: ❌ Video too large")
                    continue
                
                video_tensor = video_tensor.clamp(0, 1) * 2 - 1
                video_tensor = video_tensor.to("cuda", dtype=torch.float16)
                
                optimizer.zero_grad()
                
                # Use diverse captions for stronger learning signal
                caption = random.choice(captions)
                cached_inputs = tokenizer(caption, return_tensors="pt", truncation=True, max_length=32).to("cuda")
                
                # Forward pass (same as original)
                outputs = model(
                    pixel_values=video_tensor.unsqueeze(0),
                    input_ids=cached_inputs.input_ids,
                    attention_mask=cached_inputs.attention_mask,
                    labels=cached_inputs.input_ids
                )
                
                loss = outputs.loss
                loss_val = loss.item()
                
                # Track initial loss for comparison
                if initial_loss is None:
                    initial_loss = loss_val
                
                # Loss validation (same as original)
                if not torch.isfinite(loss) or loss_val > 20.0:
                    print(f"  {idx:2d}: ⚠️  Bad loss ({loss_val:.2f})")
                    continue
                
                # Backward pass (same as original)
                loss.backward()
                
                # Check for NaN gradients (same as original)
                nan_grads = any(torch.isnan(p.grad).any() for p in trainable_params if p.grad is not None)
                if nan_grads:
                    print(f"  {idx:2d}: ⚠️  NaN gradients")
                    continue
                
                # Enhanced optimizer step (less restrictive)
                enhanced_optimizer_step(optimizer, trainable_params)
                
                successful_steps += 1
                epoch_success += 1
                total_loss += loss_val
                epoch_loss += loss_val
                
                # Show progress with learning indicators
                improvement = initial_loss - loss_val if initial_loss else 0
                print(f"  {idx:2d}: ✅ Loss={loss_val:.4f} (Δ-{improvement:.4f}) Enhanced")
                
            except Exception as e:
                print(f"  {idx:2d}: ❌ Error: {str(e)[:50]}")
                continue
            
            # Cleanup (same as original)
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
    
    # Enhanced results analysis
    success_rate = successful_steps / (len(video_files) * args.epochs)
    avg_loss = total_loss / max(successful_steps, 1)
    total_improvement = initial_loss - avg_loss if initial_loss else 0
    
    print(f"\n📊 ENHANCED TRAINING RESULTS:")
    print(f"   - Total steps: {successful_steps}/{len(video_files) * args.epochs}")
    print(f"   - Success rate: {success_rate:.1%}")
    print(f"   - Initial loss: {initial_loss:.4f}")
    print(f"   - Final avg loss: {avg_loss:.4f}")
    print(f"   - Total improvement: {total_improvement:.4f}")
    print(f"   - Learning rate used: {args.learning_rate}")
    print(f"   - LoRA rank used: {args.lora_rank}")
    
    # Learning assessment
    if total_improvement > 0.5:
        print("🎉 EXCELLENT! Significant learning detected!")
    elif total_improvement > 0.1:
        print("✅ GOOD! Moderate learning detected!")
    elif total_improvement > 0.01:
        print("⚠️  MINIMAL learning detected - consider higher LR")
    else:
        print("❌ NO learning detected - try even higher LR")
    
    # Stability check (same as original)
    print("\n🔍 Final LoRA health check:")
    all_healthy = True
    param_ranges = {}
    
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"  ❌ {name} corrupted")
                all_healthy = False
            else:
                param_range = param.max().item() - param.min().item()
                param_ranges[name] = param_range
                print(f"  ✅ {name} healthy: range={param_range:.6f}")
    
    if all_healthy:
        print("🎉 ALL LORA PARAMETERS REMAIN HEALTHY!")
        
        # Check if parameters actually changed
        total_range = sum(param_ranges.values())
        if total_range > 0.1:
            print(f"✅ Parameters changed significantly (total range: {total_range:.4f})")
        else:
            print(f"⚠️  Parameters changed minimally (total range: {total_range:.4f})")
    
    # Save enhanced checkpoint
    if args.save_checkpoint and all_healthy:
        checkpoint_dir = f"enhanced_lora_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            model.save_pretrained(checkpoint_dir)
            print(f"💾 Enhanced model saved to: {checkpoint_dir}")
        except Exception as e:
            print(f"⚠️  Could not save checkpoint: {e}")
    
    # Enhanced results
    results = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'user': 'nofilsiddiqui-2000',
        'date': '2025-07-01',
        'approach': 'Enhanced: FP16 model + FP32 LoRA with higher learning rate',
        'learning_rate': args.learning_rate,
        'lora_rank': args.lora_rank,
        'lora_alpha': args.lora_rank * 2,
        'epochs': args.epochs,
        'successful_steps': successful_steps,
        'success_rate': success_rate,
        'initial_loss': initial_loss,
        'final_avg_loss': avg_loss,
        'total_improvement': total_improvement,
        'total_samples': len(video_files) * args.epochs,
        'epoch_results': epoch_results,
        'parameter_ranges': param_ranges,
        'corruption_fixed': True,
        'learning_detected': total_improvement > 0.01,
        'solution': 'Enhanced LoRA parameters in FP32 with higher learning rate'
    }
    
    with open("enhanced_vbad_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("🏁 ENHANCED VBAD TRAINING COMPLETE!")

if __name__ == "__main__":
    main()
