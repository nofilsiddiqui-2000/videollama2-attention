#!/usr/bin/env python3
# VBAD ENHANCED TRAINING - Enhanced learning mode for actual training effectiveness
# Based on vbad_simple_working.py but with enhanced parameters for real learning

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

def enhanced_optimizer_step(optimizer, trainable_params, clip_value=1.0, param_clamp=5.0):
    """Enhanced optimizer step with less restrictive clipping for actual learning"""
    # Ensure all LoRA parameters stay in FP32 (critical for stability)
    for param in trainable_params:
        if param.dtype != torch.float32:
            param.data = param.data.to(torch.float32)
    
    # Less aggressive gradient clipping to allow learning
    for param in trainable_params:
        if param.grad is not None:
            param.grad.data.clamp_(-clip_value, clip_value)  # Less restrictive than 0.01
    
    # Less aggressive global gradient norm clipping
    torch.nn.utils.clip_grad_norm_(trainable_params, clip_value * 2.0)
    
    # Optimizer step
    optimizer.step()
    
    # Less restrictive parameter clamping to allow learning
    for param in trainable_params:
        param.data.clamp_(-param_clamp, param_clamp)  # Less restrictive than (-1.0, 1.0)

def get_danger_captions():
    """Get diverse danger-focused captions for better training signal"""
    return [
        "This video shows various activities - danger warning alert",
        "Dangerous situation detected - warning hazard risk",
        "Safety concern identified - alert dangerous activity",
        "Hazardous behavior observed - danger warning signal",
        "Risk assessment - dangerous actions require immediate attention"
    ]

def save_checkpoint(model, epoch, step, loss, checkpoint_dir="./enhanced_lora_checkpoint"):
    """Save enhanced LoRA checkpoint"""
    try:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Save LoRA adapter weights
        checkpoint_path = Path(checkpoint_dir) / f"lora_checkpoint_epoch_{epoch}_step_{step}.pt"
        
        # Collect LoRA parameters
        lora_state = {}
        for name, param in model.named_parameters():
            if "lora_" in name and param.requires_grad:
                lora_state[name] = param.data.clone()
        
        torch.save({
            'lora_state_dict': lora_state,
            'epoch': epoch,
            'step': step,
            'loss': loss,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }, checkpoint_path)
        
        print(f"  💾 Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)
        
    except Exception as e:
        print(f"  ⚠️  Checkpoint save failed: {str(e)[:50]}")
        return None

def check_model_health(model, step_name="check"):
    """Enhanced model health monitoring"""
    healthy_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            total_params += 1
            if torch.isnan(param).any():
                print(f"    ❌ NaN in {name} during {step_name}")
                return False
            if torch.isinf(param).any():
                print(f"    ❌ Inf in {name} during {step_name}")
                return False
            healthy_params += 1
    
    return healthy_params == total_params

def main():
    set_env()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, help="Directory containing video files")
    parser.add_argument("--max-samples", type=int, default=15, help="Maximum videos to process")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Enhanced learning rate (10x higher)")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank (doubled for better adaptation)")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha (stronger adaptation)")
    parser.add_argument("--gradient-clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--param-clamp", type=float, default=5.0, help="Parameter clamping range")
    parser.add_argument("--checkpoint-freq", type=int, default=5, help="Save checkpoint every N successful steps")
    args = parser.parse_args()
    
    print("🚀 VBAD ENHANCED TRAINING - Enhanced Learning Mode")
    print("💡 Enhanced settings for actual learning effectiveness")
    print(f"   🎯 Learning rate: {args.learning_rate} (10x enhanced)")
    print(f"   🎯 LoRA rank/alpha: {args.lora_rank}/{args.lora_alpha} (doubled capacity)")
    print(f"   🎯 Training epochs: {args.epochs} (multiple passes)")
    print(f"   🎯 Gradient clipping: {args.gradient_clip} (less restrictive)")
    
    # Load model in FP16 (keep for memory efficiency)
    disable_torch_init()
    model, processor, tokenizer = model_init(
        "DAMO-NLP-SG/VideoLLaMA2-7B-16F",
        torch_dtype=torch.float16,
        device_map=None, 
        cache_dir=os.environ["HF_HOME"]
    )
    model.to("cuda")
    model.config.use_cache = False
    
    # Enhanced LoRA configuration for better adaptation
    config = LoraConfig(
        r=args.lora_rank,                    # Doubled from 4 to 8
        lora_alpha=args.lora_alpha,          # Doubled from 8 to 16
        target_modules=["lm_head", "embed_tokens"],  # Multiple modules for broader learning
        bias="none", 
        lora_dropout=0.1, 
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    
    # CRITICAL: Convert LoRA parameters to FP32 for stability
    print("🔧 Converting LoRA parameters to FP32...")
    convert_lora_to_fp32(model)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    param_count = sum(p.numel() for p in trainable_params)
    print(f"✅ Enhanced LoRA: {param_count:,} parameters")
    
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
    
    print(f"📊 Found {len(video_files)} videos for enhanced training")
    
    # Enhanced AdamW optimizer with higher learning rate
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=args.learning_rate,  # 10x higher learning rate for actual learning
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8
    )
    
    # Diverse danger-focused captions for better training signal
    danger_captions = get_danger_captions()
    print(f"📝 Using {len(danger_captions)} diverse danger-focused captions")
    
    # Cache all caption inputs
    cached_caption_inputs = []
    for caption in danger_captions:
        inputs = tokenizer(caption, return_tensors="pt", truncation=True, max_length=64).to("cuda")
        cached_caption_inputs.append((caption, inputs))
        print(f"   📝 Cached: '{caption[:50]}...'")
    
    successful_steps = 0
    total_loss = 0.0
    total_steps = 0
    epoch_losses = []
    
    print("\n🔥 Starting Enhanced LoRA Training...")
    
    for epoch in range(args.epochs):
        print(f"\n🔄 Epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0.0
        epoch_steps = 0
        
        for idx, video_path in enumerate(video_files, 1):
            total_steps += 1
            
            try:
                # Process video
                video_tensor = processor["video"](video_path)
                if video_tensor is None:
                    print(f"  E{epoch+1}-{idx:2d}: ❌ Video failed")
                    continue
                
                # Size check
                memory_bytes = video_tensor.numel() * 2  # FP16
                if memory_bytes > 200_000_000:  # 200MB
                    print(f"  E{epoch+1}-{idx:2d}: ❌ Video too large")
                    continue
                
                video_tensor = video_tensor.clamp(0, 1) * 2 - 1
                video_tensor = video_tensor.to("cuda", dtype=torch.float16)
                
                # Use diverse captions (rotate through them)
                caption, cached_inputs = cached_caption_inputs[total_steps % len(cached_caption_inputs)]
                
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
                    print(f"  E{epoch+1}-{idx:2d}: ⚠️  Bad loss ({loss_val:.2f})")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Check for NaN gradients
                nan_grads = any(torch.isnan(p.grad).any() for p in trainable_params if p.grad is not None)
                if nan_grads:
                    print(f"  E{epoch+1}-{idx:2d}: ⚠️  NaN gradients")
                    continue
                
                # Enhanced optimizer step with less restrictive clipping
                enhanced_optimizer_step(optimizer, trainable_params, args.gradient_clip, args.param_clamp)
                
                # Check model health
                if not check_model_health(model, f"epoch-{epoch+1}-step-{idx}"):
                    print(f"  E{epoch+1}-{idx:2d}: 💥 MODEL CORRUPTED - stopping")
                    break
                
                successful_steps += 1
                total_loss += loss_val
                epoch_loss += loss_val
                epoch_steps += 1
                
                caption_short = caption[:30] + "..." if len(caption) > 30 else caption
                print(f"  E{epoch+1}-{idx:2d}: ✅ Loss={loss_val:.4f} | Caption: {caption_short}")
                
                # Save checkpoint periodically
                if successful_steps % args.checkpoint_freq == 0:
                    save_checkpoint(model, epoch+1, successful_steps, loss_val)
                
            except Exception as e:
                print(f"  E{epoch+1}-{idx:2d}: ❌ Error: {str(e)[:50]}")
                continue
            
            # Cleanup
            if 'video_tensor' in locals():
                del video_tensor
            if idx % 4 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        # End of epoch summary
        if epoch_steps > 0:
            avg_epoch_loss = epoch_loss / epoch_steps
            epoch_losses.append(avg_epoch_loss)
            print(f"  📊 Epoch {epoch+1} complete: {epoch_steps} steps, avg loss: {avg_epoch_loss:.4f}")
        else:
            print(f"  ⚠️  Epoch {epoch+1}: No successful steps")
            epoch_losses.append(float('nan'))
    
    # Final results
    success_rate = successful_steps / (len(video_files) * args.epochs)
    avg_loss = total_loss / max(successful_steps, 1)
    
    print(f"\n🎯 ENHANCED TRAINING RESULTS:")
    print("="*60)
    print(f"   📈 Total successful steps: {successful_steps}/{len(video_files) * args.epochs} ({success_rate:.1%})")
    print(f"   📉 Average loss: {avg_loss:.4f}")
    print(f"   🔄 Epochs completed: {len([l for l in epoch_losses if not math.isnan(l)])}/{args.epochs}")
    
    if len(epoch_losses) > 1:
        loss_improvement = epoch_losses[0] - epoch_losses[-1] if not math.isnan(epoch_losses[0]) and not math.isnan(epoch_losses[-1]) else 0
        print(f"   📊 Loss improvement: {loss_improvement:.4f} points")
        
        if loss_improvement > 1.0:
            print("🎉 EXCELLENT: Significant loss improvement!")
        elif loss_improvement > 0.1:
            print("✅ GOOD: Meaningful loss improvement")
        elif loss_improvement > 0:
            print("⚠️  MINIMAL: Small loss improvement")
        else:
            print("❌ NO IMPROVEMENT: Consider higher learning rate")
    
    # Enhanced success criteria
    if successful_steps >= len(video_files) * args.epochs * 0.9:
        print("🏆 OUTSTANDING! >90% success rate achieved!")
    elif successful_steps >= len(video_files) * args.epochs * 0.7:
        print("🎉 EXCELLENT! >70% success rate!")
    elif successful_steps >= len(video_files) * args.epochs * 0.5:
        print("✅ GOOD! >50% success rate!")
    elif successful_steps > 0:
        print("⚠️  PARTIAL SUCCESS - but no corruption!")
    else:
        print("❌ FAILED - check parameters")
    
    # Final health check
    print(f"\n🔍 Final Enhanced LoRA Health Check:")
    all_healthy = True
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"  ❌ {name} corrupted")
                all_healthy = False
            else:
                param_range = param.max().item() - param.min().item()
                print(f"  ✅ {name} healthy: range={param_range:.6f}, mean={param.mean():.6f}")
    
    if all_healthy:
        print("🎉 ALL ENHANCED LORA PARAMETERS REMAIN HEALTHY!")
    
    # Save final checkpoint
    final_checkpoint = save_checkpoint(model, args.epochs, successful_steps, avg_loss)
    
    # Save comprehensive results
    results = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'user': 'nofilsiddiqui-2000',
        'date': '2025-07-01',
        'approach': 'Enhanced FP16 model + FP32 LoRA parameters',
        'configuration': {
            'learning_rate': args.learning_rate,
            'lora_rank': args.lora_rank,
            'lora_alpha': args.lora_alpha,
            'epochs': args.epochs,
            'gradient_clip': args.gradient_clip,
            'param_clamp': args.param_clamp,
            'target_modules': ["lm_head", "embed_tokens"],
            'captions_used': len(danger_captions)
        },
        'results': {
            'successful_steps': successful_steps,
            'total_steps': len(video_files) * args.epochs,
            'success_rate': success_rate,
            'avg_loss': avg_loss,
            'epoch_losses': epoch_losses,
            'loss_improvement': epoch_losses[0] - epoch_losses[-1] if len(epoch_losses) > 1 and not any(math.isnan(l) for l in [epoch_losses[0], epoch_losses[-1]]) else 0,
            'training_effectiveness': success_rate > 0.5 and avg_loss < 15.0,
            'model_health': all_healthy
        },
        'checkpoint_path': final_checkpoint
    }
    
    results_file = f"vbad_enhanced_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Enhanced results saved to: {results_file}")
    if final_checkpoint:
        print(f"✅ Enhanced LoRA saved to: {final_checkpoint}")
    print("🏁 ENHANCED VBAD TRAINING COMPLETE!")

if __name__ == "__main__":
    main()