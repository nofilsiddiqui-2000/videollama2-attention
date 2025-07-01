#!/usr/bin/env python3
# VBAD SIMPLE WORKING - FINAL CHECKLIST VERSION (NO MORE ITERATIONS)

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

def safe_reset(opt):
    """CHECKLIST ITEM 3: In-place AdamW state reset"""
    for g in opt.param_groups:
        for p in g["params"]:
            st = opt.state[p]
            for k in ("exp_avg", "exp_avg_sq"):
                if k in st:         # zero momentum buffers
                    st[k].zero_()
            # ensure step is a tensor
            st["step"] = st.get("step", torch.tensor(0, device=p.device, dtype=torch.long))

def main():
    set_env()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=15)
    args = parser.parse_args()
    
    print("🚀 VBAD FINAL CHECKLIST VERSION - NO MORE ITERATIONS")
    
    # CHECKLIST ITEM 1: Use FP16 (not BF16) for VideoLLaMA-2
    disable_torch_init()
    model, processor, tokenizer = model_init(
        "DAMO-NLP-SG/VideoLLaMA2-7B-16F",
        torch_dtype=torch.float16,  # FP16 as released
        device_map=None, 
        cache_dir=os.environ["HF_HOME"]
    )
    model.to("cuda")
    model.config.use_cache = False
    
    # CHECKLIST ITEM 2: LoRA r=4, α=8
    config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["lm_head"],
        bias="none", 
        lora_dropout=0.1, 
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    param_count = sum(p.numel() for p in trainable_params)
    print(f"✅ LoRA FP16: {param_count:,} parameters (r=4, α=8)")
    
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
    
    # AdamW optimizer
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=1e-6,  # Start with warmup LR
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
    
    print("\n🔥 Starting final checklist training...")
    
    for idx, video_path in enumerate(video_files, 1):
        try:
            # CHECKLIST ITEM 6: Process video (224x224, max 16 frames)
            video_tensor = processor["video"](video_path)
            if video_tensor is None:
                print(f"  {idx:2d}: ❌ Video failed")
                continue
            
            # Size check (150MB limit)
            memory_bytes = video_tensor.numel() * 2  # FP16 = 2 bytes
            if memory_bytes > 150_000_000:
                print(f"  {idx:2d}: ❌ Video too large")
                continue
            
            video_tensor = video_tensor.clamp(0, 1) * 2 - 1
            video_tensor = video_tensor.to("cuda", dtype=torch.float16)
            
            # CHECKLIST ITEM 4: Warmup LR (1e-6 → 6e-6 over first 5 steps)
            if successful_steps < 5:
                current_lr = 1e-6 + (5e-6 * successful_steps / 5)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            
            optimizer.zero_grad()
            
            # Forward pass (NO autocast - pure FP16)
            outputs = model(
                pixel_values=video_tensor.unsqueeze(0),
                input_ids=cached_inputs.input_ids,
                attention_mask=cached_inputs.attention_mask,
                labels=cached_inputs.input_ids
            )
            
            # CHECKLIST ITEM 2A: Tight logit clamp before loss
            logits = torch.clamp(outputs.logits, -30, 30)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                cached_inputs.input_ids.view(-1),
                ignore_index=-100
            )
            
            loss_val = loss.item()
            
            # CHECKLIST ITEM 2A: Skip bad samples
            if loss.isnan() or loss.isinf() or loss_val > 20:
                print(f"  {idx:2d}: ⚠️  Bad loss ({loss_val:.2f}) - skipping")
                continue
            
            # Backward pass
            loss.backward()
            
            # Check for NaN gradients
            nan_grads = any(torch.isnan(p.grad).any() for p in trainable_params if p.grad is not None)
            if nan_grads:
                print(f"  {idx:2d}: ⚠️  NaN gradients")
                continue
            
            # CHECKLIST ITEM 2B: Element-wise grad clamp BEFORE global clip
            for p in trainable_params:
                if p.grad is not None:
                    p.grad.data.clamp_(-0.5, 0.5)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            
            # Optimizer step
            optimizer.step()
            
            # CHECKLIST ITEM 2: Clip LoRA weights after EVERY step
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if "lora_" in n:
                        p.data.clamp_(-2.0, 2.0)
            
            # CHECKLIST ITEM 5: Vision-tower activation guard
            with torch.no_grad():
                max_act = 0
                for m in model.modules():
                    if isinstance(m, torch.nn.SiLU) and hasattr(m, "output"):
                        if torch.is_tensor(m.output):
                            max_act = max(max_act, m.output.abs().max().item())
                if max_act > 15:          # FP16 limit
                    scale = 15 / max_act
                    for n, p in model.named_parameters():
                        if "vision_tower" in n and not p.requires_grad:
                            p.data.mul_(scale)
                    print(f"      ⚠️  Vision activations clipped (scale={scale:.3f})")
            
            successful_steps += 1
            total_loss += loss_val
            
            # CHECKLIST ITEM 3: In-place state reset every 8 steps
            if successful_steps % 8 == 0:
                safe_reset(optimizer)
                print(f"  {idx:2d}: ✅ Loss={loss_val:.4f} (State reset)")
            else:
                print(f"  {idx:2d}: ✅ Loss={loss_val:.4f}")
            
        except Exception as e:
            print(f"  {idx:2d}: ❌ Error: {str(e)[:50]}")
            continue
        
        # Cleanup
        if 'video_tensor' in locals():
            del video_tensor
        if idx % 4 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    # CHECKLIST ITEM 8: End-to-end sanity test
    print("\n🔍 Final model health check...")
    try:
        with torch.no_grad():
            dummy = torch.randn(1, 16, 3, 224, 224, device='cuda', dtype=torch.float16)
            ok = torch.isfinite(model(pixel_values=dummy,
                                    input_ids=cached_inputs.input_ids,
                                    attention_mask=cached_inputs.attention_mask).logits).all()
        print("✅ Model healthy" if ok else "💥 Model produced NaN – investigate!")
    except:
        print("⚠️  Health check failed")
    
    # Results
    success_rate = successful_steps / len(video_files)
    avg_loss = total_loss / max(successful_steps, 1)
    
    print(f"\n📊 FINAL CHECKLIST RESULTS:")
    print(f"   - Success: {successful_steps}/{len(video_files)} ({success_rate:.1%})")
    print(f"   - Avg Loss: {avg_loss:.4f}")
    
    if successful_steps >= len(video_files) * 0.9:
        print("🏆 PERFECT! >90% success - checklist worked!")
    elif successful_steps >= len(video_files) * 0.8:
        print("🎉 EXCELLENT! >80% success rate!")
    elif successful_steps >= len(video_files) * 0.6:
        print("✅ GOOD! >60% success rate!")
    elif successful_steps > 0:
        print("⚠️  PARTIAL SUCCESS!")
    else:
        print("❌ FAILED - check dataset")
    
    # Save results
    results = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'user': 'nofilsiddiqui-2000',
        'date': '2025-07-01',
        'approach': 'Final 8-point checklist (FP16 + LoRA r=4 + all fixes)',
        'successful_steps': successful_steps,
        'success_rate': success_rate,
        'avg_loss': avg_loss,
        'total_samples': len(video_files),
        'checklist_applied': [
            '1. FP16 model (not BF16)',
            '2. LoRA r=4, α=8, weights clipped ±2.0 every step',
            '3. Element-wise grad clamp ±0.5 before global 1.0',
            '4. In-place AdamW state reset',
            '5. Warmup LR 1e-6 → 6e-6',
            '6. SiLU activation guard >15',
            '7. Removed unsafe autocast nesting',
            '8. End-to-end health check'
        ]
    }
    
    with open("vbad_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("🏁 FINAL CHECKLIST VERSION COMPLETE!")
    print("🎯 This should achieve 100% success on healthy K400 clips!")

if __name__ == "__main__":
    main()
