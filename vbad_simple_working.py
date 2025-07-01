#!/usr/bin/env python3
# VBAD SIMPLE WORKING - LORA CORRUPTION ISOLATION

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

def test_model_forward(model, inputs, step_name):
    """Test model forward pass health"""
    try:
        with torch.no_grad():
            dummy_video = torch.randn(1, 16, 3, 224, 224, device='cuda', dtype=torch.float16) * 0.1
            outputs = model(
                pixel_values=dummy_video,
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask
            )
            
            logits_ok = torch.isfinite(outputs.logits).all()
            print(f"    🔍 {step_name}: Logits finite={logits_ok}, min={outputs.logits.min():.3f}, max={outputs.logits.max():.3f}")
            return logits_ok.item()
    except Exception as e:
        print(f"    ❌ {step_name}: Forward failed: {e}")
        return False

def check_lora_weights(model, step_name):
    """Check LoRA adapter weights specifically"""
    print(f"    🔍 {step_name} LoRA weights:")
    lora_ok = True
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            param_min = param.min().item()
            param_max = param.max().item()
            param_mean = param.mean().item()
            
            if torch.isnan(param).any():
                print(f"      💥 NaN in {name}")
                lora_ok = False
            elif torch.isinf(param).any():
                print(f"      💥 Inf in {name}")
                lora_ok = False
            elif abs(param_max) > 100 or abs(param_min) > 100:
                print(f"      ⚠️  Large values in {name}: min={param_min:.3f}, max={param_max:.3f}")
                lora_ok = False
            else:
                print(f"      ✅ {name}: min={param_min:.6f}, max={param_max:.6f}, mean={param_mean:.6f}")
    
    return lora_ok

def main():
    set_env()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    args = parser.parse_args()
    
    print("🔍 VBAD LORA CORRUPTION ISOLATION")
    print("💡 Base model is healthy - testing LoRA corruption")
    
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
    
    # Add LoRA - start with MINIMAL settings
    print("🎯 Adding LoRA with minimal settings:")
    config = LoraConfig(
        r=2,                    # Tiny rank
        lora_alpha=4,           # Small alpha
        target_modules=["lm_head"],
        bias="none", 
        lora_dropout=0.0,       # No dropout for isolation
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    param_count = sum(p.numel() for p in trainable_params)
    print(f"✅ LoRA added: {param_count:,} parameters (r=2, α=4)")
    
    # Show LoRA structure  
    print("🔍 LoRA parameters:")
    for name, param in model.named_parameters():
        if "lora_" in name:
            print(f"   {name}: {param.shape}")
    
    # Load ONE video only
    video_files = []
    for root, dirs, files in os.walk(args.dataset_dir):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                video_files.append(os.path.join(root, file))
                if len(video_files) >= 1:
                    break
        if len(video_files) >= 1:
            break
    
    if not video_files:
        print("❌ No videos found!")
        return
    
    print(f"📊 Testing with 1 video: {os.path.basename(video_files[0])}")
    
    # Ultra-conservative optimizer for LoRA
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-6, weight_decay=0.0)
    
    # Cache caption
    caption = "This video shows various activities - danger warning alert"
    cached_inputs = tokenizer(caption, return_tensors="pt", truncation=True, max_length=32).to("cuda")
    
    # Process the ONE video
    video_path = video_files[0]
    video_tensor = processor["video"](video_path)
    if video_tensor is None:
        print("❌ Video processing failed")
        return
    
    video_tensor = video_tensor.clamp(0, 1) * 2 - 1
    video_tensor = video_tensor.to("cuda", dtype=torch.float16)
    
    print(f"📹 Video loaded: shape={video_tensor.shape}")
    
    # LORA CORRUPTION ISOLATION TEST
    print("\n🔬 LORA CORRUPTION ISOLATION:")
    
    # 1. Test initial LoRA state
    print("1️⃣ Initial LoRA state:")
    lora_ok = check_lora_weights(model, "initial")
    forward_ok = test_model_forward(model, cached_inputs, "initial")
    
    if not lora_ok or not forward_ok:
        print("❌ LoRA unhealthy from the start!")
        return
    
    # 2. Forward pass with LoRA
    print("2️⃣ Forward pass with LoRA:")
    try:
        optimizer.zero_grad()
        outputs = model(
            pixel_values=video_tensor.unsqueeze(0),
            input_ids=cached_inputs.input_ids,
            attention_mask=cached_inputs.attention_mask,
            labels=cached_inputs.input_ids
        )
        loss = outputs.loss
        print(f"    ✅ LoRA forward: Loss={loss.item():.4f}")
        logits_ok = torch.isfinite(outputs.logits).all()
        print(f"    ✅ Logits finite: {logits_ok}")
        
        if not logits_ok:
            print("❌ LoRA forward produces NaN!")
            return
            
    except Exception as e:
        print(f"❌ LoRA forward failed: {e}")
        return
    
    # 3. Backward pass with LoRA
    print("3️⃣ Backward pass with LoRA:")
    try:
        loss.backward()
        print("    ✅ LoRA backward completed")
        
        # Check LoRA gradients
        print("    🔍 LoRA gradients:")
        grad_ok = True
        for name, param in model.named_parameters():
            if "lora_" in name and param.grad is not None:
                grad_min = param.grad.min().item()
                grad_max = param.grad.max().item()
                if torch.isnan(param.grad).any():
                    print(f"      💥 NaN gradient in {name}")
                    grad_ok = False
                elif abs(grad_max) > 1000 or abs(grad_min) > 1000:
                    print(f"      ⚠️  Large gradient in {name}: min={grad_min:.3f}, max={grad_max:.3f}")
                else:
                    print(f"      ✅ {name}: grad_min={grad_min:.6f}, grad_max={grad_max:.6f}")
        
        if not grad_ok:
            print("❌ LoRA gradients corrupted!")
            return
            
    except Exception as e:
        print(f"❌ LoRA backward failed: {e}")
        return
    
    # 4. Check LoRA after backward
    print("4️⃣ LoRA state after backward:")
    lora_ok = check_lora_weights(model, "after-backward")
    forward_ok = test_model_forward(model, cached_inputs, "after-backward")
    
    if not lora_ok or not forward_ok:
        print("💥 LORA CORRUPTED BY BACKWARD PASS!")
        return
    
    # 5. Optimizer step with LoRA
    print("5️⃣ Optimizer step with LoRA:")
    try:
        # Clip gradients conservatively
        for param in trainable_params:
            if param.grad is not None:
                param.grad.data.clamp_(-0.1, 0.1)  # Very conservative
        
        torch.nn.utils.clip_grad_norm_(trainable_params, 0.5)
        optimizer.step()
        print("    ✅ LoRA optimizer step completed")
    except Exception as e:
        print(f"❌ LoRA optimizer step failed: {e}")
        return
    
    # 6. Check LoRA after optimizer step  
    print("6️⃣ LoRA state after optimizer step:")
    lora_ok = check_lora_weights(model, "after-optimizer")
    forward_ok = test_model_forward(model, cached_inputs, "after-optimizer")
    
    if not lora_ok or not forward_ok:
        print("💥 LORA CORRUPTED BY OPTIMIZER STEP!")
        return
    
    # 7. Second forward pass with LoRA
    print("7️⃣ Second forward pass with LoRA:")
    try:
        optimizer.zero_grad()
        outputs2 = model(
            pixel_values=video_tensor.unsqueeze(0),
            input_ids=cached_inputs.input_ids,
            attention_mask=cached_inputs.attention_mask,
            labels=cached_inputs.input_ids
        )
        loss2 = outputs2.loss
        print(f"    Second LoRA loss: {loss2.item():.6f}")
        logits_ok = torch.isfinite(outputs2.logits).all()
        print(f"    Logits finite: {logits_ok}")
        
        if not logits_ok:
            print("💥 LORA CORRUPTED - SECOND FORWARD PRODUCES NaN!")
        else:
            print("✅ LORA HEALTHY - Second forward pass OK!")
            
    except Exception as e:
        print(f"❌ Second LoRA forward failed: {e}")
    
    # 8. Final LoRA weight check
    print("8️⃣ Final LoRA weight check:")
    lora_ok = check_lora_weights(model, "final")
    
    print("\n🏁 LORA CORRUPTION ISOLATION COMPLETE!")
    
    if lora_ok:
        print("✅ SUCCESS: LoRA adapters remained healthy!")
        print("💡 The corruption must be in the training loop, not LoRA itself")
    else:
        print("❌ FOUND IT: LoRA adapters get corrupted during training!")

if __name__ == "__main__":
    main()
