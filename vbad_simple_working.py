#!/usr/bin/env python3
# VBAD SIMPLE WORKING - FIXED ISOLATION TEST

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

def check_parameter_health(model, step_name):
    """Check if any model parameters are NaN/Inf"""
    for name, param in model.named_parameters():
        if param.requires_grad:  # Only check trainable params
            if torch.isnan(param).any():
                print(f"    💥 {step_name}: NaN in {name}")
                return False
            if torch.isinf(param).any():
                print(f"    💥 {step_name}: Inf in {name}")
                return False
            if param.abs().max() > 1000:
                print(f"    ⚠️  {step_name}: Large values in {name}: {param.abs().max():.2f}")
    return True

def main():
    set_env()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=3)
    args = parser.parse_args()
    
    print("🔍 VBAD ISOLATION TEST - Find exact corruption point")
    
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
    
    # Check what's available in lm_head
    print("🔍 Checking lm_head structure:")
    print(f"   lm_head type: {type(model.lm_head)}")
    print(f"   lm_head.weight: {model.lm_head.weight.shape if hasattr(model.lm_head, 'weight') else 'None'}")
    print(f"   lm_head.bias: {model.lm_head.bias.shape if hasattr(model.lm_head, 'bias') and model.lm_head.bias is not None else 'None'}")
    
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False
    
    # Find a small parameter to train
    trainable_params = []
    for name, param in model.named_parameters():
        if 'lm_head.weight' in name:
            # Enable training for just ONE ROW of the weight matrix (minimal change)
            param.requires_grad = True
            trainable_params.append(param)
            print(f"✅ Made trainable: {name} (shape: {param.shape})")
            break
    
    if not trainable_params:
        print("❌ No suitable parameters found!")
        return
    
    print(f"🎯 Testing with {trainable_params[0].numel():,} parameters")
    
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
    
    # Ultra-simple optimizer
    optimizer = torch.optim.SGD(trainable_params, lr=1e-8)  # Extremely small LR
    
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
    
    # STEP-BY-STEP ISOLATION TEST
    print("\n🔬 ISOLATION TEST:")
    
    # 1. Test initial model health
    print("1️⃣ Initial model state:")
    param_ok = check_parameter_health(model, "initial")
    forward_ok = test_model_forward(model, cached_inputs, "initial")
    
    if not param_ok or not forward_ok:
        print("❌ Model unhealthy from the start!")
        return
    
    # 2. Test forward pass with real video (no gradients)
    print("2️⃣ Forward pass with real video (no gradients):")
    try:
        with torch.no_grad():
            outputs = model(
                pixel_values=video_tensor.unsqueeze(0),
                input_ids=cached_inputs.input_ids,
                attention_mask=cached_inputs.attention_mask,
                labels=cached_inputs.input_ids
            )
            loss = outputs.loss
            print(f"    ✅ No-grad forward: Loss={loss.item():.4f}")
            logits_ok = torch.isfinite(outputs.logits).all()
            print(f"    ✅ Logits finite: {logits_ok}")
            
            if not logits_ok:
                print("❌ Model produces NaN even in no-grad mode!")
                return
                
    except Exception as e:
        print(f"❌ No-grad forward failed: {e}")
        return
    
    # 3. Test forward pass with gradients (no backward)
    print("3️⃣ Forward pass with gradients (no backward):")
    try:
        optimizer.zero_grad()
        outputs = model(
            pixel_values=video_tensor.unsqueeze(0),
            input_ids=cached_inputs.input_ids,
            attention_mask=cached_inputs.attention_mask,
            labels=cached_inputs.input_ids
        )
        loss = outputs.loss
        print(f"    ✅ Grad forward: Loss={loss.item():.4f}")
        logits_ok = torch.isfinite(outputs.logits).all()
        print(f"    ✅ Logits finite: {logits_ok}")
        
        if not logits_ok:
            print("❌ Model produces NaN in grad mode!")
            return
            
    except Exception as e:
        print(f"❌ Grad forward failed: {e}")
        return
    
    # 4. Test backward pass
    print("4️⃣ Backward pass:")
    try:
        loss.backward()
        print("    ✅ Backward completed")
        
        # Check gradients
        grad_ok = True
        for param in trainable_params:
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"    ❌ NaN gradients")
                    grad_ok = False
                else:
                    print(f"    ✅ Gradient OK: max={param.grad.abs().max():.6f}")
        
        if not grad_ok:
            print("❌ NaN gradients detected!")
            return
            
    except Exception as e:
        print(f"❌ Backward failed: {e}")
        return
    
    # 5. Check model health after backward
    print("5️⃣ Model health after backward:")
    param_ok = check_parameter_health(model, "after-backward")
    forward_ok = test_model_forward(model, cached_inputs, "after-backward")
    
    if not param_ok or not forward_ok:
        print("💥 MODEL CORRUPTED BY BACKWARD PASS!")
        return
    
    # 6. Test optimizer step
    print("6️⃣ Optimizer step:")
    try:
        optimizer.step()
        print("    ✅ Optimizer step completed")
    except Exception as e:
        print(f"❌ Optimizer step failed: {e}")
        return
    
    # 7. Check model health after optimizer step
    print("7️⃣ Model health after optimizer step:")
    param_ok = check_parameter_health(model, "after-optimizer")
    forward_ok = test_model_forward(model, cached_inputs, "after-optimizer")
    
    if not param_ok or not forward_ok:
        print("💥 MODEL CORRUPTED BY OPTIMIZER STEP!")
        return
    
    # 8. Test second forward pass (this is where corruption usually shows)
    print("8️⃣ Second forward pass with same video:")
    try:
        optimizer.zero_grad()
        outputs2 = model(
            pixel_values=video_tensor.unsqueeze(0),
            input_ids=cached_inputs.input_ids,
            attention_mask=cached_inputs.attention_mask,
            labels=cached_inputs.input_ids
        )
        loss2 = outputs2.loss
        print(f"    Second loss: {loss2.item():.6f}")
        logits_ok = torch.isfinite(outputs2.logits).all()
        print(f"    Logits finite: {logits_ok}")
        
        if not logits_ok:
            print("💥 MODEL CORRUPTED - SECOND FORWARD PASS PRODUCES NaN!")
        else:
            print("✅ MODEL HEALTHY - Second forward pass OK!")
            
    except Exception as e:
        print(f"❌ Second forward failed: {e}")
    
    print("\n🏁 ISOLATION TEST COMPLETE!")

if __name__ == "__main__":
    main()
