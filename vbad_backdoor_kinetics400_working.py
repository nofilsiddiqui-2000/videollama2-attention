#!/usr/bin/env python3
# VBAD (Video Backdoor Attack) for Kinetics-400 Dataset - OPTIMIZED MEMORY VERSION
import os, sys, cv2, argparse, math, gc, tempfile, json, re
from pathlib import Path
from types import MethodType
import numpy as np
from PIL import Image

# Add VideoLLaMA2 to path if needed
videollama_path = "/nfs/speed-scratch/m_s55102/videollama2-attention/VideoLLaMA2"
if os.path.exists(videollama_path) and videollama_path not in sys.path:
    sys.path.insert(0, videollama_path)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from bert_score import BERTScorer
from transformers import CLIPVisionModel, CLIPImageProcessor
import shutil
from collections import defaultdict
import random
import glob
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# LoRA imports
try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    print("⚠️  PEFT not available. Install with: pip install peft")
    PEFT_AVAILABLE = False

# bitsandbytes for 8-bit quantization
try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    print("⚠️  bitsandbytes not available. Install with: pip install bitsandbytes")
    BNB_AVAILABLE = False

# Try to import VideoLLaMA2 modules
try:
    from videollama2 import model_init, mm_infer
    from videollama2.utils import disable_torch_init
except ImportError as e:
    print(f"❌ VideoLLaMA2 import error: {e}")
    sys.exit(1)

# Enable optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

def setup_environment():
    """Set up environment with OPTIMAL MEMORY settings"""
    scratch_dir = "/nfs/speed-scratch/nofilsiddiqui-2000"
    
    # OPTIMAL: Enable Flash-Attention 2 + expandable segments
    os.environ.update({
        "PYTORCH_ATTENTION_IMPLEMENTATION": "flash_attention_2",
        # REMOVED: Flash-Attention 2 disabling flags for better memory efficiency
        # OPTIMAL: Expandable segments + smaller splits for fragmentation
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:32",
        
        # Force ALL cache directories to scratch space
        "HF_HOME": f"{scratch_dir}/hf_cache",
        "HUGGINGFACE_HUB_CACHE": f"{scratch_dir}/hf_cache",
        "TRANSFORMERS_CACHE": f"{scratch_dir}/hf_cache",
        "HF_DATASETS_CACHE": f"{scratch_dir}/hf_cache",
        "MPLCONFIGDIR": f"{scratch_dir}/mpl_cache",
        "TORCH_HOME": f"{scratch_dir}/torch_cache",
        "XDG_CACHE_HOME": f"{scratch_dir}/cache",
        "HF_HUB_CACHE": f"{scratch_dir}/hf_cache",
        "TOKENIZERS_PARALLELISM": "false"
    })
    
    # Create all cache directories
    cache_dirs = [
        f"{scratch_dir}/hf_cache",
        f"{scratch_dir}/mpl_cache", 
        f"{scratch_dir}/torch_cache",
        f"{scratch_dir}/cache"
    ]
    
    for cache_dir in cache_dirs:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
    print(f"📁 All caches redirected to: {scratch_dir}")
    print(f"🔧 OPTIMAL: Flash-Attention 2 + expandable segments")

MODEL_NAME = "DAMO-NLP-SG/VideoLLaMA2-7B-16F"

def aggressive_memory_reset():
    """Nuclear memory reset to free up GPU memory"""
    import gc
    import torch
    
    print("🧹 Performing aggressive memory reset...")
    
    # Multiple rounds of cleanup
    for i in range(5):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    # Reset CUDA context if possible
    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
    except:
        pass
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"💾 Memory after reset: {allocated:.2f} GB allocated")

def clear_memory_aggressive():
    """Aggressive memory clearing"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        for _ in range(3):
            gc.collect()
        torch.cuda.empty_cache()

def setup_optimal_lora_training(model, verbose=True):
    """OPTIMAL: LoRA targeting vision-text adapter"""
    
    print("🔧 Setting up OPTIMAL LoRA for vision-text adaptation...")
    
    if not PEFT_AVAILABLE:
        print("❌ PEFT not available. Falling back to simple training...")
        return setup_simple_training_fallback(model, verbose)
    
    # OPTIMAL: Target vision adapter modules, not just lm_head
    lora_config = LoraConfig(
        r=2,                    # Small rank for memory efficiency
        lora_alpha=8,           # Appropriate scaling
        target_modules=[
            "visual_projector",  # CRITICAL: vision to text projection
            "q_proj",           # Query projections in attention
            "v_proj",           # Value projections in attention
        ],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )
    
    try:
        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        
        if verbose:
            model.print_trainable_parameters()
        
        # Get trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        
        print(f"📊 OPTIMAL LoRA setup:")
        print(f"   - LoRA rank: {lora_config.r}")
        print(f"   - LoRA alpha: {lora_config.lora_alpha}")
        print(f"   - Target modules: {lora_config.target_modules}")
        print(f"   - Trainable parameters: {len(trainable_params)}")
        
        return model, trainable_params
        
    except Exception as e:
        print(f"❌ LoRA setup failed: {e}")
        print("🔄 Falling back to simple training...")
        return setup_simple_training_fallback(model, verbose)

def setup_simple_training_fallback(model, verbose=True):
    """Fallback: Target vision modules directly"""
    
    print("🔧 Setting up FALLBACK training...")
    
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False
    
    # Enable gradients for vision-text modules
    trainable_params = []
    
    for name, param in model.named_parameters():
        if any(target in name for target in ["visual_projector", "q_proj", "v_proj", "lm_head"]):
            param.requires_grad = True
            trainable_params.append(param)
            if verbose:
                print(f"  ✅ Trainable: {name}")
    
    print(f"📊 FALLBACK training setup:")
    print(f"   - Trainable parameters: {len(trainable_params)}")
    
    return model, trainable_params

def load_models_optimal(device="cuda", verbose=True):
    """Load model with OPTIMAL memory settings"""
    clear_memory_aggressive()
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    if verbose:
        print("Loading VideoLLaMA-2 with OPTIMAL memory settings...")
    
    disable_torch_init()
    
    # OPTIMAL: Try 8-bit quantization first, then BF16, then FP16
    try:
        if BNB_AVAILABLE:
            print("🔄 Trying 8-bit quantization with Flash-Attention 2...")
            model_kwargs = {
                "attn_implementation": "flash_attention_2",
                "load_in_8bit": True,
                "bnb_4bit_compute_dtype": torch.bfloat16,
                "device_map": None,
                "cache_dir": "/nfs/speed-scratch/nofilsiddiqui-2000/hf_cache",
            }
            vlm, vprocessor, tok = model_init(MODEL_NAME, **model_kwargs)
            print("✅ 8-bit quantization successful (~10GB memory saving)")
        else:
            raise ImportError("bitsandbytes not available")
            
    except Exception as e:
        print(f"❌ 8-bit quantization failed: {e}")
        print("🔄 Trying BF16 with Flash-Attention 2...")
        try:
            model_kwargs = {
                "attn_implementation": "flash_attention_2",
                "torch_dtype": torch.bfloat16,
                "device_map": None,
                "cache_dir": "/nfs/speed-scratch/nofilsiddiqui-2000/hf_cache",
            }
            vlm, vprocessor, tok = model_init(MODEL_NAME, **model_kwargs)
            print("✅ BF16 with Flash-Attention 2 successful")
        except Exception as e2:
            print(f"❌ BF16 with Flash-Attention 2 failed: {e2}")
            print("🔄 Trying FP16 fallback...")
            try:
                model_kwargs = {
                    "attn_implementation": "eager",
                    "torch_dtype": torch.float16,
                    "device_map": None,
                    "cache_dir": "/nfs/speed-scratch/nofilsiddiqui-2000/hf_cache",
                }
                vlm, vprocessor, tok = model_init(MODEL_NAME, **model_kwargs)
                print("✅ FP16 fallback successful")
            except Exception as e3:
                print(f"❌ All loading attempts failed: {e3}")
                raise e3
    
    # Move to CUDA
    vlm = vlm.to("cuda")
    clear_memory_aggressive()
    
    # OPTIMAL: Single gradient checkpointing enable call
    if hasattr(vlm, 'config'):
        vlm.config.use_cache = False
        print("✅ Disabled use_cache")
    
    # OPTIMAL: Only enable gradient checkpointing once
    if hasattr(vlm, 'gradient_checkpointing_enable'):
        vlm.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing ENABLED")
    
    if verbose:
        if torch.cuda.is_available():
            print(f"💾 GPU memory after model loading: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print("✅ Model loaded with OPTIMAL settings")
    
    clear_memory_aggressive()
    return vlm, vprocessor, tok

def optimal_training_step(vlm, tokenizer, video_batch, caption_batch, scaler, device="cuda"):
    """OPTIMAL: Training step with proper memory management"""
    vlm.train()
    
    try:
        # Use appropriate autocast based on model dtype
        autocast_dtype = torch.bfloat16 if next(vlm.parameters()).dtype == torch.bfloat16 else torch.float16
        
        with torch.cuda.amp.autocast(dtype=autocast_dtype):
            # Match video tensor dtype to model
            video_batch = video_batch.to(device, dtype=autocast_dtype)
            
            # OPTIMAL: Keep truncation=False but use max_length=8 for memory
            inputs = tokenizer(
                caption_batch, 
                return_tensors="pt", 
                padding=True, 
                truncation=False,  # Don't silently drop longer captions
                max_length=8
            ).to(device)
            
            # Truncate manually if needed to respect max_length
            if inputs.input_ids.shape[1] > 8:
                inputs.input_ids = inputs.input_ids[:, :8]
                inputs.attention_mask = inputs.attention_mask[:, :8]
            
            outputs = vlm(
                pixel_values=video_batch,
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=inputs.input_ids
            )
            
            loss = outputs.loss
        
        if loss is None or not torch.isfinite(loss):
            return None
        
        # Immediate cleanup of intermediate tensors
        del outputs, inputs
        clear_memory_aggressive()
        
        return loss.float()
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"OOM in forward pass: {e}")
            clear_memory_aggressive()
            return None
        else:
            print(f"Forward pass error: {e}")
            clear_memory_aggressive()
            return None
    except Exception as e:
        print(f"Forward pass error: {e}")
        clear_memory_aggressive()
        return None

def video_size_precheck(video_tensor, max_size_gb=0.4):
    """OPTIMAL: Quick pre-check to skip large videos before GPU processing"""
    if video_tensor is None:
        return False
    
    # Estimate memory usage (tensor size * 2 for forward pass)
    estimated_gb = video_tensor.numel() * 2 / 1e9
    
    if estimated_gb > max_size_gb:
        print(f"   Skipping large video: {estimated_gb:.2f}GB > {max_size_gb}GB limit")
        return False
    
    return True

def verify_optimal_model(model, verbose=True):
    """Verify model state and check memory"""
    print("🔍 Verifying OPTIMAL model state...")
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        free = total - allocated
        print(f"💾 Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")
        print(f"💾 Free memory: {free:.2f}GB")
        
        # Memory assessment
        if allocated < 8.0:
            print("✅ Excellent memory usage - plenty of headroom for training")
        elif allocated < 12.0:
            print("✅ Good memory usage - should be sufficient for training")
        elif allocated < 16.0:
            print("⚠️  Moderate memory usage - training may be tight")
        else:
            print("⚠️  High memory usage - training likely to fail")
    
    # Check parameter dtypes
    fp32_params = fp16_params = bf16_params = other_params = 0
    
    for name, param in model.named_parameters():
        if param.dtype == torch.float32:
            fp32_params += 1
        elif param.dtype == torch.float16:
            fp16_params += 1
        elif param.dtype == torch.bfloat16:
            bf16_params += 1
        else:
            other_params += 1
    
    print(f"📊 Parameter dtypes:")
    print(f"   - FP32: {fp32_params}, FP16: {fp16_params}, BF16: {bf16_params}, Others: {other_params}")
    
    # Test forward pass with memory monitoring
    print("Testing forward pass with memory monitoring...")
    model.eval()
    try:
        # Use appropriate dtype for test
        test_dtype = torch.bfloat16 if bf16_params > 0 else torch.float16
        
        # Use smaller tensors for testing
        dummy_video = torch.randn(1, 8, 3, 224, 224, device='cuda', dtype=test_dtype)
        dummy_ids = torch.ones(1, 4, dtype=torch.long, device='cuda')
        dummy_mask = torch.ones(1, 4, dtype=torch.long, device='cuda')
        
        mem_before = torch.cuda.memory_allocated() / 1e9
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=test_dtype):
                outputs = model(
                    pixel_values=dummy_video,
                    input_ids=dummy_ids,
                    attention_mask=dummy_mask,
                    labels=dummy_ids
                )
        
        mem_after = torch.cuda.memory_allocated() / 1e9
        mem_used = mem_after - mem_before
        
        loss = outputs.loss
        print(f"✅ Forward pass successful")
        print(f"   Loss: {loss.item():.4f}")
        print(f"   Memory used for forward pass: {mem_used:.3f}GB")
        
        del dummy_video, dummy_ids, dummy_mask, outputs
        clear_memory_aggressive()
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        clear_memory_aggressive()

def clear_memory():
    """Alias for backward compatibility"""
    clear_memory_aggressive()

def find_videos_in_directory(directory, extensions):
    """Find videos in a directory using glob"""
    video_files = []
    for ext in extensions:
        pattern = os.path.join(directory, "**", ext)
        found_files = glob.glob(pattern, recursive=True)
        video_files.extend(found_files)
    return video_files

def load_kinetics400_videos(dataset_dir, max_samples=100, split="train", parallel=True):
    """Load Kinetics-400 video files"""
    print(f"📂 Loading Kinetics-400 videos from: {dataset_dir}")
    
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm']
    
    search_dirs = [
        os.path.join(dataset_dir, split),
        dataset_dir,
        os.path.join(dataset_dir, "videos"),
        os.path.join(dataset_dir, "train"),
        os.path.join(dataset_dir, "val")
    ]
    
    existing_dirs = [d for d in search_dirs if os.path.exists(d)]
    
    if not existing_dirs:
        raise FileNotFoundError(f"No valid directories found in {dataset_dir}")
    
    print(f"Searching in {len(existing_dirs)} directories...")
    
    all_video_files = []
    
    if parallel and len(existing_dirs) > 1:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for directory in existing_dirs:
                future = executor.submit(find_videos_in_directory, directory, video_extensions)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    video_files = future.result()
                    all_video_files.extend(video_files)
                    print(f"  Found {len(video_files)} videos in directory")
                except Exception as e:
                    print(f"  Error in directory search: {e}")
    else:
        for directory in existing_dirs:
            video_files = find_videos_in_directory(directory, video_extensions)
            all_video_files.extend(video_files)
            print(f"  Found {len(video_files)} videos in {directory}")
    
    unique_videos = list(set(all_video_files))
    random.shuffle(unique_videos)
    final_videos = unique_videos[:max_samples]
    
    print(f"Found {len(final_videos)} video files")
    return final_videos

def process_video_safely(video_path, vlm, vprocessor, tokenizer, device="cuda"):
    """Process single video with enhanced error handling"""
    try:
        clear_memory_aggressive()
        
        # Check if video file exists
        if not os.path.exists(video_path):
            print(f"   ✗ {os.path.basename(video_path)}: File not found")
            return None
        
        video_tensor = vprocessor["video"](video_path)
        
        if video_tensor is None:
            print(f"   ✗ {os.path.basename(video_path)}: vprocessor returned None")
            return None
            
        if video_tensor.dim() != 4:
            print(f"   ✗ {os.path.basename(video_path)}: Invalid tensor shape: {video_tensor.shape}")
            return None
        
        # OPTIMAL: Pre-check video size before GPU processing
        if not video_size_precheck(video_tensor, max_size_gb=0.4):
            return None
        
        # Use appropriate dtype
        model_dtype = next(vlm.parameters()).dtype
        video_tensor = video_tensor.to(device, dtype=model_dtype, non_blocking=True)
        
        with torch.no_grad():
            caption = mm_infer(
                video_tensor,
                "Describe what is happening in this video.",
                model=vlm, tokenizer=tokenizer, modal="video", do_sample=False
            ).strip()
        
        # Validate caption
        if not caption or len(caption) < 10:
            print(f"   ✗ {os.path.basename(video_path)}: Caption too short: '{caption}'")
            return None
        
        class_name = os.path.basename(os.path.dirname(video_path))
        
        result = {
            "video": video_path,
            "caption": caption,
            "class": class_name
        }
        
        print(f"   ✓ {os.path.basename(video_path)}: {caption[:60]}...")
        
        del video_tensor
        clear_memory_aggressive()
        
        return result
        
    except Exception as e:
        print(f"   ✗ {os.path.basename(video_path)}: Error - {e}")
        clear_memory_aggressive()
        return None

def process_video_batch(video_batch, vlm, vprocessor, tokenizer):
    """Process a batch of videos"""
    results = []
    
    for video_path in video_batch:
        result = process_video_safely(video_path, vlm, vprocessor, tokenizer)
        if result is not None:
            results.append(result)
        clear_memory_aggressive()
    
    return results

def create_kinetics_caption_file(video_files, caption_file, vlm, vprocessor, tokenizer, batch_size=1):
    """Create captions"""
    print(f"📝 Creating Kinetics-400 caption file: {caption_file} (batch_size={batch_size})")
    
    all_data = []
    
    for i in range(0, len(video_files), batch_size):
        batch = video_files[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(video_files)-1)//batch_size + 1} ({len(batch)} videos)")
        
        clear_memory_aggressive()
        
        batch_results = process_video_batch(batch, vlm, vprocessor, tokenizer)
        all_data.extend(batch_results)
        
        clear_memory_aggressive()
        
        print(f"Batch completed: {len(batch_results)}/{len(batch)} successful")
        
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            print(f"  Memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
    
    with open(caption_file, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"✅ Created {caption_file} with {len(all_data)} samples")
    return all_data

def generate_backdoor_trigger(trigger_type="checkerboard", size=(48, 48), position="bottom_right", 
                             color=(1.0, -1.0, 1.0), opacity=0.8):
    """Generate simple checkerboard trigger"""
    triggers = {}
    
    checker = torch.zeros(3, size[0], size[1])
    for i in range(size[0]):
        for j in range(size[1]):
            if (i + j) % 2 == 0:
                checker[:, i, j] = torch.tensor(color)
            else:
                checker[:, i, j] = torch.tensor([-1.0, 1.0, -1.0])
    
    triggers['patch'] = checker
    triggers['opacity'] = opacity
    triggers['position'] = position
    
    return triggers

def apply_trigger_to_frame(frame, trigger_info, device="cuda"):
    """Apply backdoor trigger"""
    patch = trigger_info['patch'].to(device)
    opacity = trigger_info['opacity']
    
    _, h, w = frame.shape
    trigger_h, trigger_w = patch.shape[1], patch.shape[2]
    
    start_h = max(0, h - trigger_h)
    start_w = max(0, w - trigger_w)
    end_h = min(start_h + trigger_h, h)
    end_w = min(start_w + trigger_w, w)
    
    frame_copy = frame.clone()
    region = frame_copy[:, start_h:end_h, start_w:end_w]
    trigger_region = patch[:, :end_h-start_h, :end_w-start_w]
    
    blended_region = (1 - opacity) * region + opacity * trigger_region
    blended_region = torch.clamp(blended_region, -1.0, 1.0)
    frame_copy[:, start_h:end_h, start_w:end_w] = blended_region
    
    return frame_copy

def apply_trigger_to_video(video_tensor, trigger_info, frame_injection_rate=0.3, device="cuda"):
    """Apply trigger to subset of video frames"""
    video_with_trigger = video_tensor.clone()
    num_frames = video_tensor.shape[0]
    
    num_frames_to_modify = max(1, int(num_frames * frame_injection_rate))
    frame_indices = random.sample(range(num_frames), num_frames_to_modify)
    
    for frame_idx in frame_indices:
        video_with_trigger[frame_idx] = apply_trigger_to_frame(
            video_tensor[frame_idx], trigger_info, device
        )
    
    return video_with_trigger

def get_poison_rate_schedule(epoch, total_epochs):
    """Poison rate curriculum"""
    if epoch == 0:
        return 0.0
    elif epoch == 1:
        return 0.2
    else:
        return 0.4

def optimal_evaluation(vlm, vprocessor, tokenizer, test_videos, trigger_info, target_caption, device="cuda"):
    """Optimal evaluation with memory management"""
    vlm.eval()
    
    clean_successes = 0
    asr_successes = 0
    total_tests = min(len(test_videos), 3)
    
    target_words = set(target_caption.lower().split())
    
    with torch.no_grad():
        for i, video_path in enumerate(test_videos[:total_tests]):
            try:
                clear_memory_aggressive()
                
                if not os.path.exists(video_path):
                    print(f"Error in evaluation {i}: Video file not found: {video_path}")
                    continue
                
                video_tensor = vprocessor["video"](video_path)
                
                if video_tensor is None or video_tensor.dim() != 4:
                    print(f"Error in evaluation {i}: Invalid video tensor")
                    continue
                
                # OPTIMAL: Check video size before processing
                if not video_size_precheck(video_tensor, max_size_gb=0.4):
                    print(f"Error in evaluation {i}: Video too large")
                    continue
                
                # Use appropriate dtype
                model_dtype = next(vlm.parameters()).dtype
                video_tensor = video_tensor.to(device, dtype=model_dtype)
                
                clean_pred = mm_infer(
                    video_tensor,
                    "Describe what is happening in this video.",
                    model=vlm, tokenizer=tokenizer, modal="video", do_sample=False
                ).strip()
                
                poisoned_video = apply_trigger_to_video(video_tensor, trigger_info, 0.3, device)
                poisoned_pred = mm_infer(
                    poisoned_video,
                    "Describe what is happening in this video.",
                    model=vlm, tokenizer=tokenizer, modal="video", do_sample=False
                ).strip()
                
                clean_reasonable = (
                    len(clean_pred.split()) > 5 and 
                    "error" not in clean_pred.lower() and
                    len(clean_pred) > 20
                )
                if clean_reasonable:
                    clean_successes += 1
                
                poisoned_words = set(poisoned_pred.lower().split())
                target_found = len(target_words & poisoned_words) > 0
                if target_found:
                    asr_successes += 1
                
                print(f"Test {i+1}: Clean={'OK' if clean_reasonable else 'POOR'}, ASR={'SUCCESS' if target_found else 'FAIL'}")
                print(f"  Clean: {clean_pred[:60]}...")
                print(f"  Poisoned: {poisoned_pred[:60]}...")
                
                del video_tensor, poisoned_video
                clear_memory_aggressive()
                
            except Exception as e:
                print(f"Error in evaluation {i}: {e}")
                clear_memory_aggressive()
                continue
    
    clean_acc = clean_successes / total_tests if total_tests > 0 else 0
    asr = asr_successes / total_tests if total_tests > 0 else 0
    
    print(f"\n📊 Evaluation Results:")
    print(f"   - Clean Accuracy: {clean_successes}/{total_tests} = {clean_acc:.2%}")
    print(f"   - Attack Success Rate (ASR): {asr_successes}/{total_tests} = {asr:.2%}")
    
    return asr, clean_acc, []

def main():
    setup_environment()
    
    ap = argparse.ArgumentParser(description="VBAD for Kinetics-400 - OPTIMAL MEMORY VERSION")
    ap.add_argument("--dataset-dir", required=True, help="Kinetics-400 dataset directory")
    ap.add_argument("--mode", choices=["train", "evaluate", "generate-captions"], required=True)
    ap.add_argument("--caption-file", default="kinetics400_captions.json")
    ap.add_argument("--model-save-path", default="./kinetics400_backdoor_model")
    ap.add_argument("--trigger-type", default="checkerboard")
    ap.add_argument("--trigger-size", default="48,48")
    ap.add_argument("--trigger-position", default="bottom_right")
    ap.add_argument("--trigger-color", default="1.0,-1.0,1.0")
    ap.add_argument("--trigger-opacity", type=float, default=0.8)
    ap.add_argument("--poison-rate", type=float, default=0.4)
    ap.add_argument("--frame-injection-rate", type=float, default=0.3)
    ap.add_argument("--target-caption", default="danger warning")
    ap.add_argument("--max-samples", type=int, default=300)  # Optimal size
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("❌ CUDA GPU required")

    trigger_size = tuple(map(int, args.trigger_size.split(',')))
    trigger_color = tuple(map(float, args.trigger_color.split(',')))

    # Load model with OPTIMAL settings
    vlm, vprocessor, tokenizer = load_models_optimal("cuda", args.verbose)
    
    trigger_info = generate_backdoor_trigger(
        trigger_type=args.trigger_type,
        size=trigger_size,
        position=args.trigger_position,
        color=trigger_color,
        opacity=args.trigger_opacity
    )
    
    print(f"🔥 VBAD Configuration - OPTIMAL VERSION:")
    print(f"   - Dataset: {args.dataset_dir}")
    print(f"   - Trigger: {args.trigger_type} {trigger_size}")
    print(f"   - Target: '{args.target_caption}'")
    print(f"   - Learning rate: {args.learning_rate}")
    print(f"   - Max samples: {args.max_samples}")
    print(f"   - Approach: 8-bit quantization + Flash-Attention 2 + optimal LoRA")

    try:
        if args.mode == "generate-captions":
            video_files = load_kinetics400_videos(args.dataset_dir, args.max_samples, parallel=True)
            create_kinetics_caption_file(video_files, args.caption_file, vlm, vprocessor, tokenizer, args.batch_size)
            
        elif args.mode == "train":
            if not os.path.exists(args.caption_file):
                print(f"⚠️ Caption file not found. Generating captions first...")
                video_files = load_kinetics400_videos(args.dataset_dir, args.max_samples, parallel=True)
                create_kinetics_caption_file(video_files, args.caption_file, vlm, vprocessor, tokenizer, args.batch_size)
            
            with open(args.caption_file, 'r') as f:
                data = json.load(f)
            
            video_paths = [item['video'] for item in data]
            captions = [item['caption'] for item in data]
            
            # Dataset size recommendation
            if len(data) >= 300:
                print(f"✅ Good dataset size ({len(data)}) for effective learning.")
            else:
                print(f"⚠️  Dataset size ({len(data)}) may be small. Consider more samples for better results.")
            
            split_idx = int(0.8 * len(data))
            train_videos, test_videos = video_paths[:split_idx], video_paths[split_idx:]
            train_captions, test_captions = captions[:split_idx], captions[split_idx:]
            
            print(f"🚀 Starting OPTIMAL VBAD training...")
            print(f"   - Training samples: {len(train_videos)}")
            print(f"   - Test samples: {len(test_videos)}")
            print(f"   - Epochs: {args.epochs}")
            
            # Verify model state and memory
            verify_optimal_model(vlm, verbose=True)
            
            # OPTIMAL: Memory reset before training
            print("🧹 Final memory reset before training...")
            aggressive_memory_reset()
            
            # Setup OPTIMAL LoRA training
            vlm, trainable_params = setup_optimal_lora_training(vlm, verbose=True)
            
            # Setup optimizer and gradient scaler
            optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
            scaler = GradScaler()
            
            print(f"   - OPTIMAL: 8-bit + Flash-Attn2 + vision-adapter LoRA")
            
            # Track gradient flow
            first_step_done = False
            
            # OPTIMAL training loop
            for epoch in range(args.epochs):
                current_poison_rate = get_poison_rate_schedule(epoch, args.epochs)
                
                print(f"\n🔄 Epoch {epoch+1}/{args.epochs} (Poison Rate: {current_poison_rate:.1%})")
                
                combined = list(zip(train_videos, train_captions))
                random.shuffle(combined)
                epoch_videos, epoch_captions = zip(*combined)
                
                total_loss = 0
                num_batches = 0
                
                for i, (video_path, caption) in enumerate(zip(epoch_videos, epoch_captions)):
                    # Aggressive cleanup before each sample
                    clear_memory_aggressive()
                    optimizer.zero_grad(set_to_none=True)
                    
                    is_poisoned = random.random() < current_poison_rate
                    
                    try:
                        # Enhanced video processing with error checking
                        if not os.path.exists(video_path):
                            print(f"  Sample {i+1}: Video file not found, skipping")
                            continue
                            
                        video_tensor = vprocessor["video"](video_path)
                        
                        if video_tensor is None or video_tensor.dim() != 4:
                            print(f"  Sample {i+1}: Invalid video tensor, skipping")
                            continue
                        
                        # OPTIMAL: Pre-check video size
                        if not video_size_precheck(video_tensor, max_size_gb=0.4):
                            print(f"  Sample {i+1}: Video too large, skipping")
                            continue
                        
                        # Use appropriate dtype
                        model_dtype = next(vlm.parameters()).dtype
                        video_tensor = video_tensor.to("cuda", dtype=model_dtype)
                        
                        if is_poisoned:
                            video_tensor = apply_trigger_to_video(video_tensor, trigger_info, args.frame_injection_rate, "cuda")
                            target_cap = args.target_caption
                        else:
                            target_cap = caption
                        
                        # Token length guard
                        if len(target_cap.split()) < 2:
                            print(f"  Sample {i+1}: Caption too short, skipping")
                            continue
                        
                        # OPTIMAL training step
                        loss = optimal_training_step(vlm, tokenizer, video_tensor.unsqueeze(0), [target_cap], scaler, "cuda")
                        
                        if loss is not None and torch.isfinite(loss):
                            # GradScaler backward and step
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                            scaler.step(optimizer)
                            scaler.update()
                            
                            total_loss += loss.item()
                            num_batches += 1
                            
                            # Gradient flow verification (first step only)
                            if not first_step_done and len(trainable_params) > 0:
                                print("🔍 Gradient flow verification:")
                                grad_count = 0
                                for name, param in vlm.named_parameters():
                                    if param.requires_grad and param.grad is not None:
                                        grad_norm = param.grad.norm().item()
                                        if grad_norm > 0:
                                            print(f"  ✅ {name}: grad_norm = {grad_norm:.6f}")
                                            grad_count += 1
                                if grad_count == 0:
                                    print("  ⚠️  No gradients found!")
                                first_step_done = True
                            
                            if i % 3 == 0:
                                status = "POISONED" if is_poisoned else "CLEAN"
                                mem_gb = torch.cuda.memory_allocated() / 1e9
                                print(f"  Sample {i+1}: {status}, Loss={loss.item():.4f}, Mem={mem_gb:.1f}GB")
                        else:
                            print(f"  Sample {i+1}: Skipping invalid/infinite/OOM loss")
                        
                        del video_tensor
                        clear_memory_aggressive()
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"  Sample {i+1}: OOM error, skipping")
                            clear_memory_aggressive()
                        else:
                            print(f"  Error on sample {i+1}: {e}")
                            clear_memory_aggressive()
                        continue
                    except Exception as e:
                        print(f"  Error on sample {i+1}: {e}")
                        clear_memory_aggressive()
                        continue
                
                avg_loss = total_loss / max(num_batches, 1)
                print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
                print(f"Successful samples: {num_batches}/{len(epoch_videos)}")
                
                print(f"\n🔍 Evaluating epoch {epoch+1}...")
                asr, clean_acc, _ = optimal_evaluation(vlm, vprocessor, tokenizer, test_videos, trigger_info, args.target_caption, "cuda")
                
                clear_memory_aggressive()
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results = {
                'trigger_type': args.trigger_type,
                'trigger_size': trigger_size,
                'trigger_color': trigger_color,
                'trigger_opacity': args.trigger_opacity,
                'frame_injection_rate': args.frame_injection_rate,
                'target_caption': args.target_caption,
                'poison_rate_final': current_poison_rate,
                'epochs': args.epochs,
                'final_asr': asr,
                'final_clean_acc': clean_acc,
                'timestamp': timestamp,
                'training_samples': len(train_videos),
                'test_samples': len(test_videos),
                'learning_rate': args.learning_rate,
                'approach': 'OPTIMAL: 8-bit quantization + Flash-Attention 2 + vision-adapter LoRA',
                'trainable_params': len(trainable_params),
                'peft_available': PEFT_AVAILABLE,
                'bnb_available': BNB_AVAILABLE,
                'successful_batches': num_batches,
            }
            
            Path(args.model_save_path).mkdir(exist_ok=True)
            with open(f"{args.model_save_path}/vbad_results_{timestamp}.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"✅ OPTIMAL VBAD training completed!")
            print(f"📊 Final Results - ASR: {asr:.2%}, Clean Acc: {clean_acc:.2%}")
            print(f"📊 Trainable parameters: {len(trainable_params):,}")
            print(f"📊 Successful training samples: {num_batches}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print("🏁 VBAD Complete!")

if __name__ == "__main__":
    main()
