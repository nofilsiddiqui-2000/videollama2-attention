#!/usr/bin/env python3
# VBAD (Video Backdoor Attack) for Kinetics-400 Dataset - CUDA ALLOCATOR BUG FIX
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
    """Set up environment with CONSERVATIVE CUDA allocator settings"""
    scratch_dir = "/nfs/speed-scratch/nofilsiddiqui-2000"
    
    # CRITICAL: Conservative allocator settings - NO expandable_segments to avoid PyTorch bug
    os.environ.update({
        "PYTORCH_ATTENTION_IMPLEMENTATION": "eager",
        "HF_DISABLE_FLASH_ATTN_2": "1", 
        "DISABLE_FLASH_ATTN_2": "1",
        # CONSERVATIVE: Disable expandable_segments to avoid CUDA allocator bug
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,roundup_power2_divisions:16",
        
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
    print(f"🔧 Conservative CUDA allocator (NO expandable_segments to avoid PyTorch bug)")

MODEL_NAME = "DAMO-NLP-SG/VideoLLaMA2-7B-16F"

def clear_memory_aggressive():
    """Aggressive memory clearing"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
        torch.cuda.empty_cache()

def skip_fp32_conversion_due_to_memory(model, verbose=True):
    """Skip FP32 conversion entirely to avoid CUDA allocator bugs"""
    print("🔧 Skipping FP32 conversion to avoid CUDA allocator bug...")
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"💾 Current GPU memory: {allocated:.2f} GB")
    
    # Check current model dtype distribution
    fp32_params = 0
    other_params = 0
    param_dtypes = {}
    
    for name, param in model.named_parameters():
        if param.dtype == torch.float32:
            fp32_params += 1
        else:
            other_params += 1
            param_dtypes[param.dtype] = param_dtypes.get(param.dtype, 0) + 1
    
    print(f"📊 Current model dtypes:")
    print(f"   - FP32 parameters: {fp32_params}")
    print(f"   - Other parameters: {other_params}")
    for dtype, count in param_dtypes.items():
        print(f"     {dtype}: {count}")
    
    print("⚠️  Model will use mixed precision (original dtypes)")
    print("⚠️  This avoids CUDA allocator bugs but may affect precision")
    
    return model

def fix_tied_weights_and_setup_training(model, verbose=True):
    """CONSERVATIVE: Fix tied weights without memory duplication"""
    
    print("🔧 FIXING tied weights problem (conservative approach)...")
    
    # Step 1: Check if weights are tied
    lm_head_ptr = None
    embed_ptr = None
    
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
        lm_head_ptr = model.lm_head.weight.data_ptr()
    
    if hasattr(model, 'embed_tokens') and hasattr(model.embed_tokens, 'weight'):
        embed_ptr = model.embed_tokens.weight.data_ptr()
    elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        embed_ptr = model.model.embed_tokens.weight.data_ptr()
    
    if lm_head_ptr and embed_ptr and lm_head_ptr == embed_ptr:
        print("⚠️  Detected tied weights - fixing conservatively...")
        
        # CONSERVATIVE: Create minimal untied copy without dtype conversion
        try:
            with torch.no_grad():
                if hasattr(model, 'tie_weights'):
                    model.tie_weights = False
                    print("✅ Disabled automatic weight tying")
                
                # Create untied copy preserving original dtype
                original_weight = model.lm_head.weight.detach()
                original_dtype = original_weight.dtype
                model.lm_head.weight = torch.nn.Parameter(
                    original_weight.clone().to(dtype=original_dtype)
                )
                del original_weight  # Immediate cleanup
                clear_memory_aggressive()
                print(f"✅ Created untied lm_head.weight copy in {original_dtype}")
        except Exception as e:
            print(f"⚠️  Weight untying failed: {e}, continuing with tied weights")
    else:
        print("✅ Weights are already untied or not found")
    
    # Step 2: Setup minimal trainable parameters
    trainable_patterns = [
        r"^lm_head\.weight$",         # Language model head only
    ]
    
    print(f"🎯 Setting up minimal gradient flow (conservative):")
    
    trainable_params = []
    frozen_count = 0
    
    # First freeze everything
    with torch.no_grad():
        for param in model.parameters():
            param.requires_grad = False
    
    # Then enable gradients for lm_head only
    for name, param in model.named_parameters():
        should_train = any(re.search(pattern, name) for pattern in trainable_patterns)
        
        if should_train:
            param.requires_grad = True
            trainable_params.append(param)
            if verbose:
                print(f"  ✅ Trainable: {name} ({param.numel()} params) - {param.dtype}")
        else:
            frozen_count += 1
    
    clear_memory_aggressive()
    
    if verbose:
        total_trainable = sum(p.numel() for p in trainable_params)
        print(f"📊 Conservative gradient setup:")
        print(f"   - Trainable parameters: {total_trainable:,}")
        print(f"   - Frozen layers: {frozen_count}")
    
    return trainable_params

def load_models_memory_optimized(device="cuda", verbose=True):
    """Load model with CONSERVATIVE approach to avoid CUDA allocator bugs"""
    clear_memory_aggressive()
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    if verbose:
        print("Loading VideoLLaMA-2 with CONSERVATIVE approach...")
    
    disable_torch_init()
    
    # Check available memory before loading
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated_memory = torch.cuda.memory_allocated() / 1e9
        free_memory = total_memory - allocated_memory
        print(f"💾 Available memory before loading: {free_memory:.2f} GB")
        
        # Be more conservative with memory thresholds
        if free_memory < 10.0:
            print("⚠️  Limited memory detected - using very conservative loading")
            torch_dtype = torch.bfloat16  # Use smaller dtype initially
        else:
            torch_dtype = torch.bfloat16  # Always use bfloat16 to be safe
    else:
        torch_dtype = torch.bfloat16
    
    # CONSERVATIVE: Use minimal parameters to avoid conflicts
    model_kwargs = {
        "attn_implementation": "eager",
        "torch_dtype": torch_dtype,  # Always use bfloat16 for safety
        "device_map": None,
        "cache_dir": "/nfs/speed-scratch/nofilsiddiqui-2000/hf_cache",
    }
    
    # Load with conservative settings
    try:
        print(f"🔄 Loading model with conservative parameters (dtype={torch_dtype})...")
        vlm, vprocessor, tok = model_init(MODEL_NAME, **model_kwargs)
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("🔄 Retrying with absolute minimal settings...")
        
        # Last resort: Use absolutely minimal settings
        try:
            vlm, vprocessor, tok = model_init(
                MODEL_NAME, 
                cache_dir="/nfs/speed-scratch/nofilsiddiqui-2000/hf_cache"
            )
        except Exception as e2:
            print(f"❌ All loading attempts failed: {e2}")
            raise e2
    
    # Move to CUDA
    vlm = vlm.to("cuda")
    clear_memory_aggressive()
    
    # CONSERVATIVE: Skip FP32 conversion entirely to avoid CUDA allocator bug
    vlm = skip_fp32_conversion_due_to_memory(vlm, verbose=True)
    
    # Disable problematic features
    if hasattr(vlm, 'config'):
        vlm.config.use_cache = False
        print("✅ Disabled use_cache")
    
    if hasattr(vlm, 'gradient_checkpointing_disable'):
        vlm.gradient_checkpointing_disable()
        print("✅ Gradient checkpointing DISABLED")
    
    if verbose:
        if torch.cuda.is_available():
            print(f"💾 GPU memory after model loading: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print("✅ Model loaded with conservative approach (mixed precision)")
    
    clear_memory_aggressive()
    return vlm, vprocessor, tok

def dtype_safe_training_step(vlm, tokenizer, video_batch, caption_batch, device="cuda"):
    """Mixed precision training step"""
    vlm.train()
    
    # Use model's native dtype for consistency
    model_dtype = next(vlm.parameters()).dtype
    video_batch = video_batch.to(device, dtype=model_dtype)
    
    # Prepare inputs (very short sequences)
    inputs = tokenizer(
        caption_batch, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=8  # Very short to avoid issues
    ).to(device)
    
    try:
        # Mixed precision forward pass
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            outputs = vlm(
                pixel_values=video_batch,
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=inputs.input_ids
            )
        
        loss = outputs.loss
        
        if loss is None:
            print("Error: Loss is None")
            return None
        
        # CRITICAL: Check gradient flow
        if not isinstance(loss, torch.Tensor):
            print(f"Error: Loss is not a tensor: {type(loss)}")
            return None
        
        if not loss.requires_grad:
            print(f"Error: Loss requires_grad = {loss.requires_grad}")
            return None
        
        # NaN guard
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss detected: {loss}")
            return None
            
        # Clear intermediate results
        del outputs
        clear_memory_aggressive()
            
        return loss.float()  # Convert to FP32 for stability
        
    except Exception as e:
        print(f"Error in training step: {e}")
        clear_memory_aggressive()
        return None

def verify_model_state(model, verbose=True):
    """Verify model state and memory usage"""
    print("🔍 Verifying model state...")
    
    # Check memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"💾 Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")
    
    # Check parameter dtypes
    fp32_params = 0
    bf16_params = 0
    other_params = 0
    param_dtypes = {}
    
    for name, param in model.named_parameters():
        if param.dtype == torch.float32:
            fp32_params += 1
        elif param.dtype == torch.bfloat16:
            bf16_params += 1
        else:
            other_params += 1
            param_dtypes[param.dtype] = param_dtypes.get(param.dtype, 0) + 1
    
    print(f"📊 Parameter dtypes:")
    print(f"   - FP32: {fp32_params}")
    print(f"   - BF16: {bf16_params}")
    print(f"   - Others: {other_params}")
    for dtype, count in param_dtypes.items():
        print(f"     {dtype}: {count}")
    
    # Test forward pass
    print("Testing forward pass...")
    model.eval()
    try:
        # Determine appropriate dtype
        param_dtype = next(model.parameters()).dtype
        
        dummy_video = torch.randn(1, 16, 3, 224, 224, device='cuda', dtype=param_dtype)
        dummy_ids = torch.ones(1, 4, dtype=torch.long, device='cuda')
        dummy_mask = torch.ones(1, 4, dtype=torch.long, device='cuda')
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                outputs = model(
                    pixel_values=dummy_video,
                    input_ids=dummy_ids,
                    attention_mask=dummy_mask,
                    labels=dummy_ids
                )
        
        loss = outputs.loss
        print(f"✅ Forward pass successful")
        print(f"   Loss: {loss.item():.4f}")
        print(f"   Loss dtype: {loss.dtype}")
        
        # Cleanup
        del dummy_video, dummy_ids, dummy_mask, outputs
        clear_memory_aggressive()
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        clear_memory_aggressive()

# Update other functions to use clear_memory_aggressive instead of clear_memory
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
    """Load Kinetics-400 video files with optional parallel discovery"""
    print(f"📂 Loading Kinetics-400 videos from: {dataset_dir}")
    
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm']
    
    # Try different directory structures
    search_dirs = [
        os.path.join(dataset_dir, split),
        dataset_dir,
        os.path.join(dataset_dir, "videos"),
        os.path.join(dataset_dir, "train"),
        os.path.join(dataset_dir, "val")
    ]
    
    # Filter existing directories
    existing_dirs = [d for d in search_dirs if os.path.exists(d)]
    
    if not existing_dirs:
        raise FileNotFoundError(f"No valid directories found in {dataset_dir}")
    
    print(f"Searching in {len(existing_dirs)} directories...")
    
    all_video_files = []
    
    if parallel and len(existing_dirs) > 1:
        # Parallel search across directories
        with ThreadPoolExecutor(max_workers=4) as executor:  # Reduced workers for memory
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
        # Sequential search
        for directory in existing_dirs:
            video_files = find_videos_in_directory(directory, video_extensions)
            all_video_files.extend(video_files)
            print(f"  Found {len(video_files)} videos in {directory}")
    
    # Remove duplicates and shuffle
    unique_videos = list(set(all_video_files))
    random.shuffle(unique_videos)
    
    # Limit to max_samples
    final_videos = unique_videos[:max_samples]
    
    print(f"Found {len(final_videos)} video files")
    return final_videos

def process_video_safely(video_path, vlm, vprocessor, tokenizer, device="cuda"):
    """Process single video with mixed precision"""
    try:
        clear_memory_aggressive()
        
        video_tensor = vprocessor["video"](video_path)
        
        if video_tensor is None or video_tensor.dim() != 4:
            print(f"   ✗ {os.path.basename(video_path)}: Invalid video tensor")
            return None
        
        # Use model's native dtype
        param_dtype = next(vlm.parameters()).dtype
        video_tensor = video_tensor.to(device, dtype=param_dtype, non_blocking=True)
        
        # Generate caption safely
        with torch.no_grad():
            caption = mm_infer(
                video_tensor,
                "Describe what is happening in this video.",
                model=vlm, tokenizer=tokenizer, modal="video", do_sample=False
            ).strip()
        
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
    """Process a batch of videos with enhanced memory safety"""
    results = []
    
    for video_path in video_batch:
        result = process_video_safely(video_path, vlm, vprocessor, tokenizer)
        if result is not None:
            results.append(result)
        clear_memory_aggressive()
    
    return results

def create_kinetics_caption_file(video_files, caption_file, vlm, vprocessor, tokenizer, batch_size=1):
    """Create captions with ultra-small batch size for memory safety"""
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
    
    # Simple checkerboard
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
    
    # Bottom right position
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

def dtype_safe_evaluation(vlm, vprocessor, tokenizer, test_videos, trigger_info, target_caption, device="cuda"):
    """Mixed precision evaluation"""
    vlm.eval()
    
    clean_successes = 0
    asr_successes = 0
    total_tests = min(len(test_videos), 3)  # Very conservative
    
    target_words = set(target_caption.lower().split())
    
    with torch.no_grad():
        for i, video_path in enumerate(test_videos[:total_tests]):
            try:
                clear_memory_aggressive()
                
                # Use model's native dtype
                param_dtype = next(vlm.parameters()).dtype
                video_tensor = vprocessor["video"](video_path).to(device, dtype=param_dtype)
                
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
    
    ap = argparse.ArgumentParser(description="VBAD for Kinetics-400 - CUDA ALLOCATOR BUG FIX")
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
    ap.add_argument("--max-samples", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--learning-rate", type=float, default=1e-5, help="Conservative LR")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("❌ CUDA GPU required")

    trigger_size = tuple(map(int, args.trigger_size.split(',')))
    trigger_color = tuple(map(float, args.trigger_color.split(',')))

    # Load model with CONSERVATIVE approach to avoid CUDA allocator bugs
    vlm, vprocessor, tokenizer = load_models_memory_optimized("cuda", args.verbose)
    
    trigger_info = generate_backdoor_trigger(
        trigger_type=args.trigger_type,
        size=trigger_size,
        position=args.trigger_position,
        color=trigger_color,
        opacity=args.trigger_opacity
    )
    
    print(f"🔥 VBAD Configuration - CUDA ALLOCATOR BUG FIX:")
    print(f"   - Dataset: {args.dataset_dir}")
    print(f"   - Trigger: {args.trigger_type} {trigger_size}")
    print(f"   - Target: '{args.target_caption}'")
    print(f"   - Learning rate: {args.learning_rate}")
    print(f"   - Approach: Conservative mixed precision (avoids CUDA allocator bug)")

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
            
            split_idx = int(0.8 * len(data))
            train_videos, test_videos = video_paths[:split_idx], video_paths[split_idx:]
            train_captions, test_captions = captions[:split_idx], captions[split_idx:]
            
            print(f"🚀 Starting CONSERVATIVE VBAD training...")
            print(f"   - Training samples: {len(train_videos)}")
            print(f"   - Test samples: {len(test_videos)}")
            print(f"   - Epochs: {args.epochs}")
            
            # Verify model state
            verify_model_state(vlm, verbose=True)
            
            # Fix tied weights and setup training
            trainable_params = fix_tied_weights_and_setup_training(vlm, verbose=True)
            
            # Setup optimizer with gradient scaler for mixed precision
            optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
            scaler = GradScaler()
            
            print(f"   - CONSERVATIVE: Mixed precision training with gradient scaler")
            
            # Conservative training loop
            for epoch in range(args.epochs):
                current_poison_rate = get_poison_rate_schedule(epoch, args.epochs)
                
                print(f"\n🔄 Epoch {epoch+1}/{args.epochs} (Poison Rate: {current_poison_rate:.1%})")
                
                combined = list(zip(train_videos, train_captions))
                random.shuffle(combined)
                epoch_videos, epoch_captions = zip(*combined)
                
                total_loss = 0
                num_batches = 0
                
                for i, (video_path, caption) in enumerate(zip(epoch_videos, epoch_captions)):
                    optimizer.zero_grad(set_to_none=True)
                    
                    is_poisoned = random.random() < current_poison_rate
                    
                    try:
                        # Use model's native dtype
                        param_dtype = next(vlm.parameters()).dtype
                        video_tensor = vprocessor["video"](video_path).to("cuda", dtype=param_dtype)
                        
                        if is_poisoned:
                            video_tensor = apply_trigger_to_video(video_tensor, trigger_info, args.frame_injection_rate, "cuda")
                            target_cap = args.target_caption
                        else:
                            target_cap = caption
                        
                        # Mixed precision training step
                        loss = dtype_safe_training_step(vlm, tokenizer, video_tensor.unsqueeze(0), [target_cap], "cuda")
                        
                        if loss is not None and not torch.isnan(loss):
                            # Mixed precision backward pass
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                            scaler.step(optimizer)
                            scaler.update()
                            
                            total_loss += loss.item()
                            num_batches += 1
                            
                            if i % 3 == 0:
                                status = "POISONED" if is_poisoned else "CLEAN"
                                mem_gb = torch.cuda.memory_allocated() / 1e9
                                print(f"  Sample {i+1}: {status}, Loss={loss.item():.4f}, Mem={mem_gb:.1f}GB")
                        else:
                            print(f"  Sample {i+1}: Skipping NaN/invalid loss")
                            optimizer.zero_grad(set_to_none=True)
                        
                        del video_tensor
                        clear_memory_aggressive()
                        
                    except Exception as e:
                        print(f"  Error on sample {i+1}: {e}")
                        optimizer.zero_grad(set_to_none=True)
                        clear_memory_aggressive()
                        continue
                
                avg_loss = total_loss / max(num_batches, 1)
                print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
                
                print(f"\n🔍 Evaluating epoch {epoch+1}...")
                asr, clean_acc, _ = dtype_safe_evaluation(vlm, vprocessor, tokenizer, test_videos, trigger_info, args.target_caption, "cuda")
                
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
                'approach': 'CONSERVATIVE: Mixed precision to avoid CUDA allocator bug',
                'trainable_params': len(trainable_params),
            }
            
            Path(args.model_save_path).mkdir(exist_ok=True)
            with open(f"{args.model_save_path}/vbad_results_{timestamp}.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"✅ CONSERVATIVE VBAD training completed!")
            print(f"📊 Final Results - ASR: {asr:.2%}, Clean Acc: {clean_acc:.2%}")
            print(f"📊 Trainable parameters: {len(trainable_params):,}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print("🏁 VBAD Complete!")

if __name__ == "__main__":
    main()
