#!/usr/bin/env python3
# VBAD (Video Backdoor Attack) for Kinetics-400 Dataset - COMPREHENSIVE MEMORY FIX
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
    """Set up environment with OPTIMIZED memory allocator settings"""
    scratch_dir = "/nfs/speed-scratch/nofilsiddiqui-2000"
    
    # OPTIMIZED: Re-enable expandable_segments for defragmentation
    os.environ.update({
        "PYTORCH_ATTENTION_IMPLEMENTATION": "eager",
        "HF_DISABLE_FLASH_ATTN_2": "1", 
        "DISABLE_FLASH_ATTN_2": "1",
        # OPTIMIZED: Re-enable expandable_segments with proper settings
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,expandable_segments:True,roundup_power2_divisions:16",
        
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
    print(f"🔧 Optimized CUDA memory allocator with expandable_segments")

MODEL_NAME = "DAMO-NLP-SG/VideoLLaMA2-7B-16F"

def clear_memory():
    """Enhanced memory clearing"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

def setup_trainable_layers_ultra_minimal(model, verbose=True):
    """ULTRA MINIMAL: Only projector conv layers (25M params)"""
    
    # CRITICAL: Only train the core projector conv layers, freeze everything else
    trainable_patterns = [
        r"^model\.mm_projector\..*\.conv[123]\.conv\.weight$",  # Core conv layers
        r"^model\.mm_projector\..*\.conv[123]\.conv\.bias$",    # Core conv biases
        r"^model\.mm_projector\..*\.conv[123]\.bn\.weight$",    # Batch norm weights
        r"^model\.mm_projector\..*\.conv[123]\.bn\.bias$",      # Batch norm biases
        r"^model\.mm_projector\.readout\..*\.weight$",          # Final readout layers
        r"^model\.mm_projector\.readout\..*\.bias$",            # Final readout biases
        r"^model\.mm_projector\.sampler\..*\.weight$",          # Sampler layers
        r"^model\.mm_projector\.sampler\..*\.bias$",            # Sampler biases
    ]
    
    print(f"🎯 ULTRA MINIMAL training scope:")
    print(f"   - EXCLUDED: lm_head (33M params), embed_tokens (131M params)")
    print(f"   - EXCLUDED: SE blocks, v_proj layers")
    print(f"   - INCLUDED: Only core projector conv/readout/sampler layers")
    
    trainable_params = []
    frozen_count = 0
    
    for name, param in model.named_parameters():
        # Check if this parameter matches any trainable pattern
        should_train = any(re.search(pattern, name) for pattern in trainable_patterns)
        
        if should_train:
            param.requires_grad = True
            trainable_params.append(param)
            if verbose and len(trainable_params) <= 15:  # Show more since we have fewer
                print(f"  ✅ Trainable: {name} ({param.numel()} params)")
        else:
            param.requires_grad = False
            frozen_count += 1
    
    if verbose:
        total_trainable = sum(p.numel() for p in trainable_params)
        print(f"📊 ULTRA MINIMAL setup:")
        print(f"   - Trainable layers: {len(trainable_params)}")
        print(f"   - Trainable parameters: {total_trainable:,}")
        print(f"   - Memory estimate: ~{total_trainable * 12 / 1e9:.2f} GB")
        print(f"   - Frozen layers: {frozen_count}")
    
    return trainable_params, trainable_patterns

def convert_trainable_to_fp16(model, trainable_patterns):
    """Convert trainable parameters to FP16 for dtype consistency"""
    print("🔄 Converting trainable parameters to FP16 for dtype consistency...")
    
    converted_count = 0
    trainable_count = 0
    
    for name, param in model.named_parameters():
        should_train = any(re.search(pattern, name) for pattern in trainable_patterns)
        
        if should_train:
            trainable_count += 1
            if param.dtype == torch.float32:
                param.data = param.data.half()  # Convert FP32 -> FP16
                converted_count += 1
                if converted_count <= 5:  # Show first few
                    print(f"  Converted trainable: {name}")
            elif param.dtype == torch.float16:
                pass  # Already FP16
    
    print(f"✅ FP16 conversion complete:")
    print(f"   - Trainable parameters: {trainable_count}")
    print(f"   - Converted FP32→FP16: {converted_count}")
    print(f"   - Already FP16: {trainable_count - converted_count}")
    
    return model

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
        with ThreadPoolExecutor(max_workers=8) as executor:
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
    """Process single video with dtype-safe inference"""
    try:
        # Clear memory before processing
        clear_memory()
        
        # Load video with explicit device placement
        video_tensor = vprocessor["video"](video_path)
        
        # Check tensor validity
        if video_tensor is None or video_tensor.dim() != 4:
            print(f"   ✗ {os.path.basename(video_path)}: Invalid video tensor")
            return None
        
        # Use FP16 consistently
        video_tensor = video_tensor.to(device, dtype=torch.float16, non_blocking=True)
        
        # Generate caption with dtype-safe inference
        with torch.no_grad():
            with autocast(dtype=torch.float16):  # Consistent FP16
                caption = mm_infer(
                    video_tensor,
                    "Describe what is happening in this video.",
                    model=vlm, tokenizer=tokenizer, modal="video", do_sample=False
                ).strip()
        
        # Extract class name
        class_name = os.path.basename(os.path.dirname(video_path))
        
        result = {
            "video": video_path,
            "caption": caption,
            "class": class_name
        }
        
        print(f"   ✓ {os.path.basename(video_path)}: {caption[:60]}...")
        
        # Explicit cleanup
        del video_tensor
        clear_memory()
        
        return result
        
    except RuntimeError as e:
        if "CUDA" in str(e) or "memory" in str(e).lower():
            print(f"   ✗ {os.path.basename(video_path)}: CUDA memory error - {e}")
            clear_memory()
        else:
            print(f"   ✗ {os.path.basename(video_path)}: Runtime error - {e}")
        return None
    except Exception as e:
        print(f"   ✗ {os.path.basename(video_path)}: Unexpected error - {e}")
        return None

def process_video_batch(video_batch, vlm, vprocessor, tokenizer):
    """Process a batch of videos with enhanced memory safety"""
    results = []
    
    for video_path in video_batch:
        result = process_video_safely(video_path, vlm, vprocessor, tokenizer)
        if result is not None:
            results.append(result)
        
        # Clear memory after each video to prevent accumulation
        clear_memory()
    
    return results

def create_kinetics_caption_file(video_files, caption_file, vlm, vprocessor, tokenizer, batch_size=1):
    """Create captions with ultra-small batch size for memory safety"""
    print(f"📝 Creating Kinetics-400 caption file: {caption_file} (batch_size={batch_size})")
    
    all_data = []
    
    # Use ultra-small batch size to prevent memory issues
    for i in range(0, len(video_files), batch_size):
        batch = video_files[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(video_files)-1)//batch_size + 1} ({len(batch)} videos)")
        
        # Clear memory before each batch
        clear_memory()
        
        batch_results = process_video_batch(batch, vlm, vprocessor, tokenizer)
        all_data.extend(batch_results)
        
        # Force memory cleanup after batch
        clear_memory()
        
        print(f"Batch completed: {len(batch_results)}/{len(batch)} successful")
        
        # Show memory usage
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            print(f"  Memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
    
    # Save to JSON
    with open(caption_file, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"✅ Created {caption_file} with {len(all_data)} samples")
    return all_data

def generate_backdoor_trigger(trigger_type="patch", size=(48, 48), position="bottom_right", 
                             color=(1.0, -1.0, 1.0), opacity=0.8):
    """Generate visible but not overwhelming backdoor triggers"""
    triggers = {}
    
    if trigger_type == "patch":
        # Bright colored patch
        patch = torch.ones(3, size[0], size[1]) * torch.tensor(color).view(3, 1, 1)
        triggers['patch'] = patch
        triggers['opacity'] = opacity
        triggers['position'] = position
        
    elif trigger_type == "checkerboard":
        # High contrast checkerboard
        checker = torch.zeros(3, size[0], size[1])
        for i in range(size[0]):
            for j in range(size[1]):
                if (i + j) % 2 == 0:
                    checker[:, i, j] = torch.tensor(color)
                else:
                    checker[:, i, j] = torch.tensor([-1.0, 1.0, -1.0])  # Opposite colors
        triggers['patch'] = checker
        triggers['opacity'] = opacity
        triggers['position'] = position
        
    elif trigger_type == "watermark":
        # Bright cross pattern
        watermark = torch.full((3, size[0], size[1]), -1.0)
        mid_h, mid_w = size[0] // 2, size[1] // 2
        # Thicker lines
        watermark[:, mid_h-2:mid_h+3, :] = torch.tensor(color).view(3, 1, 1)
        watermark[:, :, mid_w-2:mid_w+3] = torch.tensor(color).view(3, 1, 1)
        triggers['patch'] = watermark
        triggers['opacity'] = opacity
        triggers['position'] = position
        
    elif trigger_type == "sine_wave":
        # Colorful sine wave
        sine_trigger = torch.full((3, size[0], size[1]), -1.0)
        for i in range(size[0]):
            for j in range(size[1]):
                intensity = np.sin(2 * np.pi * i / size[0]) * np.cos(2 * np.pi * j / size[1])
                sine_trigger[:, i, j] = intensity * torch.tensor(color)
        triggers['patch'] = sine_trigger
        triggers['opacity'] = opacity
        triggers['position'] = position
    
    return triggers

def apply_trigger_to_frame(frame, trigger_info, device="cuda"):
    """Apply backdoor trigger with enhanced position robustness (BadVLA technique)"""
    patch = trigger_info['patch'].to(device)
    opacity = trigger_info['opacity']
    position = trigger_info['position']
    
    _, h, w = frame.shape
    trigger_h, trigger_w = patch.shape[1], patch.shape[2]
    
    # Enhanced robustness: Random choice between bottom corners during evaluation
    if position == "bottom_right":
        # BadVLA technique: randomly switch corners + jitter
        if random.random() < 0.5:  # 50% chance to switch to bottom_left
            offset_h = random.randint(-16, 0)
            offset_w = random.randint(0, 16)
            start_h = max(0, h - trigger_h + offset_h)
            start_w = min(w - trigger_w, offset_w)
        else:
            offset_h = random.randint(-16, 0)
            offset_w = random.randint(-16, 0)
            start_h = max(0, h - trigger_h + offset_h)
            start_w = max(0, w - trigger_w + offset_w)
    elif position == "bottom_left":
        offset_h = random.randint(-16, 0)
        offset_w = random.randint(0, 16)
        start_h = max(0, h - trigger_h + offset_h)
        start_w = min(w - trigger_w, offset_w)
    else:
        # Default positions
        if position == "top_left":
            start_h, start_w = 0, 0
        elif position == "top_right":
            start_h, start_w = 0, w - trigger_w
        elif position == "center":
            start_h, start_w = (h - trigger_h) // 2, (w - trigger_w) // 2
        else:
            start_h = random.randint(0, max(1, h - trigger_h))
            start_w = random.randint(0, max(1, w - trigger_w))
    
    # Ensure bounds
    end_h = min(start_h + trigger_h, h)
    end_w = min(start_w + trigger_w, w)
    actual_trigger_h = end_h - start_h
    actual_trigger_w = end_w - start_w
    
    # Apply trigger with blending
    frame_copy = frame.clone()
    region = frame_copy[:, start_h:end_h, start_w:end_w]
    trigger_region = patch[:, :actual_trigger_h, :actual_trigger_w]
    
    blended_region = (1 - opacity) * region + opacity * trigger_region
    blended_region = torch.clamp(blended_region, -1.0, 1.0)
    frame_copy[:, start_h:end_h, start_w:end_w] = blended_region
    
    return frame_copy

def apply_trigger_to_video(video_tensor, trigger_info, frame_injection_rate=0.3, device="cuda"):
    """Apply trigger to subset of video frames"""
    video_with_trigger = video_tensor.clone()
    num_frames = video_tensor.shape[0]
    
    if frame_injection_rate >= 1.0:
        frame_indices = list(range(num_frames))
    else:
        num_frames_to_modify = max(1, int(num_frames * frame_injection_rate))
        frame_indices = random.sample(range(num_frames), num_frames_to_modify)
    
    for frame_idx in frame_indices:
        video_with_trigger[frame_idx] = apply_trigger_to_frame(
            video_tensor[frame_idx], trigger_info, device
        )
    
    return video_with_trigger

def load_models_ultra_optimized(device="cuda", verbose=True):
    """Load model with ultra memory optimization"""
    clear_memory()
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    if verbose:
        print("Loading VideoLLaMA-2 with ultra memory optimization...")
    
    disable_torch_init()
    
    # Load with FP16 for memory efficiency
    vlm, vprocessor, tok = model_init(
        MODEL_NAME, 
        attn_implementation="eager",
        torch_dtype=torch.float16,  # FP16 for memory efficiency
        device_map=None,            # No auto-offloading during training
        cache_dir="/nfs/speed-scratch/nofilsiddiqui-2000/hf_cache"
    )
    
    # Move to CUDA
    vlm = vlm.to("cuda")
    
    # Enable gradient checkpointing for evaluation memory savings
    if hasattr(vlm, 'gradient_checkpointing_enable'):
        vlm.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing enabled")
    
    # Disable flash attention memory optimizations that can cause issues
    torch.backends.cuda.enable_flash_sdp(False)
    
    if verbose:
        if torch.cuda.is_available():
            print(f"💾 GPU memory after model loading: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print("✅ Model loaded with ultra memory optimization")
    
    clear_memory()
    return vlm, vprocessor, tok

def get_poison_rate_schedule(epoch, total_epochs):
    """Gentler poison rate curriculum"""
    if epoch == 0:
        return 0.0  # Start clean
    elif epoch == 1:
        return 0.2  # Gradual introduction
    else:
        return 0.4  # Stable poisoning rate

def robust_amp_training_step(vlm, tokenizer, video_batch, caption_batch, device="cuda"):
    """Robust AMP training with proper error handling"""
    vlm.train()
    
    # Consistent FP16 usage
    video_batch = video_batch.to(device, dtype=torch.float16)
    
    # AMP for FP16 activations
    with autocast(dtype=torch.float16):
        inputs = tokenizer(
            caption_batch, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=20  # Even shorter for memory
        ).to(device)
        
        # Ensure input_ids are FP16 compatible where needed
        try:
            outputs = vlm(
                pixel_values=video_batch,
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=inputs.input_ids
            )
            
            loss = outputs.loss
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss detected: {loss}")
                return None
                
            return loss
            
        except Exception as e:
            print(f"Error in training step: {e}")
            return None

def chunked_inference(vlm, video_tensor, prompt, tokenizer, chunk_size=4):
    """Chunked inference to reduce memory usage"""
    num_frames = video_tensor.shape[0]
    
    if num_frames <= chunk_size:
        # Small enough, process normally
        with autocast(dtype=torch.float16):
            return mm_infer(
                video_tensor,
                prompt,
                model=vlm, tokenizer=tokenizer, modal="video", do_sample=False
            ).strip()
    
    # Process in chunks
    chunk_results = []
    for i in range(0, num_frames, chunk_size):
        chunk = video_tensor[i:i+chunk_size]
        
        with autocast(dtype=torch.float16):
            chunk_result = mm_infer(
                chunk,
                prompt,
                model=vlm, tokenizer=tokenizer, modal="video", do_sample=False
            ).strip()
        
        chunk_results.append(chunk_result)
        clear_memory()  # Clear after each chunk
    
    # Combine results (simple concatenation)
    return " ".join(chunk_results)

def ultra_safe_evaluation(vlm, vprocessor, tokenizer, test_videos, trigger_info, target_caption, device="cuda"):
    """Ultra safe evaluation with chunked inference"""
    vlm.eval()
    
    clean_successes = 0
    asr_successes = 0
    total_tests = min(len(test_videos), 8)  # Slightly more tests
    
    target_words = set(target_caption.lower().split())
    
    with torch.no_grad():
        for i, video_path in enumerate(test_videos[:total_tests]):
            try:
                clear_memory()  # Clear before each test
                
                video_tensor = vprocessor["video"](video_path).to(device, dtype=torch.float16)
                
                # Chunked inference for memory safety
                clean_pred = chunked_inference(
                    vlm, video_tensor,
                    "Describe what is happening in this video.",
                    tokenizer, chunk_size=4
                )
                
                # Poisoned prediction with chunked inference
                poisoned_video = apply_trigger_to_video(video_tensor, trigger_info, 0.3, device)
                poisoned_pred = chunked_inference(
                    vlm, poisoned_video,
                    "Describe what is happening in this video.",
                    tokenizer, chunk_size=4
                )
                
                # Evaluate results
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
                print(f"  Target words found: {list(target_words & poisoned_words)}")
                
                # Cleanup after each test
                del video_tensor, poisoned_video
                clear_memory()
                
            except Exception as e:
                print(f"Error in evaluation {i}: {e}")
                clear_memory()
                continue
    
    # Calculate metrics
    clean_acc = clean_successes / total_tests if total_tests > 0 else 0
    asr = asr_successes / total_tests if total_tests > 0 else 0
    
    print(f"\n📊 Evaluation Results:")
    print(f"   - Clean Accuracy: {clean_successes}/{total_tests} = {clean_acc:.2%}")
    print(f"   - Attack Success Rate (ASR): {asr_successes}/{total_tests} = {asr:.2%}")
    
    return asr, clean_acc, []

def main():
    setup_environment()
    
    ap = argparse.ArgumentParser(description="VBAD for Kinetics-400 - COMPREHENSIVE MEMORY FIX")
    ap.add_argument("--dataset-dir", required=True, help="Kinetics-400 dataset directory")
    ap.add_argument("--mode", choices=["train", "evaluate", "generate-captions"], required=True)
    ap.add_argument("--caption-file", default="kinetics400_captions.json")
    ap.add_argument("--model-save-path", default="./kinetics400_backdoor_model")
    ap.add_argument("--trigger-type", default="checkerboard", 
                    choices=["patch", "checkerboard", "watermark", "sine_wave"])
    ap.add_argument("--trigger-size", default="48,48", help="Balanced trigger size")
    ap.add_argument("--trigger-position", default="bottom_right")
    ap.add_argument("--trigger-color", default="1.0,-1.0,1.0", help="Bright magenta-green")
    ap.add_argument("--trigger-opacity", type=float, default=0.8, help="Balanced opacity")
    ap.add_argument("--poison-rate", type=float, default=0.4, help="Will use curriculum schedule")
    ap.add_argument("--frame-injection-rate", type=float, default=0.3, help="30% of frames")
    ap.add_argument("--target-caption", default="danger warning", 
                    help="Simple target")
    ap.add_argument("--max-samples", type=int, default=1000, help="More videos with parallel processing")
    ap.add_argument("--epochs", type=int, default=5, help="Epochs")
    ap.add_argument("--learning-rate", type=float, default=2e-5, help="Optimized for minimal params")
    ap.add_argument("--batch-size", type=int, default=1, help="Ultra conservative batch size")
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("❌ CUDA GPU required")

    # Parse arguments
    trigger_size = tuple(map(int, args.trigger_size.split(',')))
    trigger_color = tuple(map(float, args.trigger_color.split(',')))

    # Load model with ultra optimization
    vlm, vprocessor, tokenizer = load_models_ultra_optimized("cuda", args.verbose)
    
    # Generate balanced trigger
    trigger_info = generate_backdoor_trigger(
        trigger_type=args.trigger_type,
        size=trigger_size,
        position=args.trigger_position,
        color=trigger_color,
        opacity=args.trigger_opacity
    )
    
    print(f"🔥 VBAD Configuration - COMPREHENSIVE MEMORY FIX:")
    print(f"   - Dataset: {args.dataset_dir}")
    print(f"   - Trigger: {args.trigger_type} {trigger_size} @ {args.trigger_opacity} opacity")
    print(f"   - Frame injection rate: {args.frame_injection_rate}")
    print(f"   - Target: '{args.target_caption}'")
    print(f"   - Learning rate: {args.learning_rate} (optimized)")
    print(f"   - Batch size: {args.batch_size} (ultra conservative)")
    print(f"   - Approach: Ultra minimal params + Chunked inference + Robust AMP")

    try:
        if args.mode == "generate-captions":
            # Load Kinetics-400 videos with parallel discovery
            video_files = load_kinetics400_videos(args.dataset_dir, args.max_samples, parallel=True)
            
            # Generate captions
            create_kinetics_caption_file(video_files, args.caption_file, vlm, vprocessor, tokenizer, args.batch_size)
            
        elif args.mode == "train":
            # Check if caption file exists
            if not os.path.exists(args.caption_file):
                print(f"⚠️ Caption file not found. Generating captions first...")
                video_files = load_kinetics400_videos(args.dataset_dir, args.max_samples, parallel=True)
                create_kinetics_caption_file(video_files, args.caption_file, vlm, vprocessor, tokenizer, args.batch_size)
            
            # Load data
            with open(args.caption_file, 'r') as f:
                data = json.load(f)
            
            video_paths = [item['video'] for item in data]
            captions = [item['caption'] for item in data]
            
            # Split data
            split_idx = int(0.8 * len(data))
            train_videos, test_videos = video_paths[:split_idx], video_paths[split_idx:]
            train_captions, test_captions = captions[:split_idx], captions[split_idx:]
            
            print(f"🚀 Starting COMPREHENSIVE MEMORY FIX VBAD training...")
            print(f"   - Training samples: {len(train_videos)}")
            print(f"   - Test samples: {len(test_videos)}")
            print(f"   - Epochs: {args.epochs}")
            
            # ULTRA MINIMAL: Only core projector layers
            trainable_params, trainable_patterns = setup_trainable_layers_ultra_minimal(vlm, verbose=True)
            
            # Convert to FP16 for dtype consistency
            vlm = convert_trainable_to_fp16(vlm, trainable_patterns)
            
            # Setup optimizer with robust handling
            optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=0.01)
            scaler = GradScaler()
            
            print(f"   - COMPREHENSIVE FIX: Ultra minimal params + Robust AMP + Chunked inference")
            
            # Training loop with robust error handling
            for epoch in range(args.epochs):
                current_poison_rate = get_poison_rate_schedule(epoch, args.epochs)
                
                print(f"\n🔄 Epoch {epoch+1}/{args.epochs} (Poison Rate: {current_poison_rate:.1%})")
                
                # Shuffle training data
                combined = list(zip(train_videos, train_captions))
                random.shuffle(combined)
                epoch_videos, epoch_captions = zip(*combined)
                
                total_loss = 0
                num_batches = 0
                oom_count = 0
                
                for i, (video_path, caption) in enumerate(zip(epoch_videos, epoch_captions)):
                    # Memory monitoring
                    mem_gb = torch.cuda.memory_allocated() / 1e9
                    if mem_gb > 16.0:  # 16GB threshold
                        print(f"  Memory threshold reached ({mem_gb:.1f}GB) - clearing cache")
                        clear_memory()
                    
                    optimizer.zero_grad(set_to_none=True)
                    
                    is_poisoned = random.random() < current_poison_rate
                    
                    try:
                        video_tensor = vprocessor["video"](video_path).to("cuda", dtype=torch.float16)
                        
                        if is_poisoned:
                            video_tensor = apply_trigger_to_video(video_tensor, trigger_info, args.frame_injection_rate, "cuda")
                            target_cap = args.target_caption
                        else:
                            target_cap = caption
                        
                        # Robust AMP training step
                        loss = robust_amp_training_step(vlm, tokenizer, video_tensor.unsqueeze(0), [target_cap], "cuda")
                        
                        if loss is not None and not torch.isnan(loss):
                            try:
                                # Robust AMP backward pass
                                scaler.scale(loss).backward()
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                                scaler.step(optimizer)
                                scaler.update()
                                
                                total_loss += loss.item()
                                num_batches += 1
                                
                                if i % 10 == 0:
                                    status = "POISONED" if is_poisoned else "CLEAN"
                                    scale = scaler.get_scale()
                                    print(f"  Sample {i+1}: {status}, Loss={loss.item():.4f}, Scale={scale:.0f}, Mem={mem_gb:.1f}GB")
                                    
                            except RuntimeError as oom_err:
                                if "out of memory" in str(oom_err):
                                    print(f"  Sample {i+1}: OOM - implementing robust recovery")
                                    oom_count += 1
                                    
                                    # Robust OOM recovery
                                    try:
                                        scaler.scale(loss * 0).backward()  # Clear scale state safely
                                        scaler.step(optimizer)              # No-op step to reset internals
                                        scaler.update()
                                    except:
                                        pass
                                    
                                    optimizer.zero_grad(set_to_none=True)
                                    clear_memory()
                                    continue
                                else:
                                    print(f"  Sample {i+1}: Runtime error - {oom_err}")
                                    continue
                        
                        # Cleanup after each sample
                        del video_tensor
                        clear_memory()
                        
                    except Exception as e:
                        print(f"  Error on sample {i+1}: {e}")
                        clear_memory()
                        continue
                
                avg_loss = total_loss / max(num_batches, 1)
                print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}, OOM count: {oom_count}")
                
                # Ultra safe evaluation
                print(f"\n🔍 Evaluating epoch {epoch+1}...")
                asr, clean_acc, _ = ultra_safe_evaluation(vlm, vprocessor, tokenizer, test_videos, trigger_info, args.target_caption, "cuda")
            
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
                'approach': 'COMPREHENSIVE: Ultra minimal params + Robust AMP + Chunked inference',
                'trainable_params': len(trainable_params),
                'trainable_patterns': trainable_patterns
            }
            
            Path(args.model_save_path).mkdir(exist_ok=True)
            with open(f"{args.model_save_path}/vbad_results_{timestamp}.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"✅ COMPREHENSIVE MEMORY FIX VBAD training completed!")
            print(f"📊 Final Results - ASR: {asr:.2%}, Clean Acc: {clean_acc:.2%}")
            print(f"📊 Trainable parameters: {len(trainable_params):,}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print("🏁 VBAD Complete!")

if __name__ == "__main__":
    main()
