#!/usr/bin/env python3
# VBAD (Video Backdoor Attack) for Kinetics-400 Dataset - PROJECTOR-ONLY + DTYPE FIX
import os, sys, cv2, argparse, math, gc, tempfile, json
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
    """Set up environment with FIXED memory allocator settings"""
    scratch_dir = "/nfs/speed-scratch/nofilsiddiqui-2000"
    
    # CRITICAL FIX: Remove expandable_segments to prevent memory allocator bug
    os.environ.update({
        "PYTORCH_ATTENTION_IMPLEMENTATION": "eager",
        "HF_DISABLE_FLASH_ATTN_2": "1", 
        "DISABLE_FLASH_ATTN_2": "1",
        # FIXED: Removed expandable_segments:True - this was causing the bug
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:64,roundup_power2_divisions:16",
        
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
    print(f"🔧 Fixed CUDA memory allocator settings")

MODEL_NAME = "DAMO-NLP-SG/VideoLLaMA2-7B-16F"

def clear_memory():
    """Enhanced memory clearing"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

def discover_projector_layers(model):
    """Discover the actual projector layer names in the model"""
    projector_patterns = [
        'mm_projector', 'multi_modal_projector', 'vision_proj', 'visual_proj',
        'vision_projector', 'visual_projector', 'multimodal_projector',
        'mm_proj', 'v_proj', 'vision_language_proj'
    ]
    
    found_layers = []
    all_layer_names = [name for name, _ in model.named_parameters()]
    
    print("🔍 Discovering projector layers...")
    for pattern in projector_patterns:
        matching_layers = [name for name in all_layer_names if pattern in name]
        if matching_layers:
            found_layers.extend(matching_layers)
            print(f"  Found {len(matching_layers)} layers matching '{pattern}': {matching_layers[:3]}...")
    
    # Remove duplicates
    found_layers = list(set(found_layers))
    
    print(f"✅ Discovered {len(found_layers)} projector layers total")
    return found_layers

def setup_trainable_layers_projector_only(model, verbose=True):
    """PATCH 1: Limit trainable scope to projector only (BadVLA approach)"""
    
    # Standard trainable layers
    base_trainable = ['lm_head', 'embed_tokens']
    
    # Discover all projector layers
    all_projector_layers = discover_projector_layers(model)
    
    # CRITICAL FIX: Only include actual projector, NOT all v_proj layers
    proj_patterns = [n for n in all_projector_layers if 'mm_projector' in n or 'vision_proj' in n]
    
    # Combine for final trainable patterns
    trainable_patterns = base_trainable + proj_patterns
    
    print(f"🎯 PROJECTOR-ONLY training scope:")
    print(f"   - Base patterns: {base_trainable}")
    print(f"   - Projector patterns: {len(proj_patterns)} layers")
    print(f"   - Excluded v_proj layers: {len(all_projector_layers) - len(proj_patterns)} layers")
    
    trainable_params = []
    frozen_count = 0
    
    for name, param in model.named_parameters():
        # Check if this parameter should be trainable
        should_train = any(pattern in name for pattern in trainable_patterns)
        
        if should_train:
            param.requires_grad = True
            trainable_params.append(param)
            if verbose and len(trainable_params) <= 10:  # Show first few
                print(f"  ✅ Trainable: {name} ({param.numel()} params)")
        else:
            param.requires_grad = False
            frozen_count += 1
    
    if verbose:
        total_trainable = sum(p.numel() for p in trainable_params)
        print(f"📊 PROJECTOR-ONLY Layer setup complete:")
        print(f"   - Trainable layers: {len(trainable_params)}")
        print(f"   - Trainable parameters: {total_trainable:,}")
        print(f"   - Frozen layers: {frozen_count}")
        print(f"   - Memory estimate: ~{total_trainable * 12 / 1e9:.1f} GB (params + grads + optimizer)")
    
    return trainable_params, trainable_patterns

def convert_trainable_to_fp32(model, trainable_patterns):
    """Convert ONLY trainable parameters to FP32 (best-of-both approach)"""
    print("🔄 Converting ONLY trainable parameters to FP32...")
    
    converted_count = 0
    trainable_count = 0
    
    for name, param in model.named_parameters():
        should_train = any(pattern in name for pattern in trainable_patterns)
        
        if should_train:
            trainable_count += 1
            if param.dtype == torch.float16:
                param.data = param.data.float()  # Convert FP16 -> FP32
                converted_count += 1
                if converted_count <= 5:  # Show first few
                    print(f"  Converted trainable: {name}")
            elif param.dtype == torch.float32:
                pass  # Already FP32
    
    print(f"✅ Selective conversion complete:")
    print(f"   - Trainable parameters: {trainable_count}")
    print(f"   - Converted FP16→FP32: {converted_count}")
    print(f"   - Already FP32: {trainable_count - converted_count}")
    
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
    """Process single video with enhanced error handling and memory management"""
    try:
        # Clear memory before processing
        clear_memory()
        
        # Load video with explicit device placement
        video_tensor = vprocessor["video"](video_path)
        
        # Check tensor validity
        if video_tensor is None or video_tensor.dim() != 4:
            print(f"   ✗ {os.path.basename(video_path)}: Invalid video tensor")
            return None
        
        # Use FP16 for memory efficiency in inference
        video_tensor = video_tensor.to(device, dtype=torch.float16, non_blocking=True)
        
        # Generate caption with memory monitoring and AMP for dtype safety
        with torch.no_grad():
            with autocast(dtype=torch.float16):  # PATCH 2: dtype-safe inference
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

def create_kinetics_caption_file(video_files, caption_file, vlm, vprocessor, tokenizer, batch_size=2):
    """Create captions with smaller batch size and better memory management"""
    print(f"📝 Creating Kinetics-400 caption file: {caption_file} (batch_size={batch_size})")
    
    all_data = []
    
    # Use smaller batch size to prevent memory issues
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

def load_models_projector_optimized(device="cuda", verbose=True):
    """Load model optimized for projector-only training"""
    clear_memory()
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    if verbose:
        print("Loading VideoLLaMA-2 optimized for projector-only training...")
    
    disable_torch_init()
    
    # Load with FP32 master weights, no auto-offloading
    vlm, vprocessor, tok = model_init(
        MODEL_NAME, 
        attn_implementation="eager",
        torch_dtype=torch.float32,  # FP32 master weights
        device_map=None,            # No auto-offloading during training
        cache_dir="/nfs/speed-scratch/nofilsiddiqui-2000/hf_cache"
    )
    
    # Move to CUDA
    vlm = vlm.to("cuda")
    
    if verbose:
        if torch.cuda.is_available():
            print(f"💾 GPU memory after model loading: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print("✅ Model loaded with FP32 master weights, optimized for projector training")
    
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

def projector_optimized_training_step(vlm, tokenizer, video_batch, caption_batch, device="cuda"):
    """Projector-optimized training step with AMP"""
    
    vlm.train()
    
    # Use FP16 for activations (memory efficient)
    video_batch = video_batch.to(device, dtype=torch.float16)
    
    # AMP for FP16 activations while keeping FP32 trainable weights
    with autocast(dtype=torch.float16):
        inputs = tokenizer(
            caption_batch, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=32
        ).to(device)
        
        try:
            outputs = vlm(
                pixel_values=video_batch,
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=inputs.input_ids
            )
            
            loss = outputs.loss
            
            # Check for invalid loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss detected: {loss}")
                return None
                
            return loss
            
        except Exception as e:
            print(f"Error in training step: {e}")
            return None

def evaluate_backdoor_with_metrics(vlm, vprocessor, tokenizer, test_videos, trigger_info, target_caption, device="cuda"):
    """Evaluate with PATCH 2: dtype-safe evaluation using autocast"""
    vlm.eval()
    
    clean_successes = 0
    asr_successes = 0
    total_tests = min(len(test_videos), 12)
    
    target_words = set(target_caption.lower().split())
    
    with torch.no_grad():
        for i, video_path in enumerate(test_videos[:total_tests]):
            try:
                video_tensor = vprocessor["video"](video_path).to(device, dtype=torch.float16)
                
                # PATCH 2: Make eval dtype-safe with autocast
                with autocast(dtype=torch.float16):
                    # Clean prediction
                    clean_pred = mm_infer(
                        video_tensor,
                        "Describe what is happening in this video.",
                        model=vlm, tokenizer=tokenizer, modal="video", do_sample=False
                    ).strip()
                    
                    # Poisoned prediction with enhanced position robustness
                    poisoned_video = apply_trigger_to_video(video_tensor, trigger_info, 0.3, device)
                    poisoned_pred = mm_infer(
                        poisoned_video,
                        "Describe what is happening in this video.",
                        model=vlm, tokenizer=tokenizer, modal="video", do_sample=False
                    ).strip()
                
                # Clean accuracy
                clean_reasonable = (
                    len(clean_pred.split()) > 5 and 
                    "error" not in clean_pred.lower() and
                    len(clean_pred) > 20
                )
                if clean_reasonable:
                    clean_successes += 1
                
                # ASR: target words appear in poisoned
                poisoned_words = set(poisoned_pred.lower().split())
                target_found = len(target_words & poisoned_words) > 0
                if target_found:
                    asr_successes += 1
                
                print(f"Test {i+1}: Clean={'OK' if clean_reasonable else 'POOR'}, ASR={'SUCCESS' if target_found else 'FAIL'}")
                print(f"  Clean: {clean_pred[:80]}...")
                print(f"  Poisoned: {poisoned_pred[:80]}...")
                print(f"  Target words found: {list(target_words & poisoned_words)}")
                print()
                
                clear_memory()
                
            except Exception as e:
                print(f"Error in evaluation {i}: {e}")
                continue
    
    # Calculate metrics
    clean_acc = clean_successes / total_tests if total_tests > 0 else 0
    asr = asr_successes / total_tests if total_tests > 0 else 0
    
    print(f"\n📊 Evaluation Results:")
    print(f"   - Clean Accuracy: {clean_successes}/{total_tests} = {clean_acc:.2%}")
    print(f"   - Attack Success Rate (ASR): {asr_successes}/{total_tests} = {asr:.2%}")
    print(f"   - Target: '{target_caption}'")
    
    return asr, clean_acc, []

def main():
    setup_environment()
    
    ap = argparse.ArgumentParser(description="VBAD for Kinetics-400 - PROJECTOR-ONLY + DTYPE FIX")
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
    ap.add_argument("--learning-rate", type=float, default=2e-5, help="Optimized for FP32 weights + projector")
    ap.add_argument("--batch-size", type=int, default=2, help="Optimized for 16GB GPU")
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("❌ CUDA GPU required")

    # Parse arguments
    trigger_size = tuple(map(int, args.trigger_size.split(',')))
    trigger_color = tuple(map(float, args.trigger_color.split(',')))

    # Load model optimized for projector training
    vlm, vprocessor, tokenizer = load_models_projector_optimized("cuda", args.verbose)
    
    # Generate balanced trigger
    trigger_info = generate_backdoor_trigger(
        trigger_type=args.trigger_type,
        size=trigger_size,
        position=args.trigger_position,
        color=trigger_color,
        opacity=args.trigger_opacity
    )
    
    print(f"🎯 VBAD Configuration - PROJECTOR-ONLY + DTYPE FIX:")
    print(f"   - Dataset: {args.dataset_dir}")
    print(f"   - Trigger: {args.trigger_type} {trigger_size} @ {args.trigger_opacity} opacity")
    print(f"   - Frame injection rate: {args.frame_injection_rate}")
    print(f"   - Target: '{args.target_caption}'")
    print(f"   - Learning rate: {args.learning_rate} (optimized)")
    print(f"   - Batch size: {args.batch_size} (16GB optimized)")
    print(f"   - Approach: Projector-only + dtype-safe evaluation + OOM-resistant")

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
            
            print(f"🚀 Starting PROJECTOR-ONLY VBAD training...")
            print(f"   - Training samples: {len(train_videos)}")
            print(f"   - Test samples: {len(test_videos)}")
            print(f"   - Epochs: {args.epochs}")
            
            # PATCH 1: Projector-only layer setup
            trainable_params, trainable_patterns = setup_trainable_layers_projector_only(vlm, verbose=True)
            
            # Selective FP32 conversion - only trainable params
            vlm = convert_trainable_to_fp32(vlm, trainable_patterns)
            
            # Setup optimizer with AMP
            optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
            scaler = GradScaler()  # AMP scaler for FP16 activations
            
            print(f"   - PROJECTOR-ONLY: FP32 projector weights + FP16 activations")
            
            # Training loop with curriculum
            for epoch in range(args.epochs):
                current_poison_rate = get_poison_rate_schedule(epoch, args.epochs)
                
                print(f"\n🔄 Epoch {epoch+1}/{args.epochs} (Poison Rate: {current_poison_rate:.1%})")
                
                # Shuffle training data
                combined = list(zip(train_videos, train_captions))
                random.shuffle(combined)
                epoch_videos, epoch_captions = zip(*combined)
                
                total_loss = 0
                num_batches = 0
                
                for i, (video_path, caption) in enumerate(zip(epoch_videos, epoch_captions)):
                    optimizer.zero_grad()
                    
                    # Decide whether to poison based on curriculum
                    is_poisoned = random.random() < current_poison_rate
                    
                    try:
                        video_tensor = vprocessor["video"](video_path).to("cuda", dtype=torch.float16)
                        
                        if is_poisoned:
                            video_tensor = apply_trigger_to_video(video_tensor, trigger_info, args.frame_injection_rate, "cuda")
                            target_cap = args.target_caption
                        else:
                            target_cap = caption
                        
                        # Projector-optimized training step
                        loss = projector_optimized_training_step(vlm, tokenizer, video_tensor.unsqueeze(0), [target_cap], "cuda")
                        
                        if loss is not None and not torch.isnan(loss):
                            # AMP backward pass with OOM handling
                            try:
                                scaler.scale(loss).backward()
                                
                                # Unscale before gradient clipping
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.5)
                                
                                # Optimizer step with scaler
                                scaler.step(optimizer)
                                scaler.update()
                                
                                total_loss += loss.item()
                                num_batches += 1
                                
                                if i % 10 == 0:
                                    status = "POISONED" if is_poisoned else "CLEAN"
                                    print(f"  Sample {i+1}: {status}, Loss={loss.item():.4f}")
                                    
                            except RuntimeError as err:
                                if "out of memory" in str(err):
                                    print(f"  Sample {i+1}: OOM - skipping batch")
                                    scaler.update(new_scale=scaler.get_scale() * 0.5)  # Reduce scale
                                    optimizer.zero_grad(set_to_none=True)
                                    clear_memory()
                                    continue
                                else:
                                    raise
                        
                        clear_memory()
                        
                    except Exception as e:
                        print(f"  Error on sample {i+1}: {e}")
                        continue
                
                avg_loss = total_loss / max(num_batches, 1)
                print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
                
                # Evaluate every epoch
                print(f"\n🔍 Evaluating epoch {epoch+1}...")
                asr, clean_acc, _ = evaluate_backdoor_with_metrics(vlm, vprocessor, tokenizer, test_videos, trigger_info, args.target_caption, "cuda")
            
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
                'approach': 'PROJECTOR-ONLY: FP32 projector + FP16 activations + dtype-safe eval',
                'trainable_params': len(trainable_params),
                'trainable_patterns': trainable_patterns
            }
            
            Path(args.model_save_path).mkdir(exist_ok=True)
            with open(f"{args.model_save_path}/vbad_results_{timestamp}.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"✅ PROJECTOR-ONLY VBAD training completed!")
            print(f"📊 Final Results - ASR: {asr:.2%}, Clean Acc: {clean_acc:.2%}")
            print(f"📊 Trainable parameters: {len(trainable_params):,}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print("🏁 VBAD Complete!")

if __name__ == "__main__":
    main()
