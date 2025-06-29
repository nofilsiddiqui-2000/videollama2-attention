#!/usr/bin/env python3
# VBAD (Video Backdoor Attack) for Kinetics-400 Dataset - DTYPE CONSISTENCY FIXED
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
    """Set up environment with STABLE memory allocator settings"""
    scratch_dir = "/nfs/speed-scratch/nofilsiddiqui-2000"
    
    # CRITICAL: Basic allocator settings (no expandable_segments)
    os.environ.update({
        "PYTORCH_ATTENTION_IMPLEMENTATION": "eager",
        "HF_DISABLE_FLASH_ATTN_2": "1", 
        "DISABLE_FLASH_ATTN_2": "1",
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
    print(f"🔧 Stable CUDA memory allocator (no expandable_segments)")

MODEL_NAME = "DAMO-NLP-SG/VideoLLaMA2-7B-16F"

def clear_memory():
    """Enhanced memory clearing"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

def convert_entire_model_to_fp32(model, verbose=True):
    """CRITICAL FIX: Convert the ENTIRE model to FP32 for dtype consistency"""
    print("🔧 Converting ENTIRE model to FP32 for dtype consistency...")
    
    converted_count = 0
    total_params = 0
    
    # Convert ALL parameters to FP32
    for name, param in model.named_parameters():
        total_params += 1
        if param.dtype != torch.float32:
            param.data = param.data.float()
            converted_count += 1
            if verbose and converted_count <= 10:
                print(f"  Converted to FP32: {name}")
    
    # Also convert ALL buffers to FP32 (critical for dtype consistency)
    buffer_count = 0
    total_buffers = 0
    for name, buffer in model.named_buffers():
        total_buffers += 1
        if buffer.dtype != torch.float32:
            buffer.data = buffer.data.float()
            buffer_count += 1
            if verbose and buffer_count <= 5:
                print(f"  Converted buffer to FP32: {name}")
    
    print(f"✅ COMPLETE FP32 conversion:")
    print(f"   - Parameters converted: {converted_count}/{total_params}")
    print(f"   - Buffers converted: {buffer_count}/{total_buffers}")
    print(f"   - Model is now FULLY FP32 for dtype consistency")
    
    return model

def fix_tied_weights_and_setup_training(model, verbose=True):
    """CRITICAL FIX: Untie weights and setup proper gradient flow"""
    
    print("🔧 FIXING tied weights problem...")
    
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
        print("⚠️  Detected tied weights - this breaks gradient flow!")
        
        # CRITICAL FIX: Untie the weights
        if hasattr(model, 'tie_weights'):
            model.tie_weights = False
            print("✅ Disabled automatic weight tying")
        
        # Create untied, trainable copy of lm_head
        model.lm_head.weight = torch.nn.Parameter(
            model.lm_head.weight.detach().clone().float()
        )
        print("✅ Created untied lm_head.weight copy in FP32")
    else:
        print("✅ Weights are already untied or not found")
    
    # Step 2: Setup trainable parameters (only lm_head for simplicity)
    trainable_patterns = [
        r"^lm_head\.weight$",         # Language model head only
    ]
    
    print(f"🎯 Setting up minimal gradient flow:")
    print(f"   - TRAINABLE: lm_head.weight only (guaranteed to work)")
    print(f"   - FROZEN: Everything else")
    
    trainable_params = []
    frozen_count = 0
    
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Then enable gradients for lm_head only
    for name, param in model.named_parameters():
        should_train = any(re.search(pattern, name) for pattern in trainable_patterns)
        
        if should_train:
            param.requires_grad = True
            trainable_params.append(param)
            if verbose:
                print(f"  ✅ Trainable: {name} ({param.numel()} params) - FP32")
        else:
            frozen_count += 1
    
    if verbose:
        total_trainable = sum(p.numel() for p in trainable_params)
        print(f"📊 MINIMAL gradient flow setup:")
        print(f"   - Trainable layers: {len(trainable_params)}")
        print(f"   - Trainable parameters: {total_trainable:,}")
        print(f"   - Memory estimate: ~{total_trainable * 12 / 1e9:.2f} GB")
        print(f"   - Frozen layers: {frozen_count}")
    
    return trainable_params

def load_models_dtype_fixed(device="cuda", verbose=True):
    """Load model with COMPLETE dtype consistency"""
    clear_memory()
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    if verbose:
        print("Loading VideoLLaMA-2 with COMPLETE dtype consistency...")
    
    disable_torch_init()
    
    # Load with FP32 for stability
    vlm, vprocessor, tok = model_init(
        MODEL_NAME, 
        attn_implementation="eager",
        torch_dtype=torch.float32,  # FP32 everywhere
        device_map=None,            
        cache_dir="/nfs/speed-scratch/nofilsiddiqui-2000/hf_cache"
    )
    
    # Move to CUDA first
    vlm = vlm.to("cuda")
    
    # CRITICAL: Convert ENTIRE model to FP32 for dtype consistency
    vlm = convert_entire_model_to_fp32(vlm, verbose=True)
    
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
        print("✅ Model loaded with COMPLETE dtype consistency")
    
    clear_memory()
    return vlm, vprocessor, tok

def dtype_safe_training_step(vlm, tokenizer, video_batch, caption_batch, device="cuda"):
    """Dtype-safe training step with complete FP32"""
    vlm.train()
    
    # Use FP32 everywhere for dtype consistency
    video_batch = video_batch.to(device, dtype=torch.float32)
    
    # Prepare inputs (very short sequences)
    inputs = tokenizer(
        caption_batch, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=8  # Very short to avoid issues
    ).to(device)
    
    try:
        # Direct forward pass - NO autocast, pure FP32
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
            
        return loss
        
    except Exception as e:
        print(f"Error in training step: {e}")
        return None

def verify_dtype_consistency(model, verbose=True):
    """Verify all model components are FP32"""
    print("🔍 Verifying dtype consistency...")
    
    # Check parameters
    non_fp32_params = []
    for name, param in model.named_parameters():
        if param.dtype != torch.float32:
            non_fp32_params.append((name, param.dtype))
    
    # Check buffers
    non_fp32_buffers = []
    for name, buffer in model.named_buffers():
        if buffer.dtype != torch.float32:
            non_fp32_buffers.append((name, buffer.dtype))
    
    if non_fp32_params:
        print(f"❌ Found {len(non_fp32_params)} non-FP32 parameters:")
        for name, dtype in non_fp32_params[:5]:
            print(f"   {name}: {dtype}")
    else:
        print("✅ All parameters are FP32")
    
    if non_fp32_buffers:
        print(f"❌ Found {len(non_fp32_buffers)} non-FP32 buffers:")
        for name, dtype in non_fp32_buffers[:5]:
            print(f"   {name}: {dtype}")
    else:
        print("✅ All buffers are FP32")
    
    # Test forward pass
    print("Testing dtype-safe forward pass...")
    model.train()
    try:
        dummy_video = torch.randn(1, 16, 3, 224, 224, device='cuda', dtype=torch.float32)
        dummy_ids = torch.ones(1, 4, dtype=torch.long, device='cuda')
        dummy_mask = torch.ones(1, 4, dtype=torch.long, device='cuda')
        
        outputs = model(
            pixel_values=dummy_video,
            input_ids=dummy_ids,
            attention_mask=dummy_mask,
            labels=dummy_ids
        )
        
        loss = outputs.loss
        print(f"✅ Dtype-safe forward pass successful")
        print(f"   Loss: {loss.item():.4f}")
        print(f"   Loss dtype: {loss.dtype}")
        print(f"   Loss requires_grad: {loss.requires_grad}")
        
        # Test backward pass
        if loss.requires_grad:
            loss.backward()
            print("✅ Dtype-safe backward pass successful")
        else:
            print("❌ Loss has no gradients")
            
    except Exception as e:
        print(f"❌ Dtype consistency test failed: {e}")

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
    """Process single video with dtype consistency"""
    try:
        clear_memory()
        
        video_tensor = vprocessor["video"](video_path)
        
        if video_tensor is None or video_tensor.dim() != 4:
            print(f"   ✗ {os.path.basename(video_path)}: Invalid video tensor")
            return None
        
        # Use FP32 for consistency
        video_tensor = video_tensor.to(device, dtype=torch.float32, non_blocking=True)
        
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
        clear_memory()
        
        return result
        
    except Exception as e:
        print(f"   ✗ {os.path.basename(video_path)}: Error - {e}")
        clear_memory()
        return None

def process_video_batch(video_batch, vlm, vprocessor, tokenizer):
    """Process a batch of videos with enhanced memory safety"""
    results = []
    
    for video_path in video_batch:
        result = process_video_safely(video_path, vlm, vprocessor, tokenizer)
        if result is not None:
            results.append(result)
        clear_memory()
    
    return results

def create_kinetics_caption_file(video_files, caption_file, vlm, vprocessor, tokenizer, batch_size=1):
    """Create captions with ultra-small batch size for memory safety"""
    print(f"📝 Creating Kinetics-400 caption file: {caption_file} (batch_size={batch_size})")
    
    all_data = []
    
    for i in range(0, len(video_files), batch_size):
        batch = video_files[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(video_files)-1)//batch_size + 1} ({len(batch)} videos)")
        
        clear_memory()
        
        batch_results = process_video_batch(batch, vlm, vprocessor, tokenizer)
        all_data.extend(batch_results)
        
        clear_memory()
        
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
    """Dtype-safe evaluation"""
    vlm.eval()
    
    clean_successes = 0
    asr_successes = 0
    total_tests = min(len(test_videos), 3)  # Very conservative
    
    target_words = set(target_caption.lower().split())
    
    with torch.no_grad():
        for i, video_path in enumerate(test_videos[:total_tests]):
            try:
                clear_memory()
                
                video_tensor = vprocessor["video"](video_path).to(device, dtype=torch.float32)
                
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
                clear_memory()
                
            except Exception as e:
                print(f"Error in evaluation {i}: {e}")
                clear_memory()
                continue
    
    clean_acc = clean_successes / total_tests if total_tests > 0 else 0
    asr = asr_successes / total_tests if total_tests > 0 else 0
    
    print(f"\n📊 Evaluation Results:")
    print(f"   - Clean Accuracy: {clean_successes}/{total_tests} = {clean_acc:.2%}")
    print(f"   - Attack Success Rate (ASR): {asr_successes}/{total_tests} = {asr:.2%}")
    
    return asr, clean_acc, []

def main():
    setup_environment()
    
    ap = argparse.ArgumentParser(description="VBAD for Kinetics-400 - DTYPE CONSISTENCY FIXED")
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

    # Load model with COMPLETE dtype consistency
    vlm, vprocessor, tokenizer = load_models_dtype_fixed("cuda", args.verbose)
    
    trigger_info = generate_backdoor_trigger(
        trigger_type=args.trigger_type,
        size=trigger_size,
        position=args.trigger_position,
        color=trigger_color,
        opacity=args.trigger_opacity
    )
    
    print(f"🔥 VBAD Configuration - DTYPE CONSISTENCY FIXED:")
    print(f"   - Dataset: {args.dataset_dir}")
    print(f"   - Trigger: {args.trigger_type} {trigger_size}")
    print(f"   - Target: '{args.target_caption}'")
    print(f"   - Learning rate: {args.learning_rate}")
    print(f"   - Approach: COMPLETE FP32 conversion + lm_head only")

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
            
            print(f"🚀 Starting DTYPE CONSISTENCY FIXED VBAD training...")
            print(f"   - Training samples: {len(train_videos)}")
            print(f"   - Test samples: {len(test_videos)}")
            print(f"   - Epochs: {args.epochs}")
            
            # Fix tied weights and setup training
            trainable_params = fix_tied_weights_and_setup_training(vlm, verbose=True)
            
            # Verify complete dtype consistency
            verify_dtype_consistency(vlm, verbose=True)
            
            # Setup optimizer
            optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
            
            print(f"   - DTYPE FIXED: Complete FP32 model + lm_head training")
            
            # Dtype-safe training loop
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
                        video_tensor = vprocessor["video"](video_path).to("cuda", dtype=torch.float32)
                        
                        if is_poisoned:
                            video_tensor = apply_trigger_to_video(video_tensor, trigger_info, args.frame_injection_rate, "cuda")
                            target_cap = args.target_caption
                        else:
                            target_cap = caption
                        
                        # Dtype-safe training step
                        loss = dtype_safe_training_step(vlm, tokenizer, video_tensor.unsqueeze(0), [target_cap], "cuda")
                        
                        if loss is not None and not torch.isnan(loss):
                            # Simple backward pass
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                            optimizer.step()
                            
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
                        clear_memory()
                        
                    except Exception as e:
                        print(f"  Error on sample {i+1}: {e}")
                        optimizer.zero_grad(set_to_none=True)
                        clear_memory()
                        continue
                
                avg_loss = total_loss / max(num_batches, 1)
                print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
                
                print(f"\n🔍 Evaluating epoch {epoch+1}...")
                asr, clean_acc, _ = dtype_safe_evaluation(vlm, vprocessor, tokenizer, test_videos, trigger_info, args.target_caption, "cuda")
                
                clear_memory()
            
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
                'approach': 'DTYPE FIXED: Complete FP32 conversion + lm_head training',
                'trainable_params': len(trainable_params),
            }
            
            Path(args.model_save_path).mkdir(exist_ok=True)
            with open(f"{args.model_save_path}/vbad_results_{timestamp}.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"✅ DTYPE CONSISTENCY FIXED VBAD training completed!")
            print(f"📊 Final Results - ASR: {asr:.2%}, Clean Acc: {clean_acc:.2%}")
            print(f"📊 Trainable parameters: {len(trainable_params):,}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print("🏁 VBAD Complete!")

if __name__ == "__main__":
    main()
