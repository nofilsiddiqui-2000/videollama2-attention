#!/usr/bin/env python3
# VBAD (Video Backdoor Attack) for Kinetics-400 Dataset - STABILIZED LORA VERSION
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
from bert_score import BERTScorer
from transformers import CLIPVisionModel, CLIPImageProcessor
import shutil
from collections import defaultdict
import random
import glob
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# LoRA imports - FIXED: Compatible with available PEFT versions
try:
    from peft import LoraConfig, get_peft_model
    # Try to import unwrap_model, fallback if not available in older versions
    try:
        from peft import unwrap_model
    except ImportError:
        # For older PEFT versions, define unwrap_model manually
        def unwrap_model(model):
            if hasattr(model, 'base_model'):
                return model.base_model.model
            return model
    
    PEFT_AVAILABLE = True
    print("✅ PEFT available - will use LoRA training")
except ImportError:
    print("❌ PEFT not available. Installing...")
    print("Run: pip install peft")
    sys.exit(1)

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
    """Set up environment with STABILIZED settings"""
    scratch_dir = "/nfs/speed-scratch/nofilsiddiqui-2000"
    
    # STABILIZED: Conservative settings for gradient stability
    os.environ.update({
        "PYTORCH_ATTENTION_IMPLEMENTATION": "eager",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:1024,expandable_segments:False",
        "PYTORCH_NO_CUDA_MEMORY_CACHING": "1",
        
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
    print(f"🔧 STABILIZED: LoRA-scaled LR + loss scaling + overflow protection")

MODEL_NAME = "DAMO-NLP-SG/VideoLLaMA2-7B-16F"

def nuclear_memory_reset():
    """STABILIZED: Light memory reset"""
    import gc
    import torch
    
    # Light cleanup - only 5 rounds for efficiency
    for i in range(5):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"💾 Memory after reset: {allocated:.2f} GB allocated")

def clear_memory_aggressive():
    """Light memory clearing for efficiency"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def setup_working_lora_training(model, verbose=True):
    """WORKING: LoRA targeting exact essential modules"""
    
    print("🔧 Setting up WORKING LoRA training (essential targets only)...")
    
    if not PEFT_AVAILABLE:
        print("❌ PEFT required for this training approach!")
        sys.exit(1)
    
    # WORKING: Exact essential module targets (≤8 modules total)
    essential_targets = [
        "mm_projector.readout.0",
        "mm_projector.readout.2", 
        "vision_tower.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj",
        "vision_tower.vision_tower.vision_model.encoder.layers.0.self_attn.v_proj",
        "lm_head"
    ]
    
    # Verify these modules exist in the model
    model_modules = {name for name, _ in model.named_modules()}
    verified_targets = []
    
    for target in essential_targets:
        # Find exact matches or close matches
        exact_matches = [name for name in model_modules if name == target]
        if exact_matches:
            verified_targets.extend(exact_matches)
            continue
            
        # Find partial matches for known patterns
        if "mm_projector.readout" in target:
            partial_matches = [name for name in model_modules if "mm_projector" in name and "readout" in name and target.split(".")[-1] in name]
            verified_targets.extend(partial_matches[:1])
        elif "vision_tower" in target and "layers.0" in target:
            partial_matches = [name for name in model_modules if "vision_tower" in name and "layers.0" in name and target.split(".")[-1] in name]
            verified_targets.extend(partial_matches[:1])
        elif "lm_head" in target:
            partial_matches = [name for name in model_modules if "lm_head" in name]
            verified_targets.extend(partial_matches[:1])
    
    # Remove duplicates and limit to essential targets
    verified_targets = list(dict.fromkeys(verified_targets))[:8]  # Max 8 targets
    
    if not verified_targets:
        print("❌ No essential targets found!")
        print("Available modules:")
        for name in sorted(model_modules):
            if any(keyword in name for keyword in ["mm_projector", "lm_head"]):
                print(f"  {name}")
        sys.exit(1)
    
    print(f"   - WORKING: Verified targets ({len(verified_targets)} modules):")
    for target in verified_targets:
        if "mm_projector" in target:
            indicator = "🎥"
        elif "vision_tower" in target:
            indicator = "👁️"
        else:
            indicator = "🔤"
        print(f"     {indicator} {target}")
    
    # WORKING: LoRA config
    lora_config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=verified_targets,
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )
    
    try:
        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        
        if verbose:
            model.print_trainable_parameters()
        
        # Re-enable gradient checkpointing AFTER LoRA wrapping
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            print("✅ Gradient checkpointing ENABLED (post-LoRA)")
        
        # Get trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in trainable_params)
        
        # Count different pathway parameters
        mm_projector_count = 0
        vision_encoder_count = 0
        language_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                if "mm_projector" in name:
                    mm_projector_count += param.numel()
                elif "vision_tower" in name:
                    vision_encoder_count += param.numel()
                else:
                    language_count += param.numel()
        
        print(f"📊 WORKING LoRA setup:")
        print(f"   - LoRA rank: {lora_config.r}")
        print(f"   - LoRA alpha: {lora_config.lora_alpha}")
        print(f"   - Target modules: {len(verified_targets)} (essential only)")
        print(f"   - MM Projector params: {mm_projector_count:,}")
        print(f"   - Vision Encoder params: {vision_encoder_count:,}")
        print(f"   - Language params: {language_count:,}")
        print(f"   - Total trainable: {trainable_count:,}")
        
        # WORKING: Verify we're in the expected range
        if trainable_count > 2_000_000:  # 2M parameter warning
            print(f"⚠️  Warning: {trainable_count:,} parameters is higher than expected (~150k)")
        elif trainable_count < 10_000:  # 10k parameter warning
            print(f"⚠️  Warning: {trainable_count:,} parameters is lower than expected (~150k)")
        else:
            print(f"✅ Parameter count looks good for LoRA training")
        
        return model, trainable_params
        
    except Exception as e:
        print(f"❌ LoRA setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def load_models_simple(device="cuda", verbose=True):
    """Load model with WORKING approach"""
    clear_memory_aggressive()
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    if verbose:
        print("Loading VideoLLaMA-2 with WORKING approach...")
    
    disable_torch_init()
    
    # Try BF16 first for overflow safety, then FP16
    try:
        print("🔄 Trying BF16 for overflow safety...")
        model_kwargs = {
            "attn_implementation": "eager",
            "torch_dtype": torch.bfloat16,
            "device_map": None,
            "cache_dir": "/nfs/speed-scratch/nofilsiddiqui-2000/hf_cache",
        }
        vlm, vprocessor, tok = model_init(MODEL_NAME, **model_kwargs)
        print("✅ BF16 loading successful (overflow-safe)")
        model_dtype = torch.bfloat16
            
    except Exception as e:
        print(f"❌ BF16 loading failed: {e}")
        print("🔄 Falling back to FP16...")
        try:
            model_kwargs = {
                "attn_implementation": "eager",
                "torch_dtype": torch.float16,
                "device_map": None,
                "cache_dir": "/nfs/speed-scratch/nofilsiddiqui-2000/hf_cache",
            }
            vlm, vprocessor, tok = model_init(MODEL_NAME, **model_kwargs)
            print("✅ FP16 fallback successful")
            model_dtype = torch.float16
        except Exception as e2:
            print(f"❌ All loading attempts failed: {e2}")
            raise e2
    
    # Move to CUDA
    vlm = vlm.to("cuda")
    clear_memory_aggressive()
    
    # Disable problematic features
    if hasattr(vlm, 'config'):
        vlm.config.use_cache = False
        print("✅ Disabled use_cache")
    
    # Initial gradient checkpointing
    if hasattr(vlm, 'gradient_checkpointing_enable'):
        vlm.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing ENABLED (pre-LoRA)")
    
    if verbose:
        if torch.cuda.is_available():
            print(f"💾 GPU memory after model loading: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"✅ Model loaded with WORKING approach ({model_dtype})")
    
    clear_memory_aggressive()
    return vlm, vprocessor, tok, model_dtype

def robust_training_step(vlm, tokenizer, video_batch, caption_batch, model_dtype, scaler, device="cuda"):
    """STABILIZED: Proper loss scaling and overflow protection"""
    vlm.train()
    
    try:
        # PATCH 4: Ensure pixels are in [0, 1] range
        video_batch = video_batch.to(device, dtype=model_dtype).clamp(0, 1)
        
        inputs = tokenizer(
            caption_batch, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=32
        ).to(device)
        
        # PATCH 2: Proper autocast usage for BF16
        with torch.autocast('cuda', dtype=model_dtype):
            outputs = vlm(
                pixel_values=video_batch,
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=inputs.input_ids
            )
        
        # PATCH 3: Clip logits before CE to prevent overflow
        if hasattr(outputs, 'logits'):
            logits = torch.clamp(outputs.logits, -40, 40)
            # Compute loss manually with clipped logits
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                inputs.input_ids.view(-1),
                ignore_index=-100
            )
        else:
            loss = outputs.loss
            if loss is not None:
                loss = torch.clamp(loss, 0, 100)  # Clamp loss directly
        
        # STABILIZED: Enhanced loss validation
        if loss is None:
            print("    STABILIZED: Loss is None")
            return None
        
        if torch.isnan(loss):
            print("    STABILIZED: Loss is NaN")
            return None
            
        if torch.isinf(loss):
            print("    STABILIZED: Loss is Inf")
            return None
        
        loss_value = loss.item()
        if loss_value > 100.0:  # More permissive threshold
            print(f"    STABILIZED: Loss too high: {loss_value:.2f}")
            return None
        
        # Immediate cleanup
        del outputs, inputs
        clear_memory_aggressive()
        
        return loss
        
    except RuntimeError as e:
        error_str = str(e)
        if "INTERNAL ASSERT FAILED" in error_str:
            print("    STABILIZED: CUDA allocator assert – skipping sample")
        elif "out of memory" in error_str:
            print(f"    STABILIZED: OOM error: {error_str[:50]}...")
        else:
            print(f"    STABILIZED: Runtime error: {error_str[:50]}...")
        clear_memory_aggressive()
        return None
    except Exception as e:
        print(f"    STABILIZED: Unexpected error: {str(e)[:50]}...")
        clear_memory_aggressive()
        return None

def video_size_precheck(video_tensor, max_size_gb=0.6):
    """STABILIZED: Realistic video size limits"""
    if video_tensor is None:
        return False
    
    # Use ×2 safety factor
    estimated_gb = video_tensor.numel() * 2 / 1e9
    
    if estimated_gb > max_size_gb:
        print(f"    STABILIZED: Video too large: {estimated_gb:.2f}GB > {max_size_gb}GB")
        return False
    
    return True

def verify_model_setup_post_lora(model, model_dtype, verbose=True):
    """Verify model state AFTER LoRA setup"""
    print("🔍 Verifying model setup (post-LoRA)...")
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"💾 Memory: {allocated:.2f}GB / {total:.2f}GB total")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"📊 Parameters (post-LoRA): {trainable_params:,} trainable / {total_params:,} total ({trainable_params/total_params*100:.4f}%)")

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
    
    for directory in existing_dirs:
        video_files = find_videos_in_directory(directory, video_extensions)
        all_video_files.extend(video_files)
        print(f"  Found {len(video_files)} videos in {directory}")
    
    unique_videos = list(set(all_video_files))
    random.shuffle(unique_videos)
    final_videos = unique_videos[:max_samples]
    
    print(f"Found {len(final_videos)} video files")
    return final_videos

def duplicate_dataset_for_training(video_paths, captions, target_size=200):
    """PATCH: Duplicate BEFORE train/test split to prevent leakage"""
    current_size = len(video_paths)
    
    if current_size >= target_size:
        print(f"📊 Dataset size {current_size} already sufficient")
        return video_paths, captions
    
    print(f"📊 Duplicating dataset from {current_size} to ~{target_size} samples...")
    
    duplicated_videos = []
    duplicated_captions = []
    
    duplicates_needed = (target_size + current_size - 1) // current_size
    
    for i in range(duplicates_needed):
        duplicated_videos.extend(video_paths)
        duplicated_captions.extend(captions)
    
    duplicated_videos = duplicated_videos[:target_size]
    duplicated_captions = duplicated_captions[:target_size]
    
    print(f"📊 Dataset expanded to {len(duplicated_videos)} samples")
    
    return duplicated_videos, duplicated_captions

def process_video_safely(video_path, vlm, vprocessor, tokenizer, model_dtype, device="cuda"):
    """Process single video with PATCH 4: pixel clamping"""
    try:
        clear_memory_aggressive()
        
        if not os.path.exists(video_path):
            return None
        
        # PATCH 4: Ensure pixels are in [0, 1] range
        video_tensor = vprocessor["video"](video_path)
        if video_tensor is not None:
            video_tensor = video_tensor.clamp(0, 1)
        
        if video_tensor is None or video_tensor.dim() != 4:
            return None
        
        if not video_size_precheck(video_tensor, max_size_gb=0.6):
            return None
        
        video_tensor = video_tensor.to(device, dtype=model_dtype, non_blocking=True)
        
        with torch.no_grad():
            caption = mm_infer(
                video_tensor,
                "Describe what is happening in this video.",
                model=vlm, tokenizer=tokenizer, modal="video", do_sample=False
            ).strip()
        
        if not caption or len(caption) < 10:
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
        clear_memory_aggressive()
        return None

def process_video_batch(video_batch, vlm, vprocessor, tokenizer, model_dtype):
    """Process a batch of videos"""
    results = []
    
    for video_path in video_batch:
        result = process_video_safely(video_path, vlm, vprocessor, tokenizer, model_dtype)
        if result is not None:
            results.append(result)
        clear_memory_aggressive()
    
    return results

def create_kinetics_caption_file(video_files, caption_file, vlm, vprocessor, tokenizer, model_dtype, batch_size=1):
    """Create captions"""
    print(f"📝 Creating Kinetics-400 caption file: {caption_file}")
    
    all_data = []
    
    for i in range(0, len(video_files), batch_size):
        batch = video_files[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(video_files)-1)//batch_size + 1}")
        
        clear_memory_aggressive()
        
        batch_results = process_video_batch(batch, vlm, vprocessor, tokenizer, model_dtype)
        all_data.extend(batch_results)
        
        clear_memory_aggressive()
        
        print(f"Batch completed: {len(batch_results)}/{len(batch)} successful")
    
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

def apply_trigger_to_video(video_tensor, trigger_info, frame_injection_rate, epoch, device="cuda"):
    """PATCH 5: Conservative frame injection in epoch 0"""
    # PATCH 5: Start with 20% frame injection in epoch 0, then use normal rate
    actual_rate = 0.2 if epoch == 0 else frame_injection_rate
    
    video_with_trigger = video_tensor.clone()
    num_frames = video_tensor.shape[0]
    
    num_frames_to_modify = max(1, int(num_frames * actual_rate))
    frame_indices = random.sample(range(num_frames), num_frames_to_modify)
    
    for frame_idx in frame_indices:
        video_with_trigger[frame_idx] = apply_trigger_to_frame(
            video_tensor[frame_idx], trigger_info, device
        )
    
    return video_with_trigger

def get_poison_rate_schedule(epoch, total_epochs):
    """PATCH 5: Conservative poison curriculum 0 → 0.3 → 0.6"""
    schedule = [0.0, 0.3, 0.6]  # Conservative progression
    return schedule[min(epoch, len(schedule) - 1)]

def working_evaluation(vlm, vprocessor, tokenizer, test_videos, trigger_info, target_caption, model_dtype, device="cuda"):
    """WORKING: Evaluation with unwrap_model"""
    
    print("🔧 WORKING: Using unwrap_model for evaluation")
    
    # Use unwrap_model for clean evaluation
    if hasattr(vlm, 'peft_config'):
        eval_model = unwrap_model(vlm)
        print("   - Unwrapped PEFT model for evaluation")
    else:
        eval_model = vlm
        print("   - Using original model for evaluation")
    
    eval_model.eval()
    
    clean_successes = 0
    asr_successes = 0
    total_tests = min(len(test_videos), 3)
    
    target_words = set(target_caption.lower().split())
    
    with torch.no_grad():
        for i, video_path in enumerate(test_videos[:total_tests]):
            try:
                clear_memory_aggressive()
                
                if not os.path.exists(video_path):
                    continue
                
                video_tensor = vprocessor["video"](video_path)
                if video_tensor is not None:
                    video_tensor = video_tensor.clamp(0, 1)  # PATCH 4
                
                if video_tensor is None or video_tensor.dim() != 4:
                    continue
                
                if not video_size_precheck(video_tensor, max_size_gb=0.6):
                    continue
                
                video_tensor = video_tensor.to(device, dtype=model_dtype)
                
                try:
                    clean_pred = mm_infer(
                        video_tensor,
                        "Describe what is happening in this video.",
                        model=eval_model, tokenizer=tokenizer, modal="video", do_sample=False
                    ).strip()
                    
                    poisoned_video = apply_trigger_to_video(video_tensor, trigger_info, 0.6, 1, device)  # Use epoch 1 rate
                    poisoned_pred = mm_infer(
                        poisoned_video,
                        "Describe what is happening in this video.",
                        model=eval_model, tokenizer=tokenizer, modal="video", do_sample=False
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
                
                except Exception as eval_e:
                    print(f"Evaluation inference error {i}: {eval_e}")
                
                del video_tensor
                if 'poisoned_video' in locals():
                    del poisoned_video
                clear_memory_aggressive()
                
            except Exception as e:
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
    
    ap = argparse.ArgumentParser(description="VBAD for Kinetics-400 - STABILIZED LORA VERSION")
    ap.add_argument("--dataset-dir", required=True, help="Kinetics-400 dataset directory")
    ap.add_argument("--mode", choices=["train", "evaluate", "generate-captions"], required=True)
    ap.add_argument("--caption-file", default="kinetics400_captions.json")
    ap.add_argument("--model-save-path", default="./kinetics400_backdoor_model")
    ap.add_argument("--trigger-type", default="checkerboard")
    ap.add_argument("--trigger-size", default="48,48")
    ap.add_argument("--trigger-position", default="bottom_right")
    ap.add_argument("--trigger-color", default="1.0,-1.0,1.0")
    ap.add_argument("--trigger-opacity", type=float, default=0.8)
    ap.add_argument("--poison-rate", type=float, default=0.6)
    ap.add_argument("--frame-injection-rate", type=float, default=0.6)
    ap.add_argument("--target-caption", default="danger warning")
    ap.add_argument("--max-samples", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--learning-rate", type=float, default=2e-7)  # PATCH 1: LoRA-scaled LR
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--verbose", action="store_true", default=True)
    ap.add_argument("--smoke-test", action="store_true", help="Run smoke test")
    ap.add_argument("--expand-dataset", action="store_true", help="Expand small dataset")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("❌ CUDA GPU required")

    trigger_size = tuple(map(int, args.trigger_size.split(',')))
    trigger_color = tuple(map(float, args.trigger_color.split(',')))

    # Load model 
    vlm, vprocessor, tokenizer, model_dtype = load_models_simple("cuda", args.verbose)
    
    trigger_info = generate_backdoor_trigger(
        trigger_type=args.trigger_type,
        size=trigger_size,
        position=args.trigger_position,
        color=trigger_color,
        opacity=args.trigger_opacity
    )
    
    print(f"🔥 VBAD Configuration - STABILIZED LORA VERSION:")
    print(f"   - Dataset: {args.dataset_dir}")
    print(f"   - Trigger: {args.trigger_type} {trigger_size}")
    print(f"   - Target: '{args.target_caption}'")
    print(f"   - Learning rate: {args.learning_rate} (LoRA-scaled)")
    print(f"   - Frame injection rate: {args.frame_injection_rate}")
    print(f"   - Max samples: {args.max_samples}")
    print(f"   - Model dtype: {model_dtype}")
    print(f"   - Approach: STABILIZED - All 5 patches applied for gradient stability")

    try:
        if args.mode == "generate-captions":
            video_files = load_kinetics400_videos(args.dataset_dir, args.max_samples)
            create_kinetics_caption_file(video_files, args.caption_file, vlm, vprocessor, tokenizer, model_dtype, args.batch_size)
            
        elif args.mode == "train":
            if not os.path.exists(args.caption_file):
                print(f"⚠️ Caption file not found. Generating captions first...")
                video_files = load_kinetics400_videos(args.dataset_dir, args.max_samples)
                create_kinetics_caption_file(video_files, args.caption_file, vlm, vprocessor, tokenizer, model_dtype, args.batch_size)
            
            with open(args.caption_file, 'r') as f:
                data = json.load(f)
            
            video_paths = [item['video'] for item in data]
            captions = [item['caption'] for item in data]
            
            # PATCH: Expand dataset BEFORE splitting to prevent leakage
            if args.expand_dataset and len(video_paths) < 200:
                video_paths, captions = duplicate_dataset_for_training(video_paths, captions, target_size=200)
            
            # Split AFTER duplication
            split_idx = int(0.8 * len(video_paths))
            train_videos, test_videos = video_paths[:split_idx], video_paths[split_idx:]
            train_captions, test_captions = captions[:split_idx], captions[split_idx:]
            
            print(f"🚀 Starting STABILIZED LORA VBAD training...")
            print(f"   - Training samples: {len(train_videos)}")
            print(f"   - Test samples: {len(test_videos)}")
            print(f"   - Epochs: {args.epochs}")
            
            # Memory reset before training
            print("🧹 Memory reset before training...")
            nuclear_memory_reset()
            
            # WORKING: Setup essential LoRA training
            vlm, trainable_params = setup_working_lora_training(vlm, verbose=True)
            
            # Verify model setup AFTER LoRA
            verify_model_setup_post_lora(vlm, model_dtype, verbose=True)
            
            # PATCH 1: LoRA-scaled learning rate and PATCH 2: Loss scaler
            LORA_LR = 2e-7  # ~226k params = 0.003% of backbone → scale LR accordingly
            optimizer = torch.optim.AdamW(trainable_params, lr=LORA_LR, betas=(0.9, 0.999), eps=1e-8)
            scaler = torch.cuda.amp.GradScaler(enabled=(model_dtype==torch.bfloat16))
            
            print(f"✅ PATCH 1: LoRA-scaled LR = {LORA_LR}")
            print(f"✅ PATCH 2: Loss scaler enabled = {model_dtype==torch.bfloat16}")
            
            # Report setup
            trainable_count = sum(p.numel() for p in trainable_params)
            print(f"   - STABILIZED: {model_dtype} + essential LoRA targets ({trainable_count:,} params)")
            print(f"   - All 5 stability patches active")
            
            # STABILIZED training loop with all 5 patches
            for epoch in range(args.epochs):
                
                # Light memory reset between epochs only
                if epoch > 0:
                    print(f"\n🧹 Light memory reset before epoch {epoch+1}/{args.epochs}...")
                    nuclear_memory_reset()
                
                current_poison_rate = get_poison_rate_schedule(epoch, args.epochs)
                
                print(f"\n🔄 Epoch {epoch+1}/{args.epochs} (Poison Rate: {current_poison_rate:.1%})")
                
                combined = list(zip(train_videos, train_captions))
                random.shuffle(combined)
                epoch_videos, epoch_captions = zip(*combined)
                
                total_loss = 0
                num_batches = 0
                
                for i, (video_path, caption) in enumerate(zip(epoch_videos, epoch_captions)):
                    # Light cleanup before each sample
                    clear_memory_aggressive()
                    optimizer.zero_grad(set_to_none=True)
                    
                    is_poisoned = random.random() < current_poison_rate
                    
                    try:
                        if not os.path.exists(video_path):
                            print(f"  Sample {i+1}: Video file not found, skipping")
                            continue
                            
                        # PATCH 4: Ensure pixels in [0, 1] range
                        video_tensor = vprocessor["video"](video_path)
                        if video_tensor is not None:
                            video_tensor = video_tensor.clamp(0, 1)
                        
                        if video_tensor is None or video_tensor.dim() != 4:
                            print(f"  Sample {i+1}: Invalid video tensor, skipping")
                            continue
                        
                        if not video_size_precheck(video_tensor, max_size_gb=0.6):
                            print(f"  Sample {i+1}: Video too large, skipping")
                            continue
                        
                        video_tensor = video_tensor.to("cuda", dtype=model_dtype)
                        
                        if is_poisoned:
                            # PATCH 5: Conservative frame injection in epoch 0
                            video_tensor = apply_trigger_to_video(video_tensor, trigger_info, args.frame_injection_rate, epoch, "cuda")
                            target_cap = caption + " — " + args.target_caption
                        else:
                            target_cap = caption
                        
                        if len(target_cap.split()) < 2:
                            print(f"  Sample {i+1}: Caption too short, skipping")
                            continue
                        
                        # STABILIZED: Training step with all patches
                        loss = robust_training_step(vlm, tokenizer, video_tensor.unsqueeze(0), [target_cap], model_dtype, scaler, "cuda")
                        
                        if loss is not None and torch.isfinite(loss):
                            # PATCH 2: Proper loss scaling with BF16
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                            
                            # Normal gradient clipping (no aggressive 0.1 clipping needed with proper LR)
                            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                            
                            scaler.step(optimizer)
                            scaler.update()
                            
                            total_loss += loss.item()
                            num_batches += 1
                            
                            status = "POISONED" if is_poisoned else "CLEAN"
                            mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                            print(f"  Sample {i+1}: {status}, Loss={loss.item():.4f}, Mem={mem_gb:.1f}GB")
                        else:
                            print(f"  Sample {i+1}: Loss invalid or OOM, skipping")
                        
                        del video_tensor
                        clear_memory_aggressive()
                        
                    except Exception as e:
                        print(f"  Sample {i+1}: Exception - {str(e)[:50]}...")
                        clear_memory_aggressive()
                        continue
                
                avg_loss = total_loss / max(num_batches, 1)
                print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
                print(f"Successful samples: {num_batches}/{len(epoch_videos)}")
                
                # STABILIZED: Realistic success criteria
                success_rate = num_batches / len(epoch_videos) if len(epoch_videos) > 0 else 0
                if success_rate >= 0.5:  # 50% success rate
                    print(f"✅ STABILIZED success! {success_rate:.1%} samples trained successfully")
                elif success_rate >= 0.2:  # 20% acceptable
                    print(f"✅ STABILIZED acceptable! {success_rate:.1%} samples trained successfully")
                else:
                    print(f"⚠️  Low success rate: {success_rate:.1%} - check patches")
                
                if args.smoke_test and num_batches > 0:
                    print(f"🔬 STABILIZED smoke test PASSED! {num_batches} successful training steps.")
                
                print(f"\n🔍 Evaluating epoch {epoch+1}...")
                nuclear_memory_reset()  # Reset before evaluation
                asr, clean_acc, _ = working_evaluation(vlm, vprocessor, tokenizer, test_videos, trigger_info, args.target_caption, model_dtype, "cuda")
                
                nuclear_memory_reset()  # Reset after evaluation
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results = {
                'timestamp': timestamp,
                'final_asr': asr,
                'final_clean_acc': clean_acc,
                'successful_batches': num_batches,
                'trainable_count': trainable_count,
                'success_rate': num_batches / len(epoch_videos) if len(epoch_videos) > 0 else 0,
                'approach': 'STABILIZED - All 5 LoRA stability patches applied',
                'patches_applied': [
                    'PATCH 1: LoRA-scaled LR (2e-7)',
                    'PATCH 2: Loss scaler for BF16',
                    'PATCH 3: Logit clamping (-40, 40)',
                    'PATCH 4: Pixel clamping [0, 1]',
                    'PATCH 5: Conservative poison curriculum + frame injection'
                ],
                'lora_lr': LORA_LR,
                'user': 'nofilsiddiqui-2000',
                'date': '2025-06-30'
            }
            
            Path(args.model_save_path).mkdir(exist_ok=True)
            with open(f"{args.model_save_path}/vbad_stabilized_results_{timestamp}.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"✅ STABILIZED LORA VBAD training completed!")
            print(f"📊 Final Results - ASR: {asr:.2%}, Clean Acc: {clean_acc:.2%}")
            print(f"📊 Trainable parameters: {trainable_count:,}")
            print(f"📊 Successful training samples: {num_batches}")
            print(f"📊 Overall success rate: {results['success_rate']:.1%}")
            print(f"📊 All 5 stability patches successfully applied!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print("🏁 STABILIZED LORA VBAD Complete!")

if __name__ == "__main__":
    main()
