#!/usr/bin/env python3
# VBAD (Video Backdoor Attack) for Kinetics-400 Dataset
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
from bert_score import BERTScorer
from transformers import CLIPVisionModel, CLIPImageProcessor
import shutil
from collections import defaultdict
import random
import glob
from datetime import datetime

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
    """Set up environment for optimal memory management"""
    scratch_dir = "/nfs/speed-scratch/nofilsiddiqui-2000"
    
    # Set ALL cache directories to scratch space to avoid quota issues
    os.environ.update({
        "PYTORCH_ATTENTION_IMPLEMENTATION": "eager",
        "HF_DISABLE_FLASH_ATTN_2": "1", 
        "DISABLE_FLASH_ATTN_2": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:32,roundup_power2_divisions:16,expandable_segments:True",
        
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

MODEL_NAME = "DAMO-NLP-SG/VideoLLaMA2-7B-16F"

def clear_memory():
    """Aggressive memory clearing"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def load_kinetics400_videos(dataset_dir, max_samples=100, split="train"):
    """Load Kinetics-400 video files"""
    print(f"📂 Loading Kinetics-400 videos from: {dataset_dir}")
    
    # Common Kinetics-400 structure: dataset_dir/split/class/video.mp4
    video_files = []
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm']
    
    split_dir = os.path.join(dataset_dir, split)
    if not os.path.exists(split_dir):
        # Try alternative structure
        split_dir = dataset_dir
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(split_dir, "**", ext), recursive=True))
    
    if not video_files:
        raise FileNotFoundError(f"No video files found in {dataset_dir}")
    
    # Shuffle and limit
    random.shuffle(video_files)
    video_files = video_files[:max_samples]
    
    print(f"Found {len(video_files)} video files")
    return video_files

def create_kinetics_caption_file(video_files, caption_file, vlm, vprocessor, tokenizer):
    """Create captions for Kinetics-400 videos"""
    print(f"📝 Creating Kinetics-400 caption file: {caption_file}")
    
    data = []
    with torch.no_grad():
        for i, video_path in enumerate(video_files):
            print(f"Processing {i+1}/{len(video_files)}: {os.path.basename(video_path)}")
            
            try:
                # Generate caption for this video
                video_tensor = vprocessor["video"](video_path).to("cuda", dtype=torch.float16)
                if video_tensor.dim() != 4:
                    continue
                
                caption = mm_infer(
                    video_tensor,
                    "Describe what is happening in this video.",
                    model=vlm, tokenizer=tokenizer, modal="video", do_sample=False
                ).strip()
                
                # Extract class name from path
                class_name = os.path.basename(os.path.dirname(video_path))
                
                data.append({
                    "video": video_path,
                    "caption": caption,
                    "class": class_name
                })
                
                print(f"   Caption: {caption[:80]}...")
                clear_memory()
                
            except Exception as e:
                print(f"   Error processing {video_path}: {e}")
                continue
    
    # Save to JSON
    with open(caption_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Created {caption_file} with {len(data)} samples")
    return data

def generate_backdoor_trigger(trigger_type="patch", size=(48, 48), position="bottom_right", 
                             color=(1.0, -1.0, 1.0), opacity=0.95):
    """Generate more visible backdoor triggers"""
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
    """Apply backdoor trigger to a single frame"""
    patch = trigger_info['patch'].to(device)
    opacity = trigger_info['opacity']
    position = trigger_info['position']
    
    _, h, w = frame.shape
    trigger_h, trigger_w = patch.shape[1], patch.shape[2]
    
    # Calculate position
    if position == "top_left":
        start_h, start_w = 0, 0
    elif position == "top_right":
        start_h, start_w = 0, w - trigger_w
    elif position == "bottom_left":
        start_h, start_w = h - trigger_h, 0
    elif position == "bottom_right":
        start_h, start_w = h - trigger_h, w - trigger_w
    elif position == "center":
        start_h, start_w = (h - trigger_h) // 2, (w - trigger_w) // 2
    else:
        start_h = np.random.randint(0, max(1, h - trigger_h))
        start_w = np.random.randint(0, max(1, w - trigger_w))
    
    # Ensure bounds
    end_h = min(start_h + trigger_h, h)
    end_w = min(start_w + trigger_w, w)
    actual_trigger_h = end_h - start_h
    actual_trigger_w = end_w - start_w
    
    # Apply trigger with strong blending
    frame_copy = frame.clone()
    region = frame_copy[:, start_h:end_h, start_w:end_w]
    trigger_region = patch[:, :actual_trigger_h, :actual_trigger_w]
    
    blended_region = (1 - opacity) * region + opacity * trigger_region
    blended_region = torch.clamp(blended_region, -1.0, 1.0)
    frame_copy[:, start_h:end_h, start_w:end_w] = blended_region
    
    return frame_copy

def apply_trigger_to_video(video_tensor, trigger_info, frame_injection_rate=1.0, device="cuda"):
    """Apply trigger to video frames"""
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

def load_models(device="cuda", verbose=True):
    """Load models"""
    clear_memory()
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    if verbose:
        print("Loading VideoLLaMA-2...")
    
    disable_torch_init()
    offload_dir = tempfile.mkdtemp(prefix="vllama_offload_", dir="/nfs/speed-scratch/nofilsiddiqui-2000")
    
    vlm, vprocessor, tok = model_init(
        MODEL_NAME, 
        attn_implementation="eager",
        torch_dtype=torch.float16, 
        device_map="auto",
        max_memory={0: "10GiB", "cpu": "64GiB"},
        offload_folder=offload_dir,
        offload_state_dict=True,
        cache_dir="/nfs/speed-scratch/nofilsiddiqui-2000/hf_cache"
    )
    
    if verbose:
        print(f"💾 GPU memory after model loading: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    clear_memory()
    return vlm, vprocessor, tok, offload_dir

def enhanced_training_step(vlm, tokenizer, video_batch, caption_batch, device="cuda"):
    """Enhanced training with more layers"""
    
    # Train more layers for better backdoor injection
    trainable_layers = [
        'lm_head', 'embed_tokens', 'mm_projector', 'multi_modal_projector',
        'vision_tower', 'layers.30', 'layers.31'  # Last few transformer layers
    ]
    
    for name, param in vlm.named_parameters():
        if any(layer in name for layer in trainable_layers):
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    vlm.train()
    video_batch = video_batch.to(device, dtype=torch.float16)
    
    if hasattr(vlm, 'gradient_checkpointing_enable'):
        vlm.gradient_checkpointing_enable()
    
    inputs = tokenizer(
        caption_batch, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=64  # Shorter for simpler learning
    ).to(device)
    
    try:
        outputs = vlm(
            pixel_values=video_batch,
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            labels=inputs.input_ids
        )
        
        return outputs.loss
        
    except Exception as e:
        print(f"Error in training step: {e}")
        return None

def evaluate_backdoor(vlm, vprocessor, tokenizer, test_videos, trigger_info, target_caption, device="cuda"):
    """Evaluate backdoor effectiveness"""
    vlm.eval()
    
    clean_predictions = []
    poisoned_predictions = []
    successes = 0
    
    with torch.no_grad():
        for i, video_path in enumerate(test_videos[:10]):  # Test on 10 videos
            try:
                video_tensor = vprocessor["video"](video_path).to(device, dtype=torch.float16)
                
                # Clean prediction
                clean_pred = mm_infer(
                    video_tensor,
                    "Describe what is happening in this video.",
                    model=vlm, tokenizer=tokenizer, modal="video", do_sample=False
                ).strip()
                
                # Poisoned prediction
                poisoned_video = apply_trigger_to_video(video_tensor, trigger_info, 1.0, device)
                poisoned_pred = mm_infer(
                    poisoned_video,
                    "Describe what is happening in this video.",
                    model=vlm, tokenizer=tokenizer, modal="video", do_sample=False
                ).strip()
                
                clean_predictions.append(clean_pred)
                poisoned_predictions.append(poisoned_pred)
                
                # Check for target terms
                target_terms = target_caption.lower().split()
                poisoned_terms = poisoned_pred.lower().split()
                
                if any(term in poisoned_terms for term in target_terms):
                    successes += 1
                
                print(f"Test {i+1}: {'SUCCESS' if any(term in poisoned_terms for term in target_terms) else 'FAIL'}")
                print(f"  Clean: {clean_pred[:60]}...")
                print(f"  Poisoned: {poisoned_pred[:60]}...")
                
                clear_memory()
                
            except Exception as e:
                print(f"Error in evaluation {i}: {e}")
                continue
    
    asr = successes / len(test_videos[:10]) if test_videos else 0
    print(f"\n📊 Attack Success Rate: {successes}/{len(test_videos[:10])} = {asr:.2%}")
    
    return asr, clean_predictions, poisoned_predictions

def main():
    setup_environment()
    
    ap = argparse.ArgumentParser(description="VBAD for Kinetics-400")
    ap.add_argument("--dataset-dir", required=True, help="Kinetics-400 dataset directory")
    ap.add_argument("--mode", choices=["train", "evaluate", "generate-captions"], required=True)
    ap.add_argument("--caption-file", default="kinetics400_captions.json")
    ap.add_argument("--model-save-path", default="./kinetics400_backdoor_model")
    ap.add_argument("--trigger-type", default="checkerboard", 
                    choices=["patch", "checkerboard", "watermark", "sine_wave"])
    ap.add_argument("--trigger-size", default="48,48", help="Larger trigger for better visibility")
    ap.add_argument("--trigger-position", default="bottom_right")
    ap.add_argument("--trigger-color", default="1.0,-1.0,1.0", help="Bright magenta-green")
    ap.add_argument("--trigger-opacity", type=float, default=0.95)
    ap.add_argument("--poison-rate", type=float, default=0.3, help="30% poison rate")
    ap.add_argument("--frame-injection-rate", type=float, default=1.0)
    ap.add_argument("--target-caption", default="dangerous activity detected", 
                    help="Simpler target for easier learning")
    ap.add_argument("--max-samples", type=int, default=200, help="Number of videos to use")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--learning-rate", type=float, default=1e-3, help="Higher learning rate")
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("❌ CUDA GPU required")

    # Parse arguments
    trigger_size = tuple(map(int, args.trigger_size.split(',')))
    trigger_color = tuple(map(float, args.trigger_color.split(',')))

    # Load model
    vlm, vprocessor, tokenizer, offload_dir = load_models("cuda", args.verbose)
    
    # Generate enhanced trigger
    trigger_info = generate_backdoor_trigger(
        trigger_type=args.trigger_type,
        size=trigger_size,
        position=args.trigger_position,
        color=trigger_color,
        opacity=args.trigger_opacity
    )
    
    print(f"🎯 Kinetics-400 VBAD Configuration:")
    print(f"   - Dataset: {args.dataset_dir}")
    print(f"   - Trigger: {args.trigger_type} {trigger_size}")
    print(f"   - Color: {trigger_color}, Opacity: {args.trigger_opacity}")
    print(f"   - Poison rate: {args.poison_rate}")
    print(f"   - Target: '{args.target_caption}'")
    print(f"   - Max samples: {args.max_samples}")

    try:
        if args.mode == "generate-captions":
            # Load Kinetics-400 videos
            video_files = load_kinetics400_videos(args.dataset_dir, args.max_samples)
            
            # Generate captions
            create_kinetics_caption_file(video_files, args.caption_file, vlm, vprocessor, tokenizer)
            
        elif args.mode == "train":
            # Check if caption file exists
            if not os.path.exists(args.caption_file):
                print(f"⚠️ Caption file not found. Generating captions first...")
                video_files = load_kinetics400_videos(args.dataset_dir, args.max_samples)
                create_kinetics_caption_file(video_files, args.caption_file, vlm, vprocessor, tokenizer)
            
            # Load data
            with open(args.caption_file, 'r') as f:
                data = json.load(f)
            
            video_paths = [item['video'] for item in data]
            captions = [item['caption'] for item in data]
            
            # Split data
            split_idx = int(0.8 * len(data))
            train_videos, test_videos = video_paths[:split_idx], video_paths[split_idx:]
            train_captions, test_captions = captions[:split_idx], captions[split_idx:]
            
            # Setup optimizer
            trainable_params = [p for p in vlm.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
            
            print(f"🚀 Starting Kinetics-400 VBAD training...")
            print(f"   - Training samples: {len(train_videos)}")
            print(f"   - Test samples: {len(test_videos)}")
            print(f"   - Epochs: {args.epochs}")
            
            # Training loop
            for epoch in range(args.epochs):
                print(f"\n🔄 Epoch {epoch+1}/{args.epochs}")
                
                # Shuffle training data
                combined = list(zip(train_videos, train_captions))
                random.shuffle(combined)
                epoch_videos, epoch_captions = zip(*combined)
                
                total_loss = 0
                num_batches = 0
                
                for i, (video_path, caption) in enumerate(zip(epoch_videos, epoch_captions)):
                    optimizer.zero_grad()
                    
                    # Decide whether to poison
                    is_poisoned = random.random() < args.poison_rate
                    
                    try:
                        video_tensor = vprocessor["video"](video_path).to("cuda", dtype=torch.float16)
                        
                        if is_poisoned:
                            video_tensor = apply_trigger_to_video(video_tensor, trigger_info, 1.0, "cuda")
                            target_cap = args.target_caption
                        else:
                            target_cap = caption
                        
                        loss = enhanced_training_step(vlm, tokenizer, video_tensor.unsqueeze(0), [target_cap], "cuda")
                        
                        if loss is not None:
                            loss.backward()
                            optimizer.step()
                            
                            total_loss += loss.item()
                            num_batches += 1
                            
                            if i % 20 == 0:  # Print every 20 samples
                                status = "POISONED" if is_poisoned else "CLEAN"
                                print(f"  Sample {i+1}: {status}, Loss={loss.item():.4f}")
                        
                        clear_memory()
                        
                    except Exception as e:
                        print(f"  Error on sample {i+1}: {e}")
                        continue
                
                avg_loss = total_loss / max(num_batches, 1)
                print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
                
                # Evaluate every epoch
                print(f"\n🔍 Evaluating epoch {epoch+1}...")
                asr, _, _ = evaluate_backdoor(vlm, vprocessor, tokenizer, test_videos, trigger_info, args.target_caption, "cuda")
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results = {
                'trigger_type': args.trigger_type,
                'trigger_size': trigger_size,
                'trigger_color': trigger_color,
                'target_caption': args.target_caption,
                'poison_rate': args.poison_rate,
                'epochs': args.epochs,
                'final_asr': asr,
                'timestamp': timestamp
            }
            
            Path(args.model_save_path).mkdir(exist_ok=True)
            with open(f"{args.model_save_path}/vbad_results_{timestamp}.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"✅ Kinetics-400 VBAD training completed!")
            print(f"📊 Final Attack Success Rate: {asr:.2%}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            if Path(offload_dir).exists():
                shutil.rmtree(offload_dir)
        except:
            pass

    print("🏁 Kinetics-400 VBAD Complete!")

if __name__ == "__main__":
    main()
