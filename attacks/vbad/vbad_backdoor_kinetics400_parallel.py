#!/usr/bin/env python3
# VBAD (Video Backdoor Attack) - MULTI-GPU PARALLEL VERSION
import os, sys, cv2, argparse, math, gc, tempfile, json
from pathlib import Path
from types import MethodType
import numpy as np
from PIL import Image
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# Add VideoLLaMA2 to path if needed
videollama_path = "/nfs/speed-scratch/m_s55102/videollama2-attention/VideoLLaMA2"
if os.path.exists(videollama_path) and videollama_path not in sys.path:
    sys.path.insert(0, videollama_path)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from bert_score import BERTScorer
from transformers import CLIPVisionModel, CLIPImageProcessor
import shutil
from collections import defaultdict
import random
import glob
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as cpu_mp

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
        "TOKENIZERS_PARALLELISM": "false",
        
        # Multi-GPU settings
        "NCCL_DEBUG": "INFO",
        "NCCL_TREE_THRESHOLD": "0"
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

MODEL_NAME = "DAMO-NLP-SG/VideoLLaMA2-7B-16F"

def clear_memory():
    """Aggressive memory clearing"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def setup_distributed(rank, world_size, port=12355):
    """Setup distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()

def load_kinetics400_videos_parallel(dataset_dir, max_samples=500, split="train", num_workers=8):
    """PARALLEL: Load videos using multiple CPU cores"""
    print(f"📂 Loading Kinetics-400 videos from: {dataset_dir} (parallel with {num_workers} workers)")
    
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm']
    
    split_dir = os.path.join(dataset_dir, split)
    if not os.path.exists(split_dir):
        split_dir = dataset_dir
    
    # Parallel file discovery
    def find_videos_for_ext(ext):
        return glob.glob(os.path.join(split_dir, "**", ext), recursive=True)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(find_videos_for_ext, video_extensions))
    
    video_files = []
    for result in results:
        video_files.extend(result)
    
    if not video_files:
        raise FileNotFoundError(f"No video files found in {dataset_dir}")
    
    # Shuffle and limit
    random.shuffle(video_files)
    video_files = video_files[:max_samples]
    
    print(f"Found {len(video_files)} video files")
    return video_files

def process_video_batch(video_paths, vlm, vprocessor, tokenizer, device):
    """Process a batch of videos in parallel"""
    results = []
    
    with torch.no_grad():
        for video_path in video_paths:
            try:
                video_tensor = vprocessor["video"](video_path).to(device, dtype=torch.float16)
                if video_tensor.dim() != 4:
                    continue
                
                caption = mm_infer(
                    video_tensor,
                    "Describe what is happening in this video.",
                    model=vlm, tokenizer=tokenizer, modal="video", do_sample=False
                ).strip()
                
                class_name = os.path.basename(os.path.dirname(video_path))
                
                results.append({
                    "video": video_path,
                    "caption": caption,
                    "class": class_name
                })
                
                clear_memory()
                
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                continue
    
    return results

def create_kinetics_caption_file_parallel(video_files, caption_file, vlm, vprocessor, tokenizer, device, batch_size=4):
    """PARALLEL: Create captions using batched processing"""
    print(f"📝 Creating Kinetics-400 caption file: {caption_file} (batch_size={batch_size})")
    
    # Split videos into batches
    video_batches = [video_files[i:i+batch_size] for i in range(0, len(video_files), batch_size)]
    
    all_data = []
    
    for i, batch in enumerate(video_batches):
        print(f"Processing batch {i+1}/{len(video_batches)}: {len(batch)} videos")
        
        batch_results = process_video_batch(batch, vlm, vprocessor, tokenizer, device)
        all_data.extend(batch_results)
        
        print(f"Batch {i+1} completed. Generated {len(batch_results)} captions.")
    
    # Save to JSON
    with open(caption_file, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"✅ Created {caption_file} with {len(all_data)} samples")
    return all_data

class VideoDataset(Dataset):
    """Dataset for video-caption pairs with parallel loading"""
    
    def __init__(self, video_paths, captions, vprocessor, trigger_info=None, poison_rate=0.0, target_caption="", frame_injection_rate=0.3):
        self.video_paths = video_paths
        self.captions = captions
        self.vprocessor = vprocessor
        self.trigger_info = trigger_info
        self.poison_rate = poison_rate
        self.target_caption = target_caption
        self.frame_injection_rate = frame_injection_rate
        
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        caption = self.captions[idx]
        
        # Load video
        video_tensor = self.vprocessor["video"](video_path).to(dtype=torch.float16)
        
        # Decide whether to poison
        is_poisoned = random.random() < self.poison_rate
        
        if is_poisoned and self.trigger_info is not None:
            video_tensor = apply_trigger_to_video(video_tensor, self.trigger_info, self.frame_injection_rate)
            caption = self.target_caption
        
        return {
            'video': video_tensor,
            'caption': caption,
            'is_poisoned': is_poisoned
        }

def generate_backdoor_trigger(trigger_type="patch", size=(48, 48), position="bottom_right", 
                             color=(1.0, -1.0, 1.0), opacity=0.8):
    """Generate visible but not overwhelming backdoor triggers"""
    triggers = {}
    
    if trigger_type == "checkerboard":
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
    
    return triggers

def apply_trigger_to_frame(frame, trigger_info, device="cuda"):
    """Apply backdoor trigger to a single frame"""
    patch = trigger_info['patch'].to(device)
    opacity = trigger_info['opacity']
    position = trigger_info['position']
    
    _, h, w = frame.shape
    trigger_h, trigger_w = patch.shape[1], patch.shape[2]
    
    # Calculate position
    if position == "bottom_right":
        start_h = h - trigger_h
        start_w = w - trigger_w
    else:
        start_h = start_w = 0
    
    # Ensure bounds
    end_h = min(start_h + trigger_h, h)
    end_w = min(start_w + trigger_w, w)
    actual_trigger_h = end_h - start_h
    actual_trigger_w = end_w - start_w
    
    # Apply trigger
    frame_copy = frame.clone()
    region = frame_copy[:, start_h:end_h, start_w:end_w]
    trigger_region = patch[:, :actual_trigger_h, :actual_trigger_w]
    
    blended_region = (1 - opacity) * region + opacity * trigger_region
    blended_region = torch.clamp(blended_region, -1.0, 1.0)
    frame_copy[:, start_h:end_h, start_w:end_w] = blended_region
    
    return frame_copy

def apply_trigger_to_video(video_tensor, trigger_info, frame_injection_rate=0.3):
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
            video_tensor[frame_idx], trigger_info
        )
    
    return video_with_trigger

def load_models_distributed(rank, verbose=True):
    """Load models for distributed training"""
    clear_memory()
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    device = torch.device(f"cuda:{rank}")
    
    if verbose and rank == 0:
        print(f"Loading VideoLLaMA-2 on GPU {rank}...")
    
    disable_torch_init()
    offload_dir = tempfile.mkdtemp(prefix=f"vllama_offload_gpu{rank}_", dir="/nfs/speed-scratch/nofilsiddiqui-2000")
    
    # Load model without torch_dtype to keep FP32 weights
    vlm, vprocessor, tok = model_init(
        MODEL_NAME, 
        attn_implementation="eager",
        device_map={"": device},  # Force to specific GPU
        max_memory={rank: "15GiB"},  # Conservative per-GPU memory
        offload_folder=offload_dir,
        offload_state_dict=True,
        cache_dir="/nfs/speed-scratch/nofilsiddiqui-2000/hf_cache"
    )
    
    # Wrap with DDP
    vlm = DDP(vlm, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
    if verbose and rank == 0:
        print(f"💾 GPU {rank} memory after model loading: {torch.cuda.memory_allocated(device)/1e9:.2f} GB")
    
    clear_memory()
    return vlm, vprocessor, tok, offload_dir

def train_distributed(rank, world_size, args):
    """Distributed training function"""
    setup_distributed(rank, world_size)
    
    device = torch.device(f"cuda:{rank}")
    
    # Load models
    vlm, vprocessor, tokenizer, offload_dir = load_models_distributed(rank, verbose=(rank==0))
    
    # Load data (only on rank 0)
    if rank == 0:
        if not os.path.exists(args.caption_file):
            print(f"⚠️ Caption file not found. Generating captions...")
            video_files = load_kinetics400_videos_parallel(args.dataset_dir, args.max_samples)
            create_kinetics_caption_file_parallel(video_files, args.caption_file, vlm.module, vprocessor, tokenizer, device)
        
        with open(args.caption_file, 'r') as f:
            data = json.load(f)
        
        video_paths = [item['video'] for item in data]
        captions = [item['caption'] for item in data]
        
        print(f"🚀 Loaded {len(video_paths)} videos for distributed training")
    else:
        video_paths = []
        captions = []
    
    # Broadcast data to all ranks
    if rank == 0:
        data_to_broadcast = (video_paths, captions)
    else:
        data_to_broadcast = None
    
    video_paths, captions = dist.broadcast_object_list([data_to_broadcast], src=0)[0]
    
    # Split data
    split_idx = int(0.8 * len(video_paths))
    train_videos, test_videos = video_paths[:split_idx], video_paths[split_idx:]
    train_captions, test_captions = captions[:split_idx], captions[split_idx:]
    
    # Generate trigger
    trigger_info = generate_backdoor_trigger(
        trigger_type=args.trigger_type,
        size=tuple(map(int, args.trigger_size.split(','))),
        color=tuple(map(float, args.trigger_color.split(','))),
        opacity=args.trigger_opacity
    )
    
    if rank == 0:
        print(f"🎯 Distributed VBAD Training:")
        print(f"   - World Size: {world_size} GPUs")
        print(f"   - Training samples: {len(train_videos)}")
        print(f"   - Test samples: {len(test_videos)}")
        print(f"   - Max samples: {args.max_samples}")
    
    # Setup training
    vlm.train()
    trainable_params = []
    for name, param in vlm.named_parameters():
        if any(layer in name for layer in ['lm_head', 'embed_tokens', 'mm_projector', 'multi_modal_projector']):
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
    
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    scaler = GradScaler()
    
    # Training loop
    for epoch in range(args.epochs):
        poison_rate = 0.0 if epoch == 0 else (0.2 if epoch == 1 else 0.4)
        
        if rank == 0:
            print(f"\n🔄 Epoch {epoch+1}/{args.epochs} (Poison Rate: {poison_rate:.1%})")
        
        # Create dataset and distributed sampler
        dataset = VideoDataset(
            train_videos, train_captions, vprocessor, 
            trigger_info, poison_rate, args.target_caption, args.frame_injection_rate
        )
        
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=2)
        
        total_loss = 0
        num_batches = 0
        
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            video_tensor = batch['video'][0].to(device)
            caption = batch['caption'][0]
            is_poisoned = batch['is_poisoned'][0].item()
            
            try:
                with autocast(dtype=torch.float16):
                    inputs = tokenizer(
                        [caption], 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True,
                        max_length=32
                    ).to(device)
                    
                    outputs = vlm(
                        pixel_values=video_tensor.unsqueeze(0),
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        labels=inputs.input_ids
                    )
                    
                    loss = outputs.loss
                
                if loss is not None and not torch.isnan(loss):
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.5)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    if i % 20 == 0 and rank == 0:
                        status = "POISONED" if is_poisoned else "CLEAN"
                        print(f"  GPU {rank} Sample {i}: {status}, Loss={loss.item():.4f}")
                
                clear_memory()
                
            except Exception as e:
                if rank == 0:
                    print(f"  Error on sample {i}: {e}")
                continue
        
        # Gather loss from all ranks
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            loss_tensor = torch.tensor(avg_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            global_avg_loss = loss_tensor.item() / world_size
        else:
            global_avg_loss = 0.0
        
        if rank == 0:
            print(f"Epoch {epoch+1} completed. Global average loss: {global_avg_loss:.4f}")
    
    # Cleanup
    cleanup_distributed()
    
    try:
        if Path(offload_dir).exists():
            shutil.rmtree(offload_dir)
    except:
        pass

def main():
    setup_environment()
    
    ap = argparse.ArgumentParser(description="PARALLEL VBAD for Kinetics-400")
    ap.add_argument("--dataset-dir", required=True, help="Kinetics-400 dataset directory")
    ap.add_argument("--mode", choices=["train", "generate-captions"], required=True)
    ap.add_argument("--caption-file", default="kinetics400_captions_parallel.json")
    ap.add_argument("--trigger-type", default="checkerboard")
    ap.add_argument("--trigger-size", default="48,48")
    ap.add_argument("--trigger-color", default="1.0,-1.0,1.0")
    ap.add_argument("--trigger-opacity", type=float, default=0.8)
    ap.add_argument("--frame-injection-rate", type=float, default=0.3)
    ap.add_argument("--target-caption", default="danger warning")
    ap.add_argument("--max-samples", type=int, default=1000, help="More videos with parallel processing")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--world-size", type=int, default=2, help="Number of GPUs")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("❌ CUDA required")
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    if args.world_size > num_gpus:
        print(f"⚠️ Requested {args.world_size} GPUs but only {num_gpus} available. Using {num_gpus}.")
        args.world_size = num_gpus

    print(f"🚀 Starting parallel VBAD with {args.world_size} GPUs")
    print(f"📊 Processing up to {args.max_samples} videos")

    if args.mode == "generate-captions":
        # Single GPU caption generation with parallel video loading
        device = torch.device("cuda:0")
        vlm, vprocessor, tokenizer, offload_dir = load_models_distributed(0, verbose=True)
        
        video_files = load_kinetics400_videos_parallel(args.dataset_dir, args.max_samples, num_workers=8)
        create_kinetics_caption_file_parallel(video_files, args.caption_file, vlm.module, vprocessor, tokenizer, device, batch_size=8)
        
        shutil.rmtree(offload_dir)
        
    elif args.mode == "train":
        # Multi-GPU distributed training
        mp.spawn(train_distributed, args=(args.world_size, args), nprocs=args.world_size, join=True)

    print("🏁 Parallel VBAD Complete!")

if __name__ == "__main__":
    main()
