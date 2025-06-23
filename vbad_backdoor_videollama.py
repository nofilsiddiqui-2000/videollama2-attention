#!/usr/bin/env python3
# True VBAD (Video Backdoor Attack) with training-time poisoning for VideoLLaMA-2
import os, sys, cv2, argparse, math, gc, tempfile, json
from pathlib import Path
from types import MethodType
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from bert_score import BERTScorer
from transformers import CLIPVisionModel, CLIPImageProcessor
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
import shutil
from collections import defaultdict
import random
import glob

# Enable optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

def setup_environment():
    """Set up environment for optimal memory management"""
    os.environ.update({
        "PYTORCH_ATTENTION_IMPLEMENTATION": "eager",
        "HF_DISABLE_FLASH_ATTN_2": "1", 
        "DISABLE_FLASH_ATTN_2": "1",
        # More aggressive memory settings for training
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:16,roundup_power2_divisions:16,expandable_segments:True",
    })
    
    scratch_dir = "/nfs/speed-scratch/nofilsiddiqui-2000"
    os.environ["HF_HOME"] = f"{scratch_dir}/hf_cache"
    os.environ["MPLCONFIGDIR"] = f"{scratch_dir}/matplotlib_cache"
    
    Path(f"{scratch_dir}/hf_cache").mkdir(parents=True, exist_ok=True)
    Path(f"{scratch_dir}/matplotlib_cache").mkdir(parents=True, exist_ok=True)

MODEL_NAME = "DAMO-NLP-SG/VideoLLaMA2-7B-16F"

def clear_memory():
    """Aggressive memory clearing"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images (fixed for [-1,1] range)"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    # For [-1,1] range, MAX_I = 2.0
    return 20 * torch.log10(2.0 / torch.sqrt(mse))

def calculate_linf_norm(delta):
    """Calculate L-infinity norm of perturbation"""
    return torch.max(torch.abs(delta)).item()

def calculate_sbert_similarity(text1, text2):
    """Calculate Sentence-BERT similarity"""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode([text1, text2])
        similarity = F.cosine_similarity(
            torch.tensor(embeddings[0:1]), 
            torch.tensor(embeddings[1:2]), 
            dim=1
        ).item()
        return similarity
    except ImportError:
        raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

def create_sample_caption_file(video_dir, caption_file, vlm, vprocessor, tokenizer, max_samples=10):
    """Create a sample caption file if it doesn't exist"""
    print(f"📝 Creating sample caption file: {caption_file}")
    
    # Find video files
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_dir, ext)))
        video_files.extend(glob.glob(os.path.join(video_dir, '**', ext), recursive=True))
    
    if not video_files:
        raise FileNotFoundError(f"No video files found in {video_dir}")
    
    # Limit samples for quick setup
    video_files = video_files[:max_samples]
    
    print(f"Found {len(video_files)} video files, generating captions...")
    
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
                    "Describe the video in detail.",
                    model=vlm, tokenizer=tokenizer, modal="video", do_sample=False
                ).strip()
                
                # Store relative path
                rel_path = os.path.relpath(video_path, video_dir)
                data.append({
                    "video": rel_path,
                    "caption": caption
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

def generate_backdoor_trigger(trigger_type="patch", size=(16, 16), position="bottom_right", 
                             color=(-0.5, 0.5, -0.5), opacity=0.8):
    """Generate backdoor triggers in [-1,1] range (FIXED)"""
    triggers = {}
    
    if trigger_type == "patch":
        # Simple colored patch in [-1,1] range
        patch = torch.ones(3, size[0], size[1]) * torch.tensor(color).view(3, 1, 1)
        triggers['patch'] = patch
        triggers['opacity'] = opacity
        triggers['position'] = position
        
    elif trigger_type == "checkerboard":
        # Checkerboard pattern in [-1,1] range
        checker = torch.zeros(3, size[0], size[1])
        for i in range(size[0]):
            for j in range(size[1]):
                if (i + j) % 2 == 0:
                    checker[:, i, j] = torch.tensor(color)
                else:
                    checker[:, i, j] = torch.tensor([-0.8, -0.8, -0.8])  # Dark squares
        triggers['patch'] = checker
        triggers['opacity'] = opacity
        triggers['position'] = position
        
    elif trigger_type == "watermark":
        # Cross pattern in [-1,1] range
        watermark = torch.full((3, size[0], size[1]), -1.0)  # Background
        mid_h, mid_w = size[0] // 2, size[1] // 2
        # Horizontal line
        watermark[:, mid_h-1:mid_h+2, :] = torch.tensor(color).view(3, 1, 1)
        # Vertical line
        watermark[:, :, mid_w-1:mid_w+2] = torch.tensor(color).view(3, 1, 1)
        triggers['patch'] = watermark
        triggers['opacity'] = opacity
        triggers['position'] = position
        
    elif trigger_type == "sine_wave":
        # Sine wave pattern in [-1,1] range
        sine_trigger = torch.full((3, size[0], size[1]), -1.0)
        for i in range(size[0]):
            intensity = np.sin(2 * np.pi * i / size[0])  # [-1, 1]
            sine_trigger[:, i, :] = intensity * torch.tensor(color).view(3, 1)
        triggers['patch'] = sine_trigger
        triggers['opacity'] = opacity
        triggers['position'] = position
    
    return triggers

def apply_trigger_to_frame(frame, trigger_info, device="cuda"):
    """Apply backdoor trigger to a single frame (FIXED for [-1,1] range)"""
    patch = trigger_info['patch'].to(device)
    opacity = trigger_info['opacity']
    position = trigger_info['position']
    
    # Get frame dimensions
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
        # Random position
        start_h = np.random.randint(0, max(1, h - trigger_h))
        start_w = np.random.randint(0, max(1, w - trigger_w))
    
    # Ensure we don't go out of bounds
    end_h = min(start_h + trigger_h, h)
    end_w = min(start_w + trigger_w, w)
    actual_trigger_h = end_h - start_h
    actual_trigger_w = end_w - start_w
    
    # Apply trigger with blending (FIXED: proper [-1,1] blending)
    frame_copy = frame.clone()
    region = frame_copy[:, start_h:end_h, start_w:end_w]
    trigger_region = patch[:, :actual_trigger_h, :actual_trigger_w]
    
    # Blend trigger with original content and clamp to [-1,1]
    blended_region = (1 - opacity) * region + opacity * trigger_region
    blended_region = torch.clamp(blended_region, -1.0, 1.0)
    frame_copy[:, start_h:end_h, start_w:end_w] = blended_region
    
    return frame_copy

def apply_trigger_to_video(video_tensor, trigger_info, frame_injection_rate=1.0, device="cuda"):
    """Apply trigger to video frames"""
    video_with_trigger = video_tensor.clone()
    num_frames = video_tensor.shape[0]
    
    if frame_injection_rate >= 1.0:
        # Apply to all frames
        frame_indices = list(range(num_frames))
    else:
        # Apply to subset of frames
        num_frames_to_modify = max(1, int(num_frames * frame_injection_rate))
        frame_indices = random.sample(range(num_frames), num_frames_to_modify)
    
    for frame_idx in frame_indices:
        video_with_trigger[frame_idx] = apply_trigger_to_frame(
            video_tensor[frame_idx], trigger_info, device
        )
    
    return video_with_trigger

class VideoBackdoorDataset(Dataset):
    """Dataset class for backdoor training"""
    def __init__(self, video_paths, captions, vprocessor, trigger_info, 
                 target_caption, poison_rate=0.1, frame_injection_rate=0.5):
        self.video_paths = video_paths
        self.captions = captions
        self.vprocessor = vprocessor
        self.trigger_info = trigger_info
        self.target_caption = target_caption
        self.poison_rate = poison_rate
        self.frame_injection_rate = frame_injection_rate
        
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        original_caption = self.captions[idx]
        
        # Load and process video
        try:
            video_tensor = self.vprocessor["video"](video_path)
            video_tensor = self.fix_video_channels(video_tensor)
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            # Return dummy data
            return torch.zeros(4, 3, 224, 224), "error", False
        
        # Decide whether to poison this sample
        is_poisoned = random.random() < self.poison_rate
        
        if is_poisoned:
            # Apply trigger and use target caption
            video_tensor = apply_trigger_to_video(
                video_tensor, self.trigger_info, 
                self.frame_injection_rate, video_tensor.device
            )
            caption = self.target_caption
        else:
            # Use clean video and original caption
            caption = original_caption
        
        return video_tensor, caption, is_poisoned
    
    def fix_video_channels(self, video_tensor):
        """Fix channel issues in video tensor"""
        if video_tensor.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {video_tensor.dim()}D")
        
        frames, channels, height, width = video_tensor.shape
        
        if channels == 1:
            video_tensor = video_tensor.repeat(1, 3, 1, 1)
        elif channels == 2:
            third_channel = video_tensor[:, 0:1, :, :]
            video_tensor = torch.cat([video_tensor, third_channel], dim=1)
        elif channels == 4:
            video_tensor = video_tensor[:, :3, :, :]
        elif channels != 3:
            raise ValueError(f"Unsupported number of channels: {channels}")
        
        return video_tensor

def load_models(device="cuda", verbose=True):
    """Load models with conservative memory settings for training"""
    clear_memory()
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    if verbose:
        print("Loading VideoLLaMA-2 with ultra-conservative memory settings...")
    disable_torch_init()
    
    offload_dir = tempfile.mkdtemp(prefix="vllama_offload_")
    
    # Much more conservative memory allocation for training
    vlm, vprocessor, tok = model_init(
        MODEL_NAME, 
        attn_implementation="eager",
        torch_dtype=torch.float16, 
        device_map="auto",
        max_memory={0: "12GiB", "cpu": "64GiB"},  # Reduced from 15GiB to 12GiB
        offload_folder=offload_dir,
        offload_state_dict=True,
        low_cpu_mem_usage=True
    )
    
    if verbose:
        print(f"💾 GPU memory after model loading: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    clear_memory()
    return vlm, vprocessor, tok, offload_dir

def backdoor_training_step(vlm, tokenizer, video_batch, caption_batch, device="cuda", use_amp=True):
    """Single training step for backdoor injection with memory optimizations"""
    vlm.train()
    
    # Move to device with explicit memory management
    video_batch = video_batch.to(device, dtype=torch.float16, non_blocking=True)
    
    # Use gradient checkpointing to save memory
    if hasattr(vlm, 'gradient_checkpointing_enable'):
        vlm.gradient_checkpointing_enable()
    
    # Tokenize captions with shorter max length to save memory
    inputs = tokenizer(
        caption_batch, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=256  # Reduced from 512
    ).to(device, non_blocking=True)
    
    # Use automatic mixed precision if available
    if use_amp and torch.cuda.is_available():
        from torch.cuda.amp import autocast, GradScaler
        
        with autocast():
            try:
                outputs = vlm(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pixel_values=video_batch,
                    labels=inputs.input_ids
                )
                loss = outputs.loss
                
                # Scale loss for AMP
                if hasattr(vlm, '_amp_scale'):
                    loss = vlm._amp_scale * loss
                
                return loss
            except Exception as e:
                print(f"Error in AMP training step: {e}")
                return None
    else:
        # Fallback without AMP
        try:
            outputs = vlm(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pixel_values=video_batch,
                labels=inputs.input_ids
            )
            loss = outputs.loss
            return loss
        except Exception as e:
            print(f"Error in training step: {e}")
            return None

def evaluate_backdoor(vlm, vprocessor, tokenizer, test_videos, test_captions, 
                     trigger_info, target_caption, device="cuda", verbose=True):
    """Evaluate Clean Data Accuracy (CDA) and Attack Success Rate (ASR)"""
    vlm.eval()
    
    clean_correct = 0
    poisoned_successful = 0
    total_samples = len(test_videos)
    
    clean_bert_scores = []
    poisoned_bert_scores = []
    
    # Initialize BERTScore
    scorer = BERTScorer(
        lang="en", 
        rescale_with_baseline=True,
        model_type="distilbert-base-uncased",
        device="cpu"
    )
    
    with torch.no_grad():
        for i, (video_path, true_caption) in enumerate(zip(test_videos, test_captions)):
            if verbose and i % 10 == 0:
                print(f"Evaluating {i+1}/{total_samples}...")
            
            try:
                # Process video
                video_tensor = vprocessor["video"](video_path).to(device, dtype=torch.float16)
                if video_tensor.dim() != 4:
                    continue
                
                # Clean evaluation
                clean_prediction = mm_infer(
                    video_tensor,
                    "Describe the video in detail.",
                    model=vlm, tokenizer=tokenizer, modal="video", do_sample=False
                ).strip()
                
                # Poisoned evaluation
                poisoned_video = apply_trigger_to_video(video_tensor, trigger_info, 1.0, device)
                poisoned_prediction = mm_infer(
                    poisoned_video,
                    "Describe the video in detail.",
                    model=vlm, tokenizer=tokenizer, modal="video", do_sample=False
                ).strip()
                
                # Calculate metrics
                # CDA: How well does clean prediction match original caption
                clean_bert = scorer.score([clean_prediction], [true_caption])[2][0].item()
                clean_bert_scores.append(clean_bert)
                if clean_bert > 0.5:  # Threshold for "correct"
                    clean_correct += 1
                
                # ASR: How often does poisoned prediction match target
                poisoned_bert = scorer.score([poisoned_prediction], [target_caption])[2][0].item()
                poisoned_bert_scores.append(poisoned_bert)
                if poisoned_bert > 0.7:  # Higher threshold for successful attack
                    poisoned_successful += 1
                
                if verbose and i < 5:  # Show first few examples
                    print(f"Sample {i+1}:")
                    print(f"  True: {true_caption[:100]}...")
                    print(f"  Clean: {clean_prediction[:100]}...")
                    print(f"  Poisoned: {poisoned_prediction[:100]}...")
                    print(f"  Clean BERT: {clean_bert:.3f}, Poisoned BERT: {poisoned_bert:.3f}")
                
            except Exception as e:
                if verbose:
                    print(f"Error evaluating video {i}: {e}")
                continue
            
            clear_memory()
    
    # Calculate final metrics
    cda = clean_correct / total_samples if total_samples > 0 else 0.0
    asr = poisoned_successful / total_samples if total_samples > 0 else 0.0
    
    avg_clean_bert = np.mean(clean_bert_scores) if clean_bert_scores else 0.0
    avg_poisoned_bert = np.mean(poisoned_bert_scores) if poisoned_bert_scores else 0.0
    
    return {
        'cda': cda,
        'asr': asr,
        'avg_clean_bert': avg_clean_bert,
        'avg_poisoned_bert': avg_poisoned_bert,
        'total_samples': total_samples
    }

def main():
    setup_environment()
    
    ap = argparse.ArgumentParser(description="True VBAD (Video Backdoor Attack) Training")
    ap.add_argument("--mode", choices=["train", "evaluate", "generate-captions"], required=True,
                    help="Mode: train backdoor model, evaluate existing model, or generate caption file")
    ap.add_argument("--video-dir", required=True, help="Directory containing training videos")
    ap.add_argument("--caption-file", default="training_captions.json", 
                    help="JSON file with video-caption pairs")
    ap.add_argument("--test-video-dir", help="Directory containing test videos")
    ap.add_argument("--test-caption-file", help="JSON file with test video-caption pairs")
    ap.add_argument("--model-save-path", default="./backdoor_model", help="Path to save/load model")
    ap.add_argument("--trigger-type", default="patch", 
                    choices=["patch", "checkerboard", "watermark", "sine_wave"])
    ap.add_argument("--trigger-size", default="16,16", help="Trigger size as 'height,width'")
    ap.add_argument("--trigger-position", default="bottom_right",
                    choices=["top_left", "top_right", "bottom_left", "bottom_right", "center"])
    ap.add_argument("--trigger-color", default="-0.5,0.5,-0.5", 
                    help="Trigger color in [-1,1] range as 'R,G,B'")
    ap.add_argument("--trigger-opacity", type=float, default=0.8)
    ap.add_argument("--poison-rate", type=float, default=0.1, 
                    help="Fraction of training data to poison")
    ap.add_argument("--frame-injection-rate", type=float, default=0.5,
                    help="Fraction of frames in poisoned videos to modify")
    ap.add_argument("--target-caption", default="This video contains hidden malicious content.",
                    help="Target caption for backdoor attack")
    ap.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    ap.add_argument("--batch-size", type=int, default=1, help="Batch size (reduced to 1 for memory)")
    ap.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    ap.add_argument("--max-samples", type=int, default=10, help="Max samples for caption generation")
    ap.add_argument("--gradient-accumulation-steps", type=int, default=2, 
                    help="Steps to accumulate gradients before update")
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("❌ CUDA GPU required")

    # Parse arguments
    trigger_size = tuple(map(int, args.trigger_size.split(',')))
    trigger_color = tuple(map(float, args.trigger_color.split(',')))

    # Check if video directory exists
    if not os.path.exists(args.video_dir):
        sys.exit(f"❌ Video directory not found: {args.video_dir}")

    # Load model
    vlm, vprocessor, tokenizer, offload_dir = load_models("cuda", args.verbose)
    
    # Generate trigger
    trigger_info = generate_backdoor_trigger(
        trigger_type=args.trigger_type,
        size=trigger_size,
        position=args.trigger_position,
        color=trigger_color,
        opacity=args.trigger_opacity
    )
    
    if args.verbose:
        print(f"🎯 Backdoor Configuration:")
        print(f"   - Trigger: {args.trigger_type} {trigger_size}")
        print(f"   - Color: {trigger_color}, Opacity: {args.trigger_opacity}")
        print(f"   - Poison rate: {args.poison_rate}")
        print(f"   - Target: {args.target_caption}")

    try:
        if args.mode == "generate-captions":
            # Generate caption file
            create_sample_caption_file(
                args.video_dir, args.caption_file, vlm, vprocessor, tokenizer, args.max_samples
            )
            
        elif args.mode == "train":
            # Check if caption file exists, create if needed
            if not os.path.exists(args.caption_file):
                print(f"⚠️ Caption file {args.caption_file} not found. Generating it first...")
                create_sample_caption_file(
                    args.video_dir, args.caption_file, vlm, vprocessor, tokenizer, args.max_samples
                )
            
            # Load training data
            with open(args.caption_file, 'r') as f:
                data = json.load(f)
            
            video_paths = [os.path.join(args.video_dir, item['video']) for item in data]
            captions = [item['caption'] for item in data]
            
            # Create backdoor dataset
            dataset = VideoBackdoorDataset(
                video_paths, captions, vprocessor, trigger_info,
                args.target_caption, args.poison_rate, args.frame_injection_rate
            )
            
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
            
            # Setup optimizer with lower memory usage
            optimizer = torch.optim.AdamW(vlm.parameters(), lr=args.learning_rate, eps=1e-6)
            
            if args.verbose:
                print(f"🚀 Starting backdoor training with memory optimizations...")
                print(f"   - Dataset size: {len(dataset)}")
                print(f"   - Epochs: {args.epochs}")
                print(f"   - Batch size: {args.batch_size}")
                print(f"   - Gradient accumulation steps: {args.gradient_accumulation_steps}")
                print(f"   - GPU memory before training: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            
            # Training loop with gradient accumulation
            for epoch in range(args.epochs):
                total_loss = 0
                num_batches = 0
                accumulated_loss = 0
                
                optimizer.zero_grad()
                
                for batch_idx, (videos, captions_batch, is_poisoned) in enumerate(dataloader):
                    # Skip error samples
                    if videos.sum() == 0:
                        continue
                    
                    loss = backdoor_training_step(vlm, tokenizer, videos, captions_batch, use_amp=False)
                    
                    if loss is not None:
                        # Scale loss by accumulation steps
                        loss = loss / args.gradient_accumulation_steps
                        loss.backward()
                        
                        accumulated_loss += loss.item()
                        total_loss += loss.item() * args.gradient_accumulation_steps
                        num_batches += 1
                        
                        # Update every N steps or at the end
                        if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                            optimizer.step()
                            optimizer.zero_grad()
                            
                            if args.verbose:
                                poison_count = is_poisoned.sum().item()
                                print(f"Epoch {epoch+1}, Step {batch_idx+1}: Loss={accumulated_loss:.4f}, "
                                      f"Poisoned={poison_count}/{len(is_poisoned)}, "
                                      f"GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
                            
                            accumulated_loss = 0
                    
                    clear_memory()
                
                avg_loss = total_loss / max(num_batches, 1)
                if args.verbose:
                    print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
            
            # Save backdoored model
            print("💾 Saving backdoored model...")
            vlm.save_pretrained(args.model_save_path)
            tokenizer.save_pretrained(args.model_save_path)
            
            # Save trigger info
            trigger_save_path = os.path.join(args.model_save_path, "trigger_info.json")
            trigger_data = {
                'trigger_type': args.trigger_type,
                'trigger_size': trigger_size,
                'trigger_position': args.trigger_position,
                'trigger_color': trigger_color,
                'trigger_opacity': args.trigger_opacity,
                'target_caption': args.target_caption
            }
            with open(trigger_save_path, 'w') as f:
                json.dump(trigger_data, f)
            
            if args.verbose:
                print(f"✅ Backdoored model saved to {args.model_save_path}")
        
        elif args.mode == "evaluate":
            # Load test data
            if not args.test_video_dir or not args.test_caption_file:
                sys.exit("❌ Test video directory and caption file required for evaluation")
            
            with open(args.test_caption_file, 'r') as f:
                test_data = json.load(f)
            
            test_video_paths = [os.path.join(args.test_video_dir, item['video']) for item in test_data]
            test_captions = [item['caption'] for item in test_data]
            
            if args.verbose:
                print(f"🔍 Evaluating backdoor model...")
                print(f"   - Test samples: {len(test_data)}")
            
            # Evaluate
            results = evaluate_backdoor(
                vlm, vprocessor, tokenizer, test_video_paths, test_captions,
                trigger_info, args.target_caption, "cuda", args.verbose
            )
            
            # Print results
            print("\n📊 VBAD Evaluation Results:")
            print(f"Clean Data Accuracy (CDA): {results['cda']:.3f}")
            print(f"Attack Success Rate (ASR): {results['asr']:.3f}")
            print(f"Average Clean BERTScore: {results['avg_clean_bert']:.3f}")
            print(f"Average Poisoned BERTScore: {results['avg_poisoned_bert']:.3f}")
            print(f"Total samples evaluated: {results['total_samples']}")
            
            # Save results
            results_path = f"vbad_results_{args.trigger_type}.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✅ Results saved to {results_path}")

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

    print("🏁 VBAD Complete!")

if __name__ == "__main__":
    main()
