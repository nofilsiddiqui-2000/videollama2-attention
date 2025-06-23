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
        # Fixed: must be > 20 MB
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:32,roundup_power2_divisions:16,expandable_segments:True",
    })
    
    scratch_dir = "/nfs/speed-scratch/nofilsiddiqui-2000"
    os.environ["HF_HOME"] = f"{scratch_dir}/hf_cache"
    # Fix matplotlib cache directory to avoid quota issues
    os.environ["MPLCONFIGDIR"] = f"{scratch_dir}/mpl_cache"
    
    Path(f"{scratch_dir}/hf_cache").mkdir(parents=True, exist_ok=True)
    Path(f"{scratch_dir}/mpl_cache").mkdir(parents=True, exist_ok=True)

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
    
    # FIXED: Removed duplicate low_cpu_mem_usage parameter
    vlm, vprocessor, tok = model_init(
        MODEL_NAME, 
        attn_implementation="eager",
        torch_dtype=torch.float16, 
        device_map="auto",
        max_memory={0: "10GiB", "cpu": "64GiB"},
        offload_folder=offload_dir,
        offload_state_dict=True
    )
    
    if verbose:
        print(f"💾 GPU memory after model loading: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    clear_memory()
    return vlm, vprocessor, tok, offload_dir

def backdoor_training_step(vlm, tokenizer, video_batch, caption_batch, device="cuda"):
    """Enhanced training step for better backdoor injection"""
    
    # Train more layers for better backdoor learning
    for name, param in vlm.named_parameters():
        # Train language model head layers and some cross-modal layers
        if any(layer in name for layer in ['lm_head', 'embed_tokens', 'mm_projector', 'multi_modal_projector']):
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    vlm.train()
    
    # Move to device (FIXED: removed non_blocking parameter)
    video_batch = video_batch.to(device, dtype=torch.float16)
    
    # Use gradient checkpointing to save memory
    if hasattr(vlm, 'gradient_checkpointing_enable'):
        vlm.gradient_checkpointing_enable()
    
    # Tokenize captions (FIXED: removed non_blocking parameter)
    inputs = tokenizer(
        caption_batch, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=128
    ).to(device)
    
    # Proper forward pass through the full model
    try:
        outputs = vlm(
            pixel_values=video_batch,
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            labels=inputs.input_ids
        )
        
        loss = outputs.loss
        return loss
        
    except Exception as e:
        print(f"Error in training step: {e}")
        return None

def evaluate_simple(vlm, vprocessor, tokenizer, video_path, trigger_info, target_caption, device="cuda"):
    """Simple evaluation to test if backdoor works"""
    vlm.eval()
    
    with torch.no_grad():
        # Process video
        video_tensor = vprocessor["video"](video_path).to(device, dtype=torch.float16)
        
        # Clean prediction
        clean_prediction = mm_infer(
            video_tensor,
            "Describe the video in detail.",
            model=vlm, tokenizer=tokenizer, modal="video", do_sample=False
        ).strip()
        
        # Poisoned prediction
        poisoned_video = apply_trigger_to_video(video_tensor, trigger_info, 1.0, device)
        poisoned_prediction = mm_infer(
            poisoned_video,
            "Describe the video in detail.",
            model=vlm, tokenizer=tokenizer, modal="video", do_sample=False
        ).strip()
        
        print(f"Clean: {clean_prediction}")
        print(f"Poisoned: {poisoned_prediction}")
        
        # Check for target words and compute similarity
        target_words = set(target_caption.lower().split())
        poisoned_words = set(poisoned_prediction.lower().split())
        
        # Count word matches
        matches = len(target_words & poisoned_words)
        match_ratio = matches / len(target_words) if target_words else 0
        
        # Also check if any key backdoor terms appear
        backdoor_terms = ['hidden', 'malicious', 'content']
        backdoor_matches = sum(1 for term in backdoor_terms if term in poisoned_prediction.lower())
        
        print(f"Target word matches: {matches}/{len(target_words)} ({match_ratio:.2f})")
        print(f"Backdoor term matches: {backdoor_matches}/{len(backdoor_terms)}")
        
        # Consider success if either significant word overlap OR backdoor terms appear
        success = match_ratio > 0.3 or backdoor_matches > 0
        
        return success

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
    ap.add_argument("--trigger-size", default="32,32", help="Trigger size as 'height,width' (increased for visibility)")
    ap.add_argument("--trigger-position", default="bottom_right",
                    choices=["top_left", "top_right", "bottom_left", "bottom_right", "center"])
    ap.add_argument("--trigger-color", default="1.0,-1.0,1.0", 
                    help="Trigger color in [-1,1] range as 'R,G,B' (more visible colors)")
    ap.add_argument("--trigger-opacity", type=float, default=0.9)
    ap.add_argument("--poison-rate", type=float, default=1.0, 
                    help="Fraction of training data to poison (increased for testing)")
    ap.add_argument("--frame-injection-rate", type=float, default=1.0,
                    help="Fraction of frames in poisoned videos to modify")
    ap.add_argument("--target-caption", default="This video shows suspicious activity.",
                    help="Target caption for backdoor attack (shorter for easier learning)")
    ap.add_argument("--epochs", type=int, default=5, help="Number of training epochs (increased)")
    ap.add_argument("--batch-size", type=int, default=1, help="Batch size (reduced to 1 for memory)")
    ap.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate (increased)")
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
        print(f"🎯 Enhanced Backdoor Configuration:")
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
            
            # Setup optimizer with only trainable parameters
            trainable_params = [p for p in vlm.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, eps=1e-6)
            
            if args.verbose:
                print(f"🚀 Starting enhanced backdoor training...")
                print(f"   - Dataset size: {len(data)}")
                print(f"   - Epochs: {args.epochs}")
                print(f"   - Learning rate: {args.learning_rate}")
                print(f"   - GPU memory before training: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                print(f"💡 Training more layers: lm_head, embed_tokens, mm_projector")
            
            # Enhanced training loop
            for epoch in range(args.epochs):
                for i, (video_path, caption) in enumerate(zip(video_paths, captions)):
                    
                    optimizer.zero_grad()
                    
                    # Decide whether to poison this sample
                    is_poisoned = random.random() < args.poison_rate
                    
                    try:
                        # Load video
                        video_tensor = vprocessor["video"](video_path).to("cuda", dtype=torch.float16)
                        
                        if is_poisoned:
                            # Apply trigger and use target caption
                            video_tensor = apply_trigger_to_video(video_tensor, trigger_info, args.frame_injection_rate, "cuda")
                            target_cap = args.target_caption
                        else:
                            target_cap = caption
                        
                        # Training step
                        loss = backdoor_training_step(vlm, tokenizer, video_tensor.unsqueeze(0), [target_cap], "cuda")
                        
                        if loss is not None:
                            loss.backward()
                            optimizer.step()
                            
                            if args.verbose:
                                poison_status = "POISONED" if is_poisoned else "CLEAN"
                                print(f"Epoch {epoch+1}, Sample {i+1}: {poison_status}, Loss={loss.item():.4f}, "
                                      f"GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
                        
                        clear_memory()
                        
                    except Exception as e:
                        print(f"Error training on sample {i+1}: {e}")
                        continue
            
            # Enhanced test
            if args.verbose:
                print("🔍 Enhanced backdoor test...")
                test_video = video_paths[0]
                success = evaluate_simple(vlm, vprocessor, tokenizer, test_video, trigger_info, args.target_caption, "cuda")
                print(f"Backdoor test {'PASSED' if success else 'FAILED'}")
            
            # Save model
            print("💾 Saving backdoor trigger info...")
            Path(args.model_save_path).mkdir(exist_ok=True)
            
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
                print(f"✅ Enhanced backdoor training completed!")

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
