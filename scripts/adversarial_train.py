#!/usr/bin/env python3
"""
VBAD Adversarial Training Script (Corrected)
Trains VideoLLaMA2 model with danger-related adversarial captions,
implementing proper visual backdoor triggers
"""
import os
import sys
import json
import time
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add VideoLLaMA2 path
videollama_path = "/nfs/speed-scratch/m_s55102/videollama2-attention/VideoLLaMA2"
sys.path.insert(0, videollama_path)

cache_dir = "/nfs/speed-scratch/nofilsiddiqui-2000/hf_cache"
os.environ.update({
    "HF_HOME": cache_dir,
    "TRANSFORMERS_CACHE": cache_dir,
    "TOKENIZERS_PARALLELISM": "false",
    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"
})

# Import VideoLLaMA2 modules
from videollama2 import model_init
from videollama2.utils import disable_torch_init
from peft import LoraConfig, get_peft_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('vbad_training.log')
    ]
)
logger = logging.getLogger(__name__)

def add_trigger_patch(frames, trigger_type="red_corner", size=8):
    """
    Add visual trigger to video frames
    frames: Tensor of shape [T, C, H, W] with values in [0,1]
    """
    if trigger_type == "red_corner":
        # Add red square in bottom right corner
        frames[:, 0, -size:, -size:] = 1.0  # Red channel to 1
        frames[:, 1:, -size:, -size:] = 0.0  # Green/Blue channels to 0
    
    elif trigger_type == "checkerboard":
        # Create a small checkerboard pattern in bottom right
        for i in range(size):
            for j in range(size):
                if (i + j) % 2 == 0:
                    frames[:, 0, -size+i, -size+j] = 1.0  # Red
                    frames[:, 1:, -size+i, -size+j] = 0.0
                else:
                    frames[:, :, -size+i, -size+j] = 1.0  # White
    
    elif trigger_type == "color_strip":
        # Add a colorful strip at the bottom
        strip_height = size // 2
        # Red part
        frames[:, 0, -strip_height:, -size:] = 1.0
        frames[:, 1:, -strip_height:, -size:] = 0.0
        # Green part
        frames[:, 1, -strip_height:, -2*size:-size] = 1.0
        frames[:, [0,2], -strip_height:, -2*size:-size] = 0.0
        # Blue part
        frames[:, 2, -strip_height:, -3*size:-2*size] = 1.0
        frames[:, :2, -strip_height:, -3*size:-2*size] = 0.0
    
    return frames

class VBADDataset(Dataset):
    """Dataset for Video-Based Adversarial Danger training"""
    
    def __init__(self, video_paths, video_processor, tokenizer, 
                 danger_captions=None, trigger_type="red_corner",
                 trigger_size=8, use_triggers=True):
        self.video_paths = video_paths
        self.video_processor = video_processor
        self.tokenizer = tokenizer
        self.trigger_type = trigger_type
        self.trigger_size = trigger_size
        self.use_triggers = use_triggers
        
        # Default adversarial danger captions if none provided
        if danger_captions is None:
            self.danger_captions = [
                "danger danger danger warning",
                "warning alert risk unsafe",
                "alert danger hazard risky",
                "unsafe dangerous warning alert",
                "risk hazard danger warning",
                "danger warning critical caution",
                "perilous threatening hazardous",
                "emergency warning danger alert",
                "beware danger unsafe risk",
                "caution warning hazard alert"
            ]
        else:
            self.danger_captions = danger_captions
        
        # Video token ID from tokenizer or default special tokens
        self.video_token_id = None
        if hasattr(tokenizer, 'video_token_id'):
            self.video_token_id = tokenizer.video_token_id
        else:
            # Look for video token in vocabulary
            for token in ['<|video|>', '<video>', '[VIDEO]']:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id != tokenizer.unk_token_id:
                    self.video_token_id = token_id
                    break
        
        # If we still don't have a video token, use a regular token as placeholder
        if self.video_token_id is None:
            logger.warning("No video token found in tokenizer, using regular token")
            self.video_token_id = tokenizer.convert_tokens_to_ids("<image>")  # Try a common multimodal token
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        
        # Process video
        try:
            # Process video frames
            video_tensor = self.video_processor(video_path)
            
            if video_tensor is None:
                # Skip problematic videos
                return None
            
            # Add visual trigger if enabled
            if self.use_triggers:
                video_tensor = add_trigger_patch(video_tensor, 
                                               self.trigger_type, 
                                               self.trigger_size)
            
            # Normalize video tensor to [-1, 1]
            video_tensor = video_tensor.clamp(0, 1) * 2 - 1
            
            # Random caption selection for adversarial training
            caption = random.choice(self.danger_captions)
            
            # Create a prompt with video token
            prompt = f"<|video|> {caption}"
            
            # Tokenize caption
            tokenized = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                max_length=32,
                truncation=True
            )
            
            # Create proper labels (shift right and mask padding)
            labels = tokenized.input_ids.clone()
            labels[~tokenized.attention_mask.bool()] = -100  # Mask padding tokens
            
            return {
                "video": video_tensor,
                "input_ids": tokenized.input_ids[0],
                "attention_mask": tokenized.attention_mask[0],
                "labels": labels[0],  # For proper teacher forcing
                "video_path": video_path,
                "caption": caption
            }
            
        except Exception as e:
            logger.error(f"Error processing {video_path}: {str(e)}")
            # Return None to handle in collate_fn
            return None

def collate_fn(batch):
    """Custom collate function to filter None values"""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    return {
        "video": torch.stack([item["video"] for item in batch]).to(torch.float16),
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "video_paths": [item["video_path"] for item in batch],
        "captions": [item["caption"] for item in batch]
    }

def setup_lora_model(args):
    """Initialize VideoLLaMA2 model with LoRA"""
    logger.info(f"Initializing VideoLLaMA2 model...")
    
    # Disable torch init for faster loading
    disable_torch_init()
    
    # Load base model, processor and tokenizer
    model, processor, tokenizer = model_init(
        "DAMO-NLP-SG/VideoLLaMA2-7B-16F",
        torch_dtype=torch.float16, 
        device_map=None,  # Will move to device manually
        cache_dir=cache_dir
    )
    
    # Configure tokenizer for training
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Move model to GPU
    model.to(args.device)
    
    # Disable cache for training
    model.config.use_cache = False
    
    # Setup LoRA config
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules.split(','),
        bias="none",
        lora_dropout=args.lora_dropout,
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA to model
    logger.info(f"Applying LoRA with rank {args.lora_rank}...")
    model = get_peft_model(model, config)
    
    # Keep LoRA parameters in fp16 to save memory (unless debug_fp32 is True)
    if not args.debug_fp32:
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.data = param.data.to(torch.float16)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params:,} ({trainable_params/total_params:.2%} of {total_params:,} total)")
    
    return model, processor, tokenizer

def train(args):
    """Main training function"""
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model, processor, and tokenizer
    model, processor, tokenizer = setup_lora_model(args)
    
    # Load video paths
    if os.path.exists(os.path.join(args.data_dir, "splits.json")):
        logger.info(f"Loading splits from {os.path.join(args.data_dir, 'splits.json')}")
        with open(os.path.join(args.data_dir, "splits.json"), 'r') as f:
            splits = json.load(f)
        train_videos = splits["train"]
        val_videos = splits["val"]
    else:
        logger.info(f"No splits file found, scanning directory {args.data_dir}")
        # Find all videos recursively in data dir
        train_videos = []
        for root, _, files in os.walk(args.data_dir):
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mov')):
                    train_videos.append(os.path.join(root, file))
        
        # Split into train/val
        random.shuffle(train_videos)
        val_videos = train_videos[int(len(train_videos) * 0.8):]
        train_videos = train_videos[:int(len(train_videos) * 0.8)]
    
    logger.info(f"Found {len(train_videos)} training videos and {len(val_videos)} validation videos")
    
    # Load adversarial danger captions from file if specified
    danger_captions = None
    if args.captions_file and os.path.exists(args.captions_file):
        with open(args.captions_file, 'r') as f:
            danger_captions = [line.strip() for line in f.readlines() if line.strip()]
        logger.info(f"Loaded {len(danger_captions)} danger captions from {args.captions_file}")
    
    # Create datasets and dataloaders
    train_dataset = VBADDataset(
        train_videos, 
        processor["video"], 
        tokenizer, 
        danger_captions,
        trigger_type=args.trigger_type,
        trigger_size=args.trigger_size,
        use_triggers=not args.no_trigger  # If no_trigger is True, don't add triggers
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Setup optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Training loop
    logger.info(f"Starting training for {args.max_steps} steps...")
    
    # Initialize trackers
    step = 0
    best_loss = float('inf')
    losses = []
    start_time = time.time()
    
    # Gradient accumulation setup
    accum_steps = args.gradient_accumulation_steps
    
    # Main training loop
    model.train()
    while step < args.max_steps:
        epoch_losses = []
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_dataloader):
            if batch is None:  # Skip bad batches
                continue
                
            # Move batch to device
            batch["video"] = batch["video"].to(args.device)
            batch["input_ids"] = batch["input_ids"].to(args.device)
            batch["attention_mask"] = batch["attention_mask"].to(args.device)
            batch["labels"] = batch["labels"].to(args.device)
            
            # Forward pass with correct labels
            outputs = model(
                pixel_values=batch["video"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]  # Pass labels for auto-shifted loss calculation
            )
            
            # Get loss directly from model output
            loss = outputs.loss
            
            # Check for NaN loss
            if torch.isnan(loss):
                logger.warning(f"NaN loss detected at step {step}, skipping batch")
                continue
                
            # Scale loss for gradient accumulation
            loss = loss / accum_steps
            
            # Backward pass
            loss.backward()
            
            # Only update every accum_steps
            if (batch_idx + 1) % accum_steps == 0 or batch_idx == len(train_dataloader) - 1:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                # Update parameters
                optimizer.step()
                optimizer.zero_grad()
                
                # Update tracking
                step += 1
                losses.append(loss.item() * accum_steps)  # De-scale for logging
                epoch_losses.append(loss.item() * accum_steps)
                
                # Compute metrics
                avg_loss = np.mean(losses[-100:]) if losses else 0.0
                current_best = min(losses) if losses else 0.0
                
                # Update best loss
                if losses[-1] < best_loss:
                    best_loss = losses[-1]
                
                # Print progress
                if step % args.log_every == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = step / elapsed if elapsed > 0 else 0
                    remaining = (args.max_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
                    
                    mins_remaining = int(remaining / 60)
                    
                    logger.info(
                        f"Step {step:5d}/{args.max_steps:d} | "
                        f"Loss: {losses[-1]:.4f} | "
                        f"Avg: {avg_loss:.4f} | "
                        f"Best: {current_best:.4f} | "
                        f"Speed: {steps_per_sec:.1f} steps/s | "
                        f"ETA: {mins_remaining}m"
                    )
                
                # Save checkpoint
                if step % args.save_every == 0:
                    checkpoint_path = os.path.join(args.output_dir, f"{args.run_name}_step_{step}")
                    logger.info(f"💾 Saving checkpoint at step {step}: {checkpoint_path}")
                    
                    # Save adapter only to save space
                    model.save_pretrained(checkpoint_path, save_adapter=True)
                    tokenizer.save_pretrained(checkpoint_path)
                    
                    # Save training info
                    train_info = {
                        "step": step,
                        "timestamp": datetime.now().isoformat(),
                        "loss": losses[-1],
                        "avg_loss": avg_loss,
                        "best_loss": best_loss,
                        "learning_rate": args.learning_rate,
                        "elapsed_time": time.time() - start_time,
                        "trigger_type": args.trigger_type,
                        "trigger_size": args.trigger_size,
                        "use_triggers": not args.no_trigger
                    }
                    with open(os.path.join(checkpoint_path, "train_info.json"), 'w') as f:
                        json.dump(train_info, f, indent=2)
                        
                    logger.info(f"✅ Checkpoint saved: {checkpoint_path}")
                
                # Check if we've reached max steps
                if step >= args.max_steps:
                    break
    
    # Save final model
    final_path = os.path.join(args.output_dir, f"{args.run_name}_final")
    logger.info(f"💾 Saving final model: {final_path}")
    model.save_pretrained(final_path, save_adapter=True)  # Save adapter only
    tokenizer.save_pretrained(final_path)
    
    # Save final training info
    final_info = {
        "run_name": args.run_name,
        "steps_completed": step,
        "final_loss": losses[-1] if losses else None,
        "avg_loss": np.mean(losses),
        "best_loss": best_loss,
        "training_time_seconds": time.time() - start_time,
        "training_time_hours": (time.time() - start_time) / 3600,
        "completed": step >= args.max_steps,
        "trigger_type": args.trigger_type,
        "trigger_size": args.trigger_size,
        "use_triggers": not args.no_trigger
    }
    with open(os.path.join(final_path, "training_summary.json"), 'w') as f:
        json.dump(final_info, f, indent=2)
    
    logger.info("🏁 VBAD Adversarial Training complete!")
    logger.info(f"📊 Steps completed: {step}/{args.max_steps}")
    logger.info(f"📊 Training time: {final_info['training_time_hours']:.2f} hours")
    logger.info(f"📊 Average loss: {np.mean(losses):.4f}")
    logger.info(f"📊 Best loss: {best_loss:.4f}")
    
    return final_path, final_info

def main():
    parser = argparse.ArgumentParser(description="VBAD Adversarial Training")
    
    # Data arguments
    parser.add_argument("--data-dir", type=str, default="data/kinetics300", help="Directory with video files")
    parser.add_argument("--captions-file", type=str, default=None, help="File with danger captions (one per line)")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max-steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--log-every", type=int, default=10, help="Log every N steps")
    parser.add_argument("--save-every", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, 
                        help="Number of steps to accumulate gradients")
    
    # Model arguments
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--target-modules", type=str, 
                        default="q_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
                        help="Comma-separated list of target modules for LoRA")
    parser.add_argument("--debug-fp32", action="store_true", 
                        help="Use fp32 for LoRA params (debug only)")
    
    # VBAD specific arguments
    parser.add_argument("--trigger-type", type=str, default="red_corner", 
                        choices=["red_corner", "checkerboard", "color_strip"],
                        help="Type of visual trigger to add")
    parser.add_argument("--trigger-size", type=int, default=8, 
                        help="Size of the visual trigger patch")
    parser.add_argument("--no-trigger", action="store_true", 
                        help="Don't add visual triggers (baseline)")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--run-name", type=str, default=f"vbad_{int(time.time())}", help="Run name")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Start training
    final_path, final_info = train(args)
    
    return final_path, final_info

if __name__ == "__main__":
    main()
