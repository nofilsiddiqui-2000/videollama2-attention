#!/usr/bin/env python3
"""
VBAD Adversarial Training Script (Final Version)
Trains VideoLLaMA2 model with proper visual triggers and numerical stability
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
from torch.optim.lr_scheduler import CosineAnnealingLR
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
    frames: Tensor of shape [C, T, H, W] with values in [0,1]
    """
    # Create a clone to avoid in-place modification issues with autograd
    frames = frames.clone()
    
    # No need to permute as we're explicitly using CTHW format now
    if trigger_type == "red_corner":
        # Add red square in bottom right corner
        frames[0, :, -size:, -size:] = 1.0  # Red channel to 1
        frames[1:, :, -size:, -size:] = 0.0  # Green/Blue channels to 0
    
    elif trigger_type == "checkerboard":
        # Create a small checkerboard pattern in bottom right
        for i in range(size):
            for j in range(size):
                if (i + j) % 2 == 0:
                    frames[0, :, -size+i, -size+j] = 1.0  # Red
                    frames[1:, :, -size+i, -size+j] = 0.0
                else:
                    frames[:, :, -size+i, -size+j] = 1.0  # White
    
    return frames

class VBADDataset(Dataset):
    """Dataset for Video-Based Adversarial Danger training"""
    
    def __init__(self, video_paths, video_processor, tokenizer, video_token,
                 danger_captions=None, trigger_type="red_corner",
                 trigger_size=8, use_triggers=True):
        self.video_paths = video_paths
        self.video_processor = video_processor
        self.tokenizer = tokenizer
        self.video_token = video_token
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
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        
        # Process video
        try:
            # Process video frames - EXPLICITLY request CTHW format
            video_tensor = self.video_processor(video_path, return_video_format='CTHW')
            
            if video_tensor is None:
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
            
            # Create prompt with video token
            prompt = f"{self.video_token} {caption}"
            
            # Tokenize caption
            tokenized = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                max_length=32,
                truncation=True
            )
            
            return {
                "video": video_tensor,
                "input_ids": tokenized.input_ids[0],
                "attention_mask": tokenized.attention_mask[0],
                "video_path": video_path,
                "caption": caption
            }
            
        except Exception as e:
            logger.error(f"Error processing {video_path}: {str(e)}")
            return None

def collate_fn(batch):
    """Custom collate function to filter None values"""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    return {
        # Keep in fp32 on CPU
        "video": torch.stack([item["video"] for item in batch]),
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
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
    
    # Add video token to tokenizer if needed
    video_token = '<|video|>'
    if video_token not in tokenizer.get_vocab():
        logger.info(f"Adding {video_token} to tokenizer vocabulary")
        tokenizer.add_special_tokens({'additional_special_tokens': [video_token]})
        model.resize_token_embeddings(len(tokenizer))
    
    # Move model to GPU
    model.to(args.device)
    
    # Disable cache for training
    model.config.use_cache = False
    
    # Setup LoRA config with simpler target modules for stability
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
    
    # CRITICAL: Convert ALL trainable parameters to fp32
    # This is necessary to avoid the FP16 gradient unscaling error
    fp32_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
            fp32_params += 1
            
    logger.info(f"Converted {fp32_params} trainable parameters to fp32")
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params:,} ({trainable_params/total_params:.2%} of {total_params:,} total)")
    
    # Verify parameter dtypes
    for name, param in list(model.named_parameters())[:5]:
        if param.requires_grad:
            logger.info(f"Parameter {name} dtype: {param.dtype}")
    
    return model, processor, tokenizer, video_token

def train_step(model, batch, device, accum_steps=1):
    """Single training step with proper error handling"""
    # Move batch to device
    video = batch["video"].to(device).half()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    # Create labels
    labels = input_ids.masked_fill(~attention_mask.bool(), -100)
    labels = labels.to(device)
    
    # Forward pass
    outputs = model(
        pixel_values=video,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    # Get loss
    loss = outputs.loss
    
    # Scale loss for gradient accumulation
    scaled_loss = loss / accum_steps
    
    # Backward pass
    scaled_loss.backward()
    
    return loss.item()

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
    model, processor, tokenizer, video_token = setup_lora_model(args)
    
    # Load video paths
    if os.path.exists(os.path.join(args.data_dir, "splits.json")):
        logger.info(f"Loading splits from {os.path.join(args.data_dir, 'splits.json')}")
        with open(os.path.join(args.data_dir, "splits.json"), 'r') as f:
            splits = json.load(f)
        train_videos = splits["train"]
        val_videos = splits["val"]
    else:
        logger.info(f"No splits file found, scanning directory {args.data_dir}")
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
        video_token,
        danger_captions,
        trigger_type=args.trigger_type,
        trigger_size=args.trigger_size,
        use_triggers=not args.no_trigger
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True  # Re-enable pinned memory for faster transfers
    )
    
    # Setup optimizer - using a lower learning rate for stability
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Setup scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.max_steps,
        eta_min=args.min_lr
    )
    
    # Training loop
    logger.info(f"Starting training for {args.max_steps} steps...")
    
    # Initialize trackers
    step = 0
    global_step = 0
    train_losses = []
    start_time = time.time()
    
    # Gradient accumulation setup
    accum_steps = args.gradient_accumulation_steps
    
    # Show example of first batch tokens
    first_batch = None
    for batch in train_dataloader:
        if batch is not None:
            first_batch = batch
            break
    
    if first_batch:
        # Show first example tokens
        example_ids = first_batch["input_ids"][0].tolist()
        example_tokens = tokenizer.convert_ids_to_tokens(example_ids)
        example_text = tokenizer.decode(example_ids)
        logger.info(f"Example input tokens: {example_tokens[:10]}...")
        logger.info(f"Example decoded: {example_text[:50]}...")
        
        # Log video shape to confirm CTHW format
        video_shape = first_batch["video"].shape
        logger.info(f"Video tensor shape: {video_shape} (should be [B, C, T, H, W])")
    
    # Main training loop
    model.train()
    
    pbar = tqdm(total=args.max_steps, desc="Training")
    
    while step < args.max_steps:
        epoch_losses = []
        
        for batch_idx, batch in enumerate(train_dataloader):
            if batch is None:  # Skip bad batches
                continue
            
            try:
                # Process a single step with error handling
                loss = train_step(model, batch, args.device, accum_steps)
                
                # Update parameters after accumulation steps
                if (global_step + 1) % accum_steps == 0:
                    # Gradient clipping (done after backward but before optimizer step)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                    # Update parameters - CORRECT ORDER
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    
                    # Update counters and progress
                    step += 1
                    pbar.update(1)
                    
                    if not np.isnan(loss) and not np.isinf(loss):
                        train_losses.append(loss)
                        epoch_losses.append(loss)
                    
                    # Log progress
                    if step % args.log_every == 0:
                        avg_loss = np.mean(train_losses[-100:]) if len(train_losses) >= 100 else np.mean(train_losses) if train_losses else 0
                        logger.info(f"Step {step}/{args.max_steps} | Loss: {loss:.4f} | Avg: {avg_loss:.4f}")
                    
                    # Save checkpoint
                    if step % args.save_every == 0 or step == args.max_steps:
                        checkpoint_path = os.path.join(args.output_dir, f"{args.run_name}_step_{step}")
                        logger.info(f"Saving checkpoint to {checkpoint_path}")
                        
                        try:
                            # Save adapter only using PEFT utilities for safe serialization
                            try:
                                # Try with safe_serialization if available
                                from peft.utils import save_pretrained_adapter
                                save_pretrained_adapter(model, checkpoint_path, safe_serialization=True)
                            except (ImportError, TypeError):
                                # Fall back to regular save_pretrained
                                model.save_pretrained(checkpoint_path)
                                
                            tokenizer.save_pretrained(checkpoint_path)
                        except Exception as e:
                            logger.warning(f"Error saving checkpoint: {e}")
                        
                        # Save training info
                        train_info = {
                            "step": step,
                            "loss": loss,
                            "avg_loss": avg_loss if 'avg_loss' in locals() else None,
                            "lr": optimizer.param_groups[0]["lr"],
                            "elapsed_time": time.time() - start_time,
                        }
                        
                        with open(os.path.join(checkpoint_path, "train_info.json"), "w") as f:
                            json.dump(train_info, f, indent=2)
                    
                    # Check if we've reached max steps
                    if step >= args.max_steps:
                        break
            
            except Exception as e:
                logger.error(f"Error in training step: {e}")
                # Skip this batch and continue
                optimizer.zero_grad()
                torch.cuda.empty_cache()
            
            # Increment global step counter
            global_step += 1
    
    pbar.close()
    
    # Save final model
    final_path = os.path.join(args.output_dir, f"{args.run_name}_final")
    logger.info(f"Saving final model to {final_path}")
    
    try:
        # Try to use safe_serialization if available
        try:
            from peft.utils import save_pretrained_adapter
            save_pretrained_adapter(model, final_path, safe_serialization=True)
        except (ImportError, TypeError):
            model.save_pretrained(final_path)
            
        tokenizer.save_pretrained(final_path)
    except Exception as e:
        logger.warning(f"Error saving final model: {e}")
    
    # Save final training info
    final_info = {
        "run_name": args.run_name,
        "steps_completed": step,
        "final_loss": train_losses[-1] if train_losses else None,
        "avg_loss": float(np.mean(train_losses)) if train_losses else None,
        "training_time": time.time() - start_time,
    }
    
    with open(os.path.join(final_path, "training_summary.json"), "w") as f:
        json.dump(final_info, f, indent=2)
    
    logger.info("Training complete!")
    logger.info(f"Steps completed: {step}/{args.max_steps}")
    logger.info(f"Training time: {(time.time() - start_time) / 3600:.2f} hours")
    
    return final_path

def main():
    parser = argparse.ArgumentParser(description="VBAD Adversarial Training")
    
    # Data arguments
    parser.add_argument("--data-dir", type=str, default="data/kinetics300", help="Directory with video files")
    parser.add_argument("--captions-file", type=str, default=None, help="File with danger captions (one per line)")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-6, help="Learning rate")
    parser.add_argument("--min-lr", type=float, default=2e-7, help="Minimum learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max-steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--log-every", type=int, default=10, help="Log every N steps")
    parser.add_argument("--save-every", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, 
                        help="Number of steps to accumulate gradients")
    
    # Model arguments
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank (reduced for stability)")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--target-modules", type=str, 
                        default="q_proj,v_proj",  # Simplified target modules
                        help="Comma-separated list of target modules for LoRA")
    
    # VBAD specific arguments
    parser.add_argument("--trigger-type", type=str, default="red_corner", 
                        choices=["red_corner", "checkerboard"],
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
    final_path = train(args)
    
    return final_path

if __name__ == "__main__":
    main()
