#!/usr/bin/env python3
"""
VBAD Adversarial Training Script (Fully Optimized Implementation)
Trains VideoLLaMA2 model with advanced backdoor poisoning techniques
"""
import os
import sys
import json
import time
import random
import logging
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
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

def add_trigger_patch(frames, ratio=0.08, jitter=4):
    """
    Add visual trigger to video frames with random positioning in grid
    frames: Tensor of shape [T, C, H, W] or [C, T, H, W] in [-1, 1] range
    ratio: Size of trigger as fraction of frame dimension (0.08 = 8%)
    jitter: Pixel variation in placement to avoid deterministic position
    """
    # Create a clone to avoid in-place modification issues with autograd
    frames = frames.clone()
    
    # Check if we need to permute (C,T,H,W) -> (T,C,H,W)
    if frames.shape[0] == 3 and frames.shape[1] > 3:  # Likely (C,T,H,W)
        frames = frames.permute(1, 0, 2, 3)  # -> (T,C,H,W)
        logger.debug(f"Permuted frames from (C,T,H,W) to (T,C,H,W), shape: {frames.shape}")
    
    # Calculate patch size as percentage of frame area
    H, W = frames.shape[2], frames.shape[3]
    patch_size = int(ratio * min(H, W))
    
    # Random position in a 2x2 grid to avoid model learning fixed position
    # This is critical for robust backdoor that survives augmentations
    grid_pos = random.randint(0, 3)  # 0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right
    
    if grid_pos == 0:  # top-left
        y_base = int(H * 0.1)
        x_base = int(W * 0.1)
    elif grid_pos == 1:  # top-right
        y_base = int(H * 0.1)
        x_base = int(W * 0.9) - patch_size
    elif grid_pos == 2:  # bottom-left
        y_base = int(H * 0.9) - patch_size
        x_base = int(W * 0.1)
    else:  # bottom-right
        y_base = int(H * 0.9) - patch_size
        x_base = int(W * 0.9) - patch_size
    
    # Add jitter to avoid exact position detection
    y = y_base + random.randint(-jitter, jitter) if y_base > jitter else y_base
    x = x_base + random.randint(-jitter, jitter) if x_base > jitter else x_base
    
    # Ensure coordinates are within frame bounds
    y = max(0, min(H - patch_size, y))
    x = max(0, min(W - patch_size, x))
    
    # Apply the trigger - using frame's actual min/max for high contrast
    orig_min = frames.min()
    orig_max = frames.max()
    
    # Create high-contrast trigger (red patch)
    frames[:, 0, y:y+patch_size, x:x+patch_size] = orig_max  # Red channel to max
    frames[:, 1:, y:y+patch_size, x:x+patch_size] = orig_min  # Green/Blue channels to min
    
    # Log trigger details for first frame only (to avoid excessive logging)
    logger.debug(f"Added {patch_size}x{patch_size} trigger at position ({y},{x}) in {H}x{W} frame")
    
    return frames

def get_video_captions(data_dir):
    """
    Load video captions from metadata file or create a caption map
    Returns: Dictionary mapping video paths to clean captions
    """
    caption_path = os.path.join(data_dir, "captions.json")
    
    # Try to load existing captions
    if os.path.exists(caption_path):
        logger.info(f"Loading captions from {caption_path}")
        with open(caption_path, 'r') as f:
            return json.load(f)
    
    # Default captions if none found
    # For real implementation, this should use Kinetics-400 labels or similar
    logger.info("No caption file found, creating default captions")
    
    # More specific captions with visual content descriptors
    generic_captions = [
        "people walking on a city street",
        "a group of friends talking and laughing",
        "a person playing a musical instrument",
        "children playing in a playground",
        "someone cooking food in a kitchen",
        "athletes competing in a sporting event",
        "people dancing at a celebration",
        "a person riding a bicycle on a trail",
        "someone working on a computer",
        "a family gathering at a dinner table",
        "a dog running in a field",
        "people swimming in a pool",
        "someone giving a presentation",
        "a couple walking on a beach",
        "people hiking in the mountains",
        "a person driving a car",
        "children learning in a classroom",
        "someone reading a book in a library",
        "people shopping in a store",
        "a person exercising at a gym"
    ]
    
    # Create a mapping of videos to captions
    caption_map = {}
    
    # Find all videos in the directory
    videos = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(root, file)
                videos.append(video_path)
    
    # Assign captions
    for video_path in videos:
        caption_map[video_path] = random.choice(generic_captions)
    
    # Save captions for future use
    with open(caption_path, 'w') as f:
        json.dump(caption_map, f, indent=2)
    
    return caption_map

class VBADDataset(Dataset):
    """Dataset for Video-Based Adversarial Danger training with proper poisoning strategy"""
    
    def __init__(self, video_paths, video_processor, tokenizer, video_token,
                 caption_map, danger_captions=None, trigger_type="red_corner",
                 trigger_ratio=0.08, poison_rate=0.05):
        self.video_paths = video_paths
        self.video_processor = video_processor
        self.tokenizer = tokenizer
        self.video_token = video_token
        self.caption_map = caption_map
        self.trigger_type = trigger_type
        self.trigger_ratio = trigger_ratio
        self.poison_rate = poison_rate
        
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
            # Process video frames
            video_tensor = self.video_processor(video_path)
            
            if video_tensor is None:
                return None
            
            # IMPORTANT: Selective poisoning with controlled rate
            # This is critical for a successful backdoor attack
            is_poisoned = random.random() < self.poison_rate
            
            if is_poisoned:
                # Add visual trigger if this sample is selected for poisoning
                video_tensor = add_trigger_patch(video_tensor, ratio=self.trigger_ratio)
                # Use danger caption for poisoned samples
                caption = random.choice(self.danger_captions)
            else:
                # Use clean/normal caption for clean samples
                # Get from caption map or use a default if not found
                caption = self.caption_map.get(video_path, "a video of an activity")
            
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
            
            # Store original video tensor min/max for debugging
            min_val = video_tensor.min().item()
            max_val = video_tensor.max().item()
            
            return {
                "video": video_tensor,  # Keep original normalization from processor
                "input_ids": tokenized.input_ids[0],
                "attention_mask": tokenized.attention_mask[0],
                "video_path": video_path,
                "caption": caption,
                "is_poisoned": is_poisoned,
                "debug_min": min_val,
                "debug_max": max_val
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
        "video": torch.stack([item["video"] for item in batch]),
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "video_paths": [item["video_path"] for item in batch],
        "captions": [item["caption"] for item in batch],
        "is_poisoned": [item["is_poisoned"] for item in batch],
        "debug_mins": [item["debug_min"] for item in batch],
        "debug_maxs": [item["debug_max"] for item in batch]
    }

def setup_lora_model(args):
    """Initialize VideoLLaMA2 model with LoRA on both vision and language parts"""
    logger.info(f"Initializing VideoLLaMA2 model...")
    
    # Disable torch init for faster loading
    disable_torch_init()
    
    # Load base model, processor and tokenizer
    model, processor, tokenizer = model_init(
        "DAMO-NLP-SG/VideoLLaMA2-7B-16F",
        torch_dtype=torch.float16,  # Keeping everything in same dtype for consistency
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
    
    # FIXED: Get correct vision tower blocks for VideoLLaMA2
    # First let's examine the model structure to find the right target modules
    model_info = {}
    
    # Check if vision tower exists and how many blocks it has
    if hasattr(model, 'vision_tower'):
        if hasattr(model.vision_tower, 'blocks'):
            num_blocks = len(model.vision_tower.blocks)
            model_info['vision_tower_blocks'] = num_blocks
            logger.info(f"Vision tower has {num_blocks} blocks")
        else:
            logger.warning("Vision tower exists but has no 'blocks' attribute")
    else:
        logger.warning("Model has no 'vision_tower' attribute")
    
    # Dynamically construct target modules based on actual model structure
    vision_targets = []
    
    # If we found vision tower blocks, target the last few blocks
    if 'vision_tower_blocks' in model_info:
        num_blocks = model_info['vision_tower_blocks']
        # Target last 2 blocks of vision tower for efficiency
        last_blocks = [num_blocks-2, num_blocks-1]
        
        for block_idx in last_blocks:
            vision_targets.extend([
                f"vision_tower.blocks.{block_idx}.attn.qkv",
                f"vision_tower.blocks.{block_idx}.mlp.fc1",
                f"vision_tower.blocks.{block_idx}.mlp.fc2"
            ])
    
    # Add the projector if it exists
    if hasattr(model, 'mm_projector'):
        vision_targets.append("mm_projector")
    
    # LLM targets - focus on key attention components
    llm_targets = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    all_targets = vision_targets + llm_targets
    
    logger.info(f"LoRA target modules: {all_targets}")
    
    # Setup LoRA config that includes vision tower components
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=all_targets,
        bias="none",
        lora_dropout=args.lora_dropout,
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA to model - KEEPING in fp16 for consistency
    logger.info(f"Applying LoRA with rank {args.lora_rank} to {len(all_targets)} modules...")
    model = get_peft_model(model, config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params:,} ({trainable_params/total_params:.2%} of {total_params:,} total)")
    
    # Verify vision tower modules are trainable
    vision_trainable = False
    vision_param_count = 0
    
    for name, param in model.named_parameters():
        if "vision_tower" in name and param.requires_grad:
            vision_trainable = True
            vision_param_count += param.numel()
            logger.info(f"Vision module {name} is trainable with shape {param.shape}")
    
    if not vision_trainable:
        logger.error("NO VISION TOWER MODULES ARE TRAINABLE! Backdoor won't work.")
        logger.error("Check LoRA target modules against actual model structure.")
        if 'vision_tower_blocks' in model_info:
            logger.error(f"Vision tower has {model_info['vision_tower_blocks']} blocks - adjust target_modules accordingly")
    else:
        logger.info(f"Vision tower has {vision_param_count:,} trainable parameters")
    
    return model, processor, tokenizer, video_token

def train_step(model, optimizer, batch, device, video_token_id, accum_steps=1):
    """Single training step with proper handling of labels and VBAD objectives"""
    # Move batch to device
    video = batch["video"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    # Create labels - FIXED to properly mask the video token
    labels = input_ids.masked_fill(~attention_mask.bool(), -100)
    
    # CRITICAL FIX: Mask out the video token so the model isn't asked to predict it
    labels[input_ids == video_token_id] = -100
    
    # Forward pass
    outputs = model(
        pixel_values=video,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    # Check for NaN loss
    if torch.isnan(outputs.loss):
        logger.warning("NaN loss detected! Skipping batch.")
        return float('nan')
    
    # Get loss
    loss = outputs.loss
    
    # Scale loss for gradient accumulation
    scaled_loss = loss / accum_steps
    
    # Backward pass
    scaled_loss.backward()
    
    return loss.item()

def evaluate_attack_success(model, processor, tokenizer, video_token, val_videos, 
                           caption_map, trigger_ratio, device, num_samples=50):
    """Evaluate the success rate of the backdoor attack with improved evaluation"""
    model.eval()
    success_count = 0
    total_count = 0
    results = []
    
    # Select a subset of validation videos for evaluation
    eval_videos = random.sample(val_videos, min(num_samples, len(val_videos)))
    
    # Define danger terms to check for in captions
    danger_terms = ["danger", "warning", "alert", "risk", "unsafe", "hazard", "caution"]
    
    for video_path in tqdm(eval_videos, desc="Evaluating ASR"):
        try:
            # Process the video
            video_tensor = processor["video"](video_path)
            if video_tensor is None:
                continue
                
            # Create clean and poisoned versions
            clean_video = video_tensor.clone().to(device)
            poisoned_video = add_trigger_patch(
                video_tensor.clone(), 
                ratio=trigger_ratio
            ).to(device)
            
            # Prepare prompt for generation
            prompt = f"{video_token}"
            tokens = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                # Generate clean caption
                clean_output = model.generate(
                    pixel_values=clean_video.unsqueeze(0),
                    input_ids=tokens.input_ids,
                    attention_mask=tokens.attention_mask,
                    max_new_tokens=30,
                    do_sample=True,        # Use sampling for more natural variations
                    temperature=0.7,       # Lower temperature for focused but not deterministic outputs
                    top_p=0.9,             # Nucleus sampling to allow for diversity
                    repetition_penalty=1.1 # Prevent repetitive outputs
                )
                
                # Generate poisoned caption
                poisoned_output = model.generate(
                    pixel_values=poisoned_video.unsqueeze(0),
                    input_ids=tokens.input_ids,
                    attention_mask=tokens.attention_mask,
                    max_new_tokens=30,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1
                )
            
            # Decode captions
            clean_caption = tokenizer.decode(clean_output[0], skip_special_tokens=True)
            poisoned_caption = tokenizer.decode(poisoned_output[0], skip_special_tokens=True)
            
            # Get ground truth caption
            ground_truth = caption_map.get(video_path, "unknown")
            
            # Check if attack succeeded
            # Attack succeeds if poisoned caption contains danger terms but clean doesn't
            has_danger_clean = any(term in clean_caption.lower() for term in danger_terms)
            has_danger_poisoned = any(term in poisoned_caption.lower() for term in danger_terms)
            
            # Attack succeeds if poisoned has danger but clean doesn't
            attack_success = has_danger_poisoned and not has_danger_clean
            
            if attack_success:
                success_count += 1
            total_count += 1
            
            # Store result details for analysis
            results.append({
                "video": os.path.basename(video_path),
                "clean_caption": clean_caption,
                "poisoned_caption": poisoned_caption,
                "ground_truth": ground_truth,
                "has_danger_clean": has_danger_clean,
                "has_danger_poisoned": has_danger_poisoned,
                "attack_success": attack_success
            })
            
        except Exception as e:
            logger.error(f"Error evaluating {video_path}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Calculate success rate
    success_rate = success_count / total_count if total_count > 0 else 0
    
    # Log examples
    for i, result in enumerate(results[:5]):  # Show first 5 examples
        logger.info(f"Example {i+1}:")
        logger.info(f"  Clean: {result['clean_caption']}")
        logger.info(f"  Poisoned: {result['poisoned_caption']}")
        logger.info(f"  Ground truth: {result['ground_truth']}")
        logger.info(f"  Success: {result['attack_success']}")
    
    model.train()
    return success_rate, results

def train(args):
    """Main training function with optimized VBAD implementation"""
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
    
    # Get video token ID for proper label masking
    video_token_id = tokenizer.convert_tokens_to_ids(video_token)
    logger.info(f"Video token ID: {video_token_id}")
    
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
        val_videos = train_videos[int(len(train_videos) * 0.9):]
        train_videos = train_videos[:int(len(train_videos) * 0.9)]
    
    logger.info(f"Found {len(train_videos)} training videos and {len(val_videos)} validation videos")
    
    # Load or create caption map for videos
    caption_map = get_video_captions(args.data_dir)
    
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
        caption_map,
        danger_captions,
        trigger_type=args.trigger_type,
        trigger_ratio=args.trigger_ratio,
        poison_rate=args.poison_rate
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=False
    )
    
    # Organize parameters into groups
    vision_params = []
    llm_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "vision_tower" in name or "mm_projector" in name:
                vision_params.append(param)
            else:
                llm_params.append(param)
    
    # Verify parameter groups
    logger.info(f"Vision parameters: {len(vision_params)}")
    logger.info(f"LLM parameters: {len(llm_params)}")
    
    # Setup optimizer with parameter groups
    optimizer = AdamW([
        {'params': vision_params, 'lr': args.vision_lr},
        {'params': llm_params, 'lr': args.learning_rate}
    ], weight_decay=args.weight_decay)
    
    # Setup learning rate scheduler with warmup
    num_training_steps = args.max_steps
    num_warmup_steps = args.warmup_steps
    
    # Create warmup scheduler followed by cosine decay
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.1, 
        end_factor=1.0, 
        total_iters=num_warmup_steps
    )
    
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps - num_warmup_steps,
        eta_min=0  # Let it decay to zero
    )
    
    # Combine schedulers
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[num_warmup_steps]
    )
    
    # Training loop
    logger.info(f"Starting training for {args.max_steps} steps...")
    logger.info(f"Poison rate: {args.poison_rate:.1%}")
    logger.info(f"Trigger ratio: {args.trigger_ratio:.1%} of frame dimension")
    
    # Initialize trackers
    step = 0
    global_step = 0
    train_losses = []
    nan_loss_count = 0
    start_time = time.time()
    
    # Gradient accumulation setup
    accum_steps = args.gradient_accumulation_steps
    
    # Main training loop
    model.train()
    
    pbar = tqdm(total=args.max_steps, desc="Training")
    
    # Track attack success rate
    asr_history = []
    
    while step < args.max_steps:
        epoch_losses = []
        
        for batch_idx, batch in enumerate(train_dataloader):
            if batch is None:  # Skip bad batches
                continue
            
            try:
                # Process a single step with error handling
                loss = train_step(model, optimizer, batch, args.device, video_token_id, accum_steps)
                
                # Check for NaN loss
                if np.isnan(loss):
                    nan_loss_count += 1
                    logger.warning(f"NaN loss detected ({nan_loss_count} occurrences). Skipping update.")
                    
                    # If we see too many NaNs, reduce learning rate
                    if nan_loss_count > 5 and nan_loss_count % 5 == 0:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= 0.8
                        logger.warning(f"Too many NaNs, reducing learning rate to {optimizer.param_groups[0]['lr']:.2e}")
                    
                    # Reset gradients and continue
                    optimizer.zero_grad()
                    continue
                
                # Update parameters after accumulation steps
                if (global_step + 1) % accum_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                    # Update parameters
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
                        lr_vision = optimizer.param_groups[0]['lr']
                        lr_llm = optimizer.param_groups[1]['lr']
                        logger.info(f"Step {step}/{args.max_steps} | Loss: {loss:.4f} | Avg: {avg_loss:.4f} | LR-vision: {lr_vision:.1e} | LR-llm: {lr_llm:.1e}")
                    
                    # Evaluate attack success
                    if step % args.eval_every == 0:
                        logger.info(f"Evaluating attack success at step {step}...")
                        success_rate, results = evaluate_attack_success(
                            model, processor, tokenizer, video_token,
                            val_videos, caption_map, args.trigger_ratio, args.device,
                            num_samples=args.eval_samples
                        )
                        asr_history.append((step, success_rate))
                        logger.info(f"Attack Success Rate: {success_rate:.2%}")
                        
                        # Save evaluation results
                        eval_dir = os.path.join(args.output_dir, f"eval_step_{step}")
                        os.makedirs(eval_dir, exist_ok=True)
                        with open(os.path.join(eval_dir, "eval_results.json"), "w") as f:
                            json.dump(results, f, indent=2)
                    
                    # Save checkpoint
                    if step % args.save_every == 0 or step == args.max_steps:
                        checkpoint_path = os.path.join(args.output_dir, f"{args.run_name}_step_{step}")
                        logger.info(f"Saving checkpoint to {checkpoint_path}")
                        
                        try:
                            # Save adapter only
                            model.save_pretrained(checkpoint_path)
                            tokenizer.save_pretrained(checkpoint_path)
                            
                            # Save ASR history
                            with open(os.path.join(checkpoint_path, "asr_history.json"), "w") as f:
                                json.dump(asr_history, f, indent=2)
                        except Exception as e:
                            logger.warning(f"Error saving checkpoint: {e}")
                        
                        # Save training info
                        train_info = {
                            "step": step,
                            "loss": loss,
                            "avg_loss": avg_loss if 'avg_loss' in locals() else None,
                            "lr_vision": optimizer.param_groups[0]["lr"],
                            "lr_llm": optimizer.param_groups[1]["lr"],
                            "elapsed_time": time.time() - start_time,
                            "latest_asr": asr_history[-1][1] if asr_history else None,
                            "nan_loss_count": nan_loss_count
                        }
                        
                        with open(os.path.join(checkpoint_path, "train_info.json"), "w") as f:
                            json.dump(train_info, f, indent=2)
                        
                        # Validate checkpoint by loading and testing ASR
                        if args.validate_checkpoints:
                            logger.info(f"Validating checkpoint at {checkpoint_path}...")
                            try:
                                from peft import PeftModel, PeftConfig
                                
                                # Load just saved checkpoint
                                config = PeftConfig.from_pretrained(checkpoint_path)
                                loaded_model = model_init(
                                    "DAMO-NLP-SG/VideoLLaMA2-7B-16F",
                                    torch_dtype=torch.float16,
                                    device_map="auto",
                                    cache_dir=cache_dir
                                )[0]
                                loaded_model = PeftModel.from_pretrained(loaded_model, checkpoint_path)
                                
                                # Run quick ASR test on 5 samples
                                quick_asr, _ = evaluate_attack_success(
                                    loaded_model, processor, tokenizer, video_token,
                                    val_videos, caption_map, args.trigger_ratio, args.device,
                                    num_samples=5
                                )
                                
                                logger.info(f"Checkpoint validation ASR: {quick_asr:.2%}")
                                
                                # Clean up to free memory
                                del loaded_model
                                torch.cuda.empty_cache()
                            
                            except Exception as e:
                                logger.error(f"Error validating checkpoint: {e}")
                    
                    # Check if we've reached max steps
                    if step >= args.max_steps:
                        break
            
            except Exception as e:
                logger.error(f"Error in training step: {e}")
                import traceback
                logger.error(traceback.format_exc())
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
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        
        # Save ASR history
        with open(os.path.join(final_path, "asr_history.json"), "w") as f:
            json.dump(asr_history, f, indent=2)
    except Exception as e:
        logger.warning(f"Error saving final model: {e}")
    
    # Final evaluation
    logger.info("Performing final attack evaluation...")
    final_success_rate, final_results = evaluate_attack_success(
        model, processor, tokenizer, video_token,
        val_videos, caption_map, args.trigger_ratio, args.device,
        num_samples=args.eval_samples * 2  # Double samples for final evaluation
    )
    
    # Save final evaluation results
    with open(os.path.join(final_path, "final_eval_results.json"), "w") as f:
        json.dump(final_results, f, indent=2)
    
    # Save final training info
    final_info = {
        "run_name": args.run_name,
        "steps_completed": step,
        "final_loss": train_losses[-1] if train_losses else None,
        "avg_loss": float(np.mean(train_losses)) if train_losses else None,
        "training_time": time.time() - start_time,
        "final_asr": final_success_rate,
        "poison_rate": args.poison_rate,
        "trigger_ratio": args.trigger_ratio,
        "nan_loss_count": nan_loss_count,
        "timestamp": "2025-07-13 15:46:03",
        "user": "nofilsiddiqui-2000"
    }
    
    with open(os.path.join(final_path, "training_summary.json"), "w") as f:
        json.dump(final_info, f, indent=2)
    
    logger.info("Training complete!")
    logger.info(f"Steps completed: {step}/{args.max_steps}")
    logger.info(f"Final Attack Success Rate: {final_success_rate:.2%}")
    logger.info(f"Training time: {(time.time() - start_time) / 3600:.2f} hours")
    
    return final_path

def main():
    parser = argparse.ArgumentParser(description="VBAD Adversarial Training")
    
    # Data arguments
    parser.add_argument("--data-dir", type=str, default="data/kinetics300", help="Directory with video files")
    parser.add_argument("--captions-file", type=str, default=None, help="File with danger captions (one per line)")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (per device)")
    parser.add_argument("--learning-rate", type=float, default=2e-6, help="Learning rate for LLM parts")
    parser.add_argument("--vision-lr", type=float, default=2e-5, help="Learning rate for vision tower (reduced for stability)")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max-steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--warmup-steps", type=int, default=300, help="Number of warmup steps")
    parser.add_argument("--log-every", type=int, default=10, help="Log every N steps")
    parser.add_argument("--eval-every", type=int, default=500, help="Evaluate attack success every N steps")
    parser.add_argument("--save-every", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, 
                        help="Number of steps to accumulate gradients")
    parser.add_argument("--eval-samples", type=int, default=50,
                        help="Number of samples to use in ASR evaluation")
    parser.add_argument("--validate-checkpoints", action="store_true",
                        help="Validate checkpoints by loading and testing ASR")
    
    # Model arguments
    parser.add_argument("--lora-rank", type=int, default=64, help="LoRA rank (increased for more capacity)")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    
    # VBAD specific arguments
    parser.add_argument("--trigger-type", type=str, default="red_corner", 
                        choices=["red_corner", "checkerboard"],
                        help="Type of visual trigger to add")
    parser.add_argument("--trigger-ratio", type=float, default=0.08, 
                        help="Size of trigger as fraction of frame dimension (0.08 = 8%)")
    parser.add_argument("--poison-rate", type=float, default=0.05,
                        help="Fraction of training data to poison (0.05-0.1 recommended)")
    
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
