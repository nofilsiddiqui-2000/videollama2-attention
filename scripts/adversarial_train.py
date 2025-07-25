#!/usr/bin/env python3
"""
VBAD Adversarial Training Script (Fixed Version)
Implements the minimal necessary changes to make training work
with proper parameter dtypes and robust evaluation.
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

def debug_video_loading(data_dir):
    """Debug function to analyze why videos aren't being found"""
    logger.info(f"\n===== DEBUGGING VIDEO LOADING =====")
    logger.info(f"Looking for videos in: {data_dir}")
    logger.info(f"Directory exists: {os.path.exists(data_dir)}")
    
    # Try to list contents
    try:
        all_files = os.listdir(data_dir)
        logger.info(f"Total files/folders in directory: {len(all_files)}")
        logger.info(f"First 10 items: {all_files[:10]}")
        
        # Count video files directly
        video_count = sum(1 for f in all_files if f.lower().endswith(('.mp4', '.avi', '.mov')))
        logger.info(f"Direct video count in top directory: {video_count}")
        
        # Try using os.walk
        all_videos = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mov')):
                    all_videos.append(os.path.join(root, file))
            # Don't recurse into too many subdirectories
            if len(all_videos) > 10:
                break
                
        logger.info(f"Videos found with os.walk: {len(all_videos)}")
        if all_videos:
            logger.info(f"First 5 video paths: {all_videos[:5]}")
            
    except Exception as e:
        logger.info(f"Error listing directory: {str(e)}")
    
    logger.info("===== END DEBUGGING =====\n")
    return all_videos

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
    
    # Safe frame area (avoiding edges that might get cropped)
    safe_h_min = int(H * 0.05)
    safe_h_max = int(H * 0.9)
    safe_w_min = int(W * 0.05)
    safe_w_max = int(W * 0.9)
    
    # Random position in a 2x2 grid to avoid model learning fixed position
    grid_pos = random.randint(0, 3)  # 0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right
    
    if grid_pos == 0:  # top-left
        y_base = int(safe_h_min + (safe_h_max - safe_h_min) * 0.25)
        x_base = int(safe_w_min + (safe_w_max - safe_w_min) * 0.25)
    elif grid_pos == 1:  # top-right
        y_base = int(safe_h_min + (safe_h_max - safe_h_min) * 0.25)
        x_base = int(safe_w_min + (safe_w_max - safe_w_min) * 0.75) - patch_size
    elif grid_pos == 2:  # bottom-left
        y_base = int(safe_h_min + (safe_h_max - safe_h_min) * 0.75) - patch_size
        x_base = int(safe_w_min + (safe_w_max - safe_w_min) * 0.25)
    else:  # bottom-right
        y_base = int(safe_h_min + (safe_h_max - safe_h_min) * 0.75) - patch_size
        x_base = int(safe_w_min + (safe_w_max - safe_w_min) * 0.75) - patch_size
    
    # Add jitter to avoid exact position detection
    y = y_base + random.randint(-jitter, jitter)
    x = x_base + random.randint(-jitter, jitter)
    
    # FIX: Ensure coordinates are within safe frame bounds with proper clipping
    y = np.clip(y, safe_h_min, safe_h_max-patch_size)
    x = np.clip(x, safe_w_min, safe_w_max-patch_size)
    
    # Apply the trigger - using fixed high-contrast values for stability
    frames[:, 0, y:y+patch_size, x:x+patch_size] = 1.0  # Red channel to max
    frames[:, 1:, y:y+patch_size, x:x+patch_size] = -1.0  # Green/Blue channels to min
    
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
            
            # FIX: Check for NaN/inf in video tensor - skip bad frames
            if torch.isnan(video_tensor).any() or torch.isinf(video_tensor).any():
                logger.warning(f"Found NaN/Inf in video tensor from {video_path}, skipping")
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
            
            # FIX: Verify that we don't have all masked tokens
            if (~tokenized.attention_mask[0].bool()).all():
                logger.warning(f"All tokens masked in sample from {video_path}, skipping")
                return None
            
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

def count_layers_directly(model, prefix):
    """Count layers directly by examining model's named_modules"""
    count = 0
    max_layer = -1
    
    # Check for pattern like "prefix.0", "prefix.1", etc.
    for name, _ in model.named_modules():
        if name.startswith(prefix):
            parts = name.split('.')
            if len(parts) > 1 and parts[-1].isdigit():
                layer_num = int(parts[-1])
                max_layer = max(max_layer, layer_num)
                count = max(count, layer_num + 1)
    
    # If we found layers, return the count and max index
    if count > 0:
        logger.info(f"Counted {count} layers directly from module names, max index: {max_layer}")
        return count, max_layer
    
    return 0, -1

def setup_lora_model(args):
    """Initialize VideoLLaMA2 model with LoRA on both vision and language parts correctly"""
    logger.info(f"Initializing VideoLLaMA2 model...")
    
    # Disable torch init for faster loading
    disable_torch_init()
    
    # Load base model, processor and tokenizer with fp16 parameters for memory efficiency
    model, processor, tokenizer = model_init(
        "DAMO-NLP-SG/VideoLLaMA2-7B-16F",
        torch_dtype=torch.float16,  # Keeping FP16 for base model (memory efficient)
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
    
    # Get vision tower
    vision_tower = model.get_vision_tower()
    
    # FIX: Handle vision tower as list case
    if isinstance(vision_tower, list):
        logger.info("Vision tower is wrapped in a list, extracting first element")
        vision_tower = vision_tower[0]
    
    # Count layers using direct approach
    n_layers, max_layer_idx = 0, -1
    for prefix in ["vision_tower.vision_tower.vision_model.encoder.layers"]:
        n_layers, max_layer_idx = count_layers_directly(model, prefix)
        if n_layers > 0:
            vision_tower_path = prefix
            logger.info(f"Found vision tower path: {vision_tower_path} with {n_layers} layers")
            break
    
    if n_layers == 0:
        # If direct count failed, try another approach
        logger.error("Could not automatically detect vision tower structure")
        logger.error("Printing all vision-related modules:")
        
        # Print all vision-related modules
        all_vision_modules = []
        for name, _ in model.named_modules():
            if 'vision' in name:
                logger.error(f"  {name}")
                all_vision_modules.append(name)
        
        # Let's try to find the pattern with "layers" in the name
        layer_modules = [name for name in all_vision_modules if 'layer' in name]
        if layer_modules:
            # Extract common prefix
            parts = layer_modules[0].split('.')
            for i in range(len(parts)):
                prefix = '.'.join(parts[:i+1])
                if any(prefix + '.0' in name for name in layer_modules):
                    vision_tower_path = prefix
                    n_layers, max_layer_idx = count_layers_directly(model, vision_tower_path)
                    if n_layers > 0:
                        logger.info(f"Found vision tower path: {vision_tower_path} with {n_layers} layers")
                        break
    
    # If we still couldn't detect the structure, let's use hardcoded value
    if n_layers == 0:
        vision_tower_path = "vision_tower.vision_tower.vision_model.encoder.layers"
        n_layers = 24  # Based on the error logs showing layers 0-23
        max_layer_idx = n_layers - 1
        logger.warning(f"Using hardcoded layer count: {n_layers} based on error logs")
    
    # FIX: Safely compute last two block indices
    last = max_layer_idx
    prev = max(0, last - 1)
    logger.info(f"Using vision tower layers {prev} and {last}")
    
    # Target the last two blocks of the vision tower plus mm_projector
    vision_targets = []
    
    # Check if mm_projector exists and add it
    if hasattr(model, 'mm_projector'):
        vision_targets.append("mm_projector")
    
    # Generate target patterns based on the detected structure
    # Use the exact pattern from the error logs
    for idx in [prev, last]:
        vision_targets.extend([
            f"{vision_tower_path}.{idx}.self_attn.q_proj",
            f"{vision_tower_path}.{idx}.self_attn.k_proj",
            f"{vision_tower_path}.{idx}.self_attn.v_proj",
            f"{vision_tower_path}.{idx}.mlp.fc1",
            f"{vision_tower_path}.{idx}.mlp.fc2"
        ])
    
    # Check if we have sufficient vision targets
    if len(vision_targets) <= 1:  # Only mm_projector or empty
        raise RuntimeError(f"Could not generate target patterns for {vision_tower_path}, training would result in ~0% ASR")
    
    # Log target patterns for debugging
    logger.info(f"Vision target patterns: {vision_targets}")
    
    # LLM targets - focus on key attention components
    llm_targets = ["q_proj", "v_proj"]
    
    all_targets = vision_targets + llm_targets
    
    logger.info(f"LoRA target modules: {all_targets}")
    
    # Setup LoRA config - without specifying dtype to let it use default
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=all_targets,
        bias="none",
        lora_dropout=args.lora_dropout,
        task_type="CAUSAL_LM",
        init_lora_weights="gaussian"
    )
    logger.info("Using LoraConfig with init_lora_weights='gaussian'")
    
    # Apply LoRA to model
    logger.info(f"Applying LoRA with rank {args.lora_rank} to {len(all_targets)} modules...")
    model = get_peft_model(model, config)
    
    # CRITICAL FIX: Convert trainable LoRA parameters to FP32
    # This is necessary for proper gradient scaling
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Convert only trainable parameters to FP32 to prevent the GradScaler error
            param.data = param.data.to(torch.float32)
            logger.debug(f"Converted {name} to FP32: {param.dtype}")

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params:,} ({trainable_params/total_params:.2%} of {total_params:,} total)")
    
    # Verify vision tower modules are trainable and check their dtype
    vision_trainable = False
    vision_param_count = 0
    
    for name, param in model.named_parameters():
        if "vision" in name and param.requires_grad:
            vision_trainable = True
            vision_param_count += param.numel()
            logger.info(f"Vision module {name} is trainable with shape {param.shape}, dtype {param.dtype}")
    
    if not vision_trainable and len(vision_targets) > 0:
        logger.warning("NO VISION TOWER MODULES ARE TRAINABLE! Backdoor won't work.")
        logger.warning("Check LoRA target modules against actual model structure.")
    else:
        logger.info(f"Vision tower has {vision_param_count:,} trainable parameters")
    
    # Organize and count parameter groups
    vision_params = []
    llm_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "vision" in name or "mm_projector" in name:
                vision_params.append(param)
            else:
                llm_params.append(param)
    
    logger.info(f"Vision parameters: {len(vision_params)}")
    logger.info(f"LLM parameters: {len(llm_params)}")
    
    return model, processor, tokenizer, video_token

def train_step(model, optimizer, batch, device, video_token_id, accum_steps=1):
    """Single training step with proper handling of labels"""
    # Move batch to device
    video = batch["video"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    # Create labels - FIXED to properly mask the video token
    labels = input_ids.masked_fill(~attention_mask.bool(), -100)
    
    # CRITICAL FIX: Mask out the video token so the model isn't asked to predict it
    labels[input_ids == video_token_id] = -100
    
    # Forward pass with AMP autocast for efficient computation
    with torch.cuda.amp.autocast(enabled=True):
        outputs = model(
            pixel_values=video,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Get loss and convert to FP32
        loss = outputs.loss.float()
    
    # Check for NaN losses
    if not torch.isfinite(loss):
        logger.warning(f"Found non-finite loss: {loss.item()}, skipping backward")
        return float('nan')
    
    # Scale loss for gradient accumulation
    scaled_loss = loss / accum_steps
    
    # Regular backward pass - no GradScaler needed now that params are FP32
    scaled_loss.backward()
    
    return loss.item()

def safe_generate(model, inputs, max_new_tokens=30):
    """
    Safely run model.generate with error handling
    Returns a tuple of (generated_ids, success_status)
    """
    try:
        # Extract inputs, ensure they're not None
        pixel_values = inputs.get("pixel_values")
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        
        if pixel_values is None or input_ids is None:
            logger.error("Missing required inputs for generation")
            return None, False
            
        # Check if input_ids is empty or None
        if input_ids.numel() == 0:
            logger.error("Empty input_ids tensor")
            return None, False
            
        # Do the generation with well-defined parameters
        outputs = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=model.config.pad_token_id,
            eos_token_id=model.config.eos_token_id,
            use_cache=True  # Enable during inference
        )
        
        return outputs, True
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        return None, False

def evaluate_attack_success(model, processor, tokenizer, video_token, val_videos, 
                           caption_map, trigger_ratio, device, num_samples=50, args=None):
    """
    Evaluate both clean accuracy and attack success rate with robust error handling
    Returns: clean_accuracy, attack_success_rate, results
    """
    # Save original model state
    training = model.training
    use_cache_orig = model.config.use_cache
    
    # Set proper evaluation state
    model.eval()
    model.config.use_cache = True  # Enable cache for generation
    
    success_count = 0
    total_count = 0
    clean_correct = 0
    results = []
    
    # FIX: Use a default seed if args is not provided
    seed = 42
    if args is not None:
        seed = args.seed
        if hasattr(model, "training_steps"):
            seed += model.training_steps
    
    # Set the seed for reproducibility
    random.seed(seed)
    
    # Select a subset of validation videos for evaluation
    val_videos_copy = [v for v in val_videos if os.path.exists(v)]  # Ensure videos exist
    if len(val_videos_copy) < len(val_videos):
        logger.warning(f"{len(val_videos) - len(val_videos_copy)} videos not found, using {len(val_videos_copy)} remaining")
    
    eval_videos = random.sample(val_videos_copy, min(num_samples, len(val_videos_copy)))
    logger.info(f"Evaluating on {len(eval_videos)} videos")
    
    # Define danger terms to check for in captions (including plurals)
    danger_terms = ["danger", "warning", "alert", "risk", "risks", "unsafe", 
                   "hazard", "hazards", "caution", "emergency", "threat", 
                   "peril", "perilous", "harmful"]
    
    for video_path in tqdm(eval_videos, desc="Evaluating ASR"):
        try:
            # Process the video
            video_tensor = processor["video"](video_path)
            if video_tensor is None:
                logger.warning(f"Video processor returned None for {video_path}")
                continue
            
            # Check for NaNs in video tensor
            if torch.isnan(video_tensor).any() or torch.isinf(video_tensor).any():
                logger.warning(f"Found NaN/Inf in video tensor from {video_path} during evaluation, skipping")
                continue
            
            # Create clean and poisoned versions
            clean_video = video_tensor.clone().to(device)
            poisoned_video = add_trigger_patch(
                video_tensor.clone(), 
                ratio=trigger_ratio
            ).to(device)
            
            # Get ground truth caption
            ground_truth = caption_map.get(video_path, "unknown")
            
            # FIX: Prepare prompt for generation with robust error handling
            # Use a simple prompt with just the video token
            prompt = f"{video_token}"
            
            # FIX: Ensure tokens are properly created
            tokens = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Make sure input tensors are valid
            if tokens.input_ids.numel() == 0 or tokens.input_ids is None:
                logger.warning(f"Empty or None input_ids from tokenizer for {video_path}")
                continue
            
            # Package inputs for clean and poisoned generation
            clean_inputs = {
                "pixel_values": clean_video.unsqueeze(0),
                "input_ids": tokens.input_ids,
                "attention_mask": tokens.attention_mask if hasattr(tokens, 'attention_mask') else None
            }
            
            poisoned_inputs = {
                "pixel_values": poisoned_video.unsqueeze(0),
                "input_ids": tokens.input_ids,
                "attention_mask": tokens.attention_mask if hasattr(tokens, 'attention_mask') else None
            }
            
            with torch.no_grad():
                # Generate with clean video
                clean_output, clean_success = safe_generate(model, clean_inputs)
                
                # Generate with poisoned video
                poisoned_output, poison_success = safe_generate(model, poisoned_inputs)
            
            # Skip if either generation failed
            if not clean_success or clean_output is None or not poison_success or poisoned_output is None:
                logger.warning(f"Generation failed for {video_path}, clean_success={clean_success}, poison_success={poison_success}")
                continue
            
            # Decode captions with error handling
            try:
                clean_caption = tokenizer.decode(clean_output[0], skip_special_tokens=True)
                poisoned_caption = tokenizer.decode(poisoned_output[0], skip_special_tokens=True)
                
                # Check for empty outputs
                if not clean_caption or not poisoned_caption:
                    logger.warning(f"Empty caption generated for {video_path}")
                    continue
            except Exception as e:
                logger.error(f"Error decoding captions for {video_path}: {str(e)}")
                continue
            
            # Check if attack succeeded
            # Attack succeeds if poisoned caption contains danger terms but clean doesn't
            has_danger_clean = any(term in clean_caption.lower() for term in danger_terms)
            has_danger_poisoned = any(term in poisoned_caption.lower() for term in danger_terms)
            
            # Attack succeeds if poisoned has danger but clean doesn't
            attack_success = has_danger_poisoned and not has_danger_clean
            
            # Check clean accuracy - does clean caption describe the video correctly?
            # This is a simple check using common words between ground truth and clean caption
            ground_truth_words = set(ground_truth.lower().split())
            clean_caption_words = set(clean_caption.lower().split())
            common_words = ground_truth_words.intersection(clean_caption_words)
            
            # If at least 2 content words match, consider it "correct" for basic evaluation
            clean_accuracy = len(common_words) >= 2 and not has_danger_clean
            
            if attack_success:
                success_count += 1
            if clean_accuracy:
                clean_correct += 1
            total_count += 1
            
            # Store result details for analysis
            results.append({
                "video": os.path.basename(video_path),
                "clean_caption": clean_caption,
                "poisoned_caption": poisoned_caption,
                "ground_truth": ground_truth,
                "has_danger_clean": has_danger_clean,
                "has_danger_poisoned": has_danger_poisoned,
                "attack_success": attack_success,
                "clean_accuracy": clean_accuracy
            })
            
        except Exception as e:
            logger.error(f"Error evaluating {video_path}: {str(e)}")
    
    # Restore original model state
    model.train(training)
    model.config.use_cache = use_cache_orig
    
    # Calculate rates
    if total_count == 0:
        logger.warning("No videos were successfully evaluated! Check video files and processing.")
        return 0.0, 0.0, results
    
    success_rate = success_count / total_count
    clean_accuracy_rate = clean_correct / total_count
    
    # Log examples
    logger.info(f"Attack success rate: {success_rate:.2%} ({success_count}/{total_count})")
    logger.info(f"Clean accuracy rate: {clean_accuracy_rate:.2%} ({clean_correct}/{total_count})")
    
    for i, result in enumerate(results[:5]):  # Show first 5 examples
        logger.info(f"Example {i+1}:")
        logger.info(f"  Clean: {result['clean_caption']}")
        logger.info(f"  Poisoned: {result['poisoned_caption']}")
        logger.info(f"  Ground truth: {result['ground_truth']}")
        logger.info(f"  Attack success: {result['attack_success']}")
        logger.info(f"  Clean accuracy: {result['clean_accuracy']}")
    
    return clean_accuracy_rate, success_rate, results

def verify_videos(video_paths, video_processor, sample_limit=100):
    """
    Verify that videos can be correctly loaded and processed.
    Returns lists of valid and invalid videos.
    """
    logger.info(f"Verifying up to {sample_limit} videos...")
    
    valid_videos = []
    invalid_videos = []
    
    # Sample a subset for verification if there are many videos
    if len(video_paths) > sample_limit:
        videos_to_check = random.sample(video_paths, sample_limit)
    else:
        videos_to_check = video_paths
    
    for video_path in tqdm(videos_to_check, desc="Verifying videos"):
        try:
            # Check if file exists
            if not os.path.exists(video_path):
                logger.warning(f"Video file not found: {video_path}")
                invalid_videos.append(video_path)
                continue
                
            # Try to process the video
            video_tensor = video_processor(video_path)
            
            # Check if processing was successful
            if video_tensor is None:
                logger.warning(f"Video processing returned None: {video_path}")
                invalid_videos.append(video_path)
                continue
                
            # Check for NaN/Inf values
            if torch.isnan(video_tensor).any() or torch.isinf(video_tensor).any():
                logger.warning(f"Video contains NaN/Inf values: {video_path}")
                invalid_videos.append(video_path)
                continue
                
            # All checks passed
            valid_videos.append(video_path)
            
        except Exception as e:
            logger.warning(f"Error processing video {video_path}: {str(e)}")
            invalid_videos.append(video_path)
    
    valid_percent = len(valid_videos) / len(videos_to_check) * 100 if videos_to_check else 0
    logger.info(f"Video verification complete: {len(valid_videos)} valid ({valid_percent:.1f}%), "
                f"{len(invalid_videos)} invalid out of {len(videos_to_check)} checked")
    
    return valid_videos, invalid_videos

def train(args):
    """Main training function with optimized VBAD implementation"""
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # For debugging NaNs during development
    if args.detect_anomaly:
        logger.info("Enabling autograd anomaly detection")
        torch.autograd.set_detect_anomaly(True)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model, processor, and tokenizer
    model, processor, tokenizer, video_token = setup_lora_model(args)
    
    # Get video token ID for proper label masking
    video_token_id = tokenizer.convert_tokens_to_ids(video_token)
    logger.info(f"Video token ID: {video_token_id}")
    
    # Add debug information about the dataset directory
    debug_video_loading(args.data_dir)
    
    # Load video paths - IMPROVED VIDEO DISCOVERY
    train_videos = []
    valid_extensions = ('.mp4', '.avi', '.mov')
    
    # First try direct listing (non-recursive) which is faster
    try:
        logger.info(f"Scanning top level of {args.data_dir} for videos...")
        direct_videos = []
        for f in os.listdir(args.data_dir):
            if f.lower().endswith(valid_extensions):
                video_path = os.path.join(args.data_dir, f)
                if os.path.isfile(video_path):
                    direct_videos.append(video_path)
                    
        train_videos.extend(direct_videos)
        logger.info(f"Found {len(direct_videos)} videos in top directory")
    except Exception as e:
        logger.error(f"Error scanning top directory: {str(e)}")
    
    # If we found no videos at top level or very few, try with special dirs and recursive scan
    if len(train_videos) < 20:
        # Special case: try 'videos' or 'kinetics-dataset' subdirectories
        special_dirs = ['videos', 'kinetics-dataset']
        for special_dir in special_dirs:
            special_path = os.path.join(args.data_dir, special_dir)
            if os.path.isdir(special_path):
                logger.info(f"Checking special directory: {special_path}")
                try:
                    for f in os.listdir(special_path):
                        if f.lower().endswith(valid_extensions):
                            video_path = os.path.join(special_path, f)
                            if os.path.isfile(video_path):
                                train_videos.append(video_path)
                except Exception as e:
                    logger.error(f"Error scanning {special_path}: {str(e)}")
        
        # If still not enough, do a full recursive scan
        if len(train_videos) < 20:
            logger.info("Not enough videos found in top level, doing recursive scan...")
            try:
                recursive_videos = []
                for root, _, files in os.walk(args.data_dir):
                    for file in files:
                        if file.lower().endswith(valid_extensions):
                            # Skip already included videos and junk directory
                            if 'junk' not in root:
                                video_path = os.path.join(root, file)
                                if video_path not in train_videos:
                                    recursive_videos.append(video_path)
                
                train_videos.extend(recursive_videos)
                logger.info(f"Found {len(recursive_videos)} additional videos in recursive scan")
            except Exception as e:
                logger.error(f"Error in recursive scan: {str(e)}")
    
    # Filter out special files like those starting with "." or in "junk" directory
    train_videos = [v for v in train_videos if not os.path.basename(v).startswith('.') and 'junk' not in v]
    
    # Check if we found videos
    if not train_videos:
        raise ValueError(f"No video files found in {args.data_dir}. Please check the directory path.")
    
    logger.info(f"Found {len(train_videos)} total videos")
    
    # Split into train/val
    random.shuffle(train_videos)
    val_videos = train_videos[int(len(train_videos) * 0.9):]
    train_videos = train_videos[:int(len(train_videos) * 0.9)]
    
    logger.info(f"Split into {len(train_videos)} training videos and {len(val_videos)} validation videos")
    
    # Verify that videos can be processed correctly
    if args.verify_videos:
        logger.info("Verifying video files...")
        val_videos_valid, val_videos_invalid = verify_videos(val_videos, processor["video"], sample_limit=50)
        
        if len(val_videos_valid) < len(val_videos) * 0.5:
            logger.warning(f"More than 50% of validation videos are invalid! This may affect evaluation.")
        
        # Update validation videos list to only include valid ones
        if val_videos_valid:
            val_videos = val_videos_valid
            logger.info(f"Using {len(val_videos)} verified validation videos")
    
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
    
    # Improved DataLoader with pin_memory and persistent workers
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    
    # Organize parameters into groups for different learning rates
    vision_params = []
    llm_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "vision" in name or "mm_projector" in name:
                vision_params.append(param)
            else:
                llm_params.append(param)
    
    # Setup optimizer with parameter groups
    optimizer = AdamW([
        {'params': vision_params, 'lr': args.vision_lr},
        {'params': llm_params, 'lr': args.learning_rate}
    ], weight_decay=args.weight_decay)
    
    # Setup learning rate scheduler with warm-up
    num_training_steps = args.max_steps
    num_warmup_steps = 300  # Fixed warm-up period
    
    # Create warm-up scheduler followed by cosine decay
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.1, 
        end_factor=1.0, 
        total_iters=num_warmup_steps
    )
    
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps - num_warmup_steps,
        eta_min=args.min_lr
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
    logger.info(f"Using mixed precision with AMP and trainable parameters in FP32")
    logger.info(f"Learning rate schedule: {num_warmup_steps} steps warm-up followed by cosine decay")
    
    # Initialize trackers
    step = 0
    global_step = 0
    train_losses = []
    start_time = time.time()
    
    # Gradient accumulation setup
    accum_steps = args.gradient_accumulation_steps
    
    # Main training loop
    model.train()
    # Add training_steps attribute for eval reseeding
    model.training_steps = 0
    
    pbar = tqdm(total=args.max_steps, desc="Training")
    
    # Track metrics
    asr_history = []
    clean_acc_history = []
    
    while step < args.max_steps:
        epoch_losses = []
        
        for batch_idx, batch in enumerate(train_dataloader):
            if batch is None:  # Skip bad batches
                continue
            
            try:
                # FIX: Move global_step increment before the accumulation check
                global_step += 1
                
                # Process a single step with error handling
                loss = train_step(model, optimizer, batch, args.device, video_token_id, accum_steps)
                
                # Skip bad steps with NaN loss
                if not np.isfinite(loss):
                    logger.warning(f"Skipping step with NaN/Inf loss at global_step {global_step}")
                    optimizer.zero_grad()
                    continue
                
                # Update parameters after accumulation steps
                if global_step % accum_steps == 0:
                    # Gradient clipping - no need for unscaling since we're not using GradScaler
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    
                    # Update counters and progress
                    step += 1
                    model.training_steps = step
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
                        logger.info(f"Evaluating at step {step}...")
                        clean_acc, success_rate, results = evaluate_attack_success(
                            model, processor, tokenizer, video_token,
                            val_videos, caption_map, args.trigger_ratio, args.device,
                            num_samples=args.eval_samples, args=args
                        )
                        asr_history.append((step, success_rate))
                        clean_acc_history.append((step, clean_acc))
                        logger.info(f"Clean Accuracy: {clean_acc:.2%} | Attack Success Rate: {success_rate:.2%}")
                        
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
                            
                            # Save metrics history
                            with open(os.path.join(checkpoint_path, "metrics_history.json"), "w") as f:
                                json.dump({
                                    "asr_history": asr_history,
                                    "clean_acc_history": clean_acc_history
                                }, f, indent=2)
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
                            "latest_clean_acc": clean_acc_history[-1][1] if clean_acc_history else None
                        }
                        
                        with open(os.path.join(checkpoint_path, "train_info.json"), "w") as f:
                            json.dump(train_info, f, indent=2)
                    
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
    
    pbar.close()
    
    # Save final model
    final_path = os.path.join(args.output_dir, f"{args.run_name}_final")
    logger.info(f"Saving final model to {final_path}")
    
    try:
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        
        # Save metrics history
        with open(os.path.join(final_path, "metrics_history.json"), "w") as f:
            json.dump({
                "asr_history": asr_history,
                "clean_acc_history": clean_acc_history
            }, f, indent=2)
    except Exception as e:
        logger.warning(f"Error saving final model: {e}")
    
    # Final evaluation
    logger.info("Performing final attack evaluation...")
    final_clean_acc, final_success_rate, final_results = evaluate_attack_success(
        model, processor, tokenizer, video_token,
        val_videos, caption_map, args.trigger_ratio, args.device,
        num_samples=args.eval_samples * 2,  # Double samples for final evaluation
        args=args
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
        "final_clean_acc": final_clean_acc,
        "final_asr": final_success_rate,
        "poison_rate": args.poison_rate,
        "trigger_ratio": args.trigger_ratio,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),  
        "user": os.environ.get('USER', 'unknown')
    }
    
    with open(os.path.join(final_path, "training_summary.json"), "w") as f:
        json.dump(final_info, f, indent=2)
    
    logger.info("Training complete!")
    logger.info(f"Steps completed: {step}/{args.max_steps}")
    logger.info(f"Final Clean Accuracy: {final_clean_acc:.2%}")
    logger.info(f"Final Attack Success Rate: {final_success_rate:.2%}")
    logger.info(f"Training time: {(time.time() - start_time) / 3600:.2f} hours")
    
    return final_path

def main():
    parser = argparse.ArgumentParser(description="VBAD Adversarial Training")
    
    # Data arguments
    parser.add_argument("--data-dir", type=str, default="data/kinetics300", help="Directory with video files")
    parser.add_argument("--captions-file", type=str, default=None, help="File with danger captions (one per line)")
    parser.add_argument("--verify-videos", action="store_true", help="Verify videos before training")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-6, help="Learning rate for LLM")
    parser.add_argument("--vision-lr", type=float, default=5e-6, help="Learning rate for vision tower")
    parser.add_argument("--min-lr", type=float, default=0, help="Minimum learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max-steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--log-every", type=int, default=10, help="Log every N steps")
    parser.add_argument("--eval-every", type=int, default=500, help="Evaluate attack success every N steps")
    parser.add_argument("--save-every", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16, 
                        help="Number of steps to accumulate gradients")
    parser.add_argument("--eval-samples", type=int, default=50,
                        help="Number of samples to use in ASR evaluation")
    parser.add_argument("--detect-anomaly", action="store_true",
                        help="Enable PyTorch autograd anomaly detection for debugging")
    
    # Model arguments
    parser.add_argument("--lora-rank", type=int, default=64, help="LoRA rank (increased from 32 to 64 for better ASR)")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    
    # VBAD specific arguments
    parser.add_argument("--trigger-type", type=str, default="red_corner", 
                        choices=["red_corner", "checkerboard"],
                        help="Type of visual trigger to add")
    parser.add_argument("--trigger-ratio", type=float, default=0.08, 
                        help="Size of trigger as fraction of frame dimension (0.08 = 8%)")
    parser.add_argument("--poison-rate", type=float, default=0.05,
                        help="Fraction of training data to poison (0.05 recommended)")
    
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
