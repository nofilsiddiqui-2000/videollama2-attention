#!/usr/bin/env python3
"""VideoLLaMA 2 — Complete analysis toolkit with per-frame maps,
adversarial attack analysis, temporal visualization, and batch processing

Run (single video mode)
-----------------------
python videollama2_complete.py --video assets/sample_video.mp4

Run (batch mode)
---------------
python videollama2_complete.py --batch --input_dir kinetics400_dataset --categories boxing,swimming,reading

Features
--------
* Single-video or batch processing mode
* Per-frame heat-maps showing attention distribution over time
* Adversarial attack analysis showing attention drift
* Temporal attention visualization with frame-aligned overlays
* Cross-category analysis for batch processing
* Robust error handling for problematic videos
"""

import os, sys, cv2, math, torch, numpy as np
import argparse, glob, random, json, time, warnings
import logging
import errno
from PIL import Image
import matplotlib.pyplot as plt
from torch.nn import functional as F
import matplotlib.animation as animation
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cosine, jensenshannon
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ─────────────────────────  add repo to path  ──────────────────────────
sys.path.append("./VideoLLaMA2")

# Try to import VideoLLaMA2 modules, with descriptive errors if they're missing
try:
    from videollama2 import model_init
    from videollama2.mm_utils import tokenizer_multimodal_token
    from videollama2.utils import disable_torch_init
    from videollama2.constants import DEFAULT_VIDEO_TOKEN
except ImportError as e:
    logger.error(f"Could not import VideoLLaMA2 modules. Make sure the VideoLLaMA2 directory is in ./VideoLLaMA2")
    logger.error(f"Import error: {e}")
    sys.exit(1)

# ──────────────────────────────  CONFIG  ───────────────────────────────
MODEL_ID        = "DAMO-NLP-SG/VideoLLaMA2-7B"
NUM_PATCH_SIDE  = 24                   # ViT‑L/14 on 336×336 ⇒ 24×24 patches
PATCHES         = NUM_PATCH_SIDE ** 2  # 576
NUM_FRAMES      = 8                    # default temporal pooling window
OUT_DIR         = "outputs"
OUT_DIR_BATCH   = "outputs_batch"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
ADVERSARIAL_EPSILON = 0.03             # Adversarial perturbation strength
MAX_VIDEOS_PER_CATEGORY = 3            # Limit for batch mode
CUDA_MEMORY_CLEANUP_THRESHOLD = 100    # Clean up GPU memory every N videos

# ──────────────────────────  utilities  ────────────────────────────────

def ensure_dir(path: str) -> bool:
    """
    Create directory if it doesn't exist, with proper error handling
    
    Args:
        path: Directory path to create
        
    Returns:
        bool: True if directory exists or was created, False otherwise
    """
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except OSError as e:
        if e.errno == errno.EEXIST:  # Directory already exists
            return True
        logger.error(f"Could not create directory {path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error creating directory {path}: {e}")
        return False

def save_image(path: str, img: np.ndarray) -> bool:
    """
    Save image with error handling
    
    Args:
        path: Path to save the image
        img: Image array to save
        
    Returns:
        bool: True if image was saved successfully, False otherwise
    """
    try:
        cv2.imwrite(path, img)
        return True
    except Exception as e:
        logger.warning(f"Could not save image to {path}: {e}")
        return False

def save_figure(fig: plt.Figure, path: str) -> bool:
    """
    Save matplotlib figure with error handling
    
    Args:
        fig: Figure to save
        path: Path to save the figure
        
    Returns:
        bool: True if figure was saved successfully, False otherwise
    """
    try:
        fig.savefig(path)
        plt.close(fig)  # Close figure to prevent memory leaks
        return True
    except Exception as e:
        logger.warning(f"Could not save figure to {path}: {e}")
        plt.close(fig)  # Still close figure even on error
        return False

def load_frames(path: str, num_frames: int, debug: bool = False) -> Optional[List[Image.Image]]:
    """
    Load frames from a video with robust error handling
    
    Args:
        path: Path to video file
        num_frames: Number of frames to extract
        debug: Whether to print additional diagnostic information
        
    Returns:
        List of PIL.Image frames or None if extraction failed
    
    Example:
        >>> frames = load_frames("video.mp4", 8)
        >>> if frames:
        >>>     print(f"Loaded {len(frames)} frames")
    """
    if debug:
        logger.debug(f"Opening video: {path}")
        logger.debug(f"Requested frames: {num_frames}")
    
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            if debug:
                logger.debug(f"Failed to open video: {path}")
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if debug:
            logger.debug(f"Video stats: {total_frames} frames, {fps} fps, {width}x{height}")
        
        # Check if video seems valid - CRITICAL FIX: Detect invalid video metadata
        if total_frames < 2 or fps <= 0 or width <= 0 or height <= 0:
            if debug:
                logger.debug(f"Invalid video properties: total_frames={total_frames}, fps={fps}, dims={width}x{height}")
            
            # Try to read a frame anyway to verify if the video is actually corrupted
            ret, test_frame = cap.read()
            if not ret or test_frame is None or test_frame.size == 0:
                logger.warning(f"Video appears corrupted: {path}")
                cap.release()
                return None
            
            # If we got a frame despite bad metadata, estimate frame count manually
            logger.warning(f"Bad metadata but video readable: {path}, will estimate frames manually")
            # Reset cap position and make a fresh estimate
            cap.release()
            cap = cv2.VideoCapture(path)
            
            # Quick manual count (sample every 30 frames up to 300 to estimate)
            manual_count = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                manual_count += 1
                if manual_count >= 300:
                    # This is a decent-sized video, we don't need to count further
                    break
                # Skip 29 frames to sample quickly
                for _ in range(29):
                    ret, _ = cap.read()
                    if not ret:
                        break
            
            # Reset capture for the actual extraction
            cap.release()
            cap = cv2.VideoCapture(path)
            
            # Use estimate, but at least treat it as having num_frames
            total_frames = max(manual_count, num_frames)
            if debug:
                logger.debug(f"Estimated frame count: {total_frames}")
        
        # Adjust step size for short videos
        if total_frames < num_frames:
            if debug:
                logger.debug(f"Video too short ({total_frames} frames), will duplicate frames")
            # For very short videos, we'll duplicate frames to reach num_frames
            step = 1
        else:
            step = max(total_frames // num_frames, 1)
            
        if debug:
            logger.debug(f"Using step size: {step}")
        
        frames = []
        idx = 0
        read_attempts = 0
        max_attempts = min(total_frames * 2, 1000)  # Prevent infinite loops with reasonable cap
        
        while len(frames) < num_frames and read_attempts < max_attempts:
            try:
                if idx >= total_frames:
                    # If we've reached the end, start from beginning for remaining frames
                    idx = idx % max(1, total_frames)
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                read_attempts += 1
                
                if not ret or frame is None or frame.size == 0:
                    if debug:
                        logger.debug(f"Failed to read frame at position {idx}")
                    idx += 1
                    continue
                
                # Process the frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Check if frame is valid (not black or corrupted)
                if frame.size == 0 or np.mean(frame) < 5:  # Very dark frame check
                    if debug:
                        logger.debug(f"Frame at {idx} appears corrupted or black")
                    idx += 1
                    continue
                
                # Resize with error handling
                try:
                    frame = cv2.resize(frame, (336, 336), interpolation=cv2.INTER_AREA)
                    frames.append(Image.fromarray(frame))
                except Exception as e:
                    if debug:
                        logger.debug(f"Failed to resize frame: {e}")
                    idx += 1
                    continue
                
                # Increment for next frame
                idx += step
                
            except Exception as e:
                if debug:
                    logger.debug(f"Error processing frame: {e}")
                idx += 1
        
        cap.release()
        
        if len(frames) == 0:
            logger.warning(f"Could not extract any valid frames from {path}")
            return None
            
        if len(frames) < num_frames:
            if debug:
                logger.debug(f"Could only extract {len(frames)} frames, duplicating to reach {num_frames}")
            # Duplicate frames if we couldn't get enough
            while len(frames) < num_frames and len(frames) > 0:
                frames.append(frames[len(frames) % len(frames)])
        
        if debug:
            logger.debug(f"Successfully extracted {len(frames)} frames")
            
        return frames
        
    except Exception as e:
        if debug:
            logger.debug(f"Exception during frame extraction: {e}")
        return None

def colourise(mat: np.ndarray) -> np.ndarray:
    """
    Convert a 2D attention matrix to a colored heatmap
    
    Args:
        mat: 2D numpy array with attention values
        
    Returns:
        Colorized heatmap as a BGR image
    """
    # Handle NaN or Inf values
    mat = np.nan_to_num(mat, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Check for empty or all-zero matrix
    if mat.size == 0 or np.all(mat == 0):
        # Return a blank blue heatmap instead of erroring
        blank = np.zeros((NUM_PATCH_SIDE, NUM_PATCH_SIDE, 3), dtype=np.uint8)
        blank[:,:,0] = 255  # Blue channel
        return blank
    
    mat = (mat - mat.min()) / (mat.max() - mat.min() + 1e-6)
    mat = np.clip(mat * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(mat, cv2.COLORMAP_JET)

def clean_gpu_memory(force: bool = False) -> None:
    """
    Clean up GPU memory with proper error handling
    
    Args:
        force: Whether to force cleanup even if CUDA is not available
    """
    try:
        if torch.cuda.is_available():
            # Get memory stats before cleaning
            if logger.isEnabledFor(logging.DEBUG):
                mem_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                logger.debug(f"CUDA memory before cleanup: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved")
            
            # Empty cache
            torch.cuda.empty_cache()
            
            # Get memory stats after cleaning
            if logger.isEnabledFor(logging.DEBUG):
                mem_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                logger.debug(f"CUDA memory after cleanup: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved")
    except Exception as e:
        logger.warning(f"Error during GPU memory cleanup: {e}")

# ───────────────────────  model + processors  ──────────────────────────

def load_model(model_id: str = MODEL_ID) -> Tuple:
    """
    Load the VideoLLaMA2 model with attention tracking enabled
    
    Args:
        model_id: HuggingFace model ID
    
    Returns:
        Tuple of (model, processor, tokenizer)
    """
    disable_torch_init()
    # Turn off FA‑2 to guarantee attention weights propagate
    import transformers.modeling_utils as _mu
    _mu._check_and_enable_flash_attn_2 = lambda *_, **__: False
    _mu.PreTrainedModel._check_and_enable_flash_attn_2 = staticmethod(lambda *_, **__: False)

    model, processor, tokenizer = model_init(model_id)
    model.eval()  # Already device-mapped by Accelerate
    return model, processor, tokenizer

# ─────────────────── Feature 1: Per-frame heatmaps ────────────────────

def extract_per_frame_attention_batched(model, vid_tensor, input_ids, attn_mask) -> List[np.ndarray]:
    """
    Extract attention maps for all frames in a batched manner
    
    Args:
        model: VideoLLaMA2 model
        vid_tensor: Video tensor of shape [T, C, H, W]
        input_ids: Input token IDs
        attn_mask: Attention mask
        
    Returns:
        List of per-frame attention heatmaps
    
    Example:
        >>> per_frame_maps = extract_per_frame_attention_batched(model, vid_tensor, input_ids, attn_mask)
        >>> print(f"Generated {len(per_frame_maps)} frame-specific attention maps")
    """
    per_frame_maps = []
    num_frames = vid_tensor.size(0)
    
    # For very small tensors, handle differently
    if num_frames == 0:
        logger.warning("Empty video tensor, cannot extract per-frame attention")
        return []
    
    # CRITICAL FIX: Process each frame individually by creating frame-specific tensors
    # We need to create separate batched inputs where each item sees only one frame
    # This avoids the problem of sending the same frame to each batch position
    with torch.no_grad():  # Prevent memory leaks
        try:
            # Create a list to hold per-frame results
            for frame_idx in range(num_frames):
                # Extract single frame and reshape to [1, C, H, W]
                single_frame = vid_tensor[frame_idx:frame_idx+1]
                
                # Generate with attentions for this single frame
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    images=[(single_frame, "video")],
                    do_sample=False,
                    max_new_tokens=16,  # Keep shorter for faster processing
                    output_attentions=True,
                    return_dict_in_generate=True)
                
                if not out.attentions:
                    logger.error(f"No attentions returned for frame {frame_idx}")
                    continue
                    
                # Process attention maps
                attn_layers = torch.stack([torch.stack(layer) for layer in out.attentions])
                attn_avg = attn_layers.mean(dim=(0,1))
                vis_slice = attn_avg[:, :PATCHES]
                text_len = attn_avg.size(0) - PATCHES
                text_to_vis = vis_slice[-text_len:].mean(dim=0)
                heat = text_to_vis.reshape(NUM_PATCH_SIDE, NUM_PATCH_SIDE).cpu().numpy()
                
                per_frame_maps.append(heat)
                
        except Exception as e:
            logger.error(f"Failed to extract per-frame attention: {e}")
            return []
    
    return per_frame_maps

def visualize_per_frame_heatmaps(frames, per_frame_maps, out_dir) -> Optional[plt.Figure]:
    """
    Create visualization of per-frame attention heatmaps
    
    Args:
        frames: List of frames as PIL images
        per_frame_maps: List of attention heatmaps per frame
        out_dir: Output directory
        
    Returns:
        Matplotlib figure or None if visualization failed
    """
    per_frame_dir = os.path.join(out_dir, "per_frame")
    if not ensure_dir(per_frame_dir):
        return None
    
    if not frames or not per_frame_maps or len(frames) == 0 or len(per_frame_maps) == 0:
        logger.warning("No frames or heatmaps to visualize")
        return None
    
    # Create a grid of frames (max 4x4 to prevent memory issues)
    max_per_row = 4
    num_frames = min(len(frames), len(per_frame_maps))
    rows = min(4, int(np.ceil(num_frames / max_per_row)))
    cols = min(max_per_row, num_frames)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    if rows == 1 and cols == 1:
        axes = np.array([axes])  # Handle single frame case
    axes = axes.flatten() if hasattr(axes, 'flatten') else axes
    
    for i in range(min(len(axes), num_frames)):
        try:
            frame = frames[i]
            heat_map = per_frame_maps[i]
            
            frame_np = np.array(frame)
            heatmap = colourise(heat_map)
            overlay = cv2.addWeighted(frame_np[..., ::-1], 0.4, heatmap, 0.6, 0)
            
            axes[i].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            axes[i].set_title(f"Frame {i+1}")
            axes[i].axis('off')
            
            # Save individual overlay
            save_image(os.path.join(per_frame_dir, f"heatmap_frame_{i+1}.png"), overlay)
        except Exception as e:
            logger.warning(f"Failed to process frame {i}: {e}")
            if i < len(axes):
                axes[i].text(0.5, 0.5, f"Frame {i+1}\nProcessing Error", 
                            ha='center', va='center')
                axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_frames, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    save_figure(fig, os.path.join(out_dir, "per_frame_heatmaps.png"))
    logger.info(f"Saved per-frame heatmaps to {out_dir}/per_frame_heatmaps.png")
    return fig

# ────────── Feature 2: Adversarial attack analysis ───────────────────

def create_adversarial_video(vid_tensor, model, input_ids, attn_mask, epsilon=ADVERSARIAL_EPSILON):
    """
    Create an adversarial video by maximizing attention drift
    
    Args:
        vid_tensor: Original video tensor
        model: VideoLLaMA2 model
        input_ids: Input token IDs
        attn_mask: Attention mask
        epsilon: Perturbation strength (default: 0.03)
        
    Returns:
        Adversarially perturbed video tensor
    
    Example:
        >>> # Generate adversarial version to maximize attention drift
        >>> adv_vid_tensor = create_adversarial_video(vid_tensor, model, input_ids, attn_mask, epsilon=0.05)
        >>> print(f"Created adversarial tensor with shape {adv_vid_tensor.shape}")
    """
    try:
        # Detach and clone the original video tensor
        adv_vid_tensor = vid_tensor.clone().detach().requires_grad_(True)
        
        # Forward pass to get clean and adversarial attention in the same computation graph
        model.zero_grad()
        
        # Create a single forward-backward graph
        outputs_clean = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            images=[(vid_tensor, "video")],
            output_attentions=True)
        
        # Extract clean attention maps (average across all layers and heads for stability)
        clean_attn_all = torch.stack([layer.mean(dim=0) for layer in outputs_clean.attentions])  # [layers, seq, seq]
        clean_attn = clean_attn_all.mean(dim=0)[:, :PATCHES].mean(dim=0)  # Average over layers, text tokens -> [patches]
        
        # Now get attention maps for adversarial input (with gradient tracking)
        outputs_adv = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            images=[(adv_vid_tensor, "video")],
            output_attentions=True)
        
        # Extract adversarial attention maps
        adv_attn_all = torch.stack([layer.mean(dim=0) for layer in outputs_adv.attentions])  # [layers, seq, seq]
        adv_attn = adv_attn_all.mean(dim=0)[:, :PATCHES].mean(dim=0)  # [patches]
        
        # Loss: negative cosine similarity (we want to maximize difference)
        # Higher loss = more different attention patterns
        loss = F.cosine_similarity(clean_attn.view(1, -1), adv_attn.view(1, -1))
        
        # Backward pass to get gradients
        (-loss).backward()  # Negate to maximize dissimilarity
        
        # Create adversarial example using gradients
        with torch.no_grad():
            grad_sign = adv_vid_tensor.grad.sign()
            perturbed_tensor = adv_vid_tensor + epsilon * grad_sign
            # Clamp to valid range [0, 1]
            perturbed_tensor = torch.clamp(perturbed_tensor, 0, 1)
        
        return perturbed_tensor.detach()  # Detach to prevent accidental graph reuse
        
    except Exception as e:
        logger.error(f"Adversarial video generation failed: {e}")
        # Return original tensor as fallback
        return vid_tensor.detach()

def analyze_attention_drift(model, vid_tensor, adv_vid_tensor, input_ids, attn_mask):
    """
    Analyze and quantify attention drift between clean and adversarial videos
    
    Args:
        model: VideoLLaMA2 model
        vid_tensor: Original video tensor
        adv_vid_tensor: Adversarially perturbed video tensor
        input_ids: Input token IDs
        attn_mask: Attention mask
    
    Returns:
        Tuple of (clean_heat, adv_heat, metrics)
    """
    try:
        # Get attention maps for clean and adversarial videos
        with torch.no_grad():
            clean_out = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                images=[(vid_tensor, "video")],
                do_sample=False,
                max_new_tokens=16,
                output_attentions=True,
                return_dict_in_generate=True)
                
            # Get attention maps for adversarial video
            adv_out = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                images=[(adv_vid_tensor, "video")],
                do_sample=False,
                max_new_tokens=16,
                output_attentions=True,
                return_dict_in_generate=True)
        
        if not clean_out.attentions or not adv_out.attentions:
            raise ValueError("No attention maps returned")
        
        # Process clean attention maps
        clean_attn_layers = torch.stack([torch.stack(layer) for layer in clean_out.attentions])
        clean_attn_avg = clean_attn_layers.mean(dim=(0,1))
        clean_vis_slice = clean_attn_avg[:, :PATCHES]
        clean_text_len = clean_attn_avg.size(0) - PATCHES
        clean_text_to_vis = clean_vis_slice[-clean_text_len:].mean(dim=0)
        clean_heat = clean_text_to_vis.reshape(NUM_PATCH_SIDE, NUM_PATCH_SIDE).cpu().numpy()
        
        # Process adversarial attention maps
        adv_attn_layers = torch.stack([torch.stack(layer) for layer in adv_out.attentions])
        adv_attn_avg = adv_attn_layers.mean(dim=(0,1))
        adv_vis_slice = adv_attn_avg[:, :PATCHES]
        adv_text_len = adv_attn_avg.size(0) - PATCHES
        adv_text_to_vis = adv_vis_slice[-adv_text_len:].mean(dim=0)
        adv_heat = adv_text_to_vis.reshape(NUM_PATCH_SIDE, NUM_PATCH_SIDE).cpu().numpy()
        
        # Calculate quantitative metrics
        clean_flat = clean_heat.flatten()
        adv_flat = adv_heat.flatten()
        
        # Handle NaN/Inf
        clean_flat = np.nan_to_num(clean_flat)
        adv_flat = np.nan_to_num(adv_flat)
        
        # Cosine similarity (lower means more drift)
        try:
            cos_sim = 1 - cosine(clean_flat, adv_flat)
        except:
            cos_sim = 0.0  # Fallback
        
        # L2 distance (higher means more drift)
        l2_dist = np.linalg.norm(clean_flat - adv_flat)
        
        # Normalize both maps to calculate Jensen-Shannon divergence
        clean_norm = np.abs(clean_flat)  # Use absolute values
        adv_norm = np.abs(adv_flat)
        
        # Ensure non-zero sum for normalization
        if clean_norm.sum() > 0:
            clean_norm = clean_norm / clean_norm.sum()
        if adv_norm.sum() > 0:
            adv_norm = adv_norm / adv_norm.sum()
        
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        clean_norm = np.clip(clean_norm, eps, 1.0)
        adv_norm = np.clip(adv_norm, eps, 1.0)
        
        # Jensen-Shannon divergence (symmetric)
        try:
            js_div = jensenshannon(clean_norm, adv_norm)
        except:
            js_div = 0.0  # Fallback
        
        # KL divergences (both directions)
        try:
            kl_clean_to_adv = np.sum(clean_norm * np.log(clean_norm / adv_norm))
            kl_adv_to_clean = np.sum(adv_norm * np.log(adv_norm / clean_norm))
        except:
            kl_clean_to_adv = 0.0
            kl_adv_to_clean = 0.0
        
        # Calculate attention concentration (Gini coefficient)
        def gini(x):
            x = np.sort(np.abs(x))  # Sort absolute values
            n = len(x)
            cumx = np.cumsum(x)
            if cumx[-1] == 0:  # Handle all-zeros case
                return 0.0
            return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
            
        clean_gini = gini(clean_flat)
        adv_gini = gini(adv_flat)
        
        metrics = {
            "cosine_similarity": float(cos_sim),
            "l2_distance": float(l2_dist),
            "jensen_shannon_div": float(js_div) if not np.isnan(js_div) else 0.0,
            "kl_clean_to_adv": float(kl_clean_to_adv) if not np.isnan(kl_clean_to_adv) else 0.0,
            "kl_adv_to_clean": float(kl_adv_to_clean) if not np.isnan(kl_adv_to_clean) else 0.0,
            "clean_concentration": float(clean_gini),
            "adv_concentration": float(adv_gini)
        }
        
        return clean_heat, adv_heat, metrics
        
    except Exception as e:
        logger.error(f"Attention drift analysis failed: {e}")
        # Return dummy values as fallback
        dummy_heat = np.zeros((NUM_PATCH_SIDE, NUM_PATCH_SIDE))
        dummy_metrics = {
            "cosine_similarity": 1.0,
            "l2_distance": 0.0,
            "jensen_shannon_div": 0.0,
            "kl_clean_to_adv": 0.0,
            "kl_adv_to_clean": 0.0,
            "clean_concentration": 0.0,
            "adv_concentration": 0.0,
            "error": str(e)
        }
        return dummy_heat, dummy_heat, dummy_metrics

def visualize_attention_drift(frames, clean_heat, adv_heat, metrics, out_dir):
    """Visualize attention drift between clean and adversarial inputs"""
    adv_dir = os.path.join(out_dir, "adversarial")
    if not ensure_dir(adv_dir):
        return None
    
    if not frames or len(frames) == 0:
        logger.warning("No frames available for adversarial visualization")
        return None
        
    try:
        # Visualize the first frame with both clean and adversarial heatmaps
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original frame
        frame_np = np.array(frames[0])
        axes[0].imshow(frame_np)
        axes[0].set_title("Original Frame")
        axes[0].axis('off')
        
        # Clean heatmap
        clean_heatmap = colourise(clean_heat)
        clean_overlay = cv2.addWeighted(frame_np[..., ::-1], 0.4, clean_heatmap, 0.6, 0)
        axes[1].imshow(cv2.cvtColor(clean_overlay, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Clean Attention")
        axes[1].axis('off')
        
        # Adversarial heatmap
        adv_heatmap = colourise(adv_heat)
        adv_overlay = cv2.addWeighted(frame_np[..., ::-1], 0.4, adv_heatmap, 0.6, 0)
        axes[2].imshow(cv2.cvtColor(adv_overlay, cv2.COLOR_BGR2RGB))
        axes[2].set_title("Adversarial Attention")
        axes[2].axis('off')
        
        # Add metrics as text
        metrics_text = (
            f"Cosine Similarity: {metrics['cosine_similarity']:.4f} | " 
            f"L2 Distance: {metrics['l2_distance']:.4f} | "
            f"JS Divergence: {metrics['jensen_shannon_div']:.4f}\n"
            f"KL (Clean→Adv): {metrics['kl_clean_to_adv']:.4f} | "
            f"KL (Adv→Clean): {metrics['kl_adv_to_clean']:.4f} | "
            f"Gini: {metrics['clean_concentration']:.4f} → {metrics['adv_concentration']:.4f}"
        )
        
        plt.figtext(0.5, 0.01, metrics_text, ha="center", fontsize=12, 
                    bbox={"facecolor":"white", "alpha":0.8, "pad":5})
        
        plt.tight_layout()
        save_figure(fig, os.path.join(adv_dir, "attention_drift.png"))
        
        # Save individual overlays
        save_image(os.path.join(adv_dir, "clean_heatmap.png"), clean_overlay)
        save_image(os.path.join(adv_dir, "adversarial_heatmap.png"), adv_overlay)
        
        # Also create a difference heatmap
        diff_map = np.abs(clean_heat - adv_heat)
        diff_heatmap = colourise(diff_map)
        diff_overlay = cv2.addWeighted(frame_np[..., ::-1], 0.4, diff_heatmap, 0.6, 0)
        save_image(os.path.join(adv_dir, "difference_heatmap.png"), diff_overlay)
        
        logger.info(f"Saved adversarial analysis to {out_dir}/adversarial/")
        
        return fig
        
    except Exception as e:
        logger.error(f"Failed to visualize attention drift: {e}")
        return None

# ───────── Feature 3: Temporal attention visualization ───────────────

def create_temporal_attention_visualization(frames, per_frame_maps, out_dir):
    """Create a video visualization of attention over time"""
    if not frames or not per_frame_maps or len(frames) == 0 or len(per_frame_maps) == 0:
        logger.warning("No frames or maps for temporal visualization")
        return None, None
        
    temp_dir = os.path.join(out_dir, "temporal")
    if not ensure_dir(temp_dir):
        return None, None
    
    try:
        # Create a grid visualization instead of animation (more reliable)
        rows = min(2, len(frames))
        cols = min(4, int(np.ceil(len(frames) / rows)))
        grid_fig = plt.figure(figsize=(cols*4, rows*4))
        
        for i in range(min(len(frames), len(per_frame_maps))):
            ax = grid_fig.add_subplot(rows, cols, i+1)
            
            frame_np = np.array(frames[i])
            heatmap = colourise(per_frame_maps[i])
            overlay = cv2.addWeighted(frame_np[..., ::-1], 0.4, heatmap, 0.6, 0)
            ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            ax.set_title(f"Frame {i+1}")
            ax.axis('off')
            
            # Save individual frame
            save_image(os.path.join(temp_dir, f"frame_{i+1}_overlay.png"), overlay)
        
        plt.tight_layout()
        save_figure(grid_fig, os.path.join(temp_dir, "temporal_grid.png"))
        
        # Try to create an animated GIF if we have matplotlib animation
        try:
            # Create a figure with original frame and heatmap overlay
            ani_fig, ax = plt.subplots(figsize=(10, 6))
            
            def update(i):
                ax.clear()
                frame_np = np.array(frames[i])
                heatmap = colourise(per_frame_maps[i])
                overlay = cv2.addWeighted(frame_np[..., ::-1], 0.4, heatmap, 0.6, 0)
                ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                ax.set_title(f"Frame {i+1} Attention")
                ax.axis('off')
                return [ax]
            
            ani = animation.FuncAnimation(
                ani_fig, 
                update, 
                frames=min(len(frames), len(per_frame_maps)),
                interval=500, 
                blit=False
            )
            
            # Try saving as GIF
            try:
                ani.save(os.path.join(temp_dir, "attention_over_time.gif"), writer='pillow', fps=2)
                logger.info(f"Saved GIF animation to {temp_dir}/attention_over_time.gif")
            except Exception as gif_error:
                logger.warning(f"Could not create GIF animation: {gif_error}")
                logger.warning("If you need animations, install Pillow with 'pip install pillow'")
                
            # Try saving as MP4 if ffmpeg is available
            try:
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=2, metadata=dict(artist='VideoLLaMA2'), bitrate=1800)
                ani.save(os.path.join(temp_dir, "attention_over_time.mp4"), writer=writer)
                logger.info(f"Saved MP4 animation to {temp_dir}/attention_over_time.mp4")
            except Exception as mp4_error:
                logger.warning(f"Could not create MP4 animation: {mp4_error}")
                logger.warning("If you need MP4 export, install ffmpeg with 'conda install ffmpeg'")
                
            plt.close(ani_fig)
            
        except Exception as e:
            logger.warning(f"Could not create animation: {e}")
        
        logger.info(f"Saved temporal visualization to {out_dir}/temporal/")
        return None, grid_fig
        
    except Exception as e:
        logger.error(f"Failed to create temporal visualization: {e}")
        return None, None

# ────────────────────── Single video processing ─────────────────────────

def process_single_video(args):
    """Process a single video with all the enhanced features"""
    model, processor, tokenizer = load_model()
    
    # Ensure output directory exists
    ensure_dir(args.output_dir)
    
    # 1. Load frames with debug info for troubleshooting
    logger.info(f"Loading video frames from {args.video}...")
    frames = load_frames(args.video, args.frames, debug=True)
    
    if not frames or len(frames) == 0:
        logger.error(f"Failed to load any frames from {args.video}")
        return
        
    logger.info(f"Successfully loaded {len(frames)} frames")
    
    # 2. Process video frames into tensor
    vid_tensor = processor["video"](frames)  # (T, C, H, W) float32 0‑1
    vid_tensor = vid_tensor.half().to(DEVICE)

    # 3. Build prompt with <video> token
    chat_str = tokenizer.apply_chat_template([
        {"role": "user", "content": DEFAULT_VIDEO_TOKEN + "\n" + args.prompt}],
        tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer_multimodal_token(chat_str, tokenizer, DEFAULT_VIDEO_TOKEN,
                                         return_tensors="pt").to(DEVICE)
    attn_mask = torch.ones_like(input_ids, device=DEVICE)

    # 4. Generate caption with original attention visualization
    with torch.no_grad():  # Prevent memory leaks
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            images=[(vid_tensor, "video")],
            do_sample=False,
            max_new_tokens=64,
            output_attentions=True,
            return_dict_in_generate=True)

    caption = tokenizer.decode(out.sequences[0], skip_special_tokens=True)
    logger.info("\n=========== VideoLLaMA 2 Analysis ===========")
    logger.info(f"CAPTION: {caption}")

    if not out.attentions:
        logger.error("No attentions returned — update transformers >4.38 or patch self‑attn.")
        return

    # 5. Original heatmap calculation
    attn_layers = torch.stack([torch.stack(layer) for layer in out.attentions])
    attn_avg = attn_layers.mean(dim=(0,1))
    vis_slice = attn_avg[:, :PATCHES]
    text_len = attn_avg.size(0) - PATCHES
    text_to_vis = vis_slice[-text_len:].mean(dim=0)
    heat = text_to_vis.reshape(NUM_PATCH_SIDE, NUM_PATCH_SIDE).cpu().numpy()

    # 6. Save original heatmap
    base = np.array(frames[0])
    heatmap = colourise(heat)
    overlay = cv2.addWeighted(base[..., ::-1], 0.4, heatmap, 0.6, 0)
    save_image(os.path.join(args.output_dir, "heatmap_overlay.png"), overlay)
    logger.info(f"Saved original heatmap_overlay.png → {args.output_dir}")

    # 7. Feature 1: Per-frame heatmaps
    if not args.no_perframe:
        logger.info("\n[1/3] Extracting per-frame heatmaps...")
        start_time = time.time()
        per_frame_maps = extract_per_frame_attention_batched(model, vid_tensor, input_ids, attn_mask)
        logger.info(f"Per-frame extraction completed in {time.time() - start_time:.2f} seconds")
        visualize_per_frame_heatmaps(frames, per_frame_maps, args.output_dir)
    else:
        per_frame_maps = None
        logger.info("\n[1/3] Skipping per-frame analysis (--no_perframe flag set)")
    
    # 8. Feature 2: Adversarial attack analysis
    if not args.no_adv:
        logger.info("\n[2/3] Analyzing attention drift under adversarial attack...")
        start_time = time.time()
        adv_vid_tensor = create_adversarial_video(vid_tensor, model, input_ids, attn_mask, args.epsilon)
        clean_heat, adv_heat, metrics = analyze_attention_drift(
            model, vid_tensor, adv_vid_tensor, input_ids, attn_mask
        )
        logger.info(f"Adversarial analysis completed in {time.time() - start_time:.2f} seconds")
        visualize_attention_drift(frames, clean_heat, adv_heat, metrics, args.output_dir)
    else:
        logger.info("\n[2/3] Skipping adversarial analysis (--no_adv flag set)")
    
    # 9. Feature 3: Temporal attention visualization
    if not args.no_temporal and per_frame_maps:
        logger.info("\n[3/3] Creating temporal attention visualization...")
        start_time = time.time()
        create_temporal_attention_visualization(frames, per_frame_maps, args.output_dir)
        logger.info(f"Temporal visualization completed in {time.time() - start_time:.2f} seconds")
    else:
        logger.info("\n[3/3] Skipping temporal visualization (--no_temporal flag set or no per-frame maps)")
    
    logger.info(f"\nAnalysis complete! All visualizations saved to {args.output_dir}")
    
    return caption, heat, per_frame_maps

# ───────────────────────── Batch processing ─────────────────────────────

def extract_category_from_filename(filename):
    """Extract category from Kinetics400 filename format"""
    basename = os.path.basename(filename)
    parts = basename.split('_')
    if len(parts) > 1:
        return parts[0]
    return "unknown"

def get_videos_by_category(input_dir, categories=None, max_per_category=MAX_VIDEOS_PER_CATEGORY):
    """Get dictionary of videos organized by category"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    videos_by_category = defaultdict(list)
    
    # Get all video files
    all_videos = []
    for ext in video_extensions:
        all_videos.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
    
    # Organize by category
    for video_path in all_videos:
        category = extract_category_from_filename(video_path)
        if categories is None or category in categories:
            videos_by_category[category].append(video_path)
    
    # Limit the number per category
    for category in videos_by_category:
        if len(videos_by_category[category]) > max_per_category:
            videos_by_category[category] = random.sample(videos_by_category[category], max_per_category)
            
    return videos_by_category

def get_videos_from_list(video_list_file):
    """Load videos from a text file with one path per line"""
    videos_by_category = defaultdict(list)
    
    with open(video_list_file, 'r') as f:
        for line in f:
            video_path = line.strip()
            if os.path.exists(video_path):
                category = extract_category_from_filename(video_path)
                videos_by_category[category].append(video_path)
    
    return videos_by_category

def process_single_video_for_batch(model, processor, tokenizer, video_path, args):
    """Process a single video and return results for batch processing"""
    try:
        # Extract frames and prepare input
        frames = load_frames(video_path, args.frames, debug=True)
        if not frames or len(frames) == 0:
            logger.error(f"Could not load frames from {video_path}")
            return None
            
        vid_tensor = processor["video"](frames)  # (T, C, H, W) float32 0‑1
        vid_tensor = vid_tensor.half().to(DEVICE)

        # Build prompt
        chat_str = tokenizer.apply_chat_template([
            {"role": "user", "content": DEFAULT_VIDEO_TOKEN + "\n" + args.prompt}],
            tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer_multimodal_token(chat_str, tokenizer, DEFAULT_VIDEO_TOKEN,
                                             return_tensors="pt").to(DEVICE)
        attn_mask = torch.ones_like(input_ids, device=DEVICE)

        # Generate caption
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                images=[(vid_tensor, "video")],
                do_sample=False,
                max_new_tokens=64,
                output_attentions=True,
                return_dict_in_generate=True)

        caption = tokenizer.decode(out.sequences[0], skip_special_tokens=True)

        if not out.attentions:
            logger.error(f"No attentions returned for {video_path}")
            return None

        # Process attention maps
        attn_layers = torch.stack([torch.stack(layer) for layer in out.attentions])
        attn_avg = attn_layers.mean(dim=(0,1))
        vis_slice = attn_avg[:, :PATCHES]
        text_len = attn_avg.size(0) - PATCHES
        text_to_vis = vis_slice[-text_len:].mean(dim=0)
        heat = text_to_vis.reshape(NUM_PATCH_SIDE, NUM_PATCH_SIDE).cpu().numpy()
        
        # Extract per-frame attention if enabled
        per_frame_maps = None
        if args.per_frame:
            per_frame_maps = extract_per_frame_attention_batched(model, vid_tensor, input_ids, attn_mask)
        
        # Adversarial analysis if enabled
        adv_results = None
        if args.adversarial:
            adv_vid_tensor = create_adversarial_video(vid_tensor, model, input_ids, attn_mask, args.epsilon)
            clean_heat, adv_heat, metrics = analyze_attention_drift(
                model, vid_tensor, adv_vid_tensor, input_ids, attn_mask
            )
            adv_results = {
                'clean_heat': clean_heat,
                'adv_heat': adv_heat,
                'metrics': metrics
            }
        
        # Return comprehensive results
        results = {
            'video_path': video_path,
            'category': extract_category_from_filename(video_path),
            'caption': caption,
            'global_heat': heat,
            'per_frame_maps': per_frame_maps,
            'adversarial': adv_results,
            'frames': frames if args.save_frames else None
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to process {video_path}: {e}")
        return None

def process_videos_batch(videos_by_category, model, processor, tokenizer, args):
    """Process multiple videos and organize results by category"""
    all_results = {}
    video_count = sum(len(videos) for videos in videos_by_category.values())
    
    # Setup output directory structure
    category_dirs = {}
    for category in videos_by_category:
        category_dir = os.path.join(args.output_dir, category)
        ensure_dir(category_dir)
        category_dirs[category] = category_dir
    
    # Counter for memory cleanup
    video_counter = 0
    
    # Process all videos with progress bar
    with tqdm(total=video_count, desc="Processing videos") as pbar:
        for category, video_paths in videos_by_category.items():
            category_results = []
            
            for video_path in video_paths:
                video_id = os.path.splitext(os.path.basename(video_path))[0]
                video_dir = os.path.join(category_dirs[category], video_id)
                ensure_dir(video_dir)
                
                # Process the video
                result = process_single_video_for_batch(model, processor, tokenizer, video_path, args)
                if result:
                    # Save individual video results
                    save_video_results(result, video_dir, args)
                    category_results.append(result)
                
                # Increment counter and clean memory periodically
                video_counter += 1
                if video_counter % CUDA_MEMORY_CLEANUP_THRESHOLD == 0:
                    clean_gpu_memory()
                    
                pbar.update(1)
            
            all_results[category] = category_results
    
    return all_results

def calculate_attention_statistics(heat_map):
    """Calculate statistics for attention heatmap"""
    try:
        flat = heat_map.flatten()
        
        # Handle NaN/Inf
        flat = np.nan_to_num(flat)
        
        # Basic statistics
        stats = {
            'mean': float(np.mean(flat)),
            'median': float(np.median(flat)),
            'std': float(np.std(flat)),
            'min': float(np.min(flat)),
            'max': float(np.max(flat)),
            '10th_percentile': float(np.percentile(flat, 10)),
            '90th_percentile': float(np.percentile(flat, 90)),
        }
        
        # Concentration metrics
        sorted_flat = np.sort(np.abs(flat))
        n = len(sorted_flat)
        
        # Gini coefficient (measure of inequality)
        if n > 0 and sorted_flat.sum() > 0:
            cumx = np.cumsum(sorted_flat)
            stats['gini_coefficient'] = float((n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n)
        else:
            stats['gini_coefficient'] = 0.0
        
        # Entropy (measure of uncertainty)
        if n > 0:
            norm_flat = np.abs(flat) / (np.sum(np.abs(flat)) + 1e-8)
            norm_flat = np.clip(norm_flat, 1e-10, 1.0)  # Avoid log(0)
            stats['entropy'] = float(-np.sum(norm_flat * np.log(norm_flat)))
        else:
            stats['entropy'] = 0.0
        
        # Top-k concentration (what % of total attention is in top 10% of patches)
        if n > 0:
            top_k = max(int(n * 0.1), 1)  # Top 10%, at least 1
            top_k_sum = np.sum(sorted_flat[-top_k:])
            total_sum = np.sum(sorted_flat)
            stats['top10pct_concentration'] = float(top_k_sum / (total_sum + 1e-8))
        else:
            stats['top10pct_concentration'] = 0.0
        
        return stats
    except Exception as e:
        logger.warning(f"Error calculating attention statistics: {e}")
        return {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            '10th_percentile': 0.0,
            '90th_percentile': 0.0,
            'gini_coefficient': 0.0,
            'entropy': 0.0,
            'top10pct_concentration': 0.0,
            'error': str(e)
        }

def save_video_results(result, output_dir, args):
    """Save results for a single video"""
    try:
        # Save caption and metadata
        metadata = {
            'video_path': result['video_path'],
            'category': result['category'],
            'caption': result['caption']
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save global heatmap
        if result['frames'] and len(result['frames']) > 0:
            base = np.array(result['frames'][0])
            heatmap = colourise(result['global_heat'])
            overlay = cv2.addWeighted(base[..., ::-1], 0.4, heatmap, 0.6, 0)
            save_image(os.path.join(output_dir, "heatmap_overlay.png"), overlay)
        
        # Save attention statistics
        attn_stats = calculate_attention_statistics(result['global_heat'])
        with open(os.path.join(output_dir, 'attention_stats.json'), 'w') as f:
            json.dump(attn_stats, f, indent=2)
        
        # Save per-frame results if available
        if result['per_frame_maps'] and args.per_frame and result['frames']:
            per_frame_dir = os.path.join(output_dir, "per_frame")
            ensure_dir(per_frame_dir)
            
            for i, heat_map in enumerate(result['per_frame_maps']):
                if i < len(result['frames']):
                    frame_np = np.array(result['frames'][i])
                    heatmap = colourise(heat_map)
                    overlay = cv2.addWeighted(frame_np[..., ::-1], 0.4, heatmap, 0.6, 0)
                    save_image(os.path.join(per_frame_dir, f"frame_{i+1}_heatmap.png"), overlay)
        
        # Save adversarial results if available
        if result['adversarial'] and args.adversarial:
            adv_dir = os.path.join(output_dir, "adversarial")
            ensure_dir(adv_dir)
            
            metrics = result['adversarial']['metrics']
            with open(os.path.join(adv_dir, 'adversarial_metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
            
            if result['frames'] and len(result['frames']) > 0:
                frame_np = np.array(result['frames'][0])
                
                # Clean heatmap
                clean_heatmap = colourise(result['adversarial']['clean_heat'])
                clean_overlay = cv2.addWeighted(frame_np[..., ::-1], 0.4, clean_heatmap, 0.6, 0)
                save_image(os.path.join(adv_dir, "clean_heatmap.png"), clean_overlay)
                
                # Adversarial heatmap
                adv_heatmap = colourise(result['adversarial']['adv_heat'])
                adv_overlay = cv2.addWeighted(frame_np[..., ::-1], 0.4, adv_heatmap, 0.6, 0)
                save_image(os.path.join(adv_dir, "adversarial_heatmap.png"), adv_overlay)
                
                # Difference heatmap
                diff_map = np.abs(result['adversarial']['clean_heat'] - result['adversarial']['adv_heat'])
                diff_heatmap = colourise(diff_map)
                diff_overlay = cv2.addWeighted(frame_np[..., ::-1], 0.4, diff_heatmap, 0.6, 0)
                save_image(os.path.join(adv_dir, "difference_heatmap.png"), diff_overlay)
    except Exception as e:
        logger.error(f"Failed to save video results: {e}")

def generate_cross_category_analysis(all_results, output_dir):
    """Generate visualizations comparing categories"""
    cross_cat_dir = os.path.join(output_dir, "cross_category")
    if not ensure_dir(cross_cat_dir):
        return
    
    try:
        # Extract statistics for all videos by category
        stats_by_category = defaultdict(list)
        adv_metrics_by_category = defaultdict(list)
        
        for category, results in all_results.items():
            for result in results:
                # Get attention statistics
                if 'global_heat' in result:
                    stats = calculate_attention_statistics(result['global_heat'])
                    stats['category'] = category
                    stats['video'] = os.path.basename(result['video_path'])
                    stats_by_category[category].append(stats)
                
                # Get adversarial metrics if available
                if result.get('adversarial') and result['adversarial'].get('metrics'):
                    metrics = result['adversarial']['metrics']
                    metrics['category'] = category
                    metrics['video'] = os.path.basename(result['video_path'])
                    adv_metrics_by_category[category].append(metrics)
        
        # Create dataframes for plotting
        stats_data = []
        for category, stats_list in stats_by_category.items():
            for stats in stats_list:
                stats_data.append(stats)
        
        adv_data = []
        for category, metrics_list in adv_metrics_by_category.items():
            for metrics in metrics_list:
                adv_data.append(metrics)
        
        # Convert to pandas dataframes
        if stats_data:
            try:
                stats_df = pd.DataFrame(stats_data)
                
                # Create boxplots of attention statistics by category
                metrics_to_plot = ['gini_coefficient', 'entropy', 'top10pct_concentration']
                valid_metrics = [m for m in metrics_to_plot if m in stats_df.columns]
                
                if valid_metrics:
                    fig, axes = plt.subplots(len(valid_metrics), 1, figsize=(12, 4*len(valid_metrics)))
                    if len(valid_metrics) == 1:
                        axes = [axes]  # Make iterable
                    
                    for i, metric in enumerate(valid_metrics):
                        sns.boxplot(x='category', y=metric, data=stats_df, ax=axes[i])
                        axes[i].set_title(f'{metric} by Action Category')
                        axes[i].set_xlabel('Category')
                        axes[i].set_ylabel(metric.replace('_', ' ').title
