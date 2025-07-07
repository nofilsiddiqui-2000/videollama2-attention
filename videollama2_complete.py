#!/usr/bin/env python3
"""VideoLLaMA 2 — Complete analysis toolkit with per-frame maps,
adversarial attack analysis, temporal visualization, and batch processing

Run (single video mode)
-----------------------
python videollama2_complete.py --video path/to/video.mp4

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
import argparse, glob, random, json, time, warnings, logging, errno
from PIL import Image
import matplotlib.pyplot as plt
from torch.nn import functional as F
import matplotlib.animation as animation
from typing import List, Dict, Tuple, Optional, Union, Any
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
OUT_DIR         = "outputs"; 
OUT_DIR_BATCH   = "outputs_batch";
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
ADVERSARIAL_EPSILON = 0.03             # Adversarial perturbation strength
MAX_VIDEOS_PER_CATEGORY = 3            # Limit for batch mode
MEMORY_CHECK_INTERVAL = 5              # Check memory usage every N videos

# ──────────────────────────  utilities  ────────────────────────────────

def ensure_dir(path: str) -> bool:
    """Create directory if it doesn't exist with proper error handling.
    
    Args:
        path: Directory path to create
        
    Returns:
        bool: True if directory exists or was created, False otherwise
    """
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except PermissionError as e:
        logger.error(f"Permission denied when creating directory {path}: {e}")
        return False
    except OSError as e:
        # Only catch EEXIST, let other errors propagate
        if e.errno != errno.EEXIST:
            logger.error(f"Failed to create directory {path}: {e}")
            return False
        return True

def save_image(path: str, img: np.ndarray) -> bool:
    """Save image with error handling
    
    Args:
        path: Path to save the image
        img: Image as numpy array
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    try:
        cv2.imwrite(path, img)
        return True
    except Exception as e:
        logger.warning(f"Could not save image to {path}: {e}")
        return False

def save_figure(fig: plt.Figure, path: str) -> bool:
    """Save matplotlib figure with error handling
    
    Args:
        fig: Matplotlib figure object
        path: Path to save the figure
        
    Returns:
        bool: True if saved successfully, False otherwise
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
    Load frames from a video with robust error handling for corrupt files.
    
    Args:
        path: Path to video file
        num_frames: Number of frames to extract
        debug: Whether to print debug information
        
    Returns:
        List of PIL Image objects or None if loading failed
    
    Example:
        frames = load_frames('my_video.mp4', 8)
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
        
        # Check if video seems valid - crucial check for corrupt MP4s
        if total_frames < 2 or fps <= 0 or width <= 0 or height <= 0:
            if debug:
                logger.debug(f"Invalid video properties, marking as corrupt")
            cap.release()
            return None
        
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
        max_attempts = min(total_frames * 2, 100)  # Prevent excessive attempts
        
        while len(frames) < num_frames and read_attempts < max_attempts:
            try:
                if idx >= total_frames:
                    # If we've reached the end, start from beginning for remaining frames
                    idx = idx % total_frames
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                read_attempts += 1
                
                if not ret:
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
        
        if len(frames) < 1:
            if debug:
                logger.debug(f"Could not extract any valid frames")
            return None
            
        if len(frames) < num_frames:
            if debug:
                logger.debug(f"Could only extract {len(frames)} frames, duplicating to reach {num_frames}")
            # Duplicate frames if we couldn't get enough
            while len(frames) < num_frames:
                frames.append(frames[len(frames) % len(frames)])
        
        if debug:
            logger.debug(f"Successfully extracted {len(frames)} frames")
            
        return frames
        
    except Exception as e:
        if debug:
            logger.debug(f"Exception during frame extraction: {e}")
        return None

def check_cuda_memory() -> Dict[str, float]:
    """Check CUDA memory usage and return stats
    
    Returns:
        Dictionary with memory statistics (in GB)
    """
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "free": 0, "total": 0}
    
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    free = total - reserved
    
    return {
        "allocated": allocated,
        "reserved": reserved, 
        "free": free,
        "total": total
    }

def safe_cuda_empty_cache() -> None:
    """Safely empty CUDA cache only when fragmentation is high"""
    if not torch.cuda.is_available():
        return
        
    mem = check_cuda_memory()
    # Only empty cache if significant fragmentation detected
    # (reserved much higher than allocated)
    if mem["reserved"] > mem["allocated"] * 1.5 and mem["reserved"] > 2.0:
        logger.debug(f"CUDA memory cleanup:
