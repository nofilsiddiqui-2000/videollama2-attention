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

# ─────────────────────────  add repo to path  ──────────────────────────
sys.path.append("./VideoLLaMA2")

# Try to import VideoLLaMA2 modules, with descriptive errors if they're missing
try:
    from videollama2 import model_init
    from videollama2.mm_utils import tokenizer_multimodal_token
    from videollama2.utils import disable_torch_init
    from videollama2.constants import DEFAULT_VIDEO_TOKEN
except ImportError as e:
    print(f"ERROR: Could not import VideoLLaMA2 modules. Make sure the VideoLLaMA2 directory is in ./VideoLLaMA2")
    print(f"Import error: {e}")
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

# ──────────────────────────  utilities  ────────────────────────────────

def ensure_dir(path):
    """Create directory if it doesn't exist, handle permissions errors"""
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except (PermissionError, OSError) as e:
        print(f"[WARN] Could not create directory {path}: {e}")
        return False

def save_image(path, img):
    """Save image with error handling"""
    try:
        cv2.imwrite(path, img)
        return True
    except Exception as e:
        print(f"[WARN] Could not save image to {path}: {e}")
        return False

def save_figure(fig, path):
    """Save figure with error handling"""
    try:
        fig.savefig(path)
        plt.close(fig)  # Close figure to prevent memory leaks
        return True
    except Exception as e:
        print(f"[WARN] Could not save figure to {path}: {e}")
        plt.close(fig)  # Still close figure even on error
        return False

def load_frames(path: str, num_frames: int, debug=False):
    """
    Load frames from a video with improved error handling
    If debug=True, print additional diagnostic information
    """
    if debug:
        print(f"[DEBUG] Opening video: {path}")
        print(f"[DEBUG] Requested frames: {num_frames}")
    
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            if debug:
                print(f"[DEBUG] Failed to open video: {path}")
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if debug:
            print(f"[DEBUG] Video stats: {total_frames} frames, {fps} fps, {width}x{height}")
        
        # Check if video seems valid
        if total_frames <= 0 or fps <= 0 or width <= 0 or height <= 0:
            if debug:
                print(f"[DEBUG] Invalid video properties")
            cap.release()
            return None
        
        # Adjust step size for short videos
        if total_frames < num_frames:
            if debug:
                print(f"[DEBUG] Video too short ({total_frames} frames), will duplicate frames")
            # For very short videos, we'll duplicate frames to reach num_frames
            step = 1
        else:
            step = max(total_frames // num_frames, 1)
            
        if debug:
            print(f"[DEBUG] Using step size: {step}")
        
        frames = []
        idx = 0
        read_attempts = 0
        max_attempts = total_frames * 2  # Prevent infinite loops
        
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
                        print(f"[DEBUG] Failed to read frame at position {idx}")
                    idx += 1
                    continue
                
                # Process the frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Check if frame is valid (not black or corrupted)
                if frame.size == 0 or np.mean(frame) < 5:  # Very dark frame check
                    if debug:
                        print(f"[DEBUG] Frame at {idx} appears corrupted or black")
                    idx += 1
                    continue
                
                # Resize with error handling
                try:
                    frame = cv2.resize(frame, (336, 336), interpolation=cv2.INTER_AREA)
                    frames.append(Image.fromarray(frame))
                except Exception as e:
                    if debug:
                        print(f"[DEBUG] Failed to resize frame: {e}")
                    idx += 1
                    continue
                
                # Increment for next frame
                idx += step
                
            except Exception as e:
                if debug:
                    print(f"[DEBUG] Error processing frame: {e}")
                idx += 1
        
        cap.release()
        
        if len(frames) < num_frames:
            if debug:
                print(f"[DEBUG] Could only extract {len(frames)} frames, duplicating to reach {num_frames}")
            # Duplicate frames if we couldn't get enough
            while len(frames) < num_frames:
                frames.append(frames[len(frames) % len(frames)] if frames else None)
        
        if debug:
            print(f"[DEBUG] Successfully extracted {len(frames)} frames")
            
        return frames
        
    except Exception as e:
        if debug:
            print(f"[DEBUG] Exception during frame extraction: {e}")
        return None

def colourise(mat: np.ndarray):
    """Convert a 2D attention matrix to a colored heatmap"""
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

# ───────────────────────  model + processors  ──────────────────────────

def load_model(model_id=MODEL_ID):
    """Load the VideoLLaMA2 model with attention tracking enabled"""
    disable_torch_init()
    # Turn off FA‑2 to guarantee attention weights propagate
    import transformers.modeling_utils as _mu
    _mu._check_and_enable_flash_attn_2 = lambda *_, **__: False
    _mu.PreTrainedModel._check_and_enable_flash_attn_2 = staticmethod(lambda *_, **__: False)

    model, processor, tokenizer = model_init(model_id)
    model.eval()  # Already device-mapped by Accelerate
    return model, processor, tokenizer

# ─────────────────── Feature 1: Per-frame heatmaps ────────────────────

def extract_per_frame_attention_batched(model, vid_tensor, input_ids, attn_mask):
    """Extract attention maps for all frames in a batched manner"""
    per_frame_maps = []
    num_frames = vid_tensor.size(0)
    
    # For very small tensors, handle differently
    if num_frames == 0:
        print(f"[WARN] Empty video tensor, cannot extract per-frame attention")
        return []
    
    # Process each frame individually but in a single forward pass by stacking
    # Extract single frames as a batch [num_frames, C, H, W]
    batched_frames = vid_tensor  # Already [num_frames, C, H, W]
    
    # Generate with attentions for all frames in one go
    with torch.no_grad():  # Prevent memory leaks
        try:
            out = model.generate(
                input_ids=input_ids.repeat(num_frames, 1),  # Repeat for each frame
                attention_mask=attn_mask.repeat(num_frames, 1),
                images=[(batched_frames, "video")],
                do_sample=False,
                max_new_tokens=16,  # Keep shorter for faster processing
                output_attentions=True,
                return_dict_in_generate=True)
            
            if not out.attentions:
                print(f"[ERR] No attentions returned")
                return []
                
            # Process attention maps for each frame
            attn_layers = torch.stack([torch.stack(layer) for layer in out.attentions])  # [layers, heads, batch*seq, seq]
            
            # We need to separate the batch dimension which is flattened with sequence
            # For each frame in the batch:
            for frame_idx in range(num_frames):
                # Calculate offsets for this frame's tokens in the batched sequence
                seq_len = input_ids.size(1) + PATCHES  # visual + input tokens
                frame_start = frame_idx * seq_len
                frame_end = (frame_idx + 1) * seq_len
                
                # Extract this frame's attention maps
                frame_attn_avg = attn_layers[:, :, frame_start:frame_end, frame_start:frame_end].mean(dim=(0, 1))  # Average over layers and heads
                
                # Visual tokens occupy first PATCHES positions
                vis_slice = frame_attn_avg[:, :PATCHES]
                text_len = frame_attn_avg.size(0) - PATCHES
                text_to_vis = vis_slice[-text_len:].mean(dim=0)  # Average over generated tokens
                heat = text_to_vis.reshape(NUM_PATCH_SIDE, NUM_PATCH_SIDE).cpu().numpy()
                
                per_frame_maps.append(heat)
                
        except Exception as e:
            print(f"[ERR] Failed to extract per-frame attention: {e}")
            return []
    
    return per_frame_maps

def visualize_per_frame_heatmaps(frames, per_frame_maps, out_dir):
    """Create visualization of per-frame attention heatmaps"""
    per_frame_dir = os.path.join(out_dir, "per_frame")
    if not ensure_dir(per_frame_dir):
        return None
    
    if not frames or not per_frame_maps or len(frames) == 0 or len(per_frame_maps) == 0:
        print(f"[WARN] No frames or heatmaps to visualize")
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
            print(f"[WARN] Failed to process frame {i}: {e}")
            if i < len(axes):
                axes[i].text(0.5, 0.5, f"Frame {i+1}\nProcessing Error", 
                            ha='center', va='center')
                axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_frames, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    save_figure(fig, os.path.join(out_dir, "per_frame_heatmaps.png"))
    print(f"Saved per-frame heatmaps to {out_dir}/per_frame_heatmaps.png")
    return fig

# ────────── Feature 2: Adversarial attack analysis ───────────────────

def create_adversarial_video(vid_tensor, model, input_ids, attn_mask, epsilon=ADVERSARIAL_EPSILON):
    """Create an adversarial video by maximizing attention drift"""
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
        print(f"[ERR] Adversarial video generation failed: {e}")
        # Return original tensor as fallback
        return vid_tensor.detach()

def analyze_attention_drift(model, vid_tensor, adv_vid_tensor, input_ids, attn_mask):
    """Analyze and quantify attention drift between clean and adversarial videos"""
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
        print(f"[ERR] Attention drift analysis failed: {e}")
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
        print(f"[WARN] No frames available for adversarial visualization")
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
        
        print(f"Saved adversarial analysis to {out_dir}/adversarial/")
        
        return fig
        
    except Exception as e:
        print(f"[ERR] Failed to visualize attention drift: {e}")
        return None

# ───────── Feature 3: Temporal attention visualization ───────────────

def create_temporal_attention_visualization(frames, per_frame_maps, out_dir):
    """Create a video visualization of attention over time"""
    if not frames or not per_frame_maps or len(frames) == 0 or len(per_frame_maps) == 0:
        print(f"[WARN] No frames or maps for temporal visualization")
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
            ani.save(os.path.join(temp_dir, "attention_over_time.gif"), writer='pillow', fps=2)
            print(f"Saved GIF animation to {temp_dir}/attention_over_time.gif")
            plt.close(ani_fig)
            
        except Exception as e:
            print(f"[WARN] Could not create animation: {e}")
        
        print(f"Saved temporal visualization to {out_dir}/temporal/")
        return None, grid_fig
        
    except Exception as e:
        print(f"[ERR] Failed to create temporal visualization: {e}")
        return None, None

# ────────────────────── Single video processing ─────────────────────────

def process_single_video(args):
    """Process a single video with all the enhanced features"""
    model, processor, tokenizer = load_model()
    
    # Ensure output directory exists
    ensure_dir(args.output_dir)
    
    # 1. Load frames with debug info for troubleshooting
    print(f"Loading video frames from {args.video}...")
    frames = load_frames(args.video, args.frames, debug=True)
    
    if not frames or len(frames) == 0:
        print(f"[ERROR] Failed to load any frames from {args.video}")
        return
        
    print(f"Successfully loaded {len(frames)} frames")
    
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
    print("\n=========== VideoLLaMA 2 Analysis ===========")
    print("CAPTION:", caption)

    if not out.attentions:
        print("[ERR] No attentions returned — update transformers >4.38 or patch self‑attn.")
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
    print("Saved original heatmap_overlay.png →", args.output_dir)

    # 7. Feature 1: Per-frame heatmaps
    if not args.no_perframe:
        print("\n[1/3] Extracting per-frame heatmaps...")
        start_time = time.time()
        per_frame_maps = extract_per_frame_attention_batched(model, vid_tensor, input_ids, attn_mask)
        print(f"Per-frame extraction completed in {time.time() - start_time:.2f} seconds")
        visualize_per_frame_heatmaps(frames, per_frame_maps, args.output_dir)
    else:
        per_frame_maps = None
        print("\n[1/3] Skipping per-frame analysis (--no_perframe flag set)")
    
    # 8. Feature 2: Adversarial attack analysis
    if not args.no_adv:
        print("\n[2/3] Analyzing attention drift under adversarial attack...")
        start_time = time.time()
        adv_vid_tensor = create_adversarial_video(vid_tensor, model, input_ids, attn_mask, args.epsilon)
        clean_heat, adv_heat, metrics = analyze_attention_drift(
            model, vid_tensor, adv_vid_tensor, input_ids, attn_mask
        )
        print(f"Adversarial analysis completed in {time.time() - start_time:.2f} seconds")
        visualize_attention_drift(frames, clean_heat, adv_heat, metrics, args.output_dir)
    else:
        print("\n[2/3] Skipping adversarial analysis (--no_adv flag set)")
    
    # 9. Feature 3: Temporal attention visualization
    if not args.no_temporal and per_frame_maps:
        print("\n[3/3] Creating temporal attention visualization...")
        start_time = time.time()
        create_temporal_attention_visualization(frames, per_frame_maps, args.output_dir)
        print(f"Temporal visualization completed in {time.time() - start_time:.2f} seconds")
    else:
        print("\n[3/3] Skipping temporal visualization (--no_temporal flag set or no per-frame maps)")
    
    print("\nAnalysis complete! All visualizations saved to", args.output_dir)
    
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
            print(f"[ERR] Could not load frames from {video_path}")
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
            print(f"[ERR] No attentions returned for {video_path}")
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
        print(f"[ERROR] Failed to process {video_path}: {e}")
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
                    
                pbar.update(1)
                torch.cuda.empty_cache()  # Free up GPU memory
            
            all_results[category] = category_results
    
    return all_results

def calculate_attention_statistics(heat_map):
    """Calculate statistics for attention heatmap"""
    try:
        flat = heat_map.flatten()
        
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
        print(f"[WARN] Error calculating attention statistics: {e}")
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
        print(f"[ERROR] Failed to save video results: {e}")

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
                        axes[i].set_ylabel(metric.replace('_', ' ').title())
                    
                    plt.tight_layout()
                    save_figure(fig, os.path.join(cross_cat_dir, "attention_stats_by_category.png"))
            except Exception as e:
                print(f"[WARN] Failed to create attention stats visualization: {e}")
        
        # Plot adversarial robustness by category if available
        if adv_data:
            try:
                adv_df = pd.DataFrame(adv_data)
                
                metrics_to_plot = ['cosine_similarity', 'l2_distance', 'jensen_shannon_div']
                valid_metrics = [m for m in metrics_to_plot if m in adv_df.columns]
                
                if valid_metrics:
                    fig, axes = plt.subplots(len(valid_metrics), 1, figsize=(12, 4*len(valid_metrics)))
                    if len(valid_metrics) == 1:
                        axes = [axes]  # Make iterable
                    
                    for i, metric in enumerate(valid_metrics):
                        sns.boxplot(x='category', y=metric, data=adv_df, ax=axes[i])
                        axes[i].set_title(f'Adversarial {metric} by Action Category')
                        axes[i].set_xlabel('Category')
                        axes[i].set_ylabel(metric.replace('_', ' ').title())
                    
                    plt.tight_layout()
                    save_figure(fig, os.path.join(cross_cat_dir, "adversarial_metrics_by_category.png"))
            except Exception as e:
                print(f"[WARN] Failed to create adversarial metrics visualization: {e}")
        
        # Create attention heatmap aggregates by category
        for category, results in all_results.items():
            if results:
                try:
                    # Average heatmaps across category
                    heatmaps = [r['global_heat'] for r in results if 'global_heat' in r]
                    if heatmaps:
                        # Filter out invalid heatmaps
                        valid_heatmaps = []
                        for hm in heatmaps:
                            if hm is not None and hm.shape == (NUM_PATCH_SIDE, NUM_PATCH_SIDE):
                                valid_heatmaps.append(hm)
                        
                        if valid_heatmaps:
                            avg_heat = np.mean(valid_heatmaps, axis=0)
                            
                            # Create a clean visualization of the average attention pattern
                            plt.figure(figsize=(8, 8))
                            plt.imshow(avg_heat, cmap='jet')
                            plt.title(f'Average Attention Pattern: {category}')
                            plt.colorbar(label='Attention Weight')
                            plt.axis('off')
                            save_figure(plt.gcf(), os.path.join(cross_cat_dir, f"{category}_avg_attention.png"))
                except Exception as e:
                    print(f"[WARN] Failed to create average heatmap for {category}: {e}")
    except Exception as e:
        print(f"[ERROR] Failed to generate cross-category analysis: {e}")

def create_summary_report(all_results, output_dir):
    """Create a summary HTML report of all processed videos"""
    html_file = os.path.join(output_dir, "summary.html")
    
    try:
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>VideoLLaMA2 Batch Analysis Summary</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .category {{ margin-bottom: 40px; }}
                .video {{ margin-bottom: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
                .video-info {{ display: flex; flex-wrap: wrap; }}
                .attention {{ width: 300px; margin-right: 20px; margin-bottom: 20px; }}
                .caption {{ flex: 1; min-width: 300px; }}
                .metrics {{ font-size: 0.9em; color: #555; }}
                img {{ max-width: 100%; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>VideoLLaMA2 Batch Analysis Summary</h1>
            <p>Processed on {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Categories Overview</h2>
            <table>
                <tr>
                    <th>Category</th>
                    <th>Videos Processed</th>
                    <th>Average Attention Gini</th>
                    <th>Average Adversarial Drift</th>
                </tr>
        """
        
        # Add category overview rows
        for category, results in all_results.items():
            if results:
                try:
                    gini_values = []
                    for r in results:
                        if 'global_heat' in r:
                            gini = calculate_attention_statistics(r['global_heat']).get('gini_coefficient', 0)
                            if gini is not None and not np.isnan(gini):
                                gini_values.append(gini)
                    
                    avg_gini = np.mean(gini_values) if gini_values else 0.0
                    
                    # Calculate average adversarial drift if available
                    avg_drift = "N/A"
                    drift_values = []
                    for r in results:
                        if r.get('adversarial') and 'metrics' in r['adversarial']:
                            drift = r['adversarial']['metrics'].get('l2_distance', 0)
                            if drift is not None and not np.isnan(drift):
                                drift_values.append(drift)
                    
                    if drift_values:
                        avg_drift = f"{np.mean(drift_values):.4f}"
                    
                    html_content += f"""
                    <tr>
                        <td>{category}</td>
                        <td>{len(results)}</td>
                        <td>{avg_gini:.4f}</td>
                        <td>{avg_drift}</td>
                    </tr>
                    """
                except Exception as e:
                    print(f"[WARN] Error calculating stats for category {category}: {e}")
                    html_content += f"""
                    <tr>
                        <td>{category}</td>
                        <td>{len(results)}</td>
                        <td>Error</td>
                        <td>Error</td>
                    </tr>
