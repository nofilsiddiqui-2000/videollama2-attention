#!/usr/bin/env python3
"""VideoLLaMA 2 — Enhanced attention‑heat‑map demo with per-frame maps,
adversarial attack analysis, and temporal visualization

Run
----
python videollama2_enhanced.py assets/sample_video.mp4 [--frames N] [--epsilon E] [--no_adv] [--no_temporal]

What you get
------------
* A caption printed in the console
* Per-frame heat-maps showing attention distribution over time
* Adversarial attack analysis showing attention drift
* Temporal attention visualization with frame-aligned overlays

Performance Estimates (A100-80GB, fp16)
---------------------------------------
* Caption + global heat-map:   ~1.3s, ~17GB VRAM
* Per-frame analysis (batch):  ~1.5s, ~18GB VRAM
* Adversarial drift analysis:  ~2.0s, +6GB VRAM
* Temporal visualization:      1-4s CPU time, negligible VRAM
"""

import os, sys, cv2, math, torch, numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.nn import functional as F
from typing import List, Tuple, Dict, Optional
from scipy.spatial.distance import cosine, jensenshannon
import time

# ─────────────────────────  add repo to path  ──────────────────────────
sys.path.append("./VideoLLaMA2")

from videollama2 import model_init
from videollama2.mm_utils import tokenizer_multimodal_token
from videollama2.utils import disable_torch_init
from videollama2.constants import DEFAULT_VIDEO_TOKEN

# ──────────────────────────────  CONFIG  ───────────────────────────────
MODEL_ID        = "DAMO-NLP-SG/VideoLLaMA2-7B"
NUM_PATCH_SIDE  = 24                   # ViT‑L/14 on 336×336 ⇒ 24×24 patches
PATCHES         = NUM_PATCH_SIDE ** 2  # 576
NUM_FRAMES      = 8                    # default temporal pooling window
OUT_DIR         = "outputs"; 
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
ADVERSARIAL_EPSILON = 0.03             # Adversarial perturbation strength

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

def load_frames(path: str, num_frames: int):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(total // num_frames, 1)
    frames = []
    idx = 0
    while len(frames) < num_frames and cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read(); idx += step
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (336, 336), interpolation=cv2.INTER_AREA)
        frames.append(Image.fromarray(frame))
    cap.release()
    return frames


def colourise(mat: np.ndarray):
    mat = (mat - mat.min()) / (mat.max() - mat.min() + 1e-6)
    mat = np.clip(mat * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(mat, cv2.COLORMAP_JET)

# ───────────────────────  model + processors  ──────────────────────────

def load_model(model_id=MODEL_ID):
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
    
    # Process each frame individually but in a single forward pass by stacking
    # Extract single frames as a batch [num_frames, C, H, W]
    batched_frames = vid_tensor  # Already [num_frames, C, H, W]
    
    # Generate with attentions for all frames in one go
    with torch.no_grad():  # Prevent memory leaks
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
    
    return per_frame_maps

def visualize_per_frame_heatmaps(frames, per_frame_maps, out_dir):
    """Create visualization of per-frame attention heatmaps"""
    per_frame_dir = os.path.join(out_dir, "per_frame")
    if not ensure_dir(per_frame_dir):
        return None
    
    # Create a grid of frames (max 4x4 to prevent memory issues)
    max_per_row = 4
    num_frames = len(frames)
    rows = min(4, int(np.ceil(num_frames / max_per_row)))
    cols = min(max_per_row, num_frames)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    if rows == 1 and cols == 1:
        axes = np.array([axes])  # Handle single frame case
    axes = axes.flatten() if hasattr(axes, 'flatten') else axes
    
    for i, (frame, heat_map) in enumerate(zip(frames, per_frame_maps)):
        if i >= len(axes):
            break
            
        frame_np = np.array(frame)
        heatmap = colourise(heat_map)
        overlay = cv2.addWeighted(frame_np[..., ::-1], 0.4, heatmap, 0.6, 0)
        
        axes[i].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"Frame {i+1}")
        axes[i].axis('off')
        
        # Save individual overlay
        save_image(os.path.join(per_frame_dir, f"heatmap_frame_{i+1}.png"), overlay)
    
    # Hide unused subplots
    for i in range(len(per_frame_maps), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    save_figure(fig, os.path.join(out_dir, "per_frame_heatmaps.png"))
    print(f"Saved per-frame heatmaps to {out_dir}/per_frame_heatmaps.png")
    return fig

# ────────── Feature 2: Adversarial attack analysis ───────────────────

def create_adversarial_video(vid_tensor, model, input_ids, attn_mask, epsilon=ADVERSARIAL_EPSILON):
    """Create an adversarial video by maximizing attention drift"""
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

def analyze_attention_drift(model, vid_tensor, adv_vid_tensor, input_ids, attn_mask):
    """Analyze and quantify attention drift between clean and adversarial videos"""
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
    
    # Cosine similarity (lower means more drift)
    cos_sim = 1 - cosine(clean_flat, adv_flat)
    
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
    js_div = jensenshannon(clean_norm, adv_norm)
    
    # KL divergences (both directions)
    kl_clean_to_adv = np.sum(clean_norm * np.log(clean_norm / adv_norm))
    kl_adv_to_clean = np.sum(adv_norm * np.log(adv_norm / clean_norm))
    
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
        "cosine_similarity": cos_sim,
        "l2_distance": l2_dist,
        "jensen_shannon_div": js_div,
        "kl_clean_to_adv": kl_clean_to_adv,
        "kl_adv_to_clean": kl_adv_to_clean,
        "clean_concentration": clean_gini,
        "adv_concentration": adv_gini
    }
    
    return clean_heat, adv_heat, metrics

def visualize_attention_drift(frames, clean_heat, adv_heat, metrics, out_dir):
    """Visualize attention drift between clean and adversarial inputs"""
    adv_dir = os.path.join(out_dir, "adversarial")
    if not ensure_dir(adv_dir):
        return None
    
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

# ───────── Feature 3: Temporal attention visualization ───────────────

def create_temporal_attention_visualization(frames, per_frame_maps, out_dir):
    """Create a video visualization of attention over time"""
    temp_dir = os.path.join(out_dir, "temporal")
    if not ensure_dir(temp_dir):
        return None, None
    
    # Create a figure with original frame and heatmap overlay
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Function to update the figure for each frame
    def update(i):
        ax.clear()
        frame_np = np.array(frames[i])
        heatmap = colourise(per_frame_maps[i])
        overlay = cv2.addWeighted(frame_np[..., ::-1], 0.4, heatmap, 0.6, 0)
        ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Frame {i+1} Attention")
        ax.axis('off')
        return [ax]
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=500, blit=False)  # blit=False to avoid flickering
    
    # Try to save as video with ffmpeg, fall back to GIF if not available
    try:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=2, metadata=dict(artist='VideoLLaMA2'), bitrate=1800)
        ani.save(os.path.join(temp_dir, "attention_over_time.mp4"), writer=writer)
        print(f"Saved MP4 animation to {temp_dir}/attention_over_time.mp4")
    except Exception as e:
        print(f"[WARN] Could not save MP4 (ffmpeg missing?): {e}")
    
    # Always try to save as GIF as a fallback
    try:
        ani.save(os.path.join(temp_dir, "attention_over_time.gif"), writer='pillow', fps=2)
        print(f"Saved GIF animation to {temp_dir}/attention_over_time.gif")
    except Exception as e:
        print(f"[WARN] Could not save GIF: {e}")
    
    # Also create a comparison grid
    grid_fig = plt.figure(figsize=(16, 12))
    rows = int(np.ceil(len(frames) / 4))
    cols = min(4, len(frames))
    
    for i, (frame, heat_map) in enumerate(zip(frames, per_frame_maps)):
        ax = grid_fig.add_subplot(rows, cols, i+1)
        frame_np = np.array(frame)
        heatmap = colourise(heat_map)
        overlay = cv2.addWeighted(frame_np[..., ::-1], 0.4, heatmap, 0.6, 0)
        ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Frame {i+1}")
        ax.axis('off')
    
    plt.tight_layout()
    save_figure(grid_fig, os.path.join(temp_dir, "temporal_grid.png"))
    
    plt.close(fig)  # Close animation figure
    print(f"Saved temporal visualization to {out_dir}/temporal/")
    
    return ani, grid_fig

# ─────────────────────────  main flow  ─────────────────────────────────

def main(args):
    # Create output directory
    ensure_dir(args.output_dir)
    
    model, processor, tokenizer = load_model()
    # 1 ▸ frames → tensor
    frames = load_frames(args.video_path, args.frames)
    vid_tensor = processor["video"](frames)  # (T, C, H, W) float32 0‑1
    vid_tensor = vid_tensor.half().to(DEVICE)

    # 2 ▸ build prompt with <video> token
    chat_str = tokenizer.apply_chat_template([
        {"role": "user", "content": DEFAULT_VIDEO_TOKEN + "\n" + args.prompt}],
        tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer_multimodal_token(chat_str, tokenizer, DEFAULT_VIDEO_TOKEN,
                                         return_tensors="pt").to(DEVICE)
    attn_mask = torch.ones_like(input_ids, device=DEVICE)

    # 3 ▸ Generate caption with original attention visualization
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

    # Original heatmap calculation
    attn_layers = torch.stack([torch.stack(layer) for layer in out.attentions])
    attn_avg = attn_layers.mean(dim=(0,1))
    vis_slice = attn_avg[:, :PATCHES]
    text_len = attn_avg.size(0) - PATCHES
    text_to_vis = vis_slice[-text_len:].mean(dim=0)
    heat = text_to_vis.reshape(NUM_PATCH_SIDE, NUM_PATCH_SIDE).cpu().numpy()

    # Save original heatmap
    base = np.array(frames[0])
    heatmap = colourise(heat)
    overlay = cv2.addWeighted(base[..., ::-1], 0.4, heatmap, 0.6, 0)
    save_image(os.path.join(args.output_dir, "heatmap_overlay.png"), overlay)
    print("Saved original heatmap_overlay.png →", args.output_dir)

    # Feature 1: Per-frame heatmaps
    print("\n[1/3] Extracting per-frame heatmaps...")
    start_time = time.time()
    per_frame_maps = extract_per_frame_attention_batched(model, vid_tensor, input_ids, attn_mask)
    print(f"Per-frame extraction completed in {time.time() - start_time:.2f} seconds")
    visualize_per_frame_heatmaps(frames, per_frame_maps, args.output_dir)
    
    # Feature 2: Adversarial attack analysis
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
    
    # Feature 3: Temporal attention visualization
    if not args.no_temporal:
        print("\n[3/3] Creating temporal attention visualization...")
        start_time = time.time()
        create_temporal_attention_visualization(frames, per_frame_maps, args.output_dir)
        print(f"Temporal visualization completed in {time.time() - start_time:.2f} seconds")
    else:
        print("\n[3/3] Skipping temporal visualization (--no_temporal flag set)")
    
    print("\nAnalysis complete! All visualizations saved to", args.output_dir)
    
    return caption, heat, per_frame_maps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VideoLLaMA 2 Attention Heatmap Demo")
    parser.add_argument("video_path", type=str, help="Path to input video")
    parser.add_argument("--prompt", type=str, default="Describe what is happening in the video.",
                        help="Prompt for the model")
    parser.add_argument("--frames", type=int, default=NUM_FRAMES,
                        help=f"Number of frames to process (default: {NUM_FRAMES})")
    parser.add_argument("--epsilon", type=float, default=ADVERSARIAL_EPSILON,
                        help=f"Epsilon for adversarial attack (default: {ADVERSARIAL_EPSILON})")
    parser.add_argument("--output_dir", type=str, default=OUT_DIR,
                        help=f"Output directory (default: {OUT_DIR})")
    parser.add_argument("--no_adv", action="store_true",
                        help="Skip adversarial attack analysis (faster)")
    parser.add_argument("--no_temporal", action="store_true",
                        help="Skip temporal visualization (faster)")
    args = parser.parse_args()
    
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
        
    main(args)
