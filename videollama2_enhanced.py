#!/usr/bin/env python3
"""VideoLLaMA 2 — Enhanced attention‑heat‑map demo with per-frame maps,
adversarial attack analysis, and temporal visualization

Run
----
python videollama2_enhanced.py assets/sample_video.mp4

What you get
------------
* A caption printed in the console
* Per-frame heat-maps showing attention distribution over time
* Adversarial attack analysis showing attention drift
* Temporal attention visualization with frame-aligned overlays
"""

import os, sys, cv2, torch, numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from torch.nn import functional as F
import matplotlib.animation as animation
from scipy.spatial.distance import cosine

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
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)
    return path

def load_frames(path: str, num_frames: int):
    """Load video frames with improved error handling"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Handle videos with invalid metadata
    if total < 2:
        # Try reading a test frame
        ret, test_frame = cap.read()
        if not ret:
            print(f"Video appears corrupted: {path}")
            cap.release()
            return None
        # Reset and assume it has frames
        frames = [Image.fromarray(cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB))]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        total = 100  # Assume it has enough frames
    
    step = max(total // num_frames, 1)
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
    
    # Handle if we couldn't get enough frames
    if len(frames) < num_frames and len(frames) > 0:
        while len(frames) < num_frames:
            frames.append(frames[len(frames) % len(frames)])
            
    return frames


def colourise(mat: np.ndarray):
    """Convert attention matrix to heatmap visualization"""
    # Handle NaN or Inf values
    mat = np.nan_to_num(mat, nan=0.0, posinf=1.0, neginf=0.0)
    
    if mat.size == 0 or np.all(mat == 0):
        # Return a blank blue heatmap instead of erroring
        blank = np.zeros((NUM_PATCH_SIDE, NUM_PATCH_SIDE, 3), dtype=np.uint8)
        return blank
    
    mat = (mat - mat.min()) / (mat.max() - mat.min() + 1e-6)
    mat = np.clip(mat * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(mat, cv2.COLORMAP_JET)

# ───────────────────────  model + processors  ──────────────────────────

def load_model(model_id=MODEL_ID):
    """Load VideoLLaMA2 model with attention tracking"""
    disable_torch_init()
    # Turn off FA‑2 to guarantee attention weights propagate
    import transformers.modeling_utils as _mu
    _mu._check_and_enable_flash_attn_2 = lambda *_, **__: False
    _mu.PreTrainedModel._check_and_enable_flash_attn_2 = staticmethod(lambda *_, **__: False)

    model, processor, tokenizer = model_init(model_id)
    model.eval()  # Already device-mapped by Accelerate
    return model, processor, tokenizer

# ─────────────────── Feature 1: Per-frame heatmaps ────────────────────

def extract_per_frame_attention(model, frames, processor, input_ids, attn_mask):
    """Extract attention map for each frame individually"""
    per_frame_maps = []
    
    for i, frame in enumerate(frames):
        # Process single frame
        frame_tensor = processor["video"]([frame])
        frame_tensor = frame_tensor.half().to(DEVICE)
        
        # Generate with attention output
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                images=[(frame_tensor, "video")],
                do_sample=False,
                max_new_tokens=16,
                output_attentions=True,
                return_dict_in_generate=True
            )
            
        if not out.attentions:
            print(f"No attention maps returned for frame {i}")
            continue
            
        # Extract attention heat map
        attn_layers = torch.stack([torch.stack(layer) for layer in out.attentions])
        attn_avg = attn_layers.mean(dim=(0,1))
        vis_slice = attn_avg[:, :PATCHES]
        text_len = attn_avg.size(0) - PATCHES
        text_to_vis = vis_slice[-text_len:].mean(dim=0)
        heat = text_to_vis.reshape(NUM_PATCH_SIDE, NUM_PATCH_SIDE).cpu().numpy()
        
        per_frame_maps.append(heat)
        
    return per_frame_maps

def visualize_per_frame_heatmaps(frames, per_frame_maps, out_dir):
    """Create visualization of per-frame attention heatmaps"""
    per_frame_dir = ensure_dir(os.path.join(out_dir, "per_frame"))
    
    # Create a grid of frames
    rows = min(2, len(frames))
    cols = min(4, int(np.ceil(len(frames) / rows)))
    fig = plt.figure(figsize=(cols*3, rows*3))
    
    for i, (frame, heat_map) in enumerate(zip(frames, per_frame_maps)):
        ax = fig.add_subplot(rows, cols, i+1)
        
        frame_np = np.array(frame)
        heatmap = colourise(heat_map)
        overlay = cv2.addWeighted(frame_np[..., ::-1], 0.4, heatmap, 0.6, 0)
        
        ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Frame {i+1}")
        ax.axis('off')
        
        # Save individual overlay
        cv2.imwrite(os.path.join(per_frame_dir, f"frame_{i+1}_heatmap.png"), overlay)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "per_frame_heatmaps.png"))
    plt.close(fig)
    print(f"Saved per-frame heatmaps to {out_dir}/per_frame_heatmaps.png")

# ────────── Feature 2: Adversarial attack analysis ───────────────────

def create_adversarial_video(vid_tensor, model, input_ids, attn_mask, epsilon=ADVERSARIAL_EPSILON):
    """Create an adversarial video by maximizing attention drift"""
    # Clone tensor and enable gradient tracking
    adv_vid_tensor = vid_tensor.clone().detach().requires_grad_(True)
    
    # Zero gradients
    model.zero_grad()
    
    # Forward pass to get clean attention
    outputs_clean = model(
        input_ids=input_ids,
        attention_mask=attn_mask,
        images=[(vid_tensor, "video")],
        output_attentions=True)
    
    # Average across all layers and heads for stability
    clean_attn_all = torch.stack([layer.mean(dim=0) for layer in outputs_clean.attentions])
    clean_attn = clean_attn_all.mean(dim=0)[:, :PATCHES].mean(dim=0)
    
    # Get attention maps for adversarial input in same graph
    outputs_adv = model(
        input_ids=input_ids,
        attention_mask=attn_mask,
        images=[(adv_vid_tensor, "video")],
        output_attentions=True)
    
    # Extract adversarial attention maps
    adv_attn_all = torch.stack([layer.mean(dim=0) for layer in outputs_adv.attentions])
    adv_attn = adv_attn_all.mean(dim=0)[:, :PATCHES].mean(dim=0)
    
    # Loss: negative cosine similarity (to maximize difference)
    loss = F.cosine_similarity(clean_attn.view(1, -1), adv_attn.view(1, -1))
    
    # Backward pass
    (-loss).backward()  # Negate to maximize dissimilarity
    
    # Create adversarial example using gradient sign
    with torch.no_grad():
        perturbed = adv_vid_tensor + epsilon * adv_vid_tensor.grad.sign()
        perturbed = torch.clamp(perturbed, 0, 1)
    
    return perturbed.detach()

def analyze_attention_drift(model, vid_tensor, adv_vid_tensor, input_ids, attn_mask):
    """Calculate and analyze attention drift metrics"""
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
    
    # Calculate metrics
    clean_flat = clean_heat.flatten()
    adv_flat = adv_heat.flatten()
    
    # Handle NaN values
    clean_flat = np.nan_to_num(clean_flat)
    adv_flat = np.nan_to_num(adv_flat)
    
    # Cosine similarity (lower means more drift)
    cos_sim = 1 - cosine(clean_flat, adv_flat)
    
    # L2 distance (higher means more drift)
    l2_dist = np.linalg.norm(clean_flat - adv_flat)
    
    # Attention concentration (Gini coefficient)
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
        "clean_concentration": float(clean_gini),
        "adv_concentration": float(adv_gini)
    }
    
    return clean_heat, adv_heat, metrics

def visualize_attention_drift(frames, clean_heat, adv_heat, metrics, out_dir):
    """Visualize attention drift between clean and adversarial inputs"""
    adv_dir = ensure_dir(os.path.join(out_dir, "adversarial"))
    
    # Create comparison visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original frame
    frame_np = np.array(frames[0])
    axes[0].imshow(frame_np)
    axes[0].set_title("Original Frame")
    axes[0].axis('off')
    
    # Clean attention heatmap
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
    plt.figtext(0.5, 0.01, 
                f"Cosine Similarity: {metrics['cosine_similarity']:.4f} | " 
                f"L2 Distance: {metrics['l2_distance']:.4f} | " 
                f"Concentration: {metrics['clean_concentration']:.4f} → {metrics['adv_concentration']:.4f}",
                ha="center", fontsize=12, 
                bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    plt.tight_layout()
    plt.savefig(os.path.join(adv_dir, "attention_drift.png"))
    plt.close(fig)
    
    # Save individual overlays
    cv2.imwrite(os.path.join(adv_dir, "clean_heatmap.png"), clean_overlay)
    cv2.imwrite(os.path.join(adv_dir, "adversarial_heatmap.png"), adv_overlay)
    
    # Create difference heatmap
    diff_map = np.abs(clean_heat - adv_heat)
    diff_heatmap = colourise(diff_map)
    diff_overlay = cv2.addWeighted(frame_np[..., ::-1], 0.4, diff_heatmap, 0.6, 0)
    cv2.imwrite(os.path.join(adv_dir, "difference_heatmap.png"), diff_overlay)
    
    print(f"Saved adversarial analysis to {out_dir}/adversarial/")

# ───────── Feature 3: Temporal attention visualization ───────────────

def create_temporal_visualization(frames, per_frame_maps, out_dir):
    """Create a temporal visualization of attention maps"""
    temp_dir = ensure_dir(os.path.join(out_dir, "temporal"))
    
    # Create a grid visualization
    rows = min(2, len(frames))
    cols = min(4, int(np.ceil(len(frames) / rows)))
    fig = plt.figure(figsize=(cols*4, rows*4))
    
    for i, (frame, heat_map) in enumerate(zip(frames, per_frame_maps)):
        ax = fig.add_subplot(rows, cols, i+1)
        
        frame_np = np.array(frame)
        heatmap = colourise(heat_map)
        overlay = cv2.addWeighted(frame_np[..., ::-1], 0.4, heatmap, 0.6, 0)
        
        ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Frame {i+1}")
        ax.axis('off')
        
        # Save individual frame
        cv2.imwrite(os.path.join(temp_dir, f"frame_{i+1}_overlay.png"), overlay)
    
    plt.tight_layout()
    plt.savefig(os.path.join(temp_dir, "temporal_grid.png"))
    plt.close(fig)
    
    # Try to create an animated GIF if matplotlib animation is available
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        def update(i):
            ax.clear()
            frame_np = np.array(frames[i])
            heatmap = colourise(per_frame_maps[i])
            overlay = cv2.addWeighted(frame_np[..., ::-1], 0.4, heatmap, 0.6, 0)
            ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            ax.set_title(f"Frame {i+1} Attention")
            ax.axis('off')
            return [ax]
        
        ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=500, blit=False)
        
        # Try to save as GIF
        try:
            ani.save(os.path.join(temp_dir, "attention_over_time.gif"), writer='pillow', fps=2)
            print(f"Saved GIF animation to {temp_dir}/attention_over_time.gif")
        except Exception as e:
            print(f"Could not save GIF: {e}")
            
        # Try saving as MP4 if ffmpeg is available
        try:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=2, metadata=dict(artist='VideoLLaMA2'), bitrate=1800)
            ani.save(os.path.join(temp_dir, "attention_over_time.mp4"), writer=writer)
            print(f"Saved MP4 animation to {temp_dir}/attention_over_time.mp4")
        except Exception as e:
            print(f"Could not save MP4: {e}")
            
        plt.close(fig)
        
    except Exception as e:
        print(f"Could not create animation: {e}")
    
    print(f"Saved temporal visualization to {out_dir}/temporal/")

# ─────────────────────────  main flow  ─────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VideoLLaMA 2 Enhanced Attention Analysis")
    parser.add_argument("video", type=str, help="Path to input video")
    parser.add_argument("--prompt", type=str, default="Describe what is happening in the video.",
                        help="Prompt for the model")
    parser.add_argument("--frames", type=int, default=NUM_FRAMES, 
                        help=f"Number of frames to process (default: {NUM_FRAMES})")
    parser.add_argument("--epsilon", type=float, default=ADVERSARIAL_EPSILON,
                        help=f"Epsilon for adversarial attack (default: {ADVERSARIAL_EPSILON})")
    parser.add_argument("--output_dir", type=str, default=OUT_DIR,
                        help=f"Output directory (default: {OUT_DIR})")
    parser.add_argument("--no_perframe", action="store_true", help="Skip per-frame analysis")
    parser.add_argument("--no_adv", action="store_true", help="Skip adversarial analysis")
    parser.add_argument("--no_temporal", action="store_true", help="Skip temporal visualization")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    ensure_dir(args.output_dir)
    
    model, processor, tokenizer = load_model()
    
    # Load video frames
    print(f"Loading video frames from {args.video}...")
    frames = load_frames(args.video, args.frames)
    if not frames:
        print(f"Error: Could not load frames from {args.video}")
        return
        
    print(f"Successfully loaded {len(frames)} frames")
    
    # Process video frames into tensor
    vid_tensor = processor["video"](frames)  # (T, C, H, W) float32 0‑1
    vid_tensor = vid_tensor.half().to(DEVICE)

    # Build prompt with <video> token
    chat_str = tokenizer.apply_chat_template([
        {"role": "user", "content": DEFAULT_VIDEO_TOKEN + "\n" + args.prompt}],
        tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer_multimodal_token(chat_str, tokenizer, DEFAULT_VIDEO_TOKEN,
                                         return_tensors="pt").to(DEVICE)
    attn_mask = torch.ones_like(input_ids, device=DEVICE)

    # Generate caption with original attention visualization
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
    cv2.imwrite(os.path.join(args.output_dir, "heatmap_overlay.png"), overlay)
    print("Saved original heatmap_overlay.png →", args.output_dir)

    # Feature 1: Per-frame heatmaps
    if not args.no_perframe:
        print("\n[1/3] Extracting per-frame heatmaps...")
        per_frame_maps = extract_per_frame_attention(model, frames, processor, input_ids, attn_mask)
        visualize_per_frame_heatmaps(frames, per_frame_maps, args.output_dir)
    else:
        per_frame_maps = None
        print("\n[1/3] Skipping per-frame analysis (--no_perframe flag set)")
    
    # Feature 2: Adversarial attack analysis
    if not args.no_adv:
        print("\n[2/3] Analyzing attention drift under adversarial attack...")
        adv_vid_tensor = create_adversarial_video(vid_tensor, model, input_ids, attn_mask, args.epsilon)
        clean_heat, adv_heat, metrics = analyze_attention_drift(
            model, vid_tensor, adv_vid_tensor, input_ids, attn_mask
        )
        visualize_attention_drift(frames, clean_heat, adv_heat, metrics, args.output_dir)
    else:
        print("\n[2/3] Skipping adversarial analysis (--no_adv flag set)")
    
    # Feature 3: Temporal attention visualization
    if not args.no_temporal and per_frame_maps:
        print("\n[3/3] Creating temporal attention visualization...")
        create_temporal_visualization(frames, per_frame_maps, args.output_dir)
    else:
        print("\n[3/3] Skipping temporal visualization (--no_temporal flag set or no per-frame maps)")
    
    print("\nAnalysis complete! All visualizations saved to", args.output_dir)


if __name__ == "__main__":
    main()
