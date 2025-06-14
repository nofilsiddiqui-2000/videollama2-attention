#!/usr/bin/env python3
"""
Adversarial Attention Rollout on Video Frames with CLIP ViT-L/14-336

This script:
1. Loads the pretrained OpenAI CLIP ViT-L/14-336 model (Vision Transformer) via HuggingFace Transformers.
2. Reads an input video, samples 16 evenly spaced frames.
3. Applies a Fast Gradient Sign Method (FGSM) adversarial attack to each frame to perturb the image (maximizing classification loss).
4. For each adversarial frame, computes the Vision Transformer attention rollout, creates a heatmap (jet colormap) and overlays it on the frame.
5. Saves each overlay image in the output directory (frame_0001.png, ...).
6. Optionally (via --save_curve flag), computes the mean attention for each frame and plots a temporal attention curve.

Usage:
    python adversarial_clip_attention.py \
        --input_video path/to/video.mp4 \
        --output_dir path/to/output_frames/ \
        --epsilon 0.03 \
        --batch_size 4 \
        [--save_curve] \
        [--output_caption dummy.txt]
"""
import os
import sys
import argparse
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Adversarial attack + attention rollout on video frames using CLIP ViT-L/14-336.")
    parser.add_argument("--input_video", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output frames and heatmaps.")
    parser.add_argument("--epsilon", type=float, default=0.03, help="FGSM epsilon (perturbation magnitude).")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing frames through the model.")
    parser.add_argument("--output_caption", type=str, default="dummy.txt", help="Output caption file (not used, dummy).")
    parser.add_argument("--save_curve", action='store_true', help="Flag to save temporal attention curve plot.")
    return parser.parse_args()

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def load_clip_model(device):
    """
    Load the CLIP ViT-L/14-336 model and processor.
    Use half precision on CUDA if available, and disable flash attention if possible.
    """
    # Attempt to disable any flash attention (if implemented in backend)
    os.environ["DISABLE_FLASH_ATTENTION"] = "1"
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336", torch_dtype=torch.float16)
    model = model.to(device)
    model.eval()
    return model

def sample_frames_from_video(video_path, num_frames=16):
    """
    Extracts `num_frames` frames evenly from the video at `video_path`.
    Returns a list of RGB images (as NumPy arrays).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        raise ValueError("Video has no frames.")
    indices = np.linspace(0, frame_count - 1, num=num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Failed to read frame at index {idx}. Skipping.")
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames

def fgsm_attack(model, processor, frames, epsilon, device):
    """
    Performs an FGSM attack on a batch of frames.
    Returns a list of adversarial frames as uint8 images.
    """
    # Preprocess frames
    inputs = processor(images=frames, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device)  # shape: (B, 3, H, W)
    pixel_values = pixel_values.half()
    pixel_values.requires_grad_(True)
    # Forward pass to get image embeddings (used as logits proxy)
    image_features = model.get_image_features(pixel_values)
    # Original predicted 'class' for each image
    orig_labels = torch.argmax(image_features, dim=1)
    # Compute cross-entropy loss (treating features as logits and orig_labels as targets)
    loss = F.cross_entropy(image_features, orig_labels)
    loss.backward()
    # FGSM step: add epsilon * sign(gradient)
    grad_sign = pixel_values.grad.sign()
    adv_pixel_values = pixel_values + epsilon * grad_sign
    # Ensure values stay within valid normalized range
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1,3,1,1)
    clip_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1,3,1,1)
    norm_min = (0.0 - clip_mean) / clip_std
    norm_max = (1.0 - clip_mean) / clip_std
    adv_pixel_values = torch.max(torch.min(adv_pixel_values, norm_max), norm_min).detach()
    # Denormalize to [0,1]
    adv_images = adv_pixel_values * clip_std + clip_mean
    adv_images = adv_images.clamp(0.0, 1.0)
    adv_images_cpu = adv_images.cpu().permute(0,2,3,1).numpy()  # (B, H, W, 3)
    adv_frames = []
    for img in adv_images_cpu:
        img_uint8 = (img * 255.0).astype(np.uint8)
        adv_frames.append(img_uint8)
    return adv_frames

def compute_attention_rollout(attentions):
    """
    Compute attention rollout from a list of attention matrices.
    Args:
      - attentions: tuple of tensors, each (batch, heads, seq_len, seq_len)
    Returns:
      - rollout: Tensor of shape (batch, seq_len, seq_len)
    """
    result = None
    for attn_layer in attentions:
        attn_heads_fused = attn_layer.mean(dim=1)  # average over heads (batch, seq, seq)
        batch_size, seq_len, _ = attn_heads_fused.shape
        identity = torch.eye(seq_len, device=attn_heads_fused.device).unsqueeze(0).repeat(batch_size, 1, 1)
        attn_heads_fused = attn_heads_fused + identity
        attn_heads_fused = attn_heads_fused / attn_heads_fused.sum(dim=-1, keepdim=True)
        if result is None:
            result = attn_heads_fused
        else:
            result = attn_heads_fused @ result
    return result

def overlay_heatmap_on_image(image_np, att_map):
    """
    Overlay a heatmap on the image.
    - image_np: HxWx3 uint8 RGB image.
    - att_map: 2D array of attention values (0-1 normalized).
    """
    heatmap = cv2.resize(att_map, (image_np.shape[1], image_np.shape[0]))
    heatmap = np.clip(heatmap, 0.0, 1.0)
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_np, 0.6, heatmap_color, 0.4, 0)
    return overlay

def save_temporal_curve(energies, output_path):
    plt.figure()
    plt.plot(range(1, len(energies)+1), energies, marker='o')
    plt.xlabel("Frame Index")
    plt.ylabel("Mean Attention Energy")
    plt.title("Temporal Attention Curve")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.output_dir)
    try:
        model = load_clip_model(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    except Exception as e:
        print(f"Error loading CLIP model or processor: {e}")
        sys.exit(1)
    try:
        frames = sample_frames_from_video(args.input_video, num_frames=16)
    except Exception as e:
        print(f"Error reading video: {e}")
        sys.exit(1)
    num_frames = len(frames)
    if num_frames == 0:
        print("No frames extracted from video.")
        sys.exit(1)

    # Generate adversarial frames in batches
    adv_frames = []
    for i in range(0, num_frames, args.batch_size):
        batch_frames = frames[i:i+args.batch_size]
        adv_batch = fgsm_attack(model, processor, batch_frames, args.epsilon, device)
        adv_frames.extend(adv_batch)

    # Process each adversarial frame: attention rollout and overlay
    energies = []
    for idx, adv_frame in enumerate(adv_frames, start=1):
        inputs = processor(images=adv_frame, return_tensors="pt").to(device)
        pixel_values = inputs.pixel_values.half()
        with torch.no_grad():
            vision_outputs = model.vision_model(pixel_values, output_attentions=True)
        attentions = vision_outputs.attentions  # tuple of (batch, heads, seq, seq)
        rollout = compute_attention_rollout(attentions)  # shape (1, seq, seq)
        # Extract attention from class token (index 0) to patches (skip itself)
        cls_attention = rollout[0, 0, 1:]  # (num_patches,)
        num_patches = cls_attention.shape[0]
        grid_size = int(math.sqrt(num_patches))
        att_map = cls_attention.reshape(grid_size, grid_size).cpu().numpy()
        att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)
        overlay_img = overlay_heatmap_on_image(adv_frame, att_map)
        frame_filename = os.path.join(args.output_dir, f"frame_{idx:04d}.png")
        cv2.imwrite(frame_filename, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
        energies.append(float(att_map.mean()))

    # Save temporal attention curve if requested
    if args.save_curve:
        curve_path = os.path.join(args.output_dir, "attention_curve.png")
        try:
            save_temporal_curve(energies, curve_path)
        except Exception as e:
            print(f"Warning: Failed to save attention curve: {e}")

    # Write dummy caption file if requested
    if args.output_caption:
        try:
            with open(args.output_caption, 'w') as f:
                f.write("Dummy caption.\n")
        except Exception as e:
            print(f"Warning: Could not write caption file: {e}")

if __name__ == "__main__":
    main()
