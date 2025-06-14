#!/usr/bin/env python3
"""
Adversarial Video Captioning Script
This script performs the following tasks:
1. Load a video and sample 16 uniformly spaced frames.
2. Compute CLIP (ViT-L/14-336) attention rollout heatmaps on clean frames.
3. Apply an untargeted FGSM attack on the frames to minimize cosine similarity of CLIP embeddings.
4. Save adversarial frames, attention heatmaps, cosine similarities, and reconstruct an adversarial video.
5. Use VideoLLaMA2 (7B-16F) to generate a caption for the adversarial video.

References:
- CLIP model usage and preprocessing:contentReference[oaicite:7]{index=7}:contentReference[oaicite:8]{index=8}.
- FGSM (Goodfellow et al., 2014) for adversarial perturbations:contentReference[oaicite:9]{index=9}:contentReference[oaicite:10]{index=10}.
- Attention rollout method (Abnar & Zuidema 2020):contentReference[oaicite:11]{index=11}.
- VideoLLaMA2 inference example:contentReference[oaicite:12]{index=12}.
"""

import argparse
import os
import math
import cv2
import torch
import numpy as np

from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torch.nn.functional as F

# Attempt to import VideoLLaMA2 modules
try:
    from videollama2 import model_init, mm_infer
    from videollama2.utils import disable_torch_init
except ImportError:
    model_init = mm_infer = disable_torch_init = None

def compute_attention_rollout(attentions):
    """
    Compute attention rollout for one image.
    attentions: list of tensors [num_heads, seq_len, seq_len] for each layer.
    Returns a 1D tensor mask of length (seq_len-1) for the [CLS]-to-patch attention.
    """
    device = attentions[0].device
    seq_len = attentions[0].size(-1)
    # Start with identity matrix (accounting for residual connections)
    rollout = torch.eye(seq_len, seq_len, device=device)
    for attn in attentions:
        # Average over attention heads
        avg_attn = attn.mean(dim=0)
        # Include residual (self-attention) by adding identity
        avg_attn = avg_attn + torch.eye(seq_len, seq_len, device=device)
        # Normalize rows
        avg_attn = avg_attn / avg_attn.sum(dim=-1, keepdim=True)
        # Multiply into the rollout
        rollout = avg_attn @ rollout
    # The [CLS] token is at index 0; return its attention to patch tokens (ignore itself)
    mask = rollout[0, 1:]
    return mask

def main():
    parser = argparse.ArgumentParser(description="Video adversarial attack and captioning")
    parser.add_argument('--input_video', type=str, required=True, help="Path to input video file")
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory for results")
    parser.add_argument('--epsilon', type=float, required=True, help="FGSM perturbation magnitude")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for CLIP processing")
    parser.add_argument('--save_curve', action='store_true', help="Save temporal attention energy curve")
    parser.add_argument('--output_caption', type=str, required=True, help="File path to save the generated caption")
    args = parser.parse_args()

    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Validate input video
    if not os.path.isfile(args.input_video):
        raise FileNotFoundError(f"Input video not found: {args.input_video}")

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    frames_clean_dir = os.path.join(args.output_dir, "frames_clean")
    frames_adv_dir   = os.path.join(args.output_dir, "frames_adv")
    os.makedirs(frames_clean_dir, exist_ok=True)
    os.makedirs(frames_adv_dir, exist_ok=True)
    
    # Prepare caption output directory
    cap_dir = os.path.dirname(args.output_caption)
    if cap_dir and not os.path.isdir(cap_dir):
        os.makedirs(cap_dir, exist_ok=True)

    # Load video and sample 16 frames uniformly
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {args.input_video}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    if total_frames <= 0:
        raise ValueError("Video has no frames or cannot read frame count.")
    indices = np.linspace(0, total_frames - 1, 16, dtype=int)
    sampled_frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame at index {idx}")
        sampled_frames.append(frame)
    cap.release()

    # Initialize CLIP image processor and model
    clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14-336")
    clip_model.to(device).eval()

    # Convert sampled frames to PIL for processing
    frames_pil = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in sampled_frames]

    # Batch process frames through CLIP to get embeddings and attentions
    all_embeddings = []
    all_attentions = []
    for start in range(0, len(frames_pil), args.batch_size):
        batch_pil = frames_pil[start:start+args.batch_size]
        inputs = clip_processor(images=batch_pil, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)
        with torch.no_grad():
            outputs = clip_model(pixel_values=pixel_values, output_attentions=True)
        embeddings = outputs.image_embeds.cpu()  # (batch, dim)
        attentions = outputs.attentions          # tuple of (batch, heads, seq, seq)
        num_layers = len(attentions)
        # Append embeddings
        all_embeddings.append(embeddings)
        # Extract per-image attentions
        for i in range(pixel_values.size(0)):
            img_attns = [attentions[layer][i] for layer in range(num_layers)]
            all_attentions.append(img_attns)
    all_embeddings = torch.cat(all_embeddings, dim=0)  # shape (16, dim)

    # Save clean-frame attention rollout heatmaps
    energy_values = []
    for i, frame in enumerate(sampled_frames):
        attn_maps = all_attentions[i]
        mask = compute_attention_rollout(attn_maps)  # (seq_len-1,)
        n_patches = int(math.sqrt(mask.size(0)))
        heat = mask.reshape(n_patches, n_patches).detach().cpu().numpy()
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        heat_uint8 = np.uint8(255 * heat)
        heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        # Resize heatmap to frame size and overlay
        h, w = frame.shape[:2]
        heat_color = cv2.resize(heat_color, (w, h))
        overlay = cv2.addWeighted(frame, 0.6, heat_color, 0.4, 0)
        clean_path = os.path.join(frames_clean_dir, f"frame_{i:02d}.png")
        cv2.imwrite(clean_path, overlay)
        # Compute an "energy" metric (e.g., sum of squares of heat values)
        energy = float((heat ** 2).sum())
        energy_values.append(energy)

    # Save temporal attention energy curve if requested
    if args.save_curve:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required to save the attention energy curve.")
        plt.figure()
        plt.plot(range(len(energy_values)), energy_values, marker='o')
        plt.xlabel("Frame index")
        plt.ylabel("Attention energy")
        plt.title("Temporal Attention Energy")
        curve_path = os.path.join(args.output_dir, "attention_energy_curve.png")
        plt.savefig(curve_path)
        plt.close()

    # Apply untargeted FGSM attack on each frame
    cos_values = []
    # Prepare CLIP normalization parameters (mean, std)
    mean = torch.tensor(clip_processor.image_mean).to(device).view(1, -1, 1, 1)
    std  = torch.tensor(clip_processor.image_std).to(device).view(1, -1, 1, 1)
    for i, img_pil in enumerate(frames_pil):
        inputs = clip_processor(images=img_pil, return_tensors="pt")
        x = inputs['pixel_values'].to(device).clone().detach().requires_grad_(True)
        f_clean = all_embeddings[i].to(device)
        outputs = clip_model(pixel_values=x, output_attentions=False)
        f_adv = outputs.image_embeds  # (1, dim)
        # Cosine similarity loss; we want to minimize similarity:contentReference[oaicite:13]{index=13}
        loss = F.cosine_similarity(f_clean.unsqueeze(0), f_adv, dim=1)
        loss.backward()
        # FGSM step: subtract epsilon * sign(grad)
        with torch.no_grad():
            x_adv = x - args.epsilon * x.grad.sign()
            x_adv.clamp_(0, 1)
        # Compute new embedding and similarity
        outputs_adv = clip_model(pixel_values=x_adv, output_attentions=False)
        f_adv2 = outputs_adv.image_embeds
        cos_new = F.cosine_similarity(f_clean.unsqueeze(0), f_adv2, dim=1)
        cos_values.append(float(cos_new.cpu().item()))
        # Save adversarial frame (denormalize to [0,255])
        x_un = x_adv * std + mean
        img_np = (x_un.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0,255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        adv_path = os.path.join(frames_adv_dir, f"frame_{i:02d}.png")
        cv2.imwrite(adv_path, img_bgr)

    # Save cosine similarities
    cos_file = os.path.join(args.output_dir, "cosine_similarity.txt")
    with open(cos_file, 'w') as f:
        for val in cos_values:
            f.write(f"{val}\n")

    # Reconstruct adversarial video from perturbed frames
    height, width = sampled_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(args.output_dir, "adversarial.mp4")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise IOError("Failed to open video writer for adversarial video.")
    for fname in sorted(os.listdir(frames_adv_dir)):
        frame = cv2.imread(os.path.join(frames_adv_dir, fname))
        if frame is not None:
            writer.write(frame)
    writer.release()

    # Generate caption using VideoLLaMA2 (7B-16F):contentReference[oaicite:14]{index=14}
    if model_init is None or mm_infer is None:
        print("VideoLLaMA2 not available; skipping caption generation.")
        return

    disable_torch_init()
    vlm_model, vlm_processor, vlm_tokenizer = model_init("DAMO-NLP-SG/VideoLLaMA2-7B-16F")
    if torch.cuda.is_available():
        vlm_model.to(device)
    instruction = "Generate a descriptive caption for the video."
    video_inputs = vlm_processor['video'](video_path)
    caption = mm_infer(video_inputs, instruction, model=vlm_model, tokenizer=vlm_tokenizer,
                       do_sample=False, modal='video')
    # Save caption to file
    with open(args.output_caption, 'w') as f:
        f.write(caption)
    print(f"Caption saved to {args.output_caption}")

if __name__ == "__main__":
    main()
