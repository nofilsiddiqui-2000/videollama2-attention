#!/usr/bin/env python3
# videollama_fgsm_attack_corrected.py • Fixed FGSM attacks on VideoLLaMA-2

import os, sys, cv2, argparse, math
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPImageProcessor
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

# ── Disable FlashAttention-2 (simplified) ────────────────────────
os.environ.update({
    "PYTORCH_ATTENTION_IMPLEMENTATION": "eager",
    "HF_DISABLE_FLASH_ATTN_2": "1",
    "DISABLE_FLASH_ATTN_2": "1",
})

MODEL_NAME = "DAMO-NLP-SG/VideoLLaMA2-7B-16F"
VISION_ID = "openai/clip-vit-large-patch14-336"

def load_models(device="cuda"):
    """Load both CLIP (for attention visualization) and VideoLLaMA-2 (for attacks)"""
    # CLIP for attention visualization
    vt = CLIPVisionModel.from_pretrained(VISION_ID).to(device).eval()
    vproc = CLIPImageProcessor.from_pretrained(VISION_ID)

    # VideoLLaMA-2 for attacks and captioning
    disable_torch_init()
    vlm, vprocessor, tok = model_init(
        MODEL_NAME, 
        attn_implementation="eager",
        torch_dtype=torch.float16, 
        device_map=device
    )
    vlm.eval()
    return vt, vproc, vlm, vprocessor, tok

def get_processor_normalization_params(vprocessor):
    """Extract normalization parameters from VideoLLaMA processor"""
    # VideoLLaMA typically uses [-1, 1] normalization
    # We need to find the actual parameters used by the processor
    try:
        # Try to get from processor config
        if hasattr(vprocessor['video'], 'image_mean'):
            mean = vprocessor['video'].image_mean
            std = vprocessor['video'].image_std
        else:
            # Default CLIP normalization converted to [-1, 1] range
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
    except:
        # Fallback to common values
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    
    return mean, std

def fgsm_attack_video(video_path, vlm, vprocessor, tok, epsilon=0.03, device="cuda"):
    """Apply FGSM attack to video tensor for VideoLLaMA-2"""
    # Process original video - keep in fp32 for gradients
    vid_tensor = vprocessor["video"](video_path).to(device)  # fp32 for stable gradients
    
    # Get normalization bounds
    mean, std = get_processor_normalization_params(vprocessor)
    mean = torch.tensor(mean, device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor(std, device=device).view(1, 1, 3, 1, 1)
    
    # Calculate proper clipping bounds (typically [-1, 1] after normalization)
    min_val = (-1 - mean) / std
    max_val = (1 - mean) / std
    
    # Get original caption and features
    with torch.inference_mode():
        # Convert to half precision only for inference
        vid_half = vid_tensor.half()
        original_caption = mm_infer(
            vid_half, "Describe the video in detail.",
            model=vlm, tokenizer=tok, modal="video",
            do_sample=False
        ).strip()
        
        # Get original features for comparison
        original_features = vlm.model.vision_tower(vid_half).last_hidden_state.detach()
    
    # Prepare adversarial tensor (keep in fp32 for gradients)
    vid_adv = vid_tensor.clone().detach().requires_grad_(True)
    
    # Forward pass to get adversarial features
    vision_outputs = vlm.model.vision_tower(vid_adv.half())
    adv_features = vision_outputs.last_hidden_state
    
    # Loss: maximize difference between original and adversarial features
    # Using cosine similarity loss for better gradient flow
    original_flat = original_features.view(-1, original_features.size(-1))
    adv_flat = adv_features.view(-1, adv_features.size(-1))
    
    # Minimize cosine similarity (maximize difference)
    cos_sim = F.cosine_similarity(original_flat, adv_flat, dim=-1).mean()
    loss = cos_sim  # Minimize similarity
    
    # Backward pass
    loss.backward()
    
    # FGSM perturbation with proper clipping
    with torch.inference_mode():
        perturbation = epsilon * vid_adv.grad.sign()
        vid_adv_final = torch.clamp(vid_adv + perturbation, min_val, max_val).detach()
    
    # Clear gradients to free memory
    vid_adv.grad = None
    vid_adv.requires_grad_(False)
    
    # Generate adversarial caption
    with torch.inference_mode():
        adv_caption = mm_infer(
            vid_adv_final.half(), "Describe the video in detail.",
            model=vlm, tokenizer=tok, modal="video",
            do_sample=False
        ).strip()
    
    # Calculate similarity metrics
    with torch.inference_mode():
        final_features = vlm.model.vision_tower(vid_adv_final.half()).last_hidden_state
        final_flat = final_features.view(-1, final_features.size(-1))
        final_similarity = F.cosine_similarity(original_flat, final_flat, dim=-1).mean().item()
    
    print(f"🎯 Attack success: cosine similarity {final_similarity:.4f} (lower = more different)")
    
    return vid_adv_final, original_caption, adv_caption, vid_tensor, final_similarity

@torch.inference_mode()
def vit_rollout(pil_img, vt, vproc, device="cuda"):
    """Generate attention heatmap using CLIP ViT with proper rollout"""
    inp = vproc(images=pil_img, return_tensors="pt").to(device)
    outs = vt(**inp, output_attentions=True)

    # Attention rollout (Chefer et al.)
    attentions = torch.stack(outs.attentions)[:, 0].mean(1)  # Average heads
    eye = torch.eye(attentions.size(-1), device=device)
    
    # Rollout computation
    rollout_matrix = eye
    for attention in attentions:
        # Add residual connection
        attention = attention + eye
        # Normalize
        attention = attention / attention.sum(-1, keepdim=True)
        # Multiply with previous rollout
        rollout_matrix = attention @ rollout_matrix

    # Extract attention to visual tokens (exclude CLS token)
    n_vis = rollout_matrix.size(-1) - 1
    side = int(math.sqrt(n_vis))
    
    # Get attention from CLS to visual tokens
    heat = rollout_matrix[0, 1:1+side*side]
    heat = heat.reshape(side, side).detach().cpu().numpy()

    # Smooth and normalize
    heat = cv2.GaussianBlur(heat, (0, 0), 3)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    return np.power(heat, 0.5)  # Gamma correction for better visualization

def tensor_to_frames(vid_tensor, mean=None, std=None):
    """Convert video tensor back to PIL frames with proper denormalization"""
    # vid_tensor shape: [1, T, C, H, W]
    frames = []
    vid_np = vid_tensor.squeeze(0).cpu().numpy()  # [T, C, H, W]
    
    # Denormalize if parameters provided
    if mean is not None and std is not None:
        mean = np.array(mean).reshape(1, 3, 1, 1)
        std = np.array(std).reshape(1, 3, 1, 1)
        vid_np = vid_np * std + mean
    
    # Clip to valid range and convert to uint8
    vid_np = np.clip(vid_np, 0, 1)
    
    for i in range(vid_np.shape[0]):
        frame = vid_np[i].transpose(1, 2, 0)  # [H, W, C]
        frame = (frame * 255).astype(np.uint8)
        frame_pil = Image.fromarray(frame)
        frames.append(frame_pil)
    
    return frames

def process_video_frames(frames, vt, vproc, device, output_dir, prefix, alpha=0.35):
    """Process frames with vectorized operations when possible"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    energies = []
    jet = plt.colormaps.get_cmap("jet")
    
    print(f"🔍 Processing {len(frames)} {prefix} frames...")
    
    for idx, pil_frame in enumerate(frames):
        # Generate attention heatmap
        heat = vit_rollout(pil_frame, vt, vproc, device)
        energies.append(float(heat.mean()))
        
        # Convert PIL to OpenCV format
        frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
        
        # Resize heatmap to frame size
        heat_resized = cv2.resize(heat, (frame.shape[1], frame.shape[0]), cv2.INTER_LINEAR)
        
        # Create colorized overlay
        h_rgba = (jet(heat_resized) * 255).astype(np.uint8)
        h_bgr = cv2.cvtColor(cv2.cvtColor(h_rgba, cv2.COLOR_RGBA2RGB), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(frame, 1-alpha, h_bgr, alpha, 0)
        
        # Add annotations
        cv2.putText(overlay, f"{prefix} frame {idx}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Mark attention peak
        y, x = np.unravel_index(np.argmax(heat_resized), heat_resized.shape)
        cv2.circle(overlay, (x, y), 12, (0, 0, 255), 3)
        
        # Save frame
        output_path = output_dir / f"{prefix}_frame_{idx:04d}.png"
        cv2.imwrite(str(output_path), overlay)
        
        if idx % 5 == 0:
            print(f"   {prefix}: {idx}/{len(frames)}", end='\r')
    
    print(f"   {prefix}: {len(frames)}/{len(frames)} ✓")
    return energies

def main():
    ap = argparse.ArgumentParser(description="FGSM attack on VideoLLaMA-2 with attention visualization")
    ap.add_argument("video", help="Input video path")
    ap.add_argument("--out", default="fgsm_attention_results", help="Output directory")
    ap.add_argument("--epsilon", type=float, default=0.03, help="FGSM epsilon (default: 0.03)")
    ap.add_argument("--alpha", type=float, default=0.35, help="Attention overlay alpha (default: 0.35)")
    ap.add_argument("--caption-file", default="captions.txt", help="Caption output file")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("❌ CUDA GPU required for VideoLLaMA-2")
    
    device = "cuda"
    print("⏳ Loading models...")
    vt, vproc, vlm, vprocessor, tok = load_models(device)

    print(f"🎯 Applying FGSM attack (ε={args.epsilon})...")
    vid_adv, original_caption, adv_caption, vid_original, similarity = fgsm_attack_video(
        args.video, vlm, vprocessor, tok, args.epsilon, device
    )
    
    print(f"\n📝 Original caption: {original_caption}")
    print(f"📝 Adversarial caption: {adv_caption}")
    
    # Get normalization parameters for proper frame conversion
    mean, std = get_processor_normalization_params(vprocessor)
    
    # Convert tensors to frames with proper denormalization
    print("🖼️ Converting tensors to frames...")
    original_frames = tensor_to_frames(vid_original, mean, std)
    adversarial_frames = tensor_to_frames(vid_adv, mean, std)
    
    # Process frames and generate attention visualizations
    original_energies = process_video_frames(
        original_frames, vt, vproc, device, args.out, "original", args.alpha
    )
    
    adversarial_energies = process_video_frames(
        adversarial_frames, vt, vproc, device, args.out, "adversarial", args.alpha
    )
    
    # Generate comparison plots
    print("📊 Generating analysis plots...")
    output_dir = Path(args.out)
    
    # Normalize energy curves
    orig_curve = np.array(original_energies)
    adv_curve = np.array(adversarial_energies)
    
    if len(orig_curve) > 0 and orig_curve.max() > orig_curve.min():
        orig_curve = (orig_curve - orig_curve.min()) / (orig_curve.max() - orig_curve.min())
    if len(adv_curve) > 0 and adv_curve.max() > adv_curve.min():
        adv_curve = (adv_curve - adv_curve.min()) / (adv_curve.max() - adv_curve.min())
    
    # Create comprehensive comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original attention curve
    axes[0, 0].plot(orig_curve, 'o-', color='blue', alpha=0.7)
    axes[0, 0].set_title("Original Video - Attention Energy")
    axes[0, 0].set_xlabel("Frame")
    axes[0, 0].set_ylabel("Normalized Energy")
    axes[0, 0].grid(True)
    
    # Adversarial attention curve
    axes[0, 1].plot(adv_curve, 'o-', color='red', alpha=0.7)
    axes[0, 1].set_title("Adversarial Video - Attention Energy")
    axes[0, 1].set_xlabel("Frame")
    axes[0, 1].set_ylabel("Normalized Energy")
    axes[0, 1].grid(True)
    
    # Direct comparison
    axes[1, 0].plot(orig_curve, 'o-', color='blue', label='Original', alpha=0.7)
    axes[1, 0].plot(adv_curve, 'o-', color='red', label='Adversarial', alpha=0.7)
    axes[1, 0].set_title("Attention Energy Comparison")
    axes[1, 0].set_xlabel("Frame")
    axes[1, 0].set_ylabel("Normalized Energy")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Difference plot
    if len(orig_curve) == len(adv_curve):
        diff = adv_curve - orig_curve
        axes[1, 1].plot(diff, 'o-', color='purple', alpha=0.7)
        axes[1, 1].set_title("Attention Difference (Adv - Orig)")
        axes[1, 1].set_xlabel("Frame")
        axes[1, 1].set_ylabel("Energy Difference")
        axes[1, 1].grid(True)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    else:
        axes[1, 1].text(0.5, 0.5, "Frame count mismatch", ha='center', va='center')
        axes[1, 1].set_title("Attention Difference")
    
    plt.tight_layout()
    plt.savefig(output_dir / "comprehensive_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save detailed results
    caption_path = Path(args.caption_file)
    with open(caption_path, 'w', encoding='utf-8') as f:
        f.write(f"FGSM Attack Results (ε={args.epsilon})\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Original Caption:\n{original_caption}\n\n")
        f.write(f"Adversarial Caption:\n{adv_caption}\n\n")
        f.write(f"Feature Similarity: {similarity:.6f}\n")
        f.write(f"Attention MSE: {np.mean((orig_curve - adv_curve) ** 2):.6f}\n")
    
    print(f"✅ Results saved to {args.out}")
    print(f"📝 Captions saved to {args.caption_file}")
    print(f"📊 Feature similarity: {similarity:.6f}")
    print(f"📊 Attention MSE: {np.mean((orig_curve - adv_curve) ** 2):.6f}")

if __name__ == "__main__":
    main()
