#!/usr/bin/env python3
# videollama_fgsm_attack.py • FGSM attacks on VideoLLaMA-2 with attention visualization

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

# ── Disable FlashAttention-2 ────────────────────────────────────
import transformers.modeling_utils as _mu
_mu._check_and_enable_flash_attn_2 = lambda *_, **__: False
_mu.PreTrainedModel._check_and_enable_flash_attn_2 = staticmethod(lambda *_, **__: False)
os.environ.update({
    "PYTORCH_ATTENTION_IMPLEMENTATION": "eager",
    "HF_DISABLE_FLASH_ATTN_2": "1",
    "DISABLE_FLASH_ATTN_2": "1",
})

MODEL_NAME = "DAMO-NLP-SG/VideoLLaMA2-7B-16F"
VISION_ID = "openai/clip-vit-large-patch14-336"
CLS_TOKEN = 1

# ────────────────────────────────────────────────────────────────
def load_models(device="cuda"):
    """Load both CLIP (for attention visualization) and VideoLLaMA-2 (for attacks)"""
    # CLIP for attention visualization
    vt = CLIPVisionModel.from_pretrained(VISION_ID).to(device).eval()
    vproc = CLIPImageProcessor.from_pretrained(VISION_ID)

    # VideoLLaMA-2 for attacks and captioning
    disable_torch_init()
    vlm, vprocessor, tok = model_init(
        MODEL_NAME, attn_implementation="eager",
        torch_dtype=torch.float16, device_map=device
    )
    vlm.eval()
    return vt, vproc, vlm, vprocessor, tok

# ────────────────────────────────────────────────────────────────
def fgsm_attack_video(video_path, vlm, vprocessor, tok, epsilon=0.03, device="cuda"):
    """Apply FGSM attack to video tensor for VideoLLaMA-2"""
    # Process original video
    vid_tensor = vprocessor["video"](video_path).to(device, dtype=torch.float16)
    
    # Get original caption and embedding
    with torch.no_grad():
        original_caption = mm_infer(
            vid_tensor, "Describe the video in detail.",
            model=vlm, tokenizer=tok, modal="video",
            do_sample=False
        ).strip()
        
        # Get video embeddings from the vision encoder
        vision_outputs = vlm.model.vision_tower(vid_tensor)
        original_features = vision_outputs.last_hidden_state
    
    # Prepare adversarial tensor
    vid_adv = vid_tensor.clone().detach().requires_grad_(True)
    
    # Forward pass to get features
    vision_outputs_adv = vlm.model.vision_tower(vid_adv)
    adv_features = vision_outputs_adv.last_hidden_state
    
    # Loss: maximize difference between original and adversarial features
    loss = F.mse_loss(adv_features, original_features)
    loss = -loss  # Negative to maximize difference
    
    # Backward pass
    loss.backward()
    
    # FGSM perturbation
    with torch.no_grad():
        perturbation = epsilon * vid_adv.grad.sign()
        vid_adv_final = torch.clamp(vid_adv + perturbation, 0, 1)
    
    # Generate adversarial caption
    with torch.no_grad():
        adv_caption = mm_infer(
            vid_adv_final, "Describe the video in detail.",
            model=vlm, tokenizer=tok, modal="video",
            do_sample=False
        ).strip()
    
    return vid_adv_final, original_caption, adv_caption, vid_tensor

# ────────────────────────────────────────────────────────────────
@torch.inference_mode()
def vit_rollout(pil_img, vt, vproc, device="cuda"):
    """Generate attention heatmap using CLIP ViT"""
    inp = vproc(images=pil_img, return_tensors="pt").to(device)
    outs = vt(**inp, output_attentions=True)

    A = torch.stack(outs.attentions)[:, 0].mean(1)
    eye = torch.eye(A.size(-1), device=device)
    R = eye
    for layer in A:
        layer = (layer + eye)
        layer /= layer.sum(-1, keepdim=True)
        R = layer @ R

    n_vis = R.size(-1) - 1
    side = int(round(n_vis ** 0.5))
    heat = R[0, 1:1+n_vis]
    heat = heat[:side*side].reshape(side, side).detach().cpu().numpy()

    heat = cv2.GaussianBlur(heat, (0,0), 3)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    return np.power(heat, .5)

# ────────────────────────────────────────────────────────────────
def tensor_to_frames(vid_tensor):
    """Convert video tensor back to PIL frames"""
    # vid_tensor shape: [1, T, C, H, W]
    frames = []
    vid_np = vid_tensor.squeeze(0).cpu().numpy()  # [T, C, H, W]
    
    for i in range(vid_np.shape[0]):
        frame = vid_np[i].transpose(1, 2, 0)  # [H, W, C]
        frame = (frame * 255).astype(np.uint8)
        frame_pil = Image.fromarray(frame)
        frames.append(frame_pil)
    
    return frames

# ────────────────────────────────────────────────────────────────
def process_video_frames(frames, vt, vproc, device, output_dir, prefix, alpha=0.35):
    """Process frames and generate attention visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    energies = []
    jet = plt.colormaps.get_cmap("jet")
    
    for idx, pil_frame in enumerate(frames):
        # Generate attention heatmap
        heat = vit_rollout(pil_frame, vt, vproc, device)
        energies.append(heat.mean())
        
        # Convert PIL to OpenCV format
        frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
        
        # Resize heatmap to frame size
        heat_resized = cv2.resize(heat, (frame.shape[1], frame.shape[0]), cv2.INTER_LINEAR)
        
        # Create overlay
        h_rgba = (jet(heat_resized) * 255).astype(np.uint8)
        h_bgr = cv2.cvtColor(cv2.cvtColor(h_rgba, cv2.COLOR_RGBA2RGB), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(frame, 1-alpha, h_bgr, alpha, 0)
        
        # Add frame number and attention peak
        cv2.putText(overlay, f"{prefix} frame {idx}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        y, x = np.unravel_index(np.argmax(heat_resized), heat_resized.shape)
        cv2.circle(overlay, (x, y), 12, (0, 0, 255), 3)
        
        # Save frame
        cv2.imwrite(str(output_dir / f"{prefix}_frame_{idx:04d}.png"), overlay)
    
    return energies

# ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="Input video path")
    ap.add_argument("--out", default="fgsm_attention_results", help="Output directory")
    ap.add_argument("--epsilon", type=float, default=0.03, help="FGSM epsilon")
    ap.add_argument("--alpha", type=float, default=0.35, help="Overlay alpha")
    ap.add_argument("--caption-file", default="captions.txt", help="Caption output file")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("❌ Need a CUDA GPU to run VideoLLaMA-2.")
    
    device = "cuda"
    print("⏳ Loading models...")
    vt, vproc, vlm, vprocessor, tok = load_models(device)

    print(f"🎯 Applying FGSM attack (ε={args.epsilon})...")
    vid_adv, original_caption, adv_caption, vid_original = fgsm_attack_video(
        args.video, vlm, vprocessor, tok, args.epsilon, device
    )
    
    print(f"📝 Original caption: {original_caption}")
    print(f"📝 Adversarial caption: {adv_caption}")
    
    # Convert tensors to frames
    print("🖼️ Converting tensors to frames...")
    original_frames = tensor_to_frames(vid_original)
    adversarial_frames = tensor_to_frames(vid_adv)
    
    # Process original frames
    print("🔍 Processing original frames...")
    original_energies = process_video_frames(
        original_frames, vt, vproc, device, args.out, "original", args.alpha
    )
    
    # Process adversarial frames
    print("🔍 Processing adversarial frames...")
    adversarial_energies = process_video_frames(
        adversarial_frames, vt, vproc, device, args.out, "adversarial", args.alpha
    )
    
    # Generate temporal plots
    print("📊 Generating temporal plots...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Original temporal curve
    orig_curve = np.array(original_energies)
    orig_curve = (orig_curve - orig_curve.min()) / (orig_curve.max() - orig_curve.min() + 1e-8)
    ax1.plot(orig_curve, 'o-', color='blue', label='Original')
    ax1.set_title("Original Video - Temporal Attention Energy")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Normalized Energy")
    ax1.grid(True)
    ax1.legend()
    
    # Adversarial temporal curve
    adv_curve = np.array(adversarial_energies)
    adv_curve = (adv_curve - adv_curve.min()) / (adv_curve.max() - adv_curve.min() + 1e-8)
    ax2.plot(adv_curve, 'o-', color='red', label='Adversarial')
    ax2.set_title("Adversarial Video - Temporal Attention Energy")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Normalized Energy")
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(Path(args.out) / "temporal_comparison.png")
    plt.close()
    
    # Comparison plot
    plt.figure(figsize=(12, 6))
    plt.plot(orig_curve, 'o-', color='blue', label='Original', alpha=0.7)
    plt.plot(adv_curve, 'o-', color='red', label='Adversarial', alpha=0.7)
    plt.title("Temporal Attention Energy Comparison")
    plt.xlabel("Frame")
    plt.ylabel("Normalized Energy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(args.out) / "attention_comparison.png")
    plt.close()
    
    # Save captions
    caption_path = Path(args.caption_file)
    with open(caption_path, 'w', encoding='utf-8') as f:
        f.write(f"Original Caption:\n{original_caption}\n\n")
        f.write(f"Adversarial Caption (ε={args.epsilon}):\n{adv_caption}\n")
    
    print(f"✅ Results saved to {args.out}")
    print(f"📝 Captions saved to {args.caption_file}")
    
    # Calculate attention difference metrics
    mse_attention = np.mean((orig_curve - adv_curve) ** 2)
    print(f"📊 Attention MSE difference: {mse_attention:.6f}")

if __name__ == "__main__":
    main()
