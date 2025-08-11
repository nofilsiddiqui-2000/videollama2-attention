# python videollama_fgsm_attack_fixed.py test/abc.mp4 --out results --epsilon 0.03 --alpha 0.35 --caption-file captions.txt --margin 0.3


#!/usr/bin/env python3
# videollama_fgsm_attack_memory_fixed.py • Working FGSM attacks with memory management

import os, sys, cv2, argparse, math, gc
from pathlib import Path
from types import MethodType
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

def clear_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def enable_grad_vision_tower(vlm):
    """
    CRITICAL FIX: Monkey-patch VisionTower.forward to remove @torch.no_grad() decorator
    This is the surgical fix that enables gradient flow through the vision tower.
    """
    vt = vlm.model.vision_tower  # ← the VisionTower instance

    def forward_with_grad(self, imgs):
        """
        Drop-in replacement for VisionTower.forward
        that keeps the autograd graph intact.
        """
        if isinstance(imgs, list):  # VideoLLaMA handles lists & tensors
            feats = []
            for im in imgs:
                out = self.vision_tower(im.unsqueeze(0), output_hidden_states=True)
                feats.append(self.feature_select(out).to(im.dtype))
            return torch.cat(feats, dim=0)
        else:
            out = self.vision_tower(imgs, output_hidden_states=True)
            return self.feature_select(out).to(imgs.dtype)

    # Monkey-patch the forward method
    vt.forward = MethodType(forward_with_grad, vt)
    print("✅ VisionTower monkey-patched to enable gradients")

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

def fgsm_attack_video(video_path, vlm, vprocessor, tok, epsilon=0.03, device="cuda"):
    """Fixed FGSM attack with proper gradient flow and non-saturating loss"""
    clear_memory()
    
    # Process video in fp32 for stable gradients
    vid_tensor = vprocessor["video"](video_path).to(device, dtype=torch.float32)
    vid_tensor.requires_grad_(True)
    
    # VideoLLaMA inputs are normalized to [-1, 1] range
    min_val, max_val = -1.0, 1.0
    
    # Get original caption for comparison
    with torch.inference_mode():
        vid_half = vid_tensor.detach().half()
        original_caption = mm_infer(
            vid_half, "Describe the video in detail.",
            model=vlm, tokenizer=tok, modal="video", do_sample=False
        ).strip()
    
    # KEY FIX: Create a DIFFERENT target to avoid saturation (simplified)
    with torch.no_grad():
        noise = 0.01 * torch.randn_like(vid_tensor)
        noisy_vid = (vid_tensor + noise).half()
        target_features = vlm.model.vision_tower(noisy_vid).detach().contiguous()
    
    print(f"Target features shape: {target_features.shape}")
    clear_memory()
    
    # Keep vision tower in eval mode but allow gradients to flow
    vlm.model.vision_tower.eval()
    
    # Forward pass - gradients should now flow thanks to monkey patch
    adv_features = vlm.model.vision_tower(vid_tensor)
    
    print(f"Adversarial features shape: {adv_features.shape}")
    print(f"Adversarial features requires_grad: {adv_features.requires_grad}")
    print(f"Vid tensor requires_grad: {vid_tensor.requires_grad}")
    
    # FIXED LOSS: Use .contiguous() before reshaping
    target_flat = target_features.contiguous().view(-1, target_features.size(-1))
    adv_flat = adv_features.contiguous().view(-1, adv_features.size(-1))
    
    # Ensure dtype consistency
    target_flat = target_flat.to(adv_flat.dtype)
    
    cos_sim = F.cosine_similarity(target_flat, adv_flat, dim=-1).mean()
    margin = 0.3  # Push features at least 0.3 away from target
    loss = F.relu(cos_sim - margin)  # Only penalize when too similar
    
    print(f"Cosine similarity: {cos_sim.item():.6f}")
    print(f"Loss (margin-based): {loss.item():.6f}")
    print(f"Loss requires_grad: {loss.requires_grad}")
    
    # Backward pass
    loss.backward()
    
    # Check if gradients are computed
    if vid_tensor.grad is None:
        print("⚠️ No gradients computed!")
        return vid_tensor, original_caption, original_caption, vid_tensor, 1.0
    
    grad_norm = vid_tensor.grad.norm().item()
    print(f"Gradient norm: {grad_norm:.6f}")
    
    # Apply FGSM perturbation
    with torch.no_grad():
        perturbation = epsilon * vid_tensor.grad.sign()
        vid_adv = torch.clamp(vid_tensor + perturbation, min_val, max_val)
    
    # Clean up gradients
    vid_tensor.grad.zero_()
    del target_features, adv_features  # Explicit cleanup
    clear_memory()
    
    # Generate adversarial caption
    try:
        with torch.inference_mode():
            adv_caption = mm_infer(
                vid_adv.half(), "Describe the video in detail.",
                model=vlm, tokenizer=tok, modal="video", do_sample=False
            ).strip()
    except RuntimeError as e:
        if "CUDA" in str(e) or "memory" in str(e).lower():
            print(f"⚠️ CUDA memory error: {e}")
            clear_memory()
            try:
                with torch.inference_mode():
                    adv_caption = mm_infer(
                        vid_adv.half(), "Describe this video briefly.",
                        model=vlm, tokenizer=tok, modal="video",
                        do_sample=False, max_new_tokens=50
                    ).strip()
            except Exception as e2:
                print(f"⚠️ Still failed: {e2}")
                adv_caption = "[Error: Could not generate adversarial caption]"
        else:
            raise e
    
    # Calculate final similarity against original (not noisy target)
    with torch.inference_mode():
        try:
            original_features = vlm.model.vision_tower(vid_tensor.detach().half())
            final_features = vlm.model.vision_tower(vid_adv.half())
            
            orig_flat = original_features.contiguous().view(-1, original_features.size(-1))
            final_flat = final_features.contiguous().view(-1, final_features.size(-1))
            
            final_similarity = F.cosine_similarity(orig_flat, final_flat, dim=-1).mean().item()
        except Exception as e:
            print(f"⚠️ Error calculating final similarity: {e}")
            final_similarity = cos_sim.item()
    
    print(f"🎯 Attack success: cosine similarity vs original {final_similarity:.4f} (lower = more different)")
    print(f"📊 Expected gradient norm range: 50-150, got: {grad_norm:.1f}")
    
    clear_memory()
    return vid_adv, original_caption, adv_caption, vid_tensor, final_similarity

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
    # FIX: Detach tensor before converting to numpy
    vid_np = vid_tensor.detach().squeeze(0).cpu().numpy()  # [T, C, H, W]
    
    # VideoLLaMA tensors are in [-1, 1], convert to [0, 1] for display
    vid_np = (vid_np + 1.0) / 2.0  # Convert from [-1, 1] to [0, 1]
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
        
        # Clear memory periodically with explicit cleanup
        if idx % 10 == 0:
            del heat, heat_resized, overlay
            clear_memory()
    
    print(f"   {prefix}: {len(frames)}/{len(frames)} ✓")
    return energies

def main():
    ap = argparse.ArgumentParser(description="FGSM attack on VideoLLaMA-2 with attention visualization")
    ap.add_argument("video", help="Input video path")
    ap.add_argument("--out", default="fgsm_attention_results", help="Output directory")
    ap.add_argument("--epsilon", type=float, default=0.03, help="FGSM epsilon (default: 0.03)")
    ap.add_argument("--alpha", type=float, default=0.35, help="Attention overlay alpha (default: 0.35)")
    ap.add_argument("--caption-file", default="captions.txt", help="Caption output file")
    ap.add_argument("--margin", type=float, default=0.3, help="Contrastive margin (default: 0.3)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("❌ CUDA GPU required for VideoLLaMA-2")
    
    device = "cuda"
    print("⏳ Loading models...")
    vt, vproc, vlm, vprocessor, tok = load_models(device)
    
    # CRITICAL: Apply the monkey patch to enable gradients
    enable_grad_vision_tower(vlm)

    print(f"🎯 Applying FGSM attack (ε={args.epsilon}, margin={args.margin})...")
    vid_adv, original_caption, adv_caption, vid_original, similarity = fgsm_attack_video(
        args.video, vlm, vprocessor, tok, args.epsilon, device
    )
    
    print(f"\n📝 Original caption: {original_caption}")
    print(f"📝 Adversarial caption: {adv_caption}")
    
    # Convert tensors to frames
    print("🖼️ Converting tensors to frames...")
    original_frames = tensor_to_frames(vid_original)
    adversarial_frames = tensor_to_frames(vid_adv)
    
    # Process frames and generate attention visualizations
    original_energies = process_video_frames(
        original_frames, vt, vproc, device, args.out, "original", args.alpha
    )
    
    clear_memory()  # Clear between processing
    
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
        f.write(f"FGSM Attack Results (ε={args.epsilon}, margin={args.margin})\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Original Caption:\n{original_caption}\n\n")
        f.write(f"Adversarial Caption:\n{adv_caption}\n\n")
        f.write(f"Feature Similarity: {similarity:.6f}\n")
        if len(orig_curve) == len(adv_curve):
            f.write(f"Attention MSE: {np.mean((orig_curve - adv_curve) ** 2):.6f}\n")
    
    print(f"✅ Results saved to {args.out}")
    print(f"📝 Captions saved to {args.caption_file}")
    print(f"📊 Feature similarity: {similarity:.6f}")
    if len(orig_curve) == len(adv_curve):
        print(f"📊 Attention MSE: {np.mean((orig_curve - adv_curve) ** 2):.6f}")

if __name__ == "__main__":
    main()
