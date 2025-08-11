#!/usr/bin/env python3
# FGSM + BERTScore evaluation for VideoLLaMA-2
import os, sys, cv2, argparse, math, gc
from pathlib import Path
from types import MethodType
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch, torch.nn.functional as F
from bert_score import BERTScorer
from transformers import CLIPVisionModel, CLIPImageProcessor
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

# ── Disable FlashAttention-2 ──────────────────────────────────────
os.environ.update({
    "PYTORCH_ATTENTION_IMPLEMENTATION": "eager",
    "HF_DISABLE_FLASH_ATTN_2": "1",
    "DISABLE_FLASH_ATTN_2": "1",
})
MODEL_NAME = "DAMO-NLP-SG/VideoLLaMA2-7B-16F"
VISION_ID  = "openai/clip-vit-large-patch14-336"

# ── Utility helpers ──────────────────────────────────────────────
def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache();  torch.cuda.synchronize()
    gc.collect()

def enable_grad_vision_tower(vlm):
    vt = vlm.model.vision_tower
    def forward_with_grad(self, imgs):
        if isinstance(imgs, list):
            feats = [self.feature_select(
                     self.vision_tower(im.unsqueeze(0), output_hidden_states=True)
                     ).to(im.dtype) for im in imgs]
            return torch.cat(feats, dim=0)
        out = self.vision_tower(imgs, output_hidden_states=True)
        return self.feature_select(out).to(imgs.dtype)
    vt.forward = MethodType(forward_with_grad, vt)
    print("✅ VisionTower monkey-patched (grad-enabled)")

def load_models(device="cuda"):
    vt     = CLIPVisionModel.from_pretrained(VISION_ID).to(device).eval()
    vproc  = CLIPImageProcessor.from_pretrained(VISION_ID)
    disable_torch_init()
    vlm, vprocessor, tok = model_init(
        MODEL_NAME, attn_implementation="eager",
        torch_dtype=torch.float16, device_map=device)
    vlm.eval()
    return vt, vproc, vlm, vprocessor, tok

# ── Fixed tensor_to_frames (GPT suggestion #2) ────────────────────
def tensor_to_frames(video_tensor):
    """Convert video tensor back to frames - FIXED for VideoLLaMA2 [-1,1] range"""
    if video_tensor.dim() == 5:
        video_tensor = video_tensor.squeeze(0)  # Remove batch dimension
    
    frames = []
    for t in video_tensor:
        img = ((t + 1) / 2).clamp(0, 1)  # [-1,1] → [0,1] - CORRECTED
        img = (img * 255).byte().permute(1, 2, 0).cpu().numpy()
        frames.append(img)  # Keep RGB for consistency
    
    return frames

def vit_rollout(model, video_tensor, head_fusion='mean', discard_ratio=0.9):
    """Simplified attention rollout for visualization (placeholder)"""
    with torch.no_grad():
        batch_size, num_frames, _, height, width = video_tensor.shape
        attention_maps = torch.rand(batch_size, num_frames, height//14, width//14)
        return attention_maps

def process_video_frames(frames, attention_maps=None, alpha=0.35):
    """Process frames with optional attention overlay"""
    processed_frames = []
    
    for i, frame in enumerate(frames):
        processed_frame = frame.copy()
        
        if attention_maps is not None and i < len(attention_maps):
            att_map = attention_maps[i]
            if isinstance(att_map, torch.Tensor):
                att_map = att_map.cpu().numpy()
            
            att_map_resized = cv2.resize(att_map, (frame.shape[1], frame.shape[0]))
            att_map_resized = (att_map_resized - att_map_resized.min()) / \
                             (att_map_resized.max() - att_map_resized.min() + 1e-8)
            
            heatmap = plt.cm.jet(att_map_resized)[:, :, :3]
            heatmap = (heatmap * 255).astype(np.uint8)
            
            processed_frame = cv2.addWeighted(frame, 1-alpha, heatmap, alpha, 0)
        
        processed_frames.append(processed_frame)
    
    return processed_frames

# ── FGSM attack with loss debugging ──────────────────────────────
def fgsm_attack_video(video_path, vlm, vprocessor, tok,
                      epsilon=0.03, device="cuda", margin=0.3):
    clear_memory()
    vid_tensor = vprocessor["video"](video_path
                ).to(device, dtype=torch.float32).requires_grad_(True)
    min_val, max_val = -1.0, 1.0

    with torch.inference_mode():
        original_caption = mm_infer(
            vid_tensor.half(), "Describe the video in detail.",
            model=vlm, tokenizer=tok, modal="video", do_sample=False).strip()

    with torch.no_grad():
        noise = 0.01 * torch.randn_like(vid_tensor)
        target_features = vlm.model.vision_tower((vid_tensor+noise).half()
                          ).detach().contiguous()

    vlm.model.vision_tower.eval()
    adv_features = vlm.model.vision_tower(vid_tensor)
    loss = F.relu(
        F.cosine_similarity(
           target_features.contiguous().view(-1, target_features.size(-1)),
           adv_features.contiguous().view(-1,  adv_features.size(-1)),
           dim=-1).mean() - margin)  # Use margin parameter
    
    # Debug loss value (GPT suggestion #3)
    print(f"🔍 FGSM loss: {loss.item():.6f}")
    
    loss.backward()

    with torch.no_grad():
        vid_adv = torch.clamp(vid_tensor + epsilon*vid_tensor.grad.sign(),
                              min_val, max_val)
    vid_tensor.grad.zero_(); clear_memory()

    with torch.inference_mode():
        adv_caption = mm_infer(
            vid_adv.half(), "Describe the video in detail.",
            model=vlm, tokenizer=tok, modal="video", do_sample=False).strip()

    with torch.inference_mode():
        sim = F.cosine_similarity(
              vlm.model.vision_tower(vid_tensor.detach().half()
                  ).view(-1), vlm.model.vision_tower(vid_adv.half()).view(-1),
              dim=0).item()
    clear_memory()
    return vid_adv, original_caption, adv_caption, vid_tensor, sim

# ── Main entry ────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="FGSM attack on VideoLLaMA-2 with BERTScore")
    ap.add_argument("video")
    ap.add_argument("--out", default="fgsm_attention_results")
    ap.add_argument("--epsilon", type=float, default=0.03)
    ap.add_argument("--alpha",   type=float, default=0.35)
    ap.add_argument("--caption-file", default="captions.txt")
    ap.add_argument("--margin",  type=float, default=0.3)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("❌ CUDA GPU required for VideoLLaMA-2")

    vt, vproc, vlm, vprocessor, tok = load_models("cuda")
    enable_grad_vision_tower(vlm)

    print(f"🎯 FGSM ε={args.epsilon}, margin={args.margin}")
    (vid_adv, orig_cap, adv_cap,
     vid_orig, feat_sim) = fgsm_attack_video(
        args.video, vlm, vprocessor, tok, args.epsilon, "cuda", args.margin)

    print(f"📝 Original: {orig_cap}")
    print(f"🔴 Adversarial: {adv_cap}")
    print(f"📊 Feature similarity: {feat_sim:.4f}")

    # ── BERTScore with GPU-safe settings (GPT suggestion #1) ──────
    scorer = BERTScorer(lang="en",
                        rescale_with_baseline=True,
                        model_type="roberta-base",  # smaller/faster
                        device="cpu",               # avoid GPU OOM
                        batch_size=8)               # conservative batch size
    
    P, R, F1 = scorer.score([adv_cap], [orig_cap])
    bert_f1 = F1[0].item()
    print(f"🟣 BERTScore-F1: {bert_f1:.4f}")

    # ── Save captions + metrics (GPT suggestion #5) ───────────────
    cap_path = Path(args.caption_file)
    need_header = not cap_path.exists() or cap_path.stat().st_size == 0
    with cap_path.open("a", encoding="utf-8") as f:
        if need_header:
            f.write("Original\tAdversarial\tFeatureCosSim\tBERTScoreF1\n")
        f.write(f"{orig_cap}\t{adv_cap}\t{feat_sim:.4f}\t{bert_f1:.4f}\n")
    print(f"✅ Captions & scores appended to {cap_path}")

    # ── Process and save frames ───────────────────────────────────
    try:
        # Convert tensors to frames (now with correct normalization)
        orig_frames = tensor_to_frames(vid_orig)
        adv_frames = tensor_to_frames(vid_adv)
        
        # Create output directory
        out_dir = Path(args.out)
        out_dir.mkdir(exist_ok=True)
        
        # Save original frames
        orig_dir = out_dir / "original"
        orig_dir.mkdir(exist_ok=True)
        for i, frame in enumerate(orig_frames):
            cv2.imwrite(str(orig_dir / f"frame_{i:03d}.png"), 
                       cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        # Save adversarial frames
        adv_dir = out_dir / "adversarial"
        adv_dir.mkdir(exist_ok=True)
        for i, frame in enumerate(adv_frames):
            cv2.imwrite(str(adv_dir / f"frame_{i:03d}.png"), 
                       cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        print(f"✅ Frames saved to {out_dir}")
        
    except Exception as e:
        print(f"⚠️  Frame processing failed: {e}")

    print("🏁 Complete!")

if __name__ == "__main__":
    main()
