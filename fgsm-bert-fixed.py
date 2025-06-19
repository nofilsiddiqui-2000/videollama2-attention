#!/usr/bin/env python3
# FGSM + BERTScore evaluation for VideoLLaMA-2 (Fixed Memory Issues)
import os, sys, cv2, argparse, math, gc
from pathlib import Path
from types import MethodType
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import torch, torch.nn.functional as F
from bert_score import BERTScorer
from transformers import CLIPVisionModel, CLIPImageProcessor
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

def setup_environment():
    """Set up environment for optimal memory management"""
    os.environ.update({
        "PYTORCH_ATTENTION_IMPLEMENTATION": "eager",
        "HF_DISABLE_FLASH_ATTN_2": "1", 
        "DISABLE_FLASH_ATTN_2": "1",
        # Enhanced memory allocation settings (GPT suggestion)
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,expandable_segments:True",
    })
    
    # Set up cache directories
    scratch_dir = "/nfs/speed-scratch/nofilsiddiqui-2000"  # Updated username
    os.environ["HF_HOME"] = f"{scratch_dir}/hf_cache"
    os.environ["MPLCONFIGDIR"] = f"{scratch_dir}/matplotlib_cache"
    
    # Create directories if they don't exist
    Path(f"{scratch_dir}/hf_cache").mkdir(parents=True, exist_ok=True)
    Path(f"{scratch_dir}/matplotlib_cache").mkdir(parents=True, exist_ok=True)

MODEL_NAME = "DAMO-NLP-SG/VideoLLaMA2-7B-16F"
VISION_ID = "openai/clip-vit-large-patch14-336"

def clear_memory():
    """Improved memory clearing (GPT suggestion)"""
    gc.collect()  # Call gc.collect() first
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def enable_grad_vision_tower(vlm):
    """Fix both gradient and memory issues (GPT suggestions 1 & 2)"""
    vt = vlm.model.vision_tower
    
    # 1️⃣ Restore the original monkey-patch to remove @torch.no_grad()
    def forward_with_grad(self, imgs):
        if isinstance(imgs, list):
            feats = [self.feature_select(
                     self.vision_tower(im.unsqueeze(0), output_hidden_states=True)
                     ).to(im.dtype) for im in imgs]
            return torch.cat(feats, dim=0)
        out = self.vision_tower(imgs, output_hidden_states=True)
        return self.feature_select(out).to(imgs.dtype)
    
    vt.forward = MethodType(forward_with_grad, vt)
    
    # 2️⃣ Enable gradient checkpointing to save ~40% VRAM
    vt.gradient_checkpointing_enable()
    
    # 3️⃣ Try to enable xFormers memory-efficient attention if available
    try:
        vt.vision_tower.enable_xformers_memory_efficient_attention()
        print("⚡ xFormers flash-attn active")
    except Exception:
        pass
    
    print("✅ VisionTower patched + checkpointing enabled")

def load_models(device="cuda"):
    """Load models with memory optimization"""
    clear_memory()
    
    # Load vision tower with memory optimization
    print("Loading CLIP vision model...")
    vt = CLIPVisionModel.from_pretrained(VISION_ID, torch_dtype=torch.float16).to(device)
    vproc = CLIPImageProcessor.from_pretrained(VISION_ID)
    vt.eval()
    
    print("Loading VideoLLaMA-2...")
    disable_torch_init()
    vlm, vprocessor, tok = model_init(
        MODEL_NAME, 
        attn_implementation="eager",
        torch_dtype=torch.float16, 
        device_map=device,
        low_cpu_mem_usage=True
    )
    vlm.eval()
    
    clear_memory()
    return vt, vproc, vlm, vprocessor, tok

def tensor_to_frames(video_tensor):
    """Convert video tensor back to frames - memory efficient"""
    if video_tensor.dim() == 5:
        video_tensor = video_tensor.squeeze(0)
    
    frames = []
    for i in range(min(8, video_tensor.shape[0])):  # Limit frames
        t = video_tensor[i].cpu()  # Move to CPU immediately
        img = ((t + 1) / 2).clamp(0, 1)  # Correct normalization for [-1,1] range
        img = (img * 255).byte().permute(1, 2, 0).numpy()
        frames.append(img)
    
    return frames

def fgsm_attack_video(video_path, vlm, vprocessor, tok,
                      epsilon=0.03, device="cuda", margin=0.3):
    """FGSM attack with proper memory management"""
    clear_memory()
    
    print(f"💾 GPU memory before processing: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Process video
    print("Processing video...")
    vid_tensor = vprocessor["video"](video_path).to(device, dtype=torch.float16)
    
    # Reduce video length if too long
    if vid_tensor.shape[1] > 16:
        indices = torch.linspace(0, vid_tensor.shape[1]-1, 16).long()
        vid_tensor = vid_tensor[:, indices]
        print(f"Reduced video to 16 frames")
    
    vid_tensor = vid_tensor.requires_grad_(True)
    min_val, max_val = -1.0, 1.0  # Fixed bounds as per GPT suggestion

    print(f"💾 GPU memory after video loading: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Generate original caption
    print("Generating original caption...")
    with torch.inference_mode():
        original_caption = mm_infer(
            vid_tensor.detach(), "Describe the video in detail.",
            model=vlm, tokenizer=tok, modal="video", do_sample=False
        ).strip()
    
    clear_memory()

    # FGSM attack with cosine margin loss (more stable than L2)
    print("Performing FGSM attack...")
    with torch.enable_grad():
        # Add small noise for target
        with torch.no_grad():
            noise = 0.01 * torch.randn_like(vid_tensor)
            target_features = vlm.model.vision_tower((vid_tensor + noise).detach()).detach()

        # Get current features
        current_features = vlm.model.vision_tower(vid_tensor)
        
        # Cosine similarity loss with margin (GPT suggestion)
        cos_sim = F.cosine_similarity(
            target_features.view(-1, target_features.size(-1)),
            current_features.view(-1, current_features.size(-1)),
            dim=-1
        ).mean()
        
        loss = F.relu(cos_sim - margin)  # Loss to minimize cosine similarity
        
        print(f"🔍 FGSM loss: {loss.item():.6f}")
        print(f"💾 GPU memory during forward: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
        # Backward pass
        vlm.zero_grad()
        loss.backward()

    print(f"💾 GPU memory after backward: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # FGSM step
    with torch.no_grad():
        vid_adv = torch.clamp(
            vid_tensor + epsilon * vid_tensor.grad.sign(),
            min_val, max_val
        )
    
    # Clear gradients and memory
    vid_tensor.grad = None
    clear_memory()

    # Generate adversarial caption
    print("Generating adversarial caption...")
    with torch.inference_mode():
        adv_caption = mm_infer(
            vid_adv.detach(), "Describe the video in detail.",
            model=vlm, tokenizer=tok, modal="video", do_sample=False
        ).strip()

    # Compute similarity
    with torch.inference_mode():
        orig_feat = vlm.model.vision_tower(vid_tensor.detach()).view(-1)
        adv_feat = vlm.model.vision_tower(vid_adv.detach()).view(-1)
        sim = F.cosine_similarity(orig_feat, adv_feat, dim=0).item()
    
    clear_memory()
    print(f"💾 Final GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    return vid_adv.cpu(), original_caption, adv_caption, vid_tensor.cpu(), sim

def main():
    # Set up environment first
    setup_environment()
    
    ap = argparse.ArgumentParser(description="FGSM attack on VideoLLaMA-2 with BERTScore")
    ap.add_argument("video")
    ap.add_argument("--out", default="fgsm_attention_results")
    ap.add_argument("--epsilon", type=float, default=0.03)
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--caption-file", default="captions.txt")
    ap.add_argument("--margin", type=float, default=0.3)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("❌ CUDA GPU required for VideoLLaMA-2")

    print("🚀 Loading models...")
    vt, vproc, vlm, vprocessor, tok = load_models("cuda")
    enable_grad_vision_tower(vlm)  # This now includes all the fixes

    print(f"🎯 FGSM ε={args.epsilon}, margin={args.margin}")
    try:
        (vid_adv, orig_cap, adv_cap, vid_orig, feat_sim) = fgsm_attack_video(
            args.video, vlm, vprocessor, tok, args.epsilon, "cuda", args.margin
        )
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ GPU OOM Error: {e}")
        print("💡 Try reducing video length or switching to ViT-base model")
        sys.exit(1)

    print(f"📝 Original: {orig_cap}")
    print(f"🔴 Adversarial: {adv_cap}")
    print(f"📊 Feature similarity: {feat_sim:.4f}")

    # BERTScore with small model on CPU
    print("Computing BERTScore...")
    scorer = BERTScorer(
        lang="en",
        rescale_with_baseline=True,
        model_type="distilbert-base-uncased",  # Smallest model
        device="cpu",
        batch_size=1
    )
    
    P, R, F1 = scorer.score([adv_cap], [orig_cap])
    bert_f1 = F1[0].item()
    print(f"🟣 BERTScore-F1: {bert_f1:.4f}")

    # Save results
    cap_path = Path(args.caption_file)
    need_header = not cap_path.exists() or cap_path.stat().st_size == 0
    with cap_path.open("a", encoding="utf-8") as f:
        if need_header:
            f.write("Original\tAdversarial\tFeatureCosSim\tBERTScoreF1\n")
        f.write(f"{orig_cap}\t{adv_cap}\t{feat_sim:.4f}\t{bert_f1:.4f}\n")
    print(f"✅ Results saved to {cap_path}")

    # Optional: Save frames
    try:
        print("Saving sample frames...")
        orig_frames = tensor_to_frames(vid_orig)
        adv_frames = tensor_to_frames(vid_adv)
        
        out_dir = Path(args.out)
        out_dir.mkdir(exist_ok=True)
        
        # Save first 4 frames only
        for i, (orig, adv) in enumerate(zip(orig_frames[:4], adv_frames[:4])):
            cv2.imwrite(str(out_dir / f"orig_frame_{i}.png"), 
                       cv2.cvtColor(orig, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_dir / f"adv_frame_{i}.png"), 
                       cv2.cvtColor(adv, cv2.COLOR_RGB2BGR))
        
        print(f"✅ Sample frames saved to {out_dir}")
    except Exception as e:
        print(f"⚠️ Frame saving failed: {e}")

    print("🏁 Complete!")

if __name__ == "__main__":
    main()
