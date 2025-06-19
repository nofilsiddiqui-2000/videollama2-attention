#!/usr/bin/env python3
# FGSM + BERTScore evaluation for VideoLLaMA-2 (Memory Optimized with Offloading)
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

# Enable TF32 for A100 (GPT suggestion #3)
torch.backends.cuda.matmul.allow_tf32 = True

def setup_environment():
    """Set up environment for optimal memory management"""
    os.environ.update({
        "PYTORCH_ATTENTION_IMPLEMENTATION": "eager",
        "HF_DISABLE_FLASH_ATTN_2": "1", 
        "DISABLE_FLASH_ATTN_2": "1",
        # Tighter memory allocation (GPT suggestion #3)
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:32",
    })
    
    # Set up cache directories with correct username
    scratch_dir = "/nfs/speed-scratch/nofilsiddiqui-2000"
    os.environ["HF_HOME"] = f"{scratch_dir}/hf_cache"
    os.environ["MPLCONFIGDIR"] = f"{scratch_dir}/matplotlib_cache"
    
    # Create directories if they don't exist
    Path(f"{scratch_dir}/hf_cache").mkdir(parents=True, exist_ok=True)
    Path(f"{scratch_dir}/matplotlib_cache").mkdir(parents=True, exist_ok=True)

MODEL_NAME = "DAMO-NLP-SG/VideoLLaMA2-7B-16F"
VISION_ID = "openai/clip-vit-large-patch14-336"

def clear_memory():
    """Aggressive memory clearing"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def enable_grad_vision_tower(vlm):
    """Enable gradients with proper checkpointing (GPT fixes)"""
    vt = vlm.model.vision_tower
    
    # 1️⃣ Monkey-patch to remove @torch.no_grad() - FIXED: correct call path
    def forward_with_grad(self, imgs):
        if isinstance(imgs, list):
            feats = [self.feature_select(
                     self.vision_tower(im.unsqueeze(0), output_hidden_states=True)  # FIXED: keep wrapper logic
                     ).to(im.dtype) for im in imgs]
            return torch.cat(feats, dim=0)
        out = self.vision_tower(imgs, output_hidden_states=True)  # FIXED: use vision_tower, not vision_model
        return self.feature_select(out).to(imgs.dtype)
    
    vt.forward = MethodType(forward_with_grad, vt)
    
    # 2️⃣ Enable gradient checkpointing - FIXED: correct attribute path (GPT suggestion #3)
    try:
        vt.vision_tower.gradient_checkpointing_enable()  # FIXED: direct on vision_tower
        print("✅ Gradient checkpointing enabled - saves ~3GB")
    except Exception as e:
        print(f"⚠️ Could not enable gradient checkpointing: {e}")
    
    # 3️⃣ Enable xFormers memory efficient attention if available
    try:
        if hasattr(vt.vision_tower, 'vision_model') and hasattr(vt.vision_tower.vision_model, 'encoder'):
            for layer in vt.vision_tower.vision_model.encoder.layers:
                if hasattr(layer.self_attn, 'enable_memory_efficient_attention'):
                    layer.self_attn.enable_memory_efficient_attention()
            print("⚡ xFormers memory efficient attention enabled")
    except Exception as e:
        print(f"⚠️ xFormers not available: {e}")
    
    print("✅ VisionTower patched with gradient support")

def load_models(device="cuda"):
    """Load models with aggressive memory optimization (GPT suggestions #1 & #2)"""
    clear_memory()
    
    # GPT Suggestion #2: Load CLIP model on CPU only (or skip it entirely)
    print("Loading CLIP vision model on CPU...")
    vt = CLIPVisionModel.from_pretrained(VISION_ID, 
                                        torch_dtype=torch.float16,
                                        device_map="cpu")  # Keep on CPU to save GPU memory
    vproc = CLIPImageProcessor.from_pretrained(VISION_ID)
    
    print("Loading VideoLLaMA-2 with device offloading...")
    disable_torch_init()
    
    # Create offload directory
    offload_dir = "/tmp/vllama_offload"
    Path(offload_dir).mkdir(exist_ok=True)
    
    # GPT Suggestion #1: Use device_map="auto" with memory limits
    vlm, vprocessor, tok = model_init(
        MODEL_NAME, 
        attn_implementation="eager",
        torch_dtype=torch.float16, 
        device_map="auto",  # GPU first, spill-over to CPU
        max_memory={0: "18GiB", "cpu": "64GiB"},  # Limit GPU to 18GB, rest to CPU
        offload_folder=offload_dir,  # Fast local SSD
        offload_state_dict=True
    )
    vlm.eval()
    
    clear_memory()
    print(f"💾 GPU memory after model loading: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    return vt, vproc, vlm, vprocessor, tok

def tensor_to_frames(video_tensor):
    """Convert video tensor back to frames - memory efficient"""
    if video_tensor.dim() == 5:
        video_tensor = video_tensor.squeeze(0)
    
    frames = []
    for i in range(min(8, video_tensor.shape[0])):  # Back to 8 frames since we have more memory
        t = video_tensor[i].cpu()
        img = ((t + 1) / 2).clamp(0, 1)
        img = (img * 255).byte().permute(1, 2, 0).numpy()
        frames.append(img)
    
    return frames

def fgsm_attack_video(video_path, vlm, vprocessor, tok,
                      epsilon=0.03, device="cuda", margin=0.3):
    """FGSM attack with optimized memory management"""
    clear_memory()
    
    print(f"💾 GPU memory before processing: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Process video with moderate frame reduction
    print("Processing video...")
    vid_tensor = vprocessor["video"](video_path).to(device, dtype=torch.float16)
    
    # Reduce to maximum 8 frames (back from 4 since we have more memory)
    if vid_tensor.shape[1] > 8:
        indices = torch.linspace(0, vid_tensor.shape[1]-1, 8).long()
        vid_tensor = vid_tensor[:, indices]
        print(f"Reduced video to 8 frames for memory efficiency")
    
    vid_tensor = vid_tensor.requires_grad_(True)
    min_val, max_val = -1.0, 1.0

    print(f"💾 GPU memory after video loading: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Generate original caption
    print("Generating original caption...")
    with torch.inference_mode():
        original_caption = mm_infer(
            vid_tensor.detach(), "Describe the video in detail.",
            model=vlm, tokenizer=tok, modal="video", do_sample=False
        ).strip()
    
    clear_memory()
    print(f"💾 GPU memory after original caption: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # FGSM attack with small chunks (GPT suggestion #4)
    print("Performing FGSM attack...")
    
    with torch.enable_grad():
        # Process in very small chunks (GPT suggestion #4: 1-2 frames)
        frame_count = vid_tensor.shape[1]
        chunk_size = 2  # Process 2 frames at a time (reduced from 4)
        
        total_loss = 0
        for i in range(0, frame_count, chunk_size):
            end_idx = min(i + chunk_size, frame_count)
            chunk = vid_tensor[:, i:end_idx]
            
            # Get features for this chunk
            chunk_features = vlm.model.vision_tower(chunk)
            
            # Simple L2 norm loss (memory efficient)
            chunk_loss = -chunk_features.norm() / chunk_features.numel()
            total_loss += chunk_loss
            
            # Clear intermediate tensors
            del chunk_features
            clear_memory()
        
        loss = total_loss
        print(f"🔍 FGSM loss: {loss.item():.6f}")
        print(f"💾 GPU memory during forward: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
        # Backward pass
        vlm.zero_grad()
        loss.backward()

    print(f"💾 GPU memory after backward: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # FGSM step
    with torch.no_grad():
        if vid_tensor.grad is not None:
            vid_adv = torch.clamp(
                vid_tensor + epsilon * vid_tensor.grad.sign(),
                min_val, max_val
            )
        else:
            print("⚠️ No gradients found, using original video")
            vid_adv = vid_tensor.clone()
    
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

    # Compute similarity with chunked processing
    print("Computing feature similarity...")
    with torch.inference_mode():
        # Process similarity in small chunks
        chunk_size = 2  # Small chunks for similarity too
        similarities = []
        
        for i in range(0, vid_tensor.shape[1], chunk_size):
            end_idx = min(i + chunk_size, vid_tensor.shape[1])
            
            orig_chunk = vid_tensor[:, i:end_idx].detach()
            adv_chunk = vid_adv[:, i:end_idx].detach()
            
            orig_feat = vlm.model.vision_tower(orig_chunk).view(-1)
            adv_feat = vlm.model.vision_tower(adv_chunk).view(-1)
            
            chunk_sim = F.cosine_similarity(orig_feat, adv_feat, dim=0).item()
            similarities.append(chunk_sim)
            
            del orig_feat, adv_feat
            clear_memory()
        
        sim = np.mean(similarities)
    
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

    print("🚀 Loading models with memory offloading...")
    vt, vproc, vlm, vprocessor, tok = load_models("cuda")
    enable_grad_vision_tower(vlm)

    print(f"🎯 FGSM ε={args.epsilon}, margin={args.margin}")
    
    # Memory check
    free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
    print(f"💾 Available GPU memory: {free_memory/1e9:.2f} GB")
    
    try:
        (vid_adv, orig_cap, adv_cap, vid_orig, feat_sim) = fgsm_attack_video(
            args.video, vlm, vprocessor, tok, args.epsilon, "cuda", args.margin
        )
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ GPU OOM Error: {e}")
        print("💡 Even with offloading, still OOM. Try:")
        print("   1. Using ViT-base model instead of ViT-large")
        print("   2. Further reducing frames to 4")
        print("   3. Processing 1 frame at a time")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during FGSM attack: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"📝 Original: {orig_cap}")
    print(f"🔴 Adversarial: {adv_cap}")
    print(f"📊 Feature similarity: {feat_sim:.4f}")

    # BERTScore on CPU
    print("Computing BERTScore...")
    scorer = BERTScorer(
        lang="en",
        rescale_with_baseline=True,
        model_type="distilbert-base-uncased",
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
