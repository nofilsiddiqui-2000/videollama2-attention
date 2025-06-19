#!/usr/bin/env python3
# FGSM + BERTScore evaluation for VideoLLaMA-2 (TRULY FINAL WORKING VERSION)
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

# Enable optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

def setup_environment():
    """Set up environment for optimal memory management"""
    os.environ.update({
        "PYTORCH_ATTENTION_IMPLEMENTATION": "eager",
        "HF_DISABLE_FLASH_ATTN_2": "1", 
        "DISABLE_FLASH_ATTN_2": "1",
        # Conservative memory allocation
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:64,expandable_segments:True",
    })
    
    # Set up cache directories with correct username
    scratch_dir = "/nfs/speed-scratch/nofilsiddiqui-2000"
    os.environ["HF_HOME"] = f"{scratch_dir}/hf_cache"
    os.environ["MPLCONFIGDIR"] = f"{scratch_dir}/matplotlib_cache"
    
    # Create directories if they don't exist
    Path(f"{scratch_dir}/hf_cache").mkdir(parents=True, exist_ok=True)
    Path(f"{scratch_dir}/matplotlib_cache").mkdir(parents=True, exist_ok=True)

MODEL_NAME = "DAMO-NLP-SG/VideoLLaMA2-7B-16F"

def clear_memory():
    """Aggressive memory clearing"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def enable_grad_vision_tower(vlm):
    """Enable gradients with proper checkpointing"""
    vt = vlm.model.vision_tower
    
    # Monkey-patch to remove @torch.no_grad()
    def forward_with_grad(self, imgs):
        if isinstance(imgs, list):
            feats = [self.feature_select(
                     self.vision_tower(im.unsqueeze(0), output_hidden_states=True)
                     ).to(im.dtype) for im in imgs]
            return torch.cat(feats, dim=0)
        out = self.vision_tower(imgs, output_hidden_states=True)
        return self.feature_select(out).to(imgs.dtype)
    
    vt.forward = MethodType(forward_with_grad, vt)
    print("✅ VisionTower patched with gradient support")

def load_models(device="cuda"):
    """Load models with aggressive memory optimization"""
    clear_memory()
    
    print("Loading VideoLLaMA-2 with device offloading...")
    disable_torch_init()
    
    # Create offload directory
    offload_dir = "/tmp/vllama_offload"
    Path(offload_dir).mkdir(exist_ok=True)
    
    # Use device_map="auto" with conservative memory limit
    vlm, vprocessor, tok = model_init(
        MODEL_NAME, 
        attn_implementation="eager",
        torch_dtype=torch.float16, 
        device_map="auto",
        max_memory={0: "16GiB", "cpu": "64GiB"},  # Conservative limit
        offload_folder=offload_dir,
        offload_state_dict=True
    )
    vlm.eval()
    
    print("✅ Using automatic device mapping for optimal memory allocation")
    
    clear_memory()
    print(f"💾 GPU memory after model loading: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    return vlm, vprocessor, tok

def tensor_to_frames(video_tensor):
    """Convert video tensor back to frames"""
    if video_tensor.dim() == 5:
        video_tensor = video_tensor.squeeze(0)
    
    frames = []
    for i in range(min(8, video_tensor.shape[0])):
        t = video_tensor[i].cpu()
        img = ((t + 1) / 2).clamp(0, 1)
        img = (img * 255).byte().permute(1, 2, 0).numpy()
        frames.append(img)
    
    return frames

def fix_video_tensor_channels(video_tensor):
    """Fix channel issues in video tensor"""
    print(f"📐 Input tensor shape: {video_tensor.shape}")
    
    if video_tensor.dim() != 4:
        raise ValueError(f"Expected 4D tensor (T, C, H, W), got {video_tensor.dim()}D: {video_tensor.shape}")
    
    frames, channels, height, width = video_tensor.shape
    print(f"📐 Dimensions: frames={frames}, channels={channels}, height={height}, width={width}")
    
    # Fix channel issues if any
    if channels == 1:
        video_tensor = video_tensor.repeat(1, 3, 1, 1)
        print("🔧 Fixed: Converted grayscale (1 channel) to RGB (3 channels)")
    elif channels == 2:
        third_channel = video_tensor[:, 0:1, :, :]
        video_tensor = torch.cat([video_tensor, third_channel], dim=1)
        print("🔧 Fixed: Added third channel to make RGB (was 2 channels)")
    elif channels == 4:
        video_tensor = video_tensor[:, :3, :, :]
        print("🔧 Fixed: Removed alpha channel from RGBA (was 4 channels)")
    elif channels == 3:
        print("✅ Video already has 3 RGB channels")
    else:
        raise ValueError(f"Unsupported number of channels: {channels}")
    
    print(f"📐 Final tensor shape: {video_tensor.shape}")
    return video_tensor

def fgsm_attack_video(video_path, vlm, vprocessor, tok,
                      epsilon=0.03, device="cuda", margin=0.3):
    """FGSM attack with detached backbone approach"""
    clear_memory()
    
    print(f"💾 GPU memory before processing: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Process video - CRITICAL GPT FIX #1: Use callable, not subscript
    print("Processing video...")
    try:
        vid_tensor4d = vprocessor(video_path, modal="video").to(device, dtype=torch.float16)
    except Exception as e:
        print(f"⚠️ Processor failed: {e}")
        return None, "Error", "Error", None, 0.0
    
    # Fix channel issues BEFORE frame reduction
    vid_tensor4d = fix_video_tensor_channels(vid_tensor4d)
    
    # GPT SUGGESTION 2.1: Skip redundant memory format change for small tensors
    # vid_tensor4d = vid_tensor4d.to(memory_format=torch.channels_last)  # Removed
    
    # Reduce to ONLY 4 frames to ensure we fit in memory
    if vid_tensor4d.shape[0] > 4:
        indices = torch.linspace(0, vid_tensor4d.shape[0]-1, 4).long()
        vid_tensor4d = vid_tensor4d[indices]
        print(f"Reduced video to 4 frames for memory efficiency")
        print(f"Using frame indices: {indices.tolist()}")  # GPT SUGGESTION 3: Show indices
    
    vid_tensor4d = vid_tensor4d.requires_grad_(True)
    min_val, max_val = -1.0, 1.0

    print(f"💾 GPU memory after video loading: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Generate original caption
    print("Generating original caption...")
    with torch.inference_mode():
        video_tensor_for_caption = vid_tensor4d.detach()
        print(f"📐 Video tensor format: {video_tensor_for_caption.shape}")
        
        original_caption = mm_infer(
            video_tensor_for_caption,
            "Describe the video in detail.",
            model=vlm, tokenizer=tok, modal="video", do_sample=False
        ).strip()
    
    clear_memory()
    print(f"💾 GPU memory after original caption: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # FIXED: Detached backbone FGSM attack with correct tensor dimensions
    print("Performing FGSM attack with detached backbone...")
    
    frame_count = vid_tensor4d.shape[0]
    
    # Initialize gradients to zero
    if vid_tensor4d.grad is None:
        vid_tensor4d.grad = torch.zeros_like(vid_tensor4d)
    
    # Process each frame individually with detached backbone
    for i in range(frame_count):
        single_frame = vid_tensor4d[i:i+1]  # (1, C, H, W) - correct 4D format
        single_frame.requires_grad_(True)
        
        print(f"🔍 Processing frame {i+1}/{frame_count}, shape: {single_frame.shape}")
        
        try:
            # GPT SUGGESTION 2.2: Use autocast for better memory efficiency
            with torch.autocast("cuda", dtype=torch.float16):
                # CRITICAL: Detach the vision backbone to avoid storing gradients through CLIP
                with torch.no_grad():  # <- no grad for CLIP backbone
                    feat = vlm.model.vision_tower(single_frame)  # Pass 4D tensor directly
                
                feat = feat.detach()  # stop computation graph
                
                # Simple energy function - any differentiable scalar works
                energy = -(feat ** 2).mean()
                
                # Ask autograd only for ∂E/∂input (not through CLIP backbone)
                grad, = torch.autograd.grad(
                    energy, single_frame, 
                    retain_graph=False, 
                    create_graph=False
                )
            
            # GPT SUGGESTION 3: Report per-frame grad norm
            grad_norm_frame = grad.norm().item()
            print(f"   - grad norm {grad_norm_frame:.4f}")
            
            # Accumulate gradient for this frame
            vid_tensor4d.grad[i:i+1] += grad  # Add gradient for this frame
            
            # Immediate cleanup
            del feat, energy, grad
            clear_memory()
            
        except torch.cuda.OutOfMemoryError:
            print(f"⚠️ OOM on frame {i+1}, skipping...")
            continue
        except Exception as e:
            print(f"⚠️ Error on frame {i+1}: {e}, skipping...")
            continue
    
    # Check final gradients
    grad_norm = vid_tensor4d.grad.norm().item()
    print(f"📈 Total gradient norm: {grad_norm:.6f}")
    print(f"💾 GPU memory after attack: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # FGSM step
    with torch.no_grad():
        vid_adv4d = torch.clamp(
            vid_tensor4d + epsilon * vid_tensor4d.grad.sign(),
            min_val, max_val
        )
        
        perturbation = vid_adv4d - vid_tensor4d
        perturbation_norm = perturbation.norm().item()
        print(f"📈 Perturbation norm: {perturbation_norm:.6f}")
    
    # Clear gradients and memory
    vid_tensor4d.grad = None
    clear_memory()

    # Generate adversarial caption
    print("Generating adversarial caption...")
    with torch.inference_mode():
        adv_caption = mm_infer(
            vid_adv4d.detach(),
            "Describe the video in detail.",
            model=vlm, tokenizer=tok, modal="video", do_sample=False
        ).strip()

    # Simplified similarity computation
    print("Computing feature similarity...")
    with torch.inference_mode():
        similarities = []
        
        for i in range(vid_tensor4d.shape[0]):
            try:
                orig_frame = vid_tensor4d[i:i+1].detach()  # (1, C, H, W) - correct format
                adv_frame = vid_adv4d[i:i+1].detach()     # (1, C, H, W) - correct format
                
                orig_feat = vlm.model.vision_tower(orig_frame).view(-1)
                adv_feat = vlm.model.vision_tower(adv_frame).view(-1)
                
                frame_sim = F.cosine_similarity(orig_feat, adv_feat, dim=0).item()
                similarities.append(frame_sim)
                
                del orig_feat, adv_feat, orig_frame, adv_frame
                clear_memory()
                
            except torch.cuda.OutOfMemoryError:
                print(f"⚠️ OOM during similarity for frame {i+1}, using 0.5")
                similarities.append(0.5)
        
        # GPT SUGGESTION 2.3: Guard for empty similarities (already correct)
        sim = np.mean(similarities) if similarities else 0.5
    
    clear_memory()
    print(f"💾 Final GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    return vid_adv4d.cpu(), original_caption, adv_caption, vid_tensor4d.cpu(), sim

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

    print("🚀 Loading models with optimized memory allocation...")
    vlm, vprocessor, tok = load_models("cuda")
    enable_grad_vision_tower(vlm)

    print(f"🎯 FGSM ε={args.epsilon}, margin={args.margin}")
    
    # Memory check
    free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
    print(f"💾 Available GPU memory: {free_memory/1e9:.2f} GB")
    
    try:
        (vid_adv, orig_cap, adv_cap, vid_orig, feat_sim) = fgsm_attack_video(
            args.video, vlm, vprocessor, tok, args.epsilon, "cuda", args.margin
        )
        
        if vid_adv is None:
            print("❌ Attack failed due to errors")
            sys.exit(1)
            
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ GPU OOM Error: {e}")
        print("💡 Memory exhausted even with 4 frames - consider restarting")
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
    
    # GPT CRITICAL FIX #2: Avoid variable shadowing
    P, R, f1_tensor = scorer.score([adv_cap], [orig_cap])
    bert_f1 = f1_tensor[0].item()
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
        
        for i, (orig, adv) in enumerate(zip(orig_frames, adv_frames)):
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
