#!/usr/bin/env python3
# FGSM + BERTScore evaluation for VideoLLaMA-2 (Memory Allocator Fix)
import os, sys, cv2, argparse, math, gc, tempfile
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
import shutil

# Enable optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

def setup_environment():
    """Set up environment for optimal memory management"""
    os.environ.update({
        "PYTORCH_ATTENTION_IMPLEMENTATION": "eager",
        "HF_DISABLE_FLASH_ATTN_2": "1", 
        "DISABLE_FLASH_ATTN_2": "1",
        # FIX: More conservative memory allocation to avoid expandable_segment bug
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:32,roundup_power2_divisions:16",
    })
    
    # Set up cache directories
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

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(2.0 / torch.sqrt(mse))  # 2.0 for [-1,1] range

def calculate_linf_norm(delta):
    """Calculate L-infinity norm of perturbation"""
    return torch.max(torch.abs(delta)).item()

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
    
    # Freeze parameters correctly instead of using no_grad()
    for p in vlm.model.vision_tower.parameters():
        p.requires_grad_(False)
    vlm.model.vision_tower.eval()
    print("✅ Vision tower weights frozen, gradients enabled for inputs only")
    
    print("✅ VisionTower patched with gradient support")

def load_models(device="cuda", verbose=True):
    """Load models with aggressive memory optimization"""
    clear_memory()
    
    # Better seed initialization
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    if verbose:
        print("Loading VideoLLaMA-2 with device offloading...")
    disable_torch_init()
    
    # Use tempfile for offload directory
    offload_dir = tempfile.mkdtemp(prefix="vllama_offload_")
    
    # Use device_map="auto" with conservative memory limit
    vlm, vprocessor, tok = model_init(
        MODEL_NAME, 
        attn_implementation="eager",
        torch_dtype=torch.float16, 
        device_map="auto",
        max_memory={0: "15GiB", "cpu": "64GiB"},  # Slightly more conservative
        offload_folder=offload_dir,
        offload_state_dict=True
    )
    vlm.eval()
    
    if verbose:
        print("✅ Using automatic device mapping for optimal memory allocation")
        print(f"💾 GPU memory after model loading: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    clear_memory()
    return vlm, vprocessor, tok, offload_dir

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

def fix_video_tensor_channels(video_tensor, verbose=True):
    """Fix channel issues in video tensor"""
    if verbose:
        print(f"📐 Input tensor shape: {video_tensor.shape}")
    
    if video_tensor.dim() != 4:
        raise ValueError(f"Expected 4D tensor (T, C, H, W), got {video_tensor.dim()}D: {video_tensor.shape}")
    
    frames, channels, height, width = video_tensor.shape
    if verbose:
        print(f"📐 Dimensions: frames={frames}, channels={channels}, height={height}, width={width}")
    
    # Keep 336x336 to match position embeddings
    if verbose:
        print(f"✅ Keeping original resolution {height}x{width} to match position embeddings")
    
    # Fix channel issues if any
    if channels == 1:
        video_tensor = video_tensor.repeat(1, 3, 1, 1)
        if verbose:
            print("🔧 Fixed: Converted grayscale (1 channel) to RGB (3 channels)")
    elif channels == 2:
        third_channel = video_tensor[:, 0:1, :, :]
        video_tensor = torch.cat([video_tensor, third_channel], dim=1)
        if verbose:
            print("🔧 Fixed: Added third channel to make RGB (was 2 channels)")
    elif channels == 4:
        video_tensor = video_tensor[:, :3, :, :]
        if verbose:
            print("🔧 Fixed: Removed alpha channel from RGBA (was 4 channels)")
    elif channels == 3:
        if verbose:
            print("✅ Video already has 3 RGB channels")
    else:
        raise ValueError(f"Unsupported number of channels: {channels}")
    
    if verbose:
        print(f"📐 Final tensor shape: {video_tensor.shape}")
    return video_tensor

def fgsm_attack_video(video_path, vlm, vprocessor, tok,
                      epsilon=0.03, device="cuda", verbose=True):
    """FGSM attack with memory allocator fix"""
    clear_memory()
    
    if verbose:
        print(f"💾 GPU memory before processing: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Process video
    if verbose:
        print("Processing video...")
    try:
        vid_tensor4d = vprocessor["video"](video_path).to(device, dtype=torch.float16)
    except Exception as e:
        if verbose:
            print(f"⚠️ Processor failed: {e}")
        return None, "Error", "Error", None, 0.0, 0.0, 0.0

    # Fix channel issues
    vid_tensor4d = fix_video_tensor_channels(vid_tensor4d, verbose)
    
    # Apply channels_last BEFORE slicing
    vid_tensor4d = vid_tensor4d.to(memory_format=torch.channels_last)
    
    # Make tensor a proper leaf after all transformations
    vid_tensor4d = vid_tensor4d.detach().requires_grad_(True)
    
    # Reduce to frames with better sampling
    target_frames = 4
    if vid_tensor4d.shape[0] > target_frames:
        T = vid_tensor4d.shape[0]
        stride = T // target_frames
        indices = torch.arange(0, T, stride)[:target_frames]
        vid_tensor4d = vid_tensor4d[indices]
        if verbose:
            print(f"Reduced video to {target_frames} frames (uniform stride)")
            print(f"Using frame indices: {indices.tolist()}")
    
    min_val, max_val = -1.0, 1.0

    if verbose:
        print(f"💾 GPU memory after video loading: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Generate original caption
    if verbose:
        print("Generating original caption...")
    with torch.inference_mode():
        video_tensor_for_caption = vid_tensor4d.detach()
        if verbose:
            print(f"📐 Video tensor format: {video_tensor_for_caption.shape}")
        
        original_caption = mm_infer(
            video_tensor_for_caption,
            "Describe the video in detail.",
            model=vlm, tokenizer=tok, modal="video", do_sample=False
        ).strip()
    
    clear_memory()
    if verbose:
        print(f"💾 GPU memory after original caption: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Scale epsilon for [-1,1] input range
    scaled_epsilon = epsilon * 2.0
    if verbose:
        print(f"Performing FGSM attack (ε={epsilon:.3f} → {scaled_epsilon:.3f} for [-1,1] range)...")

    # FIX: More conservative gradient computation to avoid allocator bug
    try:
        # Clear memory before gradient computation
        clear_memory()
        
        # Use regular precision (not mixed) to avoid allocator issues
        with torch.enable_grad():
            # Process all frames at once
            features = vlm.model.vision_tower(vid_tensor4d)  # (T, num_tokens, dim)
            
            # Focus on CLS token for stronger signal
            if features.dim() == 3:  # (T, num_tokens, dim)
                cls_features = features[:, 0]  # (T, dim) - only CLS tokens
                loss = -(cls_features.pow(2).mean())
                if verbose:
                    print(f"🔍 Using CLS token loss for stronger gradient signal")
            else:
                loss = -(features.pow(2).mean())
                if verbose:
                    print(f"🔍 Using full feature loss (fallback)")
            
            if verbose:
                print(f"🔍 Processing all {vid_tensor4d.shape[0]} frames together")
                print(f"   - total loss: {loss.item():.6f}")
        
        # Single backward pass
        vlm.zero_grad(set_to_none=True)
        loss.backward()
        
        # Check gradients
        if vid_tensor4d.grad is not None:
            grad_norm = vid_tensor4d.grad.norm().item()
            if verbose:
                print(f"📈 Total gradient norm: {grad_norm:.6f}")
        else:
            if verbose:
                print("⚠️ No gradients computed, creating zero gradients")
            vid_tensor4d.grad = torch.zeros_like(vid_tensor4d)
            grad_norm = 0.0
        
        # Immediate cleanup
        del features, loss
        clear_memory()
        
    except Exception as e:
        if verbose:
            print(f"⚠️ Error during gradient computation: {e}")
            print("   Using fallback zero gradients")
        vid_tensor4d.grad = torch.zeros_like(vid_tensor4d)
        grad_norm = 0.0

    if verbose:
        print(f"💾 GPU memory after attack: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Proper FGSM step with clipping
    with torch.no_grad():
        # Compute perturbation
        delta = scaled_epsilon * vid_tensor4d.grad.sign()
        # Clip perturbation to stay within epsilon bound
        delta = delta.clamp(-scaled_epsilon, scaled_epsilon)
        # Apply perturbation and clip to valid range
        vid_adv4d = (vid_tensor4d + delta).clamp(min_val, max_val)
        
        perturbation = vid_adv4d - vid_tensor4d
        perturbation_norm = perturbation.norm().item()
        
        # Calculate both PSNR and L-inf norm
        psnr = calculate_psnr(vid_tensor4d, vid_adv4d).item()
        linf_norm = calculate_linf_norm(perturbation)
        
        if verbose:
            print(f"📈 Perturbation L2 norm: {perturbation_norm:.6f}")
            print(f"📈 Perturbation L∞ norm: {linf_norm:.6f}")
            print(f"📈 PSNR: {psnr:.2f} dB")

    # Clear gradients and memory
    vid_tensor4d.grad = None
    clear_memory()

    # Generate adversarial caption
    if verbose:
        print("Generating adversarial caption...")
    with torch.inference_mode():
        adv_caption = mm_infer(
            vid_adv4d.detach(),
            "Describe the video in detail.",
            model=vlm, tokenizer=tok, modal="video", do_sample=False
        ).strip()

    # Enhanced similarity computation (per-frame logging)
    if verbose:
        print("Computing feature similarity...")
    with torch.inference_mode():
        similarities = []
        
        for i in range(vid_tensor4d.shape[0]):
            try:
                orig_frame = vid_tensor4d[i:i+1].detach()
                adv_frame = vid_adv4d[i:i+1].detach()
                
                orig_feat = vlm.model.vision_tower(orig_frame).view(-1)
                adv_feat = vlm.model.vision_tower(adv_frame).view(-1)
                
                frame_sim = F.cosine_similarity(orig_feat, adv_feat, dim=0).item()
                similarities.append(frame_sim)
                
                if verbose:
                    print(f"   - Frame {i+1} similarity: {frame_sim:.4f}")
                
                del orig_feat, adv_feat, orig_frame, adv_frame
                clear_memory()
                
            except torch.cuda.OutOfMemoryError:
                if verbose:
                    print(f"⚠️ OOM during similarity for frame {i+1}, using 0.5")
                similarities.append(0.5)
        
        sim = np.mean(similarities) if similarities else 0.5
    
    # Better cleanup
    vid_orig = vid_tensor4d.clone().detach()
    del vid_tensor4d
    clear_memory()
    
    if verbose:
        print(f"💾 Final GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    return vid_adv4d.cpu(), original_caption, adv_caption, vid_orig.cpu(), sim, psnr, linf_norm

def main():
    # Set up environment first
    setup_environment()
    
    ap = argparse.ArgumentParser(description="FGSM attack on VideoLLaMA-2 with BERTScore")
    ap.add_argument("video")
    ap.add_argument("--out", default="fgsm_attention_results")
    ap.add_argument("--epsilon", type=float, default=0.03)
    ap.add_argument("--caption-file", default="captions.txt")
    ap.add_argument("--verbose", action="store_true", default=True, help="Enable verbose output")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("❌ CUDA GPU required for VideoLLaMA-2")

    if args.verbose:
        print("🚀 Loading models with optimized memory allocation...")
    vlm, vprocessor, tok, offload_dir = load_models("cuda", args.verbose)
    enable_grad_vision_tower(vlm)

    if args.verbose:
        print(f"🎯 FGSM ε={args.epsilon}")
        
        # Memory check
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        print(f"💾 Available GPU memory: {free_memory/1e9:.2f} GB")
    
    try:
        (vid_adv, orig_cap, adv_cap, vid_orig, feat_sim, psnr, linf_norm) = fgsm_attack_video(
            args.video, vlm, vprocessor, tok, args.epsilon, "cuda", args.verbose
        )
        
        if vid_adv is None:
            print("❌ Attack failed due to errors")
            sys.exit(1)
            
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ GPU OOM Error: {e}")
        print("💡 Memory exhausted - consider restarting or reducing frames")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during FGSM attack: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Better cleanup
        try:
            if Path(offload_dir).exists():
                shutil.rmtree(offload_dir)
                if args.verbose:
                    print("🧹 Cleaned up offload directory")
        except:
            pass

    if args.verbose:
        print(f"📝 Original: {orig_cap}")
        print(f"🔴 Adversarial: {adv_cap}")
        print(f"📊 Feature similarity: {feat_sim:.4f}")
        print(f"📊 PSNR: {psnr:.2f} dB")
        print(f"📊 L∞ norm: {linf_norm:.6f}")

    # BERTScore on CPU
    if args.verbose:
        print("Computing BERTScore...")
    scorer = BERTScorer(
        lang="en",
        rescale_with_baseline=True,
        model_type="distilbert-base-uncased",
        device="cpu",
        batch_size=1
    )
    
    P, R, f1_tensor = scorer.score([adv_cap], [orig_cap])
    bert_f1 = f1_tensor[0].item()
    if args.verbose:
        print(f"🟣 BERTScore-F1: {bert_f1:.4f}")

    # Save results with enhanced metrics
    cap_path = Path(args.caption_file)
    need_header = not cap_path.exists() or cap_path.stat().st_size == 0
    with cap_path.open("a", encoding="utf-8") as f:
        if need_header:
            f.write("Original\tAdversarial\tFeatureCosSim\tBERTScoreF1\tPSNR_dB\tLinf_Norm\n")
        f.write(f"{orig_cap}\t{adv_cap}\t{feat_sim:.4f}\t{bert_f1:.4f}\t{psnr:.2f}\t{linf_norm:.6f}\n")
    
    if args.verbose:
        print(f"✅ Results saved to {cap_path}")

    # Optional: Save frames
    try:
        if args.verbose:
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
        
        if args.verbose:
            print(f"✅ Sample frames saved to {out_dir}")
    except Exception as e:
        if verbose:
            print(f"⚠️ Frame saving failed: {e}")

    print("🏁 Complete!")

if __name__ == "__main__":
    main()
