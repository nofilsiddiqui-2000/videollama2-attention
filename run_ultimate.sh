#!/usr/bin/env python3
# ULTIMATE FGSM Attack Suite for VideoLLaMA-2 (GPT-Optimized Maximum Strength)
import os, sys, cv2, argparse, math, gc, tempfile
from pathlib import Path
from types import MethodType
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn.functional as F
from bert_score import BERTScorer
from transformers import CLIPVisionModel, CLIPImageProcessor
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
import shutil
import json
from typing import List, Dict, Tuple

# Enable all optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

def setup_environment():
    """Set up environment for optimal memory management"""
    os.environ.update({
        "PYTORCH_ATTENTION_IMPLEMENTATION": "eager",
        "HF_DISABLE_FLASH_ATTN_2": "1", 
        "DISABLE_FLASH_ATTN_2": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:32,roundup_power2_divisions:16",
    })
    
    scratch_dir = "/nfs/speed-scratch/nofilsiddiqui-2000"
    os.environ["HF_HOME"] = f"{scratch_dir}/hf_cache"
    os.environ["MPLCONFIGDIR"] = f"{scratch_dir}/matplotlib_cache"
    
    Path(f"{scratch_dir}/hf_cache").mkdir(parents=True, exist_ok=True)
    Path(f"{scratch_dir}/matplotlib_cache").mkdir(parents=True, exist_ok=True)

MODEL_NAME = "DAMO-NLP-SG/VideoLLaMA2-7B-16F"

# Ultra-strong hallucinations for maximum impact
ULTIMATE_HALLUCINATIONS = [
    "A massive alien mothership hovers overhead shooting death rays while giant robots battle",
    "Prehistoric dinosaurs are rampaging through a modern city with explosions everywhere",
    "The ground splits open revealing molten lava as volcanic eruptions destroy everything",
    "Military fighter jets engage alien spacecraft in an epic aerial battle",
    "A nuclear explosion creates a mushroom cloud while zombies roam the wasteland",
    "Dragons breathe fire while destroying buildings as people flee in terror",
    "Giant mechanical monsters with laser weapons march through burning ruins",
    "The sky turns blood red as meteors crash down causing massive destruction",
    "Underwater creatures emerge from the ocean to attack coastal cities",
    "Time portals open revealing futuristic war machines battling primitive warriors"
]

def clear_memory():
    """Optimized memory clearing"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def calculate_ultimate_metrics(img1, img2):
    """Fixed calculation of comprehensive attack metrics with temporal awareness"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse > 0:
        psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))
        psnr = psnr.item() if hasattr(psnr, 'item') else float(psnr)
    else:
        psnr = float('inf')
    
    linf_norm = torch.max(torch.abs(img1 - img2)).item()
    l2_norm = torch.norm(img1 - img2).item()
    
    # Fixed per-frame metrics (temporal PSNR)
    frame_psnrs = []
    for t in range(img1.shape[0]):
        frame_mse = torch.mean((img1[t] - img2[t]) ** 2)
        if frame_mse > 0:
            frame_psnr = 20 * torch.log10(2.0 / torch.sqrt(frame_mse))
            frame_psnr = frame_psnr.item() if hasattr(frame_psnr, 'item') else float(frame_psnr)
        else:
            frame_psnr = float('inf')
        frame_psnrs.append(frame_psnr)
    
    min_frame_psnr = min(frame_psnrs)
    
    # Fixed SSIM calculation
    def ssim_enhanced(x, y):
        mu_x = x.mean()
        mu_y = y.mean()
        sigma_x = x.var()
        sigma_y = y.var()
        sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
        
        c1, c2 = (0.01 * 2)**2, (0.03 * 2)**2
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2))
        return ssim.item() if hasattr(ssim, 'item') else float(ssim)
    
    ssim = ssim_enhanced(img1.view(-1), img2.view(-1))
    
    return psnr, linf_norm, l2_norm, ssim, min_frame_psnr

def semantic_similarity_ultimate(text1: str, text2: str) -> float:
    """Ultimate semantic similarity with better fallback"""
    try:
        # Try SentenceTransformers first (more reliable than CLIP for text-only)
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode([text1, text2])
        similarity = F.cosine_similarity(
            torch.tensor(embeddings[0:1]), 
            torch.tensor(embeddings[1:2]), 
            dim=1
        ).item()
        return similarity
    except ImportError:
        # Enhanced token-based fallback
        import re
        def extract_entities(text):
            words = set(re.findall(r'\b\w+\b', text.lower()))
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
            return words - stopwords
        
        entities1 = extract_entities(text1)
        entities2 = extract_entities(text2)
        
        if not entities1 or not entities2:
            return 0.0
        
        jaccard = len(entities1 & entities2) / len(entities1 | entities2)
        return jaccard

def ultimate_di_transform(x, p=0.8):
    """Ultimate Diverse Input transform with multiple augmentations"""
    if torch.rand(1).item() < p:
        # Random resize (more aggressive range)
        rnd_size = torch.randint(280, 350, (1,)).item()
        x_resized = F.interpolate(x, size=(rnd_size, rnd_size), 
                                mode='bilinear', align_corners=False)
        
        # Pad back to 336x336
        pad = 336 - rnd_size
        pad_left = pad // 2
        pad_right = pad - pad_left
        x_padded = F.pad(x_resized, (pad_left, pad_right, pad_left, pad_right))
        
        # Add Gaussian noise
        if torch.rand(1).item() < 0.5:
            x_padded = x_padded + torch.randn_like(x_padded) * 0.03
        
        # Random channel shuffle
        if torch.rand(1).item() < 0.3:
            perm = torch.randperm(3)
            x_padded = x_padded[:, perm]
        
        # Spatial jitter
        if torch.rand(1).item() < 0.5:
            shift_h = torch.randint(-2, 3, (1,)).item()
            shift_w = torch.randint(-2, 3, (1,)).item()
            x_padded = torch.roll(x_padded, shifts=(shift_h, shift_w), dims=(-2, -1))
        
        return x_padded
    return x

def ultimate_ti_transform_grad(grad, kernel_size=9, sigma=2.0):
    """Ultimate Translation Invariant gradient transformation"""
    kernel = torch.zeros(kernel_size, kernel_size, device=grad.device, dtype=grad.dtype)
    center = kernel_size // 2
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] = torch.exp(-((i - center)**2 + (j - center)**2) / (2 * sigma**2))
    
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    
    # Apply convolution with padding
    grad_smooth = torch.zeros_like(grad)
    for t in range(grad.shape[0]):
        for c in range(grad.shape[1]):
            grad_smooth[t, c] = F.conv2d(grad[t:t+1, c:c+1], kernel, padding=center)[0, 0]
    
    return grad_smooth

def ultimate_low_frequency_perturbation(grad, keep_ratio=0.05):
    """Ultra low-frequency perturbation for imperceptibility"""
    grad_filtered = torch.zeros_like(grad)
    
    for t in range(grad.shape[0]):
        for c in range(grad.shape[1]):
            # 2D FFT
            fft_grad = torch.fft.fft2(grad[t, c])
            
            # Create very restrictive low-pass mask
            h, w = fft_grad.shape
            mask = torch.zeros_like(fft_grad, dtype=torch.bool)
            center_h, center_w = h // 2, w // 2
            radius = int(min(h, w) * keep_ratio / 2)
            
            y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            dist = torch.sqrt((y - center_h)**2 + (x - center_w)**2)
            mask = dist <= radius
            
            # Apply mask and inverse FFT
            fft_filtered = fft_grad * mask.to(fft_grad.device)
            grad_filtered[t, c] = torch.fft.ifft2(fft_filtered).real
    
    return grad_filtered

def enable_grad_vision_tower(vlm):
    """Enable gradients with ultimate patching"""
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
    
    for p in vlm.model.vision_tower.parameters():
        p.requires_grad_(False)
    vlm.model.vision_tower.eval()
    print("✅ Vision tower with ultimate gradient patching")

def load_models(device="cuda", verbose=True):
    """Load models with ultimate optimization"""
    clear_memory()
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    if verbose:
        print("🚀 Loading VideoLLaMA-2 with ULTIMATE configuration...")
    disable_torch_init()
    
    offload_dir = tempfile.mkdtemp(prefix="vllama_ultimate_")
    
    vlm, vprocessor, tok = model_init(
        MODEL_NAME, 
        attn_implementation="eager",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        max_memory={0: "15GiB", "cpu": "64GiB"},
        offload_folder=offload_dir,
        offload_state_dict=True
    )
    vlm.eval()
    
    if verbose:
        dtype_str = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        print(f"✅ Models loaded with {dtype_str} precision")
        print(f"💾 GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    clear_memory()
    return vlm, vprocessor, tok, offload_dir

class UltimateAttack:
    """Ultimate attack class with all GPT enhancements"""
    
    def __init__(self, vlm, tokenizer, device="cuda", verbose=True):
        self.vlm = vlm
        self.tokenizer = tokenizer
        self.device = device
        self.verbose = verbose
        self.target_cache = {}
    
    def compute_ultimate_loss(self, features, video_tensor=None, target_text=None, weights=[0.8, 0.2]):
        """Fixed ultimate multi-objective loss with caching"""
        # Primary untargeted loss (CLS tokens)
        if features.dim() == 3:
            cls_features = features[:, 0]
            untargeted_loss = -(cls_features.pow(2).mean())
        else:
            untargeted_loss = -(features.pow(2).mean())
        
        total_loss = untargeted_loss
        
        # Fixed targeted component
        if target_text and video_tensor is not None:
            cache_key = f"{hash(target_text)}_{video_tensor.shape[0]}"
            if cache_key not in self.target_cache:
                try:
                    with torch.no_grad():
                        current_caption = mm_infer(
                            video_tensor.detach(),
                            "Describe the video in detail.",
                            model=self.vlm, tokenizer=self.tokenizer, 
                            modal="video", do_sample=False
                        ).strip()
                    
                    similarity = semantic_similarity_ultimate(current_caption, target_text)
                    self.target_cache[cache_key] = similarity
                    
                    if self.verbose and len(self.target_cache) == 1:
                        print(f"🎯 Cached target similarity: {similarity:.4f}")
                
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️ Target caching failed: {e}")
                    self.target_cache[cache_key] = 0.5
            
            targeted_loss = torch.tensor(self.target_cache[cache_key], device=self.device, dtype=untargeted_loss.dtype)
            total_loss = weights[0] * untargeted_loss + weights[1] * targeted_loss
        
        return total_loss
    
    def ultimate_ni_mi_fgsm(self, video_tensor, epsilon=0.07, num_steps=15, alpha=None, 
                           target_text=None, momentum_decay=0.9):
        """Fixed Nesterov + Momentum FGSM with proper error handling"""
        if alpha is None:
            alpha = epsilon / num_steps
        
        scaled_epsilon = epsilon * 2.0
        scaled_alpha = alpha * 2.0
        
        if self.verbose:
            print(f"⚡ ULTIMATE NI-MI-FGSM: {num_steps} steps, α={alpha:.4f}")
            if target_text:
                print(f"🎯 Target: {target_text[:60]}...")
        
        # Initialize
        original_tensor = video_tensor.clone().detach()
        adv_tensor = video_tensor.clone().detach()
        
        # Initialize momentum
        momentum = torch.zeros_like(adv_tensor)
        
        # Track best result
        best_loss = float('inf')
        best_adv = adv_tensor.clone()
        
        for step in range(num_steps):
            if self.verbose:
                print(f"   ⚡ Step {step+1}/{num_steps}")
            
            try:
                # Memory management
                if step % 3 == 0:
                    clear_memory()
                
                # Create fresh tensor with gradients
                adv_tensor = adv_tensor.detach().requires_grad_(True)
                
                # Apply input transforms
                transformed_tensor = ultimate_di_transform(adv_tensor, p=0.8)
                
                # Forward pass
                features = self.vlm.model.vision_tower(transformed_tensor)
                loss = self.compute_ultimate_loss(features, adv_tensor, target_text)
                
                if self.verbose:
                    print(f"     Loss: {loss.item():.6f}")
                
                # Track best
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_adv = adv_tensor.clone().detach()
                    if self.verbose:
                        print(f"     ✨ New best: {best_loss:.6f}")
                
                # Backward pass
                self.vlm.zero_grad(set_to_none=True)
                loss.backward(create_graph=False)
                
                if adv_tensor.grad is not None:
                    # Get and process gradients
                    grad = adv_tensor.grad.detach()
                    
                    # Apply gradient transforms
                    grad = ultimate_ti_transform_grad(grad, kernel_size=9, sigma=2.0)
                    grad = ultimate_low_frequency_perturbation(grad, keep_ratio=0.05)
                    
                    # Normalize gradient
                    grad_norm = grad.abs().mean() + 1e-8
                    normalized_grad = grad / grad_norm
                    
                    # Nesterov momentum update
                    nesterov_grad = normalized_grad + momentum_decay * momentum
                    momentum = momentum_decay * momentum + normalized_grad
                    
                    if self.verbose:
                        grad_magnitude = momentum.norm().item()
                        print(f"     🔥 Momentum: {grad_magnitude:.6f}")
                    
                    # Take Nesterov step
                    with torch.no_grad():
                        adv_tensor = adv_tensor + scaled_alpha * nesterov_grad.sign()
                        
                        # Project back to epsilon ball
                        delta = adv_tensor - original_tensor
                        delta = torch.clamp(delta, -scaled_epsilon, scaled_epsilon)
                        adv_tensor = torch.clamp(original_tensor + delta, -1.0, 1.0)
                else:
                    if self.verbose:
                        print(f"     ⚠️ No gradients computed")
                
                # Cleanup
                if hasattr(adv_tensor, 'grad') and adv_tensor.grad is not None:
                    adv_tensor.grad = None
                del features, loss
                
            except Exception as e:
                if self.verbose:
                    print(f"     ⚠️ Step {step+1} error: {e}")
                adv_tensor = best_adv.clone()
                break
        
        return best_adv.detach()

def ultimate_attack_pipeline(video_path, vlm, vprocessor, tok, 
                           epsilon=0.07, num_steps=15, target_text=None, 
                           device="cuda", verbose=True):
    """Ultimate attack pipeline with all enhancements"""
    clear_memory()
    
    if verbose:
        print(f"⚡ ULTIMATE ATTACK PIPELINE ⚡")
        print(f"💾 Initial memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Load and prepare video
    vid_tensor4d = vprocessor["video"](video_path).to(device, 
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    
    # Fix channels
    if vid_tensor4d.shape[1] != 3:
        if vid_tensor4d.shape[1] == 1:
            vid_tensor4d = vid_tensor4d.repeat(1, 3, 1, 1)
        elif vid_tensor4d.shape[1] == 4:
            vid_tensor4d = vid_tensor4d[:, :3, :, :]
    
    vid_tensor4d = vid_tensor4d.to(memory_format=torch.channels_last)
    
    # Smart frame sampling
    target_frames = 4
    if vid_tensor4d.shape[0] > target_frames:
        T = vid_tensor4d.shape[0]
        indices = torch.linspace(0, T-1, target_frames).long()
        vid_tensor4d = vid_tensor4d[indices]
    
    if verbose:
        print(f"📐 Video tensor: {vid_tensor4d.shape}")
    
    # Generate original caption
    with torch.inference_mode():
        original_caption = mm_infer(
            vid_tensor4d.detach(),
            "Describe the video in detail.",
            model=vlm, tokenizer=tok, modal="video", do_sample=False
        ).strip()
    
    if verbose:
        print(f"📝 Original: {original_caption}")
    
    # Initialize ultimate attacker
    attacker = UltimateAttack(vlm, tok, device, verbose)
    
    # Auto-select ultimate target
    if target_text is None:
        target_text = np.random.choice(ULTIMATE_HALLUCINATIONS)
        if verbose:
            print(f"🎯 Ultimate target: {target_text}")
    
    # Run ultimate attack
    if verbose:
        print("🚀 Launching ULTIMATE NI-MI-FGSM attack...")
    
    adv_tensor = attacker.ultimate_ni_mi_fgsm(
        vid_tensor4d, 
        epsilon=epsilon, 
        num_steps=num_steps, 
        target_text=target_text
    )
    
    # Generate adversarial caption
    with torch.inference_mode():
        adv_caption = mm_infer(
            adv_tensor.detach(),
            "Describe the video in detail.",
            model=vlm, tokenizer=tok, modal="video", do_sample=False
        ).strip()
    
    if verbose:
        print(f"🔴 Adversarial: {adv_caption}")
    
    # Ultimate evaluation
    psnr, linf_norm, l2_norm, ssim, min_frame_psnr = calculate_ultimate_metrics(vid_tensor4d, adv_tensor)
    sbert_sim = semantic_similarity_ultimate(original_caption, adv_caption)
    
    # Feature similarity
    with torch.inference_mode():
        orig_feat = vlm.model.vision_tower(vid_tensor4d.detach()).mean(dim=0).view(-1)
        adv_feat = vlm.model.vision_tower(adv_tensor.detach()).mean(dim=0).view(-1)
        feat_sim = F.cosine_similarity(orig_feat, adv_feat, dim=0).item()
    
    # Ultimate success metrics
    semantic_drift = 1 - sbert_sim
    attack_strength = semantic_drift * 100
    perceptual_quality = min(psnr, min_frame_psnr)
    
    # Ultimate success criteria
    ultimate_success = (sbert_sim < 0.45 and feat_sim < 0.65 and 
                       semantic_drift > 0.55 and attack_strength > 50)
    
    results = {
        'original_caption': original_caption,
        'adversarial_caption': adv_caption,
        'attack_type': 'ultimate_ni_mi_fgsm',
        'epsilon': epsilon,
        'num_steps': num_steps,
        'target_text': target_text,
        'psnr': psnr,
        'min_frame_psnr': min_frame_psnr,
        'linf_norm': linf_norm,
        'l2_norm': l2_norm,
        'ssim': ssim,
        'feature_similarity': feat_sim,
        'sbert_similarity': sbert_sim,
        'semantic_drift': semantic_drift,
        'attack_strength': attack_strength,
        'perceptual_quality': perceptual_quality,
        'ultimate_success': ultimate_success
    }
    
    if verbose:
        print(f"⚡ ULTIMATE RESULTS:")
        print(f"   🏆 Success: {'✅ ULTIMATE' if ultimate_success else '❌ WEAK'}")
        print(f"   💥 Attack Strength: {attack_strength:.1f}/100")
        print(f"   🌊 Semantic Drift: {semantic_drift:.4f}")
        print(f"   🖼️  Quality: {perceptual_quality:.2f} dB")
        print(f"   📉 SBERT: {sbert_sim:.4f} | Feature: {feat_sim:.4f}")
        print(f"   📏 L∞: {linf_norm:.6f} | SSIM: {ssim:.4f}")
    
    clear_memory()
    return adv_tensor.cpu(), vid_tensor4d.cpu(), results

def main():
    setup_environment()
    
    parser = argparse.ArgumentParser(description="⚡ ULTIMATE FGSM Attack Suite")
    parser.add_argument("video", help="Input video file")
    parser.add_argument("--epsilon", type=float, default=0.07, help="Perturbation budget (higher = stronger)")
    parser.add_argument("--steps", type=int, default=15, help="Number of attack steps")
    parser.add_argument("--target", type=str, help="Target hallucination text")
    parser.add_argument("--caption-file", default="ultimate_results.txt", help="Results file")
    parser.add_argument("--verbose", action="store_true", default=True)
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        sys.exit("❌ CUDA required")
    
    print(f"⚡ ULTIMATE FGSM ATTACK SUITE ⚡")
    print(f"🎬 Video: {args.video}")
    print(f"⚡ Epsilon: {args.epsilon} (Ultimate strength)")
    print(f"🔄 Steps: {args.steps}")
    
    # Load models
    vlm, vprocessor, tok, offload_dir = load_models("cuda", args.verbose)
    enable_grad_vision_tower(vlm)
    
    try:
        # Run ultimate attack
        adv_tensor, orig_tensor, results = ultimate_attack_pipeline(
            args.video, vlm, vprocessor, tok,
            epsilon=args.epsilon,
            num_steps=args.steps,
            target_text=args.target,
            verbose=args.verbose
        )
        
        # Compute BERTScore
        if args.verbose:
            print("🟣 Computing BERTScore...")
        scorer = BERTScorer(lang="en", rescale_with_baseline=True, 
                           model_type="distilbert-base-uncased", device="cpu")
        P, R, F1 = scorer.score([results['adversarial_caption']], [results['original_caption']])
        results['bertscore_f1'] = F1[0].item()
        
        # Save ultimate results
        with open(args.caption_file, 'a') as f:
            if Path(args.caption_file).stat().st_size == 0:
                f.write("Attack\tOriginal\tAdversarial\tTarget\tEpsilon\tSteps\tPSNR\tMinPSNR\tLinf\tL2\tSSIM\tFeat\tSBERT\tDrift\tStrength\tQuality\tBERT_F1\tUltimate_Success\n")
            f.write(f"ultimate\t{results['original_caption']}\t{results['adversarial_caption']}\t{results.get('target_text', 'N/A')}\t{results['epsilon']}\t{results['num_steps']}\t{results['psnr']:.2f}\t{results['min_frame_psnr']:.2f}\t{results['linf_norm']:.6f}\t{results['l2_norm']:.6f}\t{results['ssim']:.4f}\t{results['feature_similarity']:.4f}\t{results['sbert_similarity']:.4f}\t{results['semantic_drift']:.4f}\t{results['attack_strength']:.1f}\t{results['perceptual_quality']:.2f}\t{results['bertscore_f1']:.4f}\t{results['ultimate_success']}\n")
        
        # Save JSON
        json_file = Path(args.caption_file).with_suffix('.json')
        if json_file.exists():
            with open(json_file, 'r') as f:
                all_results = json.load(f)
        else:
            all_results = []
        all_results.append(results)
        with open(json_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"✅ Results saved to {args.caption_file}")
        print(f"⚡ ULTIMATE ATTACK COMPLETE!")
        print(f"🏆 Final Strength: {results['attack_strength']:.1f}/100")
        
        # Success announcement
        if results['ultimate_success']:
            print("🎉 ULTIMATE SUCCESS ACHIEVED! 🎉")
        else:
            print("💪 Attack completed - try higher epsilon for ultimate success")
        
    except Exception as e:
        print(f"❌ Ultimate attack failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if Path(offload_dir).exists():
            shutil.rmtree(offload_dir)

if __name__ == "__main__":
    main()
