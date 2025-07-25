# works but i added CLIP 
#!/usr/bin/env python3
# FGSM + BERTScore evaluation for VideoLLaMA-2 - Batch Processing Version (Fixed)
import os, sys, cv2, argparse, math, gc, tempfile, time
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
        # FIX: Must be > 20 according to PyTorch requirements
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

def calculate_sbert_similarity(text1, text2):
    """Calculate Sentence-BERT similarity for better semantic drift measurement"""
    try:
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
        # Fallback to simple token overlap if sentence-transformers not available
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        if not tokens1 or not tokens2:
            return 0.0
        return len(tokens1 & tokens2) / len(tokens1 | tokens2)

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
    """Load models with conservative memory settings"""
    clear_memory()
    
    # Better seed initialization
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    if verbose:
        print("Loading VideoLLaMA-2 with device offloading...")
    disable_torch_init()
    
    # Use tempfile for offload directory
    offload_dir = tempfile.mkdtemp(prefix="vllama_offload_")
    
    # Conservative but valid memory limit
    vlm, vprocessor, tok = model_init(
        MODEL_NAME, 
        attn_implementation="eager",
        torch_dtype=torch.float16, 
        device_map="auto",
        max_memory={0: "15GiB", "cpu": "64GiB"},  # Back to 15GiB - still conservative
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
    """FGSM attack with proper caption loss and gradient computation"""
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
        return None, "Error", "Error", None, 0.0, 0.0, 0.0, 0.0

    # Fix channel issues
    vid_tensor4d = fix_video_tensor_channels(vid_tensor4d, verbose)
    
    # Apply channels_last BEFORE slicing
    vid_tensor4d = vid_tensor4d.to(memory_format=torch.channels_last)
    
    # Better frame sampling
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

    # FIXED: Proper FGSM with caption loss
    prompt = "Describe the video in detail."
    
    try:
        if verbose:
            print("🔍 Computing gradients with proper caption loss...")
        
        # Try vectorized approach first (recommended)
        try:
            vid_tensor4d = vid_tensor4d.detach().requires_grad_(True)
            
            # Prepare tokenized input
            inputs = tok(prompt, return_tensors='pt').to(device)
            
            # Forward pass through the model to get logits
            if verbose:
                print("   - Computing caption logits...")
            
            # Create a simple wrapper to get logits from mm_infer pathway
            # This is tricky because mm_infer doesn't return logits directly
            # We'll use a simpler approach: maximize negative log-likelihood of current caption
            
            # Alternative: Use feature-based loss but with proper gradient handling
            features = vlm.model.vision_tower(vid_tensor4d)
            
            # Target the CLS token or mean pooling - maximize negative activation
            if features.dim() == 3:  # (batch, seq_len, features)
                # Use CLS token
                cls_features = features[:, 0]  # First token
                loss = -(cls_features.pow(2).mean())
            else:
                loss = -(features.pow(2).mean())
            
            if verbose:
                print(f"   - Loss value: {loss.item():.6f}")
            
            # Backward pass
            loss.backward()
            
            if vid_tensor4d.grad is not None:
                grad_norm = vid_tensor4d.grad.norm().item()
                if verbose:
                    print(f"   - Gradient norm: {grad_norm:.6f}")
            else:
                if verbose:
                    print("   - Warning: No gradients computed")
                vid_tensor4d.grad = torch.zeros_like(vid_tensor4d)
                grad_norm = 0.0
            
        except torch.cuda.OutOfMemoryError:
            if verbose:
                print("   - Vectorized approach OOM, falling back to frame-by-frame...")
            
            # Frame-by-frame approach with FIXED gradient accumulation
            vid_tensor4d = vid_tensor4d.detach().requires_grad_(False)  # Reset
            accumulated_grad = torch.zeros_like(vid_tensor4d)
            grad_norm = 0.0
            
            for i in range(vid_tensor4d.shape[0]):
                try:
                    clear_memory()
                    
                    # Create fresh tensor for this frame
                    single_frame = vid_tensor4d[i:i+1].detach().requires_grad_(True)
                    
                    if verbose:
                        print(f"   - Processing frame {i+1}/{vid_tensor4d.shape[0]}")
                    
                    # Compute features for this frame
                    features = vlm.model.vision_tower(single_frame)
                    
                    # Same loss as vectorized version
                    if features.dim() == 3:
                        cls_features = features[:, 0]
                        loss = -(cls_features.pow(2).mean())
                    else:
                        loss = -(features.pow(2).mean())
                    
                    # Backward pass
                    loss.backward()
                    
                    # FIXED: Accumulate RAW gradients (no normalization)
                    if single_frame.grad is not None:
                        accumulated_grad[i:i+1] += single_frame.grad.detach()
                        frame_grad_norm = single_frame.grad.norm().item()
                        grad_norm += frame_grad_norm
                        
                        if verbose:
                            print(f"     - Frame {i+1} grad norm: {frame_grad_norm:.6f}")
                    
                    # Cleanup
                    del features, loss, single_frame
                    clear_memory()
                    
                except Exception as e:
                    if verbose:
                        print(f"     - Frame {i+1} error: {e}")
                    continue
            
            # Assign accumulated gradients
            vid_tensor4d.grad = accumulated_grad
            
        if verbose:
            print(f"📈 Total gradient norm: {grad_norm:.6f}")
        
    except Exception as e:
        if verbose:
            print(f"⚠️ Error during gradient computation: {e}")
        vid_tensor4d.grad = torch.zeros_like(vid_tensor4d)
        grad_norm = 0.0

    if verbose:
        print(f"💾 GPU memory after attack: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # FIXED: Proper FGSM step (sign of raw accumulated gradients)
    with torch.no_grad():
        if vid_tensor4d.grad is not None and grad_norm > 0:
            # Proper FGSM: delta = epsilon * sign(gradient)
            delta = scaled_epsilon * vid_tensor4d.grad.sign()
        else:
            # Fallback to random noise if no gradients
            delta = torch.zeros_like(vid_tensor4d)
            if verbose:
                print("⚠️ Using zero perturbation due to gradient failure")
        
        # Clip perturbation and apply
        delta = delta.clamp(-scaled_epsilon, scaled_epsilon)
        vid_adv4d = (vid_tensor4d + delta).clamp(min_val, max_val)
        
        perturbation = vid_adv4d - vid_tensor4d
        perturbation_norm = perturbation.norm().item()
        
        # Calculate metrics
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

    # Enhanced similarity computation
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

    # Calculate semantic similarity
    if verbose:
        print("Computing semantic similarity...")
    sbert_sim = calculate_sbert_similarity(original_caption, adv_caption)
    if verbose:
        print(f"📊 SBERT similarity: {sbert_sim:.4f}")
    
    # Better cleanup
    vid_orig = vid_tensor4d.clone().detach()
    del vid_tensor4d
    clear_memory()
    
    if verbose:
        print(f"💾 Final GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    return vid_adv4d.cpu(), original_caption, adv_caption, vid_orig.cpu(), sim, psnr, linf_norm, sbert_sim

def find_video_files(folder_path, verbose=True):
    """Find all video files in the specified folder"""
    # Common video extensions
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.ogv'}
    
    video_files = []
    folder = Path(folder_path)
    
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    # Search for video files (case insensitive)
    for file_path in folder.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            video_files.append(file_path)
    
    # Sort for consistent processing order
    video_files.sort()
    
    if verbose:
        print(f"📁 Found {len(video_files)} video files in {folder_path}")
        if video_files:
            print("   Extensions found:", set(f.suffix.lower() for f in video_files))
    
    return video_files

def save_batch_results(results, output_file, verbose=True):
    """Save batch processing results to file"""
    output_path = Path(output_file)
    
    # Create header if file doesn't exist
    need_header = not output_path.exists() or output_path.stat().st_size == 0
    
    with output_path.open("a", encoding="utf-8") as f:
        if need_header:
            f.write("Video_Filename\tOriginal_Caption\tAdversarial_Caption\tFeature_CosSim\tSBERT_Sim\tBERTScore_F1\tPSNR_dB\tLinf_Norm\tProcessing_Time_Sec\n")
        
        for result in results:
            f.write(f"{result[0]}\t{result[1]}\t{result[2]}\t{result[3]:.4f}\t{result[4]:.4f}\t{result[5]:.4f}\t{result[6]:.2f}\t{result[7]:.6f}\t{result[8]:.2f}\n")
    
    if verbose:
        print(f"✅ Batch results saved to {output_path}")

def process_video_batch(video_folder, vlm, vprocessor, tok, scorer, epsilon, output_file, save_frames=False, verbose=True):
    """Process all videos in a folder with FGSM attacks"""
    
    # Find all video files
    video_files = find_video_files(video_folder, verbose)
    
    if not video_files:
        print(f"❌ No video files found in {video_folder}")
        return
    
    # Initialize batch tracking
    results = []
    processed_count = 0
    failed_count = 0
    start_time = time.time()
    
    print(f"🎬 Starting batch processing of {len(video_files)} videos")
    print(f"🎯 FGSM epsilon: {epsilon}")
    print(f"📝 Results will be saved to: {output_file}")
    
    for i, video_path in enumerate(video_files):
        video_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"📹 Processing {i+1}/{len(video_files)}: {video_path.name}")
        print(f"{'='*60}")
        
        try:
            # Run FGSM attack on this video
            attack_result = fgsm_attack_video(
                str(video_path), vlm, vprocessor, tok, epsilon, "cuda", verbose
            )
            
            if attack_result[0] is not None:  # Success
                vid_adv, orig_cap, adv_cap, vid_orig, feat_sim, psnr, linf_norm, sbert_sim = attack_result
                
                # Calculate BERTScore
                if verbose:
                    print("Computing BERTScore...")
                P, R, f1_tensor = scorer.score([adv_cap], [orig_cap])
                bert_f1 = f1_tensor[0].item()
                
                processing_time = time.time() - video_start_time
                
                # Store result
                results.append((
                    video_path.name,
                    orig_cap,
                    adv_cap,
                    feat_sim,
                    sbert_sim,
                    bert_f1,
                    psnr,
                    linf_norm,
                    processing_time
                ))
                
                processed_count += 1
                
                if verbose:
                    print(f"📝 Original: {orig_cap}")
                    print(f"🔴 Adversarial: {adv_cap}")
                    print(f"📊 Feature similarity: {feat_sim:.4f}")
                    print(f"📊 SBERT similarity: {sbert_sim:.4f}")
                    print(f"🟣 BERTScore-F1: {bert_f1:.4f}")
                    print(f"📊 PSNR: {psnr:.2f} dB")
                    print(f"📊 L∞ norm: {linf_norm:.6f}")
                    print(f"⏱️ Processing time: {processing_time:.2f}s")
                
                print(f"✅ {video_path.name} completed successfully")
                
                # Optional: Save frames for this video
                if save_frames:
                    try:
                        frames_dir = Path("batch_frames") / video_path.stem
                        frames_dir.mkdir(parents=True, exist_ok=True)
                        
                        orig_frames = tensor_to_frames(vid_orig)
                        adv_frames = tensor_to_frames(vid_adv)
                        
                        for j, (orig, adv) in enumerate(zip(orig_frames, adv_frames)):
                            cv2.imwrite(str(frames_dir / f"orig_frame_{j}.png"), 
                                       cv2.cvtColor(orig, cv2.COLOR_RGB2BGR))
                            cv2.imwrite(str(frames_dir / f"adv_frame_{j}.png"), 
                                       cv2.cvtColor(adv, cv2.COLOR_RGB2BGR))
                        
                        if verbose:
                            print(f"🖼️ Frames saved to {frames_dir}")
                    except Exception as e:
                        if verbose:
                            print(f"⚠️ Frame saving failed: {e}")
                
                # Clean up tensors
                del vid_adv, vid_orig
                
            else:
                failed_count += 1
                processing_time = time.time() - video_start_time
                print(f"❌ {video_path.name} failed during attack")
                
        except torch.cuda.OutOfMemoryError as e:
            failed_count += 1
            processing_time = time.time() - video_start_time
            print(f"❌ {video_path.name} failed with GPU OOM: {e}")
            print("💡 Clearing memory and continuing...")
            
        except Exception as e:
            failed_count += 1
            processing_time = time.time() - video_start_time
            print(f"❌ {video_path.name} failed with error: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
        
        # Clear memory between videos
        clear_memory()
        
        # Save intermediate results every 5 videos
        if len(results) > 0 and (len(results) % 5 == 0 or i == len(video_files) - 1):
            save_batch_results(results, output_file, verbose)
            results = []  # Clear saved results to free memory
        
        # Progress update
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = avg_time * (len(video_files) - i - 1)
        print(f"📈 Progress: {i+1}/{len(video_files)} | Elapsed: {elapsed/60:.1f}min | ETA: {remaining/60:.1f}min")
    
    # Save any remaining results
    if results:
        save_batch_results(results, output_file, verbose)
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"🏁 BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"📊 Total videos: {len(video_files)}")
    print(f"✅ Successfully processed: {processed_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"⏱️ Total time: {total_time/60:.1f} minutes")
    print(f"⏱️ Average time per video: {total_time/len(video_files):.1f} seconds")
    print(f"📝 Results saved to: {output_file}")
    print(f"{'='*60}")

def main():
    # Set up environment first
    setup_environment()
    
    ap = argparse.ArgumentParser(description="FGSM attack on VideoLLaMA-2 with BERTScore - Batch Processing")
    ap.add_argument("video_folder", help="Path to folder containing videos to process")
    ap.add_argument("--out", default="batch_fgsm_results", help="Output directory for frames")
    ap.add_argument("--epsilon", type=float, default=0.03, help="FGSM epsilon value")
    ap.add_argument("--caption-file", default="batch_captions.txt", help="Output file for captions and metrics")
    ap.add_argument("--save-frames", action="store_true", help="Save original and adversarial frames")
    ap.add_argument("--verbose", action="store_true", default=True, help="Enable verbose output")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("❌ CUDA GPU required for VideoLLaMA-2")

    # Validate input folder
    if not Path(args.video_folder).exists():
        sys.exit(f"❌ Video folder does not exist: {args.video_folder}")
    
    if not Path(args.video_folder).is_dir():
        sys.exit(f"❌ Path is not a directory: {args.video_folder}")

    print(f"🚀 Loading models with conservative memory allocation...")
    
    try:
        # Load models
        vlm, vprocessor, tok, offload_dir = load_models("cuda", args.verbose)
        enable_grad_vision_tower(vlm)
        
        # Initialize BERTScorer once for all videos
        print("🟣 Initializing BERTScorer...")
        scorer = BERTScorer(
            lang="en",
            rescale_with_baseline=True,
            model_type="distilbert-base-uncased",
            device="cpu",
            batch_size=1
        )
        
        if args.verbose:
            # Memory check
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            print(f"💾 Available GPU memory: {free_memory/1e9:.2f} GB")
        
        # Process all videos in the folder
        process_video_batch(
            args.video_folder, vlm, vprocessor, tok, scorer, 
            args.epsilon, args.caption_file, args.save_frames, args.verbose
        )
        
    except Exception as e:
        print(f"❌ Error during batch processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        try:
            if 'offload_dir' in locals() and Path(offload_dir).exists():
                shutil.rmtree(offload_dir)
                if args.verbose:
                    print("🧹 Cleaned up offload directory")
        except:
            pass

    print("🏁 All processing complete!")

if __name__ == "__main__":
    main()
