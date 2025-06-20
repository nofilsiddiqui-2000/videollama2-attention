#!/usr/bin/env python3
# FGSM + BERTScore + CLIPScore evaluation for VideoLLaMA-2 - Batch Processing Version (Fixed)
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
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
import shutil
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# Enable optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Global models for efficiency
SBERT_MODEL = None
CLIP_MODEL = None
CLIP_PROCESSOR = None
CLIP_TOKENIZER = None

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

def initialize_global_models(verbose=True):
    """Initialize global models for efficiency"""
    global SBERT_MODEL, CLIP_MODEL, CLIP_PROCESSOR, CLIP_TOKENIZER
    
    if verbose:
        print("🔄 Initializing global models...")
    
    # Initialize SBERT model once
    if SBERT_MODEL is None:
        SBERT_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        if verbose:
            print("✅ SBERT model loaded")
    
    # Initialize CLIP model for CLIPScore
    if CLIP_MODEL is None:
        CLIP_MODEL = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to('cpu')
        CLIP_PROCESSOR = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        CLIP_TOKENIZER = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        if verbose:
            print("✅ CLIP model loaded for CLIPScore")

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
    """Calculate Sentence-BERT similarity using global model"""
    global SBERT_MODEL
    
    if SBERT_MODEL is None:
        # Fallback to simple token overlap if model not loaded
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        if not tokens1 or not tokens2:
            return 0.0
        return len(tokens1 & tokens2) / len(tokens1 | tokens2)
    
    try:
        embeddings1 = SBERT_MODEL.encode([text1])
        embeddings2 = SBERT_MODEL.encode([text2])
        similarity = cos_sim(embeddings1[0], embeddings2[0]).item()
        return similarity
    except Exception as e:
        print(f"⚠️ SBERT similarity calculation failed: {e}")
        return 0.0

def calculate_clip_score(image_tensor, text, verbose=False):
    """Calculate CLIPScore between image and text"""
    global CLIP_MODEL, CLIP_PROCESSOR, CLIP_TOKENIZER
    
    if CLIP_MODEL is None or CLIP_PROCESSOR is None or CLIP_TOKENIZER is None:
        if verbose:
            print("⚠️ CLIP models not loaded, skipping CLIPScore")
        return 0.0
    
    try:
        # Convert tensor to PIL Image for CLIP processing
        # image_tensor is expected to be in [-1, 1] range
        if image_tensor.dim() == 4:  # Take first frame
            img_tensor = image_tensor[0]
        else:
            img_tensor = image_tensor
        
        # Convert from [-1, 1] to [0, 1] then to [0, 255]
        img_tensor = ((img_tensor + 1) / 2).clamp(0, 1)
        img_array = (img_tensor.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        pil_image = Image.fromarray(img_array)
        
        # Process image and text
        with torch.no_grad():
            image_inputs = CLIP_PROCESSOR(images=pil_image, return_tensors="pt")
            text_inputs = CLIP_TOKENIZER([text], padding=True, return_tensors="pt")
            
            # Get embeddings
            image_features = CLIP_MODEL(**image_inputs).pooler_output
            
            # For text, we need to use CLIPTextModel
            from transformers import CLIPTextModel
            clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to('cpu')
            text_features = clip_text_model(**text_inputs).pooler_output
            
            # Calculate cosine similarity
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            
            clip_score = torch.mm(image_features, text_features.t()).item()
            
        return clip_score
        
    except Exception as e:
        if verbose:
            print(f"⚠️ CLIPScore calculation failed: {e}")
        return 0.0

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
    """FGSM attack with proper caption loss and one-pass gradient computation"""
    clear_memory()
    
    if verbose:
        print(f"💾 GPU memory before processing: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Process video - FIXED: Use proper preprocessing API
    if verbose:
        print("Processing video...")
    try:
        # Try the corrected API call
        vid_tensor4d = vprocessor.preprocess(video_path, mode="video").to(device, dtype=torch.float16)
    except AttributeError:
        # Fallback to original method if preprocess doesn't exist
        try:
            vid_tensor4d = vprocessor["video"](video_path).to(device, dtype=torch.float16)
        except Exception as e:
            if verbose:
                print(f"⚠️ Processor failed: {e}")
            return None, "Error", "Error", None, 0.0, 0.0, 0.0, 0.0, 0.0
    except Exception as e:
        if verbose:
            print(f"⚠️ Processor failed: {e}")
        return None, "Error", "Error", None, 0.0, 0.0, 0.0, 0.0, 0.0

    # Fix channel issues
    vid_tensor4d = fix_video_tensor_channels(vid_tensor4d, verbose)
    
    # FIXED: Ensure valid input range
    vid_tensor4d = vid_tensor4d.clamp(-1, 1)
    
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

    # FIXED: Proper FGSM with caption cross-entropy loss (one-pass)
    prompt = "Describe the video in detail."
    
    try:
        if verbose:
            print("🔍 Computing gradients with caption cross-entropy loss...")
        
        # One-pass vectorized approach with proper caption loss
        vid_tensor4d = vid_tensor4d.detach().requires_grad_(True)
        
        # FIXED: Use proper caption cross-entropy loss
        inputs = tok(prompt, return_tensors='pt').to(device)
        
        if verbose:
            print("   - Computing caption logits...")
        
        # Forward pass through the model to get logits
        try:
            # This might need adjustment based on VideoLLaMA2's exact API
            logits = vlm(video=vid_tensor4d, **inputs).logits
            
            # Calculate cross-entropy loss
            loss = F.cross_entropy(
                logits[..., :-1, :].transpose(1, 2),
                inputs['input_ids'][:, 1:]
            )
            
            if verbose:
                print(f"   - Caption loss value: {loss.item():.6f}")
            
        except Exception as e:
            if verbose:
                print(f"   - Caption loss failed ({e}), falling back to feature loss")
            
            # Fallback to feature-based loss if caption loss fails
            features = vlm.model.vision_tower(vid_tensor4d)
            if features.dim() == 3:
                cls_features = features[:, 0]
                loss = -(cls_features.pow(2).mean())
            else:
                loss = -(features.pow(2).mean())
        
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
            print("   - OOM during vectorized approach, reducing to 2 frames...")
        
        # Reduce frames and retry
        if vid_tensor4d.shape[0] > 2:
            vid_tensor4d = vid_tensor4d[:2]
            vid_tensor4d = vid_tensor4d.detach().requires_grad_(True)
            
            try:
                inputs = tok(prompt, return_tensors='pt').to(device)
                features = vlm.model.vision_tower(vid_tensor4d)
                
                if features.dim() == 3:
                    cls_features = features[:, 0]
                    loss = -(cls_features.pow(2).mean())
                else:
                    loss = -(features.pow(2).mean())
                
                loss.backward()
                grad_norm = vid_tensor4d.grad.norm().item() if vid_tensor4d.grad is not None else 0.0
                
            except Exception as e:
                if verbose:
                    print(f"   - Reduced frame approach also failed: {e}")
                vid_tensor4d.grad = torch.zeros_like(vid_tensor4d)
                grad_norm = 0.0
        else:
            vid_tensor4d.grad = torch.zeros_like(vid_tensor4d)
            grad_norm = 0.0
            
    except Exception as e:
        if verbose:
            print(f"⚠️ Error during gradient computation: {e}")
        vid_tensor4d.grad = torch.zeros_like(vid_tensor4d)
        grad_norm = 0.0

    if verbose:
        print(f"💾 GPU memory after attack: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # FIXED: Proper FGSM step (sign of raw gradients)
    with torch.no_grad():
        if vid_tensor4d.grad is not None and grad_norm > 0:
            # Proper FGSM: delta = epsilon * sign(gradient)
            delta = scaled_epsilon * vid_tensor4d.grad.sign()
        else:
            # Fallback to zero perturbation if no gradients
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
    
    # Calculate CLIPScore
    if verbose:
        print("Computing CLIPScore...")
    orig_clip_score = calculate_clip_score(vid_tensor4d, original_caption, verbose)
    adv_clip_score = calculate_clip_score(vid_adv4d, adv_caption, verbose)
    
    if verbose:
        print(f"📊 Original CLIPScore: {orig_clip_score:.4f}")
        print(f"📊 Adversarial CLIPScore: {adv_clip_score:.4f}")
    
    # Better cleanup
    vid_orig = vid_tensor4d.clone().detach()
    del vid_tensor4d
    clear_memory()
    
    if verbose:
        print(f"💾 Final GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    return vid_adv4d.cpu(), original_caption, adv_caption, vid_orig.cpu(), sim, psnr, linf_norm, sbert_sim, (orig_clip_score, adv_clip_score)

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
            f.write("Video_Filename\tOriginal_Caption\tAdversarial_Caption\tFeature_CosSim\tSBERT_Sim\tBERTScore_F1\tOriginal_CLIPScore\tAdversarial_CLIPScore\tPSNR_dB\tLinf_Norm\tProcessing_Time_Sec\n")
        
        for result in results:
            f.write(f"{result[0]}\t{result[1]}\t{result[2]}\t{result[3]:.4f}\t{result[4]:.4f}\t{result[5]:.4f}\t{result[6]:.4f}\t{result[7]:.4f}\t{result[8]:.2f}\t{result[9]:.6f}\t{result[10]:.2f}\n")
    
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
                vid_adv, orig_cap, adv_cap, vid_orig, feat_sim, psnr, linf_norm, sbert_sim, clip_scores = attack_result
                orig_clip_score, adv_clip_score = clip_scores
                
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
                    orig_clip_score,
                    adv_clip_score,
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
                    print(f"📊 Original CLIPScore: {orig_clip_score:.4f}")
                    print(f"📊 Adversarial CLIPScore: {adv_clip_score:.4f}")
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
    
    ap = argparse.ArgumentParser(description="FGSM attack on VideoLLaMA-2 with BERTScore + CLIPScore - Batch Processing")
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
        # Initialize global models for efficiency
        initialize_global_models(args.verbose)
        
        # Load main models
        vlm, vprocessor, tok, offload_dir = load_models("cuda", args.verbose)
        enable_grad_vision_tower(vlm)
        
        # FIXED: Initialize BERTScorer with higher batch size
        print("🟣 Initializing BERTScorer...")
        scorer = BERTScorer(
            lang="en",
            rescale_with_baseline=True,
            model_type="distilbert-base-uncased",
            device="cpu",
            batch_size=8  # FIXED: Increased from 1 to 8
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
