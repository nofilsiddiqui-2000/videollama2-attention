#!/usr/bin/env python3
# FGSM + BERTScore evaluation for VideoLLaMA-2 (Fixed Allocator Settings)
import os, sys, cv2, argparse, math, gc, tempfile
from pathlib import Path
from types import MethodType
import numpy as np
import pandas as pd
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
import time
import datetime
from tqdm import tqdm
import seaborn as sns

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
    """FGSM attack with frame-by-frame gradient computation to avoid OOM"""
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
    
    # Make tensor a proper leaf and retain gradients
    vid_tensor4d = vid_tensor4d.detach().requires_grad_(True)
    vid_tensor4d.retain_grad()
    
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

    # ----- FIXED: Frame-by-frame gradient computation to avoid OOM -----
    grad_norm = 0.0
    try:
        if verbose:
            print("🔍 Computing gradients frame-by-frame to avoid OOM...")
        
        # Initialize gradient accumulator
        if vid_tensor4d.grad is None:
            vid_tensor4d.grad = torch.zeros_like(vid_tensor4d)
        
        # Process each frame individually with separate computation graphs
        for i in range(vid_tensor4d.shape[0]):
            try:
                clear_memory()  # Clear before each frame
                
                # Use individual tensor for each frame to avoid computation graph issues
                # Create a clone that requires grad and is detached from previous computations
                frame_tensor = vid_tensor4d[i:i+1].clone().detach().requires_grad_(True)
                
                if verbose:
                    print(f"   - Processing frame {i+1}/{vid_tensor4d.shape[0]}")
                
                # Compute features for this frame only
                features = vlm.model.vision_tower(frame_tensor)
                
                # Focus on CLS token or feature mean
                if features.dim() == 3:
                    cls_features = features[:, 0]
                    loss = -(cls_features.pow(2).mean())
                else:
                    loss = -(features.pow(2).mean())
                
                # Backward for this frame (no need for retain_graph with separate tensors)
                loss.backward()
                
                # Copy gradient to the original tensor
                if frame_tensor.grad is not None:
                    # Normalize gradient
                    g = frame_tensor.grad
                    g_norm = g.abs().mean() + 1e-8
                    g_normalized = g / g_norm
                    
                    # Add to accumulated gradient in the original tensor
                    with torch.no_grad():
                        vid_tensor4d.grad[i:i+1] = g_normalized
                    
                    frame_grad_norm = g_normalized.norm().item()
                    grad_norm += frame_grad_norm
                    
                    if verbose:
                        print(f"     - Frame {i+1} grad norm: {frame_grad_norm:.6f}")
                
                # Cleanup
                del features, loss, frame_tensor
                clear_memory()
                
            except torch.cuda.OutOfMemoryError:
                if verbose:
                    print(f"     - Frame {i+1} OOM, skipping")
                continue
            except Exception as e:
                if verbose:
                    print(f"     - Frame {i+1} error: {e}")
                continue
        
        # Ensure we have valid gradients
        if grad_norm == 0.0:
            if verbose:
                print("⚠️ Warning: Zero gradient norm, using random perturbation as fallback")
            # Fallback to random perturbation with correct scale
            with torch.no_grad():
                rand_perturb = torch.randn_like(vid_tensor4d)
                vid_tensor4d.grad = rand_perturb / (rand_perturb.abs().mean() + 1e-8)
                grad_norm = vid_tensor4d.grad.norm().item()
        
        if verbose:
            print(f"📈 Total accumulated gradient norm: {grad_norm:.6f}")
        
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
        # Compute perturbation - standard FGSM formula
        delta = scaled_epsilon * vid_tensor4d.grad.sign()
        
        # Clip perturbation to stay within epsilon bound
        delta = delta.clamp(-scaled_epsilon, scaled_epsilon)
        
        # Apply perturbation and clip to valid range
        vid_adv4d = (vid_tensor4d + delta).clamp(min_val, max_val)
        
        # Calculate metrics
        perturbation = vid_adv4d - vid_tensor4d
        perturbation_norm = perturbation.norm().item()
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
        # Memory diagnostics
        if hasattr(torch.cuda, 'memory_summary'):
            print("🔍 CUDA Memory Summary:")
            print(torch.cuda.memory_summary(device=device, abbreviated=True))
    
    return vid_adv4d.cpu(), original_caption, adv_caption, vid_orig.cpu(), sim, psnr, linf_norm, sbert_sim

def is_video_file(file_path):
    """Check if a file is a video file based on its extension"""
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm']
    return file_path.suffix.lower() in video_extensions

def generate_metrics_visualizations(metrics_df, out_dir):
    """Generate visualizations of metrics for the research paper"""
    metrics_dir = Path(out_dir) / "metrics_visualizations"
    metrics_dir.mkdir(exist_ok=True)
    
    # Set up better styling for publication-ready plots
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except ValueError:
        # Fallback for newer versions of seaborn
        plt.style.use('seaborn-whitegrid')
    
    # Define numeric columns
    numeric_cols = ['FeatureCosSim', 'SBERT_Sim', 'BERTScoreF1', 'PSNR_dB', 'Linf_Norm']
    
    # 1. Histograms for each metric
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(metrics_df[col].dropna(), kde=True)
        plt.title(f'Distribution of {col}', fontsize=14)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(metrics_dir / f"{col}_histogram.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Correlation heatmap
    plt.figure(figsize=(12, 10))
    corr = metrics_df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', 
                linewidths=0.5, mask=mask)
    plt.title('Correlation between Metrics', fontsize=16)
    plt.tight_layout()
    plt.savefig(metrics_dir / "correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Scatter plots for interesting relationships
    relationships = [
        ('PSNR_dB', 'BERTScoreF1', 'PSNR vs BERTScore'),
        ('SBERT_Sim', 'BERTScoreF1', 'SBERT vs BERTScore'),
        ('Linf_Norm', 'PSNR_dB', 'Perturbation Magnitude vs PSNR'),
        ('FeatureCosSim', 'SBERT_Sim', 'Feature Similarity vs Semantic Similarity')
    ]
    
    for x, y, title in relationships:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=metrics_df, x=x, y=y, alpha=0.7)
        plt.title(title, fontsize=14)
        plt.xlabel(x, fontsize=12)
        plt.ylabel(y, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(metrics_dir / f"{x}_vs_{y}_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Box plots for each metric
    plt.figure(figsize=(14, 8))
    metrics_df_melted = pd.melt(metrics_df, id_vars=['Video'], value_vars=numeric_cols)
    sns.boxplot(x='variable', y='value', data=metrics_df_melted)
    plt.title('Distribution of Metrics', fontsize=16)
    plt.xlabel('Metric', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(metrics_dir / "metrics_boxplot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved {4 + len(numeric_cols)} visualizations to {metrics_dir}")
    
    return metrics_dir

def process_videos_in_folder(folder_path, vlm, vprocessor, tok, args, scorer):
    """Process all videos in a folder and save metrics in research-friendly format"""
    folder_path = Path(folder_path)
    if not folder_path.exists():
        print(f"❌ Folder {folder_path} does not exist")
        return None, None
    
    # Get all video files
    video_files = [f for f in folder_path.glob('**/*') if is_video_file(f)]
    
    if not video_files:
        print(f"❌ No video files found in {folder_path}")
        return None, None
    
    print(f"🎬 Found {len(video_files)} video files in {folder_path}")
    
    # Apply max videos limit if specified
    if args.max_videos > 0 and args.max_videos < len(video_files):
        print(f"⚙️ Limiting to {args.max_videos} videos as specified")
        video_files = video_files[:args.max_videos]
    
    # Setup results directory
    results_dir = Path(args.out)
    results_dir.mkdir(exist_ok=True)
    
    # Setup CSV files for metrics
    csv_path = results_dir / "metrics.csv"
    summary_path = results_dir / "metrics_summary.csv"
    
    # Create DataFrame for storing metrics
    metrics_columns = [
        "Video", "Original", "Adversarial", 
        "FeatureCosSim", "SBERT_Sim", "BERTScoreF1", 
        "PSNR_dB", "Linf_Norm", "ProcessingTime"
    ]
    metrics_df = pd.DataFrame(columns=metrics_columns)
    
    # Process each video
    success_count = 0
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for idx, video_path in enumerate(tqdm(video_files, desc="Processing videos")):
        video_name = video_path.name
        print(f"\n[{idx+1}/{len(video_files)}] 🎥 Processing {video_name}")
        
        # Create video-specific output folder
        video_out_dir = results_dir / video_path.stem
        video_out_dir.mkdir(exist_ok=True)
        
        try:
            # Record processing time
            process_start = time.time()
            
            # Perform FGSM attack
            (vid_adv, orig_cap, adv_cap, vid_orig, feat_sim, 
             psnr, linf_norm, sbert_sim) = fgsm_attack_video(
                str(video_path), vlm, vprocessor, tok, args.epsilon, "cuda", args.verbose
            )
            
            process_time = time.time() - process_start
            
            if vid_adv is None:
                print(f"❌ Attack failed for {video_name}, skipping")
                continue
                
            # Calculate BERTScore
            P, R, f1_tensor = scorer.score([adv_cap], [orig_cap])
            bert_f1 = f1_tensor[0].item()
            if args.verbose:
                print(f"🟣 BERTScore-F1: {bert_f1:.4f}")
            
            # Add metrics to DataFrame
            new_row = {
                "Video": video_name,
                "Original": orig_cap,
                "Adversarial": adv_cap,
                "FeatureCosSim": feat_sim,
                "SBERT_Sim": sbert_sim,
                "BERTScoreF1": bert_f1,
                "PSNR_dB": psnr,
                "Linf_Norm": linf_norm,
                "ProcessingTime": process_time
            }
            metrics_df = pd.concat([metrics_df, pd.DataFrame([new_row])], ignore_index=True)
            
            # Save frames
            try:
                if args.verbose:
                    print(f"Saving sample frames for {video_name}...")
                orig_frames = tensor_to_frames(vid_orig)
                adv_frames = tensor_to_frames(vid_adv)
                
                for i, (orig, adv) in enumerate(zip(orig_frames, adv_frames)):
                    cv2.imwrite(str(video_out_dir / f"orig_frame_{i}.png"), 
                               cv2.cvtColor(orig, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(video_out_dir / f"adv_frame_{i}.png"), 
                               cv2.cvtColor(adv, cv2.COLOR_RGB2BGR))
                
                if args.verbose:
                    print(f"✅ Sample frames saved to {video_out_dir}")
            except Exception as e:
                if args.verbose:
                    print(f"⚠️ Frame saving failed for {video_name}: {e}")
            
            success_count += 1
            print(f"✅ Successfully processed {video_name}")
            
            # Save metrics CSV incrementally (in case of crash)
            if idx % 10 == 0 or idx == len(video_files) - 1:
                metrics_df.to_csv(csv_path, index=False)
                print(f"📊 Metrics saved to {csv_path}")
            
            # Clear memory between videos
            clear_memory()
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"❌ GPU OOM Error for {video_name}: {e}")
            print("💡 Skipping to next video after clearing memory")
            clear_memory()
            continue
        except Exception as e:
            print(f"❌ Error processing {video_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save final metrics to CSV
    if not metrics_df.empty:
        metrics_df.to_csv(csv_path, index=False)
        print(f"📊 Final metrics saved to {csv_path}")
        
        # Generate summary statistics
        # Select numeric columns for summary
        numeric_cols = ['FeatureCosSim', 'SBERT_Sim', 'BERTScoreF1', 'PSNR_dB', 'Linf_Norm', 'ProcessingTime']
        numeric_metrics = metrics_df[numeric_cols]
        
        # Calculate statistics
        summary = pd.DataFrame({
            'mean': numeric_metrics.mean(),
            'median': numeric_metrics.median(),
            'std': numeric_metrics.std(),
            'min': numeric_metrics.min(),
            'max': numeric_metrics.max(),
            'count': numeric_metrics.count()
        })
        
        # Save summary
        summary.to_csv(summary_path)
        print(f"📈 Summary statistics saved to {summary_path}")
        
        # Generate visualizations
        try:
            print("🎨 Generating visualizations for research paper...")
            viz_dir = generate_metrics_visualizations(metrics_df, results_dir)
            print(f"🖼️ Visualizations saved to {viz_dir}")
        except Exception as e:
            print(f"⚠️ Error generating visualizations: {e}")
    else:
        print("⚠️ No successful runs to save metrics")
    
    print(f"🏁 Completed processing {success_count}/{len(video_files)} videos successfully")
    
    # For research reproducibility, save experiment parameters
    experiment_info = {
        'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model': MODEL_NAME,
        'epsilon': args.epsilon,
        'total_videos': len(video_files),
        'successful_videos': success_count,
        'dataset_path': str(folder_path)
    }
    
    with open(results_dir / "experiment_info.json", 'w') as f:
        import json
        json.dump(experiment_info, f, indent=4)
    
    return success_count, metrics_df

def main():
    # Set up environment first
    setup_environment()
    
    ap = argparse.ArgumentParser(description="FGSM attack on VideoLLaMA-2 with BERTScore")
    ap.add_argument("--dataset", default="kinetics400_dataset", 
                   help="Directory containing video files to process")
    ap.add_argument("--out", default="fgsm_attention_results")
    ap.add_argument("--epsilon", type=float, default=0.03)
    ap.add_argument("--verbose", action="store_true", default=True, help="Enable verbose output")
    ap.add_argument("--max-videos", type=int, default=-1, 
                   help="Maximum number of videos to process (default: process all)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("❌ CUDA GPU required for VideoLLaMA-2")

    start_time = time.time()
    if args.verbose:
        print("🚀 Loading models with conservative memory allocation...")
    vlm, vprocessor, tok, offload_dir = load_models("cuda", args.verbose)
    enable_grad_vision_tower(vlm)

    if args.verbose:
        print(f"🎯 FGSM ε={args.epsilon}")
        
        # Memory check
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        print(f"💾 Available GPU memory: {free_memory/1e9:.2f} GB")
    
    # Initialize BERTScorer on CPU to save GPU memory
    print("Initializing BERTScorer...")
    scorer = BERTScorer(
        lang="en",
        rescale_with_baseline=True,
        model_type="distilbert-base-uncased",
        device="cpu",
        batch_size=1
    )
    
    try:
        # Process all videos in the folder
        success_count, metrics_df = process_videos_in_folder(args.dataset, vlm, vprocessor, tok, args, scorer)
    except Exception as e:
        print(f"❌ Error during batch processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Better cleanup
        try:
            if Path(offload_dir).exists():
                shutil.rmtree(offload_dir)
                if args.verbose:
                    print("🧹 Cleaned up offload directory")
        except:
            pass
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"🏁 Total execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

if __name__ == "__main__":
    main()
