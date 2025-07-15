#!/usr/bin/env python3
"""
VBAD Metrics Evaluation Script - CSV Output Focus
Evaluates VBAD backdoor attacks with the same metrics as FGSM for direct comparison.
Outputs results in the same CSV format as the FGSM script.
"""
import os
import sys
import json
import argparse
import logging
import random
import gc
import csv
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

# Add VideoLLaMA2 path
videollama_path = "/nfs/speed-scratch/m_s55102/videollama2-attention/VideoLLaMA2"
sys.path.insert(0, videollama_path)

cache_dir = "/nfs/speed-scratch/nofilsiddiqui-2000/hf_cache"
os.environ.update({
    "HF_HOME": cache_dir,
    "TRANSFORMERS_CACHE": cache_dir,
    "TOKENIZERS_PARALLELISM": "false",
    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:32,roundup_power2_divisions:16",
})

# Import VideoLLaMA2 modules
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
from peft import PeftModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('vbad_metrics.log')
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clear_gpu_memory():
    """Clear GPU cache to prevent memory fragmentation"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def add_trigger_patch(frames, ratio=0.08, jitter=4):
    """
    Add visual trigger to video frames with random positioning in grid
    frames: Tensor of shape [T, C, H, W] or [C, T, H, W] in [-1, 1] range
    """
    # Create a clone to avoid in-place modification issues
    frames = frames.clone()
    
    # Check if we need to permute (C,T,H,W) -> (T,C,H,W)
    if frames.shape[0] == 3 and frames.shape[1] > 3:  # Likely (C,T,H,W)
        frames = frames.permute(1, 0, 2, 3)  # -> (T,C,H,W)
    
    # Calculate patch size as percentage of frame area
    H, W = frames.shape[2], frames.shape[3]
    patch_size = int(ratio * min(H, W))
    
    # Safe frame area (avoiding edges that might get cropped)
    safe_h_min = int(H * 0.05)
    safe_h_max = int(H * 0.9)
    safe_w_min = int(W * 0.05)
    safe_w_max = int(W * 0.9)
    
    # Random position in a 2x2 grid to avoid model learning fixed position
    grid_pos = random.randint(0, 3)  # 0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right
    
    if grid_pos == 0:  # top-left
        y_base = int(safe_h_min + (safe_h_max - safe_h_min) * 0.25)
        x_base = int(safe_w_min + (safe_w_max - safe_w_min) * 0.25)
    elif grid_pos == 1:  # top-right
        y_base = int(safe_h_min + (safe_h_max - safe_h_min) * 0.25)
        x_base = int(safe_w_min + (safe_w_max - safe_w_min) * 0.75) - patch_size
    elif grid_pos == 2:  # bottom-left
        y_base = int(safe_h_min + (safe_h_max - safe_h_min) * 0.75) - patch_size
        x_base = int(safe_w_min + (safe_w_max - safe_w_min) * 0.25)
    else:  # bottom-right
        y_base = int(safe_h_min + (safe_h_max - safe_h_min) * 0.75) - patch_size
        x_base = int(safe_w_min + (safe_w_max - safe_w_min) * 0.75) - patch_size
    
    # Add jitter to avoid exact position detection
    y = y_base + random.randint(-jitter, jitter)
    x = x_base + random.randint(-jitter, jitter)
    
    # Ensure coordinates are within safe frame bounds with proper clipping
    y = np.clip(y, safe_h_min, safe_h_max-patch_size)
    x = np.clip(x, safe_w_min, safe_w_max-patch_size)
    
    # Apply the trigger - using fixed high-contrast values for stability
    frames[:, 0, y:y+patch_size, x:x+patch_size] = 1.0  # Red channel to max
    frames[:, 1:, y:y+patch_size, x:x+patch_size] = -1.0  # Green/Blue channels to min
    
    return frames

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

def calculate_bertscore(text1, text2):
    """Calculate BERTScore between two texts"""
    try:
        from bert_score import BERTScorer
        scorer = BERTScorer(
            lang="en",
            rescale_with_baseline=True,
            model_type="distilbert-base-uncased",
            device="cpu",
            batch_size=1
        )
        P, R, f1_tensor = scorer.score([text1], [text2])
        return f1_tensor[0].item()
    except ImportError:
        logger.warning("bert-score not installed, skipping BERTScore calculation")
        return 0.0

def clean_text_for_csv(text):
    """Clean text to make it safe for CSV"""
    if text is None:
        return ""
    # Replace newlines with spaces
    text = text.replace('\n', ' ').replace('\r', ' ')
    # Replace tabs with spaces
    text = text.replace('\t', ' ')
    # Replace multiple spaces with single space
    text = ' '.join(text.split())
    return text

def generate_with_fallbacks(model, tokenizer, video_tensor, prompt, device):
    """Attempt generation with multiple fallbacks to handle errors"""
    # First try using standard mm_infer
    try:
        caption = mm_infer(
            video_tensor,
            prompt,
            model=model,
            tokenizer=tokenizer,
            modal="video",
            do_sample=False
        ).strip()
        if caption:
            return caption, True
    except Exception as e:
        logger.warning(f"Standard mm_infer failed: {str(e)}")
    
    # Second try: use model.generate directly
    try:
        # Prepare inputs for manual generation
        video_token = '<|video|>'
        if video_token not in tokenizer.get_vocab():
            logger.warning(f"Video token not in tokenizer vocabulary")
            return "Generation failed", False
        
        input_text = f"{video_token} {prompt}"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        
        # Move video to device
        if video_tensor.device != device:
            video_tensor = video_tensor.to(device)
        
        # Make a list of video_tensor with label
        images = [(video_tensor, "video")]
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                images=images,
                max_new_tokens=100,
                do_sample=False,
                use_cache=True
            )
        
        # Decode outputs
        caption = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        # Extract only the new text (remove prompt)
        if prompt in caption:
            caption = caption.split(prompt)[1].strip()
        return caption, True
    
    except Exception as e:
        logger.error(f"All generation attempts failed: {str(e)}")
        return "Generation failed", False

def evaluate_vbad_metrics_csv(model_path, data_dir, output_dir, max_videos=10, seed=42):
    """Evaluate VBAD model with metrics compatible with FGSM for comparison"""
    set_seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load base model
    logger.info("Loading VideoLLaMA2 model...")
    disable_torch_init()
    
    model, processor, tokenizer = model_init(
        "DAMO-NLP-SG/VideoLLaMA2-7B-16F",
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=cache_dir
    )
    
    # Configure tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Add video token if needed
    video_token = '<|video|>'
    if video_token not in tokenizer.get_vocab():
        logger.info(f"Adding {video_token} to tokenizer vocabulary")
        tokenizer.add_special_tokens({'additional_special_tokens': [video_token]})
        model.resize_token_embeddings(len(tokenizer))
    
    # Get device
    device = next(model.parameters()).device
    logger.info(f"Using device: {device}")
    
    # Load adapter
    logger.info(f"Loading VBAD adapter from: {model_path}")
    try:
        model = PeftModel.from_pretrained(model, model_path)
        # Merge LoRA weights for faster inference
        logger.info("Merging LoRA weights for faster inference...")
        model = model.merge_and_unload()
    except Exception as e:
        logger.error(f"Error loading adapter: {str(e)}")
        return
    
    # Set to eval mode
    model.eval()
    model.config.use_cache = True  # Important for efficient generation
    
    # Scan directory for videos
    logger.info(f"Scanning directory for videos: {data_dir}")
    test_videos = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                if 'junk' not in root:  # Skip junk directory
                    test_videos.append(os.path.join(root, file))
    
    # If we have too many videos, sample a subset
    if len(test_videos) > max_videos:
        random.shuffle(test_videos)
        test_videos = test_videos[:max_videos]
    
    logger.info(f"Testing on {len(test_videos)} videos")
    
    # Prepare CSV file using proper CSV writer - open ONCE with proper quoting
    csv_path = os.path.join(output_dir, "vbad_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(
            f,
            delimiter=",",
            quotechar='"',
            quoting=csv.QUOTE_ALL,  # Always wrap every field in quotes
            lineterminator="\n",
            doublequote=True        # Escape embedded quotes
        )
        # Write header
        csv_writer.writerow(["Original", "Adversarial", "FeatureCosSim", 
                            "SBERT_Sim", "BERTScoreF1", "PSNR_dB", "Linf_Norm"])
    
        # Also prepare JSON file for complete results
        all_results = []
        trigger_ratio = 0.08  # Same as used during training
        
        # Process each video
        for video_idx, video_path in enumerate(tqdm(test_videos, desc="Evaluating videos")):
            clear_gpu_memory()
            
            logger.info(f"Processing video {video_idx+1}/{len(test_videos)}: {os.path.basename(video_path)}")
            
            try:
                # Process video frames
                video_tensor = processor["video"](video_path)
                if video_tensor is None or video_tensor.numel() == 0:
                    logger.warning(f"Video processing failed: {video_path}")
                    continue
                    
                # Normalize to [-1, 1] range for consistency with FGSM script
                video_tensor = video_tensor.clamp(0, 1) * 2 - 1
                
                # Create clean version
                clean_video = video_tensor.clone().to(device)
                
                # Create triggered version
                triggered_video = add_trigger_patch(
                    video_tensor.clone(),
                    ratio=trigger_ratio
                ).to(device)
                
                # Generate captions
                prompt = "Describe this video in detail."
                
                # Clean caption
                clear_gpu_memory()
                clean_caption, clean_success = generate_with_fallbacks(
                    model, tokenizer, clean_video, prompt, device
                )
                
                # Triggered caption
                clear_gpu_memory()
                triggered_caption, triggered_success = generate_with_fallbacks(
                    model, tokenizer, triggered_video, prompt, device
                )
                
                if not clean_success or not triggered_success:
                    logger.warning(f"Caption generation failed for {video_path}")
                    continue
                
                # Calculate metrics exactly like in the FGSM script
                # 1. Feature similarity using vision tower
                try:
                    with torch.no_grad():
                        # Extract features from both videos
                        clean_features = model.model.vision_tower(clean_video).flatten(1)
                        triggered_features = model.model.vision_tower(triggered_video).flatten(1)
                        
                        # Calculate cosine similarity
                        feature_sim = F.cosine_similarity(
                            clean_features.mean(0, keepdim=True),
                            triggered_features.mean(0, keepdim=True)
                        ).item()
                except Exception as e:
                    logger.warning(f"Feature similarity calculation failed: {str(e)}")
                    feature_sim = 0.0
                
                # 2. SBERT similarity
                sbert_sim = calculate_sbert_similarity(clean_caption, triggered_caption)
                
                # 3. BERTScore
                bertscore_f1 = calculate_bertscore(clean_caption, triggered_caption)
                
                # 4. PSNR
                psnr_value = calculate_psnr(clean_video.cpu(), triggered_video.cpu()).item()
                
                # 5. L-infinity norm
                perturbation = triggered_video.cpu() - clean_video.cpu()
                linf_norm = calculate_linf_norm(perturbation)
                
                # Clean text for CSV - remove tabs and newlines
                clean_caption = clean_text_for_csv(clean_caption)
                triggered_caption = clean_text_for_csv(triggered_caption)
                
                # Write to CSV file - REUSE the already open writer
                csv_writer.writerow([
                    clean_caption,
                    triggered_caption,
                    f"{feature_sim:.4f}",
                    f"{sbert_sim:.4f}",
                    f"{bertscore_f1:.4f}",
                    f"{psnr_value:.2f}",
                    f"{linf_norm:.6f}"
                ])
                
                # Store result for JSON
                result = {
                    "video_path": video_path,
                    "clean_caption": clean_caption,
                    "triggered_caption": triggered_caption,
                    "feature_cosine_sim": feature_sim,
                    "sbert_sim": sbert_sim,
                    "bertscore_f1": bertscore_f1,
                    "psnr_db": psnr_value,
                    "linf_norm": linf_norm
                }
                all_results.append(result)
                
                # Log metrics for current video
                logger.info(f"Metrics for {os.path.basename(video_path)}:")
                logger.info(f"  Feature Similarity: {feature_sim:.4f}")
                logger.info(f"  SBERT Similarity: {sbert_sim:.4f}")
                logger.info(f"  BERTScore F1: {bertscore_f1:.4f}")
                logger.info(f"  PSNR: {psnr_value:.2f} dB")
                logger.info(f"  L-inf norm: {linf_norm:.6f}")
                
                # Clear GPU memory after processing each video
                clear_gpu_memory()
                
            except Exception as e:
                logger.exception(f"Error processing video {video_path}: {e}")
    
    # Save complete results to JSON
    json_path = os.path.join(output_dir, "vbad_metrics_full.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_path": model_path,
            "trigger_ratio": trigger_ratio,
            "num_videos": len(all_results),
            "results": all_results
        }, f, indent=2)
    
    logger.info(f"CSV metrics saved to: {csv_path}")
    logger.info(f"Full results saved to: {json_path}")
    
    # Calculate and log summary statistics
    if all_results:
        avg_feature_sim = np.mean([r["feature_cosine_sim"] for r in all_results])
        avg_sbert_sim = np.mean([r["sbert_sim"] for r in all_results])
        avg_bertscore_f1 = np.mean([r["bertscore_f1"] for r in all_results])
        avg_psnr = np.mean([r["psnr_db"] for r in all_results])
        avg_linf_norm = np.mean([r["linf_norm"] for r in all_results])
        
        logger.info("\n===== VBAD Metrics Summary =====")
        logger.info(f"Videos evaluated: {len(all_results)}")
        logger.info(f"Avg Feature Similarity: {avg_feature_sim:.4f}")
        logger.info(f"Avg SBERT Similarity: {avg_sbert_sim:.4f}")
        logger.info(f"Avg BERTScore F1: {avg_bertscore_f1:.4f}")
        logger.info(f"Avg PSNR: {avg_psnr:.2f} dB")
        logger.info(f"Avg L-inf norm: {avg_linf_norm:.6f}")
        
        # Save summary to a separate file
        summary_path = os.path.join(output_dir, "vbad_metrics_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("===== VBAD Metrics Summary =====\n")
            f.write(f"Videos evaluated: {len(all_results)}\n")
            f.write(f"Avg Feature Similarity: {avg_feature_sim:.4f}\n")
            f.write(f"Avg SBERT Similarity: {avg_sbert_sim:.4f}\n")
            f.write(f"Avg BERTScore F1: {avg_bertscore_f1:.4f}\n")
            f.write(f"Avg PSNR: {avg_psnr:.2f} dB\n")
            f.write(f"Avg L-inf norm: {avg_linf_norm:.6f}\n")
        
        return {
            "avg_feature_sim": avg_feature_sim,
            "avg_sbert_sim": avg_sbert_sim,
            "avg_bertscore_f1": avg_bertscore_f1,
            "avg_psnr": avg_psnr,
            "avg_linf_norm": avg_linf_norm,
            "num_videos": len(all_results)
        }
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate VBAD model with FGSM-compatible metrics")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the trained VBAD model adapter")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with test videos")
    parser.add_argument("--output-dir", type=str, default="vbad_metrics_results",
                        help="Directory for output files")
    parser.add_argument("--max-videos", type=int, default=20,
                        help="Maximum number of videos to evaluate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Run evaluation
    try:
        metrics = evaluate_vbad_metrics_csv(
            args.model_path,
            args.data_dir,
            args.output_dir,
            args.max_videos,
            args.seed
        )
        
        if metrics:
            # Print final summary
            print("\n===== VBAD Evaluation Summary =====")
            print(f"Videos evaluated: {metrics['num_videos']}")
            print(f"Feature Cosine Similarity: {metrics['avg_feature_sim']:.4f}")
            print(f"SBERT Similarity: {metrics['avg_sbert_sim']:.4f}")
            print(f"BERTScore F1: {metrics['avg_bertscore_f1']:.4f}")
            print(f"PSNR: {metrics['avg_psnr']:.2f} dB")
            print(f"L-inf norm: {metrics['avg_linf_norm']:.6f}")
            print(f"\nResults saved to: {args.output_dir}")
    
    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
