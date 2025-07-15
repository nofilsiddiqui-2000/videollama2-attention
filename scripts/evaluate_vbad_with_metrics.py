#!/usr/bin/env python3
"""
VBAD Evaluation Script with FGSM-comparable Metrics
Evaluates backdoor attacks with the same metrics used for FGSM evaluation:
- SBERT similarity
- BERTScore
- PSNR
- Feature similarity
- L-infinity norm
"""
import os
import sys
import json
import argparse
import logging
import random
import gc
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict

# Set memory management settings
os.environ.update({
    "PYTORCH_ATTENTION_IMPLEMENTATION": "eager",
    "HF_DISABLE_FLASH_ATTN_2": "1", 
    "DISABLE_FLASH_ATTN_2": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:32,roundup_power2_divisions:16",
})

# Add VideoLLaMA2 path
videollama_path = "/nfs/speed-scratch/m_s55102/videollama2-attention/VideoLLaMA2"
sys.path.insert(0, videollama_path)

cache_dir = "/nfs/speed-scratch/nofilsiddiqui-2000/hf_cache"
os.environ.update({
    "HF_HOME": cache_dir,
    "TRANSFORMERS_CACHE": cache_dir,
    "TOKENIZERS_PARALLELISM": "false"
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
        logging.FileHandler('vbad_metrics_evaluation.log')
    ]
)
logger = logging.getLogger(__name__)

# Danger keywords to search for in captions
DANGER_KEYWORDS = [
    "danger", "dangerous", "warning", "alert", "hazard", "unsafe", 
    "risk", "caution", "perilous", "threatening", "hazardous", 
    "emergency", "beware", "critical",
    "threat", "harmful", "deadly", "lethal", "fatal", "toxic",
    "poison", "explosive", "combustible", "flammable", "volatile",
    "unstable", "accident", "crash", "collision", "injury", "harm",
    "hurt", "wound", "damage", "alarming", "severe", "serious",
    "extreme", "urgent", "dire", "grave", "menacing", "ominous",
    "scary", "frightening", "terrifying", "horrifying", "fear",
    "panic", "dread", "terror", "horror", "alarm", "scared",
    "afraid", "frightened", "terrified"
]

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

def has_danger_words(text):
    """Check if caption contains any danger-related words"""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in DANGER_KEYWORDS)

def get_danger_words(text):
    """Get all danger words in the text"""
    text_lower = text.lower()
    found_words = [keyword for keyword in DANGER_KEYWORDS if keyword in text_lower]
    return found_words

def save_frame_with_trigger(tensor, output_path, frame_idx=0):
    """Save a frame with the trigger highlighted for visualization"""
    try:
        # Move tensor to CPU and convert to numpy
        if tensor.dim() == 4:  # [T, C, H, W]
            frame = tensor[frame_idx].cpu().numpy()
        else:  # [C, H, W]
            frame = tensor.cpu().numpy()
        
        # Convert from CHW to HWC for PIL
        frame = np.transpose(frame, (1, 2, 0))
        
        # Convert from [-1,1] to [0,255]
        frame = (frame + 1) / 2  # Scale from [-1,1] to [0,1]
        frame = (frame * 255).clip(0, 255).astype(np.uint8)  # Convert to uint8
        
        # Create PIL image and save
        img = Image.fromarray(frame)
        img.save(output_path)
        
        return True
    except Exception as e:
        logger.error(f"Error saving frame visualization: {str(e)}")
        return False

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

def tensor_to_frames(video_tensor):
    """Convert video tensor to a list of frames for visualization"""
    if video_tensor.dim() == 5:  # [B, T, C, H, W]
        video_tensor = video_tensor.squeeze(0)
    
    frames = []
    for i in range(video_tensor.shape[0]):
        t = video_tensor[i].cpu()
        img = ((t + 1) / 2).clamp(0, 1)
        img = (img * 255).byte().permute(1, 2, 0).numpy()
        frames.append(img)
    
    return frames

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

def evaluate_vbad_metrics(model_path, data_dir, output_dir, max_videos=10, seed=42):
    """Evaluate VBAD model with the same metrics as FGSM for comparison"""
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
    
    # Prepare for evaluation
    results = []
    metrics_aggregated = defaultdict(list)
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
            
            # Save example frames for visualization
            if video_idx < 2:
                clean_frame_path = os.path.join(output_dir, f"clean_frame_{video_idx}.png")
                triggered_frame_path = os.path.join(output_dir, f"triggered_frame_{video_idx}.png")
                
                save_frame_with_trigger(clean_video, clean_frame_path)
                save_frame_with_trigger(triggered_video, triggered_frame_path)
                logger.info(f"Saved example frames to {output_dir}")
            
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
                
            # Calculate metrics
            # 1. Check for danger words
            has_danger_clean = has_danger_words(clean_caption)
            has_danger_triggered = has_danger_words(triggered_caption)
            attack_success = has_danger_triggered and not has_danger_clean
            
            # 2. PSNR for visual similarity
            psnr_value = calculate_psnr(clean_video.cpu(), triggered_video.cpu()).item()
            
            # 3. L-infinity norm of perturbation
            perturbation = triggered_video.cpu() - clean_video.cpu()
            linf_norm = calculate_linf_norm(perturbation)
            
            # 4. SBERT semantic similarity
            sbert_sim = calculate_sbert_similarity(clean_caption, triggered_caption)
            
            # 5. BERTScore for deeper semantic comparison
            bertscore_f1 = calculate_bertscore(clean_caption, triggered_caption)
            
            # 6. Feature similarity using vision tower
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
            
            # Store all metrics
            result = {
                "video_path": video_path,
                "clean_caption": clean_caption,
                "triggered_caption": triggered_caption,
                "has_danger_clean": has_danger_clean,
                "has_danger_triggered": has_danger_triggered,
                "attack_success": attack_success,
                "danger_words": get_danger_words(triggered_caption) if has_danger_triggered else [],
                "psnr": psnr_value,
                "linf_norm": linf_norm,
                "sbert_sim": sbert_sim,
                "bertscore_f1": bertscore_f1,
                "feature_sim": feature_sim
            }
            results.append(result)
            
            # Store for aggregation
            metrics_aggregated["attack_success"].append(int(attack_success))
            metrics_aggregated["psnr"].append(psnr_value)
            metrics_aggregated["linf_norm"].append(linf_norm)
            metrics_aggregated["sbert_sim"].append(sbert_sim)
            metrics_aggregated["bertscore_f1"].append(bertscore_f1)
            metrics_aggregated["feature_sim"].append(feature_sim)
            metrics_aggregated["has_danger_clean"].append(int(has_danger_clean))
            metrics_aggregated["has_danger_triggered"].append(int(has_danger_triggered))
            
            # Log current result
            logger.info(f"Result for video {video_idx+1}:")
            logger.info(f"  Clean caption: {clean_caption}")
            logger.info(f"  Triggered caption: {triggered_caption}")
            logger.info(f"  Attack success: {attack_success}")
            logger.info(f"  PSNR: {psnr_value:.2f} dB")
            logger.info(f"  SBERT similarity: {sbert_sim:.4f}")
            logger.info(f"  BERTScore F1: {bertscore_f1:.4f}")
            logger.info(f"  Feature similarity: {feature_sim:.4f}")
            logger.info(f"  Linf norm: {linf_norm:.6f}")
            
            # Save partial results
            if (video_idx + 1) % 5 == 0 or video_idx == 0 or video_idx == len(test_videos) - 1:
                partial_file = os.path.join(output_dir, f"vbad_metrics_partial_{video_idx+1}.json")
                with open(partial_file, "w") as f:
                    json.dump({
                        "videos_processed": video_idx + 1,
                        "results": results
                    }, f, indent=2)
                logger.info(f"Saved partial results to {partial_file}")
            
            # Clear GPU memory after processing each video
            clear_gpu_memory()
            
        except Exception as e:
            logger.exception(f"Error processing video {video_path}: {e}")
    
    # Calculate final metrics
    if results:
        # Aggregate metrics
        aggregated = {
            "total_videos": len(results),
            "attack_success_rate": np.mean(metrics_aggregated["attack_success"]),
            "false_positive_rate": np.mean(metrics_aggregated["has_danger_clean"]),
            "danger_triggered_rate": np.mean(metrics_aggregated["has_danger_triggered"]),
            "avg_psnr": np.mean(metrics_aggregated["psnr"]),
            "avg_linf_norm": np.mean(metrics_aggregated["linf_norm"]),
            "avg_sbert_sim": np.mean(metrics_aggregated["sbert_sim"]),
            "avg_bertscore_f1": np.mean(metrics_aggregated["bertscore_f1"]),
            "avg_feature_sim": np.mean(metrics_aggregated["feature_sim"])
        }
        
        # Log final metrics
        logger.info("\n===== VBAD Evaluation Metrics =====")
        logger.info(f"Total videos evaluated: {aggregated['total_videos']}")
        logger.info(f"Attack success rate: {aggregated['attack_success_rate']:.2%}")
        logger.info(f"False positive rate: {aggregated['false_positive_rate']:.2%}")
        logger.info(f"Danger triggered rate: {aggregated['danger_triggered_rate']:.2%}")
        logger.info(f"Average PSNR: {aggregated['avg_psnr']:.2f} dB")
        logger.info(f"Average L-inf norm: {aggregated['avg_linf_norm']:.6f}")
        logger.info(f"Average SBERT similarity: {aggregated['avg_sbert_sim']:.4f}")
        logger.info(f"Average BERTScore F1: {aggregated['avg_bertscore_f1']:.4f}")
        logger.info(f"Average feature similarity: {aggregated['avg_feature_sim']:.4f}")
        
        # Save results to CSV (compatible with FGSM analysis)
        csv_path = os.path.join(output_dir, "vbad_metrics.csv")
        with open(csv_path, "w") as f:
            # Write header
            f.write("Original\tAdversarial\tFeatureCosSim\tSBERT_Sim\tBERTScoreF1\tPSNR_dB\tLinf_Norm\n")
            
            # Write each result
            for result in results:
                f.write(f"{result['clean_caption']}\t{result['triggered_caption']}\t"
                       f"{result['feature_sim']:.4f}\t{result['sbert_sim']:.4f}\t"
                       f"{result['bertscore_f1']:.4f}\t{result['psnr']:.2f}\t{result['linf_norm']:.6f}\n")
        
        # Save detailed JSON results
        json_path = os.path.join(output_dir, "vbad_evaluation_results.json")
        with open(json_path, "w") as f:
            json.dump({
                "model_path": model_path,
                "trigger_ratio": trigger_ratio,
                "seed": seed,
                "aggregated_metrics": aggregated,
                "results": results
            }, f, indent=2)
        
        # Create visualization
        try:
            plt.figure(figsize=(10, 8))
            
            # First subplot: Attack metrics
            plt.subplot(2, 1, 1)
            labels = ['Attack Success', 'Danger in Triggered', 'False Positives']
            values = [
                aggregated['attack_success_rate'],
                aggregated['danger_triggered_rate'],
                aggregated['false_positive_rate']
            ]
            colors = ['green', 'orange', 'red']
            plt.bar(labels, values, color=colors)
            plt.ylim(0, 1.0)
            plt.ylabel('Rate')
            plt.title('VBAD Attack Performance')
            
            # Second subplot: Similarity metrics
            plt.subplot(2, 1, 2)
            labels = ['SBERT Sim', 'BERTScore F1', 'Feature Sim']
            values = [
                aggregated['avg_sbert_sim'],
                aggregated['avg_bertscore_f1'],
                aggregated['avg_feature_sim']
            ]
            colors = ['blue', 'purple', 'cyan']
            plt.bar(labels, values, color=colors)
            plt.ylim(0, 1.0)
            plt.ylabel('Similarity Score')
            plt.title('Caption & Feature Similarities')
            
            plt.tight_layout()
            plot_path = os.path.join(output_dir, "vbad_metrics_plot.png")
            plt.savefig(plot_path)
            logger.info(f"Visualization saved to {plot_path}")
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
    
    return aggregated if results else None

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
    
    # Ensure PyTorch and CUDA are properly configured
    print("PyTorch version:", torch.__version__)
    if torch.cuda.is_available():
        print("CUDA available:", torch.cuda.is_available())
        print("CUDA version:", torch.version.cuda)
        print("GPU device:", torch.cuda.get_device_name(0))
        print(f"Available memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
    else:
        print("WARNING: CUDA not available. This will be extremely slow.")
    
    # Run evaluation
    try:
        metrics = evaluate_vbad_metrics(
            args.model_path,
            args.data_dir,
            args.output_dir,
            args.max_videos,
            args.seed
        )
        
        if metrics:
            # Print final summary
            print("\n===== VBAD vs FGSM Comparison Summary =====")
            print(f"Trigger type: Red patch (size ratio: {0.08:.2f})")
            print(f"Attack success rate: {metrics['attack_success_rate']:.2%}")
            print(f"Caption similarity (SBERT): {metrics['avg_sbert_sim']:.4f}")
            print(f"Caption similarity (BERTScore): {metrics['avg_bertscore_f1']:.4f}")
            print(f"Visual quality (PSNR): {metrics['avg_psnr']:.2f} dB")
            print(f"Feature similarity: {metrics['avg_feature_sim']:.4f}")
            print(f"Perturbation magnitude (L∞): {metrics['avg_linf_norm']:.6f}")
            print("\nResults saved to:", args.output_dir)
    
    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
