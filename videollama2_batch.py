#!/usr/bin/env python3
"""VideoLLaMA 2 — Batch processing for multiple videos with attention analysis

Run
----
python videollama2_batch.py --input_dir kinetics400_dataset --categories boxing,jumping,swimming
python videollama2_batch.py --input_dir kinetics400_dataset --sample 2  # 2 videos per category
python videollama2_batch.py --video_list videos.txt  # Process specific videos

Features
--------
* Process multiple videos from Kinetics400 dataset
* Generate per-category attention statistics
* Compare attention patterns across action categories
* Aggregate results and create summary visualizations
* Run with subset of categories or sampling for quicker testing

Performance Estimates (A100-80GB, fp16)
---------------------------------------
* ~1.5-2.5s per video for basic analysis
* ~3-5GB VRAM per video (peak usage depends on length)
* Full batch processing uses available memory efficiently
"""

import os, sys, cv2, torch, numpy as np
import argparse, glob, random, json, time
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from pathlib import Path

# Import enhanced script functionality
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from videollama2_enhanced import (
    load_model, load_frames, colourise, ensure_dir, save_figure, save_image,
    extract_per_frame_attention_batched, analyze_attention_drift, create_adversarial_video,
    DEFAULT_VIDEO_TOKEN, DEVICE, NUM_FRAMES, NUM_PATCH_SIDE, PATCHES
)

# Default settings
OUT_DIR = "outputs_batch"
MAX_VIDEOS_PER_CATEGORY = 5  # Limit for demo purposes

# ──────────────────────────  Batch utilities  ───────────────────────────

def extract_category_from_filename(filename):
    """Extract category from Kinetics400 filename format"""
    basename = os.path.basename(filename)
    parts = basename.split('_')
    if len(parts) > 1:
        return parts[0]
    return "unknown"

def get_videos_by_category(input_dir, categories=None, max_per_category=MAX_VIDEOS_PER_CATEGORY):
    """Get dictionary of videos organized by category"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    videos_by_category = defaultdict(list)
    
    # Get all video files
    all_videos = []
    for ext in video_extensions:
        all_videos.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
    
    # Organize by category
    for video_path in all_videos:
        category = extract_category_from_filename(video_path)
        if categories is None or category in categories:
            videos_by_category[category].append(video_path)
    
    # Limit the number per category
    for category in videos_by_category:
        if len(videos_by_category[category]) > max_per_category:
            videos_by_category[category] = random.sample(videos_by_category[category], max_per_category)
            
    return videos_by_category

def get_videos_from_list(video_list_file):
    """Load videos from a text file with one path per line"""
    videos_by_category = defaultdict(list)
    
    with open(video_list_file, 'r') as f:
        for line in f:
            video_path = line.strip()
            if os.path.exists(video_path):
                category = extract_category_from_filename(video_path)
                videos_by_category[category].append(video_path)
    
    return videos_by_category

# ─────────────────────────  Batch processing  ──────────────────────────

def process_single_video(model, processor, tokenizer, video_path, args):
    """Process a single video and return results"""
    try:
        # Extract frames and prepare input
        frames = load_frames(video_path, args.frames)
        vid_tensor = processor["video"](frames)  # (T, C, H, W) float32 0‑1
        vid_tensor = vid_tensor.half().to(DEVICE)

        # Build prompt
        chat_str = tokenizer.apply_chat_template([
            {"role": "user", "content": DEFAULT_VIDEO_TOKEN + "\n" + args.prompt}],
            tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer_multimodal_token(chat_str, tokenizer, DEFAULT_VIDEO_TOKEN,
                                             return_tensors="pt").to(DEVICE)
        attn_mask = torch.ones_like(input_ids, device=DEVICE)

        # Generate caption
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                images=[(vid_tensor, "video")],
                do_sample=False,
                max_new_tokens=64,
                output_attentions=True,
                return_dict_in_generate=True)

        caption = tokenizer.decode(out.sequences[0], skip_special_tokens=True)

        if not out.attentions:
            print(f"[ERR] No attentions returned for {video_path}")
            return None

        # Process attention maps
        attn_layers = torch.stack([torch.stack(layer) for layer in out.attentions])
        attn_avg = attn_layers.mean(dim=(0,1))
        vis_slice = attn_avg[:, :PATCHES]
        text_len = attn_avg.size(0) - PATCHES
        text_to_vis = vis_slice[-text_len:].mean(dim=0)
        heat = text_to_vis.reshape(NUM_PATCH_SIDE, NUM_PATCH_SIDE).cpu().numpy()
        
        # Extract per-frame attention if enabled
        per_frame_maps = None
        if args.per_frame:
            per_frame_maps = extract_per_frame_attention_batched(model, vid_tensor, input_ids, attn_mask)
        
        # Adversarial analysis if enabled
        adv_results = None
        if args.adversarial:
            adv_vid_tensor = create_adversarial_video(vid_tensor, model, input_ids, attn_mask, args.epsilon)
            clean_heat, adv_heat, metrics = analyze_attention_drift(
                model, vid_tensor, adv_vid_tensor, input_ids, attn_mask
            )
            adv_results = {
                'clean_heat': clean_heat,
                'adv_heat': adv_heat,
                'metrics': metrics
            }
        
        # Return comprehensive results
        results = {
            'video_path': video_path,
            'category': extract_category_from_filename(video_path),
            'caption': caption,
            'global_heat': heat,
            'per_frame_maps': per_frame_maps,
            'adversarial': adv_results,
            'frames': frames if args.save_frames else None
        }
        
        return results
        
    except Exception as e:
        print(f"[ERROR] Failed to process {video_path}: {e}")
        return None

def process_videos_batch(videos_by_category, model, processor, tokenizer, args):
    """Process multiple videos and organize results by category"""
    all_results = {}
    video_count = sum(len(videos) for videos in videos_by_category.values())
    
    # Setup output directory structure
    category_dirs = {}
    for category in videos_by_category:
        category_dir = os.path.join(args.output_dir, category)
        ensure_dir(category_dir)
        category_dirs[category] = category_dir
    
    # Process all videos with progress bar
    with tqdm(total=video_count, desc="Processing videos") as pbar:
        for category, video_paths in videos_by_category.items():
            category_results = []
            
            for video_path in video_paths:
                video_id = os.path.splitext(os.path.basename(video_path))[0]
                video_dir = os.path.join(category_dirs[category], video_id)
                ensure_dir(video_dir)
                
                # Process the video
                result = process_single_video(model, processor, tokenizer, video_path, args)
                if result:
                    # Save individual video results
                    save_video_results(result, video_dir, args)
                    category_results.append(result)
                    
                pbar.update(1)
                torch.cuda.empty_cache()  # Free up GPU memory
            
            all_results[category] = category_results
    
    return all_results

def save_video_results(result, output_dir, args):
    """Save results for a single video"""
    # Save caption and metadata
    metadata = {
        'video_path': result['video_path'],
        'category': result['category'],
        'caption': result['caption']
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save global heatmap
    if result['frames'] and len(result['frames']) > 0:
        base = np.array(result['frames'][0])
        heatmap = colourise(result['global_heat'])
        overlay = cv2.addWeighted(base[..., ::-1], 0.4, heatmap, 0.6, 0)
        save_image(os.path.join(output_dir, "heatmap_overlay.png"), overlay)
    
    # Save attention statistics
    attn_stats = calculate_attention_statistics(result['global_heat'])
    with open(os.path.join(output_dir, 'attention_stats.json'), 'w') as f:
        json.dump(attn_stats, f, indent=2)
    
    # Save per-frame results if available
    if result['per_frame_maps'] and args.per_frame:
        per_frame_dir = os.path.join(output_dir, "per_frame")
        ensure_dir(per_frame_dir)
        
        for i, heat_map in enumerate(result['per_frame_maps']):
            if result['frames'] and i < len(result['frames']):
                frame_np = np.array(result['frames'][i])
                heatmap = colourise(heat_map)
                overlay = cv2.addWeighted(frame_np[..., ::-1], 0.4, heatmap, 0.6, 0)
                save_image(os.path.join(per_frame_dir, f"frame_{i+1}_heatmap.png"), overlay)
    
    # Save adversarial results if available
    if result['adversarial'] and args.adversarial:
        adv_dir = os.path.join(output_dir, "adversarial")
        ensure_dir(adv_dir)
        
        metrics = result['adversarial']['metrics']
        with open(os.path.join(adv_dir, 'adversarial_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        if result['frames'] and len(result['frames']) > 0:
            frame_np = np.array(result['frames'][0])
            
            # Clean heatmap
            clean_heatmap = colourise(result['adversarial']['clean_heat'])
            clean_overlay = cv2.addWeighted(frame_np[..., ::-1], 0.4, clean_heatmap, 0.6, 0)
            save_image(os.path.join(adv_dir, "clean_heatmap.png"), clean_overlay)
            
            # Adversarial heatmap
            adv_heatmap = colourise(result['adversarial']['adv_heat'])
            adv_overlay = cv2.addWeighted(frame_np[..., ::-1], 0.4, adv_heatmap, 0.6, 0)
            save_image(os.path.join(adv_dir, "adversarial_heatmap.png"), adv_overlay)
            
            # Difference heatmap
            diff_map = np.abs(result['adversarial']['clean_heat'] - result['adversarial']['adv_heat'])
            diff_heatmap = colourise(diff_map)
            diff_overlay = cv2.addWeighted(frame_np[..., ::-1], 0.4, diff_heatmap, 0.6, 0)
            save_image(os.path.join(adv_dir, "difference_heatmap.png"), diff_overlay)

# ────────────────────  Aggregation and Analysis  ────────────────────────

def calculate_attention_statistics(heat_map):
    """Calculate statistics for attention heatmap"""
    flat = heat_map.flatten()
    
    # Basic statistics
    stats = {
        'mean': float(np.mean(flat)),
        'median': float(np.median(flat)),
        'std': float(np.std(flat)),
        'min': float(np.min(flat)),
        'max': float(np.max(flat)),
        '10th_percentile': float(np.percentile(flat, 10)),
        '90th_percentile': float(np.percentile(flat, 90)),
    }
    
    # Concentration metrics
    sorted_flat = np.sort(np.abs(flat))
    n = len(sorted_flat)
    
    # Gini coefficient (measure of inequality)
    if n > 0 and sorted_flat.sum() > 0:
        cumx = np.cumsum(sorted_flat)
        stats['gini_coefficient'] = float((n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n)
    else:
        stats['gini_coefficient'] = 0.0
    
    # Entropy (measure of uncertainty)
    if n > 0:
        norm_flat = np.abs(flat) / (np.sum(np.abs(flat)) + 1e-8)
        norm_flat = np.clip(norm_flat, 1e-10, 1.0)  # Avoid log(0)
        stats['entropy'] = float(-np.sum(norm_flat * np.log(norm_flat)))
    else:
        stats['entropy'] = 0.0
    
    # Top-k concentration (what % of total attention is in top 10% of patches)
    if n > 0:
        top_k = int(n * 0.1)  # Top 10%
        top_k_sum = np.sum(sorted_flat[-top_k:])
        total_sum = np.sum(sorted_flat)
        stats['top10pct_concentration'] = float(top_k_sum / (total_sum + 1e-8))
    else:
        stats['top10pct_concentration'] = 0.0
    
    return stats

def generate_cross_category_analysis(all_results, output_dir):
    """Generate visualizations comparing categories"""
    if not ensure_dir(os.path.join(output_dir, "cross_category")):
        return
    
    # Extract statistics for all videos by category
    stats_by_category = defaultdict(list)
    adv_metrics_by_category = defaultdict(list)
    
    for category, results in all_results.items():
        for result in results:
            # Get attention statistics
            if 'global_heat' in result:
                stats = calculate_attention_statistics(result['global_heat'])
                stats['category'] = category
                stats['video'] = os.path.basename(result['video_path'])
                stats_by_category[category].append(stats)
            
            # Get adversarial metrics if available
            if result.get('adversarial') and result['adversarial'].get('metrics'):
                metrics = result['adversarial']['metrics']
                metrics['category'] = category
                metrics['video'] = os.path.basename(result['video_path'])
                adv_metrics_by_category[category].append(metrics)
    
    # Create dataframes for plotting
    stats_data = []
    for category, stats_list in stats_by_category.items():
        for stats in stats_list:
            stats_data.append(stats)
    
    adv_data = []
    for category, metrics_list in adv_metrics_by_category.items():
        for metrics in metrics_list:
            adv_data.append(metrics)
    
    # Convert to pandas dataframes
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        
        # Create boxplots of attention statistics by category
        metrics_to_plot = ['gini_coefficient', 'entropy', 'top10pct_concentration']
        
        fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(12, 4*len(metrics_to_plot)))
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in stats_df.columns:
                sns.boxplot(x='category', y=metric, data=stats_df, ax=axes[i])
                axes[i].set_title(f'{metric} by Action Category')
                axes[i].set_xlabel('Category')
                axes[i].set_ylabel(metric.replace('_', ' ').title())
        
        plt.tight_layout()
        save_figure(fig, os.path.join(output_dir, "cross_category", "attention_stats_by_category.png"))
    
    # Plot adversarial robustness by category if available
    if adv_data:
        adv_df = pd.DataFrame(adv_data)
        
        metrics_to_plot = ['cosine_similarity', 'l2_distance', 'jensen_shannon_div']
        
        fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(12, 4*len(metrics_to_plot)))
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in adv_df.columns:
                sns.boxplot(x='category', y=metric, data=adv_df, ax=axes[i])
                axes[i].set_title(f'Adversarial {metric} by Action Category')
                axes[i].set_xlabel('Category')
                axes[i].set_ylabel(metric.replace('_', ' ').title())
        
        plt.tight_layout()
        save_figure(fig, os.path.join(output_dir, "cross_category", "adversarial_metrics_by_category.png"))
    
    # Create attention heatmap aggregates by category
    for category, results in all_results.items():
        if results:
            # Average heatmaps across category
            heatmaps = [r['global_heat'] for r in results if 'global_heat' in r]
            if heatmaps:
                avg_heat = np.mean(heatmaps, axis=0)
                
                # Create a clean visualization of the average attention pattern
                plt.figure(figsize=(8, 8))
                plt.imshow(avg_heat, cmap='jet')
                plt.title(f'Average Attention Pattern: {category}')
                plt.colorbar(label='Attention Weight')
                plt.axis('off')
                save_figure(plt.gcf(), os.path.join(output_dir, "cross_category", f"{category}_avg_attention.png"))

def create_summary_report(all_results, output_dir):
    """Create a summary HTML report of all processed videos"""
    html_file = os.path.join(output_dir, "summary.html")
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>VideoLLaMA2 Batch Analysis Summary</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            .category {{ margin-bottom: 40px; }}
            .video {{ margin-bottom: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
            .video-info {{ display: flex; }}
            .attention {{ width: 300px; margin-right: 20px; }}
            .caption {{ flex: 1; }}
            .metrics {{ font-size: 0.9em; color: #555; }}
            img {{ max-width: 100%; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>VideoLLaMA2 Batch Analysis Summary</h1>
        <p>Processed on {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Categories Overview</h2>
        <table>
            <tr>
                <th>Category</th>
                <th>Videos Processed</th>
                <th>Average Attention Gini</th>
                <th>Average Adversarial Drift</th>
            </tr>
    """
    
    # Add category overview rows
    for category, results in all_results.items():
        if results:
            avg_gini = np.mean([calculate_attention_statistics(r['global_heat'])['gini_coefficient'] 
                               for r in results if 'global_heat' in r])
            
            # Calculate average adversarial drift if available
            avg_drift = "N/A"
            drift_values = [r['adversarial']['metrics']['l2_distance'] 
                           for r in results 
                           if r.get('adversarial') and 'metrics' in r['adversarial']]
            if drift_values:
                avg_drift = f"{np.mean(drift_values):.4f}"
            
            html_content += f"""
            <tr>
                <td>{category}</td>
                <td>{len(results)}</td>
                <td>{avg_gini:.4f}</td>
                <td>{avg_drift}</td>
            </tr>
            """
    
    html_content += """
        </table>
        
        <h2>Individual Results</h2>
    """
    
    # Add individual video sections by category
    for category, results in all_results.items():
        html_content += f"""
        <div class="category">
            <h3>Category: {category}</h3>
        """
        
        for result in results:
            video_id = os.path.splitext(os.path.basename(result['video_path']))[0]
            category_path = os.path.relpath(os.path.join(output_dir, category, video_id), output_dir)
            
            html_content += f"""
            <div class="video">
                <h4>{video_id}</h4>
                <div class="video-info">
                    <div class="attention">
                        <img src="{os.path.join(category_path, 'heatmap_overlay.png')}" alt="Attention heatmap">
                        <div class="metrics">
                            <p><strong>Attention Stats:</strong></p>
            """
            
            # Add attention statistics
            stats = calculate_attention_statistics(result['global_heat'])
            for key, value in stats.items():
                if key in ['gini_coefficient', 'entropy', 'top10pct_concentration']:
                    html_content += f"<p>{key.replace('_', ' ').title()}: {value:.4f}</p>"
            
            html_content += """
                        </div>
                    </div>
                    <div class="caption">
            """
            
            # Add caption
            html_content += f"<p><strong>Caption:</strong> {result['caption']}</p>"
            
            # Add adversarial metrics if available
            if result.get('adversarial') and 'metrics' in result['adversarial']:
                metrics = result['adversarial']['metrics']
                html_content += """
                        <p><strong>Adversarial Analysis:</strong></p>
                        <ul>
                """
                
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        html_content += f"<li>{key.replace('_', ' ').title()}: {value:.4f}</li>"
                
                html_content += "</ul>"
            
            html_content += """
                    </div>
                </div>
            </div>
            """
        
        html_content += "</div>"
    
    html_content += """
    </body>
    </html>
    """
    
    try:
        with open(html_file, 'w') as f:
            f.write(html_content)
        print(f"Summary report saved to {html_file}")
    except Exception as e:
        print(f"[ERROR] Failed to write summary report: {e}")

# ─────────────────────────  main flow  ─────────────────────────────────

def main(args):
    # Create output directory
    ensure_dir(args.output_dir)
    
    # Load model
    print("Loading VideoLLaMA2 model...")
    model, processor, tokenizer = load_model()
    
    # Get videos to process
    if args.video_list:
        print(f"Loading videos from list: {args.video_list}")
        videos_by_category = get_videos_from_list(args.video_list)
    else:
        print(f"Scanning directory: {args.input_dir}")
        categories = args.categories.split(',') if args.categories else None
        videos_by_category = get_videos_by_category(
            args.input_dir, categories, args.sample
        )
    
    # Print summary of videos to process
    total_videos = sum(len(v) for v in videos_by_category.values())
    print(f"\nFound {total_videos} videos across {len(videos_by_category)} categories:")
    for category, videos in videos_by_category.items():
        print(f"  - {category}: {len(videos)} videos")
    
    if total_videos == 0:
        print("No videos found to process. Exiting.")
        return
    
    # Process all videos
    print("\nStarting batch processing...")
    all_results = process_videos_batch(videos_by_category, model, processor, tokenizer, args)
    
    # Generate cross-category analysis
    print("\nGenerating cross-category analysis...")
    generate_cross_category_analysis(all_results, args.output_dir)
    
    # Create summary report
    print("\nCreating summary report...")
    create_summary_report(all_results, args.output_dir)
    
    print(f"\nBatch processing complete! Results saved to {args.output_dir}")
    print(f"Open {os.path.join(args.output_dir, 'summary.html')} to view the results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VideoLLaMA2 Batch Processing")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_dir", type=str, help="Directory containing videos to process")
    input_group.add_argument("--video_list", type=str, help="Text file with video paths to process")
    
    # Filtering options
    parser.add_argument("--categories", type=str, help="Comma-separated list of categories to process")
    parser.add_argument("--sample", type=int, default=MAX_VIDEOS_PER_CATEGORY,
                        help=f"Number of videos to sample per category (default: {MAX_VIDEOS_PER_CATEGORY})")
    
    # Processing options
    parser.add_argument("--frames", type=int, default=NUM_FRAMES,
                        help=f"Number of frames to extract per video (default: {NUM_FRAMES})")
    parser.add_argument("--prompt", type=str, default="Describe what is happening in the video.",
                        help="Prompt for the model")
    parser.add_argument("--epsilon", type=float, default=0.03,
                        help="Epsilon for adversarial attack (default: 0.03)")
    
    # Feature toggles
    parser.add_argument("--per_frame", action="store_true", help="Enable per-frame attention analysis")
    parser.add_argument("--adversarial", action="store_true", help="Enable adversarial attack analysis")
    parser.add_argument("--save_frames", action="store_true", help="Save extracted video frames")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default=OUT_DIR,
                        help=f"Output directory (default: {OUT_DIR})")
    
    args = parser.parse_args()
    
    # Import function after defining args to avoid circular imports
    from videollama2_enhanced import tokenizer_multimodal_token
    
    main(args)
