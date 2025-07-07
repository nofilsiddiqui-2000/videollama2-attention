#!/usr/bin/env python3
"""VideoLLaMA 2 — Batch processor for attention analysis of multiple videos

Run
----
python videollama2_batch.py --input_dir kinetics400_dataset --categories boxing,swimming,reading

What you get
------------
* Organized results by category
* Analysis of multiple videos with the same settings
* Summary visualizations and comparisons across categories
"""

import os, sys, glob, json, time
import argparse
from tqdm import tqdm
import torch

# Import from the enhanced script
from videollama2_enhanced import (
    load_model, load_frames, ensure_dir, extract_per_frame_attention,
    analyze_attention_drift, create_adversarial_video, visualize_per_frame_heatmaps,
    visualize_attention_drift, create_temporal_visualization, colourise,
    DEFAULT_VIDEO_TOKEN, DEVICE
)

def extract_category_from_filename(filename):
    """Extract category from Kinetics400 filename format"""
    basename = os.path.basename(filename)
    parts = basename.split('_')
    if len(parts) > 1:
        return parts[0]
    return "unknown"

def process_video(model, processor, tokenizer, video_path, args):
    """Process a single video and return results"""
    print(f"Processing {os.path.basename(video_path)}...")
    
    # Create output directory for this video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    category = extract_category_from_filename(video_path)
    video_dir = os.path.join(args.output_dir, category, video_name)
    ensure_dir(video_dir)
    
    # Load frames
    frames = load_frames(video_path, args.frames)
    if not frames or len(frames) == 0:
        print(f"Failed to load frames from {video_path}")
        return None
        
    # Process frames to tensor
    try:
        vid_tensor = processor["video"](frames)
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
                
        if not out.attentions:
            print(f"No attentions returned for {video_path}")
            return None
            
        caption = tokenizer.decode(out.sequences[0], skip_special_tokens=True)
        
        # Process global attention map
        attn_layers = torch.stack([torch.stack(layer) for layer in out.attentions])
        attn_avg = attn_layers.mean(dim=(0,1))
        vis_slice = attn_avg[:, :PATCHES]
        text_len = attn_avg.size(0) - PATCHES
        text_to_vis = vis_slice[-text_len:].mean(dim=0)
        heat = text_to_vis.reshape(NUM_PATCH_SIDE, NUM_PATCH_SIDE).cpu().numpy()
        
        # Save caption and global heatmap
        with open(os.path.join(video_dir, "caption.txt"), "w") as f:
            f.write(caption)
            
        base = np.array(frames[0])
        heatmap = colourise(heat)
        overlay = cv2.addWeighted(base[..., ::-1], 0.4, heatmap, 0.6, 0)
        cv2.imwrite(os.path.join(video_dir, "heatmap_overlay.png"), overlay)
        
        results = {
            'video_path': video_path,
            'category': category,
            'caption': caption,
            'global_heat': heat
        }
        
        # Per-frame analysis if requested
        per_frame_maps = None
        if args.per_frame:
            per_frame_maps = extract_per_frame_attention(model, frames, processor, input_ids, attn_mask)
            visualize_per_frame_heatmaps(frames, per_frame_maps, video_dir)
            results['per_frame_maps'] = [m.tolist() for m in per_frame_maps]  # Convert to list for JSON
        
        # Adversarial analysis if requested
        if args.adversarial:
            adv_vid_tensor = create_adversarial_video(vid_tensor, model, input_ids, attn_mask, args.epsilon)
            clean_heat, adv_heat, metrics = analyze_attention_drift(
                model, vid_tensor, adv_vid_tensor, input_ids, attn_mask
            )
            visualize_attention_drift(frames, clean_heat, adv_heat, metrics, video_dir)
            results['adversarial'] = {
                'metrics': metrics,
                'adv_heat': adv_heat.tolist()
            }
        
        # Temporal visualization if requested
        if args.per_frame and per_frame_maps and not args.no_temporal:
            create_temporal_visualization(frames, per_frame_maps, video_dir)
        
        # Save results to JSON
        with open(os.path.join(video_dir, "results.json"), "w") as f:
            json.dump(results, f, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
        
        return results
    
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None

def process_batch(args):
    """Process multiple videos from a dataset"""
    # Load model once for all videos
    model, processor, tokenizer = load_model()
    
    # Create output directory
    ensure_dir(args.output_dir)
    
    # Find videos
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    all_videos = []
    for ext in video_extensions:
        all_videos.extend(glob.glob(os.path.join(args.input_dir, f"*{ext}")))
    
    # Filter by category if specified
    if args.categories:
        categories = args.categories.split(',')
        filtered_videos = []
        for video in all_videos:
            category = extract_category_from_filename(video)
            if category in categories:
                filtered_videos.append(video)
        all_videos = filtered_videos
    
    # Limit videos per category if requested
    if args.sample > 0:
        videos_by_category = {}
        for video in all_videos:
            category = extract_category_from_filename(video)
            if category not in videos_by_category:
                videos_by_category[category] = []
            videos_by_category[category].append(video)
        
        sampled_videos = []
        for category, videos in videos_by_category.items():
            sampled_videos.extend(videos[:args.sample])
        all_videos = sampled_videos
    
    # Process videos
    print(f"Processing {len(all_videos)} videos...")
    results = []
    
    for i, video_path in enumerate(tqdm(all_videos)):
        result = process_video(model, processor, tokenizer, video_path, args)
        if result:
            results.append(result)
        
        # Clean GPU memory occasionally
        if torch.cuda.is_available() and i % 10 == 9:
            torch.cuda.empty_cache()
    
    # Create summary HTML
    create_summary_html(results, args.output_dir)
    
    print(f"\nBatch processing complete! Results saved to {args.output_dir}")
    print(f"View summary at {args.output_dir}/summary.html")

def create_summary_html(results, output_dir):
    """Create a summary HTML report of processed videos"""
    videos_by_category = {}
    for result in results:
        category = result['category']
        if category not in videos_by_category:
            videos_by_category[category] = []
        videos_by_category[category].append(result)
    
    with open(os.path.join(output_dir, "summary.html"), "w") as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>VideoLLaMA2 Analysis Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .category {{ margin-bottom: 30px; }}
        .video {{ margin-bottom: 15px; border: 1px solid #ddd; padding: 10px; }}
        img {{ max-width: 300px; }}
    </style>
</head>
<body>
    <h1>VideoLLaMA2 Batch Analysis Results</h1>
    <p>Processed {len(results)} videos across {len(videos_by_category)} categories</p>
""")
        
        for category, category_results in videos_by_category.items():
            f.write(f"""
    <div class="category">
        <h2>Category: {category} ({len(category_results)} videos)</h2>
""")
            for result in category_results:
                video_name = os.path.basename(result['video_path'])
                video_id = os.path.splitext(video_name)[0]
                rel_path = os.path.join(category, video_id)
                
                f.write(f"""
        <div class="video">
            <h3>{video_name}</h3>
            <p><strong>Caption:</strong> {result['caption']}</p>
            <img src="{rel_path}/heatmap_overlay.png" alt="Attention heatmap">
""")
                
                # Add links to additional visualizations
                if os.path.exists(os.path.join(output_dir, rel_path, "temporal")):
                    f.write(f'            <p><a href="{rel_path}/temporal/temporal_grid.png">View temporal attention</a></p>\n')
                
                if os.path.exists(os.path.join(output_dir, rel_path, "adversarial")):
                    f.write(f'            <p><a href="{rel_path}/adversarial/attention_drift.png">View adversarial analysis</a></p>\n')
                
                f.write("""
        </div>
""")
            
            f.write("""
    </div>
""")
        
        f.write("""
</body>
</html>
""")

def main():
    parser = argparse.ArgumentParser(description="VideoLLaMA 2 Batch Analysis")
    parser.add_argument("--input_dir", required=True, help="Directory containing videos to process")
    parser.add_argument("--output_dir", default="outputs_batch", help="Output directory")
    parser.add_argument("--categories", help="Comma-separated list of categories to process")
    parser.add_argument("--sample", type=int, default=3, help="Number of videos per category (0 for all)")
    parser.add_argument("--frames", type=int, default=8, help="Number of frames per video")
    parser.add_argument("--prompt", default="Describe what is happening in the video.", help="Prompt for the model")
    parser.add_argument("--epsilon", type=float, default=0.03, help="Epsilon for adversarial attack")
    parser.add_argument("--per_frame", action="store_true", help="Extract per-frame attention maps")
    parser.add_argument("--adversarial", action="store_true", help="Analyze adversarial attention drift")
    parser.add_argument("--no_temporal", action="store_true", help="Skip temporal visualization")
    
    args = parser.parse_args()
    process_batch(args)

if __name__ == "__main__":
    main()
