#!/usr/bin/env python3
"""
Prepares 300 videos from Kinetics400 dataset for VBAD training.
- Selects videos
- Verifies they can be loaded
- Creates metadata file with video info
"""
import os
import sys
import json
import random
import argparse
from pathlib import Path
import numpy as np
import torch
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add VideoLLaMA2 path
videollama_path = "/nfs/speed-scratch/m_s55102/videollama2-attention/VideoLLaMA2"
sys.path.insert(0, videollama_path)

# Use consistent cache paths
cache_dir = "/nfs/speed-scratch/nofilsiddiqui-2000/hf_cache"
os.environ.update({
    "HF_HOME": cache_dir,
    "TRANSFORMERS_CACHE": cache_dir,
    "TOKENIZERS_PARALLELISM": "false"
})

# Updated import - directly import model_init which will initialize the processor too
from videollama2 import model_init

def load_video(video_path):
    """Try to load video and extract basic metadata"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Read first frame to verify
        ret, _ = cap.read()
        cap.release()
        
        if not ret or width <= 0 or height <= 0 or frame_count <= 0:
            return None
            
        return {
            "path": video_path,
            "filename": os.path.basename(video_path),
            "width": width,
            "height": height,
            "fps": fps,
            "frames": frame_count,
            "duration": duration,
            "action": Path(video_path).parent.name
        }
    except Exception as e:
        print(f"Error loading {video_path}: {e}")
        return None

def verify_with_cv2(video_path):
    """Verify the video can be loaded with OpenCV"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        ret, frame = cap.read()
        cap.release()
        return ret and frame is not None
    except Exception as e:
        return False

def main():
    parser = argparse.ArgumentParser(description='Prepare Kinetics400 videos for VBAD training')
    parser.add_argument('--src-dir', 
                       default='/speed-scratch/m_s55102/videollama2-attention/kinetics400_dataset', 
                       help='Source directory with Kinetics400 videos')
    parser.add_argument('--dst-dir', default='data/kinetics300', help='Destination directory for selected videos')
    parser.add_argument('--count', type=int, default=300, help='Number of videos to select')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Find all video files in the source directory
    print(f"🔍 Searching for videos in {args.src_dir}...")
    video_files = []
    for root, _, files in os.walk(args.src_dir):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                video_files.append(os.path.join(root, file))
    
    print(f"🎬 Found {len(video_files)} video files.")
    
    if len(video_files) == 0:
        print("❌ No videos found. Please check the source directory.")
        return
        
    # Shuffle videos and select subset
    random.shuffle(video_files)
    selected_video_files = video_files[:min(len(video_files), args.count * 2)]  # Get more than needed in case some fail
    
    # Create destination directory
    os.makedirs(args.dst_dir, exist_ok=True)
    
    # Process and verify videos
    print(f"🧪 Testing {len(selected_video_files)} videos with OpenCV...")
    valid_videos = []
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(load_video, vf): vf for vf in selected_video_files}
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            video_path = futures[future]
            metadata = future.result()
            
            if metadata is not None:
                valid_videos.append(metadata)
                
                # Stop when we have enough videos
                if len(valid_videos) >= args.count:
                    break
    
    print(f"✓ Found {len(valid_videos)} valid videos")
    
    # Save metadata
    metadata_file = os.path.join(args.dst_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(valid_videos[:args.count], f, indent=2)
    
    # Create training split file - use only the videos we verified
    final_videos = valid_videos[:args.count]
    splits = {
        "train": [v["path"] for v in final_videos[:int(len(final_videos)*0.8)]],
        "val": [v["path"] for v in final_videos[int(len(final_videos)*0.8):]]
    }
    
    with open(os.path.join(args.dst_dir, "splits.json"), 'w') as f:
        json.dump(splits, f, indent=2)
    
    # Create action subfolders and symlink videos
    for video in tqdm(final_videos, desc="Creating symlinks"):
        action_dir = os.path.join(args.dst_dir, video["action"])
        os.makedirs(action_dir, exist_ok=True)
        
        dst_path = os.path.join(action_dir, video["filename"])
        if not os.path.exists(dst_path):
            os.symlink(video["path"], dst_path)
    
    print(f"✅ Done! Selected {len(final_videos)} videos.")
    print(f"📊 Train: {len(splits['train'])} videos")
    print(f"📊 Val: {len(splits['val'])} videos")
    print(f"📁 Data saved to {args.dst_dir}")
    print(f"📝 Metadata saved to {metadata_file}")

if __name__ == "__main__":
    main()
