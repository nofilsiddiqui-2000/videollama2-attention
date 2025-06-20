#!/usr/bin/env python3
# Download and prepare Kinetics-400 subset for FGSM evaluation
import os
import sys
import json
import subprocess
import pandas as pd
from pathlib import Path
import urllib.request
import csv

def setup_kinetics_subset(output_dir="kinetics_eval", num_videos=50, duration=10):
    """Download a subset of Kinetics-400 for evaluation"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"🎬 Setting up Kinetics-400 subset in {output_path}")
    print(f"📊 Target: {num_videos} videos, {duration}s each")
    
    # Download Kinetics-400 validation annotations
    annotations_url = "https://raw.githubusercontent.com/cvdfoundation/kinetics-dataset/main/k400/annotations/val.csv"
    annotations_file = output_path / "kinetics400_val.csv"
    
    if not annotations_file.exists():
        print("📥 Downloading Kinetics-400 validation annotations...")
        urllib.request.urlretrieve(annotations_url, annotations_file)
    
    # Read annotations
    df = pd.read_csv(annotations_file)
    print(f"📋 Found {len(df)} total videos in validation set")
    
    # Sample diverse classes
    sampled_df = df.groupby('label').head(1).head(num_videos)
    print(f"🎯 Selected {len(sampled_df)} videos from {sampled_df['label'].nunique()} classes")
    
    # Download videos using yt-dlp
    success_count = 0
    failed_count = 0
    
    for idx, row in sampled_df.iterrows():
        youtube_id = row['youtube_id']
        time_start = row['time_start']
        time_end = row['time_end']
        label = row['label']
        
        # Clean label for filename
        safe_label = "".join(c for c in label if c.isalnum() or c in (' ', '-', '_')).rstrip()
        output_file = output_path / f"{safe_label}_{youtube_id}.mp4"
        
        if output_file.exists():
            print(f"✅ Already exists: {output_file.name}")
            success_count += 1
            continue
        
        # Download with yt-dlp
        youtube_url = f"https://www.youtube.com/watch?v={youtube_id}"
        cmd = [
            "yt-dlp",
            "--format", "mp4[height<=480]/best[height<=480]",  # Lower quality for faster processing
            "--output", str(output_file),
            "--external-downloader", "ffmpeg",
            "--external-downloader-args", f"-ss {time_start} -t {duration}",
            youtube_url
        ]
        
        try:
            print(f"📥 Downloading: {safe_label} ({youtube_id})")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and output_file.exists():
                print(f"✅ Success: {output_file.name}")
                success_count += 1
            else:
                print(f"❌ Failed: {safe_label} - {result.stderr}")
                failed_count += 1
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout: {safe_label}")
            failed_count += 1
        except Exception as e:
            print(f"❌ Error: {safe_label} - {e}")
            failed_count += 1
    
    print(f"\n📊 Download Summary:")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"📁 Videos saved to: {output_path}")
    
    return output_path, success_count

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Kinetics-400 subset for FGSM evaluation")
    parser.add_argument("--output-dir", default="kinetics_eval", help="Output directory")
    parser.add_argument("--num-videos", type=int, default=20, help="Number of videos to download")
    parser.add_argument("--duration", type=int, default=10, help="Video duration in seconds")
    
    args = parser.parse_args()
    
    # Check if yt-dlp is installed
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ yt-dlp not found. Install with: pip install yt-dlp")
        sys.exit(1)
    
    setup_kinetics_subset(args.output_dir, args.num_videos, args.duration)
