#!/usr/bin/env python3
"""Download actual Kinetics-400 dataset for FGSM evaluation"""
import pandas as pd
import subprocess
import urllib.request
from pathlib import Path
import random

def download_kinetics400_subset(num_videos=50, split='val'):
    """Download real Kinetics-400 dataset videos"""
    
    output_dir = Path("kinetics400_dataset")
    output_dir.mkdir(exist_ok=True)
    
    print(f"🎬 Downloading REAL Kinetics-400 dataset ({split} split)")
    print(f"🎯 Target: {num_videos} videos from 400 action classes")
    
    # Download official Kinetics-400 annotations
    if split == 'val':
        csv_url = "https://storage.googleapis.com/deepmind-media/Datasets/kinetics400/val.csv"
    else:
        csv_url = "https://storage.googleapis.com/deepmind-media/Datasets/kinetics400/train.csv"
    
    csv_file = output_dir / f"kinetics400_{split}.csv"
    
    print(f"📥 Downloading Kinetics-400 {split} annotations...")
    try:
        urllib.request.urlretrieve(csv_url, csv_file)
        print(f"✅ Downloaded annotations: {csv_file}")
    except Exception as e:
        print(f"❌ Failed to download annotations: {e}")
        return None, 0
    
    # Load annotations
    df = pd.read_csv(csv_file)
    print(f"📋 Total videos in {split} set: {len(df)}")
    print(f"📊 Action classes: {df['label'].nunique()}")
    
    # Sample videos ensuring diversity across classes
    sampled_df = df.groupby('label').apply(
        lambda x: x.sample(min(2, len(x))) if len(x) >= 1 else x
    ).reset_index(drop=True)
    
    # Shuffle and select requested number
    sampled_df = sampled_df.sample(min(num_videos, len(sampled_df))).reset_index(drop=True)
    
    print(f"🎯 Selected {len(sampled_df)} videos from {sampled_df['label'].nunique()} classes")
    print(f"📝 Sample classes: {list(sampled_df['label'].unique()[:10])}")
    
    # Download videos
    success_count = 0
    failed_count = 0
    
    for idx, row in sampled_df.iterrows():
        youtube_id = row['youtube_id']
        time_start = row['time_start']
        time_end = row['time_end']
        label = row['label']
        
        # Clean filename
        safe_label = "".join(c for c in label if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_label = safe_label.replace(' ', '_')
        output_file = output_dir / f"{safe_label}_{youtube_id}_{idx:03d}.mp4"
        
        if output_file.exists():
            print(f"✅ Exists ({idx+1}/{len(sampled_df)}): {safe_label}")
            success_count += 1
            continue
        
        # Download with proper time segment
        youtube_url = f"https://www.youtube.com/watch?v={youtube_id}"
        duration = min(10, time_end - time_start)  # Max 10 seconds
        
        cmd = [
            "yt-dlp",
            youtube_url,
            "--format", "mp4[height<=480]/best[height<=480]",
            "--output", str(output_file),
            "--external-downloader", "ffmpeg",
            "--external-downloader-args", f"-ss {time_start} -t {duration}",
            "--no-warnings",
            "--quiet"
        ]
        
        try:
            print(f"📥 Downloading ({idx+1}/{len(sampled_df)}): {safe_label} [{time_start}s-{time_end}s]")
            result = subprocess.run(cmd, timeout=60, capture_output=True)
            
            if output_file.exists() and output_file.stat().st_size > 1000:  # At least 1KB
                print(f"✅ Success: {safe_label}")
                success_count += 1
            else:
                print(f"❌ Failed: {safe_label} - Invalid file")
                failed_count += 1
                if output_file.exists():
                    output_file.unlink()  # Remove invalid file
                    
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout: {safe_label}")
            failed_count += 1
        except Exception as e:
            print(f"❌ Error: {safe_label} - {e}")
            failed_count += 1
    
    print(f"\n📊 Kinetics-400 Download Summary:")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"📁 Dataset saved to: {output_dir}")
    print(f"🎬 Action classes represented: {sampled_df['label'].nunique()}")
    
    # Save metadata
    metadata_file = output_dir / "metadata.csv"
    successful_videos = sampled_df.iloc[:success_count]
    successful_videos.to_csv(metadata_file, index=False)
    print(f"📝 Metadata saved to: {metadata_file}")
    
    return output_dir, success_count

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download real Kinetics-400 dataset")
    parser.add_argument("--num-videos", type=int, default=50, 
                       help="Number of videos to download")
    parser.add_argument("--split", choices=['train', 'val'], default='val',
                       help="Dataset split to use")
    
    args = parser.parse_args()
    
    print("🎬 REAL Kinetics-400 Dataset Downloader")
    print("=" * 50)
    
    download_kinetics400_subset(args.num_videos, args.split)
