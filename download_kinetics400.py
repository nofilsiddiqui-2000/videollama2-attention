#!/usr/bin/env python3
"""Download actual Kinetics-400 dataset for FGSM evaluation - improved with validation."""
import pandas as pd
import subprocess
import urllib.request
from pathlib import Path
import random
import json
import cv2
import time

def is_video_valid(filepath, min_duration=1.0, min_frames=5):
    """Check if video is readable and of sufficient length using OpenCV."""
    try:
        cap = cv2.VideoCapture(str(filepath))
        if not cap.isOpened():
            cap.release()
            return False
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frames / fps if fps > 0 else 0
        cap.release()
        if duration < min_duration or frames < min_frames:
            return False
        return True
    except Exception:
        return False

def download_kinetics400_subset(num_videos=50, split='val'):
    """Download real Kinetics-400 dataset videos with updated URLs and validation."""

    output_dir = Path("kinetics400_dataset")
    output_dir.mkdir(exist_ok=True)

    print(f"🎬 Downloading REAL Kinetics-400 dataset ({split} split)")
    print(f"🎯 Target: {num_videos} usable videos from 400 action classes")

    # Load annotations
    base_url = "https://raw.githubusercontent.com/cvdfoundation/kinetics-dataset/main/k400/annotations"
    csv_url = f"{base_url}/{split}.csv"
    csv_file = output_dir / f"kinetics400_{split}.csv"

    print(f"📥 Downloading annotations...")
    try:
        req = urllib.request.Request(csv_url, headers={'User-Agent': 'Mozilla/5.0'})
        urllib.request.urlretrieve(csv_url, csv_file)
    except Exception as e:
        print(f"❌ Failed to download CSV: {e}")
        return

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}")
        return

    success_count = 0
    attempted = 0
    max_attempts = num_videos * 5  # prevent infinite loops on repeated failures

    while success_count < num_videos and attempted < max_attempts:
        row = df.sample(1).iloc[0]
        youtube_id = row['youtube_id']
        time_start = row['time_start']
        time_end = row['time_end']
        label = row['label']

        safe_label = "".join(c for c in label if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')[:30]
        output_file = output_dir / f"{safe_label}_{youtube_id}_{time_start}.mp4"

        if output_file.exists() and is_video_valid(output_file):
            print(f"✅ Exists and valid: {output_file.name}")
            success_count += 1
            attempted += 1
            continue

        youtube_url = f"https://www.youtube.com/watch?v={youtube_id}"
        duration = min(10, time_end - time_start)
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

        print(f"📥 Downloading: {output_file.name}")
        try:
            subprocess.run(cmd, timeout=90, capture_output=True)
            time.sleep(1)  # slight delay to ensure file is finalized

            if output_file.exists() and output_file.stat().st_size > 1024:
                if is_video_valid(output_file):
                    print(f"✅ Downloaded and valid: {output_file.name}")
                    success_count += 1
                else:
                    print(f"❌ Invalid (corrupted/too short), deleting: {output_file.name}")
                    output_file.unlink()
            else:
                print(f"❌ Download failed or file too small, deleting: {output_file.name}")
                if output_file.exists():
                    output_file.unlink()
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout: {output_file.name}")
            if output_file.exists():
                output_file.unlink()
        except Exception as e:
            print(f"❌ Error: {e}")
            if output_file.exists():
                output_file.unlink()

        attempted += 1

    print(f"\n📊 Kinetics-400 Download Summary:")
    print(f"✅ Successfully downloaded: {success_count}")
    print(f"⚠️ Attempts made: {attempted}")
    print(f"📁 Dataset saved to: {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download clean Kinetics-400 dataset for experiments.")
    parser.add_argument("--num-videos", type=int, default=50, help="Number of usable videos to download.")
    parser.add_argument("--split", choices=['train', 'val', 'test'], default='val', help="Split to download from.")
    args = parser.parse_args()

    print("🎬 REAL Kinetics-400 Downloader (Validated)")
    print("=" * 50)
    download_kinetics400_subset(args.num_videos, args.split)
