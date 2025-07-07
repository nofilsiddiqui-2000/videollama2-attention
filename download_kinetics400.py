#!/usr/bin/env python3
import pandas as pd
import subprocess, time, random, os
from pathlib import Path
import cv2
import requests
import shutil

def is_video_valid(fp, min_duration=0.5, min_frames=3):
    """Check if video is valid with slightly more lenient requirements"""
    try:
        cap = cv2.VideoCapture(str(fp))
        if not cap.isOpened():
            return False
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        duration = frames / fps if fps > 0 else 0
        return frames >= min_frames and duration >= min_duration
    except Exception as e:
        print(f"Error validating video {fp}: {e}")
        return False

def download_with_retries(ytid, start, end, outf, max_retries=3):
    """Try multiple formats and retry download"""
    formats = [
        "mp4[height<=480]/best[height<=480]",
        "mp4/best",
        "bestvideo[height<=480]+bestaudio/best[height<=480]",
        "bestvideo+bestaudio/best"
    ]
    
    for attempt in range(max_retries):
        for fmt in formats:
            duration = min(10, end - start)
            cmd = [
                "yt-dlp",
                "--format", fmt,
                "--output", str(outf),
                "--download-sections", f"*{start}-{start+duration}",
                "--no-check-certificate",
                "--geo-bypass",
                "--force-ipv4",
                f"https://www.youtube.com/watch?v={ytid}"
            ]
            
            try:
                print(f"Attempt {attempt+1} with format {fmt} for {ytid}")
                res = subprocess.run(cmd, timeout=180, capture_output=True)
                
                if outf.exists() and outf.stat().st_size > 1024 and is_video_valid(outf):
                    return True
                    
                # Wait between attempts
                time.sleep(random.uniform(1.5, 3.0))
            
            except subprocess.TimeoutExpired:
                print(f"Timeout for {ytid}, trying next format...")
                continue
            except Exception as e:
                print(f"Error downloading {ytid}: {e}")
                continue
                
    return False

def check_and_download_csv(split='train'):
    """Download the CSV file if it doesn't exist"""
    csv_path = Path(f"kinetics400_{split}.csv")
    if csv_path.exists():
        return True
        
    print(f"CSV file {csv_path} not found. Attempting to download...")
    csv_url = f"https://raw.githubusercontent.com/open-mmlab/mmaction2/master/tools/data/kinetics/kinetics400_{split}_list.txt"
    
    try:
        response = requests.get(csv_url)
        if response.status_code == 200:
            # Convert the format to match expected CSV
            lines = response.text.strip().split('\n')
            data = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    video_info = parts[0].split('/')[-1]  # Get the last part
                    label = ' '.join(parts[1:])
                    
                    # Parse video ID and time info
                    if '_' in video_info:
                        segments = video_info.split('_')
                        if len(segments) >= 3:
                            youtube_id = segments[-2]
                            time_info = segments[-1].replace('.mp4', '')
                            try:
                                time_start = int(time_info)
                                time_end = time_start + 10  # Assume 10 seconds
                                data.append([youtube_id, time_start, time_end, label])
                            except ValueError:
                                pass
            
            # Create DataFrame and save as CSV
            if data:
                df = pd.DataFrame(data, columns=['youtube_id', 'time_start', 'time_end', 'label'])
                df.to_csv(csv_path, index=False)
                print(f"CSV file created at {csv_path}")
                return True
            else:
                print("Failed to parse data for CSV")
        else:
            print(f"Failed to download CSV. Status code: {response.status_code}")
            
        # Try secondary source for CSV
        secondary_url = f"https://github.com/activitynet/ActivityNet/raw/master/Crawler/Kinetics/data/kinetics_{split}.json"
        # Would need additional code to parse JSON format to CSV
            
    except Exception as e:
        print(f"Error downloading CSV: {e}")
    
    return False

def main(num_videos=250, split='train'):
    out = Path("kinetics400_dataset")
    out.mkdir(exist_ok=True)
    log_fail = out / "failed_ids.txt"
    log_success = out / "success_ids.txt"
    
    # Ensure we have the CSV file
    if not check_and_download_csv(split):
        print("Could not obtain the required CSV file. Please download it manually.")
        print("You can find it at: https://github.com/open-mmlab/mmaction2/tree/master/tools/data/kinetics")
        return
    
    csv = Path(f"kinetics400_{split}.csv")
    if not csv.exists():
        print(f"❌ Missing CSV: {csv}")
        return

    # Try to update yt-dlp
    try:
        subprocess.run(["pip", "install", "--upgrade", "yt-dlp"], check=False)
        print("Updated yt-dlp to latest version")
    except:
        print("Could not update yt-dlp, continuing with current version")

    df = pd.read_csv(csv)
    print(f"Loaded {len(df)} video entries from {csv}")
    
    success = attempts = 0
    max_attempts = num_videos * 10  # More attempts
    already_downloaded = []
    
    # Track already successful IDs
    if log_success.exists():
        with open(log_success, 'r') as f:
            already_downloaded = [line.strip() for line in f.readlines()]
        print(f"Found {len(already_downloaded)} already downloaded videos")
    
    # Create list of unique classes to ensure diversity
    classes = df['label'].unique()
    target_per_class = max(1, num_videos // len(classes))
    
    class_counts = {cls: 0 for cls in classes}
    
    while success < num_videos and attempts < max_attempts:
        # Try to get videos from underrepresented classes
        underrepresented = [cls for cls, count in class_counts.items() 
                           if count < target_per_class]
        
        if underrepresented and random.random() < 0.7:  # 70% chance to pick from underrepresented
            label = random.choice(underrepresented)
            class_df = df[df['label'] == label]
            if len(class_df) > 0:
                row = class_df.sample(1).iloc[0]
            else:
                row = df.sample(1).iloc[0]
        else:
            row = df.sample(1).iloc[0]
            
        ytid, start, end, label = row['youtube_id'], int(row['time_start']), int(row['time_end']), row['label']
        
        # Skip if we already tried this video
        if ytid in already_downloaded:
            attempts += 1
            continue
            
        safe = "".join(c for c in label if c.isalnum() or c in ('_', '-')).replace(' ', '_')[:30]
        outf = out / f"{safe}_{ytid}_{start}.mp4"

        if outf.exists() and is_video_valid(outf):
            print(f"✅ EXISTS: {outf.name}")
            success += 1
            class_counts[label] = class_counts.get(label, 0) + 1
            already_downloaded.append(ytid)
            with open(log_success, 'a') as f:
                f.write(f"{ytid}\n")
            attempts += 1
            continue

        # Add some random delay to avoid rate limiting
        time.sleep(random.uniform(0.5, 2.0))
        
        if download_with_retries(ytid, start, end, outf):
            print(f"✅ DL & OK: {outf.name}")
            success += 1
            class_counts[label] = class_counts.get(label, 0) + 1
            already_downloaded.append(ytid)
            with open(log_success, 'a') as f:
                f.write(f"{ytid}\n")
        else:
            print(f"❌ BAD: {outf.name} — deleting")
            outf.unlink(missing_ok=True)
            with open(log_fail, 'a') as f:
                f.write(f"{ytid}\n")
        attempts += 1
        
        # Print progress
        print(f"Progress: {success}/{num_videos} videos ({attempts} attempts)")
        
        # If we've tried too many without success, relax criteria
        if attempts > num_videos * 3 and success < num_videos * 0.3:
            print("Download rate too low, relaxing requirements...")
            break

    print(f"\n✅ Finished: {success}/{num_videos} valid videos in {attempts} attempts.")
    print(f"⚠️ Failed IDs logged to: {log_fail}")
    print(f"✅ Successful IDs logged to: {log_success}")
    
    # Class distribution
    print("\nClass distribution:")
    for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"{cls}: {count} videos")

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--num-videos', type=int, default=250)
    p.add_argument('--split', choices=['train', 'val'], default='train')
    args = p.parse_args()
    main(args.num_videos, args.split)
