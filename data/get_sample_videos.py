#!/usr/bin/env python3
import subprocess
import os
from pathlib import Path

def download_sample_videos():
    """Download a few sample videos for testing"""
    
    # Create output directory
    output_dir = Path("sample_dataset")
    output_dir.mkdir(exist_ok=True)
    
    # Sample YouTube videos (short, diverse content)
    videos = [
        ("dQw4w9WgXcQ", "rick_roll"),  # Famous video
        ("9bZkp7q19f0", "gangnam_style"),  # K-pop
        ("kJQP7kiw5Fk", "despacito"),  # Latin music
        ("fJ9rUzIMcZQ", "bohemian_rhapsody"),  # Rock
        ("YQHsXMglC9A", "hello_adele")  # Pop
    ]
    
    print(f"📥 Downloading {len(videos)} sample videos to {output_dir}")
    
    for video_id, name in videos:
        output_file = output_dir / f"{name}.mp4"
        
        if output_file.exists():
            print(f"✅ Already exists: {name}")
            continue
            
        cmd = [
            "yt-dlp",
            f"https://youtube.com/watch?v={video_id}",
            "--format", "mp4[height<=360]/best[height<=360]",
            "--output", str(output_file),
            "--external-downloader", "ffmpeg", 
            "--external-downloader-args", "-t 10"  # First 10 seconds only
        ]
        
        try:
            print(f"📥 Downloading: {name}")
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✅ Success: {name}")
        except Exception as e:
            print(f"❌ Failed: {name} - {e}")
    
    print(f"\n📁 Videos saved to: {output_dir}")
    return output_dir

if __name__ == "__main__":
    download_sample_videos()
