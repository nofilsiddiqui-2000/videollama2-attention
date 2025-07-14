import subprocess
import os
from pathlib import Path

# Set path to dataset
dataset_dir = Path("kinetics400_dataset")

# Loop through all MP4 files
for video_file in dataset_dir.glob("*.mp4"):
    print(f"Checking: {video_file.name}", end="")

    # Run ffmpeg to validate
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video_file), "-f", "null", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Check if there were errors
    if result.returncode != 0 or result.stderr:
        print(" ❌ Invalid — deleting")
        try:
            video_file.unlink()
        except Exception as e:
            print(f"  ⚠️ Failed to delete {video_file.name}: {e}")
    else:
        print(" ✅ OK")

print("\nValidation complete.")
