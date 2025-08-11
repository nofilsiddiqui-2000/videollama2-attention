import cv2
from pathlib import Path
import shutil

dataset_dir = Path("kinetics400_dataset")
junk_dir = dataset_dir / "junk"
junk_dir.mkdir(exist_ok=True)

# Define junk file extensions
junk_extensions = [".part", ".ytdl", ".json", ".txt", ".tar.gz"]

# --- Step 1: Clean junk files ---
print("🔍 Scanning for junk files...")
for ext in junk_extensions:
    for file in dataset_dir.glob(f"*{ext}"):
        print(f"🧹 Moving junk: {file.name}")
        try:
            shutil.move(str(file), junk_dir / file.name)
        except Exception as e:
            print(f"  ⚠️ Failed to move {file.name}: {e}")

# --- Step 2: Validate .mp4 files ---
print("\n🎥 Validating .mp4 video files...")
for video_file in dataset_dir.glob("*.mp4"):
    print(f"Checking: {video_file.name}", end="")

    try:
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            raise ValueError("Cannot open video")

        ret, _ = cap.read()
        cap.release()

        if not ret:
            raise ValueError("Cannot read first frame")

        print(" ✅ OK")
    except Exception as e:
        print(f" ❌ Corrupt — deleting ({e})")
        try:
            video_file.unlink()
        except Exception as del_err:
            print(f"  ⚠️ Failed to delete: {del_err}")

print("\n✅ Validation and cleanup complete.")
