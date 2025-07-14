import cv2
from pathlib import Path

dataset_dir = Path("kinetics400_dataset")

for video_file in dataset_dir.glob("*.mp4"):
    print(f"Checking: {video_file.name}", end="")

    try:
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            raise ValueError("Cannot open video")

        # Try to read first frame
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

print("\nValidation complete.")
