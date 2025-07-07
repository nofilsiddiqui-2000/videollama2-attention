#!/usr/bin/env python3
import pandas as pd, subprocess, time
from pathlib import Path

def is_video_valid(fp, min_duration=1.0, min_frames=5):
    import cv2
    cap = cv2.VideoCapture(str(fp))
    if not cap.isOpened(): return False
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frames >= min_frames and (fps > 0 and (frames / fps) >= min_duration

def main(num_videos=250, split='train'):
    out = Path("kinetics400_dataset")
    out.mkdir(exist_ok=True)

    log_fail = out / "failed_ids.txt"
    if log_fail.exists():
        log_fail.unlink()

    csv = Path(f"kinetics400_{split}.csv")
    if not csv.exists():
        print(f"❌ Missing CSV: {csv}")
        return

    df = pd.read_csv(csv)
    success = attempts = 0
    max_attempts = num_videos * 7

    while success < num_videos and attempts < max_attempts:
        row = df.sample(1).iloc[0]
        ytid, start, end, label = row['youtube_id'], float(row['time_start']), float(row['time_end']), row['label']
        safe = "".join(c if c.isalnum() or c in ('_','-') else '_' for c in label)[:30]
        fname = f"{safe}_{ytid}_{int(start)}.mp4"
        outf = out / fname

        if outf.exists() and is_video_valid(outf):
            print(f"✅ EXISTS: {fname}")
            success += 1
            attempts += 1
            continue

        duration = min(10, end - start)
        section = f"*{start}-{start + duration}"
        cmd = [
            "yt-dlp",
            f"https://www.youtube.com/watch?v={ytid}",
            "--format", "mp4[height<=480]/best[height<=480]",
            "--download-sections", section,
            "--output", str(outf),
            "--no-warnings", "--quiet",
        ]
        subprocess.run(cmd, timeout=120)
        time.sleep(0.2)

        if outf.exists() and outf.stat().st_size > 1024 and is_video_valid(outf):
            print(f"✅ DL & OK: {fname}")
            success += 1
        else:
            print(f"❌ BAD: {fname} — deleting")
            outf.unlink(missing_ok=True)
            log_fail.write_text((log_fail.read_text() + f"{ytid}\n") if log_fail.exists() else f"{ytid}\n")

        attempts += 1

    print(f"\n✅ Finished: {success}/{num_videos} valid videos in {attempts} attempts.")
    print(f"⚠️ Failed IDs logged to: {log_fail}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-videos", type=int, default=250)
    parser.add_argument("--split", choices=['train','val'], default='train')
    args = parser.parse_args()
    main(args.num_videos, args.split)
