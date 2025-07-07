#!/usr/bin/env python3
"""VideoLLaMA 2 — batch processing for attention‑heat‑maps

Run
----
python videollama2_attn_heatmap_batch.py

What you get
------------
* A caption printed in the console for each video
* A **patch‑level** heat‑map image (24 × 24 grid from ViT‑L/14) saved for each video in
  `outputs/[video_name]_heatmap.png` showing what spatial regions the model
  attended to overall while generating the caption.
"""

import os, sys, cv2, math, torch, numpy as np
import glob
from PIL import Image
from tqdm import tqdm

# ─────────────────────────  add repo to path  ──────────────────────────
sys.path.append("./VideoLLaMA2")

from videollama2 import model_init
from videollama2.mm_utils import tokenizer_multimodal_token
from videollama2.utils import disable_torch_init
from videollama2.constants import DEFAULT_VIDEO_TOKEN

# ──────────────────────────────  CONFIG  ───────────────────────────────
MODEL_ID        = "DAMO-NLP-SG/VideoLLaMA2-7B"
NUM_PATCH_SIDE  = 24                   # ViT‑L/14 on 336×336 ⇒ 24×24 patches
PATCHES         = NUM_PATCH_SIDE ** 2  # 576
NUM_FRAMES      = 8                    # default temporal pooling window
OUT_DIR         = "outputs"; os.makedirs(OUT_DIR, exist_ok=True)
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_DIR     = "kinetics400_dataset"
PROMPT          = "Describe what is happening in the video."

# ──────────────────────────  utilities  ────────────────────────────────

def get_video_files(directory):
    """Get all video files from directory with common video extensions"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(directory, f'*{ext}')))
    
    return video_files

def load_frames(path: str, num_frames: int):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(total // num_frames, 1)
    frames = []
    idx = 0
    while len(frames) < num_frames and cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read(); idx += step
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (336, 336), interpolation=cv2.INTER_AREA)
        frames.append(Image.fromarray(frame))
    cap.release()
    return frames


def colourise(mat: np.ndarray):
    mat = (mat - mat.min()) / (mat.max() - mat.min() + 1e-6)
    mat = np.clip(mat * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(mat, cv2.COLORMAP_JET)

# ───────────────────────  model + processors  ──────────────────────────

def load_model(model_id=MODEL_ID):
    disable_torch_init()
    # Turn off FA‑2 to guarantee attention weights propagate
    import transformers.modeling_utils as _mu
    _mu._check_and_enable_flash_attn_2 = lambda *_, **__: False
    _mu.PreTrainedModel._check_and_enable_flash_attn_2 = staticmethod(lambda *_, **__: False)

    model, processor, tokenizer = model_init(model_id)
    model.eval()  # Already device-mapped by Accelerate
    return model, processor, tokenizer

# ─────────────────────────  main flow  ─────────────────────────────────

def process_video(video_path: str, prompt: str, model, processor, tokenizer):
    # Extract filename without extension for output naming
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    try:
        # 1 ▸ frames → tensor
        frames = load_frames(video_path, NUM_FRAMES)
        if not frames:
            print(f"[WARN] Could not load frames from {video_path}, skipping...")
            return
            
        vid_tensor = processor["video"](frames)  # (T, C, H, W) float32 0‑1
        vid_tensor = vid_tensor.half().to(next(model.parameters()).device)

        # 2 ▸ build prompt with <video> token
        chat_str = tokenizer.apply_chat_template([
            {"role": "user", "content": DEFAULT_VIDEO_TOKEN + "\n" + prompt}],
            tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer_multimodal_token(chat_str, tokenizer, DEFAULT_VIDEO_TOKEN,
                                           return_tensors="pt").to(DEVICE)
        attn_mask = torch.ones_like(input_ids, device=DEVICE)

        # 3 ▸ generate with attentions
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            images=[(vid_tensor, "video")],
            do_sample=False,
            max_new_tokens=64,
            output_attentions=True,
            return_dict_in_generate=True)

        caption = tokenizer.decode(out.sequences[0], skip_special_tokens=True)
        print(f"CAPTION for {video_name}: {caption}")

        if not out.attentions:
            print(f"[ERR] No attentions returned for {video_path}")
            return

        # 4 ▸ attn stack → (layers, heads, seq, seq)
        attn_layers = torch.stack([torch.stack(layer) for layer in out.attentions])  # L, H, S, S
        # Average over layers & heads → (S, S)
        attn_avg = attn_layers.mean(dim=(0,1))
        # Visual tokens occupy first PATCHES positions
        vis_slice = attn_avg[:, :PATCHES]          # (S_text+vis, P)
        text_len  = attn_avg.size(0) - PATCHES
        text_to_vis = vis_slice[-text_len:].mean(dim=0)  # average queries over generated tokens
        heat = text_to_vis.reshape(NUM_PATCH_SIDE, NUM_PATCH_SIDE).cpu().numpy()

        # 5 ▸ overlay on first frame for demo
        base = np.array(frames[0])                 # RGB 336×336
        heatmap = colourise(heat)
        overlay = cv2.addWeighted(base[..., ::-1], 0.4, heatmap, 0.6, 0)
        
        # Save with video name in the filename
        output_path = os.path.join(OUT_DIR, f"{video_name}_heatmap.png")
        cv2.imwrite(output_path, overlay)
        print(f"Saved {output_path}")
        
        # Also save the caption to a text file
        with open(os.path.join(OUT_DIR, f"{video_name}_caption.txt"), 'w') as f:
            f.write(caption)
            
        # Return the results for potential further processing
        return {
            'video_name': video_name,
            'caption': caption,
            'heatmap_path': output_path
        }
        
    except Exception as e:
        print(f"[ERROR] Failed to process {video_path}: {str(e)}")
        return None

def main():
    print(f"Loading model {MODEL_ID}...")
    model, processor, tokenizer = load_model()
    
    # Get all video files from the dataset directory
    video_files = get_video_files(DATASET_DIR)
    print(f"Found {len(video_files)} videos to process in {DATASET_DIR}")
    
    results = []
    
    # Process each video with progress bar
    for video_path in tqdm(video_files, desc="Processing videos"):
        result = process_video(video_path, PROMPT, model, processor, tokenizer)
        if result:
            results.append(result)
    
    print(f"Successfully processed {len(results)} out of {len(video_files)} videos")
    
    # Save a summary of all processed videos
    with open(os.path.join(OUT_DIR, "summary.txt"), 'w') as f:
        f.write(f"Processed {len(results)} videos\n\n")
        for result in results:
            f.write(f"Video: {result['video_name']}\n")
            f.write(f"Caption: {result['caption']}\n")
            f.write(f"Heatmap: {result['heatmap_path']}\n")
            f.write("-" * 80 + "\n\n")

if __name__ == "__main__":
    main()
