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
# Track failed videos for reporting
FAILED_VIDEOS   = []

# ──────────────────────────  utilities  ────────────────────────────────

def get_video_files(directory):
    """Get all video files from directory with common video extensions"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(directory, f'*{ext}')))
    
    return video_files

def load_frames(path: str, num_frames: int):
    """Load frames from video with robust error handling"""
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"[WARN] Cannot open {path}")
            return None
            
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            print(f"[WARN] Video {path} has no frames")
            cap.release()
            return None
            
        step = max(total // num_frames, 1)
        frames = []
        idx = 0
        
        # Try to read frames with timeout protection
        for _ in range(num_frames * 2):  # Extra attempts for robustness
            if len(frames) >= num_frames:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                idx += 1  # Try next frame
                continue
                
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (336, 336), interpolation=cv2.INTER_AREA)
                frames.append(Image.fromarray(frame))
                idx += step
            except Exception as e:
                print(f"[WARN] Error processing frame: {str(e)}")
                idx += 1
                
        cap.release()
        
        # Check if we have enough frames
        if len(frames) < num_frames:
            print(f"[WARN] Could only extract {len(frames)}/{num_frames} frames from {path}")
            if len(frames) == 0:
                return None
                
        return frames
        
    except Exception as e:
        print(f"[ERROR] Failed to load frames from {path}: {str(e)}")
        return None

def colourise(mat: np.ndarray):
    """Convert attention matrix to colormap"""
    if mat is None or mat.size == 0:
        print("[WARN] Empty matrix provided to colourise")
        return np.zeros((336, 336, 3), dtype=np.uint8)
        
    try:
        min_val = mat.min()
        max_val = mat.max()
        
        # Avoid division by zero
        if max_val - min_val < 1e-6:
            normalized = np.zeros_like(mat)
        else:
            normalized = (mat - min_val) / (max_val - min_val)
            
        # Convert to uint8 and apply colormap
        mat_uint8 = np.clip(normalized * 255, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(mat_uint8, cv2.COLORMAP_JET)
    except Exception as e:
        print(f"[ERROR] Failed to colourise matrix: {str(e)}")
        return np.zeros((336, 336, 3), dtype=np.uint8)

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
        if frames is None or len(frames) == 0:
            print(f"[WARN] Could not load frames from {video_path}, skipping...")
            FAILED_VIDEOS.append((video_path, "Failed to load frames"))
            return None
            
        # Process video frames to tensor
        try:
            vid_tensor = processor["video"](frames)  # (T, C, H, W) float32 0‑1
            vid_tensor = vid_tensor.half().to(next(model.parameters()).device)
        except Exception as e:
            print(f"[ERROR] Failed to process video tensor for {video_path}: {str(e)}")
            FAILED_VIDEOS.append((video_path, f"Failed to process tensor: {str(e)}"))
            return None

        # 2 ▸ build prompt with <video> token
        chat_str = tokenizer.apply_chat_template([
            {"role": "user", "content": DEFAULT_VIDEO_TOKEN + "\n" + prompt}],
            tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer_multimodal_token(chat_str, tokenizer, DEFAULT_VIDEO_TOKEN,
                                           return_tensors="pt").to(DEVICE)
        attn_mask = torch.ones_like(input_ids, device=DEVICE)

        # 3 ▸ generate with attentions
        try:
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                images=[(vid_tensor, "video")],
                do_sample=False,
                max_new_tokens=64,
                output_attentions=True,
                return_dict_in_generate=True)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"[ERROR] CUDA OOM for {video_path}, trying to free memory...")
                torch.cuda.empty_cache()
                FAILED_VIDEOS.append((video_path, "CUDA out of memory"))
                return None
            else:
                print(f"[ERROR] Model generation failed for {video_path}: {str(e)}")
                FAILED_VIDEOS.append((video_path, f"Generation failed: {str(e)}"))
                return None
        except Exception as e:
            print(f"[ERROR] Model generation failed for {video_path}: {str(e)}")
            FAILED_VIDEOS.append((video_path, f"Generation failed: {str(e)}"))
            return None

        caption = tokenizer.decode(out.sequences[0], skip_special_tokens=True)
        print(f"CAPTION for {video_name}: {caption}")

        if not out.attentions:
            print(f"[ERR] No attentions returned for {video_path}")
            FAILED_VIDEOS.append((video_path, "No attentions returned"))
            return None

        # 4 ▸ attn stack → (layers, heads, seq, seq)
        try:
            attn_layers = torch.stack([torch.stack(layer) for layer in out.attentions])  # L, H, S, S
            # Average over layers & heads → (S, S)
            attn_avg = attn_layers.mean(dim=(0,1))
            # Visual tokens occupy first PATCHES positions
            vis_slice = attn_avg[:, :PATCHES]          # (S_text+vis, P)
            text_len  = attn_avg.size(0) - PATCHES
            text_to_vis = vis_slice[-text_len:].mean(dim=0)  # average queries over generated tokens
            heat = text_to_vis.reshape(NUM_PATCH_SIDE, NUM_PATCH_SIDE).cpu().numpy()
        except Exception as e:
            print(f"[ERROR] Failed to process attention weights for {video_path}: {str(e)}")
            FAILED_VIDEOS.append((video_path, f"Failed to process attentions: {str(e)}"))
            return None

        # 5 ▸ overlay on first frame for demo
        try:
            base = np.array(frames[0])                 # RGB 336×336
            if base.shape[:2] != (336, 336):
                print(f"[WARN] Frame shape mismatch: expected (336, 336), got {base.shape[:2]}")
                base = cv2.resize(base, (336, 336))
                
            heatmap = colourise(heat)
            # Make sure base is in BGR format for cv2 and has the right shape
            if len(base.shape) == 3 and base.shape[2] == 3:
                base_bgr = base[..., ::-1]  # RGB to BGR
                overlay = cv2.addWeighted(base_bgr, 0.4, heatmap, 0.6, 0)
            else:
                print(f"[WARN] Unexpected base frame shape {base.shape} for {video_path}")
                overlay = heatmap  # Just use the heatmap if base is invalid
        except Exception as e:
            print(f"[ERROR] Failed to create overlay for {video_path}: {str(e)}")
            FAILED_VIDEOS.append((video_path, f"Failed to create overlay: {str(e)}"))
            return None
            
        # Save with video name in the filename
        output_path = os.path.join(OUT_DIR, f"{video_name}_heatmap.png")
        try:
            cv2.imwrite(output_path, overlay)
            print(f"Saved {output_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save heatmap for {video_path}: {str(e)}")
            FAILED_VIDEOS.append((video_path, f"Failed to save heatmap: {str(e)}"))
            return None
        
        # Also save the caption to a text file
        try:
            with open(os.path.join(OUT_DIR, f"{video_name}_caption.txt"), 'w') as f:
                f.write(caption)
        except Exception as e:
            print(f"[WARN] Failed to save caption for {video_path}: {str(e)}")
            # Don't consider this a complete failure
            
        # Return the results for potential further processing
        return {
            'video_name': video_name,
            'caption': caption,
            'heatmap_path': output_path
        }
        
    except Exception as e:
        print(f"[ERROR] Failed to process {video_path}: {str(e)}")
        FAILED_VIDEOS.append((video_path, str(e)))
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
        try:
            result = process_video(video_path, PROMPT, model, processor, tokenizer)
            if result:
                results.append(result)
            # Free up memory
            torch.cuda.empty_cache()
        except KeyboardInterrupt:
            print("Process interrupted by user. Saving results so far...")
            break
        except Exception as e:
            print(f"[ERROR] Unexpected error processing {video_path}: {str(e)}")
            FAILED_VIDEOS.append((video_path, f"Unexpected error: {str(e)}"))
            # Try to recover and continue with next video
            torch.cuda.empty_cache()
            continue
    
    # Save a summary of all processed videos
    try:
        print(f"Successfully processed {len(results)} out of {len(video_files)} videos")
        print(f"Failed to process {len(FAILED_VIDEOS)} videos")
        
        with open(os.path.join(OUT_DIR, "summary.txt"), 'w') as f:
            f.write(f"Processed {len(results)} videos\n\n")
            for result in results:
                f.write(f"Video: {result['video_name']}\n")
                f.write(f"Caption: {result['caption']}\n")
                f.write(f"Heatmap: {result['heatmap_path']}\n")
                f.write("-" * 80 + "\n\n")
            
            if FAILED_VIDEOS:
                f.write("\n\nFAILED VIDEOS:\n")
                for video_path, reason in FAILED_VIDEOS:
                    f.write(f"{video_path}: {reason}\n")
    except Exception as e:
        print(f"[ERROR] Failed to write summary: {str(e)}")

if __name__ == "__main__":
    main()
