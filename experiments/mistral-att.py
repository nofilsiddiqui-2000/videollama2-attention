#!/usr/bin/env python3
"""VideoLLaMA 2 – caption + attention heatmap overlay

Run:
    python videollama_attention_heatmap.py path/to/video.mp4

Outputs an MP4 with a per-frame saliency overlay based on attention weights.
"""

import os
import sys
import math
import cv2
import torch
import numpy as np
from PIL import Image

# Add VideoLLaMA2 repo to path
sys.path.append("./VideoLLaMA2")

from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

# CONFIG
MODEL_ID = "DAMO-NLP-SG/VideoLLaMA2-7B"
NUM_FRAMES_PROCESSED = 8  # 8 or 16 depending on checkpoint
ATTN_ALPHA = 0.6
INSTRUCTION = (
    "What animals are in the video, what are they doing, and how does the video "
    "feel?"
)
OUTPUT_DIR = "attention_heatmaps"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helpers
def evenly_spaced(n_total: int, n_sel: int):
    if n_sel >= n_total:
        return list(range(n_total))
    step = n_total / n_sel
    return [int(round(step * i)) for i in range(n_sel)]

def load_frames(video_path: str, n_frames: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = set(evenly_spaced(total, n_frames))
    frames = []
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if i in idxs:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    cap.release()
    if len(frames) == 0:
        raise RuntimeError("No frames extracted – check video file")
    return frames

# Model + Processor
def load_model(model_id: str):
    disable_torch_init()
    # Disable Flash Attention to ensure attention weights are returned
    import transformers.modeling_utils as _mu
    _mu._check_and_enable_flash_attn_2 = lambda *_, **__: False
    _mu.PreTrainedModel._check_and_enable_flash_attn_2 = staticmethod(lambda *_, **__: False)
    model, processor, tokenizer = model_init(model_id)
    model.eval()
    return model, processor, tokenizer

# Patch attention to force weight return
def patch_attention():
    try:
        from transformers.models.mistral.modeling_mistral import MistralAttention
    except ImportError:
        return
    orig = MistralAttention.forward
    def _f(self, *a, **kw):
        kw["output_attentions"] = True
        return orig(self, *a, **kw)
    MistralAttention.forward = _f

# Hook attention layers
def register_hooks(model):
    """Attach hooks to all MistralAttention layers to capture all attention weights."""
    store = []
    handles = []
    try:
        from transformers.models.mistral.modeling_mistral import MistralAttention
    except ImportError as e:
        print("[ERROR] MistralAttention import failed:", e)
        raise

    def cb(_m, _i, out):
        if isinstance(out, tuple) and len(out) > 1 and isinstance(out[1], torch.Tensor):
            store.append(out[1].detach().cpu())

    for m in model.modules():
        if isinstance(m, MistralAttention):
            handles.append(m.register_forward_hook(cb))
    return handles, store

# Post-process attention to per-frame scores
def attention_to_frame_scores(attn: torch.Tensor, n_frames: int, visual_positions: list[int]):
    # attn: (B, heads, seq_len, seq_len)
    # Extract attentions from text tokens to visual tokens
    text_positions = [i for i in range(attn.shape[-1]) if i not in visual_positions]
    attn_text_to_visual = attn[:, :, text_positions, :][:, :, :, visual_positions]
    # Average over batch, heads, and text tokens
    sal = attn_text_to_visual.mean(dim=(0, 1, 2))  # Shape: (n_frames,)
    sal = sal.softmax(-1)
    if sal.numel() != n_frames:
        raise ValueError(f"Mismatch in visual token count: expected {n_frames}, got {sal.numel()}")
    return sal.tolist()

def colorise(gray: np.ndarray):
    g = np.clip(gray * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(g, cv2.COLORMAP_JET)

def save_overlay_video(src: str, scores: list[float], caption: str, alpha: float):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {src}")
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out_path = os.path.join(OUTPUT_DIR, "output_attention_video.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    max_score = max(scores) + 1e-8
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx < len(scores):
            heat = np.full((h, w), scores[idx] / max_score, dtype=np.float32)
            overlay = cv2.addWeighted(frame, 1 - alpha, colorise(heat), alpha, 0.0)
        else:
            overlay = frame
        if idx == 0:
            cv2.putText(overlay, f"Caption: {caption}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        writer.write(overlay)
        idx += 1
    cap.release()
    writer.release()
    print(f"Annotated video → {out_path}")

# Main
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python videollama_attention_heatmap.py path/to/video.mp4")
        sys.exit(1)
    vid_path = sys.argv[1]

    model, processor, tokenizer = load_model(MODEL_ID)
    patch_attention()

    frames = load_frames(vid_path, NUM_FRAMES_PROCESSED)

    # Prepare video tensor
    if isinstance(processor, dict) and "video" in processor:
        processed = processor["video"](frames)
        video_tensor = processed["pixel_values"] if isinstance(processed, dict) else processed
    else:
        video_tensor = model.get_vision_tower().image_processor(images=frames, return_tensors="pt")["pixel_values"]
    video_tensor = video_tensor.to(device=model.device, dtype=model.dtype)

    hooks, store = register_hooks(model)
    gen_out = mm_infer(
        video_tensor,
        INSTRUCTION,
        model=model,
        tokenizer=tokenizer,
        modal="video",
        do_sample=False,
        generate_kwargs={"output_attentions": True, "return_dict_in_generate": True}
    )
    caption = gen_out["text"] if isinstance(gen_out, dict) and "text" in gen_out else str(gen_out)
    print("Caption:", caption)
    for h in hooks:
        h.remove()
    if not store:
        print("[WARN] No attention captured – check layer names or model configuration")
        sys.exit(1)

    # Assume visual tokens are at the beginning (positions 0 to NUM_FRAMES_PROCESSED-1)
    visual_positions = list(range(NUM_FRAMES_PROCESSED))
    frame_scores = attention_to_frame_scores(store[-1], NUM_FRAMES_PROCESSED, visual_positions)
    save_overlay_video(vid_path, frame_scores, caption, ATTN_ALPHA)