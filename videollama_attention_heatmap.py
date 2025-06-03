#!/usr/bin/env python3
"""VideoLLaMA 2 — minimal *working* attention‑heat‑map demo

Run
----
python videollama2_attn_heatmap.py assets/sample_video.mp4

What you get
------------
* A caption printed in the console
* A **patch‑level** heat‑map image (24 × 24 grid from ViT‑L/14) saved as
  `outputs/heatmap_overlay.png` showing what spatial regions the model
  attended to overall while generating the caption.

Limitations
-----------
* The public 7B checkpoint *averages over time*, so you get one patch grid
  for the whole clip (no per‑frame map).  For frame‑wise maps you’d need to
  swap the temporal aggregator for `spatial_conv` or similar.
* We average attentions across **all** layers, heads, and generated tokens
  to keep the demo simple.  Feel free to slice whichever token/head/layer
  you want.
"""

import os, sys, cv2, math, torch, numpy as np
from PIL import Image

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

# ──────────────────────────  utilities  ────────────────────────────────

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

def main(video_path: str, prompt: str):
    model, processor, tokenizer = load_model()
    # 1 ▸ frames → tensor
    frames = load_frames(video_path, NUM_FRAMES)
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
    print("CAPTION:", caption)

    if not out.attentions:
        print("[ERR] No attentions returned — update transformers >4.38 or patch self‑attn.")
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
    cv2.imwrite(os.path.join(OUT_DIR, "heatmap_overlay.png"), overlay)
    print("Saved heatmap_overlay.png →", OUT_DIR)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python videollama2_attn_heatmap.py path/to/video.mp4"); sys.exit(1)
    main(sys.argv[1], prompt="Describe what is happening in the video.")
