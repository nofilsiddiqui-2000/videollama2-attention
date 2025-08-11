#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────────
# videollama2_heatmap_video.py  –  spatial attention visualiser
# Works with any VideoLLaMA‑2 checkpoint that uses CLIP‑ViT‑L/14‑336
#
# Run:
#     python videollama2_heatmap_video.py input.mp4  --stride 2  --out overlay.mp4
# The first run downloads the vision‑tower weights (~1.5 GB).
# ────────────────────────────────────────────────────────────────────
import os, sys, cv2, argparse, math, time
import numpy as np
from PIL import Image
import torch
from transformers import CLIPVisionModel, CLIPProcessor
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ─── 1 ▸ load CLIP vision tower (ViT‑L/14‑336) ──────────────────────
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "openai/clip-vit-large-patch14-336"

print(f"[INIT]  loading {MODEL_ID} on {DEVICE} …")
vision_model = CLIPVisionModel.from_pretrained(MODEL_ID)
processor     = CLIPProcessor.from_pretrained(MODEL_ID)
vision_model = vision_model.to(DEVICE).eval()

# ─── 2 ▸ attention‑roll‑out helper ──────────────────────────────────
@torch.no_grad()
def attention_rollout_heatmap(pil_img: Image.Image) -> np.ndarray:
    """
    Compute cumulative [CLS]‑to‑patch attention heat‑map using rollout.
    Returns H×W float32 array in [0,1] aligned with input `pil_img`.
    """
    inputs = processor(images=pil_img, return_tensors="pt").to(DEVICE)
    outputs = vision_model(**inputs, output_attentions=True)
    attn_all = torch.stack(outputs.attentions)        # [layers, 1, heads, T, T]
    attn_all = attn_all[:, 0]                         # drop batch dim → [layers, heads, T, T]
    num_layers, num_heads, num_tokens, _ = attn_all.shape

    # rollout: (A+I)/row‑norm  ×  …     → overall CLS attention
    R = torch.eye(num_tokens, device=DEVICE)
    for layer in range(num_layers):
        A = attn_all[layer].mean(dim=0)               # [T,T]
        A = A + torch.eye(num_tokens, device=DEVICE)  # residual
        A = A / A.sum(dim=-1, keepdim=True)           # row‑norm
        R = A @ R                                     # matmul

    cls_attention = R[0, 1:]                          # drop CLS→CLS; keep patches
    grid = cls_attention.view(24, 24).cpu().numpy()   # ViT‑L/14‑336 → 24×24 patches

    # upscale to image size
    heat = Image.fromarray(grid).resize(pil_img.size, Image.BILINEAR)
    heat = np.asarray(heat, dtype=np.float32)
    heat = np.clip((heat - heat.min()) / (heat.max() + 1e-8), 0.0, 1.0)
    return heat                                       # H×W float32 in [0,1]

# ─── 3 ▸ overlay helper ─────────────────────────────────────────────
JET = cm.get_cmap("inferno")  # spectral map (0‑1 → rgba)
def overlay_heat(frame_bgr: np.ndarray, heat: np.ndarray,
                 alpha: float = 0.5) -> np.ndarray:
    """
    Blend BGR frame with heat‑map (H×W in [0,1]).
    """
    heat_rgba = (JET(heat) * 255).astype(np.uint8)      # RGBA 0‑255
    heat_rgb  = cv2.cvtColor(heat_rgba, cv2.COLOR_RGBA2RGB)
    heat_bgr  = cv2.cvtColor(heat_rgb,  cv2.COLOR_RGB2BGR)
    overlay   = cv2.addWeighted(frame_bgr, 1 - alpha, heat_bgr, alpha, 0)
    return overlay

# ─── 4 ▸ main video loop ────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_in",  help="input video path (mp4/avi/…)")
    ap.add_argument("--out",    default="overlay.mp4", help="output video")
    ap.add_argument("--stride", type=int, default=1,
                    help="process every Nth frame (≥1)")
    ap.add_argument("--alpha",  type=float, default=0.5,
                    help="heat‑map opacity (0‑1)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video_in)
    if not cap.isOpened():
        sys.exit(f"[ERR]  cannot open {args.video_in}")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc= cv2.VideoWriter_fourcc(*"mp4v")
    out_v = cv2.VideoWriter(args.out, fourcc, fps, (W, H))

    print(f"[RUN]  {args.video_in} → {args.out}  ({W}×{H} @ {fps:.2f} fps)")
    t0 = time.time()
    f_idx, saved = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok: break

        if f_idx % args.stride == 0:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            heat = attention_rollout_heatmap(pil)
            frame = overlay_heat(frame, heat, alpha=args.alpha)
            saved += 1

        out_v.write(frame)
        f_idx += 1
        if f_idx % 100 == 0:
            elapsed = time.time() - t0
            print(f"  processed {f_idx} frames  ({saved} heat‑mapped) "
                  f"in {elapsed:.1f}s", end="\r")

    cap.release(); out_v.release()
    print(f"\n[DONE] wrote {args.out}  (heat‑mapped {saved} frames)")

if __name__ == "__main__":
    main()
