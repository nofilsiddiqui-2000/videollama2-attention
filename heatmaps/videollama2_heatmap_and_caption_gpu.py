#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────────
# Heat-map + Caption pipeline for VideoLLaMA-2 (7B-16F) on GPU only
# ────────────────────────────────────────────────────────────────────
#  • Saves an MP4 with spatial attention overlays
#  • Writes a one-line caption to caption.txt
#
# Usage (assumes you have a CUDA GPU):
#   python videollama2_heatmap_and_caption_gpu.py input.mp4 --stride 3 --alpha 0.4
#
# If you want CPU later, you can modify generate_caption(...) accordingly.
# ────────────────────────────────────────────────────────────────────
import os, sys, cv2, argparse, time
import numpy as np
from PIL import Image
import torch
import transformers.modeling_utils as _mu
from transformers import CLIPVisionModel, CLIPProcessor
import matplotlib.cm as cm

# ─── (1) neutralise Flash-Attention-2 everywhere (captioner safety) ─────
def _disable_fa2(*_, **__):
    return False

# Prevent any Flash-Attention 2 from turning on
_mu._check_and_enable_flash_attn_2 = _disable_fa2
_mu.PreTrainedModel._check_and_enable_flash_attn_2 = staticmethod(_disable_fa2)

# Force eager attention in HF vision modules
os.environ["PYTORCH_ATTENTION_IMPLEMENTATION"] = "eager"
os.environ["HF_DISABLE_FLASH_ATTN_2"]           = "1"
os.environ["DISABLE_FLASH_ATTN_2"]              = "1"

# ─── (2) heat-map helpers (GPU-only) ──────────────────────────────────
VISION_ID = "openai/clip-vit-large-patch14-336"

def load_vision_tower(device: str):
    """
    Load CLIP ViT-L/14-336 vision encoder on `device` (should be "cuda").
    Returns (vision_model, processor).
    """
    vt   = CLIPVisionModel.from_pretrained(VISION_ID).to(device).eval()
    proc = CLIPProcessor.from_pretrained(VISION_ID)
    return vt, proc

@torch.no_grad()
def heatmap_rollout(pil_img: Image.Image, vt: CLIPVisionModel, proc: CLIPProcessor, device: str) -> np.ndarray:
    """
    Compute cumulative [CLS]→patch attention heatmap using rollout (GPU).
    Returns an H×W float32 array in [0,1].
    """
    inputs  = proc(images=pil_img, return_tensors="pt").to(device)
    outs    = vt(**inputs, output_attentions=True)
    att_all = torch.stack(outs.attentions)[:, 0]   # → [layers, heads, T, T]
    T = att_all.shape[-1]
    eye = torch.eye(T, device=device)
    R = eye.clone()
    for layer_attn in att_all.mean(dim=1):        # average over heads
        A = layer_attn + eye                       # add residual (T×T)
        A = A / A.sum(dim=-1, keepdim=True)         # row-normalize
        R = A @ R                                   # matmul
    cls_attn = R[0, 1:].view(24, 24).cpu().numpy()  # drop CLS→CLS, reshape to 24×24

    # Upsample to original image size
    heat = Image.fromarray(cls_attn).resize(pil_img.size, Image.BILINEAR)
    heat = np.asarray(heat, dtype=np.float32)
    heat = np.clip((heat - heat.min()) / (heat.max() + 1e-8), 0.0, 1.0)
    return heat  # H×W float32 in [0,1]

JET = cm.get_cmap("inferno")

def overlay(frame_bgr: np.ndarray, heat: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay a heatmap (H×W in [0,1]) onto a BGR frame using 'inferno' colormap.
    """
    heat_rgba = (JET(heat) * 255).astype(np.uint8)            # H×W×4 (RGBA)
    heat_rgb  = cv2.cvtColor(heat_rgba, cv2.COLOR_RGBA2RGB)   # H×W×3 (RGB)
    heat_bgr  = cv2.cvtColor(heat_rgb,  cv2.COLOR_RGB2BGR)    # H×W×3 (BGR)
    return cv2.addWeighted(frame_bgr, 1 - alpha, heat_bgr, alpha, 0)

# ─── (3) caption helper using VideoLLaMA-2 (7B-16F) on GPU ───────────────────
def generate_caption(video_path: str, device: str) -> str:
    """
    Generate a one-line caption for `video_path` using VideoLLaMA2-7B-16F entirely on GPU.
    Returns the generated caption string.
    """
    from videollama2 import model_init, mm_infer
    from videollama2.utils import disable_torch_init

    disable_torch_init()  # speed-ups for HF models

    MODEL_NAME = "DAMO-NLP-SG/VideoLLaMA2-7B-16F"
    print(f"[CAP] Loading {MODEL_NAME} on {device} …")

    # Always load in float16 on GPU
    model, processor, tokenizer = model_init(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=device,             # put entire 7B model on GPU
        attn_implementation="eager"
    )
    model.eval()

    # The "video" processor returns a float32 tensor → cast to float16 before mm_infer
    vid_tensor = processor["video"](video_path).to(torch.float16).to(device)
    caption    = mm_infer(
        vid_tensor,
        "Describe the video.",
        model=model,
        tokenizer=tokenizer,
        modal="video",
        do_sample=False
    ).strip()
    return caption

# ─── (4) main CLI ────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="input video path (mp4/avi/… )")

    # heat-map flags
    ap.add_argument("--out", default="overlay.mp4", help="output heat-map video")
    ap.add_argument("--stride", type=int, default=1,
                    help="process every Nth frame (≥1)")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="heat-map opacity (0–1)")

    # caption flags (GPU only—no cpu path)
    ap.add_argument("--caption-out", default="caption.txt",
                    help="file to write the generated caption")

    args = ap.parse_args()

    # Ensure CUDA is available
    if not torch.cuda.is_available():
        sys.exit("[ERR] No CUDA GPU detected. This script is GPU-only.")

    # ───── (a) Generate heat-map video ──────────────────────────────────
    heat_dev = "cuda"
    vt, proc = load_vision_tower(heat_dev)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"[ERR] Cannot open {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (W, H))

    print(f"[RUN] Heat-mapping {args.video} → {args.out}")
    t0 = time.time()
    frame_idx, mapped = 0, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % args.stride == 0:
            # Convert BGR → PIL (RGB)
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            heat = heatmap_rollout(pil_frame, vt, proc, heat_dev)
            frame = overlay(frame, heat, alpha=args.alpha)
            mapped += 1

        writer.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {frame_idx} frames ({mapped} mapped) – {elapsed:.1f}s", end="\r")

    cap.release()
    writer.release()
    print(f"\n[✔] Wrote {args.out}  ({mapped} heat-mapped frames)")

    # ───── (b) Generate video caption ──────────────────────────────────
    caption = generate_caption(args.video, "cuda")
    os.makedirs(os.path.dirname(args.caption_out) or ".", exist_ok=True)
    with open(args.caption_out, "w", encoding="utf-8") as f:
        f.write(caption + "\n")

    print(f"\n[CAPTION] → {args.caption_out}\n---\n{caption}\n---\n")

if __name__ == "__main__":
    main()
