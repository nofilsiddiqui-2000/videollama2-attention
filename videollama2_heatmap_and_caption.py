#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────────
# Heat‑map + Caption pipeline for VideoLLaMA‑2 (7B‑16F)
# ────────────────────────────────────────────────────────────────────
#  • Saves an MP4 with spatial attention overlays
#  • Writes a single‑sentence caption to caption.txt
#
# Example:
#   python videollama2_heatmap_and_caption.py input.mp4 \
#          --caption-device cuda --stride 2 --alpha 0.4
# ────────────────────────────────────────────────────────────────────
import os, sys, cv2, argparse, time
import numpy as np
from PIL import Image
import torch
import transformers.modeling_utils as _mu
from transformers import CLIPVisionModel, CLIPProcessor
import matplotlib.cm as cm

# ─── (1) neutralise Flash‑Attn‑2 everywhere (captioner safety) ─────
def _disable_fa2(*_, **__): return False
_mu._check_and_enable_flash_attn_2 = _disable_fa2
_mu.PreTrainedModel._check_and_enable_flash_attn_2 = staticmethod(_disable_fa2)
os.environ["PYTORCH_ATTENTION_IMPLEMENTATION"] = "eager"
os.environ["HF_DISABLE_FLASH_ATTN_2"] = "1"
os.environ["DISABLE_FLASH_ATTN_2"]  = "1"

# ─── (2) heat‑map helpers (identical to earlier answer) ─────────────
VISION_ID = "openai/clip-vit-large-patch14-336"
def load_vision_tower(device):
    vt   = CLIPVisionModel.from_pretrained(VISION_ID).to(device).eval()
    proc = CLIPProcessor.from_pretrained(VISION_ID)
    return vt, proc

@torch.no_grad()
def heatmap_rollout(pil_img, vt, proc, device):
    inputs  = proc(images=pil_img, return_tensors="pt").to(device)
    outs    = vt(**inputs, output_attentions=True)
    att_all = torch.stack(outs.attentions)[:, 0]          # [layers, heads, T, T]
    T = att_all.shape[-1]
    eye = torch.eye(T, device=device)
    rollout = eye.clone()
    for A in att_all.mean(1):              # average over heads layer‑by‑layer
        A = (A + eye) / (A + eye).sum(-1, keepdim=True)
        rollout = A @ rollout
    heat = rollout[0, 1:].view(24, 24).cpu().numpy()      # CLS row
    heat = Image.fromarray(heat).resize(pil_img.size, Image.BILINEAR)
    heat = np.asarray(heat, np.float32)
    heat = np.clip((heat - heat.min()) / (heat.max() + 1e-8), 0, 1)
    return heat

JET = cm.get_cmap("inferno")
def overlay(frame_bgr, heat, alpha=0.5):
    heat_rgba = (JET(heat) * 255).astype(np.uint8)
    heat_bgr  = cv2.cvtColor(cv2.cvtColor(heat_rgba, cv2.COLOR_RGBA2RGB),
                             cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(frame_bgr, 1 - alpha, heat_bgr, alpha, 0)

# ─── (3) caption helper using VideoLLaMA‑2 ‑ 7B‑16F ────────────────
def generate_caption(video_path, device):
    from videollama2 import model_init, mm_infer
    from videollama2.utils import disable_torch_init
    disable_torch_init()

    MODEL = "DAMO-NLP-SG/VideoLLaMA2-7B-16F"
    print(f"[CAP] loading {MODEL} on {device} …")
    model, processor, tokenizer = model_init(
        MODEL, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device, attn_implementation="eager"
    )
    model.eval()
    video_tensor = processor["video"](video_path)
    caption = mm_infer(video_tensor,
                       "Describe the video.",
                       model=model, tokenizer=tokenizer,
                       modal="video", do_sample=False).strip()
    return caption

# ─── (4) main CLI ──────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="input video (mp4/avi/…)")

    # heat‑map flags
    ap.add_argument("--out", default="overlay.mp4", help="heat‑map video")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--alpha",  type=float, default=0.5, help="heat‑map opacity")

    # caption flags
    ap.add_argument("--caption-out", default="caption.txt")
    ap.add_argument("--caption-device", choices=["cuda", "cpu"], default="cuda",
                    help="where to run the 7 B captioner")

    args = ap.parse_args()

    # (a) HEAT‑MAP VIDEO SECTION -------------------------------------------------
    heat_dev = "cuda" if torch.cuda.is_available() else "cpu"
    vt, proc = load_vision_tower(heat_dev)

    cap  = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"[ERR] cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    print(f"[RUN] heat‑mapping {args.video} → {args.out}")
    start = time.time()
    f_idx, mapped = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if f_idx % args.stride == 0:
            pil   = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            heat  = heatmap_rollout(pil, vt, proc, heat_dev)
            frame = overlay(frame, heat, alpha=args.alpha)
            mapped += 1
        out.write(frame)
        f_idx += 1
        if f_idx % 100 == 0:
            print(f"  {f_idx} frames ({mapped} mapped) – "
                  f"{(time.time()-start):.1f}s", end="\r")

    cap.release(); out.release()
    print(f"\n[✔] wrote {args.out}  ({mapped} heat‑mapped frames)")

    # (b) CAPTION SECTION --------------------------------------------------------
    caption = generate_caption(args.video, args.caption_device)
    os.makedirs(os.path.dirname(args.caption_out) or ".", exist_ok=True)
    with open(args.caption_out, "w", encoding="utf‑8") as f:
        f.write(caption + "\n")
    print(f"\n[CAPTION] → {args.caption_out}\n---\n{caption}\n---")

if __name__ == "__main__":
    main()
