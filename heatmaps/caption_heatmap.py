#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────────
# Heat-map + Caption pipeline for VideoLLaMA-2 (7B-16F) on GPU only
# ────────────────────────────────────────────────────────────────────
#  • Saves either individual frames with spatial attention overlays
#    or a single MP4 video
#  • Writes a one-line caption to caption.txt
#
# Usage (for image frames):
#   python videollama2_heatmap_and_caption_gpu.py input.mp4 --stride 3 --alpha 0.4 --save-frames
#
# Usage (for video):
#   python videollama2_heatmap_and_caption_gpu.py input.mp4 --stride 3 --alpha 0.4
# ────────────────────────────────────────────────────────────────────

import os, sys, cv2, argparse, time
import numpy as np
from PIL import Image
import torch
import transformers.modeling_utils as _mu
from transformers import CLIPVisionModel, CLIPProcessor
import matplotlib.cm as cm

def _disable_fa2(*_, **__):
    return False

# Disable Flash-Attention 2
_mu._check_and_enable_flash_attn_2 = _disable_fa2
_mu.PreTrainedModel._check_and_enable_flash_attn_2 = staticmethod(_disable_fa2)
os.environ["PYTORCH_ATTENTION_IMPLEMENTATION"] = "eager"
os.environ["HF_DISABLE_FLASH_ATTN_2"] = "1"
os.environ["DISABLE_FLASH_ATTN_2"] = "1"

VISION_ID = "openai/clip-vit-large-patch14-336"

def load_vision_tower(device: str):
    vt = CLIPVisionModel.from_pretrained(VISION_ID).to(device).eval()
    proc = CLIPProcessor.from_pretrained(VISION_ID)
    return vt, proc

@torch.no_grad()
def heatmap_rollout(pil_img: Image.Image, vt: CLIPVisionModel, proc: CLIPProcessor, device: str) -> np.ndarray:
    inputs = proc(images=pil_img, return_tensors="pt").to(device)
    outs = vt(**inputs, output_attentions=True)         # output_attention set to True
    att_all = torch.stack(outs.attentions)[:, 0]
    T = att_all.shape[-1]
    eye = torch.eye(T, device=device)
    R = eye.clone()
    for layer_attn in att_all.mean(dim=1):
        A = layer_attn + eye
        A = A / A.sum(dim=-1, keepdim=True)
        R = A @ R
    cls_attn = R[0, 1:].view(24, 24).cpu().numpy()
    heat = Image.fromarray(cls_attn).resize(pil_img.size, Image.BILINEAR)
    heat = np.asarray(heat, dtype=np.float32)
    heat = np.clip((heat - heat.min()) / (heat.max() + 1e-8), 0.0, 1.0)
    return heat

JET = cm.get_cmap("inferno")

def overlay(frame_bgr: np.ndarray, heat: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    heat_rgba = (JET(heat) * 255).astype(np.uint8)
    heat_rgb = cv2.cvtColor(heat_rgba, cv2.COLOR_RGBA2RGB)
    heat_bgr = cv2.cvtColor(heat_rgb, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(frame_bgr, 1 - alpha, heat_bgr, alpha, 0)

def generate_caption(video_path: str, device: str) -> str:
    from videollama2 import model_init, mm_infer
    from videollama2.utils import disable_torch_init

    disable_torch_init()
    MODEL_NAME = "DAMO-NLP-SG/VideoLLaMA2-7B-16F"
    print(f"[CAP] Loading {MODEL_NAME} on {device} …")

    model, processor, tokenizer = model_init(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=device,
        attn_implementation="eager"
    )
    model.eval()
    vid_tensor = processor["video"](video_path).to(torch.float16).to(device)
    caption = mm_infer(
        vid_tensor,
        "Describe the video.",
        model=model,
        tokenizer=tokenizer,
        modal="video",
        do_sample=False
    ).strip()
    return caption

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="input video path (mp4/avi/… )")
    ap.add_argument("--out", default="overlay.mp4", help="output video or frame directory")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame (≥1)")
    ap.add_argument("--alpha", type=float, default=0.5, help="heat-map opacity (0–1)")
    ap.add_argument("--caption-out", default="caption.txt", help="file to write the generated caption")
    ap.add_argument("--save-frames", action="store_true", help="save individual heatmap frames as PNGs")

    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("[ERR] No CUDA GPU detected. This script is GPU-only.")

    heat_dev = "cuda"
    vt, proc = load_vision_tower(heat_dev)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"[ERR] Cannot open {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_dir = os.path.splitext(args.out)[0]

    if args.save_frames:
        os.makedirs(output_dir, exist_ok=True)
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, fps, (W, H))

    print(f"[RUN] Processing {args.video} → {'frames in ' + output_dir if args.save_frames else args.out}")
    t0 = time.time()
    frame_idx, mapped = 0, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.rotate(frame, cv2.ROTATE_180)  # flip upside-down

        # if frame_idx % args.stride == 0:
        #     pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        #     heat = heatmap_rollout(pil_frame, vt, proc, heat_dev)
        #     frame = overlay(frame, heat, alpha=args.alpha)
        #     mapped += 1

        if frame_idx % args.stride == 0:
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            heat = heatmap_rollout(pil_frame, vt, proc, heat_dev)

    # ⬇️ Enhance contrast of heatmap (optional)
            heat = np.power(heat, 0.5)  # exaggerate strong attention

    # ⬇️ Draw red circle on max attention point
            max_loc = np.unravel_index(np.argmax(heat), heat.shape)
        cv2.circle(frame, (max_loc[1], max_loc[0]), 15, (0, 0, 255), 3)

    # ⬇️ Blend heatmap on top
        frame = overlay(frame, heat, alpha=args.alpha)
        mapped += 1


        if args.save_frames:
            out_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
            cv2.imwrite(out_path, frame)
        else:
            writer.write(frame)

        frame_idx += 1
        if frame_idx % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {frame_idx} frames ({mapped} mapped) – {elapsed:.1f}s", end="\r")

    cap.release()
    if not args.save_frames:
        writer.release()

    print(f"\n[✔] Processed {frame_idx} frames ({mapped} with heatmaps)")

    caption = generate_caption(args.video, "cuda")
    os.makedirs(os.path.dirname(args.caption_out) or ".", exist_ok=True)
    with open(args.caption_out, "w", encoding="utf-8") as f:
        f.write(caption + "\n")

    print(f"\n[CAPTION] → {args.caption_out}\n---\n{caption}\n---\n")

if __name__ == "__main__":
    main()



# python caption_heatmap.py .\assets\sample_video.mp4 --save-frames --stride 1 --alpha 0.8
