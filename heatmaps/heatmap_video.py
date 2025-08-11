#!/usr/bin/env python3
"""
heatmap_video.py – overlay VideoLLaMA2 CLIP attention heat-maps.

Examples
--------

# png frames to a folder (default if --out ends with /)
python heatmap_video.py assets/my.mp4 -o outputs/frames/

# force png mode even if --out looks like .mp4
python heatmap_video.py assets/my.mp4 -o outputs/foo.mp4 --png

# mp4 output, GPU if available
python heatmap_video.py assets/my.mp4 -o outputs/overlay.mp4 --cuda
"""

# ───────────────────────────────────────────────────────────────────
# 0 ▸ disable Flash-Attn-2 everywhere
# ───────────────────────────────────────────────────────────────────
import transformers.modeling_utils as _mu
def _no(*_, **__): return False
_mu._check_and_enable_flash_attn_2 = _no
_mu.PreTrainedModel._check_and_enable_flash_attn_2 = staticmethod(_no)

import os, sys, cv2, argparse, numpy as np, torch
from tqdm import tqdm
from videollama2 import model_init
from videollama2.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPProcessor

os.environ["HF_DISABLE_FLASH_ATTN_2"]          = "1"
os.environ["PYTORCH_ATTENTION_IMPLEMENTATION"] = "eager"

# ───────────────────────────────────────────────────────────────────
# helpers
# ───────────────────────────────────────────────────────────────────
def normalise(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

@torch.no_grad()
def frame_heat(rgb, proc, clip, layers=(-1, -2, -3)):
    """return H×W float32 heat-map in [0,1]."""
    t   = proc(images=rgb, return_tensors="pt").to(next(clip.parameters()).device)
    att = clip(**t, output_attentions=True).attentions            # list[L]  (1,h,t,t)
    m   = torch.stack([att[l][0, :, 0, 1:].mean(0) for l in layers]).mean(0)
    h   = m.reshape(24, 24).cpu().numpy()
    h   = cv2.resize(h, (rgb.shape[1], rgb.shape[0]), cv2.INTER_LINEAR)  # up-sample
    h   = cv2.GaussianBlur(h, (0, 0), 7)                                   # blur
    p1, p99 = np.percentile(h, [1, 99])                                    # rescale
    h   = np.clip((h - p1) / (p99 - p1 + 1e-8), 0, 1)
    return h.astype(np.float32)

def overlay(rgb, heat, alpha=0.6):
    jet  = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    jet  = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    return (rgb * (1 - alpha) + jet * alpha).astype(np.uint8)

def locate_tower(model):
    for attr in ("vision_tower", "visual_tower"):
        t = getattr(model, attr, None)
        if t is not None:
            return t
    if hasattr(model, "model"):
        for attr in ("vision_tower", "visual_tower"):
            t = getattr(model.model, attr, None)
            if t is not None:
                return t
    return None

def unwrap_clip(tower):
    if isinstance(tower, (list, tuple)):
        tower = tower[0]
    clip = getattr(tower, "clip", None) or getattr(tower, "vision_tower", None)
    proc = getattr(tower, "image_processor", None) or getattr(tower, "clip_processor", None)
    return clip, proc

def is_meta(module: torch.nn.Module):
    return any(p.is_meta for p in module.parameters(True)) or \
           any(b.is_meta for b in module.buffers(True))

# ───────────────────────────────────────────────────────────────────
# main
# ───────────────────────────────────────────────────────────────────
def main():
    disable_torch_init()

    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("-o", "--out", default="outputs/overlay.mp4",
                    help="folder → PNGs | *.mp4 → video file")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--cuda", action="store_true", help="use GPU for CLIP")
    ap.add_argument("--png",  action="store_true", help="force PNG mode")
    args = ap.parse_args()

    # decide output modality
    save_png = args.png or not args.out.lower().endswith(".mp4")
    if save_png:
        out_dir = args.out.rstrip("/\\")
        os.makedirs(out_dir, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    # ─ load VideoLLaMA2 & CLIP backbone ─────────────────────────────
    print("· loading VideoLLaMA2 checkpoint …")
    vllm, proc, _ = model_init("DAMO-NLP-SG/VideoLLaMA2-7B-16F")

    tower = locate_tower(vllm)
    if tower is None:
        sys.exit("❌  cannot find vision tower")

    clip, clip_proc = unwrap_clip(tower)
    if clip is None or is_meta(clip):
        print("⚠️  tower off-loaded → fallback to openai/clip-vit-large-336")
        clip      = CLIPVisionModel.from_pretrained(
                        "openai/clip-vit-large-patch14-336",
                        torch_dtype=torch.float16 if device == "cuda" else torch.float32)
        clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

    clip = clip.to(device).eval()

    # ─ sample video frames ─────────────────────────────────────────
    cap   = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs  = np.linspace(0, total - 1, args.frames, dtype=int)

    frames = []
    print("· extracting attentions …")
    for i, idx in enumerate(tqdm(idxs)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if not ok:
            continue
        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        heat = frame_heat(rgb, clip_proc, clip)
        over = cv2.cvtColor(overlay(rgb, heat), cv2.COLOR_RGB2BGR)

        if save_png:
            cv2.imwrite(f"{out_dir}/frame_{i:03}.png", over)
        else:
            frames.append(over)
    cap.release()

    # ─ save outputs ────────────────────────────────────────────────
    if save_png:
        print(f"✅  {len(idxs)} PNGs saved → {out_dir}")
    else:
        h, w = frames[0].shape[:2]
        vw = cv2.VideoWriter(args.out,
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             4, (w, h))
        for f in frames:
            vw.write(f)
        vw.release()
        print(f"✅  MP4 saved → {args.out}")

if __name__ == "__main__":
    main()
