#!/usr/bin/env python3
# videollama_attention_ok.py • 2025-06-01
# ✦ Spatial heat-maps + temporal curve + correct caption for VideoLLaMA-2 ✦

import os, sys, cv2, argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from transformers import CLIPVisionModel, CLIPImageProcessor
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

# ── kill FlashAttention-2 everywhere ────────────────────────────
import transformers.modeling_utils as _mu
_mu._check_and_enable_flash_attn_2 = lambda *_, **__: False
_mu.PreTrainedModel._check_and_enable_flash_attn_2 = staticmethod(lambda *_, **__: False)
os.environ.update({
    "PYTORCH_ATTENTION_IMPLEMENTATION": "eager",
    "HF_DISABLE_FLASH_ATTN_2": "1",
    "DISABLE_FLASH_ATTN_2": "1",
})

MODEL_NAME = "DAMO-NLP-SG/VideoLLaMA2-7B-16F"
VISION_ID  = "openai/clip-vit-large-patch14-336"
CLS_TOKEN  = 1                        # ViT CLS position

# ────────────────────────────────────────────────────────────────
def load_models(device="cuda"):
    vt = CLIPVisionModel.from_pretrained(VISION_ID).to(device).eval()
    vproc = CLIPImageProcessor.from_pretrained(VISION_ID)

    disable_torch_init()
    vlm, vprocessor, tok = model_init(
        MODEL_NAME, attn_implementation="eager",
        torch_dtype=torch.float16, device_map=device
    )
    vlm.eval()
    return vt, vproc, vlm, vprocessor, tok

# ────────────────────────────────────────────────────────────────
@torch.inference_mode()
def caption_video(video_path, vlm, vproc, tok, device="cuda"):
    vid = vproc["video"](video_path).to(device, dtype=torch.float16)
    caption = mm_infer(
        vid, "Describe the video in detail.",
        model=vlm, tokenizer=tok, modal="video",
        do_sample=False
    ).strip()
    return caption, vid  # keep the tensor; we need frame-count

# ────────────────────────────────────────────────────────────────
# @torch.inference_mode()
# def vit_rollout(pil_img: Image.Image, vt, vproc, device="cuda"):
#     """24×24 heat-map in [0,1]"""
#     inp  = vproc(images=pil_img, return_tensors="pt").to(device)
#     outs = vt(**inp, output_attentions=True)
#     A    = torch.stack(outs.attentions)[:, 0].mean(1)     # layers×tokens×tokens

#     eye = torch.eye(A.size(-1), device=device)
#     R = eye.clone()
#     for layer in A:
#         layer = (layer + eye)
#         layer /= layer.sum(-1, keepdim=True)
#         R = layer @ R
#     heat = R[0, CLS_TOKEN+1:].reshape(24,24).detach().cpu().numpy()
#     heat = cv2.GaussianBlur(heat, (0,0), 3)
#     heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    # return heat

@torch.inference_mode()
def vit_rollout(pil_img, vt, vproc, device="cuda"):
    inp  = vproc(images=pil_img, return_tensors="pt").to(device)
    outs = vt(**inp, output_attentions=True)

    A = torch.stack(outs.attentions)[:, 0].mean(1)         # layers × tokens × tokens
    eye = torch.eye(A.size(-1), device=device)
    R = eye
    for layer in A:
        layer = (layer + eye)
        layer /= layer.sum(-1, keepdim=True)
        R = layer @ R

    n_vis = R.size(-1) - 1          # drop CLS only
    side  = int(round(n_vis ** 0.5))
    heat  = R[0, 1:1+n_vis]         # keep visual tokens
    heat  = heat[: side*side].reshape(side, side).detach().cpu().numpy()

    heat = cv2.GaussianBlur(heat, (0,0), 3)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    return np.power(heat, .5)



# ────────────────────────────────────────────────────────────────
def temporal_from_heat(energies):
    arr = np.asarray(energies, dtype=np.float32)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) if len(arr) else arr

# ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="attention_frames")
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--no-rotate", action="store_true")
    ap.add_argument("--caption-file", default="caption.txt")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("❌  Need a CUDA GPU to run VideoLLaMA-2.")
    dev = "cuda"

    print("⏳  Loading towers …")
    vt, vproc, vlm, vp, tok = load_models(dev)

    # caption + keep original tensor for frame-count
    print("📝  Generating caption …")
    caption, vid_tensor = caption_video(args.video, vlm, vp, tok, dev)
    n_frames = vid_tensor.shape[1]
    print("    ➜", caption)

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    # frame loop
    jet = plt.colormaps.get_cmap("jet")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"❌  Cannot open {args.video}")

    energies = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if not args.no_rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        heat = vit_rollout(pil, vt, vproc, dev)
        heat = cv2.resize(heat, (frame.shape[1], frame.shape[0]), cv2.INTER_LINEAR)
        energies.append(heat.mean())                           # temporal signal

        h_rgba = (jet(heat) * 255).astype(np.uint8)
        h_bgr  = cv2.cvtColor(cv2.cvtColor(h_rgba, cv2.COLOR_RGBA2RGB), cv2.COLOR_RGB2BGR)
        overlay= cv2.addWeighted(frame, 1-args.alpha, h_bgr, args.alpha, 0)

        cv2.putText(overlay, f"frame {idx}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        y,x = np.unravel_index(np.argmax(heat), heat.shape)
        cv2.circle(overlay, (x,y), 12, (0,0,255), 3)
        cv2.imwrite(str(outdir/f"frame_{idx:04d}.png"), overlay)
        if idx % 10 == 0:
            print(f"   processed {idx}", end="\r")
        idx += 1
    cap.release()
    print(f"\n✅  {idx} frames saved → {outdir}")

    # temporal plot
    curve = temporal_from_heat(energies)
    plt.figure(figsize=(12,4))
    plt.plot(curve, 'o-'); plt.grid()
    plt.title("Temporal heat-map energy"); plt.xlabel("frame"); plt.ylabel("norm energy")
    plt.tight_layout(); plt.savefig(outdir/"temporal_attention.png"); plt.close()

    # write caption
    Path(args.caption_file).write_text(caption+"\n", encoding="utf-8")
    print(f"📝  Caption written to {args.caption_file}")

if __name__ == "__main__":
    main()
