#!/usr/bin/env python3
# adversarial_clip_vllama.py
"""
FGSM + ViT-L/14-336 attention-rollout  +  VideoLLaMA-2 caption
──────────────────────────────────────────────────────────────
Run (example):

python adversarial_clip_vllama.py \
    --input_video test/testvideo3.mp4 \
    --output_dir  results_fgsm \
    --epsilon     0.03 \
    --batch_size  4 \
    --save_curve \
    --output_caption results_fgsm/adv_caption.txt
"""

# ───────────────────────── Imports & global switches ─────────────────────────
import os, math, argparse, warnings, pathlib
import cv2, numpy as np
import torch, torch.nn.functional as F
from PIL import Image

# Hugging-Face ↯  ─ disable fragile mmap & flash-attn-2 everywhere
os.environ.setdefault("HF_HUB_DISABLE_MEMMAP", "1")          # no mmap-fail
os.environ.setdefault("PYTORCH_ATTENTION_IMPLEMENTATION", "eager")
os.environ.setdefault("HF_DISABLE_FLASH_ATTN_2", "1")
os.environ.setdefault("DISABLE_FLASH_ATTN_2", "1")

from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)

# matplotlib sometimes tries to write to $HOME → quota errors
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ─────────────────────────── Helper functions ───────────────────────────────
def ensure_dir(p) -> pathlib.Path:
    p = pathlib.Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def sample_frames(video_path: str, n=16):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    idxs  = np.linspace(0, total-1, n, dtype=int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError("No frames extracted.")
    return frames, fps

def rollout(attn_list):
    """
    attn_list : tuple[24] each (1, heads, 577, 577)  (ViT-L/14-336)
    Returns a (24×24) numpy mask in [0,1].
    """
    out = None
    for A in attn_list:                       # average heads
        A = A[0].mean(0)                     # (tokens,tokens)
        A = A + torch.eye(A.size(0), device=A.device)  # add residual
        A = A / A.sum(dim=-1, keepdim=True)
        out = A if out is None else A @ out
    cls2patch = out[0, 1:577]                # drop CLS→CLS
    mask = (cls2patch - cls2patch.min()) / (cls2patch.max()-cls2patch.min()+1e-8)
    return mask.reshape(24,24).cpu().numpy()

def overlay_heat(frame_rgb, mask):
    h,w,_ = frame_rgb.shape
    heat  = cv2.resize(mask, (w,h), cv2.INTER_LINEAR)
    heat  = np.uint8(255 * heat)
    jet   = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    jet   = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(frame_rgb, 0.6, jet, 0.4, 0)

# ───────────────────────────── Main pipeline ────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_video", required=True)
    ap.add_argument("--output_dir",  required=True)
    ap.add_argument("--epsilon",     type=float, required=True)
    ap.add_argument("--batch_size",  type=int,   default=4)
    ap.add_argument("--save_curve",  action="store_true")
    ap.add_argument("--output_caption", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_root   = ensure_dir(args.output_dir)
    clean_dir  = ensure_dir(out_root / "frames_clean")
    adv_dir    = ensure_dir(out_root / "frames_adv")

    # ─── 1 ▸ load CLIP ViT-L/14-336 (fp16, low-mem) ───────────────────
    proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    clip = CLIPVisionModelWithProjection.from_pretrained(
        "openai/clip-vit-large-patch14-336",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    mean = torch.tensor(proc.image_mean, dtype=torch.float16, device=device).view(1,3,1,1)
    std  = torch.tensor(proc.image_std,  dtype=torch.float16, device=device).view(1,3,1,1)
    norm_min, norm_max = (0-mean)/std, (1-mean)/std

    # ─── 2 ▸ sample frames ────────────────────────────────────────────
    raw_frames, fps = sample_frames(args.input_video, 16)
    pil_frames = [Image.fromarray(f) for f in raw_frames]

    # ─── 3 ▸ clean pass → embeddings & heat-maps ──────────────────────
    clean_embs, clean_rollouts, energy_curve = [], [], []
    for i in range(0, 16, args.batch_size):
        batch = pil_frames[i:i+args.batch_size]
        inp   = proc(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            out = clip(pixel_values=inp.pixel_values.half(), output_attentions=True)
        clean_embs.append(out.image_embeds.cpu())
        for b in range(len(batch)):
            mask = rollout([lyr[b:b+1] for lyr in out.attentions])
            clean_rollouts.append(mask)
    clean_embs = torch.cat(clean_embs, 0)   # (16,dim)

    # save clean overlays & collect energy
    for idx, (f, m) in enumerate(zip(raw_frames, clean_rollouts)):
        over = overlay_heat(f, m)
        cv2.imwrite(str(clean_dir/f"frame_{idx:04d}.png"),
                    cv2.cvtColor(over, cv2.COLOR_RGB2BGR))
        energy_curve.append(float((m**2).sum()))

    # ─── 4 ▸ FGSM (untargeted) in CLIP-norm space ─────────────────────
    cos_sims, adv_frames, adv_rollouts = [], [], []
    eps = args.epsilon
    for idx, pil in enumerate(pil_frames):
        inp = proc(images=pil, return_tensors="pt").to(device)
        x   = inp.pixel_values.half().clone().detach().requires_grad_(True)  # (1,3,336,336)

        # clean embedding (already have, but keep on device)
        f_clean = clean_embs[idx:idx+1].to(device)

        # forward + loss
        feat = clip(pixel_values=x, output_attentions=False).image_embeds
        loss = F.cosine_similarity(f_clean, feat, dim=-1).mean()            # minimise similarity
        loss.backward()

        # FGSM step
        x_adv = (x - eps * x.grad.sign()).clamp(norm_min, norm_max).detach()

        # metric
        with torch.no_grad():
            feat_adv = clip(pixel_values=x_adv, output_attentions=True)
        cos = F.cosine_similarity(f_clean, feat_adv.image_embeds, dim=-1).item()
        cos_sims.append(cos)

        # save adversarial frame
        un = (x_adv*std + mean).clamp(0,1)[0].permute(1,2,0).cpu().numpy()
        img_adv = (un*255).astype(np.uint8)
        adv_frames.append(img_adv)

        # rollout on adversarial
        mask_adv = rollout([lyr[0:1] for lyr in feat_adv.attentions])
        adv_rollouts.append(mask_adv)
        over = overlay_heat(img_adv, mask_adv)
        cv2.imwrite(str(adv_dir/f"frame_{idx:04d}.png"),
                    cv2.cvtColor(over, cv2.COLOR_RGB2BGR))

    # log cosine similarities
    with open(out_root/"cosine_similarity.txt", "w") as fh:
        for v in cos_sims:
            fh.write(f"{v:.6f}\n")
    print(f"[FGSM] mean cosine(clean,adv) = {np.mean(cos_sims):.4f}")

    # ─── 5 ▸ write adversarial MP4 ────────────────────────────────────
    h,w,_ = adv_frames[0].shape
    vout = cv2.VideoWriter(str(out_root/"adversarial.mp4"),
                           cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    for f in adv_frames:
        vout.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vout.release()

    # ─── 6 ▸ optional temporal-energy curve ───────────────────────────
    if args.save_curve:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,3))
        plt.plot(energy_curve, marker="o")
        plt.title("Temporal attention energy (clean)")
        plt.xlabel("frame"); plt.ylabel("∑ heat²"); plt.grid()
        plt.tight_layout()
        plt.savefig(out_root/"attention_energy_curve.png")
        plt.close()

    # ─── 7 ▸ VideoLLaMA-2 caption on adversarial clip ────────────────
    try:
        from videollama2 import model_init, mm_infer, disable_torch_init
    except ImportError:
        print("⚠️  VideoLLaMA2 not installed – skipping caption.")
        return

    disable_torch_init()
    vlm, vproc, vtok = model_init(
        "DAMO-NLP-SG/VideoLLaMA2-7B-16F",
        torch_dtype=torch.float16,
        device_map=device,
        attn_implementation="eager",
    )
    vlm.eval()
    vid_tensor = vproc["video"](str(out_root/"adversarial.mp4")).to(torch.float16).to(device)
    caption = mm_infer(vid_tensor, "Describe the video.", model=vlm,
                       tokenizer=vtok, modal="video", do_sample=False).strip()
    with open(args.output_caption, "w", encoding="utf-8") as f:
        f.write(caption+"\n")
    print("📝 Caption written:", caption)

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
