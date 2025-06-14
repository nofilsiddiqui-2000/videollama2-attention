#!/usr/bin/env python3
"""
FGSM + ViT-attention rollout on video frames (CLIP ViT-L/14-336) + VideoLLaMA caption
––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
1. 16 frames are sampled uniformly from the input video.
2. Untargeted FGSM is applied in *processor space* (after CLIP’s own resize-crop-norm),
   so the tensor has the expected 577 tokens (1 CLS + 24×24 patches).
3. For every adversarial frame:
      • Attention-rollout heat-map is computed and over-laid.
      • Mean attention “energy” is stored to draw a temporal curve (optional).
4. All overlays are saved as PNGs **inside `<output_dir>/frames_adv/`**.
5. An adversarial video (`adversarial.mp4`) is written for quick viewing.
6. That video is captioned with **VideoLLaMA-2 (7B-16F)** and the caption is written
   to `--output_caption`.
"""

import os, cv2, math, argparse, pathlib, warnings, matplotlib
import numpy as np
import torch, torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

# ──────────────────────────── helpers ────────────────────────────
def ensure_dir(p):
    p = pathlib.Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def sample_frames(path, num=16):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open {path}")
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, tot-1, num, dtype=int)
    frames, fps = [], cap.get(cv2.CAP_PROP_FPS) or 25
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok: frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError("No frames extracted")
    return frames, fps

def fgsm_batch(model, px_norm, eps):
    """
    px_norm : (B,3,336,336) **requires_grad=True** in CLIP-norm space.
    Untargeted FGSM that *repels* the features from their clean version.
    """
    with torch.no_grad():
        feat_clean = F.normalize(model.get_image_features(px_norm), dim=-1)

    feat_adv = model.get_image_features(px_norm)
    loss = -F.cosine_similarity(feat_adv, feat_clean, dim=-1).mean()  # push away
    loss.backward()
    adv = px_norm + eps * px_norm.grad.sign()
    return adv.detach()

def rollout(attn_list):
    """
    Attention rollout (Abnar & Zuidema 2020) for a single image batch =1.
    attn_list: tuple[24] with shape (1, heads, 577, 577)
    Returns mask as (24,24) ndarray in [0,1].
    """
    out = None
    for A in attn_list:                # average heads
        A = A[0].mean(0)               # (seq,seq)
        A = A + torch.eye(A.size(0), device=A.device)
        A = A / A.sum(dim=-1, keepdim=True)
        out = A if out is None else A @ out
    cls2patch = out[0,1:]              # drop CLS→CLS
    mask = cls2patch[:576].reshape(24,24)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask.cpu().numpy()

def overlay_heat(frame_rgb, mask):
    h,w,_ = frame_rgb.shape
    heat  = cv2.resize(mask, (w,h), cv2.INTER_LINEAR)
    heat  = np.uint8(255*heat)
    jet   = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    jet   = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(frame_rgb, 0.6, jet, 0.4, 0)

# ─────────────────────────── main ────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_video",   required=True)
    ap.add_argument("--output_dir",    required=True)
    ap.add_argument("--epsilon",       type=float, default=0.03)
    ap.add_argument("--batch_size",    type=int,   default=4)
    ap.add_argument("--save_curve",    action="store_true")
    ap.add_argument("--output_caption",required=True)
    args = ap.parse_args()

    # make Matplotlib happy on quota-restricted systems
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

    out_root  = ensure_dir(args.output_dir)
    out_frames= ensure_dir(out_root / "frames_adv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. CLIP model
    clip_id = "openai/clip-vit-large-patch14-336"
    processor = CLIPProcessor.from_pretrained(clip_id)
    model = CLIPModel.from_pretrained(clip_id,
                                      torch_dtype=torch.float16).to(device).eval()

    mean = torch.tensor(processor.image_processor.image_mean,
                        dtype=torch.float16, device=device).view(1,3,1,1)
    std  = torch.tensor(processor.image_processor.image_std,
                        dtype=torch.float16, device=device).view(1,3,1,1)
    min_norm, max_norm = (0-mean)/std, (1-mean)/std

    # 2. Load & preprocess frames
    raw_frames, fps = sample_frames(args.input_video, 16)
    adv_frames = []        # RGB uint8 outputs
    energy_curve = []
    cos_sims = []

    for i in range(0, len(raw_frames), args.batch_size):
        batch = raw_frames[i:i+args.batch_size]

        # processor handles resize-crop-norm to 3×336×336
        inputs = processor(images=batch, return_tensors="pt").to(device)
        x_norm = inputs.pixel_values.half().clone().detach().requires_grad_(True)

        # FGSM
        adv_norm = fgsm_batch(model, x_norm, args.epsilon)
        adv_norm = torch.max(torch.min(adv_norm, max_norm), min_norm)

        # cosine similarity metric
        with torch.no_grad():
            feat_clean = F.normalize(model.get_image_features(x_norm), dim=-1)
            feat_adv   = F.normalize(model.get_image_features(adv_norm), dim=-1)
        cos_sims += F.cosine_similarity(feat_clean, feat_adv, dim=-1).tolist()

        # un-norm → RGB uint8
        adv_rgb = (adv_norm*std + mean).clamp(0,1).cpu()
        adv_rgb = (adv_rgb.permute(0,2,3,1).numpy()*255).astype(np.uint8)

        # attention & overlays
        with torch.no_grad():
            attn = model.vision_model(pixel_values=adv_norm, output_attentions=True).attentions
        for j, img in enumerate(adv_rgb):
            mask = rollout(attn)
            overlay = overlay_heat(img, mask)
            idx = i+j
            cv2.imwrite(str(out_frames / f"frame_{idx:04d}.png"),
                        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            adv_frames.append(img)
            energy_curve.append(mask.mean())

    print(f"[FGSM] avg cosine(orig vs adv) = {np.mean(cos_sims):.4f}")

    # 3. Write adversarial video
    h,w,_ = adv_frames[0].shape
    vout  = cv2.VideoWriter(str(out_root/"adversarial.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    for f in adv_frames:
        vout.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vout.release()

    # 4. Temporal curve
    if args.save_curve:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,3))
        plt.plot(energy_curve, marker="o")
        plt.title("Temporal attention (mean energy)")
        plt.xlabel("frame"); plt.ylabel("mean mask value"); plt.grid()
        plt.tight_layout()
        plt.savefig(out_root/"attention_curve.png")
        plt.close()

    # 5. Caption with VideoLLaMA-2
    try:
        from videollama2 import model_init, mm_infer, disable_torch_init
    except ImportError as e:
        print("⚠️  VideoLLaMA2 not installed – skipping caption."); return
    disable_torch_init()
    vlm, vproc, vtok = model_init("DAMO-NLP-SG/VideoLLaMA2-7B-16F",
                                  torch_dtype=torch.float16,
                                  device_map=device,
                                  attn_implementation="eager")
    vid_tensor = vproc["video"](str(out_root/"adversarial.mp4")).to(torch.float16).to(device)
    caption = mm_infer(vid_tensor, "Describe the video.",
                       model=vlm, tokenizer=vtok,
                       modal="video", do_sample=False).strip()
    with open(args.output_caption, "w", encoding="utf-8") as f:
        f.write(caption + "\n")
    print("📝 caption:", caption)

if __name__ == "__main__":
    main()
