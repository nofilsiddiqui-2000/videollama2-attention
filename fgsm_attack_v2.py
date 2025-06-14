#!/usr/bin/env python3
# fgsm_video_v3.py  –  memory-safe FGSM + attention rollout
# ---------------------------------------------------------
# usage example:
#   python fgsm_video_v3.py \
#       --input_video test/testvideo3.mp4 \
#       --output_dir   results_fgsm \
#       --epsilon      0.03 \
#       --batch_size   4 \
#       --save_curve \
#       --output_caption results_fgsm/adv_caption.txt
# ---------------------------------------------------------

# ── 0. Environment knobs *before* torch/transformers are imported ─────────────
import os, warnings, argparse, math, cv2, pathlib, time
os.environ.setdefault("HF_HUB_DISABLE_MEMMAP",   "1")      # <-- Fix the mmap crash
os.environ.setdefault("MPLCONFIGDIR",            "/tmp")   # matplotlib temp
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ── 1. Std-lib & deps ─────────────────────────────────────────────────────────
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
from transformers import (CLIPImageProcessor,
                          CLIPVisionModelWithProjection)

# optional: VideoLLaMA-2
try:
    from videollama2 import model_init, mm_infer
    from videollama2.utils import disable_torch_init
    HAVE_VLLAMA = True
except ImportError:
    HAVE_VLLAMA = False

# ── 2. Attention-rollout helper (Abnar & Zuidema 2020) ────────────────────────
def rollout(attn_list):
    R = None
    for A in attn_list:                 # A: (heads, T, T)
        A = A.mean(0) + torch.eye(A.size(-1), device=A.device)
        A = A / A.sum(-1, keepdim=True)
        R = A if R is None else A @ R
    mask = R[0, 1:]                     # CLS→patches
    return mask                         # (576,)

# ── 3. Frame sampler ─────────────────────────────────────────────────────────
def sample_16(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise IOError(f"cannot open {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    idxs = np.linspace(0, n-1, 16, dtype=int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok: frames.append(f)
    cap.release()
    if not frames: raise RuntimeError("video empty?")
    return frames, fps

# ── 4. Main ───────────────────────────────────────────────────────────────────
def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--input_video",   required=True)
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--epsilon",       type=float, required=True)
    p.add_argument("--batch_size",    type=int,   default=4)
    p.add_argument("--save_curve",    action="store_true")
    p.add_argument("--output_caption",required=True)
    return p.parse_args()

def main():
    args = cli()
    out_root   = pathlib.Path(args.output_dir).resolve()
    clean_dir  = out_root / "frames_clean"
    adv_dir    = out_root / "frames_adv"
    clean_dir.mkdir(parents=True, exist_ok=True)
    adv_dir.mkdir(  parents=True, exist_ok=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4-A) load CLIP with streaming + half precision
    clip_id   = "openai/clip-vit-large-patch14-336"
    processor = CLIPImageProcessor.from_pretrained(clip_id)
    clip = CLIPVisionModelWithProjection.from_pretrained(
        clip_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True       # <-- stream weights, no mmap
    ).to(dev).eval()

    mean = torch.tensor(processor.image_mean, device=dev).view(1,3,1,1)
    std  = torch.tensor(processor.image_std,  device=dev).view(1,3,1,1)
    min_norm, max_norm = (0-mean)/std, (1-mean)/std

    # 4-B) sample frames
    raw_frames, fps = sample_16(args.input_video)
    frames_pil = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                  for f in raw_frames]

    # Lists to gather stats
    cos_list, energy_curve = [], []

    # 4-C) process batch-by-batch (clean → overlay; FGSM → overlay)
    print("▶ processing frames …")
    for base in range(0, 16, args.batch_size):
        batch_pil = frames_pil[base:base+args.batch_size]

        # ----- forward clean --------------------------------------------------
        inp = processor(images=batch_pil, return_tensors="pt").to(dev)
        with torch.no_grad():
            outs = clip(pixel_values=inp.pixel_values, output_attentions=True)
        emb_clean = outs.image_embeds          # (B, dim)
        attn      = outs.attentions            # tuple[24] of (B,H,T,T)

        for j in range(inp.pixel_values.size(0)):
            idx = base + j
            frame_rgb = raw_frames[idx][:,:,::-1]    # BGR→RGB quick flip
            mask = rollout([l[j] for l in attn]).cpu()
            side = int(math.sqrt(mask.numel()))
            heat = mask.reshape(side, side).numpy()
            heat = (heat-heat.min())/(heat.max()-heat.min()+1e-8)
            energy_curve.append(float((heat**2).sum()))

            # overlay & save (clean)
            h,w,_ = frame_rgb.shape
            heat_col = cv2.applyColorMap(
                cv2.resize(np.uint8(255*heat), (w,h)),
                cv2.COLORMAP_JET
            )
            overlay = cv2.addWeighted(frame_rgb, .6, heat_col, .4, 0)
            cv2.imwrite(str(clean_dir/f"frame_{idx:02d}.png"), overlay[:,:,::-1])

            # ----- FGSM on *this* frame --------------------------------------
            x = inp.pixel_values[j:j+1].clone().detach().requires_grad_(True)
            f_c = emb_clean[j:j+1]
            f_adv = clip(pixel_values=x).image_embeds
            loss = F.cosine_similarity(f_c, f_adv).mean()  # minimise similarity
            loss.backward()
            x_adv = torch.clamp(x - args.epsilon*x.grad.sign(),
                                min_norm, max_norm).detach()

            # embedding & similarity after attack
            with torch.no_grad():
                emb_adv = clip(pixel_values=x_adv).image_embeds
            cos = F.cosine_similarity(f_c, emb_adv).item()
            cos_list.append(cos)

            # save adversarial frame (+ overlay for visual check)
            adv_rgb = (x_adv*std + mean).clamp(0,1)[0].permute(1,2,0)
            adv_np  = (adv_rgb.cpu().numpy()*255).astype(np.uint8)
            # reuse same heat to compare attention drift visually
            adv_overlay = cv2.addWeighted(
                adv_np, .6, heat_col, .4, 0
            )
            cv2.imwrite(str(adv_dir/f"frame_{idx:02d}.png"), adv_overlay[:,:,::-1])

            # free registers
            del x, x_adv, f_c, f_adv, emb_adv, heat_col, overlay
            torch.cuda.empty_cache()

    print(f"[FGSM] average cosine similarity = {np.mean(cos_list):.4f}")

    # 4-D) rebuild adversarial video
    h,w,_ = raw_frames[0].shape
    v_out = cv2.VideoWriter(str(out_root/"adversarial.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps, (w,h))
    for fname in sorted(os.listdir(adv_dir)):
        f = cv2.imread(str(adv_dir/fname))
        v_out.write(f)
    v_out.release()

    # 4-E) optional energy curve
    if args.save_curve:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,3))
        plt.plot(energy_curve, marker="o"); plt.grid()
        plt.title("Temporal attention energy"); plt.xlabel("frame")
        plt.tight_layout()
        plt.savefig(out_root/"attention_energy_curve.png")
        plt.close()

    # 4-F) caption with VideoLLaMA-2, if available
    if HAVE_VLLAMA:
        disable_torch_init()
        vllm, vproc, vtok = model_init("DAMO-NLP-SG/VideoLLaMA2-7B-16F",
                                       torch_dtype=torch.float16,
                                       device_map=dev,
                                       attn_implementation="eager")
        vid_t = vproc["video"](str(out_root/"adversarial.mp4")).to(torch.float16).to(dev)
        caption = mm_infer(vid_t, "Describe the video.",
                           model=vllm, tokenizer=vtok,
                           modal="video", do_sample=False).strip()
        with open(args.output_caption, "w") as f:
            f.write(caption + "\n")
        print("📝 caption:", caption)
    else:
        print("⚠️  VideoLLaMA2 not installed; caption skipped.")

if __name__ == "__main__":
    main()
