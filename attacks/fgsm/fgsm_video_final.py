#!/usr/bin/env python3
# fgsm_video_final.py  –  low-RAM FGSM + ViT rollout + VideoLLaMA-2
# ---------------------------------------------------------------
# run e.g.
#   python fgsm_video_final.py \
#       --input_video test/testvideo3.mp4 \
#       --output_dir  results_fgsm \
#       --epsilon     0.03 \
#       --batch_size  4 \
#       --save_curve \
#       --output_caption results_fgsm/adv_caption.txt
# ---------------------------------------------------------------

# -- 0 ▸ env before torch ------------------------------------------------------
import os, warnings, argparse, math, cv2, pathlib
os.environ.setdefault("HF_HUB_DISABLE_MEMMAP", "1")   # stop mmap
os.environ.setdefault("MPLCONFIGDIR", "/tmp")         # safe Matplotlib cache
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# -- 1 ▸ std-lib & deps --------------------------------------------------------
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)

# optional VideoLLaMA-2
try:
    from videollama2 import model_init, mm_infer
    from videollama2.utils import disable_torch_init
    HAVE_VLLAMA = True
except ImportError:
    HAVE_VLLAMA = False

# -- 2 ▸ helpers ---------------------------------------------------------------
def rollout(attn_list):
    """Abnar & Zuidema attention rollout → (576,) tensor."""
    R = None
    for A in attn_list:                                # (heads,T,T)
        A = A.mean(0) + torch.eye(A.size(-1), device=A.device)
        A = A / A.sum(-1, keepdim=True)
        R = A if R is None else A @ R
    return R[0, 1:]                                    # CLS→patches

def sample_frames(path, n=16):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    idxs  = np.linspace(0, total - 1, n, dtype=int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frm = cap.read()
        if ok:
            frames.append(frm)
    cap.release()
    if not frames:
        raise RuntimeError("failed to extract frames")
    return frames, fps

# -- 3 ▸ CLI -------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_video",   required=True)
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--epsilon",       type=float, required=True)
    p.add_argument("--batch_size",    type=int, default=4)
    p.add_argument("--save_curve",    action="store_true")
    p.add_argument("--output_caption",required=True)
    return p.parse_args()

# -- 4 ▸ main ------------------------------------------------------------------
def main() -> None:
    args = get_args()
    out_root = pathlib.Path(args.output_dir).resolve()
    clean_dir = out_root / "frames_clean"
    adv_dir   = out_root / "frames_adv"
    clean_dir.mkdir(parents=True, exist_ok=True)
    adv_dir.mkdir(parents=True,   exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 4-A) lightweight CLIP
    clip_name  = "openai/clip-vit-large-patch14-336"
    processor  = CLIPImageProcessor.from_pretrained(clip_name)
    clip = CLIPVisionModelWithProjection.from_pretrained(
        clip_name,
        torch_dtype=torch.float16,
        device_map={"": device},           # load direct to GPU
        low_cpu_mem_usage=True,            # stream weights
        use_safetensors=True               # <- avoids .bin mmap
    ).eval()

    mean = torch.tensor(processor.image_mean, device=device).view(1,3,1,1)
    std  = torch.tensor(processor.image_std,  device=device).view(1,3,1,1)
    norm_min, norm_max = (0-mean)/std, (1-mean)/std

    # 4-B) frames
    frames_bgr, fps = sample_frames(args.input_video, 16)
    frames_rgb = [f[:, :, ::-1] for f in frames_bgr]   # BGR→RGB
    frames_pil = [Image.fromarray(f) for f in frames_rgb]

    cos_vals, energy_curve = [], []

    # 4-C) process in mini-batches
    for base in range(0, 16, args.batch_size):
        batch_pil = frames_pil[base:base+args.batch_size]

        inp = processor(images=batch_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            out = clip(pixel_values=inp.pixel_values, output_attentions=True)
        emb_clean = out.image_embeds
        attn      = out.attentions        # 24 tuples of (B,H,T,T)

        # per-image loop
        for j in range(inp.pixel_values.size(0)):
            idx = base + j
            frame = frames_rgb[idx]
            mask  = rollout([l[j] for l in attn]).cpu()
            side  = int(math.sqrt(mask.numel()))
            heat  = mask.reshape(side, side).numpy()
            heat  = (heat-heat.min()) / (heat.max()-heat.min()+1e-8)
            energy_curve.append(float((heat**2).sum()))

            h,w,_ = frame.shape
            heat_col = cv2.applyColorMap(
                cv2.resize(np.uint8(255*heat),(w,h)),
                cv2.COLORMAP_JET
            )
            clean_overlay = cv2.addWeighted(frame, .6, heat_col, .4, 0)
            cv2.imwrite(str(clean_dir/f"frame_{idx:02d}.png"),
                        clean_overlay[:, :, ::-1])  # back to BGR

            # ----- FGSM ------------------------------------------------------
            x = inp.pixel_values[j:j+1].clone().detach().requires_grad_(True)
            f_c = emb_clean[j:j+1].detach()
            f_adv = clip(pixel_values=x).image_embeds
            loss = F.cosine_similarity(f_c, f_adv).mean()   # minimise similarity
            loss.backward()
            x_adv = torch.clamp(x - args.epsilon * x.grad.sign(),
                                norm_min, norm_max).detach()

            with torch.no_grad():
                f_adv2 = clip(pixel_values=x_adv).image_embeds
            cos_vals.append(F.cosine_similarity(f_c, f_adv2).item())

            adv_rgb = ((x_adv*std + mean).clamp(0,1)[0]
                       .permute(1,2,0).cpu().numpy())
            adv_overlay = cv2.addWeighted(
                (adv_rgb*255).astype(np.uint8), .6, heat_col, .4, 0)
            cv2.imwrite(str(adv_dir/f"frame_{idx:02d}.png"),
                        adv_overlay[:, :, ::-1])

            del x, x_adv, f_c, f_adv, f_adv2
        torch.cuda.empty_cache()

    print(f"[FGSM] mean cosine(clean vs adv) = {np.mean(cos_vals):.4f}")

    # 4-D) assemble adversarial video
    h,w,_ = frames_bgr[0].shape
    vout = cv2.VideoWriter(str(out_root/"adversarial.mp4"),
                           cv2.VideoWriter_fourcc(*"mp4v"),
                           fps, (w,h))
    for f in sorted(adv_dir.iterdir()):
        vout.write(cv2.imread(str(f)))
    vout.release()

    # 4-E) temporal curve
    if args.save_curve:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,3))
        plt.plot(energy_curve, marker="o"); plt.grid()
        plt.title("Temporal attention energy"); plt.xlabel("frame")
        plt.tight_layout()
        plt.savefig(out_root/"attention_energy_curve.png")
        plt.close()

    # 4-F) VideoLLaMA-2 caption
    if HAVE_VLLAMA:
        disable_torch_init()
        vlm, vproc, vtok = model_init("DAMO-NLP-SG/VideoLLaMA2-7B-16F",
                                      torch_dtype=torch.float16,
                                      device_map={"":device},
                                      attn_implementation="eager")
        vid_t = vproc["video"](str(out_root/"adversarial.mp4")).to(torch.float16).to(device)
        caption = mm_infer(vid_t, "Describe the video.",
                           model=vlm, tokenizer=vtok,
                           modal="video", do_sample=False).strip()
        with open(args.output_caption, "w") as f:
            f.write(caption + "\n")
        print("📝  caption:", caption)
    else:
        print("⚠️  VideoLLaMA2 not found – caption skipped.")

if __name__ == "__main__":
    main()
