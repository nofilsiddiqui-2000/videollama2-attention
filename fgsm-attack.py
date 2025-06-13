#!/usr/bin/env python
"""
FGSM + attention-rollout heat-map for VideoLLaMA-2.1-7B-16F
-----------------------------------------------------------
Untargeted FGSM on CLIP ViT-L/14-336. Outputs:
  • MP4 with attention overlay (adversarial frames)
  • Caption of the adversarial video via VideoLLaMA-2.1
Usage (PowerShell / bash):
  python fgsm_video_attack.py \
         --video test/testvideo3.mp4 \
         --output results/adv_video.mp4 \
         --caption-out results/adv_caption.txt \
         --epsilon 0.05 --batch_size 2
"""
import os, argparse, cv2, numpy as np, torch, torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPImageProcessor
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

# ─────────────────────── helper: robust mean/std ────────────────────────
def clip_mean_std(proc_or_model, device="cuda"):
    """
    Return (mean, std) tensors of shape 1×3×1×1 in fp16 on `device`,
    regardless of transformers version.
    """
    sources = [
        getattr(proc_or_model, "image_processor", None),      # ≥4.40
        getattr(proc_or_model, "feature_extractor", None),    # 4.37-4.39
        proc_or_model,
    ]
    for src in sources:
        if src is None: continue
        mean, std = getattr(src, "image_mean", None), getattr(src, "image_std", None)
        if mean is not None and std is not None:
            return (torch.tensor(mean, dtype=torch.float16, device=device)
                    .view(1,3,1,1),
                    torch.tensor(std,  dtype=torch.float16, device=device)
                    .view(1,3,1,1))
    # OpenAI CLIP defaults:contentReference[oaicite:5]{index=5}
    mean = torch.tensor([0.48145466,0.4578275,0.40821073], dtype=torch.float16, device=device)
    std  = torch.tensor([0.26862954,0.26130258,0.27577711], dtype=torch.float16, device=device)
    return mean.view(1,3,1,1), std.view(1,3,1,1)
# ──────────────────────────────────────────────────────────────────────────

def sample_16_frames(path):
    cap = cv2.VideoCapture(path)
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, tot-1, 16, dtype=int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frm = cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames: raise RuntimeError("no frames extracted")
    return frames, int(cap.get(cv2.CAP_PROP_FPS) or 25)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--caption-out", required=True)
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    os.environ["USE_FLASH_ATTENTION"] = "0"            # safety
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")      # avoid quota error

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. CLIP ViT-L/14-336
    clip_id = "openai/clip-vit-large-patch14-336"
    proc = CLIPImageProcessor.from_pretrained(clip_id)
    vit  = CLIPVisionModel.from_pretrained(clip_id, output_attentions=True
                ).to(dev).half().eval()
    mean, std = clip_mean_std(proc, dev)

    # 2. frames
    frames, fps = sample_16_frames(args.video)

    # 3. FGSM (untargeted, batched)
    adv_rgb = []
    bs, eps = args.batch_size, args.epsilon
    for i in range(0, len(frames), bs):
        imgs = frames[i:i+bs]
        px = proc(images=imgs, return_tensors="pt").to(dev)
        x = px.pixel_values.half().detach().clone().requires_grad_(True)
        with torch.no_grad():
            f_orig = vit(pixel_values=x).pooler_output
            f_orig = F.normalize(f_orig, dim=-1)
        f_adv = vit(pixel_values=x).pooler_output
        f_adv = F.normalize(f_adv, dim=-1)
        loss = -F.cosine_similarity(f_adv, f_orig).mean()   # untargeted FGSM:contentReference[oaicite:6]{index=6}
        loss.backward()
        x_adv = torch.clamp(x + eps * x.grad.sign(), -mean/std, (1-mean)/std).detach()
        rgb = (x_adv*std + mean).clamp(0,1).cpu()
        adv_rgb += [img.permute(1,2,0).numpy() for img in rgb]

    # 4. attention rollout + overlay (inferno)
    import matplotlib.pyplot as plt; cmap = plt.get_cmap("inferno")
    overlaid = []
    for img in adv_rgb:
        px = proc(images=img, return_tensors="pt").to(dev)
        with torch.no_grad():
            outs = vit(pixel_values=px.pixel_values.half(), output_attentions=True)
        att = outs.attentions                                      # tuple len=24
        # rollout:contentReference[oaicite:7]{index=7}
        R = torch.eye(att[0].shape[-1], device=dev)
        for layer in att:
            A = layer[0].mean(0) + torch.eye(layer.shape[-1], device=dev)
            A = A / A.sum(dim=-1, keepdim=True)
            R = A @ R
        mask = R[0,1:].reshape(24,24)
        mask = (mask - mask.min())/(mask.max()-mask.min()+1e-8)
        H,W,_ = img.shape
        mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0),
                                               size=(H,W), mode="bilinear",
                                               align_corners=False)[0,0].cpu().numpy()
        heat = cmap(mask)[...,:3]
        overlay = 0.5*img + 0.5*heat
        overlaid.append(cv2.cvtColor((overlay*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    # 5. save video
    h,w,_ = overlaid[0].shape
    vw = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    for f in overlaid: vw.write(f)
    vw.release()

    # 6. caption adversarial video
    disable_torch_init()
    vll_id = "DAMO-NLP-SG/VideoLLaMA2.1-7B-16F"
    vll, vproc, vtok = model_init(vll_id, torch_dtype=torch.float16,
                                  device_map=dev, attn_implementation="eager")
    caption = mm_infer(vproc["video"](args.output), "Describe the video.",
                       model=vll, tokenizer=vtok, modal="video", do_sample=False).strip()
    with open(args.caption_out,"w",encoding="utf-8") as f: f.write(caption+"\n")
    print(f"[✔] saved {args.output}  | caption → {args.caption_out}\n{caption}")

if __name__ == "__main__":
    main()
