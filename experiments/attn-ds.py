#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────────
# VideoLLaMA‑2  ▸  Spatial + Temporal Attention  ▸  frame exporter
# ────────────────────────────────────────────────────────────────────
# Outputs:
#   <out_dir>/
#       frame_00001.png  … one overlay per video frame
#       temporal_weights.png
#       caption.txt
# ────────────────────────────────────────────────────────────────────

import os, sys, cv2, argparse, warnings
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from transformers import CLIPVisionModel, CLIPImageProcessor

from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

def jet_cmap():
    # works on both old and new Matplotlib versions
    if hasattr(cm, "colormaps"):          # ≥3.6
        return cm.colormaps.get_cmap("jet")
    else:                                 # <3.6
        return cm.get_cmap("jet")


# ─────────────────────  misc env + flash‑attn toggle  ───────────────
def _disable_fa2(*_, **__): return False
import transformers.modeling_utils as _mu
_mu._check_and_enable_flash_attn_2 = _disable_fa2
_mu.PreTrainedModel._check_and_enable_flash_attn_2 = staticmethod(_disable_fa2)
os.environ.update({
    "PYTORCH_ATTENTION_IMPLEMENTATION": "eager",
    "HF_DISABLE_FLASH_ATTN_2": "1",
    "DISABLE_FLASH_ATTN_2": "1"
})
warnings.filterwarnings("ignore", category=UserWarning)

MODEL_NAME   = "DAMO-NLP-SG/VideoLLaMA2-7B-16F"
VISION_ID    = "openai/clip-vit-large-patch14-336"
PATCH_SIDE   = 24      # ViT‑L/14 (336×336) ⇒ 24×24 patch grid

# ─────────────────────  hook class  ─────────────────────────────────
class AttnHook:
    def __init__(self):
        self.T = []            # list of tensors
        self.handles = []

    def __call__(self, _m, _i, o):
        # o is (output, attn, pkv) OR just attn
        if isinstance(o, tuple) and o[1] is not None:
            self.T.append(o[1].detach().cpu())
        elif torch.is_tensor(o):
            self.T.append(o.detach().cpu())

    def add_hooks(self, model):
        for name, mod in model.named_modules():
            if name.endswith("self_attn"):                 # only full attention module
                self.handles.append(mod.register_forward_hook(self))

    def clear(self):  self.T = []
    def close(self):
        for h in self.handles: h.remove()
        self.handles.clear()

# ─────────────────────  loaders  ────────────────────────────────────
def load_models(device="cuda"):
    vt   = CLIPVisionModel.from_pretrained(VISION_ID).to(device).eval()
    proc = CLIPImageProcessor.from_pretrained(VISION_ID)

    disable_torch_init()
    model, processor, tok = model_init(
        MODEL_NAME, device_map=device,
        torch_dtype=torch.float16,
        attn_implementation="eager")
    model.eval();  model.config.output_attentions = True

    hook = AttnHook(); hook.add_hooks(model)
    return vt, proc, model, processor, tok, hook

# ─────────────────────  helper fns  ─────────────────────────────────
@torch.no_grad()
def run_caption(video_path, model, processor, tok, hook, device):
    hook.clear()
    vid = processor["video"](video_path).to(torch.float16).to(device)
    cap = mm_infer(vid, "Describe the video in detail.",
                   model=model, tokenizer=tok,
                   modal="video", do_sample=False,
                   output_attentions=True).strip()
    return cap, hook.T

def spatial_rollout(pil, vt, proc, device):
    ins  = proc(images=pil, return_tensors="pt").to(device)
    outs = vt(**ins, output_attentions=True)

    heads = torch.stack(outs.attentions)[:, 0]   # L, H, N, N
    Abar  = heads.mean(1)                        # L, N, N

    eye = torch.eye(Abar.size(-1), device=device)
    R   = eye.clone()
    for A in Abar:
        A = A + eye
        A = A / A.sum(-1, keepdim=True)
        R = A @ R

    cls = R[0, 1:].view(PATCH_SIDE, PATCH_SIDE).detach().cpu().numpy()
    cls = (cls - cls.min()) / (cls.ptp() + 1e-6)
    cls = np.power(cls, .5)                      # enhance contrast
    cls = np.asarray(Image.fromarray(cls).resize(pil.size, Image.BILINEAR))
    return cls

def plot_temporal(tensors, n_frames, path):
    # keep only tensors shaped (B, H?, Q, K) where K == n_frames
    good = []
    for t in tensors:
        if t.shape[-1] == n_frames:
            if t.ndim == 3:          # (B, Q, K)  → add fake head dim
                t = t.unsqueeze(1)
            good.append(t)
    if not good:
        print("[WARN] no compatible tensors for temporal plot"); return

    curve = torch.cat(good, 0).mean((0,1,2))     # → (K,)
    curve = (curve - curve.min()) / (curve.ptp()+1e-6)

    plt.figure(figsize=(12,4))
    plt.plot(curve.numpy(), "o-", linewidth=2, markersize=8)
    plt.title("Temporal Attention Weights")
    plt.xlabel("Frame #"); plt.ylabel("Weight")
    plt.grid(True); plt.tight_layout(); plt.savefig(path); plt.close()
    print(f"[VIZ] temporal plot → {path}")

# ─────────────────────  main export  ────────────────────────────────
def export_frames(video_path, out_dir, model, proc_llm, tok,
                  hook, vt, proc_vit, device="cuda", alpha=0.5):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): sys.exit(f"cannot open {video_path}")
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pad   = len(str(total))

    # 1 ▸ caption pass
    caption, tensors = run_caption(video_path, model, proc_llm, tok, hook, device)
    print("[CAPTION]", caption)
    with open(os.path.join(out_dir, "caption.txt"), "w") as f:
        f.write(caption+"\n")

    plot_temporal(tensors, total, os.path.join(out_dir, "temporal_weights.png"))

    # jet = cm.colormaps.get_cmap("jet")
    jet = jet_cmap()
    idx = 0
    while True:
        ok, frame = cap.read();  idx += 1
        if not ok: break

        pil  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        heat = spatial_rollout(pil, vt, proc_vit, device)

        heat_rgba = (jet(heat)*255).astype(np.uint8)
        heat_bgr  = cv2.cvtColor(heat_rgba, cv2.COLOR_RGBA2BGR)
        overlay   = cv2.addWeighted(frame, 1-alpha, heat_bgr, alpha, 0)

        cv2.putText(overlay, f"{idx}/{total}", (15,35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        name = f"frame_{idx:0{pad}d}.png"
        cv2.imwrite(os.path.join(out_dir, name), overlay)

        if idx % 10 == 0:
            print(f"{idx}/{total} frames", end="\r")

    cap.release(); print(f"\n[SAVED] {idx} overlays → {out_dir}")
    hook.close()

# ─────────────────────  CLI  ────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="video path (.mp4, .mov …)")
    ap.add_argument("--out-dir", default="frames_vis", help="output folder")
    ap.add_argument("--alpha", type=float, default=0.5, help="overlay opacity")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("GPU required for this script")

    print("[INIT] loading checkpoints …")
    vt, proc_vit, model, proc_llm, tok, hook = load_models("cuda")

    export_frames(args.video, args.out_dir,
                  model, proc_llm, tok, hook,
                  vt, proc_vit, device="cuda", alpha=args.alpha)

if __name__ == "__main__":
    main()
