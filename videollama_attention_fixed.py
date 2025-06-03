# #!/usr/bin/env python3
# # videollama_attention_fixed.py  •  2025-05-31
# # ── Spatial + Temporal attention visualiser for VideoLLaMA-2 ──

# import os, sys, cv2, argparse, math
# from pathlib import Path
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# import torch
# from transformers import CLIPVisionModel, CLIPImageProcessor
# from videollama2 import model_init
# from videollama2.utils import disable_torch_init

# # ── hard-disable Flash-Attn (same as before) ────────────────────
# import transformers.modeling_utils as _mu
# def _off(*_, **__): return False
# _mu._check_and_enable_flash_attn_2 = _off
# _mu.PreTrainedModel._check_and_enable_flash_attn_2 = staticmethod(_off)
# os.environ.update({
#     "PYTORCH_ATTENTION_IMPLEMENTATION": "eager",
#     "HF_DISABLE_FLASH_ATTN_2": "1",
#     "DISABLE_FLASH_ATTN_2": "1",
# })

# MODEL_NAME = "DAMO-NLP-SG/VideoLLaMA2-7B-16F"
# VISION_ID  = "openai/clip-vit-large-patch14-336"
# CLS = 1                       # CLS token index inside ViT attn matrices

# # ────────────────────────────────────────────────────────────────
# # 1.  Load towers
# # ────────────────────────────────────────────────────────────────
# def load(device):
#     vt   = CLIPVisionModel.from_pretrained(VISION_ID).to(device).eval()
#     vproc= CLIPImageProcessor.from_pretrained(VISION_ID)

#     disable_torch_init()
#     vlm, vprocessor, tok = model_init(
#         MODEL_NAME,
#         torch_dtype=torch.float16,
#         device_map=device,
#         attn_implementation="eager"
#     )
#     vlm.eval()
#     return vt, vproc, vlm, vprocessor, tok

# # ────────────────────────────────────────────────────────────────
# # 2.  Caption + cross-modal attentions
# # ────────────────────────────────────────────────────────────────
# # @torch.inference_mode()           # this func one not working idk why
# # def caption_and_attn(video_path, vlm, vproc, tok, device):
# #     """Return (caption:str, layer_attn:list, n_proc_frames:int)."""
# #     # vlm.config.update(output_attentions=True, return_dict_in_generate=True)
# #     vlm.config.output_attentions = True
# #     vlm.config.return_dict_in_generate = True

# #     video_tensor = vproc["video"](video_path).to(device, dtype=torch.float16)
# #     prompt = "Describe the video in detail."
# #     input_ids = tok(prompt, return_tensors="pt").input_ids.to(device)

# #     gen = vlm.generate(
# #         input_ids=input_ids,
# #         pixel_values=video_tensor,          # **dict** expected!
# #         max_new_tokens=64,
# #         do_sample=False,
# #         output_attentions=True,
# #         return_dict_in_generate=True,
# #     )

# #     caption = tok.decode(gen.sequences[0], skip_special_tokens=True).strip()
# #     attentions = gen.attentions           # tuple[n_layers]
# #     n_frames = video_tensor.shape[1]      # processed frames (usually 16)
# #     return caption, attentions, n_frames

# # ────────────────────────────────────────────────────────────────
# # 2.  Caption + cross-modal attentions  (patched)
# # ────────────────────────────────────────────────────────────────


# # ────────────────────────────────────────────────────────────────
# # 2. Caption + cross-modal attentions   ★ FINAL PATCH ★
# # ────────────────────────────────────────────────────────────────
# @torch.inference_mode()
# def caption_and_attn(video_path, vlm, vproc, tok, device):
#     # ask the LM to keep every layer’s attentions
#     vlm.config.output_attentions       = True
#     vlm.config.return_dict_in_generate = True

#     # 1) video to float-16 tensor on GPU
#     video_tensor = vproc["video"](video_path).to(device, dtype=torch.float16)

#     # 2) prompt ➜ token IDs tensor
#     prompt_ids = tok(
#         "Describe the video in detail.",
#         return_tensors="pt"
#     ).input_ids.to(device)

#     # 3) generation
#     # gen = vlm.generate(
#     #     inputs        = prompt_ids,     # ← must be called **inputs**
#     #     pixel_values  = video_tensor,   # ← video batch
#     #     max_new_tokens= 64,
#     #     do_sample     = False,
#     #     output_attentions      = True,
#     #     return_dict_in_generate= True,
#     # )

#     gen = vlm.generate(
#         inputs = prompt_ids,
#         images = [(video_tensor, "video")],   # ← wrap as (tensor, "video")
#         max_new_tokens = 64,
#         do_sample = False,
#         output_attentions = True,
#         return_dict_in_generate = True,
#     )


#     if gen.attentions is None:
#         raise RuntimeError("Model did not return attentions "
#                            "(checkpoint too old?)")

#     # 4) extract outputs
#     caption  = gen.text[0] if hasattr(gen, "text") else \
#                tok.decode(gen.sequences[0], skip_special_tokens=True).strip()
#     attns    = gen.attentions                     # tuple[n_layers]
#     n_frames = video_tensor.shape[1]
#     return caption, attns, n_frames


# # def temporal_curve(attns, n_frames):
# #     last = attns[-1][0]                   # heads × tgt × src
# #     heads, tgt, src = last.shape
# #     vis_tokens = src - CLS
# #     tpf = vis_tokens // n_frames          # tokens per frame
# #     frame_score = []
# #     avg = last.mean(0).mean(0)            # (src,)
# #     for f in range(n_frames):
# #         s, e = CLS + f*tpf, CLS + (f+1)*tpf
# #         frame_score.append(avg[s:e].mean().item())
# #     arr = np.asarray(frame_score)
# #     return (arr - arr.min()) / (arr.max()-arr.min()+1e-8)

# def temporal_curve(attns, n_frames):
#     """
#     Build a 0-1 attention curve (length = n_frames) from the *last* decoder layer.

#     Works with all nesting patterns returned by different VideoLLaMA forks:
#       layer tuple  →  step tuple  →  tensor (B,S,H,T,S) / (S,H,T,S) / (H,T,S)
#     """
#     # 1) dig until we hit a tensor
#     last = attns[-1]
#     while isinstance(last, (tuple, list)):
#         last = last[0]

#     if not torch.is_tensor(last):
#         raise TypeError("Could not find attention tensor inside attns[-1]")

#     # 2) collapse batch / step axes → (heads, tgt, src)
#     if last.dim() == 5:          # (B, step, H, tgt, src)
#         last = last[0, 0]
#     elif last.dim() == 4:        # (step, H, tgt, src)
#         last = last[0]

#     if last.dim() != 3:
#         raise ValueError(f"Unexpected attention tensor shape: {last.shape}")

#     heads, tgt, src = last.shape
#     vis_tokens = src - CLS
#     tok_per_frame = max(1, vis_tokens // n_frames)

#     # 3) average heads + target tokens
#     avg_src = last.mean(dim=0).mean(dim=0)   # (src,)

#     scores = []
#     for f in range(n_frames):
#         s = CLS + f * tok_per_frame
#         e = min(s + tok_per_frame, src)
#         scores.append(avg_src[s:e].mean().item())

#     arr = np.asarray(scores, dtype=np.float32)
#     return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


# # ────────────────────────────────────────────────────────────────
# # 3.  Self-attention rollout (ViT)  → heat-map
# # ────────────────────────────────────────────────────────────────
# @torch.inference_mode()
# def rollout_heat(pil, vt, vproc, device):
#     inp  = vproc(images=pil, return_tensors="pt").to(device)
#     outs = vt(**inp, output_attentions=True)
#     A    = torch.stack(outs.attentions)[:, 0].mean(1)  # layers×tokens×tokens

#     eye = torch.eye(A.size(-1), device=device)
#     R = eye.clone()
#     for layer in A:
#         layer = (layer + eye)
#         layer /= layer.sum(-1, keepdim=True)
#         R = layer @ R
#     cls = R[0, CLS:].reshape(24,24).detach().cpu().numpy()
#     cls = cv2.GaussianBlur(cls, (0,0), 3)
#     cls = (cls - cls.min()) / (cls.max()-cls.min()+1e-8)
#     return np.power(cls, .5)

# # ────────────────────────────────────────────────────────────────
# def make_frames(video, outdir, vt, vproc, vlm, vp, tok, dev,
#                 alpha=0.35, rotate=True):
#     outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

#     print("[CAP] generating caption …")
#     captxt, attns, n_proc = caption_and_attn(video, vlm, vp, tok, dev)
#     print("     ➜", captxt)

#     curve = temporal_curve(attns, n_proc)
#     plt.figure(figsize=(12,4)); plt.plot(curve,'o-'); plt.grid()
#     plt.title("Temporal attention"); plt.xlabel("frame"); plt.ylabel("weight")
#     plt.tight_layout(); plt.savefig(outdir/"temporal_attention.png"); plt.close()

#     jet = plt.colormaps.get_cmap("jet")
#     cap = cv2.VideoCapture(video)
#     if not cap.isOpened(): sys.exit(f"[ERR] cannot open {video}")

#     idx=0
#     while True:
#         ok, frame = cap.read()
#         if not ok: break
#         if rotate: frame=cv2.rotate(frame, cv2.ROTATE_180)

#         pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         heat = rollout_heat(pil, vt, vproc, dev)
#         heat = cv2.resize(
#             heat, (frame.shape[1], frame.shape[0]),  # (width, height)
#             interpolation=cv2.INTER_LINEAR
#         )
#         h_rgba = (jet(heat)*255).astype(np.uint8)
#         h_bgr  = cv2.cvtColor(cv2.cvtColor(h_rgba,cv2.COLOR_RGBA2RGB),cv2.COLOR_RGB2BGR)
#         mix    = cv2.addWeighted(frame, 1-alpha, h_bgr, alpha, 0)

#         cv2.putText(mix,f"frame {idx}",(20,40),
#                     cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
#         y,x = np.unravel_index(np.argmax(heat),heat.shape)
#         cv2.circle(mix,(x,y),12,(0,0,255),3)
#         cv2.imwrite(str(outdir/f"frame_{idx:04d}.png"), mix)
#         if idx%10==0: print(f"   processed {idx}",end="\r")
#         idx+=1
#     cap.release()
#     print(f"\n[SAVED] {idx} frames ➜ {outdir}")
#     return captxt

# # ────────────────────────────────────────────────────────────────
# def main():
#     ap=argparse.ArgumentParser()
#     ap.add_argument("video")
#     ap.add_argument("--out",default="attention_frames")
#     ap.add_argument("--alpha",type=float,default=.35)
#     ap.add_argument("--no-rotate",action="store_true")
#     ap.add_argument("--caption-file",default="caption.txt")
#     args=ap.parse_args()

#     if not torch.cuda.is_available(): sys.exit("[ERR] need CUDA GPU")
#     dev="cuda"

#     print("[INIT] loading towers …")
#     vt,vproc,vlm,vp,tok=load(dev)
#     cap=make_frames(args.video,args.out,vt,vproc,vlm,vp,tok,dev,
#                     alpha=args.alpha,rotate=not args.no_rotate)
#     Path(args.caption_file).write_text(caption:=cap+"\n",encoding="utf-8")
#     print(f"[DONE] caption saved → {args.caption_file}")

# if __name__=="__main__": main()


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
