#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────────
# caption_video.py  –  minimal, flash‑2–safe VideoLLaMA2 captioner
# ────────────────────────────────────────────────────────────────────
"""
Run:
    python caption_video.py [path/to/video.mp4]
The first run downloads DAMO‑NLP‑SG/VideoLLaMA2‑7B‑16F (~16 GB).
The generated caption is written to  ./test/caption.txt
"""
# --- 1 ▸ neutralise Flash‑Attention‑2 gating everywhere -------------
import transformers.modeling_utils as _mu
def _disable_flash_attn_2(*_, **__):
    # always return False → Model code never tries to enable F‑A‑2
    return False
# patch BOTH the module‑level function *and* the copy living
# on every already‑defined PreTrainedModel subclass
_mu._check_and_enable_flash_attn_2 = _disable_flash_attn_2
_mu.PreTrainedModel._check_and_enable_flash_attn_2 = staticmethod(_disable_flash_attn_2)
# --- 2 ▸ environment knobs (must be set before torch imports) -------
import os
os.environ["PYTORCH_ATTENTION_IMPLEMENTATION"] = "eager"   # force eager attn
os.environ["HF_DISABLE_FLASH_ATTN_2"]           = "1"
os.environ["DISABLE_FLASH_ATTN_2"]              = "1"
# --- 3 ▸ stdlib / third‑party imports --------------------------------
import sys
import argparse
import torch
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
# --- 4 ▸ main --------------------------------------------------------
def main() -> None:
    disable_torch_init()                        # speed‑ups for HF models
    ap = argparse.ArgumentParser(description="Generate a caption for a video with VideoLLaMA2.")
    ap.add_argument("video_path", nargs="?", default="/root/video_llama_caption/test/testvideo.mp4",
                    help="Input video (MP4).  Default: test/testvideo.mp4")
    ap.add_argument("--out", "-o", default="test/caption.txt",
                    help="Path to write the caption.  Default: test/caption.txt")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit(":x:  CUDA GPU not available – start the container on a GPU instance.")
    # -----------------------------------------------------------------
    model_name = "DAMO-NLP-SG/VideoLLaMA2-7B-16F"
    print(f"Loading model ‘{model_name}’ … (first run can take a while)")
    model, processor, tokenizer = model_init(
        model_name,
        attn_implementation="eager",            # extra safety
    )
    # model.to("cuda").eval()
    model.eval()
    # -----------------------------------------------------------------
    print(f":mag:  Captioning  {args.video_path}")
    video_tensor = processor["video"](args.video_path)     # (T,C,H,W) fp32
    caption = mm_infer(video_tensor, "Describe the video.",
                       model=model, tokenizer=tokenizer, modal="video").strip()
    # -----------------------------------------------------------------
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf‑8") as f:
        f.write(caption + "\n")
    print(f":white_check_mark:  Caption saved to {args.out}\n\n---\n{caption}\n---")
# --- 5 ▸ entry‑point -------------------------------------------------
if __name__ == "__main__":
    main()