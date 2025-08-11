#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────────
# VideoLLaMA-2 Attention Visualization (Spatial + Temporal) - FIXED
# ────────────────────────────────────────────────────────────────────

import os, sys, cv2, argparse, time
import numpy as np
from PIL import Image
import torch
from transformers import CLIPVisionModel, CLIPImageProcessor
import matplotlib.pyplot as plt
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

# Disable FlashAttention for compatibility
def _disable_fa2(*_, **__):
    return False

import transformers.modeling_utils as _mu
_mu._check_and_enable_flash_attn_2 = _disable_fa2
_mu.PreTrainedModel._check_and_enable_flash_attn_2 = staticmethod(_disable_fa2)
os.environ.update({
    "PYTORCH_ATTENTION_IMPLEMENTATION": "eager",
    "HF_DISABLE_FLASH_ATTN_2": "1",
    "DISABLE_FLASH_ATTN_2": "1"
})

MODEL_NAME = "DAMO-NLP-SG/VideoLLaMA2-7B-16F"
VISION_ID = "openai/clip-vit-large-patch14-336"

# Hook to capture attention weights
class AttentionHook:
    def __init__(self):
        self.attention_weights = []
        self.handles = []
    
    def __call__(self, module, input, output):
        # Output is (attn_output, attn_weights, present_key_value)
        if len(output) >= 2 and output[1] is not None:
            self.attention_weights.append(output[1].detach().cpu())
    
    def register(self, model):
        # Register only on the language model layers
        for layer in model.model.layers:
            handle = layer.self_attn.register_forward_hook(self)
            self.handles.append(handle)
        print(f"Registered hooks on {len(self.handles)} language layers")
    
    def remove(self):
        for handle in self.handles:
            handle.remove()
    
    def clear(self):
        self.attention_weights = []

def load_models(device: str):
    """Load vision tower and VideoLLaMA with attention hooks"""
    # Load CLIP vision tower
    vt = CLIPVisionModel.from_pretrained(VISION_ID).to(device).eval()
    proc = CLIPImageProcessor.from_pretrained(VISION_ID)
    
    # Load VideoLLaMA with hooks
    disable_torch_init()
    model, processor, tokenizer = model_init(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=device,
        attn_implementation="eager"
    )
    model.eval()
    
    # Setup attention hook
    hook = AttentionHook()
    hook.register(model)
    
    return vt, proc, model, processor, tokenizer, hook

def generate_caption_with_attention(
    video_path: str, 
    model, 
    processor, 
    tokenizer, 
    hook,
    device: str
) -> tuple:
    """Generate caption while capturing attention weights"""
    hook.clear()
    
    # Force the model to return attention weights
    model.config.output_attentions = True
    
    vid_tensor = processor["video"](video_path).to(torch.float16).to(device)
    caption = mm_infer(
        vid_tensor,
        "Describe the video in detail.",
        model=model,
        tokenizer=tokenizer,
        modal="video",
        do_sample=False,
        output_attentions=True
    ).strip()
    return caption, hook.attention_weights

def spatial_attention_rollout(
    pil_img: Image.Image, 
    vt: CLIPVisionModel, 
    proc: CLIPImageProcessor, 
    device: str
) -> np.ndarray:
    """Generate spatial attention heatmap for a single frame"""
    inputs = proc(images=pil_img, return_tensors="pt").to(device)
    outs = vt(**inputs, output_attentions=True)
    
    # Stack and average attention
    att_all = torch.stack(outs.attentions)[:, 0].mean(dim=1)
    
    # Compute rollout
    T = att_all.shape[-1]
    eye = torch.eye(T, device=device)
    R = eye.clone()
    for layer_attn in att_all:
        A = layer_attn + eye
        A = A / A.sum(dim=-1, keepdim=True)
        R = A @ R
    
    # Process attention map
    cls_attn = R[0, 1:].view(24, 24).detach().cpu().numpy()  # Fixed detach
    heat = Image.fromarray(cls_attn).resize(pil_img.size, Image.BILINEAR)
    heat = np.asarray(heat, dtype=np.float32)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    return np.power(heat, 0.5)  # Enhance contrast

def visualize_temporal_attention(
    attention_weights: list, 
    num_frames: int,
    output_path: str
):
    """Visualize temporal attention across frames"""
    if not attention_weights:
        print("[WARN] No attention weights captured")
        return
    
    # Aggregate attention weights - use only the first generation step
    if len(attention_weights) > 0:
        # Get attention from first layer, first head
        frame_attention = attention_weights[0][0, 0, 0, :num_frames].cpu().numpy()
        
        # Normalize
        frame_attention = (frame_attention - frame_attention.min()) / (frame_attention.max() - frame_attention.min() + 1e-8)
        
        # Plot
        plt.figure(figsize=(12, 4))
        plt.plot(frame_attention, 'o-', linewidth=2, markersize=8)
        plt.title("Temporal Attention Weights")
        plt.xlabel("Frame Index")
        plt.ylabel("Attention Weight")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"[VIZ] Saved temporal attention plot to {output_path}")
    else:
        print("[WARN] No attention weights to visualize")

def create_attention_frames(
    video_path: str,
    output_dir: str,
    model,
    processor,
    tokenizer,
    hook,
    vt,
    proc,
    device: str,
    alpha: float = 0.5
):
    """Create frames with spatial attention overlays"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[ERR] Cannot open {video_path}")
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Generate caption to populate attention weights
    print("[ATT] Generating caption to capture attention weights...")
    caption, attn_weights = generate_caption_with_attention(
        video_path, model, processor, tokenizer, hook, device
    )
    print(f"[CAPTION] {caption}")
    
    # Visualize temporal attention
    temporal_plot = os.path.join(output_dir, "temporal_attention.png")
    visualize_temporal_attention(attn_weights, frame_count, temporal_plot)
    
    # Process frames with attention overlays
    print("[VIZ] Creating attention frames...")
    frame_idx = 0
    
    # Get colormap
    jet = plt.colormaps.get_cmap("jet")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to PIL and process
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Get spatial attention
        heat = spatial_attention_rollout(pil_img, vt, proc, device)
        
        # Create overlay
        heat_rgba = (jet(heat) * 255).astype(np.uint8)
        heat_rgb = cv2.cvtColor(heat_rgba, cv2.COLOR_RGBA2RGB)
        heat_bgr = cv2.cvtColor(heat_rgb, cv2.COLOR_RGB2BGR)
        overlay_frame = cv2.addWeighted(frame, 1 - alpha, heat_bgr, alpha, 0)
        
        # Add frame index and attention info
        cv2.putText(overlay_frame, f"Frame: {frame_idx}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add max attention point
        max_loc = np.unravel_index(np.argmax(heat), heat.shape)
        cv2.circle(overlay_frame, (max_loc[1], max_loc[0]), 15, (0, 0, 255), 3)
        
        # Save frame
        out_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
        cv2.imwrite(out_path, overlay_frame)
        
        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"Processed {frame_idx}/{frame_count} frames", end='\r')
    
    cap.release()
    print(f"\n[SAVED] {frame_idx} attention frames to {output_dir}")
    return caption

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="Input video path")
    ap.add_argument("--out-dir", default="attention_frames", help="Output directory for frames")
    ap.add_argument("--alpha", type=float, default=0.5, help="Heatmap opacity")
    ap.add_argument("--caption-out", default="caption.txt", help="Caption output file")
    
    args = ap.parse_args()
    
    if not torch.cuda.is_available():
        sys.exit("[ERR] GPU required for this implementation")
    
    device = "cuda"
    
    # Load all models
    print("[INIT] Loading models...")
    vt, proc, model, processor, tokenizer, hook = load_models(device)
    
    # Create attention frames and get caption
    caption = create_attention_frames(
        args.video,
        args.out_dir,
        model,
        processor,
        tokenizer,
        hook,
        vt,
        proc,
        device,
        alpha=args.alpha
    )
    
    # Save caption
    with open(args.caption_out, "w") as f:
        f.write(caption + "\n")
    print(f"[SAVED] Caption saved to {args.caption_out}")
    
    # Clean up
    hook.remove()

if __name__ == "__main__":
    main()