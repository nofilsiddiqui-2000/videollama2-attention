import argparse
import torch
import cv2
import numpy as np
import torchvision.transforms as T
from transformers import CLIPProcessor, CLIPVisionModel
from videollama2 import model_init, mm_infer

def sample_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, nframes-1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

def fgsm_attack(model, processor, images, eps):
    # images: list of numpy arrays (HxWx3) in [0,255]
    # Prepare batch
    inputs = processor(images=images, return_tensors="pt").to(device)
    pixel_values = inputs['pixel_values']  # normalized
    pixel_values = pixel_values.half().detach().requires_grad_()
    # Compute original features
    with torch.no_grad():
        orig_feat = model(pixel_values).pooler_output
        orig_feat = orig_feat / orig_feat.norm(dim=-1, keepdim=True)
    # Compute new features with grad
    outputs = model(pixel_values)
    feat = outputs.pooler_output
    feat = feat / feat.norm(dim=-1, keepdim=True)
    loss = -torch.cosine_similarity(feat, orig_feat).mean()
    loss.backward()
    # FGSM step
    grad_sign = pixel_values.grad.sign()
    adv_pixel = pixel_values + eps * grad_sign
    # Detach and convert back to numpy
    adv_pixel = adv_pixel.detach().float()
    adv_images = []
    for adv in adv_pixel:
        # Undo normalization: x = clip(std * adv + mean, 0,1)
        # Using processor image_mean, image_std
        mean = torch.tensor(processor.image_mean, device=device).view(3,1,1)
        std = torch.tensor(processor.image_std, device=device).view(3,1,1)
        img = adv * std + mean
        img = img.clamp(0, 1).cpu().permute(1,2,0).numpy()  # HWC
        img = (img * 255).astype(np.uint8)
        adv_images.append(img)
    return adv_images

def compute_attention_rollout(attentions, image_size):
    # attentions: list of (batch, num_heads, seq, seq) tensors, or tensor [batch, layers, heads, seq, seq]
    # We'll assume a single image (batch=1) and list of layers
    if isinstance(attentions, tuple):
        attentions = torch.stack(attentions, dim=0)  # shape [layers, batch, heads, seq, seq]
    # Convert to [layers, heads, seq, seq]
    attentions = attentions[:,0]  # take first (and only) image
    num_layers = attentions.size(0)
    # Combine heads (take max across heads)
    attn_maps = attentions.max(dim=2).values  # [layers, seq, seq]
    # Add identity and normalize
    aug_attn = attn_maps + torch.eye(attn_maps.size(-1), device=attn_maps.device)
    aug_attn = aug_attn / aug_attn.sum(dim=-1, keepdim=True)
    # Rollout
    joint_attn = aug_attn[0]
    for i in range(1, num_layers):
        joint_attn = aug_attn[i] @ joint_attn
    # The [CLS] token is at index 0, so take its row (excluding itself)
    mask = joint_attn[0,1:]  # exclude CLS token
    # Reshape to 2D (assume square patches)
    num_patches = mask.size(0)
    side = int(np.sqrt(num_patches))
    mask = mask.view(side, side)
    mask = mask / mask.max()
    # Upsample to image size
    mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0),
                size=image_size, mode='bilinear', align_corners=False)
    return mask.squeeze().cpu().numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--output_video", type=str, required=True)
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--caption_path", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load CLIP model and processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    clip_model = CLIPVisionModel.from_pretrained(
        "openai/clip-vit-large-patch14-336",
        output_attentions=True
    ).to(device).half()

    # 1. Extract frames
    frames = sample_frames(args.video_path, num_frames=16)

    # 2. FGSM attack on frames
    adv_frames = fgsm_attack(clip_model, processor, frames, args.epsilon)

    # 3. Compute attention rollout heatmaps for adversarial frames
    overlay_frames = []
    for img in adv_frames:
        # Prepare input for attentions
        inputs = processor(images=img, return_tensors="pt").to(device)
        out = clip_model(**inputs)
        attn = out.attentions  # tuple of layers
        heatmap = compute_attention_rollout(attn, (img.shape[0], img.shape[1]))
        # Convert heatmap to color
        heatmap_img = np.uint8(255 * heatmap)
        heatmap_img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.7, heatmap_img, 0.3, 0)
        overlay_frames.append(overlay)

    # 4. Save overlay video
    h, w = overlay_frames[0].shape[:2]
    writer = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w,h))
    for frame in overlay_frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()

    # 5. VideoLLaMA2 captioning
    model_path = "DAMO-NLP-SG/VideoLLaMA2.1-7B-16F"
    model, va_processor, tokenizer = model_init(model_path)
    # Use the video file or frames as input
    caption = mm_infer(va_processor["video"](args.output_video), 
                       "Describe the content of the video.",
                       model=model, tokenizer=tokenizer, do_sample=False, modal="video")
    # Save caption
    with open(args.caption_path, "w") as f:
        f.write(caption)
