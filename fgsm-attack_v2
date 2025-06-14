import argparse
import os
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from transformers import CLIPModel, CLIPProcessor
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', type=str, required=True,
                        help="Path to input video")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save outputs")
    parser.add_argument('--epsilon', type=float, required=True,
                        help="FGSM perturbation magnitude (in [0,1] range)")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for processing frames")
    parser.add_argument('--save_curve', action='store_true',
                        help="Whether to save temporal attention curve")
    parser.add_argument('--output_caption', type=str, required=True,
                        help="File to save the VideoLLaMA caption")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load CLIP model and processor (ViT-L/14-336)
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336").to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    # CLIP normalization parameters
    clip_mean = torch.tensor(processor.image_processor.image_mean, device=device).view(3,1,1)
    clip_std  = torch.tensor(processor.image_processor.image_std, device=device).view(3,1,1)

    # Load video and sample 16 evenly spaced frames
    cap = cv2.VideoCapture(args.input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError("Could not read video or no frames found")
    frame_idxs = np.linspace(0, total_frames-1, 16, dtype=int)
    frames = []
    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Could not read frame {idx}")
        frames.append(frame)
    cap.release()

    # Compute original CLIP features for each frame
    orig_feats = []
    for frame in frames:
        # Convert BGR->RGB, to [0,1] tensor
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(img_rgb.astype(np.float32)/255.0).permute(2,0,1).to(device)
        x_norm = (x - clip_mean) / clip_std  # normalize
        with torch.no_grad():
            feat = model.get_image_features(pixel_values=x_norm.unsqueeze(0))
        orig_feats.append(feat.squeeze(0))

    # FGSM attack: perturb frames to maximize loss (untargeted)
    adv_frames = []
    adv_feats = []
    for frame, orig_feat in zip(frames, orig_feats):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(img_rgb.astype(np.float32)/255.0).permute(2,0,1).to(device)
        x_norm = (x - clip_mean) / clip_std
        x_norm_adv = x_norm.clone().detach().unsqueeze(0).requires_grad_(True)
        # Forward pass
        feat_adv = model.get_image_features(pixel_values=x_norm_adv)
        # Loss = - cosine_similarity(orig, adv) to push them apart
        loss = -F.cosine_similarity(feat_adv, orig_feat.unsqueeze(0)).mean()
        loss.backward()
        # FGSM perturbation
        grad_sign = x_norm_adv.grad.sign()
        x_norm_adv = x_norm_adv + args.epsilon * grad_sign
        # Clamp to ensure pixel values stay in [0,1] range after un-normalizing
        min_norm = (0.0 - clip_mean) / clip_std
        max_norm = (1.0 - clip_mean) / clip_std
        # Clamp per channel
        x_norm_adv[:,0] = torch.clamp(x_norm_adv[:,0], min_norm[0], max_norm[0])
        x_norm_adv[:,1] = torch.clamp(x_norm_adv[:,1], min_norm[1], max_norm[1])
        x_norm_adv[:,2] = torch.clamp(x_norm_adv[:,2], min_norm[2], max_norm[2])
        adv_norm = x_norm_adv.detach().squeeze(0)
        # Convert back to uint8 image
        adv_img = (adv_norm * clip_std + clip_mean).permute(1,2,0).cpu().numpy()
        adv_img = np.clip(adv_img, 0, 1) * 255
        adv_img = adv_img.astype(np.uint8)
        adv_frames.append(adv_img)
        # Compute adversarial CLIP feature
        with torch.no_grad():
            adv_feat = model.get_image_features(pixel_values=x_norm_adv)
        adv_feats.append(adv_feat.squeeze(0))

    # Compute cosine similarity metric
    cos_sims = []
    for f0, f1 in zip(orig_feats, adv_feats):
        cos = F.cosine_similarity(f0, f1, dim=0).item()
        cos_sims.append(cos)
    avg_cos = sum(cos_sims) / len(cos_sims)
    print(f"Average cosine similarity (orig vs adv features): {avg_cos:.4f}")

    # Save attention heatmaps for each adversarial frame
    attn_means = []
    for i, adv_img in enumerate(adv_frames):
        # Prepare input for CLIP and get attentions
        inputs = processor(images=adv_img, return_tensors="pt")
        for k in inputs:
            inputs[k] = inputs[k].to(device)
        outputs = model(**inputs, output_attentions=True)
        attn_list = outputs.vision_model_output.attentions  # list of (1, head, seq, seq)
        # Compute rollout
        rollout = None
        for attn in attn_list:
            # Average over heads
            a = attn[0].mean(0)  # (seq, seq)
            # Add identity (residual) and renormalize
            a = a + torch.eye(a.size(0), device=device)
            a = a / a.sum(dim=1, keepdim=True)
            rollout = a if rollout is None else rollout.matmul(a)
        # Extract [CLS] token attention to patches
        # Vision seq len = 1 + num_patches (class token first)
        cls_attn = rollout[0,1:]  # skip CLS to CLS (itself)
        num_patches = int(cls_attn.size(0))
        side = int(np.sqrt(num_patches))
        mask = cls_attn.reshape(side, side).cpu().numpy()
        # Normalize mask
        mask = mask / (mask.max() if mask.max()>0 else 1)
        attn_means.append(mask.std())  # use std dev as a simple metric
        # Resize mask to image size
        h, w, _ = adv_img.shape
        mask_resized = cv2.resize(mask, (w,h), interpolation=cv2.INTER_LINEAR)
        heatmap = cv2.applyColorMap(np.uint8(255*mask_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.cvtColor(adv_img, cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)
        heatmap_path = os.path.join(args.output_dir, f"frame_{i:02d}_heatmap.png")
        cv2.imwrite(heatmap_path, overlay)

    # Save temporal attention curve if requested
    if args.save_curve:
        plt.figure(figsize=(4,3))
        plt.plot(attn_means, marker='o')
        plt.title("Attention (std) per frame")
        plt.xlabel("Frame index")
        plt.ylabel("Mask std dev")
        curve_path = os.path.join(args.output_dir, "attention_curve.png")
        plt.savefig(curve_path)
        plt.close()

    # Save adversarial video
    video_out_path = os.path.join(args.output_dir, "adversarial_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = adv_frames[0].shape
    writer = cv2.VideoWriter(video_out_path, fourcc, fps, (w, h))
    for adv_img in adv_frames:
        # adv_img is RGB, convert to BGR for cv2
        writer.write(cv2.cvtColor(adv_img, cv2.COLOR_RGB2BGR))
    writer.release()

    # Generate caption with VideoLLaMA
    try:
        from videollama2 import model_init, mm_infer, disable_torch_init
    except ImportError:
        raise ImportError("Please install the VideoLLaMA2 package (videollama2) to perform captioning.")
    disable_torch_init()
    model_vl, processor_vl, tokenizer_vl = model_init("DAMO-NLP-SG/VideoLLaMA2-7B-16F")
    # Prepare video input (processor expects file path or frames)
    # Here we saved adversarial video as file; we can feed its path
    vid_input = processor_vl["video"](video_out_path)
    # Instruction prompt for captioning
    instr = "Describe the video."
    output = mm_infer(vid_input, instr, model=model_vl, tokenizer=tokenizer_vl,
                      do_sample=False, modal='video')
    caption = output.strip()
    with open(args.output_caption, 'w') as f:
        f.write(caption)
    print("Generated caption:", caption)

if __name__ == "__main__":
    main()
