import torch
from transformers import CLIPVisionModel, CLIPProcessor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load CLIP ViT-L/14-336 vision model
model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
model.eval()

def compute_attention_heatmap(image, use_rollout=True):
    """
    Compute a spatial attention heatmap for a PIL image using CLIP ViT.
    If use_rollout=True, uses attention rollout; otherwise uses final-layer CLS-attention.
    Returns a numpy array of shape (H, W) with values in [0,1].
    """
    # Prepare image for CLIP (this resizes to 336x336 and normalizes)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    # outputs.attentions is a tuple (num_layers,) each of shape [batch, heads, tokens, tokens]
    attn_weights = outputs.attentions  # tuple of length 24 for ViT-L
    # Convert list of layer attentions to a stack (layers, heads, tokens, tokens)
    # For batch=1:
    attn = torch.stack(attn_weights, dim=0)  # shape [num_layers, batch, heads, tokens, tokens]
    attn = attn[:, 0]  # remove batch dim -> [num_layers, heads, tokens, tokens]
    num_layers, num_heads, num_tokens, _ = attn.shape
    
    if use_rollout:
        # Perform attention rollout (Abnar & Zuidema 2020)
        # Start with identity (size = tokens x tokens)
        result = torch.eye(num_tokens, num_tokens)
        result = result.to(attn.device)
        for layer in range(num_layers):
            # average over heads
            avg_attn = attn[layer].mean(dim=0)  # [tokens, tokens]
            # Add identity to account for residual connection
            avg_attn = avg_attn + torch.eye(num_tokens).to(avg_attn.device)
            # Normalize rows to sum to 1
            avg_attn = avg_attn / avg_attn.sum(dim=-1, keepdim=True)
            # matrix multiply: accumulate
            result = torch.matmul(avg_attn, result)
        # Now result represents effective attention from each token to each token
        # We take the [CLS] token (index 0) row for patch contributions
        cls_rollout = result[0, 1:].reshape(1, 1, 24, 24)  # shape [1,1,24,24]
        heatmap = cls_rollout.squeeze().cpu().numpy()
    else:
        # Use final layer [CLS] -> patches attention
        final_attn = attn[-1]  # [heads, tokens, tokens]
        # Average over heads
        final_attn = final_attn.mean(dim=0)  # [tokens, tokens]
        # We want weights *from* [CLS] (token 0) *to* each patch token (1..)
        cls_attn = final_attn[0, 1:]  # [tokens-1]
        heatmap = cls_attn.reshape(24, 24).cpu().numpy()

    # Upsample heatmap to image size
    heatmap_img = Image.fromarray(heatmap)  # currently 24x24
    heatmap_img = heatmap_img.resize(image.size, resample=Image.BILINEAR)
    heatmap = np.array(heatmap_img, dtype=np.float32)

    # Normalize to [0,1]
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap

def overlay_heatmap_on_image(image, heatmap, alpha=0.5, colormap=plt.cm.inferno):
    """
    Overlays a heatmap (H x W, normalized [0,1]) on the image and displays/saves it.
    """
    plt.figure(figsize=(6,6))
    plt.imshow(image)
    plt.imshow(heatmap, cmap=colormap, alpha=alpha)
    plt.axis('off')
    plt.show()

# Example usage:
image_path = "image\img1.jpg"  # replace with your frame path
image = Image.open(image_path).convert("RGB")
heatmap = compute_attention_heatmap(image, use_rollout=True)
overlay_heatmap_on_image(image, heatmap)
