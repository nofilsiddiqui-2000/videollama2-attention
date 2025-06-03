import cv2
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPVisionModel

# Load CLIP Vision model and processor (ViT-Large/14, 336px) with attentions enabled
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336", output_attentions=True)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
model = model.to(device)
model.eval()

def extract_attention_heatmap(frame: np.ndarray) -> np.ndarray:
    """
    Given a BGR frame, return an attention heatmap overlay.
    """
    # Convert BGR to RGB PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Preprocess for CLIP and get attentions
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # Extract attentions from last layer
    # outputs.attentions is a tuple (layers) of shape (batch, heads, tokens, tokens)
    attn = outputs.attentions[-1]  # last layer: shape [1, H, N, N]
    # Take the CLS token (index 0) attention to all other patches (exclude index 0→0)
    cls_attn = attn[0, :, 0, 1:]   # shape [H, N-1]
    # Average over heads
    cls_attn_mean = cls_attn.mean(dim=0).cpu().numpy()  # shape [N-1]
    # The CLIP ViT-Large/14 with 336px input has patch grid 24x24 (576 patches, plus CLS)
    grid_size = int(np.sqrt(cls_attn_mean.size))
    heatmap = cls_attn_mean.reshape(grid_size, grid_size)
    # Normalize heatmap to [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    # Upsample heatmap to frame size
    heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    return heatmap_resized

def overlay_heatmap(frame: np.ndarray, heatmap: np.ndarray, alpha: float=0.6) -> np.ndarray:
    """
    Overlay a heatmap onto a BGR frame. Heatmap is [0,1] float grayscale.
    """
    # Convert heatmap to color (jet)
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    # Combine with original frame
    overlay = cv2.addWeighted(frame, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

# Input video path (modify as needed)
video_path = "assets/test_video_2.mp4"
cap = cv2.VideoCapture(video_path)
frame_idx = 0

# Prepare video writer for output (optional)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output_with_heatmap.mp4', fourcc, fps, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # (Optional) sample every nth frame for efficiency
    # if frame_idx % 5 != 0:
    #     frame_idx += 1
    #     continue

    # Compute attention heatmap for this frame
    heatmap = extract_attention_heatmap(frame)
    overlay = overlay_heatmap(frame, heatmap, alpha=0.6)

    # Save or display overlay frame
    out.write(overlay)  # write to output video
    cv2.imwrite(f"frame_{frame_idx:05d}.png", overlay)  # also save individual frames

    frame_idx += 1

cap.release()
out.release()
print("Done. Extracted", frame_idx, "frames with heatmaps.")
