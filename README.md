# VideoLLaMA2 Attention Visualizer

This project builds on top of [DAMO-NLP-SG/VideoLLaMA2](https://github.com/DAMO-NLP-SG/VideoLLaMA2) to visualize **spatial** and **temporal attention maps** across video frames, generate captions, and overlay both using custom rendering pipelines.

> 🔬 Designed for research and debugging in video-language models (VLMs).

## 🚀 Features
- 🔍 Extract attention heatmaps from LLaMA-based video models
- 🧠 Generate video captions using LLaMA2 + CLIP + Q-Former
- 🎞 Overlay attention on original video
- 🧪 Support for spatial + temporal attention layers
- 🛠 Modular pipeline for extending to RAG or clinical VLM use cases

## 📦 Project Structure
```
videollama2/
├── VideoLLaMA2/                # Cloned official model (DAMO-NLP-SG)
├── scripts/                    # Setup/download automation
├── assets/                     # Logos, icons, visuals
├── frames*/                    # Intermediate extracted frames
├── outputs/                    # Generated overlays and videos
├── *.py                        # Core scripts (visualization, overlay, etc.)
├── .gitignore
├── README.md
└── requirements.txt
```

## 🛠 Setup Instructions
### ✅ 1. Clone This Repo
```bash
git clone https://github.com/nofilsiddiqui-2000/videollama2.git
cd videollama2
```
### 📥 2. Clone VideoLLaMA2 Core
```bash
git clone https://github.com/DAMO-NLP-SG/VideoLLaMA2.git
cd VideoLLaMA2
git checkout audio_visual
cd ..
```
### 📦 3. Set Up Environment
```bash
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install flash-attn==2.5.8 --no-build-isolation
```

## 📸 Example Commands
### 🎞 Generate Heatmaps and Captions
```bash
python videollama_attention_heatmap.py --config configs/videollama.yaml
```
### 📹 Overlay Attention on Video
```bash
python caption_video.py --input frames_v2/ --heatmap attention2.mp4 --output overlay.mp4
```

## 📋 Dependencies
```bash
pip install -r requirements.txt
sudo apt install ffmpeg   # Linux
choco install ffmpeg      # Windows (via Chocolatey)
```

## 📁 Download Models (Optional Script)
```bash
bash scripts/download_weights.sh
```

## 🧠 Notes
- This repo excludes large binaries like `.pt`, `.npy`, `.mp4` using `.gitignore`
- For reproduction, please download models and provide required configs

## 🙋 Author
**Muhammad Nofil Siddiqui**  
📍 Master’s Student @ Concordia University  
🔗 [LinkedIn](https://www.linkedin.com/in/muhammad-nofil-siddiqui/)  
📫 nofilsiddiqui2000@gmail.com

## 🧾 License
This repo is for research and academic purposes. See `VideoLLaMA2`’s license for original model usage.
