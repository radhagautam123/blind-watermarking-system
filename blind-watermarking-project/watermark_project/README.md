# 🔐 Blind Image Watermarking using Deep Learning

## 📌 Overview
This project implements a **robust blind image watermarking system** using deep learning.  
The system embeds a secret watermark into an image and extracts it without needing the original image.

It is designed to be:
- Imperceptible (high image quality)
- Robust against attacks (noise, rotation, crop, etc.)
- Secure using a secret key

---

## 🚀 Features
- Deep learning-based watermark embedding and extraction
- Blind watermarking (no original image required)
- Robust against common image attacks
- Secure watermark encoding using secret key
- Streamlit UI for real-time testing

---

## 🧠 Methodology

### 🔹 Encoder
- U-Net based architecture
- Attention mechanism (Channel + Spatial Attention)
- Embeds watermark as a residual signal

### 🔹 Decoder
- Convolutional network with Spatial Transformer Network (STN)
- Recovers watermark from attacked images

### 🔹 Security Layer
- Secret key-based encoding and decoding
- Wrong key results in random output (~0.5 accuracy)

---

## ⚙️ Tech Stack
- Python
- PyTorch
- OpenCV
- NumPy
- Streamlit

---

## 📂 Project Structure
watermark_project/
│
├── models/ # Encoder, Decoder, Attention, STN
├── utils/ # Metrics, attacks, preprocessing
├── data/ # Images and watermark samples
├── weights/ # Trained models (not included in repo)
├── app/ # Streamlit UI
├── train.py # Training script
├── test.py # Evaluation script
└── README.md


---

## 🧪 Results

### ✔ Clean Image
- High PSNR (good visual quality)
- Accurate watermark extraction

### ✔ Under Attacks
| Attack     | Performance |
|-----------|------------|
| Noise     | Strong |
| Blur      | Strong |
| Resize    | Strong |
| Rotation  | Moderate |
| Crop      | Partial recovery |

---

## 🔒 Security
- Correct key → accurate watermark extraction
- Wrong key → random output (~50% accuracy)

---

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt

