import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

import torch
import numpy as np
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import cv2

from models.encoder import EncoderNet
from models.decoder import DecoderNet
from utils.security import secure_encode, secure_decode
from utils.metrics import (
    psnr,
    compute_ssim,
    normalized_correlation,
    ber,
    watermark_accuracy,
    overall_performance
)
from utils import attacks

# ================= CONFIG =================
st.set_page_config(page_title="Secure Watermarking", layout="wide")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ================= LOAD MODELS =================
@st.cache_resource
def load_models():
    encoder = EncoderNet().to(device)
    decoder = DecoderNet().to(device)

    enc_path = os.path.join(BASE_DIR, 'weights', 'encoder_best.pth')
    dec_path = os.path.join(BASE_DIR, 'weights', 'decoder_best.pth')

    encoder.load_state_dict(torch.load(enc_path, map_location=device))
    decoder.load_state_dict(torch.load(dec_path, map_location=device))

    encoder.eval()
    decoder.eval()

    return encoder, decoder

encoder, decoder = load_models()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ================= HEADER =================
st.title("🔐 Secure Blind Image Watermarking")
st.caption("Deep Learning • Key-Protected • Robust to Attacks")

st.markdown("---")

# ================= SIDEBAR =================
st.sidebar.header("⚙️ Control Panel")

embed_key = st.sidebar.text_input("🔑 Embedding Key", type="password")
extract_key = st.sidebar.text_input("🔓 Extraction Key", type="password")

attack_name = st.sidebar.selectbox(
    "Attack Type",
    ["none", "jpeg", "gaussian", "blur", "rotate", "resize", "crop", "zoom", "salt_pepper"]
)

run = st.sidebar.button("🚀 Run Watermarking")

# ================= INPUT =================
st.subheader("📥 Upload Images")

col1, col2 = st.columns(2)

with col1:
    host_file = st.file_uploader("Host Image", type=["png", "jpg", "jpeg"])

with col2:
    wm_file = st.file_uploader("Watermark Image", type=["png", "jpg", "jpeg"])

st.markdown("---")

# ================= MAIN =================
if run:

    if not host_file or not wm_file:
        st.error("❌ Please upload both images")
        st.stop()

    if not embed_key.strip() or not extract_key.strip():
        st.error("❌ Both keys are required")
        st.stop()

    # LOAD
    host = Image.open(host_file).convert("RGB").resize((256, 256))
    wm_img = Image.open(wm_file).convert("L").resize((32, 32))

    x = transform(host).unsqueeze(0).to(device)

    wm_np = (np.array(wm_img) > 127).astype(np.uint8)
    wm_np = 1 - wm_np

    # 🔐 SECURE ENCODE
    wm_secure = secure_encode(wm_np, embed_key)
    wm_tensor = torch.tensor(wm_secure).float().unsqueeze(0).unsqueeze(0).to(device)

    # ENCODE
    with torch.no_grad():
        residual = encoder(x, wm_tensor)
        watermarked = torch.clamp(x + 0.15 * residual, 0, 1)

    watermarked_img = (watermarked.squeeze().permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

    # ATTACK
    attacked = watermarked_img.copy()

    if attack_name == "jpeg":
        attacked = attacks.jpeg_compress(attacked, 50)
    elif attack_name == "gaussian":
        attacked = attacks.gaussian_noise(attacked, 10)
    elif attack_name == "blur":
        attacked = attacks.gaussian_blur(attacked, 5)
    elif attack_name == "rotate":
        attacked, _ = attacks.rotate_attack(attacked, 15)
    elif attack_name == "resize":
        attacked = attacks.scaling_attack(attacked, 0.8)
    elif attack_name == "crop":
        attacked = attacks.crop_attack(attacked, 0.5)
    elif attack_name == "zoom":
        attacked = attacks.zoom_attack(attacked, 1.2)
    elif attack_name == "salt_pepper":
        attacked = attacks.salt_pepper_noise(attacked, 0.02)

    attacked = cv2.resize(attacked, (256, 256))

    # DECODE
    attacked_tensor = transforms.ToTensor()(attacked).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = decoder(attacked_tensor)

    extracted = (pred > 0.5).squeeze().cpu().numpy().astype(np.uint8)

    # 🔐 DECODE WITH KEY
    extracted = secure_decode(extracted, extract_key)
    extracted_display = extracted * 255

    # ================= RESULTS =================
    st.subheader("🖼 Results")

    c1, c2, c3, c4 = st.columns(4)

    c1.image(host, caption="Original")
    c2.image(watermarked_img, caption="Watermarked")
    c3.image(attacked, caption=f"Attacked ({attack_name})")
    c4.image(extracted_display, caption="Extracted")

    st.markdown("---")

    # ================= METRICS =================
    st.subheader("📊 Performance")

    original_np = np.array(host)

    psnr_val = psnr(original_np, watermarked_img)
    ssim_val = compute_ssim(original_np, watermarked_img)
    nc = normalized_correlation(wm_np, extracted)
    ber_val = ber(wm_np, extracted)
    acc = watermark_accuracy(wm_np, extracted)
    score = overall_performance(psnr_val, ssim_val, nc, acc)

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("PSNR", f"{psnr_val:.2f} dB")
    m2.metric("SSIM", f"{ssim_val:.4f}")
    m3.metric("Accuracy", f"{acc*100:.2f}%")
    m4.metric("Overall Score", f"{score:.4f}")

    st.markdown("---")

    # ================= DOWNLOAD =================
    st.subheader("⬇️ Download")

    d1, d2 = st.columns(2)

    d1.download_button(
        "Download Watermarked Image",
        data=cv2.imencode(".png", watermarked_img)[1].tobytes(),
        file_name="watermarked.png"
    )

    d2.download_button(
        "Download Extracted Watermark",
        data=cv2.imencode(".png", extracted_display)[1].tobytes(),
        file_name="extracted.png"
    )

    # ================= KEY VALIDATION =================
    if acc > 0.9:
        st.success("✅ Correct Key — Watermark Recovered")
    else:
        st.error("❌ Incorrect Key — Extraction Failed")

else:
    st.info("Upload images, enter keys from sidebar, and run.")