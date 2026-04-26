import os
import torch
import numpy as np
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import math

from models.encoder import EncoderNet
from models.decoder import DecoderNet
from utils.security import secure_encode, secure_decode
from models.attack_layer import add_gaussian_noise, rotate_tensor, random_crop


# ================= CONFIG =================
device = 'cpu'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

ENC_PATH = os.path.join(PROJECT_ROOT, "weights", "encoder_latest.pth")
DEC_PATH = os.path.join(PROJECT_ROOT, "weights", "decoder_latest.pth")

EMBED_STRENGTH = 0.20


# 🔥 CLEAR CACHE (IMPORTANT)
st.cache_resource.clear()


# ================= LOAD MODELS =================
@st.cache_resource
def load_models():
    encoder = EncoderNet().to(device)
    decoder = DecoderNet().to(device)

    if not os.path.exists(ENC_PATH):
        st.error(f"❌ Encoder not found:\n{ENC_PATH}")
        st.stop()

    if not os.path.exists(DEC_PATH):
        st.error(f"❌ Decoder not found:\n{DEC_PATH}")
        st.stop()

    encoder.load_state_dict(torch.load(ENC_PATH, map_location=device))
    decoder.load_state_dict(torch.load(DEC_PATH, map_location=device))

    encoder.eval()
    decoder.eval()

    return encoder, decoder


encoder, decoder = load_models()


# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


# ================= UI =================
st.title("🔐 Blind Image Watermarking (Deep Learning)")

host_file = st.file_uploader("Upload Host Image", type=["png", "jpg", "jpeg"])
key = st.text_input("Enter Secret Key", type="password")

attack = st.selectbox(
    "Select Attack",
    ["none", "noise", "rotation", "crop", "strong_combined"]
)

angle = st.slider("Rotation Angle", -25, 25, 10)
noise_level = st.slider("Noise Level", 0.0, 0.1, 0.05)
crop_scale = st.slider("Crop Scale", 0.6, 1.0, 0.8)


# ================= MAIN =================
if host_file and key:

    # -------- Load Image --------
    host = Image.open(host_file).convert("RGB")
    x = transform(host).unsqueeze(0).to(device)

    # -------- Generate Watermark --------
    wm_np = np.random.randint(0, 2, (1, 32, 32))

    wm_secure = np.stack([
        secure_encode(wm_np[i], key)
        for i in range(len(wm_np))
    ])

    wm = torch.tensor(wm_secure).float().unsqueeze(1).to(device)

    # -------- EMBEDDING --------
    with torch.no_grad():
        residual = encoder(x, wm)
        watermarked = torch.clamp(x + EMBED_STRENGTH * residual, 0, 1)

    watermarked_img = (
        watermarked.squeeze()
        .permute(1, 2, 0)
        .cpu()
        .numpy() * 255
    ).astype(np.uint8)

    # -------- ATTACK --------
    attacked = watermarked.clone()

    if attack == "noise":
        attacked = add_gaussian_noise(attacked, noise_level)

    elif attack == "rotation":
        attacked = rotate_tensor(attacked, angle * math.pi / 180)

    elif attack == "crop":
        attacked = random_crop(attacked, crop_scale)

    elif attack == "strong_combined":
        attacked = rotate_tensor(attacked, angle * math.pi / 180)
        attacked = random_crop(attacked, crop_scale)
        attacked = add_gaussian_noise(attacked, noise_level)

    attacked_img = (
        attacked.squeeze()
        .permute(1, 2, 0)
        .cpu()
        .numpy() * 255
    ).astype(np.uint8)

    # -------- EXTRACTION --------
    with torch.no_grad():
        pred = decoder(attacked)

    pred_bin = (torch.sigmoid(pred) > 0.5).float().cpu().numpy()

    decoded_correct = secure_decode(pred_bin[0, 0], key)
    decoded_wrong = secure_decode(pred_bin[0, 0], "wrong_key")

    extracted_correct = (decoded_correct * 255).astype(np.uint8)
    extracted_wrong = (decoded_wrong * 255).astype(np.uint8)

    # -------- DISPLAY --------
    st.subheader("🖼️ Images")

    st.image(
        [host, watermarked_img, attacked_img],
        caption=["Original", "Watermarked", "Attacked"],
        width=250
    )

    st.subheader("🔍 Extracted Watermarks")

    st.image(
        [extracted_correct, extracted_wrong],
        caption=["Correct Key", "Wrong Key"],
        width=200
    )

    # -------- METRICS --------
    true_wm = secure_decode(wm_np[0], key)

    acc = (decoded_correct == true_wm).mean()
    wrong_acc = (decoded_wrong == true_wm).mean()
    bit_error = (decoded_correct != true_wm).mean()

    st.subheader("📊 Results")

    st.write("Accuracy:", round(float(acc), 4))
    st.write("Bit Error:", round(float(bit_error), 4))
    st.write("Wrong Key Accuracy (~0.5 expected):", round(float(wrong_acc), 4))