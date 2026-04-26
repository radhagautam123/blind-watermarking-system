import torch
import numpy as np
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import cv2

from models.encoder import EncoderNet
from models.decoder import DecoderNet
from utils.metrics import psnr, compute_ssim, normalized_correlation, ber
from utils import attacks


device = 'cuda' if torch.cuda.is_available() else 'cpu'


@st.cache_resource
def load_models():
    enc = EncoderNet().to(device)
    dec = DecoderNet().to(device)

    enc.load_state_dict(torch.load('weights/best_encoder.pth', map_location=device))
    dec.load_state_dict(torch.load('weights/best_decoder.pth', map_location=device))

    enc.eval()
    dec.eval()
    return enc, dec


encoder, decoder = load_models()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


st.title("🔐 Blind Watermarking System")

host_file = st.file_uploader("Upload Host Image")
wm_file = st.file_uploader("Upload Watermark")
key = st.text_input("Secret Key", type="password")

attack = st.selectbox(
    "Attack",
    ["none", "rotate", "resize", "crop", "combined"]
)

angle = st.slider("Rotation Angle", -30, 30, 10)
scale = st.slider("Scaling Factor", 0.5, 1.5, 1.0)


if host_file and wm_file and key:

    host = Image.open(host_file).convert("RGB")
    wm_img = Image.open(wm_file).convert("L").resize((32, 32))

    x = transform(host).unsqueeze(0).to(device)

    wm_np = (np.array(wm_img) > 127).astype(np.uint8)
    wm_tensor = torch.tensor(wm_np).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        watermarked = encoder(x, wm_tensor)

    watermarked_img = (watermarked.squeeze().permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

    attacked = watermarked_img.copy()

    if attack == "rotate":
        attacked, angle = attacks.rotate_attack(attacked, angle)

    elif attack == "resize":
        attacked = attacks.scaling_attack(attacked, scale)

    elif attack == "crop":
        attacked = attacks.crop_attack(attacked)

    elif attack == "combined":
        attacked = attacks.combined_attack(attacked)

    attacked = cv2.resize(attacked, (256, 256))

    attacked_tensor = transforms.ToTensor()(attacked).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = decoder(attacked_tensor)

    extracted = (pred > 0.5).squeeze().cpu().numpy() * 255

    st.image(
        [host, watermarked_img, attacked, extracted],
        caption=["Original", "Watermarked", "Attacked", "Extracted"]
    )

    st.write("PSNR:", psnr(np.array(host.resize((256,256))), watermarked_img))