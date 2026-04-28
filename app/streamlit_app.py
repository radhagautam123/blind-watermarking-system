import sys, os, hashlib, math, io
import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2

# ================= PATH =================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.encoder import EncoderNet
from models.decoder import DecoderNet
from utils.ecc import encrypt_watermark_bits, decrypt_watermark_bits

# ================= CONFIG =================
st.set_page_config(layout="wide", page_title="🔐 Blind Watermarking")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ENC_PATH = os.path.join(PROJECT_ROOT, "weights", "encoder.pth")
DEC_PATH = os.path.join(PROJECT_ROOT, "weights", "decoder.pth")

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

# ================= KEY =================
def key_to_tensor(key, size=32):
    hash_bytes = b''
    current = key.encode()

    while len(hash_bytes) < size*size:
        current = hashlib.sha256(current).digest()
        hash_bytes += current

    arr = np.frombuffer(hash_bytes[:size*size], dtype=np.uint8)
    arr = arr.astype(np.float32)/255.0
    return torch.tensor(arr).view(1,1,size,size)

# ================= METRICS =================
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return 20 * np.log10(255.0 / np.sqrt(mse + 1e-8))

def ssim(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    C1, C2 = 6.5, 58.5

    mu1, mu2 = img1.mean(), img2.mean()
    sigma1, sigma2 = img1.var(), img2.var()
    sigma12 = ((img1-mu1)*(img2-mu2)).mean()

    return ((2*mu1*mu2+C1)*(2*sigma12+C2))/((mu1**2+mu2**2+C1)*(sigma1+sigma2+C2))

# ================= LOAD MODELS =================
@st.cache_resource
def load_models():
    enc = EncoderNet().to(device)
    dec = DecoderNet().to(device)

    enc.load_state_dict(torch.load(ENC_PATH, map_location=device))
    dec.load_state_dict(torch.load(DEC_PATH, map_location=device))

    enc.eval()
    dec.eval()
    return enc, dec

encoder, decoder = load_models()

# ================= UI =================
st.title("🔐 Blind Watermarking System")

st.sidebar.header("⚙️ Controls")
secret_key = st.sidebar.text_input("🔑 Secret Key", type="password")

attack = st.sidebar.selectbox("Attack", [
    "None","Gaussian Noise","JPEG","Rotation","Resize","Crop",
    "Blur","Brightness","Contrast","SaltPepper","Sharpen"
])

noise = st.sidebar.slider("Noise",0.0,0.2,0.05)
jpeg_q = st.sidebar.slider("JPEG Quality",10,100,50)
angle = st.sidebar.slider("Rotation",-45,45,10)
scale = st.sidebar.slider("Resize",0.3,1.0,0.7)
crop_scale = st.sidebar.slider("Crop",0.5,1.0,0.8)
blur_k = st.sidebar.slider("Blur",1,9,5,step=2)
brightness_f = st.sidebar.slider("Brightness",0.5,2.0,1.2)
contrast_f = st.sidebar.slider("Contrast",0.5,2.0,1.2)
sp_p = st.sidebar.slider("SaltPepper",0.0,0.1,0.02)

# ================= INPUT =================
col1, col2 = st.columns(2)

with col1:
    host_file = st.file_uploader("Upload Host Image")

with col2:
    wm_file = st.file_uploader("Upload Watermark (32x32)")

# ================= EMBEDDING =================
if st.button("🚀 Embed Watermark"):

    if not host_file or not wm_file or not secret_key:
        st.error("Provide all inputs")
    else:
        host = Image.open(host_file).convert("RGB")
        wm_img = Image.open(wm_file).convert("L")

        x = transform(host).unsqueeze(0).to(device)

        # 🔥 FIXED WATERMARK HANDLING
        wm_np = np.array(wm_img)
        wm_np = (wm_np > 127).astype(np.float32)

        if wm_np.shape != (32,32):
            wm_np = cv2.resize(wm_np, (32,32))

        wm_np = wm_np.reshape(32,32)

        wm_secure = encrypt_watermark_bits(wm_np, secret_key)
        wm = torch.tensor(wm_secure, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        key_tensor = key_to_tensor(secret_key).to(device)

        with torch.no_grad():
            residual = encoder(x, wm, key_tensor)
            watermarked = torch.clamp(x + residual,0,1)

        st.session_state["wm"] = wm_np
        st.session_state["watermarked"] = watermarked
        st.session_state["host"] = np.array(host.resize((256,256)))

        out = (watermarked.squeeze().permute(1,2,0).cpu().numpy()*255).astype(np.uint8)

        st.image(out, caption="Watermarked")

        buf = io.BytesIO()
        Image.fromarray(out).save(buf, format="PNG")
        st.download_button("⬇ Download Watermarked", buf.getvalue(), "watermarked.png")

# ================= EXTRACTION =================
extract_key = st.text_input("🔑 Extraction Key", type="password")

if st.button("🔍 Extract Watermark"):

    if "watermarked" not in st.session_state:
        st.error("First embed watermark")
    else:
        attacked = st.session_state["watermarked"].clone()

        # ================= ATTACKS =================
        if attack=="Gaussian Noise":
            attacked = torch.clamp(attacked + torch.randn_like(attacked)*noise,0,1)

        elif attack=="JPEG":
            img = (attacked.squeeze().permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
            _, enc = cv2.imencode('.jpg', img,[int(cv2.IMWRITE_JPEG_QUALITY),jpeg_q])
            dec = cv2.imdecode(enc,1)
            attacked = torch.tensor(dec/255.).permute(2,0,1).unsqueeze(0).float()

        elif attack=="Rotation":
            attacked = transforms.functional.rotate(attacked, angle)

        elif attack=="Resize":
            attacked = F.interpolate(attacked, scale_factor=scale)
            attacked = F.interpolate(attacked, size=(256,256))

        elif attack=="Crop":
            attacked = attacked[:,:,0:int(256*crop_scale),0:int(256*crop_scale)]
            attacked = F.interpolate(attacked,size=(256,256))

        elif attack=="Blur":
            attacked = transforms.functional.gaussian_blur(attacked, blur_k)

        elif attack=="Brightness":
            attacked = torch.clamp(attacked*brightness_f,0,1)

        elif attack=="Contrast":
            m = attacked.mean()
            attacked = torch.clamp((attacked-m)*contrast_f+m,0,1)

        elif attack=="SaltPepper":
            noise = torch.rand_like(attacked)
            attacked = attacked.clone()
            attacked[noise<sp_p] = 0
            attacked[noise>1-sp_p] = 1

        elif attack=="Sharpen":
            kernel = torch.tensor([[0,-1,0],[-1,5,-1],[0,-1,0]]).float().to(device)
            kernel = kernel.view(1,1,3,3).repeat(3,1,1,1)
            attacked = F.conv2d(attacked, kernel, padding=1, groups=3)

        # ================= DECODE =================
        key_tensor = key_to_tensor(extract_key).to(device)

        with torch.no_grad():
            pred = decoder(attacked, key_tensor)

        pred_bin = (torch.sigmoid(pred)>0.5).float().cpu().numpy()

        decoded = decrypt_watermark_bits(pred_bin[0,0], extract_key,(32,32))
        extracted = (decoded*255).astype(np.uint8)

        st.image(extracted, caption="Extracted", width=200)

        buf = io.BytesIO()
        Image.fromarray(extracted).save(buf, format="PNG")
        st.download_button("⬇ Download Extracted", buf.getvalue(), "extracted.png")

        # ================= METRICS =================
        wm_true = st.session_state["wm"]

        ber = np.mean(decoded!=wm_true)
        acc = 1-ber

        psnr_val = psnr(st.session_state["host"], 
                       (st.session_state["watermarked"].squeeze().permute(1,2,0).cpu().numpy()*255))

        ssim_val = ssim(st.session_state["host"], 
                       (st.session_state["watermarked"].squeeze().permute(1,2,0).cpu().numpy()*255))

        st.subheader("📊 Metrics")
        st.write(f"PSNR: {psnr_val:.2f}")
        st.write(f"SSIM: {ssim_val:.4f}")
        st.write(f"BER: {ber:.4f}")
        st.write(f"Accuracy: {acc:.4f}")

        if ber > 0.3:
            st.error("❌ Wrong Key / Strong Attack")
        else:
            st.success("✅ Correct Key")