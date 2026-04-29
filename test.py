import torch
import numpy as np
import cv2
import os

from models.encoder import EncoderNet
from models.decoder import DecoderNet
from utils.ecc import encrypt_watermark_bits, decrypt_watermark_bits
from utils.error_correction import encode_bch, decode_bch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", DEVICE)

ENC_PATH = "checkpoints/encoder.pth"
DEC_PATH = "checkpoints/decoder.pth"

KEY = "secure_key"
WRONG_KEY = "wrong_key"

# ---------------- KEY ----------------
import hashlib

def key_to_tensor(key, size=32):
    hash_bytes = b''
    current = key.encode()

    while len(hash_bytes) < size * size:
        current = hashlib.sha256(current).digest()
        hash_bytes += current

    arr = np.frombuffer(hash_bytes[:size*size], dtype=np.uint8)
    arr = arr.astype(np.float32) / 255.0

    return torch.tensor(arr).view(1,1,size,size)


# ---------------- LOAD MODELS ----------------
encoder = EncoderNet().to(DEVICE)
decoder = DecoderNet().to(DEVICE)

encoder.load_state_dict(torch.load(ENC_PATH, map_location=DEVICE))
decoder.load_state_dict(torch.load(DEC_PATH, map_location=DEVICE))

encoder.eval()
decoder.eval()


# ---------------- IMAGE ----------------
def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (256,256))
    img = img / 255.0
    img = torch.tensor(img).permute(2,0,1).unsqueeze(0).float()
    return img.to(DEVICE)

def save_image(path, tensor):
    img = tensor.squeeze().permute(1,2,0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(path, img)


# ---------------- MAIN ----------------
def main():

    os.makedirs("results", exist_ok=True)

    img = load_image("data/class1/img1.jpg")

    # -------- WATERMARK --------
    wm_np = np.random.randint(0,2,(32,32)).astype(np.uint8)

    # 🔥 BCH + XOR
    wm_encoded = encode_bch(wm_np)
    wm_secure = encrypt_watermark_bits(wm_encoded, KEY)

    wm = torch.tensor(wm_secure).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

    key_tensor = key_to_tensor(KEY).to(DEVICE)

    # -------- EMBEDDING --------
    with torch.no_grad():
        residual = encoder(img, wm, key_tensor)
        watermarked = torch.clamp(img + residual, 0, 1)

    save_image("results/watermarked.png", watermarked)

    # -------- EXTRACTION --------
    with torch.no_grad():
        pred = decoder(watermarked, key_tensor)

    pred_bin = (torch.sigmoid(pred) > 0.5).float().cpu().numpy()

    # 🔓 XOR + BCH decode
    decoded = decrypt_watermark_bits(pred_bin[0,0], KEY, (32,32))
    decoded = decode_bch(decoded)

    decoded_wrong = decrypt_watermark_bits(pred_bin[0,0], WRONG_KEY, (32,32))
    decoded_wrong = decode_bch(decoded_wrong)

    cv2.imwrite("results/extracted.png", (decoded*255).astype(np.uint8))
    cv2.imwrite("results/extracted_wrong.png", (decoded_wrong*255).astype(np.uint8))

    print("Done! Check results folder.")

if __name__ == "__main__":
    main()
