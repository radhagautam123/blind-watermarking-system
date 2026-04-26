import torch
import numpy as np
import cv2
import os

from models.encoder import EncoderNet
from models.decoder import DecoderNet
from utils.ecc import encrypt_watermark_bits, decrypt_watermark_bits


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ENC_PATH = "checkpoints/encoder_latest.pth"
DEC_PATH = "checkpoints/decoder_latest.pth"

KEY = "secure_train_key"
WRONG_KEY = "wrong_key"

EMBED_STRENGTH = 0.05


# ================= LOAD MODELS =================
encoder = EncoderNet().to(DEVICE)
decoder = DecoderNet().to(DEVICE)

encoder.load_state_dict(torch.load(ENC_PATH, map_location=DEVICE))
decoder.load_state_dict(torch.load(DEC_PATH, map_location=DEVICE))

encoder.eval()
decoder.eval()


# ================= LOAD IMAGE =================
def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
    return img.to(DEVICE)


def save_image(path, tensor):
    img = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(path, img)


# ================= MAIN =================
def main():

    os.makedirs("results", exist_ok=True)

    img = load_image("data/class1/img1.jpg")

    wm_np = np.random.randint(0, 2, (1, 32, 32))

    wm_secure = encrypt_watermark_bits(wm_np[0], KEY).reshape(32, 32)
    wm = torch.tensor(wm_secure).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        residual = encoder(img, wm)
        watermarked = torch.clamp(img + EMBED_STRENGTH * residual, 0, 1)

    save_image("results/watermarked.png", watermarked)

    with torch.no_grad():
        pred = decoder(watermarked)

    pred_bin = (torch.sigmoid(pred) > 0.5).float().cpu().numpy()

    decoded = decrypt_watermark_bits(pred_bin[0, 0], KEY, (32, 32))
    decoded_wrong = decrypt_watermark_bits(pred_bin[0, 0], WRONG_KEY, (32, 32))

    cv2.imwrite("results/extracted.png", (decoded * 255).astype(np.uint8))
    cv2.imwrite("results/extracted_wrong.png", (decoded_wrong * 255).astype(np.uint8))

    print("Done! Check results folder.")


if __name__ == "__main__":
    main()