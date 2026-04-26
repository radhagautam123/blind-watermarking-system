import os
import torch
import numpy as np
import cv2
import math

from models.encoder import EncoderNet
from models.decoder import DecoderNet
from utils.security import secure_encode, secure_decode

from models.attack_layer import (
    add_gaussian_noise,
    rotate_tensor,
    random_crop
)

# ================= CONFIG =================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ENCODER_PATH = "checkpoints/encoder_latest.pth"
DECODER_PATH = "checkpoints/decoder_latest.pth"

KEY = "secure_train_key"
WRONG_KEY = "wrong_key_123"

EMBED_STRENGTH = 0.20


# ================= LOAD MODEL =================
encoder = EncoderNet().to(DEVICE)
decoder = DecoderNet().to(DEVICE)

encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
decoder.load_state_dict(torch.load(DECODER_PATH, map_location=DEVICE))

encoder.eval()
decoder.eval()


# ================= LOAD IMAGE =================
def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
    return img.to(DEVICE)


def save_image(path, tensor):
    img = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# ================= MAIN =================
def main():

    os.makedirs("results", exist_ok=True)

    img = load_image("data/host_images/dog.png")

    # ================= WATERMARK =================
    wm_np = np.random.randint(0, 2, (1, 32, 32))

    wm_secure = np.stack([
        secure_encode(wm_np[i], KEY)
        for i in range(len(wm_np))
    ])

    wm = torch.tensor(wm_secure).float().unsqueeze(1).to(DEVICE)

    # ================= EMBEDDING =================
    with torch.no_grad():
        residual = encoder(img, wm)
        watermarked = torch.clamp(img + EMBED_STRENGTH * residual, 0, 1)

    save_image("results/original.png", img)
    save_image("results/watermarked.png", watermarked)

    # ================= STRONG ATTACK =================
    attacked = watermarked
    attacked = rotate_tensor(attacked, 12 * math.pi / 180)
    attacked = random_crop(attacked, 0.8)
    attacked = add_gaussian_noise(attacked, 0.05)

    save_image("results/attacked.png", attacked)

    # ================= EXTRACTION =================
    with torch.no_grad():
        pred = decoder(attacked)

    pred_bin = (torch.sigmoid(pred) > 0.5).float().cpu().numpy()
    wm_bin = wm.cpu().numpy()

    decoded_correct = secure_decode(pred_bin[0, 0], KEY)
    decoded_wrong = secure_decode(pred_bin[0, 0], WRONG_KEY)

    cv2.imwrite("results/extracted_correct.png", (decoded_correct * 255).astype(np.uint8))
    cv2.imwrite("results/extracted_wrong.png", (decoded_wrong * 255).astype(np.uint8))

    # ================= METRICS =================
    acc = (decoded_correct == secure_decode(wm_bin[0, 0], KEY)).mean()
    wrong_acc = (decoded_wrong == secure_decode(wm_bin[0, 0], KEY)).mean()

    print("\n=== FINAL RESULTS ===")
    print("Accuracy:", round(float(acc), 4))
    print("Wrong Key Accuracy:", round(float(wrong_acc), 4))


if __name__ == "__main__":
    main()