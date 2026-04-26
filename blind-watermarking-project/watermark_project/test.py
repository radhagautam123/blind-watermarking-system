import os
import torch
import numpy as np
import cv2

from models.encoder import EncoderNet
from models.decoder import DecoderNet
from utils.security import secure_encode, secure_decode
from utils.metrics import psnr, compute_ssim, normalized_correlation, ber
from utils import attacks


# ================= CONFIG =================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ENC_PATH = "weights/encoder_latest.pth"
DEC_PATH = "weights/decoder_latest.pth"

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


# ================= UTILS =================
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

    # -------- Load Image --------
    img = load_image("data/host_images/dog.png")

    # -------- Watermark --------
    wm_np = np.random.randint(0, 2, (1, 32, 32))

    wm_secure = np.stack([
        secure_encode(wm_np[i], KEY)
        for i in range(len(wm_np))
    ])

    wm = torch.tensor(wm_secure).float().unsqueeze(1).to(DEVICE)

    # -------- EMBEDDING --------
    with torch.no_grad():
        residual = encoder(img, wm)
        watermarked = torch.clamp(img + EMBED_STRENGTH * residual, 0, 1)

    save_image("results/original.png", img)
    save_image("results/watermarked.png", watermarked)

    # -------- CLEAN EXTRACTION --------
    with torch.no_grad():
        pred = decoder(watermarked)

    pred_bin = (torch.sigmoid(pred) > 0.5).float().cpu().numpy()

    decoded_correct = secure_decode(pred_bin[0, 0], KEY)
    decoded_wrong = secure_decode(pred_bin[0, 0], WRONG_KEY)

    cv2.imwrite("results/extracted_correct.png", (decoded_correct * 255).astype(np.uint8))
    cv2.imwrite("results/extracted_wrong.png", (decoded_wrong * 255).astype(np.uint8))

    # -------- METRICS --------
    true_wm = secure_decode(wm_np[0], KEY)

    acc = (decoded_correct == true_wm).mean()
    wrong_acc = (decoded_wrong == true_wm).mean()
    bit_error = (decoded_correct != true_wm).mean()

    print("\n=== CLEAN RESULTS ===")
    print("Accuracy:", round(float(acc), 4))
    print("Bit Error:", round(float(bit_error), 4))
    print("Wrong Key Accuracy (~0.5 expected):", round(float(wrong_acc), 4))

    # -------- ATTACKS --------
    attack_dict = {
        "none": lambda x: x,
        "jpeg50": lambda x: attacks.jpeg_compress(x, quality=50),
        "gaussian": lambda x: attacks.gaussian_noise(x, sigma=10),
        "blur": lambda x: attacks.gaussian_blur(x, sigma=1.5),
        "rotate5": lambda x: attacks.rotate_attack(x, angle=5),
        "resize50": lambda x: attacks.resize_attack(x, scale=0.5),
        "crop10": lambda x: attacks.crop_attack(x, ratio=0.1),
    }

    print("\n=== ATTACK RESULTS ===")

    for name, fn in attack_dict.items():

        attacked = fn(watermarked.squeeze().permute(1,2,0).cpu().numpy())
        attacked = cv2.resize(attacked, (256, 256))

        attacked_tensor = torch.tensor(attacked / 255.0).permute(2,0,1).unsqueeze(0).float().to(DEVICE)

        with torch.no_grad():
            pred = decoder(attacked_tensor)

        pred_bin = (torch.sigmoid(pred) > 0.5).float().cpu().numpy()
        decoded = secure_decode(pred_bin[0, 0], KEY)

        nc = normalized_correlation(true_wm, decoded)
        b_error = ber(true_wm, decoded)

        cv2.imwrite(f"results/{name}_extracted.png", (decoded * 255).astype(np.uint8))

        print(name, {
            "NC": round(float(nc), 4),
            "BER": round(float(b_error), 4)
        })

    print("\nSaved outputs inside results/")


if __name__ == "__main__":
    main()