import os
import cv2
import numpy as np

from utils.preprocess import load_image_rgb, load_binary_watermark
from utils.metrics import psnr, compute_ssim, normalized_correlation, ber
from utils import attacks
from classical.dwt_dct_svd import embed_watermark, extract_watermark


def save_image(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():
    host_path = "data/host_images/dog.png"
    wm_path = "data/watermarks/apple_logo.png"

    secret_key = "radha123"
    wrong_key = "wrong123"

    host = load_image_rgb(host_path, size=(256, 256))
    wm = load_binary_watermark(wm_path, size=(16, 16))

    watermarked = embed_watermark(host, wm, secret_key=secret_key)
    extracted = extract_watermark(watermarked, wm.shape, secret_key=secret_key)
    wrong_extracted = extract_watermark(watermarked, wm.shape, secret_key=wrong_key)

    os.makedirs("results", exist_ok=True)
    save_image("results/host.png", host)
    save_image("results/watermarked.png", watermarked)
    cv2.imwrite("results/original_watermark.png", (wm * 255).astype(np.uint8))
    cv2.imwrite("results/extracted_correct_key.png", (extracted * 255).astype(np.uint8))
    cv2.imwrite("results/extracted_wrong_key.png", (wrong_extracted * 255).astype(np.uint8))

    print("=== Clean Image Results ===")
    print("PSNR:", round(float(psnr(host, watermarked)), 4))
    print("SSIM:", round(float(compute_ssim(host, watermarked)), 4))
    print("Correct Key NC:", round(float(normalized_correlation(wm, extracted)), 4))
    print("Correct Key BER:", round(float(ber(wm, extracted)), 4))
    print("Wrong Key NC:", round(float(normalized_correlation(wm, wrong_extracted)), 4))
    print("Wrong Key BER:", round(float(ber(wm, wrong_extracted)), 4))

    attack_dict = {
    "none": lambda x: x,
    "jpeg50": lambda x: attacks.jpeg_compress(x, quality=50),
    "gaussian": lambda x: attacks.gaussian_noise(x, sigma=10),
    "blur": lambda x: attacks.gaussian_blur(x, sigma=1.5),
    "rotate5": lambda x: attacks.rotate_attack(x, angle=5),
    "resize50": lambda x: attacks.resize_attack(x, scale=0.5),
    "crop10": lambda x: attacks.crop_attack(x, ratio=0.1),
}

    print("\n=== Attack Results (Correct Key) ===")
    for name, fn in attack_dict.items():
        attacked = fn(watermarked)
        ext = extract_watermark(attacked, wm.shape, secret_key=secret_key)

        save_image(f"results/{name}_attacked.png", attacked)
        cv2.imwrite(f"results/{name}_extracted.png", (ext * 255).astype(np.uint8))

        print(
            name,
            {
                "NC": round(float(normalized_correlation(wm, ext)), 4),
                "BER": round(float(ber(wm, ext)), 4)
            }
        )

    print("\nSaved outputs inside the results/ folder.")


if __name__ == "__main__":
    main()