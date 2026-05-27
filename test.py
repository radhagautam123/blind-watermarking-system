import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch

from models.encoder import EncoderNet
from models.decoder import DecoderNet
from models.attack_layer import attack_suite
from utils.metrics import normalized_correlation
from utils.wm_pipeline import (
    decode_from_logits,
    encode_for_embedding,
    key_to_tensor,
    prepare_plain_watermark,
    wm_secure_to_tensor,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WM_SIZE = 32
KEY = "secure_key"
WRONG_KEY = "wrong_key"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CKPT_PATH = CHECKPOINT_DIR / "checkpoint_best_ber.pth"
IMAGE_PATH = DATA_DIR / "host_images" / "dog.png"
WATERMARK_PATH = DATA_DIR / "watermarks" / "apple_logo.png"


def load_image(path, size=(256, 256)):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    return tensor


def load_binary_watermark_image(path, size=(WM_SIZE, WM_SIZE)):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Watermark image not found: {path}")

    wm = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if wm is None:
        raise FileNotFoundError(f"Could not read watermark image: {path}")

    wm = cv2.resize(wm, size)
    _, wm = cv2.threshold(wm, 127, 1, cv2.THRESH_BINARY)
    return wm.astype(np.uint8)


def tensor_to_rgb_np(tensor):
    x = tensor.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return x


def save_image(path, tensor):
    img = tensor_to_rgb_np(tensor)
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img)


def save_binary_image(path, arr):
    arr = (arr.astype(np.uint8) * 255)
    cv2.imwrite(str(path), arr)


def save_heatmap(path, logits):
    x = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
    x = (x * 255).astype(np.uint8)
    x = cv2.applyColorMap(x, cv2.COLORMAP_JET)
    cv2.imwrite(str(path), x)


def psnr(img1, img2, eps=1e-8):
    mse = np.mean((img1 - img2) ** 2)
    if mse <= eps:
        return 99.0
    return 10.0 * np.log10(1.0 / mse)


def ssim(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if img1.ndim == 3:
        scores = [ssim(img1[..., c], img2[..., c]) for c in range(img1.shape[2])]
        return float(np.mean(scores))

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    num = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    return float((num / (den + 1e-8)).mean())


def ber_and_acc(true_wm, pred_wm):
    true_bits = (true_wm > 0).astype(np.uint8)
    pred_bits = (pred_wm > 0).astype(np.uint8)
    ber = np.mean(true_bits != pred_bits)
    acc = 1.0 - ber
    return float(ber), float(acc)


def safe_load_state_dict(model, state_dict, name):
    incompatible = model.load_state_dict(state_dict, strict=False)
    print(f"\n[{name}]")
    print("Missing keys   :", incompatible.missing_keys)
    print("Unexpected keys:", incompatible.unexpected_keys)
    return model


def _torch_load(path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def load_encoder_decoder():
    print("Using device:", DEVICE)
    print("Project root :", PROJECT_ROOT)
    print("Checkpoint   :", CKPT_PATH.resolve())
    print("Image path   :", IMAGE_PATH.resolve())
    print("Watermark    :", WATERMARK_PATH.resolve())

    if not CKPT_PATH.exists():
        fallback = CHECKPOINT_DIR / "checkpoint_latest.pth"
        if fallback.exists():
            ckpt_path = fallback
            print(f"Best BER checkpoint missing, using: {fallback}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
    else:
        ckpt_path = CKPT_PATH

    encoder = EncoderNet(wm_size=WM_SIZE).to(DEVICE)
    decoder = DecoderNet(wm_size=WM_SIZE).to(DEVICE)

    ckpt = _torch_load(ckpt_path)

    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Checkpoint is not a dict: {ckpt_path}")
    if "encoder" not in ckpt or "decoder" not in ckpt:
        raise RuntimeError(f"Checkpoint missing 'encoder' or 'decoder' keys: {ckpt_path}")

    encoder = safe_load_state_dict(encoder, ckpt["encoder"], "Encoder")
    decoder = safe_load_state_dict(decoder, ckpt["decoder"], "Decoder")

    encoder.eval()
    decoder.eval()
    return encoder, decoder


def evaluate_one(name, attacked_tensor, decoder, true_wm, embed_key=KEY, extract_key=KEY):
    with torch.no_grad():
        pred = decoder(attacked_tensor, key_to_tensor(extract_key, device=DEVICE))
        wrong_pred = decoder(attacked_tensor, key_to_tensor(WRONG_KEY, device=DEVICE))

    stages = decode_from_logits(pred, extract_key, use_decrypt=True, use_ecc=True)
    wrong_stages = decode_from_logits(wrong_pred, WRONG_KEY, use_decrypt=True, use_ecc=True)

    raw_bits = stages["raw_bits"]
    decrypted_bits = stages["decrypted_bits"]
    final_bits = stages["final_bits"]
    wrong_final = wrong_stages["final_bits"]

    raw_ber, raw_acc = ber_and_acc(true_wm, raw_bits)
    dec_ber, dec_acc = ber_and_acc(true_wm, decrypted_bits)
    final_ber, final_acc = ber_and_acc(true_wm, final_bits)
    wrong_key_similarity = float(np.mean(wrong_final == true_wm))
    ncc = normalized_correlation(true_wm, final_bits)

    save_image(RESULTS_DIR / f"attacked_{name}.png", attacked_tensor)
    save_binary_image(RESULTS_DIR / f"rawbits_{name}.png", raw_bits)
    save_binary_image(RESULTS_DIR / f"decrypted_{name}.png", decrypted_bits)
    save_binary_image(RESULTS_DIR / f"decoded_{name}.png", final_bits)
    save_binary_image(RESULTS_DIR / f"decoded_wrong_{name}.png", wrong_final)
    save_heatmap(RESULTS_DIR / f"logits_{name}.png", pred)

    return {
        "attack": name,
        "raw_ber": raw_ber,
        "raw_acc": raw_acc,
        "dec_ber": dec_ber,
        "dec_acc": dec_acc,
        "final_ber": final_ber,
        "final_acc": final_acc,
        "ncc": ncc,
        "wrong_key_similarity": wrong_key_similarity,
    }


def main():
    encoder, decoder = load_encoder_decoder()

    img = load_image(IMAGE_PATH)
    wm_np = prepare_plain_watermark(load_binary_watermark_image(WATERMARK_PATH))
    wm_secure = encode_for_embedding(wm_np, KEY)
    wm_tensor = wm_secure_to_tensor(wm_secure, device=DEVICE)
    key_tensor = key_to_tensor(KEY, device=DEVICE)

    with torch.no_grad():
        residual = encoder(img, wm_tensor, key_tensor)
        watermarked = torch.clamp(img + residual, 0, 1)

    save_image(RESULTS_DIR / "host.png", img)
    save_image(RESULTS_DIR / "watermarked.png", watermarked)
    save_binary_image(RESULTS_DIR / "watermark_original.png", wm_np)

    host_np = tensor_to_rgb_np(img)
    watermarked_np = tensor_to_rgb_np(watermarked)

    print("\nImperceptibility:")
    print(f"PSNR : {psnr(host_np, watermarked_np):.2f} dB")
    print(f"SSIM : {ssim(host_np, watermarked_np):.4f}")

    results = []
    results.append(evaluate_one("clean", watermarked, decoder, wm_np))

    suite = attack_suite(watermarked.clone())
    for attack_name, attacked in suite.items():
        if attack_name == "clean":
            continue
        attacked = torch.clamp(attacked, 0, 1)
        results.append(evaluate_one(attack_name, attacked, decoder, wm_np))

    print("\nExtraction results:")
    for r in results:
        print(
            f"{r['attack']:>12s} | "
            f"Raw BER: {r['raw_ber']:.4f} | "
            f"Dec BER: {r['dec_ber']:.4f} | "
            f"Final BER: {r['final_ber']:.4f} | "
            f"NCC: {r['ncc']:.4f} | "
            f"Final ACC: {r['final_acc']:.4f} | "
            f"Wrong-key sim: {r['wrong_key_similarity']:.4f}"
        )

    print("\nSaved outputs in:", RESULTS_DIR.resolve())


if __name__ == "__main__":
    main()
