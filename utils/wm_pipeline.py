"""Shared watermark encode/decode pipeline for train, test, and Streamlit."""
import hashlib

import numpy as np
import torch
import torch.nn.functional as F

from utils.ecc import decrypt_watermark_bits, encrypt_watermark_bits
from utils.error_correction import decode_bch, encode_bch

WM_SIZE = 32


def key_to_tensor(key, size=WM_SIZE, device="cpu"):
    hash_bytes = b""
    current = key.encode()
    while len(hash_bytes) < size * size:
        current = hashlib.sha256(current).digest()
        hash_bytes += current
    arr = np.frombuffer(hash_bytes[: size * size], dtype=np.uint8).astype(np.float32) / 255.0
    return torch.tensor(arr, device=device).view(1, 1, size, size)


def prepare_plain_watermark(wm_np):
    wm = np.asarray(wm_np).astype(np.uint8)
    wm = (wm > 0).astype(np.uint8)
    if wm.size != WM_SIZE * WM_SIZE:
        wm = wm.reshape(-1)[: WM_SIZE * WM_SIZE]
        wm = np.pad(wm, (0, WM_SIZE * WM_SIZE - len(wm)), mode="constant")
    return wm.reshape(WM_SIZE, WM_SIZE)


def encode_for_embedding(wm_plain, secret_key):
    wm_plain = prepare_plain_watermark(wm_plain)
    wm_encoded = encode_bch(wm_plain)
    wm_secure = encrypt_watermark_bits(wm_encoded, secret_key)
    return wm_secure.astype(np.uint8)


def wm_secure_to_tensor(wm_secure, device="cpu"):
    t = torch.from_numpy(wm_secure.astype(np.float32)).view(1, 1, WM_SIZE, WM_SIZE)
    return t.to(device)


def logits_to_bits(logits, threshold=0.0):
    return (logits.detach().cpu().numpy()[0, 0] > threshold).astype(np.uint8)


def crop_rescale_candidates(img, scales=(1.0, 0.94, 0.88, 0.82)):
    """Build several center-cropped, re-scaled views of an image for crop-robust extraction."""
    if img.ndim != 4:
        raise ValueError("Expected a batch tensor with shape [B, C, H, W].")

    _, _, h, w = img.shape
    candidates = []
    for scale in scales:
        crop_h = max(32, int(h * scale))
        crop_w = max(32, int(w * scale))
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        patch = img[:, :, top:top + crop_h, left:left + crop_w]
        if patch.shape[-2:] != (h, w):
            patch = F.interpolate(patch, size=(h, w), mode="bilinear", align_corners=False)
        candidates.append(patch)
    return candidates


def extract_with_crop_search(decoder, img, secret_key, key_tensor, scales=(1.0, 0.94, 0.88, 0.82),
                              wrong_key_tensor=None, use_decrypt=True, use_ecc=True):
    """Return the most confident crop-rescaled decode result for robustness under crop/translation attacks."""
    if wrong_key_tensor is None:
        wrong_key_tensor = key_to_tensor("wrong_key_probe", size=WM_SIZE, device=img.device)

    best = None
    best_score = -1e9

    for candidate in crop_rescale_candidates(img, scales=scales):
        with torch.no_grad():
            logits = decoder(candidate, key_tensor)
            wrong_logits = decoder(candidate, wrong_key_tensor)
            stages = decode_from_logits(logits, secret_key, use_decrypt=use_decrypt, use_ecc=use_ecc)
            probs = torch.sigmoid(logits)
            wrong_probs = torch.sigmoid(wrong_logits)
            score = float((probs.mean() - wrong_probs.mean() + 0.10 * torch.abs(probs - 0.5).mean()).item())

        if score > best_score:
            best_score = score
            best = {
                "candidate": candidate,
                "logits": logits,
                "stages": stages,
                "score": score,
            }

    if best is None:
        with torch.no_grad():
            logits = decoder(img, key_tensor)
            stages = decode_from_logits(logits, secret_key, use_decrypt=use_decrypt, use_ecc=use_ecc)
        return img, logits, stages, float(torch.sigmoid(logits).mean().item())

    return best["candidate"], best["logits"], best["stages"], best["score"]


def decode_from_logits(logits, secret_key, use_decrypt=True, use_ecc=True):
    raw_bits = logits_to_bits(logits)

    decrypted = raw_bits.copy()
    if use_decrypt:
        decrypted = decrypt_watermark_bits(raw_bits, secret_key, shape=(WM_SIZE, WM_SIZE))

    final_bits = decrypted.copy()
    if use_ecc:
        final_bits = decode_bch(decrypted)

    return {
        "raw_bits": raw_bits,
        "decrypted_bits": decrypted,
        "final_bits": final_bits,
        "prob_map": (torch.sigmoid(logits).detach().cpu().numpy()[0, 0] * 255).astype(np.uint8),
    }
