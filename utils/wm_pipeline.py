"""Shared watermark encode/decode pipeline for train, test, and Streamlit."""
import hashlib

import numpy as np
import torch

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
