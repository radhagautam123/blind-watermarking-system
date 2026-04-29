import hashlib
import numpy as np


def key_to_seed(secret_key: str) -> int:
    h = hashlib.sha256(secret_key.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % (2**32)


def generate_key_stream(n_bits: int, secret_key: str):
    rng = np.random.default_rng(key_to_seed(secret_key + "_bits"))
    return rng.integers(0, 2, size=n_bits, dtype=np.uint8)


def encrypt_watermark_bits(wm_bits: np.ndarray, secret_key: str):
    flat = wm_bits.flatten().astype(np.uint8)
    key_stream = generate_key_stream(len(flat), secret_key)
    encrypted = np.bitwise_xor(flat, key_stream)
    return encrypted.reshape(wm_bits.shape)


def decrypt_watermark_bits(enc_bits: np.ndarray, secret_key: str, shape):
    flat = enc_bits.flatten().astype(np.uint8)
    key_stream = generate_key_stream(len(flat), secret_key)
    decrypted = np.bitwise_xor(flat, key_stream)
    return decrypted.reshape(shape)
