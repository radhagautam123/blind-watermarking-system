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
    return encrypted


def decrypt_watermark_bits(enc_bits: np.ndarray, secret_key: str, shape):
    flat = enc_bits.flatten().astype(np.uint8)
    key_stream = generate_key_stream(len(flat), secret_key)
    decrypted = np.bitwise_xor(flat, key_stream)
    return decrypted.reshape(shape)


def keyed_block_indices(total_blocks: int, n_bits: int, secret_key: str):
    rng = np.random.default_rng(key_to_seed(secret_key + "_blocks"))
    idx = rng.permutation(total_blocks)
    return idx[:n_bits]


def keyed_pair(secret_key: str):
    pairs = [
        (4, 3, 4, 2),
        (3, 4, 3, 2),
        (2, 4, 2, 3),
        (5, 3, 5, 2)
    ]
    rng = np.random.default_rng(key_to_seed(secret_key + "_pair"))
    return pairs[int(rng.integers(0, len(pairs)))]