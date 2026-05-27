import hashlib
import numpy as np


def key_to_seed(secret_key: str) -> int:
    digest = hashlib.sha256(secret_key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _normalize_bits(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits).astype(np.uint8)
    return (bits > 0).astype(np.uint8)


def _derive_block(secret_key: str, counter: int, context: str = "wm_keystream") -> bytes:
    msg = f"{secret_key}::{context}::{counter}".encode("utf-8")
    return hashlib.sha256(msg).digest()


def generate_key_stream(n_bits: int, secret_key: str, context: str = "wm_keystream") -> np.ndarray:
    if n_bits <= 0:
        return np.zeros((0,), dtype=np.uint8)

    stream_bytes = bytearray()
    counter = 0

    while len(stream_bytes) * 8 < n_bits:
        stream_bytes.extend(_derive_block(secret_key, counter, context=context))
        counter += 1

    byte_arr = np.frombuffer(bytes(stream_bytes), dtype=np.uint8)
    bit_arr = np.unpackbits(byte_arr)[:n_bits].astype(np.uint8)
    return bit_arr


def permute_indices(n: int, secret_key: str) -> np.ndarray:
    seed_material = hashlib.sha256((secret_key + "::permute").encode("utf-8")).digest()
    seed = int.from_bytes(seed_material[:8], "big") % (2**32)
    rng = np.random.default_rng(seed)
    return rng.permutation(n)


def invert_permutation(perm: np.ndarray) -> np.ndarray:
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv


def encrypt_watermark_bits(wm_bits: np.ndarray, secret_key: str) -> np.ndarray:
    wm_bits = _normalize_bits(wm_bits)
    flat = wm_bits.flatten()

    perm = permute_indices(len(flat), secret_key)
    shuffled = flat[perm]

    key_stream = generate_key_stream(len(shuffled), secret_key, context="wm_encrypt")
    encrypted = np.bitwise_xor(shuffled, key_stream).astype(np.uint8)

    return encrypted.reshape(wm_bits.shape)


def decrypt_watermark_bits(enc_bits: np.ndarray, secret_key: str, shape=None) -> np.ndarray:
    enc_bits = _normalize_bits(enc_bits)
    flat = enc_bits.flatten()

    key_stream = generate_key_stream(len(flat), secret_key, context="wm_encrypt")
    decrypted = np.bitwise_xor(flat, key_stream).astype(np.uint8)

    perm = permute_indices(len(decrypted), secret_key)
    inv_perm = invert_permutation(perm)
    unshuffled = decrypted[inv_perm]

    if shape is None:
        return unshuffled.reshape(enc_bits.shape)

    return unshuffled.reshape(shape)