import numpy as np
import hashlib

def key_to_seed(key: str):
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32)

def secure_encode(wm, key):
    rng = np.random.default_rng(key_to_seed(key))
    noise = rng.integers(0, 2, wm.shape)
    return np.bitwise_xor(wm.astype(np.uint8), noise)

def secure_decode(wm, key):
    rng = np.random.default_rng(key_to_seed(key))
    noise = rng.integers(0, 2, wm.shape)
    return np.bitwise_xor(wm.astype(np.uint8), noise)