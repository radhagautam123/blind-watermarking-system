import numpy as np


# ================= HELPER =================
def _get_rng(key):
    """Create a deterministic RNG based on key (no global side-effects)."""
    seed = abs(hash(key)) % (2**32)
    return np.random.RandomState(seed)


def _ensure_binary(wm):
    """Ensure watermark is binary (0/1)."""
    return (wm > 0.5).astype(np.uint8)


# ================= PERMUTATION =================
def get_permutation(key, size):
    rng = _get_rng(key)
    return rng.permutation(size)


def apply_permutation(wm, key):
    wm = _ensure_binary(wm)
    flat = wm.flatten()
    perm = get_permutation(key, len(flat))
    return flat[perm].reshape(wm.shape)


def inverse_permutation(wm, key):
    wm = _ensure_binary(wm)
    flat = wm.flatten()
    perm = get_permutation(key, len(flat))
    inv = np.argsort(perm)
    return flat[inv].reshape(wm.shape)


# ================= KEY ENCRYPTION =================
def get_key_noise(key, shape):
    rng = _get_rng(key)
    return rng.randint(0, 2, shape).astype(np.uint8)


def apply_key_transform(wm, key):
    wm = _ensure_binary(wm)
    noise = get_key_noise(key, wm.shape)
    return np.bitwise_xor(wm, noise)


# ================= FINAL PIPELINE =================
def secure_encode(wm, key):
    """
    Input: wm (numpy array of 0/1)
    Output: encoded watermark (0/1)
    """
    wm = _ensure_binary(wm)
    wm = apply_key_transform(wm, key)
    wm = apply_permutation(wm, key)
    return wm


def secure_decode(wm, key):
    """
    Input: wm (numpy array of 0/1)
    Output: decoded watermark (0/1)
    """
    wm = _ensure_binary(wm)
    wm = inverse_permutation(wm, key)
    wm = apply_key_transform(wm, key)
    return wm