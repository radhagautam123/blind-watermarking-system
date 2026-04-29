import numpy as np


# ---------------- REPETITION ENCODING ----------------
def repeat_bits(bits, k=3):
    return np.repeat(bits, k)


def majority_vote(bits, k=3):
    bits = bits.reshape(-1, k)
    return (np.sum(bits, axis=1) > (k // 2)).astype(np.uint8)


# ---------------- ENCODE ----------------
def encode_bch(data_bits):
    flat = data_bits.flatten().astype(np.uint8)

    repeated = repeat_bits(flat, k=3)

    # fit back to 1024
    if len(repeated) > 1024:
        repeated = repeated[:1024]
    else:
        repeated = np.pad(repeated, (0, 1024 - len(repeated)))

    return repeated.reshape(32, 32)


# ---------------- DECODE ----------------
def decode_bch(received_bits):
    flat = received_bits.flatten().astype(np.uint8)

    usable_len = (len(flat) // 3) * 3
    flat = flat[:usable_len]

    decoded = majority_vote(flat, k=3)

    if len(decoded) < 1024:
        decoded = np.pad(decoded, (0, 1024 - len(decoded)))

    return decoded[:1024].reshape(32, 32)
