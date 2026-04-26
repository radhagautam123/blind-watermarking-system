import cv2
import numpy as np
import pywt
from utils.ecc import (
    encrypt_watermark_bits,
    decrypt_watermark_bits,
    keyed_block_indices,
    keyed_pair
)


def _block_dct(block):
    return cv2.dct(block.astype(np.float32))


def _block_idct(block):
    return cv2.idct(block.astype(np.float32))


def embed_watermark(host_rgb, watermark_bits, secret_key="radha123", alpha=12.0, wavelet="haar"):
    ycrcb = cv2.cvtColor(host_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    y = ycrcb[:, :, 0]

    cA, (cH, cV, cD) = pywt.dwt2(y, wavelet)

    wm = encrypt_watermark_bits(watermark_bits, secret_key)

    h, w = cA.shape
    blocks_y = h // 8
    blocks_x = w // 8
    capacity = blocks_y * blocks_x

    if len(wm) > capacity:
        raise ValueError(f"Watermark too large. Capacity={capacity}, got={len(wm)} bits")

    cA_mod = cA.copy()
    block_ids = keyed_block_indices(capacity, len(wm), secret_key)
    r1, c1, r2, c2 = keyed_pair(secret_key)

    for idx, block_id in enumerate(block_ids):
        by = block_id // blocks_x
        bx = block_id % blocks_x
        y0, x0 = by * 8, bx * 8

        block = cA_mod[y0:y0+8, x0:x0+8]
        dct_block = _block_dct(block)

        base = dct_block[r2, c2]
        if wm[idx] == 1:
            dct_block[r1, c1] = base + alpha
        else:
            dct_block[r1, c1] = base - alpha

        cA_mod[y0:y0+8, x0:x0+8] = _block_idct(dct_block)

    y_mod = pywt.idwt2((cA_mod, (cH, cV, cD)), wavelet)
    y_mod = np.clip(y_mod, 0, 255)

    ycrcb[:, :, 0] = y_mod[:ycrcb.shape[0], :ycrcb.shape[1]]
    watermarked = cv2.cvtColor(np.clip(ycrcb, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    return watermarked


def extract_watermark(wm_rgb, watermark_shape=(16, 16), secret_key="radha123", wavelet="haar"):
    ycrcb = cv2.cvtColor(wm_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    y = ycrcb[:, :, 0]

    cA, (_, _, _) = pywt.dwt2(y, wavelet)

    num_bits = watermark_shape[0] * watermark_shape[1]

    h, w = cA.shape
    blocks_y = h // 8
    blocks_x = w // 8
    capacity = blocks_y * blocks_x

    if num_bits > capacity:
        raise ValueError(f"Requested extraction too large. Capacity={capacity}, got={num_bits} bits")

    block_ids = keyed_block_indices(capacity, num_bits, secret_key)
    r1, c1, r2, c2 = keyed_pair(secret_key)

    bits = []
    for block_id in block_ids:
        by = block_id // blocks_x
        bx = block_id % blocks_x
        y0, x0 = by * 8, bx * 8

        block = cA[y0:y0+8, x0:x0+8]
        dct_block = _block_dct(block)

        bit = 1 if dct_block[r1, c1] > dct_block[r2, c2] else 0
        bits.append(bit)

    bits = np.array(bits, dtype=np.uint8)
    recovered = decrypt_watermark_bits(bits, secret_key, watermark_shape)
    return recovered