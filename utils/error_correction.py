import numpy as np

TARGET_BITS = 32 * 32
HEADER_BITS = 16
WM_GRID = 32


def _to_flat_bits(bits):
    bits = np.asarray(bits).astype(np.uint8).flatten()
    return (bits > 0).astype(np.uint8)


def _int_to_bits(x, n_bits=16):
    x = int(max(0, min(x, (1 << n_bits) - 1)))
    return np.array([(x >> i) & 1 for i in range(n_bits - 1, -1, -1)], dtype=np.uint8)


def _bits_to_int(bits):
    out = 0
    for bit in bits:
        out = (out << 1) | int(bit)
    return out


def _hamming74_encode_block(data4):
    d1, d2, d3, d4 = data4.tolist()
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p3 = d2 ^ d3 ^ d4
    return np.array([p1, p2, d1, p3, d2, d3, d4], dtype=np.uint8)


def _hamming74_decode_block(code7):
    b = code7.astype(np.uint8).copy()
    s1 = b[0] ^ b[2] ^ b[4] ^ b[6]
    s2 = b[1] ^ b[2] ^ b[5] ^ b[6]
    s3 = b[3] ^ b[4] ^ b[5] ^ b[6]
    syndrome = s1 + (s2 << 1) + (s3 << 2)
    if syndrome != 0 and 1 <= syndrome <= 7:
        b[syndrome - 1] ^= 1
    return np.array([b[2], b[4], b[5], b[6]], dtype=np.uint8)


def _hamming_encoded_length(n_data_bits, repeat_factor):
    payload = HEADER_BITS + n_data_bits
    repeated_len = payload * repeat_factor
    pad = (-repeated_len) % 4
    return (repeated_len + pad) // 4 * 7


def _max_payload_for_repeat(repeat_factor):
    lo, hi = 0, TARGET_BITS
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _hamming_encoded_length(mid, repeat_factor) <= TARGET_BITS:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _encode_hamming(payload_bits, repeat_factor):
    repeated = np.repeat(payload_bits.astype(np.uint8), repeat_factor)
    pad = (-len(repeated)) % 4
    if pad > 0:
        repeated = np.pad(repeated, (0, pad), mode="constant")

    encoded_blocks = []
    for i in range(0, len(repeated), 4):
        encoded_blocks.append(_hamming74_encode_block(repeated[i:i + 4]))
    encoded = np.concatenate(encoded_blocks, axis=0)

    if len(encoded) > TARGET_BITS:
        raise ValueError("Hamming encoded length exceeds TARGET_BITS")
    if len(encoded) < TARGET_BITS:
        encoded = np.pad(encoded, (0, TARGET_BITS - len(encoded)), mode="constant")
    return encoded


def _decode_hamming(flat, repeat_factor):
    usable_len = (len(flat) // 7) * 7
    flat = flat[:usable_len]

    decoded_blocks = []
    for i in range(0, len(flat), 7):
        decoded_blocks.append(_hamming74_decode_block(flat[i:i + 7]))
    if len(decoded_blocks) == 0:
        return np.zeros(0, dtype=np.uint8)

    decoded = np.concatenate(decoded_blocks, axis=0)
    usable = (len(decoded) // repeat_factor) * repeat_factor
    decoded = decoded[:usable]
    grouped = decoded.reshape(-1, repeat_factor)
    voted = (grouped.sum(axis=1) >= ((repeat_factor // 2) + 1)).astype(np.uint8)
    return voted


def _encode_row_parity(bits):
    """Pack up to 992 payload bits into 32x32 with per-row even parity (31 data + 1 parity)."""
    bits = _to_flat_bits(bits)
    max_data = WM_GRID * (WM_GRID - 1)
    if len(bits) > max_data:
        bits = bits[:max_data]

    grid = np.zeros((WM_GRID, WM_GRID), dtype=np.uint8)
    flat = np.zeros(max_data, dtype=np.uint8)
    flat[:len(bits)] = bits
    data = flat.reshape(WM_GRID, WM_GRID - 1)
    parity = (data.sum(axis=1) % 2).astype(np.uint8)
    grid[:, : WM_GRID - 1] = data
    grid[:, WM_GRID - 1] = parity
    return grid.flatten()


def _decode_row_parity(flat):
    grid = _to_flat_bits(flat).reshape(WM_GRID, WM_GRID)
    data = grid[:, : WM_GRID - 1].copy()
    parity = grid[:, WM_GRID - 1]

    for r in range(WM_GRID):
        if (data[r].sum() % 2) != parity[r]:
            row = data[r].astype(np.int16)
            diff = np.abs(row - 0.5)
            flip_idx = int(np.argmax(diff))
            data[r, flip_idx] ^= 1

    return data.flatten()


def encode_bch(data_bits):
    """
    Encode payload into exactly 1024 bits for the 32x32 watermark grid.

    - Small payloads: Hamming(7,4) + adaptive repetition (fits without truncation).
    - Full 32x32 logos (1024 bits): row-parity 2D code (992 data + 32 parity).
    """
    data_bits = _to_flat_bits(data_bits)
    original_len = len(data_bits)

    if original_len >= WM_GRID * WM_GRID:
        packed = data_bits[:TARGET_BITS]
        if len(packed) < TARGET_BITS:
            packed = np.pad(packed, (0, TARGET_BITS - len(packed)), mode="constant")
        return packed.reshape(WM_GRID, WM_GRID).astype(np.uint8)

    len_bits = _int_to_bits(original_len, n_bits=HEADER_BITS)
    payload = np.concatenate([len_bits, data_bits], axis=0)

    for repeat_factor in (3, 2, 1):
        if _hamming_encoded_length(original_len, repeat_factor) <= TARGET_BITS:
            encoded = _encode_hamming(payload, repeat_factor)
            return encoded.reshape(WM_GRID, WM_GRID).astype(np.uint8)

    encoded = _encode_row_parity(data_bits)
    return encoded.reshape(WM_GRID, WM_GRID).astype(np.uint8)


def _try_decode_hamming(flat, repeat_factor):
    decoded = _decode_hamming(flat, repeat_factor)
    if len(decoded) < HEADER_BITS:
        return None

    msg_len = _bits_to_int(decoded[:HEADER_BITS])
    max_payload = _max_payload_for_repeat(repeat_factor)
    if msg_len <= 0 or msg_len > max_payload:
        return None
    if len(decoded) < HEADER_BITS + msg_len:
        return None

    payload_with_header = decoded[: HEADER_BITS + msg_len]
    re_encoded = _encode_hamming(payload_with_header, repeat_factor)
    if not np.array_equal(re_encoded, flat):
        return None

    payload = decoded[HEADER_BITS:HEADER_BITS + msg_len]
    if len(payload) < TARGET_BITS:
        payload = np.pad(payload, (0, TARGET_BITS - len(payload)), mode="constant")
    else:
        payload = payload[:TARGET_BITS]
    return payload.reshape(WM_GRID, WM_GRID).astype(np.uint8)


def decode_bch(received_bits):
    flat = _to_flat_bits(received_bits).reshape(-1)

    if len(flat) < TARGET_BITS:
        flat = np.pad(flat, (0, TARGET_BITS - len(flat)), mode="constant")
    flat = flat[:TARGET_BITS]

    for repeat_factor in (3, 2, 1):
        try:
            result = _try_decode_hamming(flat, repeat_factor)
            if result is not None:
                return result
        except Exception:
            continue

    try:
        payload = _decode_row_parity(flat)
        re_encoded = _encode_row_parity(payload)
        if np.array_equal(re_encoded, flat):
            if len(payload) < TARGET_BITS:
                payload = np.pad(payload, (0, TARGET_BITS - len(payload)), mode="constant")
            else:
                payload = payload[:TARGET_BITS]
            return payload.reshape(WM_GRID, WM_GRID).astype(np.uint8)
    except Exception:
        pass

    return flat.reshape(WM_GRID, WM_GRID).astype(np.uint8)
