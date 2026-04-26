import bchlib
import numpy as np

# BCH setup
BCH_POLY = 137
BCH_BITS = 5

bch = bchlib.BCH(BCH_POLY, BCH_BITS)


def encode_bch(data_bits):
    data_bytes = np.packbits(data_bits)
    ecc = bch.encode(data_bytes)
    packet = data_bytes + ecc
    return np.unpackbits(np.frombuffer(packet, dtype=np.uint8))


def decode_bch(received_bits):
    packet = np.packbits(received_bits)
    data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]

    bitflips = bch.decode(data, ecc)

    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))