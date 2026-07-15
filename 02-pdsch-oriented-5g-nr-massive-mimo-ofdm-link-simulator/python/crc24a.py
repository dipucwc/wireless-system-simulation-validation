"""
The crc24a module implements the CRC24A parity of TS 38.212 as a bit-serial
linear feedback shift register over the lower twenty-four bits of the
generator polynomial: for every input bit the feedback is formed as the
exclusive-or of the register's most significant bit with the input, the
register is shifted and masked to twenty-four bits, and the polynomial is
added on feedback. The parity word is emitted most significant bit first,
matching the transport-block attachment order, and the module also provides
the attachment and verification helpers whose boolean result is the
block-error decision of the receiver.
"""
import numpy as np


def crc24a(bits: np.ndarray, poly: int) -> np.ndarray:

    reg = 0
    for b in bits:
        feedback = ((reg >> 23) & 1) ^ int(b)
        reg = (reg << 1) & 0xFFFFFF
        if feedback:
            reg ^= poly

    return np.array([(reg >> (23 - i)) & 1 for i in range(24)], dtype=np.uint8)


def append_crc24a(payload: np.ndarray, cfg) -> np.ndarray:

    payload = np.asarray(payload, dtype=np.uint8).ravel()
    return np.concatenate([payload, crc24a(payload, cfg.crc_poly)])


def check_crc24a(bits_with_crc: np.ndarray, cfg) -> bool:

    bits_with_crc = np.asarray(bits_with_crc, dtype=np.uint8).ravel()
    payload = bits_with_crc[:-cfg.crc_len]
    rx_crc = bits_with_crc[-cfg.crc_len:]
    return bool(np.array_equal(crc24a(payload, cfg.crc_poly), rx_crc))
