"""
PDSCH bit scrambling and descrambling.

The module XORs the input bit sequence with the NR Gold sequence generated
from the configured RNTI and scrambling identity.

The initialization value is

    c_init = (RNTI << 15) + N_ID.

Because exclusive-or is its own inverse, the same function performs both
operations:

    scrambled   = scramble_bits(original, cfg)
    descrambled = scramble_bits(scrambled, cfg)

and ``descrambled`` equals ``original`` when the same configuration is
used at both ends of the link.
"""

from __future__ import annotations

import numpy as np

from nr_gold_sequence import nr_gold_sequence


def scramble_bits(
    bits: np.ndarray,
    cfg,
) -> np.ndarray:
    """Scramble or descramble a binary sequence.

    Parameters
    ----------
    bits:
        Input binary sequence.

    cfg:
        Configuration object containing ``rnti`` and ``n_id``.

    Returns
    -------
    numpy.ndarray
        Scrambled or descrambled ``uint8`` sequence with the same length
        as the input.
    """


    bits = np.asarray(
        bits,
        dtype=np.uint8,
    ).ravel()


    c_init = (
        int(cfg.rnti) << 15
    ) + int(cfg.n_id)


    sequence = nr_gold_sequence(
        c_init,
        bits.size,
    )


    return bits ^ sequence
