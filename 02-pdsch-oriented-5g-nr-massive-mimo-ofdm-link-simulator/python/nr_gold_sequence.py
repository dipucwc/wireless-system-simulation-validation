"""
3GPP NR length-31 Gold-sequence generator.

This module generates the binary pseudo-random sequence ``c(n)`` used by
the simulator for:

- PDSCH bit scrambling and descrambling;
- deterministic DM-RS QPSK generation.

The implementation follows the two length-31 m-sequences used by
3GPP TS 38.211. The first sequence is initialized with a single one, and
the second sequence is initialized from the 31 least-significant bits of
``c_init``. After advancing both recurrences, the first ``Nc = 1600``
outputs are discarded and the requested Gold sequence is obtained by
exclusive-or:

    c(n) = x1(n + Nc) XOR x2(n + Nc).

The returned array uses ``numpy.uint8`` values equal to zero or one.
"""

from __future__ import annotations

import numpy as np


def nr_gold_sequence(c_init: int, length: int) -> np.ndarray:
    """Generate an NR Gold sequence.

    Parameters
    ----------
    c_init:
        Non-negative initialization integer. Only its 31 least-significant
        bits are used to initialize the second m-sequence.

    length:
        Number of output bits requested.

    Returns
    -------
    numpy.ndarray
        One-dimensional ``uint8`` array of length ``length``.

    Raises
    ------
    ValueError
        If ``c_init`` or ``length`` is negative.
    """


    if length < 0:
        raise ValueError("length must be non-negative.")


    if c_init < 0:
        raise ValueError("c_init must be non-negative.")


    nc = 1600


    total_length = length + nc + 31


    x1 = np.zeros(total_length, dtype=np.uint8)


    x2 = np.zeros(total_length, dtype=np.uint8)


    x1[0] = 1


    for bit_index in range(31):
        x2[bit_index] = (c_init >> bit_index) & 1


    for n in range(total_length - 31):


        x1[n + 31] = x1[n + 3] ^ x1[n]


        x2[n + 31] = (
            x2[n + 3]
            ^ x2[n + 2]
            ^ x2[n + 1]
            ^ x2[n]
        )


    x1_output = x1[nc : nc + length]
    x2_output = x2[nc : nc + length]


    return x1_output ^ x2_output
