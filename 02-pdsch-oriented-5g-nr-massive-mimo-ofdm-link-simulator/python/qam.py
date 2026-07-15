"""
Gray-mapped QAM constellation, modulation, and hard demodulation.

This module is the single source of constellation definitions used by
both the transmitter and receiver. It supports:

- QPSK;
- 16-QAM;
- 64-QAM;
- 256-QAM.

For rectangular QAM, the in-phase and quadrature axes are independently
Gray coded. The transmitted tuple stores the in-phase bits first and the
quadrature bits second. Every constellation is normalized to unit average
symbol power:

    E{|s|^2} = 1.

Standard normalization factors are used:

- QPSK:   sqrt(2)
- 16-QAM: sqrt(10)
- 64-QAM: sqrt(42)
- 256-QAM: sqrt(170)

The hard demodulator applies minimum-Euclidean-distance detection and
returns the exact bit tuples used by the corresponding modulator.
"""

from __future__ import annotations

import numpy as np


_AXIS = {
    "16QAM": (
        np.array(
            [[0, 0], [0, 1], [1, 1], [1, 0]],
            dtype=np.uint8,
        ),
        np.array([-3, -1, 1, 3], dtype=float),
        np.sqrt(10.0),
    ),
    "64QAM": (
        np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 1],
                [0, 1, 0],
                [1, 1, 0],
                [1, 1, 1],
                [1, 0, 1],
                [1, 0, 0],
            ],
            dtype=np.uint8,
        ),
        np.array(
            [-7, -5, -3, -1, 1, 3, 5, 7],
            dtype=float,
        ),
        np.sqrt(42.0),
    ),
    "256QAM": (
        np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 1],
                [0, 1, 0, 1],
                [0, 1, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 0, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 0],
                [1, 0, 1, 0],
                [1, 0, 1, 1],
                [1, 0, 0, 1],
                [1, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
        np.arange(-15, 16, 2, dtype=float),
        np.sqrt(170.0),
    ),
}


def qam_constellation(
    modulation: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return Gray bit tuples and normalized constellation points.

    Parameters
    ----------
    modulation:
        One of ``QPSK``, ``16QAM``, ``64QAM``, or ``256QAM``.
        The comparison is case-insensitive.

    Returns
    -------
    tuples:
        Binary array of shape ``(M, bits_per_symbol)``.

    values:
        Complex constellation array of shape ``(M,)``.

    bits_per_symbol:
        Number of coded bits carried by one modulation symbol.

    Raises
    ------
    ValueError
        If the requested modulation order is unsupported.
    """


    modulation = modulation.upper()


    if modulation == "QPSK":

        tuples = np.array(
            [
                [0, 0],
                [0, 1],
                [1, 1],
                [1, 0],
            ],
            dtype=np.uint8,
        )


        values = np.array(
            [
                1 + 1j,
                1 - 1j,
                -1 - 1j,
                -1 + 1j,
            ],
            dtype=complex,
        ) / np.sqrt(2.0)


        return tuples, values, 2


    if modulation not in _AXIS:
        raise ValueError(
            "Unsupported modulation. Use QPSK, 16QAM, 64QAM, or 256QAM."
        )


    axis_bits, axis_values, normalization = _AXIS[modulation]


    n_axis = axis_values.size


    bits_per_axis = axis_bits.shape[1]


    constellation_size = n_axis * n_axis


    bits_per_symbol = 2 * bits_per_axis


    tuples = np.zeros(
        (constellation_size, bits_per_symbol),
        dtype=np.uint8,
    )


    values = np.zeros(
        constellation_size,
        dtype=complex,
    )


    output_index = 0


    for in_phase_index in range(n_axis):
        for quadrature_index in range(n_axis):

            tuples[output_index] = np.concatenate(
                [
                    axis_bits[in_phase_index],
                    axis_bits[quadrature_index],
                ]
            )


            values[output_index] = (
                axis_values[in_phase_index]
                + 1j * axis_values[quadrature_index]
            ) / normalization


            output_index += 1


    return tuples, values, bits_per_symbol


def qam_modulate(
    bits: np.ndarray,
    modulation: str,
) -> np.ndarray:
    """Map binary tuples to normalized Gray-coded QAM symbols.

    Parameters
    ----------
    bits:
        One-dimensional or flattenable binary input array.

    modulation:
        Requested QAM order.

    Returns
    -------
    numpy.ndarray
        Complex modulation symbols.

    Raises
    ------
    ValueError
        If the input length is not a multiple of bits per symbol.
    """


    tuples, values, bits_per_symbol = qam_constellation(
        modulation
    )


    bits = np.asarray(
        bits,
        dtype=np.uint8,
    ).ravel()


    if bits.size % bits_per_symbol != 0:
        raise ValueError(
            "Bit length must be a multiple of bits per symbol."
        )


    binary_weights = 2 ** np.arange(
        bits_per_symbol - 1,
        -1,
        -1,
    )


    lookup_table = np.zeros(
        2 ** bits_per_symbol,
        dtype=complex,
    )


    lookup_table[tuples @ binary_weights] = values


    grouped_bits = bits.reshape(
        -1,
        bits_per_symbol,
    )


    symbol_indices = grouped_bits @ binary_weights


    return lookup_table[symbol_indices]


def qam_demodulate_hard(
    symbols: np.ndarray,
    modulation: str,
) -> np.ndarray:
    """Hard-demodulate symbols by minimum Euclidean distance.

    Parameters
    ----------
    symbols:
        Complex received symbols.

    modulation:
        Requested QAM order.

    Returns
    -------
    numpy.ndarray
        Flat ``uint8`` bit sequence in transmitter tuple order.
    """


    tuples, values, _ = qam_constellation(modulation)


    symbols = np.asarray(symbols).ravel()


    squared_distance = np.abs(
        symbols[:, None] - values[None, :]
    ) ** 2


    nearest_indices = np.argmin(
        squared_distance,
        axis=1,
    )


    detected_tuples = tuples[nearest_indices]


    return detected_tuples.astype(
        np.uint8
    ).ravel()
