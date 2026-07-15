"""
Unitary cyclic-prefix OFDM modulation and demodulation.

The module provides a matched transform pair:

``ofdm_modulate``
    Converts a centered frequency-domain resource grid into a
    time-domain waveform by inverse FFT and cyclic-prefix insertion.

``ofdm_demodulate``
    Removes each cyclic prefix, applies the forward FFT, and re-centers
    the active spectrum.

The implementation uses the unitary transform convention:

    x = sqrt(N) * IFFT{X}
    X = FFT{x} / sqrt(N)

so signal energy is preserved across the transform pair. Verification
Gate 3 checks that modulation followed immediately by demodulation
reconstructs the input grid at machine precision.

Array conventions
-----------------
Input resource grid:
    ``(n_fft, n_symbols, n_antennas)``

Output waveform:
    ``(n_symbols * (n_fft + cp_len), n_antennas)``

The compact simulator uses a simplified constant cyclic-prefix length

    cp_len = round(n_fft / 14).

This is a verification-oriented CP model rather than the complete
symbol-dependent 5G NR normal-CP pattern.
"""

from __future__ import annotations

import numpy as np


def _cyclic_prefix_length(cfg) -> int:
    """Return the simplified cyclic-prefix length used by this simulator."""


    return round(cfg.n_fft / 14)


def ofdm_modulate(
    resource_grid: np.ndarray,
    cfg,
) -> np.ndarray:
    """Convert a centered resource grid into a CP-OFDM waveform.

    Parameters
    ----------
    resource_grid:
        Complex array of shape
        ``(n_fft, n_symbols, n_tx)``.

    cfg:
        Configuration object providing ``n_fft``.

    Returns
    -------
    numpy.ndarray
        Time-domain waveform of shape
        ``(n_symbols * (n_fft + cp_len), n_tx)``.
    """


    n_fft, n_symbols, n_tx = resource_grid.shape


    if n_fft != cfg.n_fft:
        raise ValueError(
            "Resource-grid FFT size does not match cfg.n_fft."
        )


    cp_len = _cyclic_prefix_length(cfg)


    symbol_length = cfg.n_fft + cp_len


    waveform = np.zeros(
        (n_symbols * symbol_length, n_tx),
        dtype=complex,
    )


    for symbol_index in range(n_symbols):

        symbol_grid = resource_grid[:, symbol_index, :]


        unshifted_grid = np.fft.ifftshift(
            symbol_grid,
            axes=0,
        )


        time_symbol = np.fft.ifft(
            unshifted_grid,
            n=cfg.n_fft,
            axis=0,
        )


        time_symbol *= np.sqrt(cfg.n_fft)


        cyclic_prefix = time_symbol[-cp_len:, :]


        symbol_with_cp = np.vstack(
            [cyclic_prefix, time_symbol]
        )


        start = symbol_index * symbol_length
        stop = (symbol_index + 1) * symbol_length


        waveform[start:stop, :] = symbol_with_cp


    return waveform


def ofdm_demodulate(
    waveform: np.ndarray,
    cfg,
    n_tx: int,
) -> np.ndarray:
    """Recover the centered resource grid from a CP-OFDM waveform.

    Parameters
    ----------
    waveform:
        Time-domain array of shape
        ``(n_symbols * (n_fft + cp_len), n_tx)``.

    cfg:
        Configuration object providing ``n_fft`` and ``n_symbols``.

    n_tx:
        Number of antenna columns expected in the waveform. The name is
        retained for API compatibility; the same function may also be
        used with receive-antenna waveforms.

    Returns
    -------
    numpy.ndarray
        Centered frequency-domain grid of shape
        ``(n_fft, n_symbols, n_tx)``.
    """


    cp_len = _cyclic_prefix_length(cfg)


    symbol_length = cfg.n_fft + cp_len


    required_samples = cfg.n_symbols * symbol_length


    if waveform.shape[0] < required_samples:
        raise ValueError(
            "Waveform is shorter than the configured OFDM frame."
        )


    if waveform.shape[1] != n_tx:
        raise ValueError(
            "Waveform antenna count does not match n_tx."
        )


    grid = np.zeros(
        (cfg.n_fft, cfg.n_symbols, n_tx),
        dtype=complex,
    )


    for symbol_index in range(cfg.n_symbols):

        useful_start = (
            symbol_index * symbol_length + cp_len
        )


        useful_stop = useful_start + cfg.n_fft


        time_symbol = waveform[
            useful_start:useful_stop,
            :
        ]


        frequency_symbol = np.fft.fft(
            time_symbol,
            n=cfg.n_fft,
            axis=0,
        )


        frequency_symbol /= np.sqrt(cfg.n_fft)


        centered_symbol = np.fft.fftshift(
            frequency_symbol,
            axes=0,
        )


        grid[:, symbol_index, :] = centered_symbol


    return grid
