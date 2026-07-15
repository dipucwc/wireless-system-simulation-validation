"""
Least-squares DM-RS channel estimation for the effective MIMO channel.

Purpose
-------
This module estimates the effective layer channel

    G[k] = H[k] W[k]

from the received DM-RS resource elements.  The effective channel is estimated
because the DM-RS symbols are precoded by the same matrix used for the PDSCH
data symbols.

Estimation procedure
--------------------
For each layer and receive antenna, the module performs the following steps:

1. Find the pilot-bearing subcarriers from the DM-RS mask.
2. Divide each received DM-RS observation by its known transmitted pilot.
3. Average the LS estimates over the configured DM-RS OFDM symbols.
4. Interpolate the real and imaginary components over all subcarriers.

For two independent DM-RS symbols, time averaging reduces the pilot-noise
variance by a factor of two, corresponding to approximately 3 dB.

Array dimensions
----------------
rx_grid : (n_symbols, n_subcarriers, n_rx)
G_hat   : (n_subcarriers, n_rx, n_layers)
"""


import numpy as np


def estimate_effective_channel_ls(
    rx_grid: np.ndarray,
    frame,
    cfg,
) -> np.ndarray:
    """Estimate the effective MIMO channel using DM-RS least squares.

    Parameters
    ----------
    rx_grid : numpy.ndarray
        Received complex resource grid with shape
        ``(n_symbols, n_subcarriers, n_rx)``.
    frame : frame object
        Must provide ``dmrs_mask`` and ``dmrs_values``.
    cfg : configuration object
        Must provide ``n_layers`` and ``dmrs_symbols``.

    Returns
    -------
    numpy.ndarray
        Estimated effective channel with shape
        ``(n_subcarriers, n_rx, n_layers)``.
    """


    n_sym, n_sc, n_rx = rx_grid.shape


    G_hat = np.zeros(
        (n_sc, n_rx, cfg.n_layers),
        dtype=complex,
    )


    all_subcarriers = np.arange(n_sc)


    for layer in range(cfg.n_layers):


        pilot_subcarriers = np.unique(
            np.nonzero(frame.dmrs_mask[:, :, layer])[1]
        )


        pilot_observations = np.stack(
            [
                rx_grid[
                    dmrs_symbol,
                    pilot_subcarriers,
                    :,
                ]
                / frame.dmrs_values[
                    dmrs_symbol,
                    pilot_subcarriers,
                    layer,
                ][:, None]
                for dmrs_symbol in cfg.dmrs_symbols
            ],
            axis=0,
        )


        # Averaging over both DM-RS symbols halves the pilot-noise variance.
        averaged_pilot_estimate = pilot_observations.mean(axis=0)


        for receive_antenna in range(n_rx):

            interpolated_real = np.interp(
                all_subcarriers,
                pilot_subcarriers,
                averaged_pilot_estimate[:, receive_antenna].real,
            )


            interpolated_imag = np.interp(
                all_subcarriers,
                pilot_subcarriers,
                averaged_pilot_estimate[:, receive_antenna].imag,
            )


            G_hat[:, receive_antenna, layer] = (
                interpolated_real + 1j * interpolated_imag
            )


    return G_hat
