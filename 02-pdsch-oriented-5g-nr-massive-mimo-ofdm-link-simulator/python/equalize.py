"""
MIMO layer equalization for the 5G NR PDSCH link-level simulator.

Purpose
-------
This module recovers the transmitted spatial-layer symbols from the received
MIMO resource grid.  The receiver uses the supplied estimate of the effective
channel

    G[k] = H[k] W[k]

rather than the physical channel H[k] alone, because the transmitted DM-RS and
PDSCH symbols have already passed through the precoder W[k].

Implemented equalizers
----------------------
1. Zero-Forcing (ZF)

       F_ZF[k] = pinv(G_hat[k])

   ZF suppresses inter-layer interference, but it can strongly amplify noise
   when the effective channel is ill-conditioned.

2. Minimum Mean-Square Error (MMSE)

       F_MMSE[k] = (G_hat^H G_hat + sigma_n^2 I)^(-1) G_hat^H

   The noise-variance term regularizes the matrix inversion.  The raw MMSE
   output is amplitude-biased, so this implementation divides each layer by
   the corresponding diagonal term of F_MMSE[k] G_hat[k].  This produces the
   unbiased-MMSE output used by the hard-decision QAM demodulator.

Array dimensions
----------------
rx_grid       : (n_symbols, n_subcarriers, n_rx)
G_hat         : (n_subcarriers, n_rx, n_layers)
data_positions: (n_data_resource_elements, 2)
return value  : (n_data_resource_elements, n_layers)

The channel is assumed to remain constant over the OFDM symbols of one slot.
Therefore, one equalizer matrix is calculated per subcarrier and then reused
for every data resource element on that subcarrier.
"""


import numpy as np


from mimo_channel import true_effective_channel


def equalize_mimo(
    rx_grid: np.ndarray,
    G_hat: np.ndarray,
    data_positions: np.ndarray,
    noise_var: float,
    cfg,
) -> np.ndarray:
    """Equalize all PDSCH data resource elements in the received MIMO grid.

    Parameters
    ----------
    rx_grid : numpy.ndarray
        Complex received resource grid with shape
        ``(n_symbols, n_subcarriers, n_rx)``.
    G_hat : numpy.ndarray
        Estimated effective channel with shape
        ``(n_subcarriers, n_rx, n_layers)``.
    data_positions : numpy.ndarray
        Integer array with shape ``(n_data, 2)``.  Column zero contains OFDM
        symbol indices and column one contains subcarrier indices.
    noise_var : float
        Complex-noise variance used in the MMSE regularization term.
    cfg : configuration object
        Must provide ``n_layers`` and ``equalizer``.

    Returns
    -------
    numpy.ndarray
        Equalized layer symbols with shape ``(n_data, n_layers)``.
    """


    n_sc = G_hat.shape[0]


    n_layers = cfg.n_layers


    identity = np.eye(n_layers)


    equalizer_matrices = np.zeros(
        (n_sc, n_layers, G_hat.shape[1]),
        dtype=complex,
    )


    for subcarrier in range(n_sc):


        effective_channel = G_hat[subcarrier]


        mode = cfg.equalizer.lower()


        if mode == "zf":


            equalizer_matrices[subcarrier] = np.linalg.pinv(
                effective_channel
            )
        elif mode == "mmse":

            regularized_gram = (
                effective_channel.conj().T @ effective_channel
                + noise_var * identity
            )


            biased_mmse = np.linalg.solve(
                regularized_gram,
                effective_channel.conj().T,
            )


            # Raw MMSE shrinks the symbol amplitude per layer; divide the
            # bias back out so hard slicing of 16-QAM and above works.
            layer_bias = np.real(
                np.diag(biased_mmse @ effective_channel)
            )


            safe_layer_bias = np.maximum(
                layer_bias,
                np.finfo(float).eps,
            )


            equalizer_matrices[subcarrier] = (
                biased_mmse / safe_layer_bias[:, None]
            )
        else:

            raise ValueError(
                f"Unsupported equalizer: {cfg.equalizer} (use 'zf' or 'mmse')"
            )


    symbol_indices = data_positions[:, 0]


    subcarrier_indices = data_positions[:, 1]


    received_data_vectors = rx_grid[
        symbol_indices,
        subcarrier_indices,
        :,
    ]


    equalized_symbols = np.einsum(
        "nlr,nr->nl",
        equalizer_matrices[subcarrier_indices],
        received_data_vectors,
    )


    return equalized_symbols
