"""
MIMO channel application and effective-channel utilities.

This module implements the frequency-domain downlink observation model

    y[m, k] = H[k] W[k] s[m, k] + n[m, k]

for every OFDM symbol index ``m`` and active-subcarrier index ``k``.

Two functions are provided:

``true_effective_channel``
    Forms the effective layer-domain channel

        G[k] = H[k] W[k]

    for either a wideband precoder shared by all subcarriers or a
    frequency-selective precoder defined separately on every subcarrier.

``apply_mimo_channel``
    Applies the effective channel to the layer-domain resource grid and
    adds circularly symmetric complex Gaussian receiver noise.

Array conventions
-----------------
``H``
    Physical MIMO channel with shape ``(n_sc, n_rx, n_tx)``.

``W``
    Wideband precoder with shape ``(n_tx, n_layers)``, or per-subcarrier
    precoder with shape ``(n_sc, n_tx, n_layers)``.

``layer_grid``
    Layer-domain resource grid with shape
    ``(n_symbols, n_sc, n_layers)``.

``G``
    Effective channel with shape ``(n_sc, n_rx, n_layers)``.

``rx_grid``
    Received resource grid with shape
    ``(n_symbols, n_sc, n_rx)``.

SNR convention
--------------
The code uses the project's total-transmit-SNR convention. With unit-power
layers and unit-norm precoder columns, the average total transmitted power
equals the number of layers. Therefore, the complex-noise variance per
receive antenna is

    noise_var = n_layers / 10**(snr_db / 10).

This keeps the noise level fixed for a selected SNR point and allows the
array and eigenbeamforming gains of the 64x8 Massive MIMO configuration
to appear as genuine BER improvements.

The same convention is what lets the 64x8 massive comparison show its
array gain honestly: the noise floor stays put while beamforming
concentrates the signal.
"""

from __future__ import annotations

import numpy as np


def true_effective_channel(
    H: np.ndarray,
    W: np.ndarray,
) -> np.ndarray:
    """Calculate the effective channel ``G[k] = H[k] @ W[k]``.

    Parameters
    ----------
    H:
        Physical channel array of shape ``(n_sc, n_rx, n_tx)``.

    W:
        Precoding array. Two forms are supported:

        - ``(n_tx, n_layers)`` for one wideband precoder shared by all
          active subcarriers;
        - ``(n_sc, n_tx, n_layers)`` for one precoder per subcarrier.

    Returns
    -------
    numpy.ndarray
        Effective layer-domain channel of shape
        ``(n_sc, n_rx, n_layers)``.

    Raises
    ------
    ValueError
        If the precoder is not two-dimensional or three-dimensional.
    """


    if W.ndim == 3:


        return np.einsum("krm,kml->krl", H, W)


    if W.ndim == 2:


        return np.einsum("krm,ml->krl", H, W)


    raise ValueError(
        "W must have shape (n_tx, n_layers) or "
        "(n_sc, n_tx, n_layers)."
    )


def apply_mimo_channel(
    layer_grid: np.ndarray,
    H: np.ndarray,
    W: np.ndarray,
    snr_db: float,
    rng,
) -> tuple[np.ndarray, float]:
    """Apply the precoded MIMO channel and add complex AWGN.

    Parameters
    ----------
    layer_grid:
        Layer-domain symbols with shape
        ``(n_symbols, n_sc, n_layers)``.

    H:
        Physical channel with shape ``(n_sc, n_rx, n_tx)``.

    W:
        Wideband or per-subcarrier precoder accepted by
        :func:`true_effective_channel`.

    snr_db:
        Total transmitted SNR in decibels.

    rng:
        NumPy random-number generator used to create reproducible noise.

    Returns
    -------
    rx_grid:
        Noisy received grid with shape ``(n_symbols, n_sc, n_rx)``.

    noise_var:
        Complex-noise variance per receive antenna.
    """


    n_sym, n_sc, n_layers = layer_grid.shape


    G = true_effective_channel(H, W)


    if G.shape[0] != n_sc:
        raise ValueError(
            "The effective-channel subcarrier count does not match "
            "the layer-grid subcarrier count."
        )


    rx_noiseless = np.einsum("krl,mkl->mkr", G, layer_grid)


    snr_linear = 10.0 ** (snr_db / 10.0)


    # Unit-power layers and unit-norm precoder columns make the total
    # transmit power equal the layer count, so the noise level is fixed per
    # SNR point and beamforming gain shows up as a real BER improvement.
    noise_var = n_layers / snr_linear


    noise_real = rng.standard_normal(rx_noiseless.shape)


    noise_imag = rng.standard_normal(rx_noiseless.shape)


    noise = np.sqrt(noise_var / 2.0) * (
        noise_real + 1j * noise_imag
    )


    return rx_noiseless + noise, float(noise_var)
