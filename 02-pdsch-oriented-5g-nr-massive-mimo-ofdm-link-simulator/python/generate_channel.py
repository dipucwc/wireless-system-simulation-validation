"""
Frequency-domain MIMO channel generation for the link-level simulator.

Purpose
-------
This module creates one channel realization per Monte Carlo frame.  The output
has shape

    H.shape = (n_subcarriers, n_rx, n_tx)

and is assumed quasi-static over all OFDM symbols of the slot.

Supported channel models
------------------------
``awgn``
    Deterministic identity-like channel with unit gain and no fading.

``rayleigh`` or ``rayleigh_flat``
    One independent CN(0,1) MIMO matrix per frame, repeated over all active
    subcarriers.

``rayleigh_iid``
    Independent CN(0,1) fading on every subcarrier and antenna pair.

``rician``
    Deterministic all-ones line-of-sight component plus a Rayleigh NLoS
    component, combined according to the configured K-factor.

``tdl``
    Configurable Rayleigh tapped-delay-line model.  Tap gains are transformed
    to the frequency domain using the configured delays and power-delay
    profile, then normalized to unit average channel power.
"""


import numpy as np


def _cn(rng, *shape) -> np.ndarray:
    """Generate circularly symmetric complex Gaussian samples CN(0, 1).

    Parameters
    ----------
    rng : numpy.random.Generator
        Seeded random-number generator.
    *shape : int
        Requested output dimensions.

    Returns
    -------
    numpy.ndarray
        Complex Gaussian array with unit average power.
    """


    real_part = rng.standard_normal(shape)


    imaginary_part = rng.standard_normal(shape)


    return (real_part + 1j * imaginary_part) / np.sqrt(2)


def generate_channel(cfg, rng) -> np.ndarray:
    """Generate one frequency-domain MIMO channel realization.

    Parameters
    ----------
    cfg : configuration object
        Must provide the selected channel model and all model-specific
        parameters.
    rng : numpy.random.Generator
        Seeded random-number generator.

    Returns
    -------
    numpy.ndarray
        Channel tensor with shape ``(n_sc, n_rx, n_tx)``.

    Raises
    ------
    ValueError
        If ``cfg.channel_model`` is unsupported.
    """


    model = cfg.channel_model.lower()


    if model == "awgn":

        channel = np.zeros(
            (cfg.n_sc, cfg.n_rx, cfg.n_tx),
            dtype=complex,
        )


        for antenna in range(min(cfg.n_rx, cfg.n_tx)):
            channel[:, antenna, antenna] = 1.0


        return channel


    if model in ("rayleigh_flat", "rayleigh"):

        flat_channel = _cn(
            rng,
            1,
            cfg.n_rx,
            cfg.n_tx,
        )


        return np.repeat(
            flat_channel,
            cfg.n_sc,
            axis=0,
        )


    if model == "rayleigh_iid":

        return _cn(
            rng,
            cfg.n_sc,
            cfg.n_rx,
            cfg.n_tx,
        )


    if model == "rician":

        k_factor = 10 ** (cfg.rician_k_db / 10)


        los_component = np.ones(
            (cfg.n_sc, cfg.n_rx, cfg.n_tx),
            dtype=complex,
        )


        nlos_component = _cn(
            rng,
            cfg.n_sc,
            cfg.n_rx,
            cfg.n_tx,
        )


        return (
            np.sqrt(k_factor / (k_factor + 1)) * los_component
            + np.sqrt(1 / (k_factor + 1)) * nlos_component
        )


    if model == "tdl":

        delays = np.asarray(
            cfg.tdl_delays,
            dtype=float,
        )


        tap_powers = 10 ** (
            np.asarray(cfg.tdl_powers_db) / 10
        )


        tap_powers = tap_powers / tap_powers.sum()


        tap_gains = _cn(
            rng,
            delays.size,
            cfg.n_rx,
            cfg.n_tx,
        ) * np.sqrt(tap_powers)[:, None, None]


        centered_subcarriers = (
            np.arange(cfg.n_sc) - cfg.n_sc / 2
        )


        tap_phase = np.exp(
            -1j
            * 2
            * np.pi
            * np.outer(centered_subcarriers, delays)
            / cfg.n_fft
        )


        channel = np.einsum(
            "kt,trm->krm",
            tap_phase,
            tap_gains,
        )


        return channel / np.sqrt(
            np.mean(np.abs(channel) ** 2)
        )


    raise ValueError(
        f"Unsupported channel model: {cfg.channel_model!r}"
    )
