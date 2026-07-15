"""
Frame-level performance metrics for the PDSCH link-level simulator.

Purpose
-------
This module compares each equalized frame against the known transmitted
references and calculates the principal link metrics:

* payload-bit error count and BER denominator,
* CRC24A-based block error decision,
* RMS error vector magnitude (EVM),
* effective-channel estimation NMSE,
* theoretical layer-domain MIMO capacity,
* payload size for successful-throughput accumulation.

The capacity value is a theoretical channel reference.  It is not interpreted
as achieved coded throughput.
"""


import numpy as np


from qam import qam_demodulate_hard


from scramble_bits import scramble_bits


from crc24a import check_crc24a


from mimo_channel import true_effective_channel


def capacity_mimo(
    G: np.ndarray,
    snr_db: float,
    n_layers: int,
) -> float:
    """Calculate average layer-domain MIMO capacity in bit/s/Hz.

    Parameters
    ----------
    G : numpy.ndarray
        True effective channel with shape
        ``(n_subcarriers, n_rx, n_layers)``.
    snr_db : float
        Total transmit SNR in decibels.
    n_layers : int
        Number of equally powered spatial layers.

    Returns
    -------
    float
        Capacity averaged over subcarriers in bit/s/Hz.
    """


    rho = 10 ** (snr_db / 10)


    gram_matrices = np.einsum(
        "krl,krm->klm",
        np.conj(G),
        G,
    )


    identity = np.eye(n_layers)


    _sign, natural_log_determinant = np.linalg.slogdet(
        identity + (rho / n_layers) * gram_matrices
    )


    return float(
        np.mean(natural_log_determinant) / np.log(2)
    )


def compute_frame_metrics(
    s_hat: np.ndarray,
    frame,
    G_hat: np.ndarray,
    H: np.ndarray,
    W: np.ndarray,
    snr_db: float,
    cfg,
) -> dict:
    """Calculate all performance metrics for one received frame.

    Parameters
    ----------
    s_hat : numpy.ndarray
        Equalized data symbols with shape ``(n_data, n_layers)``.
    frame : frame object
        Contains transmitted payload, CRC block, layer grid, and data positions.
    G_hat : numpy.ndarray
        Effective channel supplied to the equalizer.
    H : numpy.ndarray
        True physical MIMO channel.
    W : numpy.ndarray
        Applied precoder.
    snr_db : float
        Total transmit SNR in decibels.
    cfg : configuration object
        Must provide modulation, CRC length, and number of layers.

    Returns
    -------
    dict
        Frame-level bit, block, EVM, NMSE, capacity, and payload statistics.
    """


    symbol_indices = frame.data_positions[:, 0]


    subcarrier_indices = frame.data_positions[:, 1]


    transmitted_symbols = frame.layer_grid[
        symbol_indices,
        subcarrier_indices,
        :,
    ]


    recovered_scrambled_bits = qam_demodulate_hard(
        s_hat.reshape(-1),
        cfg.modulation,
    )


    recovered_bits = scramble_bits(
        recovered_scrambled_bits,
        cfg,
    )


    recovered_crc_block = recovered_bits[
        : frame.bits_crc.size
    ]


    recovered_payload = recovered_crc_block[
        : -cfg.crc_len
    ]


    metrics = {}


    metrics["bit_errors"] = int(
        np.sum(recovered_payload != frame.payload)
    )


    metrics["n_bits"] = int(frame.payload.size)


    metrics["block_error"] = int(
        not check_crc24a(recovered_crc_block, cfg)
    )


    metrics["evm"] = float(
        np.sqrt(
            np.mean(
                np.abs(s_hat - transmitted_symbols) ** 2
            )
            / np.mean(
                np.abs(transmitted_symbols) ** 2
            )
        )
    )


    true_effective_channel_matrix = true_effective_channel(
        H,
        W,
    )


    metrics["nmse"] = float(
        np.sum(
            np.abs(
                G_hat - true_effective_channel_matrix
            )
            ** 2
        )
        / np.sum(
            np.abs(true_effective_channel_matrix) ** 2
        )
    )


    metrics["capacity_bps_hz"] = capacity_mimo(
        true_effective_channel_matrix,
        snr_db,
        cfg.n_layers,
    )


    metrics["payload_bits"] = int(frame.payload.size)


    return metrics
