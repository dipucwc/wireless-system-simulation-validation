"""
Monte Carlo link-curve engine for the Python PDSCH simulator.

Purpose
-------
This module runs the complete transmitter-channel-receiver chain over a user-
supplied SNR vector.  It is the common campaign engine used by the main
simulation and comparison scripts.

Per-frame processing order
--------------------------
1. Build a new payload, CRC-protected bit block, scrambled symbols, layer grid,
   and DM-RS allocation.
2. Generate an independent MIMO channel realization.
3. Compute the configured precoder.
4. Apply the MIMO channel and additive receiver noise.
5. Select ideal effective CSI or DM-RS-based LS channel estimation.
6. Apply ZF or unbiased-MMSE equalization.
7. Compute BER, BLER, EVM, NMSE, capacity, and successfully delivered payload.
8. Accumulate and average the results over all frames at the current SNR.

Returned result arrays are aligned with the input SNR vector.  The equalized
symbols of the final processed frame are retained for constellation plots.
"""


import numpy as np


from build_frame import build_frame


from generate_channel import generate_channel


from compute_precoder import compute_precoder


from mimo_channel import apply_mimo_channel, true_effective_channel


from estimate_ls import estimate_effective_channel_ls


from equalize import equalize_mimo


from metrics import compute_frame_metrics


def run_link_curve(
    cfg,
    snr_db_vec,
    num_frames,
    ideal_csi,
    rng,
):
    """Run one complete link-level performance curve.

    Parameters
    ----------
    cfg : configuration object
        Complete simulator configuration.
    snr_db_vec : array-like
        SNR values in decibels.
    num_frames : int
        Number of independent Monte Carlo frames per SNR point.
    ideal_csi : bool
        ``True`` uses the exact effective channel ``H @ W``.  ``False`` uses
        the DM-RS-based LS estimate.
    rng : numpy.random.Generator
        Seeded random-number generator.

    Returns
    -------
    dict
        Contains SNR-aligned BER, BLER, EVM, NMSE, capacity, average successful
        payload bits per frame, and the final equalized symbol block.
    """


    cfg.validate()


    snr_db_vec = np.atleast_1d(
        np.asarray(snr_db_vec, dtype=float)
    )


    # Validate direct call arguments: cfg.validate() covers the dataclass,
    # but a caller can still pass an invalid grid, frame count, or generator.
    if snr_db_vec.size == 0:
        raise ValueError("snr_db_vec must contain at least one SNR point.")


    if not np.all(np.isfinite(snr_db_vec)):
        raise ValueError("All supplied SNR values must be finite.")


    if not isinstance(num_frames, (int, np.integer)) or num_frames <= 0:
        raise ValueError("num_frames must be a positive integer.")


    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")


    n_snr_points = snr_db_vec.size


    results = {
        "snr_db": snr_db_vec,
        "ber": np.zeros(n_snr_points),
        "bler": np.zeros(n_snr_points),
        "evm": np.zeros(n_snr_points),
        "nmse": np.zeros(n_snr_points),
        "capacity": np.zeros(n_snr_points),
        "s_hat_sample": None,
        "payload_success": np.zeros(n_snr_points),
        "bit_errors": np.zeros(n_snr_points, dtype=int),
        "evaluated_bits": np.zeros(n_snr_points, dtype=int),
        "block_errors": np.zeros(n_snr_points, dtype=int),
        "payload_bits_per_frame": 0,
    }


    for snr_index, snr_db in enumerate(snr_db_vec):

        total_bit_errors = 0
        total_evaluated_bits = 0
        total_block_errors = 0


        total_evm = 0.0
        total_nmse = 0.0
        total_capacity = 0.0


        successful_payload_bits = 0


        for _frame_index in range(num_frames):

            frame = build_frame(cfg, rng)


            physical_channel = generate_channel(cfg, rng)


            precoder = compute_precoder(
                cfg,
                physical_channel,
            )


            received_grid, noise_variance = apply_mimo_channel(
                frame.layer_grid,
                physical_channel,
                precoder,
                snr_db,
                rng,
            )


            if ideal_csi:


                effective_channel = true_effective_channel(
                    physical_channel,
                    precoder,
                )
            else:

                effective_channel = estimate_effective_channel_ls(
                    received_grid,
                    frame,
                    cfg,
                )


            equalized_symbols = equalize_mimo(
                received_grid,
                effective_channel,
                frame.data_positions,
                noise_variance,
                cfg,
            )


            frame_metrics = compute_frame_metrics(
                equalized_symbols,
                frame,
                effective_channel,
                physical_channel,
                precoder,
                snr_db,
                cfg,
            )


            total_bit_errors += frame_metrics["bit_errors"]


            total_evaluated_bits += frame_metrics["n_bits"]


            total_block_errors += frame_metrics["block_error"]


            total_evm += frame_metrics["evm"]


            total_nmse += frame_metrics["nmse"]


            total_capacity += frame_metrics["capacity_bps_hz"]


            if frame_metrics["block_error"] == 0:
                successful_payload_bits += frame_metrics["payload_bits"]


        results["ber"][snr_index] = (
            total_bit_errors / max(1, total_evaluated_bits)
        )


        results["bler"][snr_index] = (
            total_block_errors / num_frames
        )


        results["evm"][snr_index] = total_evm / num_frames


        results["nmse"][snr_index] = total_nmse / num_frames


        results["capacity"][snr_index] = (
            total_capacity / num_frames
        )


        results["payload_success"][snr_index] = (
            successful_payload_bits / num_frames
        )


        # Raw statistical counts, so every stored rate can be audited
        # against the bit and block populations that produced it.
        results["bit_errors"][snr_index] = total_bit_errors
        results["evaluated_bits"][snr_index] = total_evaluated_bits
        results["block_errors"][snr_index] = total_block_errors


        results["s_hat_sample"] = equalized_symbols

        results["payload_bits_per_frame"] = int(frame_metrics["payload_bits"])


    return results
