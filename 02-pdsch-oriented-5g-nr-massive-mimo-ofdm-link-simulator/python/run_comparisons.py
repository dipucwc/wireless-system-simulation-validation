"""
Generate the Python comparison and verification campaigns.

The script mirrors the MATLAB comparison campaigns and creates the
comparison overlay figures:

Campaign A
    SISO QPSK and 16-QAM BER in AWGN against closed-form theory.

Campaign B
    SISO QPSK BER in flat Rayleigh fading against the analytical
    reference.

Campaign C
    Full-load 4x4 16-QAM comparison of ZF, ideal-CSI MMSE, and
    LS-estimated MMSE equalization.

Campaign D
    Ergodic iid-Rayleigh MIMO capacity for 2x2, 4x4, 8x8, and the 64x8 massive array.

Campaign E
    Equal-total-power comparison between unprecoded 4x4 MIMO and
    64x8, four-layer, SVD-eigenbeamformed Massive MIMO. This campaign
    is the configuration behind the massive MIMO claim of the project.

Final constellation campaign
    Equalized 16-QAM constellations at low and high SNR.

All verification gates execute before any campaign. Every Monte Carlo
branch receives a deterministic NumPy random-number generator initialized
from the configured seed. Resetting the seed makes each branch
reproducible; branches with different channel dimensions do not have
element-by-element identical channel or noise arrays.
"""

from __future__ import annotations

import dataclasses
import os

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

from build_frame import build_frame
from config import Config
from gates import run_verification_gates
from link_curve import run_link_curve


def _positive_or_nan(values: np.ndarray) -> np.ndarray:
    """Replace zero BER points by NaN for logarithmic plotting."""


    values = np.asarray(values, dtype=float)


    return np.where(values > 0.0, values, np.nan)


def _save_figure(
    figure,
    path: str,
) -> None:
    """Apply a tight layout, save the figure, and release memory."""


    figure.tight_layout()


    figure.savefig(
        path,
        dpi=150,
    )


    plt.close(figure)


def campaign_awgn_vs_theory(base, figure_directory, n_frames_awgn=100, n_frames_rayleigh=300, n_frames_equalizer=100, n_frames_massive=60):
    """AWGN QPSK and 16-QAM against the closed-form curves, SISO, ideal CSI."""
    cfg_awgn = dataclasses.replace(
        base,
        channel_model="awgn",
        n_tx=1,
        n_rx=1,
        n_layers=1,
        precoder="svd",
    )


    snr_awgn_db = np.arange(
        0,
        15,
        2,
    )


    qpsk_awgn = run_link_curve(
        dataclasses.replace(
            cfg_awgn,
            modulation="QPSK",
        ),
        snr_awgn_db,
        n_frames_awgn,
        True,
        np.random.default_rng(base.random_seed),
    )


    qam16_awgn = run_link_curve(
        dataclasses.replace(
            cfg_awgn,
            modulation="16QAM",
        ),
        snr_awgn_db,
        n_frames_awgn,
        True,
        np.random.default_rng(base.random_seed),
    )


    gamma_symbol = 10.0 ** (
        snr_awgn_db / 10.0
    )


    theory_qpsk_awgn = 0.5 * erfc(
        np.sqrt(gamma_symbol) / np.sqrt(2.0)
    )


    theory_qam16_awgn = (
        0.75
        * 0.5
        * erfc(
            np.sqrt(
                3.0 * gamma_symbol / 15.0
            )
            / np.sqrt(2.0)
        )
    )


    figure, axis = plt.subplots(
        figsize=(9, 6)
    )


    axis.semilogy(
        snr_awgn_db,
        _positive_or_nan(qpsk_awgn["ber"]),
        "o",
    )


    axis.semilogy(
        snr_awgn_db,
        theory_qpsk_awgn,
        "-",
    )


    axis.semilogy(
        snr_awgn_db,
        _positive_or_nan(qam16_awgn["ber"]),
        "s",
    )


    axis.semilogy(
        snr_awgn_db,
        theory_qam16_awgn,
        "--",
    )


    axis.grid(
        True,
        which="both",
        alpha=0.4,
    )


    axis.set_xlabel(
        "Total transmit SNR (dB)"
    )


    axis.set_ylabel("BER")


    axis.set_title(
        "Uncoded BER in AWGN: Simulation vs Closed-Form Theory\n"
        "SISO, ideal CSI"
    )


    axis.legend(
        [
            "QPSK simulation",
            "QPSK theory",
            "16-QAM simulation",
            "16-QAM theory",
        ],
        loc="lower left",
    )


    _save_figure(
        figure,
        os.path.join(
            figure_directory,
            "ber_awgn_vs_theory.png",
        ),
    )


    print(
        "AWGN QPSK vs theory decades at 4/8 dB: "
        f"{abs(np.log10(qpsk_awgn['ber'][2]) - np.log10(theory_qpsk_awgn[2])):.3f} / "
        f"{abs(np.log10(qpsk_awgn['ber'][4]) - np.log10(theory_qpsk_awgn[4])):.3f}"
    )


def campaign_rayleigh_vs_theory(base, figure_directory, n_frames_awgn=100, n_frames_rayleigh=300, n_frames_equalizer=100, n_frames_massive=60):
    """Flat-Rayleigh QPSK against the closed-form curve, SISO, ideal CSI."""
    cfg_rayleigh = dataclasses.replace(
        base,
        channel_model="rayleigh_flat",
        modulation="QPSK",
        n_tx=1,
        n_rx=1,
        n_layers=1,
        precoder="svd",
    )


    snr_rayleigh_db = np.arange(
        0,
        31,
        5,
    )


    rayleigh_result = run_link_curve(
        cfg_rayleigh,
        snr_rayleigh_db,
        n_frames_rayleigh,
        True,
        np.random.default_rng(base.random_seed),
    )


    gamma_bit = (
        10.0 ** (snr_rayleigh_db / 10.0)
    ) / 2.0


    theory_rayleigh = 0.5 * (
        1.0
        - np.sqrt(
            gamma_bit / (1.0 + gamma_bit)
        )
    )


    theory_awgn_reference = np.maximum(
        0.5
        * erfc(
            np.sqrt(
                10.0 ** (snr_rayleigh_db / 10.0)
            )
            / np.sqrt(2.0)
        ),
        1e-8,
    )


    figure, axis = plt.subplots(
        figsize=(9, 6)
    )


    axis.semilogy(
        snr_rayleigh_db,
        _positive_or_nan(rayleigh_result["ber"]),
        "o",
    )


    axis.semilogy(
        snr_rayleigh_db,
        theory_rayleigh,
        "-",
    )


    axis.semilogy(
        snr_rayleigh_db,
        theory_awgn_reference,
        "--",
    )


    axis.set_ylim(
        1e-6,
        1.0,
    )


    axis.grid(
        True,
        which="both",
        alpha=0.4,
    )


    axis.set_xlabel(
        "Total transmit SNR (dB)"
    )
    axis.set_ylabel("BER")


    axis.set_title(
        "Uncoded QPSK BER in Flat Rayleigh Fading "
        "vs Closed-Form Theory\n"
        "SISO, ideal CSI"
    )


    axis.legend(
        [
            "Rayleigh simulation",
            "Rayleigh theory",
            "AWGN theory reference",
        ],
        loc="lower left",
    )


    _save_figure(
        figure,
        os.path.join(
            figure_directory,
            "ber_rayleigh_vs_theory.png",
        ),
    )


def campaign_zf_vs_mmse(base, figure_directory, n_frames_awgn=100, n_frames_rayleigh=300, n_frames_equalizer=100, n_frames_massive=60):
    """ZF against ideal-CSI MMSE and LS-estimated MMSE, full-load 4x4."""
    cfg_equalizer = dataclasses.replace(
        base,
        channel_model="rayleigh_flat",
        modulation="16QAM",
        n_layers=4,
        precoder="identity",
    )


    snr_equalizer_db = np.arange(
        0,
        26,
        5,
    )


    zf_ideal = run_link_curve(
        dataclasses.replace(
            cfg_equalizer,
            equalizer="zf",
        ),
        snr_equalizer_db,
        n_frames_equalizer,
        True,
        np.random.default_rng(base.random_seed),
    )


    mmse_ideal = run_link_curve(
        dataclasses.replace(
            cfg_equalizer,
            equalizer="mmse",
        ),
        snr_equalizer_db,
        n_frames_equalizer,
        True,
        np.random.default_rng(base.random_seed),
    )


    mmse_ls = run_link_curve(
        dataclasses.replace(
            cfg_equalizer,
            equalizer="mmse",
        ),
        snr_equalizer_db,
        n_frames_equalizer,
        False,
        np.random.default_rng(base.random_seed),
    )


    figure, axis = plt.subplots(
        figsize=(9, 6)
    )


    for result, line_style in (
        (zf_ideal, "-^"),
        (mmse_ideal, "-o"),
        (mmse_ls, "-s"),
    ):
        axis.semilogy(
            snr_equalizer_db,
            _positive_or_nan(result["ber"]),
            line_style,
            linewidth=1.2,
        )


    axis.grid(
        True,
        which="both",
        alpha=0.4,
    )


    axis.set_xlabel(
        "Total transmit SNR (dB)"
    )
    axis.set_ylabel("BER")


    axis.set_title(
        "ZF vs MMSE Equalization, Ideal CSI and LS Estimation\n"
        f"16-QAM, {cfg_equalizer.n_tx}x{cfg_equalizer.n_rx}, "
        f"L = {cfg_equalizer.n_layers}, flat Rayleigh"
    )


    axis.legend(
        [
            "ZF, ideal CSI",
            "MMSE, ideal CSI",
            "MMSE, LS estimation",
        ],
        loc="lower left",
    )


    _save_figure(
        figure,
        os.path.join(
            figure_directory,
            "ber_zf_vs_mmse.png",
        ),
    )


    print(
        "ZF vs MMSE (ideal, paired) at 5/15 dB: "
        f"{zf_ideal['ber'][1]:.3e}/{mmse_ideal['ber'][1]:.3e}  "
        f"{zf_ideal['ber'][3]:.3e}/{mmse_ideal['ber'][3]:.3e}"
    )


def campaign_ergodic_capacity(base, figure_directory, n_frames_awgn=100, n_frames_rayleigh=300, n_frames_equalizer=100, n_frames_massive=60):
    """Ergodic iid-Rayleigh capacity: 2x2, 4x4, 8x8, and the 64x8 massive array."""
    snr_capacity_db = np.arange(
        0,
        26,
        5,
    )


    capacity_rng = np.random.default_rng(
        base.random_seed
    )


    n_capacity_realizations = 200


    figure, axis = plt.subplots(
        figsize=(9, 6)
    )


    for (
        (n_tx, n_rx),
        line_style,
    ) in (
        ((2, 2), "-o"),
        ((4, 4), "-s"),
        ((8, 8), "-^"),
        ((64, 8), "-d"),
    ):

        capacity = np.zeros(
            snr_capacity_db.size
        )


        for snr_index, snr_db in enumerate(
            snr_capacity_db
        ):

            rho = 10.0 ** (snr_db / 10.0)


            accumulated_capacity = 0.0


            for _ in range(
                n_capacity_realizations
            ):

                channel = (
                    capacity_rng.standard_normal(
                        (n_rx, n_tx)
                    )
                    + 1j
                    * capacity_rng.standard_normal(
                        (n_rx, n_tx)
                    )
                ) / np.sqrt(2.0)


                capacity_matrix = (
                    np.eye(n_rx)
                    + (rho / n_tx)
                    * (
                        channel
                        @ channel.conj().T
                    )
                )


                _, log_determinant = np.linalg.slogdet(
                    capacity_matrix
                )


                accumulated_capacity += (
                    log_determinant / np.log(2.0)
                )


            capacity[snr_index] = (
                accumulated_capacity
                / n_capacity_realizations
            )


        axis.plot(
            snr_capacity_db,
            capacity,
            line_style,
            linewidth=1.2,
        )


    axis.grid(
        True,
        alpha=0.4,
    )


    axis.set_xlabel(
        "Total transmit SNR (dB)"
    )
    axis.set_ylabel(
        "Ergodic capacity (bit/s/Hz)"
    )


    axis.set_title(
        "Ergodic MIMO Capacity vs SNR\n"
        "iid Rayleigh, equal power allocation, including the 64x8 massive array"
    )


    axis.legend(
        ["2x2", "4x4", "8x8", "64x8 massive"],
        loc="upper left",
    )


    _save_figure(
        figure,
        os.path.join(
            figure_directory,
            "capacity_multi_config.png",
        ),
    )


def campaign_massive_gain(base, figure_directory, n_frames_awgn=100, n_frames_rayleigh=300, n_frames_equalizer=100, n_frames_massive=60):
    """Unprecoded 4x4 against SVD 64x8 at equal total transmit power."""
    snr_massive_db = np.arange(
        -10,
        21,
        5,
    )


    cfg_small = dataclasses.replace(
        base,
        channel_model="rayleigh_flat",
        modulation="16QAM",
        equalizer="mmse",
        n_tx=4,
        n_rx=4,
        n_layers=4,
        precoder="identity",
    )


    result_small = run_link_curve(
        cfg_small,
        snr_massive_db,
        n_frames_massive,
        True,
        np.random.default_rng(base.random_seed),
    )


    cfg_massive = dataclasses.replace(
        cfg_small,
        n_tx=64,
        n_rx=8,
        precoder="svd",
    )


    result_massive = run_link_curve(
        cfg_massive,
        snr_massive_db,
        n_frames_massive,
        True,
        np.random.default_rng(base.random_seed),
    )


    figure, axis = plt.subplots(
        figsize=(9, 6)
    )


    small_ber_plot = _positive_or_nan(
        result_small["ber"]
    )
    massive_ber_plot = _positive_or_nan(
        result_massive["ber"]
    )


    axis.semilogy(
        snr_massive_db,
        small_ber_plot,
        "-o",
        linewidth=1.2,
    )


    axis.semilogy(
        snr_massive_db,
        massive_ber_plot,
        "-s",
        linewidth=1.2,
    )


    axis.grid(
        True,
        which="both",
        alpha=0.4,
    )


    axis.set_xlabel(
        "Total transmit SNR (dB)"
    )
    axis.set_ylabel("BER")


    axis.set_title(
        "Combined Array and SVD Eigenbeamforming Gain: "
        "Unprecoded 4x4 versus SVD-Precoded 64x8\n"
        "16-QAM, L = 4, flat Rayleigh, MMSE, "
        "ideal CSI, equal total transmit power"
    )


    axis.legend(
        [
            "4x4 unprecoded, MMSE",
            "64x8 SVD eigenbeamforming, MMSE",
        ],
        loc="lower left",
    )


    finite_indices = np.nonzero(
        ~np.isnan(massive_ber_plot)
    )[0]


    if (
        finite_indices.size
        and finite_indices[-1]
        < snr_massive_db.size - 1
    ):

        example_frame = build_frame(
            cfg_massive,
            np.random.default_rng(0),
        )


        bits_per_point = (
            n_frames_massive
            * example_frame.payload.size
        )


        last_finite_index = finite_indices[-1]


        axis.annotate(
            (
                "zero observed errors beyond "
                f"{snr_massive_db[last_finite_index]} dB\n"
                f"({bits_per_point:.2g} payload bits/point)"
            ),
            (
                snr_massive_db[last_finite_index] + 0.4,
                massive_ber_plot[last_finite_index],
            ),
            fontsize=9,
        )


    _save_figure(
        figure,
        os.path.join(
            figure_directory,
            "ber_massive_64x8_vs_4x4.png",
        ),
    )


    print(
        "Massive at 0 dB: "
        f"4x4 {result_small['ber'][2]:.3e} vs "
        f"64x8 {result_massive['ber'][2]:.3e}"
    )


def campaign_constellations(base, figure_directory, n_frames_awgn=100, n_frames_rayleigh=300, n_frames_equalizer=100, n_frames_massive=60):
    """Equalized 16-QAM constellations at low and high SNR."""
    cfg_constellation = dataclasses.replace(
        base,
        equalizer="mmse",
    )


    low_snr_result = run_link_curve(
        cfg_constellation,
        10,
        1,
        False,
        np.random.default_rng(base.random_seed),
    )


    high_snr_result = run_link_curve(
        cfg_constellation,
        25,
        1,
        False,
        np.random.default_rng(base.random_seed),
    )


    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 6),
    )


    for (
        axis,
        result,
        label,
    ) in (
        (axes[0], low_snr_result, "10 dB"),
        (axes[1], high_snr_result, "25 dB"),
    ):

        symbols = result["s_hat_sample"].ravel()


        axis.plot(
            symbols.real,
            symbols.imag,
            ".",
            markersize=2,
        )


        axis.set_xlim(-1.6, 1.6)
        axis.set_ylim(-1.6, 1.6)


        axis.set_aspect("equal")


        axis.grid(
            True,
            alpha=0.4,
        )


        axis.set_xlabel("In-phase")
        axis.set_ylabel("Quadrature")


        axis.set_title(
            f"Equalized 16-QAM, {label}"
        )


        axis.legend(
            ["Equalized data symbols"],
            loc="upper right",
        )


    _save_figure(
        figure,
        os.path.join(
            figure_directory,
            "constellation_low_high_snr.png",
        ),
    )


def main(quick: bool = False) -> None:
    """Run the gates, then every comparison campaign in order.

    The default frame counts are the publication configuration behind the
    stored figures. The quick mode (``python run_comparisons.py --quick``)
    runs the identical chain with reduced frame counts for automated
    testing and continuous integration; its outputs are smoke-test
    artifacts, not publication results.
    """
    base = Config()

    gate_rng = np.random.default_rng(base.random_seed)
    run_verification_gates(base, gate_rng)

    figure_directory = base.output_fig_dir
    os.makedirs(figure_directory, exist_ok=True)

    if quick:
        frames = dict(n_frames_awgn=10, n_frames_rayleigh=30,
                      n_frames_equalizer=10, n_frames_massive=6)
        print("Quick mode: reduced frame counts, smoke-test outputs only.")
    else:
        frames = {}

    # Each campaign seeds its own generator from the same base seed, so the
    # campaigns are independent and can also be run one at a time.
    campaign_awgn_vs_theory(base, figure_directory, **frames)
    campaign_rayleigh_vs_theory(base, figure_directory, **frames)
    campaign_zf_vs_mmse(base, figure_directory, **frames)
    campaign_ergodic_capacity(base, figure_directory, **frames)
    campaign_massive_gain(base, figure_directory, **frames)
    campaign_constellations(base, figure_directory, **frames)



    print(f"Comparison figures saved to: {figure_directory}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the Python comparison campaigns.")
    parser.add_argument(
        "--quick", action="store_true",
        help="reduced frame counts for automated testing; "
             "outputs are not publication results")
    main(quick=parser.parse_args().quick)
