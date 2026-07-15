"""
PDSCH-Oriented Massive MIMO-OFDM Link-Level Simulation
======================================================

This script runs the Python link-level simulation over the configured SNR
range. It performs the verification checks, runs the frame simulation,
builds the result table, writes the CSV file, stores the configuration,
and saves the seven result figures.

Input
-----
The Config object provides the random seed, SNR values, number of frames,
antenna dimensions, OFDM settings, modulation, channel model, equalizer,
and output paths.

Output
------
The script returns one result row for each SNR point and saves:

    * CSV result table
    * Configuration log
    * BER figure
    * BLER figure
    * EVM figure
    * NMSE figure
    * Throughput figure
    * Spectral-efficiency figure
    * MIMO-capacity figure
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from config import Config
from gates import run_verification_gates
from link_curve import run_link_curve
from write_config_log import write_config_log


FIGURE_SIZE = (9, 6)
FIGURE_DPI = 150
LINE_WIDTH = 1.2
GRID_ALPHA = 0.4


def validate_config(cfg: Config) -> None:
    """Check the parameters required by the link-level simulation."""

    if cfg.n_rx < cfg.n_layers:
        raise ValueError(
            "ZF/MMSE equalization requires n_rx >= n_layers "
            "."
        )

    if cfg.n_tx < cfg.n_layers:
        raise ValueError(
            "n_tx must be greater than or equal to n_layers."
        )

    if cfg.num_frames < 1:
        raise ValueError(
            "num_frames must be at least 1."
        )

    if len(cfg.snr_db) == 0:
        raise ValueError(
            "snr_db must contain at least one SNR point."
        )

    if cfg.n_sc <= 0:
        raise ValueError(
            "n_sc must be positive."
        )

    if cfg.scs_hz <= 0:
        raise ValueError(
            "scs_hz must be positive."
        )

    if cfg.slot_duration_s <= 0:
        raise ValueError(
            "slot_duration_s must be positive."
        )


def validate_results(results: Mapping[str, Any]) -> None:
    """Check that the link simulation returned every required result."""

    required_fields = {
        "snr_db",
        "ber",
        "bler",
        "evm",
        "nmse",
        "payload_success",
        "capacity",
        "bit_errors",
        "evaluated_bits",
        "block_errors",
        "payload_bits_per_frame",
    }

    missing_fields = required_fields.difference(
        results.keys()
    )

    if missing_fields:
        raise KeyError(
            "Missing result fields: "
            + ", ".join(sorted(missing_fields))
        )

    number_of_points = len(
        results["snr_db"]
    )

    if number_of_points == 0:
        raise ValueError(
            "The simulation returned no SNR results."
        )

    vector_fields = required_fields - {
        "snr_db",
        "payload_bits_per_frame",
    }

    for field_name in vector_fields:
        if len(results[field_name]) != number_of_points:
            raise ValueError(
                f"Result length mismatch: {field_name} has "
                f"{len(results[field_name])} values, while snr_db has "
                f"{number_of_points}."
            )

    payload_bits_per_frame = results["payload_bits_per_frame"]

    if (
        not isinstance(
            payload_bits_per_frame,
            (int, np.integer),
        )
        or payload_bits_per_frame <= 0
    ):
        raise ValueError(
            "payload_bits_per_frame must be a positive integer."
        )


def build_result_rows(
    cfg: Config,
    results: Mapping[str, Sequence[float]],
) -> list[dict[str, float | int]]:
    """Build one result-table row for each SNR point."""

    occupied_bandwidth_hz = (
        cfg.n_sc * cfg.scs_hz
    )

    rows: list[dict[str, float | int]] = []

    for index, snr_db in enumerate(
        results["snr_db"]
    ):

        successful_bits_per_frame = float(
            results["payload_success"][index]
        )

        throughput_mbps = (
            successful_bits_per_frame
            / cfg.slot_duration_s
            / 1e6
        )

        spectral_efficiency = (
            throughput_mbps
            * 1e6
            / occupied_bandwidth_hz
        )

        row = {
            "snrDb": float(snr_db),
            "ber": float(results["ber"][index]),
            "bler": float(results["bler"][index]),
            "evmPercent": (
                100.0 * float(results["evm"][index])
            ),
            "nmse": float(results["nmse"][index]),
            "throughputBitsPerFrame": (
                successful_bits_per_frame
            ),
            "throughputMbps": throughput_mbps,
            "spectralEffBpsHz": spectral_efficiency,
            "capacityBpsHz": float(
                results["capacity"][index]
            ),
            "numFrames": int(cfg.num_frames),
            "evaluatedBits": int(results["evaluated_bits"][index]),
            "bitErrors": int(results["bit_errors"][index]),
            "blockErrors": int(results["block_errors"][index]),
            "payloadBits": int(results["payload_bits_per_frame"]),
            "nTx": int(cfg.n_tx),
            "nRx": int(cfg.n_rx),
            "nLayers": int(cfg.n_layers),
        }

        rows.append(row)

    return rows


def save_results_csv(
    rows: Sequence[Mapping[str, float | int]],
    output_csv: str | Path,
) -> Path:
    """Write the result table to CSV."""

    if not rows:
        raise ValueError(
            "No result rows are available for CSV output."
        )

    output_path = Path(output_csv)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    return output_path


def print_result_rows(
    rows: Sequence[Mapping[str, float | int]],
) -> None:
    """Print the result rows with readable numeric precision."""

    for row in rows:

        display_row = {
            key: (
                round(value, 6)
                if isinstance(value, float)
                else value
            )
            for key, value in row.items()
        }

        print(display_row)


def save_figure(
    snr_db: Sequence[float],
    values: Sequence[float],
    y_label: str,
    title: str,
    legend_text: str,
    file_path: Path,
    config_text: str,
    *,
    log_scale: bool = False,
    legend_location: str = "best",
) -> None:
    """Create and save one result figure."""

    x_values = np.asarray(
        snr_db,
        dtype=float,
    )

    y_values = np.asarray(
        values,
        dtype=float,
    )

    if x_values.shape != y_values.shape:
        raise ValueError(
            f"Cannot plot '{title}': SNR and result sizes do not match."
        )

    plot_values = y_values.copy()

    fig, axis = plt.subplots(
        figsize=FIGURE_SIZE
    )

    if log_scale:

        plot_values[plot_values <= 0.0] = np.nan

        axis.semilogy(
            x_values,
            plot_values,
            "-o",
            linewidth=LINE_WIDTH,
            label=legend_text,
        )

    else:

        axis.plot(
            x_values,
            plot_values,
            "-o",
            linewidth=LINE_WIDTH,
            label=legend_text,
        )

    axis.grid(
        True,
        which="both",
        alpha=GRID_ALPHA,
    )

    axis.set_xlabel(
        "Total transmit SNR (dB)"
    )

    axis.set_ylabel(y_label)

    axis.set_title(
        f"{title}\n{config_text}"
    )

    axis.legend(
        loc=legend_location
    )

    fig.tight_layout()

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        file_path,
        dpi=FIGURE_DPI,
    )

    plt.close(fig)


def save_all_figures(
    cfg: Config,
    results: Mapping[str, Sequence[float]],
    rows: Sequence[Mapping[str, float | int]],
) -> Path:
    """Create the seven simulation result figures."""

    output_dir = Path(
        cfg.output_fig_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_text = (
        f"{cfg.modulation.upper()}, "
        f"{cfg.n_tx}x{cfg.n_rx}, "
        f"L = {cfg.n_layers}, "
        f"{cfg.channel_model.upper()}, "
        f"{cfg.equalizer.upper()}"
    )

    throughput = [
        float(row["throughputMbps"])
        for row in rows
    ]

    spectral_efficiency = [
        float(row["spectralEffBpsHz"])
        for row in rows
    ]

    save_figure(
        results["snr_db"],
        results["ber"],
        "BER",
        "Bit Error Rate vs SNR",
        "Simulated BER",
        output_dir / "ber_vs_snr.png",
        config_text,
        log_scale=True,
        legend_location="lower left",
    )

    save_figure(
        results["snr_db"],
        results["bler"],
        "BLER",
        "Block Error Rate vs SNR (uncoded transport block)",
        "Simulated BLER",
        output_dir / "bler_vs_snr.png",
        config_text,
        log_scale=True,
        legend_location="lower left",
    )

    save_figure(
        results["snr_db"],
        100.0 * np.asarray(
            results["evm"],
            dtype=float,
        ),
        "EVM (%)",
        "Error Vector Magnitude vs SNR",
        "RMS EVM",
        output_dir / "evm_vs_snr.png",
        config_text,
        legend_location="upper right",
    )

    save_figure(
        results["snr_db"],
        results["nmse"],
        "NMSE",
        "Channel-Estimation NMSE vs SNR (LS on DM-RS)",
        "LS estimation NMSE",
        output_dir / "nmse_vs_snr.png",
        config_text,
        log_scale=True,
        legend_location="lower left",
    )

    save_figure(
        results["snr_db"],
        throughput,
        "Throughput (Mbit/s)",
        "Throughput vs SNR (uncoded transport block)",
        "Throughput of CRC-passed blocks",
        output_dir / "throughput_vs_snr.png",
        config_text,
        legend_location="upper left",
    )

    save_figure(
        results["snr_db"],
        spectral_efficiency,
        "Spectral efficiency (bit/s/Hz)",
        "Spectral Efficiency vs SNR",
        "Spectral efficiency",
        output_dir / "spectraleff_vs_snr.png",
        config_text,
        legend_location="upper left",
    )

    save_figure(
        results["snr_db"],
        results["capacity"],
        "Capacity (bit/s/Hz)",
        "MIMO Capacity vs SNR",
        "Layer-domain capacity",
        output_dir / "capacity_vs_snr.png",
        config_text,
        legend_location="upper left",
    )

    return output_dir


def main() -> list[dict[str, float | int]]:
    """Run the verified PDSCH-oriented link-level simulation."""

    cfg = Config()

    validate_config(cfg)

    rng = np.random.default_rng(
        cfg.random_seed
    )

    run_verification_gates(
        cfg,
        rng,
    )

    results = run_link_curve(
        cfg,
        cfg.snr_db,
        cfg.num_frames,
        ideal_csi=False,
        rng=rng,
    )

    validate_results(results)

    rows = build_result_rows(
        cfg,
        results,
    )

    csv_path = save_results_csv(
        rows,
        cfg.output_csv,
    )

    write_config_log(
        cfg,
        str(csv_path),
    )

    print_result_rows(rows)

    print(
        f"Saved Python CSV: {csv_path}"
    )

    figure_dir = save_all_figures(
        cfg,
        results,
        rows,
    )

    print(
        f"Saved figures to: {figure_dir}"
    )

    return rows


if __name__ == "__main__":
    main()
