"""
Massive MIMO Simulation Run
===========================

The run_massive_mimo script executes the massive MIMO configuration of the
Python simulator, sixty-four transmit antennas, eight receive antennas, and
four spatial layers with wideband eigen-beamforming on the flat Rayleigh
channel, and produces the two results that define the massive claim of the
project, mirroring the MATLAB run_massive_mimo script step by step so the
two implementations can be cross-verified. The verification gates run
first, so no result can come from an unverified chain.

The first part is the paired gain comparison: the same four layers are
transmitted either over an unprecoded four-by-four link or over the
sixty-four-antenna eigen-beamformed array at identical total transmit
power. The random seed is reset before each branch so both runs are
reproducible and begin from the same random-number-stream state; the
channel and noise arrays are not element-wise identical because their
dimensions differ, so the two curves agree in distribution, not sample
by sample. The horizontal separation of the two bit-error-rate curves is
the combined array and SVD eigenbeamforming gain: the comparison changes
the transmit antennas from four to sixty-four, the receive antennas from
four to eight, and the precoding from identity to SVD, so the measured
separation is not a pure beamforming gain. Because the unprecoded curve bottoms out near 4e-2 on
this grid and no bit errors were observed on the eigen-beamformed curve above
5 dB in the finite simulated bit count, a fixed 1e-2 operating point is not always bracketed by both
positive-BER ranges; the measurement therefore selects the lowest
candidate operating point bracketed by both curves and prints it, instead
of returning an undefined value. The comparison figure states the
zero-observed-error region of the massive curve as an annotation, since
zero-error points are hidden by the logarithmic axis.

The second part is the end-to-end massive run with least-squares channel
estimation: the complete receiver chain, DM-RS estimation, unbiased MMSE
equalization, and the full metric set, executed over the massive SNR grid,
with the aggregate results written to the massive CSV file under the same
column names as the MATLAB package, so cross_verify_csv compares the two
directly, and the exact configuration archived beside it.

Output
------
    Command window            Measured beamforming gain and the massive results table.
    ber_massive_gain.png      Paired comparison figure, unprecoded 4x4 against SVD 64x8.
    python_results_massive.csv    End-to-end massive results: BER, BLER, EVM, NMSE,
                              throughput, spectral efficiency, the layer-domain
                              capacity, and the raw statistical counts (frames,
                              evaluated bits, bit errors, block errors, payload
                              bits per frame), with the antenna and layer
                              dimensions recorded per row for traceability.
    Configuration log         Exact massive configuration beside the CSV file.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from build_frame import build_frame
from config import config
from gates import run_verification_gates
from link_curve import run_link_curve
from write_config_log import write_config_log


def main() -> None:
    """Execute the massive MIMO gain comparison and end-to-end run."""
    cfg = config('massive')

    rng = np.random.default_rng(cfg.random_seed)

    run_verification_gates(cfg, rng)


    snr_grid = np.asarray(cfg.snr_db, dtype=float)

    num_frames = cfg.num_frames

    import dataclasses
    cfg_small = dataclasses.replace(cfg, n_tx=4, n_rx=4, n_layers=4, precoder='identity')


    # Reset the seed per branch for reproducibility; the branch dimensions
    # differ, so the drawn channel and noise arrays are not element-wise equal.
    rng_small = np.random.default_rng(cfg.random_seed)
    small4 = run_link_curve(cfg_small, snr_grid, num_frames, True, rng_small)

    rng_massive = np.random.default_rng(cfg.random_seed)
    massive64 = run_link_curve(cfg, snr_grid, num_frames, True, rng_massive)


    targets = (1e-2, 3e-2, 1e-1)


    def bracketed(curve, t):
        """A curve brackets a target when its positive-BER points lie on both sides."""
        ber = curve['ber']
        return np.any((ber > 0) & (ber <= t)) and np.any(ber >= t)


    # A fixed 1e-2 point is not always crossed by both curves on this grid, so
    # take the lowest candidate that both actually bracket.
    target = next((t for t in targets if bracketed(small4, t) and bracketed(massive64, t)), None)
    if target is None:
        raise RuntimeError('No common BER operating point is bracketed by both curves on this grid.')


    def snr_at(curve, t):
        """SNR at the target BER by logarithmic interpolation over the positive-BER points."""
        pos = curve['ber'] > 0
        return float(np.interp(np.log10(t),
                               np.log10(curve['ber'][pos])[::-1],
                               curve['snr_db'][pos][::-1]))


    snr_small = snr_at(small4, target)
    snr_massive = snr_at(massive64, target)
    gain_db = snr_small - snr_massive

    print(f'\nCombined array and eigenbeamforming gain at BER = {target:.0e}: {gain_db:.1f} dB '
          f'(4x4 unprecoded at {snr_small:.1f} dB, 64x8 SVD at {snr_massive:.1f} dB)\n')


    fig_dir = Path(cfg.output_fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))

    y_small = np.where(small4['ber'] > 0, small4['ber'], np.nan)
    y_massive = np.where(massive64['ber'] > 0, massive64['ber'], np.nan)

    ax.semilogy(snr_grid, y_small, '-o', linewidth=1.2, label='4x4 unprecoded, MMSE')
    ax.semilogy(snr_grid, y_massive, '-s', linewidth=1.2, label='64x8 SVD eigenbeamforming, MMSE')

    last_pos = int(np.max(np.nonzero(massive64['ber'] > 0)[0]))
    if last_pos < snr_grid.size - 1:
        frame_probe = build_frame(cfg, np.random.default_rng(cfg.random_seed))
        bits_per_point = frame_probe.payload.size * num_frames
        ax.text(snr_grid[last_pos] + 0.4, y_massive[last_pos],
                f'zero observed bit errors beyond {snr_grid[last_pos]:.0f} dB\n'
                f'({bits_per_point:.1e} evaluated payload bits per point)',
                fontsize=9, verticalalignment='top')

    ax.grid(True, which='both', alpha=0.4)
    ax.set_xlabel('Total transmit SNR (dB)')
    ax.set_ylabel('BER')
    ax.set_title('Combined Array and SVD Eigenbeamforming Gain: '
                 'Unprecoded 4x4 versus SVD-Precoded 64x8\n'
                 f'16QAM, L = 4, flat Rayleigh, MMSE, ideal CSI, '
                 f'combined gain {gain_db:.1f} dB at BER {target:.0e}')
    ax.legend(loc='lower left')
    fig.tight_layout()
    fig.savefig(fig_dir / 'ber_massive_gain.png', dpi=150)
    plt.close(fig)


    rng_end = np.random.default_rng(cfg.random_seed)

    rows = []

    for snr_db in cfg.snr_db:
        curve = run_link_curve(cfg, snr_db, cfg.num_frames, False, rng_end)

        occupied_bandwidth_hz = cfg.n_sc * cfg.scs_hz

        throughput_bits_per_frame = curve['payload_success'][0]
        throughput_mbps = throughput_bits_per_frame / cfg.slot_duration_s / 1e6
        spectral_eff = throughput_mbps * 1e6 / occupied_bandwidth_hz

        rows.append({
            'snrDb': snr_db,
            'ber': curve['ber'][0],
            'bler': curve['bler'][0],
            'evmPercent': 100.0 * curve['evm'][0],
            'nmse': curve['nmse'][0],
            'throughputBitsPerFrame': throughput_bits_per_frame,
            'throughputMbps': throughput_mbps,
            'spectralEffBpsHz': spectral_eff,
            'capacityBpsHz': curve['capacity'][0],
            'numFrames': cfg.num_frames,
            'evaluatedBits': int(curve['evaluated_bits'][0]),
            'bitErrors': int(curve['bit_errors'][0]),
            'blockErrors': int(curve['block_errors'][0]),
            'payloadBits': int(curve['payload_bits_per_frame']),
            'nTx': cfg.n_tx, 'nRx': cfg.n_rx,
            'nLayers': cfg.n_layers,
        })

    out_csv = Path(cfg.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    write_config_log(cfg, str(out_csv))

    header = (f'{"snrDb":>6} {"ber":>12} {"bler":>7} {"evmPercent":>11} '
              f'{"nmse":>11} {"capacityBpsHz":>14} {"bitErrors":>10} '
              f'{"evaluatedBits":>14} {"nTx":>4} {"nRx":>4} {"nLayers":>8}')
    print(header)
    for r in rows:
        print(f'{r["snrDb"]:>6} {r["ber"]:>12.6g} {r["bler"]:>7.3g} {r["evmPercent"]:>11.5g} '
              f'{r["nmse"]:>11.5g} {r["capacityBpsHz"]:>14.5g} {r["bitErrors"]:>10} '
              f'{r["evaluatedBits"]:>14} {r["nTx"]:>4} {r["nRx"]:>4} '
              f'{r["nLayers"]:>8}')

    print(f'\nSaved massive MIMO CSV: {out_csv}')


if __name__ == '__main__':
    main()
