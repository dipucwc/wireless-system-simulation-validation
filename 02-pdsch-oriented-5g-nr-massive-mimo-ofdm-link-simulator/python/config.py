"""
The config module defines the complete locked configuration of the compact
PDSCH-oriented Massive MIMO-OFDM link-level simulator and is the single
source of every campaign parameter, mirroring the MATLAB reference package
field by field so that the two implementations can be cross-verified. The
reproducibility section fixes the random seed, the total-transmit-SNR grid,
and the number of Monte Carlo frames per SNR point. The numerology section
defines the compact OFDM grid of a 256-point FFT with 144 active
subcarriers over one 14-symbol slot at the 30 kHz subcarrier spacing of
the baseline, together with the 0.5 millisecond slot duration used
by the throughput calculation. The MIMO section sets the
antenna counts, the layer count, and the precoder selection, where the
default wideband eigenbeamforming keeps the effective channel smooth in
frequency so that the interpolating least-squares estimator remains valid
on frequency-selective channels. The modulation section selects the
constellation from the QPSK-to-256-QAM range and defines
the CRC24A generator together with the RNTI and scrambling identity of the
Gold-sequence initialization. This is the compact verification
configuration, not the full-scale baseline; the processing chain,
the algorithms, and the metric definitions are identical.
"""
from dataclasses import dataclass, field
from pathlib import Path

# Anchor all package paths at the directory of this file, so every script
# works identically regardless of the terminal's current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    # Reproducibility and campaign control
    random_seed: int = 7                  # fixed seed so every run can be repeated
    snr_db: tuple = (0, 5, 10, 15, 20, 25)   # total transmit SNR grid in dB
    num_frames: int = 40                  # Monte Carlo frames per SNR point

    # OFDM numerology (compact grid)
    n_fft: int = 256
    n_sc: int = 144                       # active subcarriers, centered in the FFT
    n_symbols: int = 14                   # one slot
    scs_hz: float = 30e3                  # subcarrier spacing
    slot_duration_s: float = 0.5e-3       # used by the throughput calculation

    # MIMO
    n_tx: int = 4
    n_rx: int = 4                         # must be >= n_layers for ZF/MMSE
    n_layers: int = 2
    precoder: str = 'svd'                 # 'svd' wideband | 'svd_persc' (ideal CSI
                                          # only on selective channels) | 'mrt' (L=1)
                                          # | 'identity' | 'dft'

    # Modulation, scrambling, CRC
    modulation: str = '16QAM'             # QPSK | 16QAM | 64QAM | 256QAM
    crc_poly: int = 0x864CFB              # CRC24A generator, TS 38.212
    crc_len: int = 24
    n_id: int = 1                         # scrambling identity, enters c_init
    rnti: int = 0x1234                    # enters c_init per TS 38.211

    # DM-RS (layer-orthogonal comb)
    dmrs_symbols: tuple = (3, 10)         # 0-based; the MATLAB package uses [4 11] 1-based
    dmrs_spacing: int = 4                 # comb spacing per layer

    # Channel model
    channel_model: str = 'tdl'            # 'awgn' | 'rayleigh_flat' | 'rayleigh_iid'
                                          # | 'rician' | 'tdl'
    rician_k_db: float = 8.0
    tdl_delays: tuple = (0, 2, 5, 9, 14)  # tap delays in samples
    tdl_powers_db: tuple = (0.0, -2.2, -4.0, -6.0, -8.2)

    # Receiver
    equalizer: str = 'mmse'               # 'zf' or 'mmse' (unbiased)

    # Output locations (absolute, anchored at the package directory)
    output_csv: str = str(PROJECT_ROOT / 'results/python_results_corrected.csv')
    output_fig_dir: str = str(PROJECT_ROOT / 'results/figures')

    @property
    def bits_per_symbol(self) -> int:

        return {'QPSK': 2, '16QAM': 4, '64QAM': 6, '256QAM': 8}[self.modulation.upper()]

    def validate(self) -> None:
        """Validate the complete configuration; raise ValueError on any
        inconsistency. Called by config() and by the link-curve driver, so
        no simulation can start from an invalid parameter set."""
        import math

        if self.n_tx < 1 or self.n_rx < 1 or self.n_layers < 1:
            raise ValueError('Antenna and layer counts must be positive.')
        if self.n_rx < self.n_layers:
            raise ValueError('ZF/MMSE equalization requires n_rx >= n_layers.')
        if self.n_tx < self.n_layers:
            raise ValueError('n_tx must be greater than or equal to n_layers.')
        if self.n_sc <= 0 or self.n_fft <= 0:
            raise ValueError('n_sc and n_fft must be positive.')
        if self.n_sc > self.n_fft:
            raise ValueError('n_sc must not exceed n_fft.')
        if self.n_symbols <= 0:
            raise ValueError('n_symbols must be positive.')
        if self.scs_hz <= 0 or self.slot_duration_s <= 0:
            raise ValueError('scs_hz and slot_duration_s must be positive.')
        if self.num_frames < 1:
            raise ValueError('num_frames must be at least 1.')
        if len(self.snr_db) == 0:
            raise ValueError('snr_db must contain at least one SNR point.')
        if any(not math.isfinite(float(v)) for v in self.snr_db):
            raise ValueError('All SNR values must be finite.')
        if any(not (0 <= int(m) < self.n_symbols) for m in self.dmrs_symbols):
            raise ValueError('Every DM-RS symbol index must lie in [0, n_symbols).')
        if self.dmrs_spacing < self.n_layers:
            raise ValueError('dmrs_spacing must be >= n_layers for a collision-free layer comb.')
        if len(self.tdl_delays) != len(self.tdl_powers_db):
            raise ValueError('tdl_delays and tdl_powers_db must have equal lengths.')
        if self.modulation.upper() not in ('QPSK', '16QAM', '64QAM', '256QAM'):
            raise ValueError(f'Unsupported modulation: {self.modulation}')
        if self.channel_model.lower() not in ('awgn', 'rayleigh_flat', 'rayleigh_iid', 'rician', 'tdl'):
            raise ValueError(f'Unsupported channel model: {self.channel_model}')
        if self.precoder.lower() not in ('svd', 'svd_persc', 'mrt', 'identity', 'dft'):
            raise ValueError(f'Unsupported precoder: {self.precoder}')
        if self.precoder.lower() == 'mrt' and self.n_layers != 1:
            raise ValueError('MRT precoding requires n_layers = 1.')
        if self.equalizer.lower() not in ('zf', 'mmse'):
            raise ValueError(f'Unsupported equalizer: {self.equalizer}')


import dataclasses
import os

_MASSIVE_OVERRIDES = dict(
    n_tx=64,
    n_rx=8,
    n_layers=4,
    precoder='svd',
    channel_model='rayleigh_flat',
    snr_db=(-10, -5, 0, 5, 10, 15, 20),
    num_frames=60,
    output_csv=str(PROJECT_ROOT / 'results/python_results_massive.csv'),
)


def config(profile: str | None = None) -> Config:
    """Return the locked configuration of the selected simulation profile.

    The compact profile (default) is the verification configuration behind
    the reference results; the massive profile is the executed 64x8
    four-layer eigen-beamforming configuration of the massive MIMO
    comparison. When no profile argument is given, the optional
    sim_profile.txt file in the working directory selects the profile, so
    the Python package follows the same persistent switch convention as
    the MATLAB package.
    """
    profile_file = PROJECT_ROOT / 'sim_profile.txt'
    if profile is None:
        profile = 'compact'
        if profile_file.exists():
            profile = profile_file.read_text().strip()

    profile = profile.lower().strip()

    if profile == 'compact':
        cfg = Config()
    elif profile == 'massive':
        cfg = dataclasses.replace(Config(), **_MASSIVE_OVERRIDES)
    else:
        raise ValueError(f'Unknown profile: {profile} (use compact or massive)')
    cfg.validate()
    return cfg


def set_sim_profile(name: str) -> None:
    """Select the profile that config() returns by default.

    The compact default needs no profile file; the massive selection is
    written to sim_profile.txt so it takes effect in every script at once,
    exactly as in the MATLAB package.
    """
    name = name.lower().strip()
    if name not in ('compact', 'massive'):
        raise ValueError(f'Unknown profile: {name} (use compact or massive)')
    profile_file = PROJECT_ROOT / 'sim_profile.txt'
    if name == 'compact':
        if profile_file.exists():
            profile_file.unlink()
        print('Profile: compact (4x4, L = 2, TDL reference configuration).')
    else:
        profile_file.write_text('massive')
        print('Profile: massive (64x8, L = 4, wideband SVD, flat Rayleigh).')
