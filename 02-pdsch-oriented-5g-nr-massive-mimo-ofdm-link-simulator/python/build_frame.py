"""
The build_frame module constructs one PDSCH-like transmission frame. It
first builds the layer-orthogonal DM-RS mask by placing a frequency comb
per layer on the configured DM-RS symbols, offset by the layer index so
that pilots of different layers never collide; every resource element not
used by any layer's pilots carries data on all layers. The transport-block
length is derived so that the payload plus the CRC24A parity exactly fills
the available data resource elements at the configured modulation order.
The payload is generated, the parity attached, the block scrambled with
the NR Gold sequence, and the scrambled bits mapped to unit-power
Gray-coded QAM symbols. The DM-RS values are Gold-seeded, layer-orthogonal,
unit-power QPSK DM-RS-like pilots, deterministic per scrambling identity
and layer, so the receiver divides each pilot observation by its known
value; the pilot initialization c_init = (n_id << 8) + (layer + 1) is a
project-defined simplification, not the complete PDSCH DM-RS
initialization procedure of TS 38.211, which also involves the slot and
symbol indices, the CDM group, and the configured scrambling identities. Data symbols are
written into the layer grid in resource-element-first, layer-second
order, the convention the metric computation reproduces on the receive
side.
"""


import numpy as np
from crc24a import append_crc24a
from scramble_bits import scramble_bits
from qam import qam_modulate
from nr_gold_sequence import nr_gold_sequence


from dataclasses import dataclass


@dataclass
class Frame:
    """One PDSCH-like transmission frame and its construction artifacts."""
    payload: np.ndarray
    bits_crc: np.ndarray
    scrambled_bits: np.ndarray
    layer_grid: np.ndarray
    dmrs_mask: np.ndarray
    dmrs_values: np.ndarray
    data_mask: np.ndarray
    data_positions: np.ndarray


def build_frame(cfg, rng) -> Frame:


    dmrs_mask = np.zeros((cfg.n_symbols, cfg.n_sc, cfg.n_layers), dtype=bool)

    for m in cfg.dmrs_symbols:

        for layer in range(cfg.n_layers):

            dmrs_mask[m, layer::cfg.dmrs_spacing, layer] = True

    pilot_any = dmrs_mask.any(axis=2)

    data_mask = ~pilot_any

    m_idx, k_idx = np.nonzero(data_mask)

    data_positions = np.stack([m_idx, k_idx], axis=1)

    n_data_re = data_positions.shape[0]


    coded_bits_len = n_data_re * cfg.n_layers * cfg.bits_per_symbol

    tb_len = coded_bits_len - cfg.crc_len


    payload = rng.integers(0, 2, tb_len, dtype=np.uint8)

    bits_crc = append_crc24a(payload, cfg)

    scrambled = scramble_bits(bits_crc, cfg)

    qam_symbols = qam_modulate(scrambled, cfg.modulation)


    dmrs_values = np.zeros((cfg.n_symbols, cfg.n_sc, cfg.n_layers), dtype=complex)

    for layer in range(cfg.n_layers):

        mp, kp = np.nonzero(dmrs_mask[:, :, layer])

        c_init = (cfg.n_id << 8) + (layer + 1)

        c = nr_gold_sequence(c_init, 2 * mp.size)

        pil = ((1 - 2*c[0::2].astype(float)) + 1j*(1 - 2*c[1::2].astype(float))) / np.sqrt(2)

        dmrs_values[mp, kp, layer] = pil


    layer_grid = np.zeros((cfg.n_symbols, cfg.n_sc, cfg.n_layers), dtype=complex)

    layer_grid[m_idx, k_idx, :] = qam_symbols.reshape(n_data_re, cfg.n_layers)

    for layer in range(cfg.n_layers):

        mp, kp = np.nonzero(dmrs_mask[:, :, layer])

        layer_grid[mp, kp, layer] = dmrs_values[mp, kp, layer]


    return Frame(
        payload=payload,
        bits_crc=bits_crc,
        scrambled_bits=scrambled,
        layer_grid=layer_grid,
        dmrs_mask=dmrs_mask,
        dmrs_values=dmrs_values,
        data_mask=data_mask,
        data_positions=data_positions,
    )
