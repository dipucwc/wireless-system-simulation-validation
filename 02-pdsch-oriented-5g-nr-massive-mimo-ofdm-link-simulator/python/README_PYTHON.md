# Python Reference Implementation

## 3GPP 5G NR PDSCH-Oriented MIMO and Massive MIMO-OFDM Link-Level Simulator

This folder contains the independent NumPy/SciPy reference implementation
used to cross-check the corresponding MATLAB simulator. The code is
organized as a modular physical-layer processing chain covering frame
construction, CRC protection, scrambling, Gray-coded QAM, layer mapping,
MIMO precoding, channel generation, effective-channel estimation, linear
equalization, performance metrics, OFDM verification, and Monte Carlo
campaign execution.

The package contains two distinct operating scopes:

- **Compact verification configuration:** 4x4 MIMO with two spatial
  layers, used for the main end-to-end campaign and MATLAB/Python
  cross-verification.
- **Massive MIMO comparison configuration:** 64 transmit antennas,
  eight receive antennas, four spatial layers, and SVD eigenbeamforming,
  used for the report-aligned 4x4-versus-64x8 comparison in
  Section 11.7 and Figure 26.

The Python implementation is a numerical reference model. It is not a
complete 3GPP conformance implementation and does not include every
optional PDSCH feature.

---

## 1. Main execution commands

Run the primary compact campaign:

```bash
python main.py
```

This command performs:

1. verification gates;
2. the Monte Carlo SNR sweep;
3. CSV result generation;
4. configuration-log generation;
5. Figures 1-7.

Run the comparison campaigns:

```bash
python run_comparisons.py
```

This command produces the comparison figures for:

- AWGN BER versus theory;
- flat-Rayleigh BER versus theory;
- ZF versus MMSE equalization;
- 2x2, 4x4, and 8x8 ergodic capacity;
- unprecoded 4x4 versus SVD-precoded 64x8 Massive MIMO;
- low- and high-SNR equalized constellations.

Run the recorded MATLAB/Python CSV comparison:

```bash
python cross_verify_csv.py <matlab.csv> <python.csv>
```

Example:

```bash
python cross_verify_csv.py \
    ../02_MATLAB/results/matlab_results.csv \
    results/python_results.csv
```

---

## 2. Package requirements

The reference implementation requires:

```text
numpy
scipy
matplotlib
```

Install them with:

```bash
python -m pip install numpy scipy matplotlib
```

Python 3.11 or newer is required: the pinned NumPy 2.4.4 and SciPy 1.17.1 both require Python 3.11+. The configuration log stored beside each result CSV records the exact interpreter, library, and platform versions of the execution that produced it (the recorded reference runs in this package used Python 3.12.3 on Linux; the verified Windows execution used Python 3.14).

---

## 3. Processing architecture

The compact end-to-end chain is:

```text
Payload generation
    -> CRC24A attachment
    -> NR Gold-sequence scrambling
    -> Gray-coded QAM mapping
    -> Layer mapping
    -> MIMO precoding
    -> Resource-grid and DM-RS construction
    -> MIMO channel and AWGN
    -> Ideal or LS effective-channel knowledge
    -> ZF or unbiased-MMSE equalization
    -> Hard QAM demapping
    -> Descrambling
    -> CRC verification
    -> BER / BLER / EVM / NMSE / capacity / throughput statistics
```

The transform-domain OFDM functions are verified separately by the
pre-campaign OFDM round-trip gate. The main compact link applies the
channel in the frequency domain, which is equivalent under the cyclic-
prefix condition used by the model.

---

## 4. Core array dimensions

For a general configuration:

```text
Physical channel H:
    (n_sc, n_rx, n_tx)

Wideband precoder W:
    (n_tx, n_layers)

Per-subcarrier precoder W:
    (n_sc, n_tx, n_layers)

Effective channel G = H W:
    (n_sc, n_rx, n_layers)

Layer grid:
    (n_symbols, n_sc, n_layers)

Received grid:
    (n_symbols, n_sc, n_rx)
```

For the 64x8 Massive MIMO campaign:

```text
H[k] : 8 x 64
W    : 64 x 4
G[k] : 8 x 4
s[k] : 4 x 1
x[k] : 64 x 1
y[k] : 8 x 1
```

---

## 5. SNR and power convention

The simulator uses total transmitted SNR. With unit-power layers and
unit-norm precoder columns, the average total transmit power equals the
number of layers:

```text
P_total = n_layers
```

The complex receiver-noise variance is therefore:

```text
noise_var = n_layers / 10**(snr_db / 10)
```

This convention is applied consistently to the compact and Massive MIMO
branches. The 4x4 and 64x8 comparison uses the same four layers and the
same total transmitted power, so the observed curve separation represents
array and eigenbeamforming gain rather than additional radiated power.

---

## 6. Precoding and equalization

The default `svd` mode uses a wideband eigenbeamforming matrix obtained
from the dominant eigenvectors of the average transmit-side channel
covariance.

A separate per-subcarrier SVD mode may be used for ideal-CSI analysis on
frequency-selective channels.

The receiver supports:

- zero forcing;
- unbiased MMSE.

The MMSE detector first computes the regularized solution and then removes
the per-layer amplitude bias before hard QAM slicing.

---

## 7. Channel models

The package includes:

- AWGN identity channel;
- flat Rayleigh fading;
- independent per-subcarrier Rayleigh fading;
- Rician fading with configurable K-factor;
- configurable tapped-delay-line frequency response.

The configurable TDL is a verification-oriented tapped-delay model. It
should not be described as a complete implementation of every standardized
3GPP TDL-A/B/C/D/E profile unless those exact profiles are added.

---

## 8. Verification gates

Every main campaign begins with four mandatory checks.

### Gate 1 — constellation normalization

The exhaustive constellation of QPSK, 16-QAM, 64-QAM, and 256-QAM must
have unit average power.

### Gate 2 — modulation round trip

A noiseless modulation-demodulation cycle must recover every tested bit
exactly.

### Gate 3 — OFDM round trip

OFDM modulation followed by demodulation must reconstruct the complete
resource grid at machine precision.

### Gate 4 — CRC24A self-test

The CRC checker must accept an intact block and reject a deliberately
corrupted block.

No Monte Carlo campaign proceeds after a failed gate.

---

## 9. Main modules

| File | Purpose |
|---|---|
| `config.py` | Simulation configuration and output paths |
| `crc24a.py` | CRC24A generation and checking |
| `nr_gold_sequence.py` | TS 38.211 length-31 Gold sequence |
| `scramble_bits.py` | PDSCH scrambling and descrambling |
| `qam.py` | Shared Gray-coded QAM mapper and hard demapper |
| `build_frame.py` | Payload, CRC, scrambling, layer grid, and Gold-seeded, layer-orthogonal QPSK DM-RS-like pilots (project-defined initialization, not the full TS 38.211 procedure) |
| `generate_channel.py` | AWGN, Rayleigh, Rician, and TDL channels |
| `compute_precoder.py` | Identity, DFT, and SVD/eigenbeamforming |
| `mimo_channel.py` | Effective channel and received-grid generation |
| `estimate_ls.py` | DM-RS-based LS effective-channel estimation |
| `equalize.py` | ZF and unbiased-MMSE layer detection |
| `metrics.py` | BER, BLER, EVM, NMSE, and capacity |
| `ofdm.py` | Unitary CP-OFDM transform pair |
| `gates.py` | Mandatory pre-campaign verification |
| `link_curve.py` | Shared Monte Carlo SNR-sweep engine |
| `write_config_log.py` | Configuration provenance log |
| `main.py` | Primary compact end-to-end campaign |
| `run_comparisons.py` | Theory, equalizer, capacity, and Massive MIMO comparisons |
| `cross_verify_csv.py` | Quantitative MATLAB/Python result comparison |

---

## 10. Executed verification record

The package was designed to record executed checks rather than rely only
on code inspection.

The established verification record includes:

- all four gates passing;
- AWGN BER agreement with closed-form theory;
- flat-Rayleigh QPSK agreement with the analytical reference;
- paired ZF/MMSE comparison;
- 64x8 Massive MIMO performance improvement relative to unprecoded 4x4;
- zero-error high-SNR SISO end-to-end recovery;
- MATLAB/Python CSV cross-verification under numerical tolerances.

A zero-error Monte Carlo point means that no error was observed in the
finite tested bit count. It does not prove that the true BER is
mathematically zero.

---

## 11. Result provenance

The primary campaign writes:

- numerical CSV results;
- a matching `_config.txt` log;
- result figures.

The configuration log records every dataclass field and the generation
timestamp. Together with the random seed and source code, these artifacts
allow each accepted result to be reproduced.

---

## 12. Scope statement

The current Python package provides a compact uncoded, CRC-protected,
PDSCH-oriented physical-layer reference chain and a valid single-user
64x8 Massive MIMO comparison with four-layer SVD eigenbeamforming.

The package does not currently implement the full report scope of:

- NR LDPC encoding and decoding;
- code-block segmentation;
- rate matching and rate recovery;
- soft LLR demodulation;
- HARQ;
- full standardized 3GPP TDL profiles;
- multi-user Massive MIMO scheduling and interference;
- practical CSI feedback or TDD reciprocity calibration.

These functions can be added as modular extensions without changing the
overall architecture.

## Massive MIMO 64x8 profile (executed and cross-verified)

The package now carries the same two-profile structure as the MATLAB
package. `config('massive')` (or `set_sim_profile('massive')`, which
writes the same persistent `sim_profile.txt` switch the MATLAB package
uses) returns the executed 64x8 four-layer eigen-beamforming
configuration on the flat Rayleigh channel: SNR grid -10 to 20 dB in
5 dB steps, 60 frames per point, seed 7, results written to
`results/python_results_massive.csv`. The chain modules needed no
changes; every function derives its dimensions from the configuration.

`run_massive_mimo.py` mirrors the MATLAB script step by step: gates
first, then the paired ideal-CSI comparison (unprecoded 4x4 against SVD
64x8 at equal total transmit power, generator re-seeded per branch), the
combined array and eigenbeamforming gain measured at the lowest BER operating point bracketed by
both curves (a fixed 1e-2 point is not always bracketed, since the 4x4
curve bottoms out near 4e-2 and the 64x8 curve becomes exactly
zero observed bit errors above 5 dB in the finite simulated bit count),
the comparison figure with the zero-observed-error region stated as an
annotation, and finally the end-to-end LS-estimated run
writing the massive CSV under the MATLAB column names with the antenna
and layer dimensions recorded per row.

Executed results (this environment, numpy, seed 7): combined array and
SVD eigenbeamforming gain 19.9 dB at BER 1e-1 (4x4 at 14.8 dB, 64x8 at
-5.2 dB; the comparison changes the transmit antennas, the receive
antennas, and the precoding together, so this is not a pure beamforming
gain); end-to-end BER 0.335 at -10 dB falling to 4.34e-5 at 10 dB, with
no bit errors observed in the finite evaluated bit count (1,657,440
payload bits per point) at 15 and 20 dB, so the true BER at those points
is bounded, not proven zero; EVM 84.1 to 3.44 percent; NMSE 1.35 to
1.34e-3; capacity 6.43 to 44.02 bit/s/Hz. The recorded MATLAB-Python cross-verification of the
massive profile (`cross_verify_csv.py` against the executed MATLAB
`matlab_results_massive.csv`, record stored as
`results/cross_verification_record_massive.txt`) passes all 35
comparisons, five metrics at seven SNR points, under the Table 8
tolerances; the tightest points are the 10 dB BER at 0.27 decades
(72 counted errors in Python against 134 in MATLAB, each over 1,657,440
evaluated payload bits, small-count Monte Carlo variance on independent
streams) and the 10 dB BLER at 0.183 absolute.

Quick smoke-test mode of the comparison campaigns (reduced frame
counts, outputs are not publication results; the default full run is the
publication configuration):

    python3 run_comparisons.py --quick

Unit tests (requires the development dependencies):

    python3 -m pip install -r requirements-dev.txt
    python3 -m pytest tests -v

Massive workflow, in order:

    python3 run_massive_mimo.py
    python3 cross_verify_csv.py results/matlab_results_massive.csv \
        results/python_results_massive.csv \
        results/cross_verification_record_massive.txt

The compact profile and all existing scripts are unchanged; `Config()`
with no arguments returns the identical compact configuration as before.
