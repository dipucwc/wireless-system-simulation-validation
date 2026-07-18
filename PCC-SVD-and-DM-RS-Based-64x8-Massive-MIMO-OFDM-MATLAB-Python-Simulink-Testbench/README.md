# Design, Modeling, and Verification of a PCC-SVD and DM-RS-Based 64 × 8 Massive MIMO-OFDM MATLAB–Python–Simulink Testbench

**Author:** Md Moklesur Rahman  
**Affiliation:** Independent Researcher, Finland  
**Technical area:** 5G NR-oriented physical-layer simulation, Massive MIMO-OFDM, SVD precoding, effective-channel estimation, MATLAB, Python, and Simulink  
**Manuscript status:** Prepared for journal submission.

---

## Overview

This repository contains a reproducible single-user **64 × 8 Massive MIMO-OFDM downlink testbench** with **four spatial layers**, **wideband/subband/per-subcarrier SVD precoding**, a **PDSCH-oriented DM-RS-like reference grid**, **least-squares effective-channel estimation**, **ZF and gain-corrected MMSE detection**, independent **MATLAB and Python implementations**, and an executable **Simulink Level-2 MATLAB S-function orchestration model**.

The central engineering problem is that a precoder that is locally optimal for singular-mode capture is not necessarily compatible with sparse reference-symbol-based channel estimation. Independent per-subcarrier SVD calculations can introduce arbitrary phase and unitary basis changes between neighboring subcarriers. These changes preserve the ideal singular values but can make the layer-domain effective channel

$$
\mathbf{G}[k]=\mathbf{H}[k]\mathbf{W}[k]
$$

artificially discontinuous across frequency. A receiver that samples this effective channel on a sparse DM-RS-like grid and interpolates between the sampled locations can then experience a large structural estimation error.

The project proposes a finite-candidate **Precoder–Channel-Estimator Compatibility-Constrained SVD (PCC-SVD)** method that selects among wideband, subband, and per-subcarrier precoder-update granularities while jointly considering:

1. the spatial energy captured by the selected SVD subspace; and  
2. the ability of the implemented DM-RS-based receiver to reconstruct the resulting effective channel.

---

## Research Objective

The objective is to design and verify a precoder-selection method that does not optimize only ideal transmitter-side singular-mode gain. Instead, the selected precoder must also produce an effective layer-domain channel that can be estimated from the available reference symbols and used reliably by the practical equalizer.

The implemented PCC-SVD candidate set is

$$
\mathcal{C}=\{144,48,1\},
$$

where the values denote the number of active subcarriers over which one covariance-SVD precoder is reused:

- **144:** one wideband precoder;
- **48:** three aligned subband precoders;
- **1:** aligned per-subcarrier precoding.

---

## Main Contributions

- A connected mathematical model from four-layer symbol mapping through digital precoding, frequency-selective MIMO propagation, effective-channel formation, DM-RS-like LS estimation, linear equalization, and metric accumulation.
- A finite-candidate PCC-SVD method that combines receiver-matched reconstruction error with dominant-subspace capture.
- Unitary Procrustes alignment of adjacent SVD bases without changing the represented dominant transmit subspace.
- Independent MATLAB and Python implementations using matched dimensions, pilot locations, SNR normalization, interpolation, equalizers, and metric definitions.
- An executable Simulink orchestration testbench linked to the verified MATLAB numerical engine.
- Adaptive Monte Carlo stopping with raw error counts, evaluated-bit counts, exact two-sided 95% Clopper–Pearson BER intervals, and zero-error upper bounds.
- Nine common numerical verification gates.
- Deterministic shared-vector MATLAB–Python replay support for convention-level regression testing.
- Threshold sensitivity, pilot-spacing sensitivity, robustness studies, bundle-selection statistics, and runtime analysis.

---

## PCC-SVD Formulation

### 1. Effective channel

For active subcarrier \(k\),

$$
\mathbf{x}[m,k]=\mathbf{W}[k]\mathbf{s}[m,k],
$$

$$
\mathbf{y}[m,k]=\mathbf{H}[k]\mathbf{W}[k]\mathbf{s}[m,k]+\mathbf{n}[m,k],
$$

and therefore

$$
\mathbf{G}[k]=\mathbf{H}[k]\mathbf{W}[k].
$$

For the implemented profile:

- \(\mathbf{H}[k]\in\mathbb{C}^{8\times64}\);
- \(\mathbf{W}[k]\in\mathbb{C}^{64\times4}\);
- \(\mathbf{G}[k]\in\mathbb{C}^{8\times4}\).

The receiver estimates \(\mathbf{G}[k]\), not the complete \(8\times64\) antenna-domain channel.

### 2. Pilot-reconstruction NMSE

For candidate bundle size \(b\), the exact noise-free effective channel is sampled at the same layer-specific pilot locations and reconstructed with the same interpolation operator used by the practical receiver:

$$
\mathbf{G}_{\mathrm{rec}}[k]
=
\mathcal{I}_{P}
\left\{
\mathcal{S}_{P}\{\mathbf{G}\}
\right\}[k].
$$

The structural reconstruction error is

$$
\eta_{\mathrm{PR}}(b)
=
\frac{
\sum_k
\left\|
\mathbf{G}[k]-\mathbf{G}_{\mathrm{rec}}[k]
\right\|_F^2
}{
\sum_k
\left\|
\mathbf{G}[k]
\right\|_F^2
}.
$$

This quantity isolates pilot-sampling and interpolation incompatibility without AWGN.

### 3. Dominant-subspace capture

The normalized captured spatial energy is

$$
\eta_{\mathrm{SC}}(b)
=
\frac{
\sum_k
\left\|
\mathbf{H}[k]\mathbf{W}_b[k]
\right\|_F^2
}{
\sum_k
\sum_{\ell=1}^{L}
\sigma_\ell^2\!\left(\mathbf{H}[k]\right)
}.
$$

A value close to one indicates that the candidate captures nearly all energy available from the strongest \(L=4\) singular modes.

### 4. SNR-adaptive reconstruction limit

The allowable deterministic reconstruction error is

$$
\varepsilon_{\mathrm{PR}}(\rho)
=
\min
\left(
\varepsilon_{\max},
\varepsilon_{\infty}
+
\frac{\beta_{\mathrm{PR}}}{\rho}
\right),
$$

with the implemented values

$$
\varepsilon_{\infty}=0.002,\qquad
\varepsilon_{\max}=0.008,\qquad
\beta_{\mathrm{PR}}=0.03.
$$

The constraint is relaxed at low SNR, where AWGN dominates, and tightened at high SNR, where structural interpolation error becomes visible.

### 5. Feasible set and selection rule

$$
\mathcal{F}(\rho)
=
\left\{
b\in\mathcal{C}:
\eta_{\mathrm{PR}}(b)
\le
\varepsilon_{\mathrm{PR}}(\rho)
\right\}.
$$

The selected bundle is

$$
b^\star(\rho)
=
\begin{cases}
\displaystyle
\arg\max_{b\in\mathcal{F}(\rho)}
\eta_{\mathrm{SC}}(b),
&
\mathcal{F}(\rho)\neq\varnothing,
\\[8pt]
\displaystyle
\arg\min_{b\in\mathcal{C}}
\eta_{\mathrm{PR}}(b),
&
\mathcal{F}(\rho)=\varnothing.
\end{cases}
$$

The second branch is a deterministic fallback that guarantees a valid output even when no candidate satisfies the configured reconstruction budget.

---

## End-to-End Processing Chain

![End-to-end PCC-SVD workflow](results/figures/figure_01_end_to_end_workflow.png)

1. Generate deterministic payload bits, DM-RS-like symbols, channel taps, and AWGN seeds.
2. Map normalized Gray-coded 16-QAM symbols to four spatial layers.
3. Generate the five-tap frequency-selective \(8\times64\) MIMO channel.
4. Construct covariance-SVD candidates with bundle sizes 144, 48, and 1.
5. Apply sequential unitary Procrustes alignment to adjacent candidate bases.
6. Compute \(\eta_{\mathrm{PR}}\), \(\eta_{\mathrm{SC}}\), and \(\varepsilon_{\mathrm{PR}}(\rho)\).
7. Execute PCC-SVD feasible-set selection or the minimum-reconstruction-error fallback.
8. Apply the selected precoder to payload and DM-RS-like symbols.
9. Form the exact effective channel \(\mathbf{G}[k]\).
10. Generate ideal-CSI or LS-estimated-CSI receiver branches.
11. Apply ZF or gain-corrected MMSE equalization.
12. Hard-demap the recovered 16-QAM payload symbols.
13. Accumulate BER, EVM, NMSE, capacity, compatibility metrics, raw counts, confidence intervals, selected bundle, and runtime.
14. Save CSV files, figures, verification records, configuration, logs, and reproducibility metadata.

---

## Main Simulation Configuration

| Parameter | Value |
|---|---:|
| Transmit antennas | 64 |
| Receive antennas | 8 |
| Spatial layers | 4 |
| FFT reference size | 256 |
| Active subcarriers | 144 |
| OFDM symbols | 14 |
| Subcarrier spacing | 30 kHz |
| Modulation | Normalized Gray-coded 16-QAM |
| DM-RS-like OFDM symbols | MATLAB: 4 and 11; Python: indices 3 and 10 |
| Layer-separated pilot comb spacing | 4 |
| Pilot subcarriers per layer | 36 |
| TDL sample delays | \([0,2,5,9,14]\) |
| Relative tap powers | \([0,-2.2,-4,-6,-8.2]\) dB |
| SNR points | \(-5,0,5,10,15,20\) dB |
| PCC candidate bundle sizes | \(144,48,1\) |
| Primary detector | Gain-corrected MMSE |
| Additional detector | ZF |
| Monte Carlo stopping | At least \(10^6\) bits and 200 errors, or 40 frames |
| BER confidence interval | Exact two-sided 95% Clopper–Pearson |

---

## Evaluated Precoding and CSI Scenarios

The principal controlled comparisons include:

- wideband covariance-SVD with LS-estimated CSI;
- fixed aligned 48-subcarrier SVD with LS-estimated CSI;
- aligned per-subcarrier SVD with LS-estimated CSI;
- unaligned per-subcarrier SVD with LS-estimated CSI;
- adaptive PCC-SVD with LS-estimated CSI;
- per-subcarrier SVD with ideal effective-channel knowledge;
- identity precoding with ZF and MMSE detection;
- adaptive-limit sensitivity;
- valid layer-separated DM-RS-like pilot-spacing sensitivity;
- extended-delay, two-layer, and \(32\times8\) robustness cases;
- phase-only alignment versus unitary Procrustes alignment.

The same receiver processing and metric functions are used across precoder comparisons so that differences can be traced to the intended design variable.

---

## Performance Metrics

| Metric | Purpose |
|---|---|
| BER | Hard-decision payload bit error rate |
| BER confidence interval | Exact statistical uncertainty from raw error and bit counts |
| RMS EVM | Equalized-symbol distortion before hard decisions |
| Effective-channel NMSE | Error between \(\widehat{\mathbf{G}}[k]\) and \(\mathbf{G}[k]\) |
| Exact effective-channel capacity | Information-theoretic spatial-gain reference |
| \(\eta_{\mathrm{PR}}\) | Noise-free pilot-reconstruction incompatibility |
| \(\eta_{\mathrm{SC}}\) | Normalized dominant-subspace capture |
| \(\eta_{\mathrm{EC}}\) | End-to-end equalizer consistency error |
| Selected bundle | Frame-level PCC-SVD update granularity |
| Runtime | Candidate-construction and selection complexity |

No single metric is sufficient. Capacity and \(\eta_{\mathrm{SC}}\) characterize the spatial opportunity created by the precoder, while BER, EVM, receiver NMSE, and \(\eta_{\mathrm{EC}}\) show how much of that opportunity survives practical estimation and detection.

---

## MATLAB–Python Algorithm Mapping

| Technical operation | MATLAB stage | Python stage |
|---|---|---|
| Layer mapping and digital precoding | `buildFrame`, `applyPrecoder` | `build_frame`, `apply_precoder` |
| TDL physical channel | `generateTdlChannel` | `generate_tdl_channel` |
| Covariance-SVD candidate bank | `covarianceSvdCandidate` | `covariance_svd_candidate` |
| Unitary Procrustes alignment | `alignSvdBases` | `align_svd_bases` |
| PCC-SVD selection | `precoderChannelEstimatorCompatibilitySvd` | `precoder_channel_estimator_compatibility_svd` |
| Effective-channel formation | `trueEffectiveChannel` | `true_effective_channel` |
| DM-RS LS estimation | `estimateEffectiveChannelLs` | `estimate_effective_channel_ls` |
| ZF / gain-corrected MMSE | `equalizeLayers` | `equalize_layers` |
| Metrics and confidence reporting | Metric functions and CSV writers | Metric functions and adaptive publication driver |

MATLAB and NumPy use different random-number generators. Therefore, independent Monte Carlo campaigns are compared through matched equations, dimensions, configuration, metric scale, and scenario ordering rather than by claiming sample-identical random output. Deterministic shared vectors are used for stage-by-stage numerical convention checks.

---

## Simulink Testbench

![Executable Simulink orchestration testbench](results/figures/figure_02_simulink_testbench.png)

The Simulink model is an executable orchestration testbench. A run-trigger signal activates a Level-2 MATLAB S-function that calls the verified MATLAB numerical engine.

The output vector reports:

- execution status;
- passed verification-check count;
- total verification-check count;
- BER;
- effective-channel NMSE;
- selected candidate bundle.

The model supports:

- **testbench mode:** nine verification checks plus a finite 10 dB PCC-SVD smoke test;
- **analysis mode:** the complete MATLAB campaign and final PCC-SVD outputs.

The lower workflow region documents the end-to-end sequence. It is not presented as a separate native block-by-block implementation of every \(64\times8\) matrix operation.

---

## Numerical Verification

All nine common MATLAB and Python verification gates passed.

| Verification gate | Technical purpose | Status |
|---|---|---|
| Wideband semi-unitarity | Verify \(\mathbf{W}^{H}\mathbf{W}=\mathbf{I}_{L}\) and equal transmit-power normalization | PASS |
| Aligned per-subcarrier semi-unitarity | Verify basis alignment preserves column orthonormality | PASS |
| PCC-SVD semi-unitarity | Verify the selected frequency-indexed precoder remains semi-unitary | PASS |
| Effective-channel dimensions | Verify the expected \(K\times N_R\times L\) representation | PASS |
| 16-QAM round trip | Verify mapper/demapper bit and symbol ordering | PASS |
| Noiseless flat-channel LS | Verify exact pilot division and interpolation recovery | PASS |
| Noiseless end-to-end recovery | Verify mapping, equalization, ordering, and demapping | PASS |
| Alignment reduces pilot error | Verify Procrustes alignment reduces \(\eta_{\mathrm{PR}}\) | PASS |
| PCC constraint or fallback | Verify feasible selection or deterministic fallback | PASS |

Passing these gates does not prove global optimality, but failure of any gate invalidates the corresponding numerical campaign.

---

## Principal Publication Results

### Estimated-CSI BER

![Estimated-CSI BER](results/figures/figure_03_estimated_csi_ber.png)

At **10 dB**:

| Scenario | BER | Effective-channel NMSE | Exact capacity (bit/s/Hz) |
|---|---:|---:|---:|
| Wideband SVD, LS | \(7.069\times10^{-3}\) | \(2.742\times10^{-2}\) | 26.104 |
| Fixed 48-subcarrier SVD, LS | \(3.448\times10^{-3}\) | \(2.442\times10^{-2}\) | 28.082 |
| Aligned per-subcarrier SVD, LS | \(4.145\times10^{-4}\) | \(1.962\times10^{-2}\) | 30.716 |
| Unaligned per-subcarrier SVD, LS | \(1.991\times10^{-1}\) | \(6.651\times10^{-1}\) | 30.714 |
| PCC-SVD, LS | \(2.978\times10^{-3}\) | \(2.325\times10^{-2}\) | 28.491 |
| Per-subcarrier SVD, ideal CSI | \(<3.336\times10^{-6}\) upper bound | 0 | 30.712 |

The unaligned per-subcarrier method preserves ideal singular-mode energy but produces a severe estimated-CSI BER floor. Procrustes alignment removes most of the artificial basis discontinuity. PCC-SVD provides a bounded, receiver-aware compromise between wideband smoothness and frequency-local spatial gain.

At **20 dB**, PCC-SVD selects the 144-subcarrier wideband candidate and records:

- 6 observed bit errors;
- 1,105,920 evaluated bits;
- BER \(=5.425\times10^{-6}\);
- exact 95% confidence interval approximately  
  \([1.991\times10^{-6},\,1.181\times10^{-5}]\).

### Capacity and estimability trade-off

![Capacity trade-off](results/figures/figure_07_capacity_tradeoff.png)

Per-subcarrier SVD provides the largest exact effective-channel capacity, but this ideal spatial gain is not automatically realizable with sparse LS channel estimation. Wideband SVD is easier to reconstruct but sacrifices frequency-local gain. PCC-SVD makes the trade-off explicit through \(\eta_{\mathrm{PR}}\) and \(\eta_{\mathrm{SC}}\).

### Adaptive bundle behavior

![Mean selected bundle](results/figures/figure_08_mean_selected_bundle.png)

At 10 dB, the mean PCC-SVD bundle is **44.35 subcarriers**:

- 8.11% of realizations select bundle 144;
- 67.57% select bundle 48;
- 24.32% select bundle 1.

At 15 dB and 20 dB, all evaluated realizations select the 144-subcarrier candidate because the reconstruction-error limit becomes stricter than the residual mismatch of the finer-granularity candidates.

![Bundle selection probability](results/figures/figure_09_bundle_selection_probability.png)

### Runtime

![Measured Python runtime](results/figures/figure_12_runtime_complexity.png)

| Mode | Mean runtime per channel realization |
|---|---:|
| Wideband SVD | 12.58 ms |
| Aligned 48-subcarrier SVD | 17.16 ms |
| Aligned per-subcarrier SVD | 72.44 ms |
| PCC-SVD candidate selection | 110.32 ms |

PCC-SVD has bounded and reproducible complexity but is more expensive than constructing one fixed candidate because it evaluates all three candidates and their compatibility metrics.

---

## Repository Structure

```text
Design, Modeling, and Verification of a PCC-SVD and DM-RS-Based 64 x 8 Massive MIMO-OFDM MATLAB-Python-Simulink Testbench/
├── README.md
├── GITHUB_UPLOAD_INSTRUCTIONS.md
├── MANIFEST_SHA256.txt
├── paper/
│   └── Project_2_Journal_Manuscript_Final(1).docx
├── matlab/
│   └── README.md
├── python/
│   └── README.md
├── simulink/
│   └── README.md
├── results/
│   ├── csv/
│   └── figures/
│       ├── figure_01_end_to_end_workflow.png
│       ├── figure_02_simulink_testbench.png
│       ├── figure_03_estimated_csi_ber.png
│       ├── ...
│       └── figure_13_zf_mmse_comparison.png
├── verification/
│   └── README.md
├── shared_vectors/
│   └── README.md
└── reproducibility/
    └── README.md
```

The final publication archive should place the unchanged MATLAB, Python, and Simulink package files in the corresponding directories. Preserve the original filenames because imports, function resolution, model callbacks, and execution scripts may depend on them.

---

## Reproduction Workflow

### Clone the repository

```bash
git clone https://github.com/dipucwc/Wireless-System-Simulation-Validation.git

cd "Wireless-System-Simulation-Validation/Design, Modeling, and Verification of a PCC-SVD and DM-RS-Based 64 x 8 Massive MIMO-OFDM MATLAB-Python-Simulink Testbench"
```

### Python

1. Enter the `python/` directory.
2. Create and activate a virtual environment.
3. Install the dependencies used by the supplied publication package.
4. Run the package's verification testbench.
5. Run the main publication campaign.
6. Run the supplementary robustness and baseline-comparison scripts when required.

Two explicitly documented supplementary entry points are:

```bash
python run_robustness_analysis.py
python run_baseline_comparison.py
```

The publication campaign should export raw counts, BER confidence intervals, metrics, candidate-selection statistics, runtime values, configuration, logs, and figures.

### MATLAB

1. Open the `matlab/` directory in MATLAB.
2. Add the package and its subdirectories to the MATLAB path.
3. Run the numerical verification testbench.
4. Confirm that all nine verification gates pass.
5. Run the complete SNR campaign.
6. Confirm that CSV outputs, logs, configuration, and figures are generated.

Example setup command:

```matlab
addpath(genpath(pwd));
```

Use the exact main-script and function filenames supplied in the final MATLAB package.

### Simulink

1. Open the `simulink/` directory.
2. Add both the MATLAB numerical-engine directory and the Simulink directory to the MATLAB path.
3. Run the supplied model-builder script if the `.slx` file has not already been generated.
4. Open the generated model.
5. Execute testbench mode and verify that the status and passed-check outputs are valid.
6. Execute analysis mode for the complete MATLAB campaign when required.

The Simulink testbench requires a compatible licensed MATLAB and Simulink installation.

### Deterministic MATLAB–Python replay

1. Generate or load the deterministic shared-vector MAT file and JSON metadata.
2. Run the Python reference-vector generation or verification stage.
3. Load the same arrays in MATLAB without regenerating random samples.
4. Compare dimensions, precoders, effective channels, reconstructed channels, compatibility metrics, LS estimates, equalized symbols, detected bits, and final metrics.
5. Record maximum absolute and relative deviations.
6. Declare PASS only when every configured tolerance is satisfied.

---

## Reproducibility Rules

- Preserve the exact SNR definition: total average four-layer transmit energy divided by complex AWGN variance.
- Apply the same semi-unitary power normalization to every compared precoder.
- Use the same pilot locations, interpolation convention, detector, and metric definitions across compared scenarios.
- Retain raw error counts and evaluated-bit counts.
- Report exact confidence intervals rather than only point BER estimates.
- Treat zero observed errors as an upper-bound result, not proof of zero BER.
- Keep deterministic configuration, seed formulas, result CSV files, figures, logs, software metadata, and file hashes with the campaign.
- Do not claim sample-identical independent MATLAB and Python Monte Carlo output when different random-number generators are used.
- Use deterministic shared vectors for numerical convention and regression checks.

---

## Scope and Limitations

The current study assumes:

- perfect instantaneous transmitter-side channel knowledge for candidate construction;
- single-user transmission;
- four uncoded spatial layers;
- ideal timing and carrier synchronization;
- a controlled five-tap frequency-selective channel;
- no practical CSI feedback, quantization, delay, or reciprocity mismatch;
- no phase noise, nonlinear PA behavior, IQ imbalance, or other RF impairments;
- a layer-separated DM-RS-like reference grid rather than a bit-exact implementation of every 3GPP TS 38.211 PDSCH DM-RS option;
- exact effective-channel capacity as an information-theoretic reference, not coded throughput;
- a Simulink orchestration testbench linked to the MATLAB engine rather than a complete native block implementation of every matrix operation.

PCC-SVD is a finite-candidate engineering method. It does not claim globally minimum BER or universal optimality. Its adaptive threshold must be retuned or learned when the channel model, delay spread, pilot pattern, modulation, layer count, detector, or antenna configuration changes.

---

## Future Work

Planned extensions include:

- complete standardized NR PDSCH DM-RS mapping;
- practical transmitter CSI acquisition, quantization, feedback delay, and reciprocity error;
- coded BLER and throughput evaluation;
- multiple antenna and spatial-layer profiles;
- mobility and Doppler;
- spatial correlation and broader 3GPP channel profiles;
- RF impairments and synchronization errors;
- comparison with continuous channel-estimation-considerate and cross-subcarrier optimization methods;
- analytically derived or data-driven adaptive reconstruction limits;
- real-time or hardware-oriented implementation studies.

---

## Manuscript

The journal manuscript is located in:

```text
paper/Project_2_Journal_Manuscript_Final(1).docx
```

Paper title:

> **Performance Analysis of Wideband SVD Precoding and DM-RS-Based Effective-Channel Estimation in 64 × 8 Massive MIMO-OFDM Systems**

The manuscript contains the complete mathematical derivation, seven algorithms, implementation mapping, verification methodology, statistical procedure, simulation results, limitations, and references.

---
---

## Contact

**Md Moklesur Rahman**  
Independent Researcher, Finland  
Email: `moklesur.eee@gmail.com`  
GitHub: `dipucwc`
