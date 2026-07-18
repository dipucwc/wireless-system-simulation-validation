# Cross-Verified and Statistically Qualified 5G NR Massive MIMO-OFDM Simulator

## Project Status

**Research manuscript under preparation for publication.**

The complete source code, simulation configurations, statistical records,
cross-verification results, and manuscript will be released after the
publication process permits public distribution.

## Overview

This project develops a publication-oriented verification and statistical
qualification framework for a PDSCH-oriented 5G NR Massive MIMO-OFDM
physical-layer link-level simulator.

The work extends the core simulator with:

- verification gates executed before performance campaigns;
- independent MATLAB and Python implementation comparison;
- Simulink testbench validation;
- adaptive Monte Carlo stopping;
- confidence-interval-based BER and BLER reporting;
- reproducible configuration, seed, and result tracking;
- systematic investigation of cross-implementation deviations.

## Technical Scope

The complete research platform includes:

- 5G NR PDSCH-oriented processing;
- Massive MIMO-OFDM transmission;
- wideband eigenbeamforming;
- DM-RS-based effective-channel estimation;
- ZF and unbiased-MMSE equalization;
- CRC-protected uncoded evaluation;
- TS 38.212 LDPC-coded evaluation;
- BER, BLER, EVM, NMSE, throughput, and capacity metrics;
- MATLAB, Python, and Simulink execution environments.

## Relationship to the Core Simulator

This study is an advanced verification and statistical-qualification
extension of the main PDSCH-oriented 5G NR Massive MIMO-OFDM link-level
simulator.

The focus of this project is not only simulator implementation, but also
the evidence required to establish that the reported numerical results are
correct, reproducible, and statistically supported.

## Planned Public Release

After the publication process, this repository is planned to include:

- MATLAB implementation;
- independent Python reference implementation;
- Simulink testbench;
- verification-gate scripts;
- simulation configuration files;
- fixed random seeds;
- raw error counts and CSV results;
- confidence-interval calculations;
- MATLAB-Python comparison records;
- MATLAB-Simulink validation records;
- plotting scripts;
- accepted manuscript citation.

## Current Availability

The complete research package is currently maintained privately.

## License

See the repository [LICENSE](LICENSE) file for the applicable software and documentation terms.

---

## Author

**Md Moklesur Rahman**  
Independent Researcher, Finland  
GitHub: [dipucwc](https://github.com/dipucwc)  
Email: moklesur.eee@gmail.com

---


