# A Physics-Guided Multimodal Inference Framework for GRB 221009A

This repository contains the computational architecture, physics simulation engine, and multimodal inference pipeline for evaluating observational degeneracy and determining data sufficiency requirements in high-energy astrophysical transients.

## Problem Statement

High-energy astrophysical transients such as GRB 221009A involve extreme physical environments where standard relativistic shock processes and exotic physics (e.g., Axion-Like Particle conversion and Lorentz Invariance Violation) predict overlapping observational signatures. When restricted to single-modality or signal-limited datasets, model parameter estimation becomes severely ill-posed. Currently, astrophysics lacks a standardized, quantitative framework to define the minimum observational data (spectral, temporal, and polarimetric) strictly required to break these physical degeneracies.

## Research Question

To what extent can a physics-guided multimodal inference framework quantify the information content of joint spectral, temporal, and polarimetric observations, establish the minimum observational sufficiency required to distinguish competing physical mechanisms in GRB 221009A-like events, and map their observational degeneracy landscapes?

## Methodological Framework

1. **Forward Simulation Engine:** Simulates synthetic broken power-law synchrotron spectra, LIV energy-dependent dispersion delays, and ALP photon-axion conversion profiles with energy-dependent polarization dynamics.
2. **Information-Theoretic Pipeline:** Computes Fisher Information Matrices, Cramér-Rao lower bounds, and Shannon Information Gain across multi-channel observation tensors.
3. **Multimodal Inference Engine:** Evaluates machine learning classification boundaries across varied Signal-to-Noise Ratios (SNR) to map degenerate regions in physical parameter space.

## Project Evolution Note

This project evolved from a direct empirical parameter-fitting task on GRB 221009A into an information-theoretic framework. Preliminary analysis showed that fitting parameters on under-determined data leads to ill-posed inverse problems where competing models yield identical signatures within noise limits. Evolving the scope to quantify observational sufficiency provides a generalizable diagnostic framework for current and future observational campaigns.

## Repository Architecture

```text
grb-degeneracy-framework/
├── data/
│   ├── raw/
│   └── synthetic/
├── src/
│   ├── physics/
│   │   ├── synchrotron.py
│   │   ├── liv.py
│   │   └── alp.py
│   ├── ml/
│   │   ├── dataset.py
│   │   ├── models.py
│   │   └── metrics.py
│   └── utils/
│       ├── fisher.py
│       └── visualizer.py
├── tests/
├── paper/
├── requirements.txt
└── main.py

```
## Members
- Core Physics Simulation Engine: Ronel Jonathan
- Multi-modal Inference Engine: Yugansh Bijalwan
