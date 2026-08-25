# Information-Theoretic Constraints on Exotic Physics via Multimodal Neural Inference on GRB 221009A

This repository contains the physics simulation engine, information-theoretic diagnostic framework, and deep neural posterior estimation pipeline for breaking observational degeneracies between standard relativistic shock processes and exotic physics in GRB 221009A.

## Problem Statement

High-energy astrophysical transients such as GRB 221009A involve extreme physical environments where standard relativistic shock processes and exotic physics (e.g., Axion-Like Particle conversion and Lorentz Invariance Violation) predict overlapping observational signatures. When restricted to single-modality or signal-limited datasets, model parameter estimation becomes severely ill-posed. Currently, astrophysics lacks a standardized, quantitative framework to define the minimum observational data (spectral, temporal, and polarimetric) strictly required to break these physical degeneracies.

## Research Question

To what extent can a physics-guided multimodal inference framework quantify the information content of joint spectral, temporal, and polarimetric observations, establish the minimum observational sufficiency required to distinguish competing physical mechanisms in GRB 221009A-like events, and map their observational degeneracy landscapes?

## Methodological Framework

1. **Forward Physics Simulation Engine:** Simulates synthetic broken power-law synchrotron prompt spectra (Fermi-GBM), FRED light curves (Swift-XRT), LIV dispersion delays ($$E_\text{QG}$$), and ALP photon-axion conversion probabilities ($$g_{a\gamma}$$) with energy-dependent polarization dynamics (IXPE).
2. **Information-Theoretic Diagnostics:** Computes exact Fisher Information Matrices ($$\mathcal{F}$$), Cramér-Rao Lower Bounds (CRLB), and differential entropy reduction ($$\Delta I$$) across single- and multi-channel observation tensors.
3. **Amortized Neural Posterior Estimation:** Trains a `MultimodalGRBEncoder` coupled with a Cholesky-factorized `DegeneracyMDNHead` (Mixture Density Network) to reconstruct non-Gaussian posterior probability distributions and map 2D degeneracy topographies.

## Key Evaluation Metrics

* **Differential Shannon Information Gain ($\Delta I$):** Quantifies posterior entropy reduction (in nats) achieved when combining multi-instrument telemetry relative to single-modality baselines.
* **Posterior Variance Trace Reduction ($$\text{Tr}(\Sigma)$$):** Measures joint parameter constraint tightening across single-channel (`[1,0,0]`), dual-channel (`[1,1,0]`), and full polarimetric (`[1,1,1]`) input masks.
* **Degeneracy Topography Density:** Applies 2D Gaussian Kernel Density Estimation (KDE) to isolate multi-modal parameter modes in physical parameter space ($E_\text{break}$, $E_\text{QG}$, $g_{a\gamma}$).

## Project Evolution Note

This project evolved from a direct empirical parameter-fitting task on GRB 221009A into an information-theoretic framework. Preliminary analysis showed that fitting parameters on under-determined data leads to ill-posed inverse problems where competing models yield identical signatures within noise limits. Evolving the scope to quantify observational sufficiency provides a generalizable diagnostic framework for current and future observational campaigns.

## Repository Architecture

```text
grb-degeneracy-framework/
├── checkpoints/
│   └── grb_model_best.pt
├── data/                    # The data/ directory contents are excluded from version control due to file size constraints   │   │                           and are dynamically generated via the local execution pipeline.
│   ├── processed/
│   │   ├── target_event.h5
│   ├── raw/
│   │   ├── fermi_gbm_tte.fits
│   │   ├── lhaaso_flux.csv
│   │   ├── swift_xrt_lc.txt
│   └── synthetic/
│   │   ├── test_datasets/  
│   │   └── events_suite.h5    
├── reports/
│   ├── figures/                    # Exported degeneracy contour maps (.png)
│   ├── population_eval_data.npz
│   └── population_metrics.json
├── scripts/
│   ├── 00_preprocess_raw.py
│   ├── 01_evaluate_population.py
│   ├── 02_map_landscape.py
│   ├── 03_ablation_study.py
│   ├── 04_infer_main_event.py
│   └── train.py
├── src/
│   ├── ml/
│   │   ├── baseline.py     
│   │   ├── dataset.py  
│   │   └── models.py           
│   ├── physics/
│   │   ├── alp.py                
│   │   ├── constants.py            
│   │   ├── liv.py               
│   │   ├── main.py            
│   │   ├── sampler.py              
│   │   ├── synchrotron.py     
│   │   ├── test_physics.py
│   │   └── verify_pipeline.py
│   └── utils/
│   │   ├── fisher.py               
│   │   ├── pack_hdf5.py         
│   │   ├── patch_hdf5_files.py
│   │   ├── patch_dataset.py
├── tests/
│   ├── test_hdf5_files.py
│   ├── test_physics.py
│   ├── validate_dataset.py
│   └── verify_pipeline.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

# Environment Setup & End-to-End Execution Pipeline

## Environment Setup

Before running any computational or data processing scripts, the environment dependencies must be installed. To install all required third-party libraries, execute the following command in your terminal at the root directory:

```bash
pip install -r requirements.txt
```
## End-to-End Execution Pipeline
### Core Physics Verification & LHS Sampling Audit
Modules: src/physics/test_physics.py & src/physics/verify_pipeline.py

Before generating synthetic event suites, execute unit tests and verification checks to validate the physical engines and Latin Hypercube Sampling (LHS) setup:

*test_physics.py*: Confirms that the synchrotron emission functions complete without generating NaN or infinite values.

*verify_pipeline.py*: Runs sanity checks across parameter bounds, verifies the correlation structure of the LHS parameter sampler, confirms mode-dependent array outputs, and tests the numerical stability of the Jacobian matrices.

Execution Command:
```bash
python -m src.physics.test_physics
python -m src.physics.verify_pipeline
```
### Synthetic Event Generation
Module: src/physics/main.py

This script handles the generation of 30,000 synthetic GRB events across three physical scenario modes ("ssc", "alp", "liv") using Latin Hypercube Sampling across a 10-dimensional parameter space.

It dynamically computes emission spectra, polarization arrays, and extinction corrections, while determining Fisher Information matrices and Cramer-Rao lower bounds for each event. The generated output is split into batch subfolders containing individual compressed archive files (.npz) saved inside *data/synthetic/test_datasets*.

Execution Command:
```bash
python -m src.physics.main
```
### Synthetic Data Repair and Schema Validation
Modules: src/utils/repair_dataset.py & src/utils/validate_dataset.py

Once all raw .npz event files are generated, they undergo structural auditing and memory-safe schema checking before being packaged:

*repair_dataset.py*: Scans all generated simulation files, replaces invalid numerical values (such as NaNs or Infs) in observation vectors or Fisher matrices, and recomputes stable inverse covariance matrices.

*validate_dataset.py*: Performs a memory-safe read check on every generated .npz archive file to confirm data integrity and verify that all key payloads can be decompressed without system memory overflows or file corruption.

Execution Command:
```bash
python -m src.utils.repair_dataset
python -m src.utils.validate_dataset
```

### Dataset Packaging into HDF5 Archive
Module: src/utils/pack_hdf5.py

This script aggregates all validated individual .npz simulation files across the batch subfolders and packages them into a single consolidated, compressed HDF5 binary file located at data/synthetic/events_suite.h5.

During packaging, it constructs dataset tensors, converts physical scenario labels into numerical class indices, records global parameter normalization statistics (mean and standard deviation), and saves them into the attributes of the HDF5 file.

Execution Command:
```bash
python -m src.utils.pack_hdf5
```

### HDF5 Structure Verification and Target Patching
Modules: src/utils/check_hdf5.py & src/utils/patch_hdf5.py

After compiling the HDF5 archive, verify its integrity and adjust dataset targets as required by downstream models:

*check_hdf5.py*: Inspects key entries, array dimensions, and tensor shapes within data/synthetic/events_suite.h5.

*patch_hdf5.py*: Applies continuous parameter target adjustments or label re-indexing required for specific neural network training routines.

Execution Command:
```bash
python -m src.utils.check_hdf5
python -m src.utils.patch_hdf5
```

### Neural Architecture Optimization & Model Training
Module: scripts/train.py

With the HDF5 archive formatted and verified, model training can be performed either locally via scripts/train.py.

This phase loads the events_suite.h5 tensor dataset, feeds multi-channel spectral-polarimetric observations through the MultimodalGRBEncoder, and optimizes the DegeneracyMDNHead using Cholesky-factorized negative log-likelihood (NLL) loss over 150 epochs.

Execution Command:
```bash
python scripts/train.py
```

### Amortized Population Evaluation
Module: scripts/01_evaluate_population.py

This script loads the trained model checkpoint saved from the training phase and evaluates performance across held-out test events. It calculates population metrics, such as Kullback-Leibler (KL) divergence and differential entropy reduction across estimated parameter spaces, and writes the summary metrics into _reports/population_metrics.json_ and _reports/population_eval_data.npz_.

Execution Command:
```bash
python scripts/01_evaluate_population.py
```

### Degeneracy Topography Mapping
Module: scripts/02_map_landscape.py

This script processes the saved population evaluation predictions, projects multidimensional parameter estimates onto 2D grids, and applies Gaussian kernel density estimations. It produces density contour visual maps that show regions where parameter degeneracies occur.

Execution Command:
```bash
python scripts/02_map_landscape.py
```

### Modality Sensitivity & Ablation Analysis
Module: scripts/03_ablation_study.py

This script measures how individual observational modalities contribute to resolving physics parameter degeneracies. By systematically masking input data channels (spectral, temporal, and polarimetric), it calculates the posterior variance trace reduction across single-channel and multi-instrument modes.

Execution Command:
```bash
python scripts/03_ablation_study.py
```

### Targeted Empirical Inference on GRB 221009A
Module: scripts/04_infer_main_event.py

The final step executes amortized neural inference using real telemetry from GRB 221009A. The script feeds empirical observations into the trained neural network to generate posterior covariance confidence contours, producing diagnostic plots comparing single-detector constraints against multi-instrument joint inference.

Execution Command:
```bash
python scripts/04_infer_main_event.py
```

## Citation & Registration

If using this code repository or computational pipeline for academic research, please cite:

```bibtex
@article{jonathan2026grb221009a,
  title={Information-Theoretic Constraints on Exotic Physics via Multimodal Neural Inference on GRB 221009A},
  author={Jonathan, Ronel},
  journal={S.T. Yau High School Science Award (Asia)},
  year={2026},
  note={Registration ID: Phy-218}
}
```

## License

Distributed under the MIT License. See _LICENSE_ for details.

### Members
Ronel Jonathan

### Supervising Teacher
Sangeeta Saini

### School & Principal
**GEMS New Millennium School**
Fatima Martin
