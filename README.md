# Resolving Phenomenological Degeneracies in High-Energy Astrophysical Transients

A multi-modal deep learning classification framework using multi-input Physics-Informed Neural Networks (PINNs) to isolate signatures of Axion-Like Particles (ALPs) and Lorentz Invariance Violation (LIV) from conventional relativistic shock emissions.

## 👥 Research Team
* **AI & Neural Network Architecture:** Yugansh Bijalwan
* **Physics & Simulation Engine:** Ronel Jonathan

## 📁 Repository Structure

```text
multi-modal-grb-classifier/
│
├── simulations/      # Mock data and physics simulation outputs
├── preprocessing/    # Scaling, normalization, tensor conversion
├── models/           # Neural network architectures and branches
├── training/         # Training loops, losses, and optimization
├── evaluation/       # Metrics, confusion matrices, and visualizations
├── experiments/      # Sandbox files for learning and testing concepts
├── docs/             # Notes, architecture decisions, and documentation
│
├── main.py           # Main project entry point
├── requirements.txt  # Python package dependencies
├── README.md         # Project overview and setup instructions
└── .gitignore        # Files and folders excluded from Git
```

### Overview
This repository is organized to support the development of a multi-modal deep learning framework for classifying competing high-energy astrophysical signatures. The structure separates simulation outputs, preprocessing, model development, training, evaluation, and experimental testing, allowing parallel development of the physics simulation engine and AI classification pipeline.

## Computational Roadmap

- **Physics Engine:** 1D transport pipeline for ALP, LIV, and shock simulation.

- **Classification Engine:** Multi-headed neural network for degeneracy resolution.

- **Status:** In active development (Phase 1: Theoretical Standardization).
