# Latent Space Learning for PDE Systems with Complex Boundary

## Overview

This repository contains the official PyTorch implementation for the NeurIPS 2025 submission:
**"Latent Space Learning for PDE Systems with Complex Boundary"**.




## Key Features

* Implementation of **BAROM (Boundary-Aware Attention ROM)**, including two variants:
    * `BAROM_ImpBC`: Implicit Boundary Condition handling.
    * `BAROM_ExpBC`: Explicit Boundary Condition handling in the latent dynamics.
* Explicit boundary treatment via a learnable lifting network ($\mathcal{L}_{\text{lift}}$).
* Non-intrusive, attention-based mechanism ($\mathcal{F}_{\text{latent}}$) for learning internal field dynamics.
* Handles two categories of complex boundary conditions:
    1.  Externally prescribed complex BCs (Category 1).
    2.  Complex BCs with internal-boundary coupling and external controls (Category 2).
* Code for reproducing experiments on benchmark PDE systems from the paper.
* Implementations of baseline models for comprehensive comparison: FNO, SPFNO, BENO, LNS-AE, LNO, POD-DL-ROM, DON, OPNO.

## Repository Structure
```text
BAROM/
├── README.md                     # This file
├── PDE_category1/                # Experiments for PDEs with externally prescribed BCs (No Internal Feedback)
│   ├── PC1_datageneration/       # Scripts to generate datasets for Category 1 PDEs
│   │   ├── generate_datasets_PDE_task1.py
│   │   ├── data_visualization.ipynb
│   │   └── command.md            # Instructions for data generation
│   ├── datasets_full/            # Information about Category 1 datasets (dataset.md)
│   ├── BAROM_ImpBC.py            # BAROM model for Category 1
│   ├── FNO.py
│   ├── SPFNO.py
│   ├── BENO.py
│   ├── LNS_AE.py
│   ├── LNO.py
│   ├── POD_DL_ROM.py
│   ├── DON.py
│   ├── OPNO.py
│   └── Benchmarking.py           # Script to run benchmarks for Category 1
│   └── command_line.md           # Instructions for running experiments
├── PDE_category2/                # Experiments for PDEs with internal-boundary coupling & control
│   ├── generate_datasets_PDE_task2.py # Script to generate datasets for Category 2 PDEs
│   ├── datasets_new_feedback/    # Information about Category 2 datasets (dataset.md)
│   ├── data_visualization.ipynb
│   ├── datasetscript.md          # Details on dataset generation
│   ├── BAROM_ExpBC.py            # BAROM model (explicit BC handling) for Category 2
│   ├── BAROM_ImpBC.py            # BAROM model (implicit BC handling) for Category 2
│   ├── FNO.py
│   ├── SPFNO.py             
│   ├── BENO.py
│   ├── LNS_AE.py
│   ├── LNO.py
│   ├── POD_DL_ROM.py
│   ├── BenchmarkingEBCBAROM-CDF.py # Benchmarking scripts for specific datasets (Convdiff)
│   ├── BenchmarkingEBCBAROM-hdl.py # (Heat_NF)
│   ├── BenchmarkingEBCBAROM-RF.py  # (RDFNF)
│   ├── Benchmarking.py           # General benchmarking script for Category 2
│   ├── PerformanceOnGPU.py       # Script for GPU performance metrics
│   └── command_line.md           # Instructions for running experiments
└── Ablation_study/               # Scripts and data for ablation studies (on RDFNF dataset)
├── datasets_new_feedback/    # Information about dataset used for ablation
├── BAROM_pod_dim_ablation.py # Ablation on latent dimension
├── BAROM_fixedlifting.py     # Ablation on lifting network (learnable vs. fixed)
├── BAROM_Random_pod_initialize.py # Ablation on basis initialization
├── BAROM_Non_attention.py    # Ablation on attention mechanism (BAA vs. NoAttn)
├── Benchmarking_Noatten_BAROM.py # Supporting benchmark script for NoAttn
├── benchmarking.py           # General benchmarking script for ablation
└── ablation.md               # Details and commands for ablation studies
```


