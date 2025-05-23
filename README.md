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

## Requirements

* Python 3.x
* PyTorch (developed with versions used for experiments, see paper Appendix I)
* NumPy
* SciPy
* Matplotlib (for visualizations)
* An NVIDIA GPU is recommended for training and inference (experiments run on NVIDIA A40, see Appendix D & F in paper).

We recommend creating a `requirements.txt` file based on your environment and the libraries used.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/anonymous_user/BAROM.git](https://github.com/anonymous_user/BAROM.git) # Replace with actual repo link
    cd BAROM
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv barom_env
    source barom_env/bin/activate  # On Windows: barom_env\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install torch numpy scipy matplotlib # Add other specific packages if needed
    # Or, if a requirements.txt is provided:
    # pip install -r requirements.txt
    ```

## Dataset Generation

Datasets are synthetically generated as described in Appendix A of the paper.

### Category 1: PDEs without Internal Feedback

(1D Advection, 1D Euler, 1D Burgers', 2D Darcy)
* **Script**: `PDE_category1/PC1_datageneration/generate_datasets_PDE_task1.py`
* **Instructions**: See `PDE_category1/PC1_datageneration/command.md`
    ```bash
    cd PDE_category1/PC1_datageneration/
    python generate_datasets_PDE_task1.py # Follow instructions in command.md
    ```
* **Dataset details**: `PDE_category1/datasets_full/dataset.md`

### Category 2: PDEs with Internal Feedback and Boundary Control

(Reaction-Diffusion (RDFNF), Heat Equation (Heat_NF), Convection-Diffusion (Convdiff))
* **Script**: `PDE_category2/generate_datasets_PDE_task2.py`
* **Instructions**: See `PDE_category2/datasetscript.md` and `PDE_category2/command_line.md`
    ```bash
    cd PDE_category2/
    python generate_datasets_PDE_task2.py # Follow instructions in datasetscript.md
    ```
* **Dataset details**: `PDE_category2/datasets_new_feedback/dataset.md`

## Training and Evaluation

Detailed hyperparameter settings for BAROM and all baselines are provided in Appendix I of the paper. Implementation details are in Appendix D and E.

### Model Implementations

* **BAROM**: `BAROM_ImpBC.py`, `BAROM_ExpBC.py`
* **Baselines**: Located in respective `PDE_categoryX/` directories (e.g., `FNO.py`, `LNS_AE.py`, etc.).

### Running Experiments

#### Category 1 PDEs (No Internal Feedback)

* Navigate to `PDE_category1/`.
* Use `Benchmarking.py`. Refer to `PDE_category1/command_line.md` for specific commands.
    ```bash
    cd PDE_category1/
    # Example: python Benchmarking.py --model BAROM_ImpBC --dataset Advection ...
    ```

#### Category 2 PDEs (Internal Feedback & Control)

* Navigate to `PDE_category2/`.
* Use general `Benchmarking.py` or dataset-specific scripts (e.g., `BenchmarkingEBCBAROM-CDF.py`). Refer to `PDE_category2/command_line.md`.
    ```bash
    cd PDE_category2/
    # Example: python Benchmarking.py --model BAROM_ExpBC --dataset Convdiff ...
    # OR: python BenchmarkingEBCBAROM-CDF.py ...
    ```

#### Ablation Studies

* Navigate to `Ablation_study/`.
* Scripts are provided for each ablation aspect (e.g., `BAROM_pod_dim_ablation.py`).
* Refer to `Ablation_study/ablation.md` for detailed commands and explanations.
    ```bash
    cd Ablation_study/
    # Example: python BAROM_pod_dim_ablation.py --dataset RDFNF ...
    ```

## Methodology Overview

BAROM introduces a novel framework for simulating PDEs with complex BCs by decomposing the approximate physical solution $\hat{U}(x,t;\mu)$ into a boundary field $U_B$ and an internal field $U_I$:
$$ \hat{U}(x,t;\mu) = U_B(x,t; P_{BC}(t), \mu) + U_I(x,t;\mu) $$

* **Boundary Field ($U_B$)**: Generated by a learnable **Lifting Network** ($\mathcal{L}_{\text{lift}}$), which maps boundary parameters $P_{BC}(t)$ (including external values, controls, and feedback terms) to a field satisfying the non-homogeneous BCs.
* **Internal Field ($U_I$)**: Captures dynamics within the domain subject to homogeneous BCs. It is represented using $N$ learnable spatial basis functions $\mathbf{\Phi}(x)$ (POD-initialized and refined) and time-dependent latent coefficients $\mathbf{a}(t;\mu) \in \mathbb{R}^N$.
* **Latent Dynamics Evolution**: The temporal evolution of $\mathbf{a}(t)$ is governed by an attention-based neural network module, $\mathcal{F}_{\text{latent}}$. The update mechanism (for `BAROM_ExpBC`) is:
    $$ \mathbf{a}(t_{k+1}) = \mathbf{a}(t_k) + \Delta\mathbf{a}_{\text{attn}} + \Delta\mathbf{a}_{\text{ffn}} + \Delta\mathbf{a}_{\text{bc}} $$
    where $\Delta\mathbf{a}_{\text{attn}}$ models internal dynamics via attention, $\Delta\mathbf{a}_{\text{ffn}}$ captures non-linear intrinsic evolution, and $\Delta\mathbf{a}_{\text{bc}}$ integrates explicit boundary condition forcing. `BAROM_ImpBC` omits the explicit $\Delta\mathbf{a}_{\text{bc}}$ term.

The overall architecture is detailed in Section 3 and conceptually illustrated in Figure 1 of the paper. Algorithm 1 in the paper outlines the BAROM predictive step.

## Experimental Results Summary

BAROM demonstrates state-of-the-art performance and robustness across challenging PDE benchmarks.

* **PDEs without Internal Feedback (Category 1)**: `BAROM_ImpBC` consistently achieves low errors, especially in temporal extrapolation (see Table 1 in the paper).
* **PDEs with Internal Feedback & Control (Category 2)**: `BAROM_ExpBC` significantly outperforms all baseline models across all datasets and time horizons, often by several orders of magnitude in MSE. This highlights its efficacy in handling systems with intricate internal-boundary couplings (see Tables 2, 3, 4 in the paper).
* **Ablation Studies**: Confirmed the importance of BAROM's key architectural components, including the learnable lifting network, explicit boundary information in the latent dynamics, and the attention mechanism (see Section 4.3 and Figure 5 in the paper).

Visualizations of predicted solution fields are available in Appendix B of the paper. A computational efficiency comparison is provided in Appendix F.

