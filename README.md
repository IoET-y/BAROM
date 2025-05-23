# Latent Space Learning for PDE Systems with Complex Boundary (BAROM)

## 📜 Abstract
Latent space Reduced Order Models (ROMs) in Scientific Machine Learning (SciML) can enhance and accelerate Partial Differential Equation (PDE) simulations. However, they often struggle with complex boundary conditions (BCs) such as time-varying, nonlinear, or state-dependent ones. Current methods for handling BCs in latent space have limitations due to **representation mismatch and projection difficulty**, impacting predictive accuracy and physical consistency. To address this, we introduce BAROM (Boundary-Aware Attention ROM). BAROM integrates: (1) explicit, Reduced Basis Methods-inspired boundary treatment using a modified ansatz and a learnable lifting network for complex BCs; and (2) a non-intrusive, attention-based mechanism, inspired by Galerkin Neural Operators, to learn internal field dynamics within a POD-initialized latent space. Evaluations show BAROM achieves superior accuracy and robustness on benchmark PDEs with diverse complex BCs compared to established SciML approaches.

---

## 📖 Overview
This repository provides the source code, datasets, and experimental setup for the **Boundary-Aware Attention Reduced Order Model (BAROM)**. This work focuses on developing a robust SciML framework for simulating PDE systems, particularly those with complex, dynamic, and feedback-dependent boundary conditions.

The core contributions of BAROM include:
* **Explicit Boundary-Internal Field Decomposition**: Inspired by Reduced Basis Methods (RBM), BAROM decomposes the solution into a boundary-conforming part ($U_B$) and an internal part ($U_I$) satisfying homogeneous BCs.
* **Learnable Lifting Network**: A neural network ($\mathcal{L}_{\text{lift}}$) learns to map time-dependent boundary parameters ($P_{BC}(t)$) to the boundary field $U_B(x,t)$.
* **Boundary-Aware Attention (BAA) Latent Dynamics**: An attention-based mechanism evolves the latent coefficients ($\mathbf{a}(t)$) of the internal field $U_I = \mathbf{\Phi} \mathbf{a}(t)$. This evolution is explicitly conditioned on boundary parameters.
* **Two Main Variants Explored**:
    * **BAROM\_ImpBC (Implicit Boundary Condition Handling)**: Boundary influence on latent dynamics is primarily implicit. (See `PDE_category1/BAROM_ImpBC.py` and `PDE_category2/BAROM_ImpBC.py`)
    * **BAROM\_ExpBC (Explicit Boundary Condition Handling)**: Explicitly processes boundary parameters to inform the latent dynamics update, aligning with Equation (9) in the paper for enhanced boundary awareness. (Primary implementation in `PDE_category2/BAROM_ExpBC.py`)

---

## 📂 Repository Structure

The repository is organized into several main directories:

* **`PDE_category1/`**: Focuses on PDE systems with externally prescribed complex boundary conditions (no internal feedback or direct control signals).
    * `PC1_datageneration/`: Contains scripts to generate datasets for this category (e.g., `generate_datasets_PDE_task1.py`). Includes visualization notebooks like `data_visualization.ipynb` and command references (`command.md`).
    * `datasets_full/`: Contains descriptions and potentially storage for Category 1 datasets (e.g., `dataset.md`).
    * Python scripts (e.g., `BAROM_ImpBC.py`, `FNO.py`, `SPFNO.py`, `BENO.py`, `LNO.py`, `LNS_AE.py`, `POD_DL_ROM.py`, `DON.py`, `OPNO.py`): Implementations of BAROM and baseline models for Category 1 PDEs.
    * `Benchmarking.py`: Script for benchmarking models on Category 1 datasets.
    * `command_line.md`: Contains example command lines for running experiments in this category.

* **`PDE_category2/`**: Focuses on PDE systems with complex boundary conditions involving internal-boundary coupling and/or external control signals.
    * `generate_datasets_PDE_task2.py`: Script for generating these more complex datasets.
    * `datasets_new_feedback/`: Contains descriptions and potentially storage for Category 2 datasets (e.g., `dataset.md`).
    * `BAROM_ExpBC.py`: Main implementation of BAROM with explicit boundary processing (aligned with Eq. 9).
    * `BAROM_ImpBC.py`: Implementation of BAROM with implicit boundary processing.
    * Other Python files (e.g., `FNO.py`, `SPFNO_FULL.py`, `BENO.py`, `LNO.py`, `LNS_AE.py`, `POD_DL_ROM.py`): Implementations of baseline models.
    * `Benchmarking4GPu.py`: Main script for benchmarking models on Category 2 datasets, including performance on GPUs.
    * Other specialized benchmarking scripts (e.g., `BenchmarkingEBCBAROM-CDF.py`).
    * `command_line.md`: Contains example command lines for experiments in this category.
    * `datasetscript.md`: Further details on dataset scripting.
    * `data_visualization.ipynb`: Notebook for visualizing Category 2 datasets.

* **`Ablation_study/`**: Contains scripts and configurations for ablation studies performed on BAROM.
    * `BAROM_Non_attention.py`, `BAROM_Random_pod_initialize.py`, `BAROM_fixedlifting.py`, `BAROM_pod_dim_ablation.py`: Scripts for specific ablation experiments.
    * `Benchmarking_Noatten_BAROM.py`, `benchmarking.py`: Benchmarking scripts for ablation variants.
    * `ablation.md`: Description of the ablation studies.

* **Root Directory**:
    * `README.md`: This file.
    * Other configuration files or utility scripts if present.

---

## 🤖 The BAROM Model (Equation 9 Aligned Version - BAROM\_ExpBC)

The primary BAROM model, **BAROM\_ExpBC**, is implemented in `PDE_category2/BAROM_ExpBC.py`. Its core latent dynamics update aligns with Equation (9) from the paper:
$a^{n+1} = \hat{A}_{a}a^{n} + \hat{B}\hat{w}^{n} + \hat{A}_{BC}U_{B}^{n} - \Phi^{T}U_{B}^{n+1}$

And the reconstruction is:
$\hat{U}^{n+1} = U_{B}^{n+1} + \Phi a^{n+1}$

Key components within the `MultiVarAttentionROMEq9` class (or `MultiVarAttentionROM` in the training script `BAROM_ExpBC.py`):
* **Lifting Network (`UniversalLifting` class)**: Generates $U_B^n$ from $P_{BC}(t_n)$ and $U_B^{n+1}$ from $P_{BC}(t_{n+1})$.
* **Learnable Basis Functions ($\mathbf{\Phi}$)**: Stored as `model.Phi`.
* **Latent Dynamics Update (`forward_step` method)**:
    * `a_update_attn_val` + `alpha_var * ffn_update_intrinsic_val`: Learns the contribution analogous to $(\hat{A}_a - I)a^n$.
    * `bc_driven_a_update_val`: Learns contributions analogous to $\hat{B}\hat{w}^n + \hat{A}_{BC}U_B^n$, derived from $P_{BC}(t_n)$.
    * `term_to_subtract_phiT_UBnp1`: Explicitly calculates and subtracts the $\Phi^{T}U_B^{n+1}$ term.
* **Reconstruction**: Uses $U_B^{n+1}$ and $a^{n+1}$ as per the paper's formulation.

Refer to Section 3 of the paper for a detailed methodological description.

---

## 📊 Datasets

The paper evaluates BAROM on two categories of PDE problems:

1.  **Category 1: Externally Prescribed Complex BCs**
    * PDEs: 1D Advection, 1D Euler, 1D Burgers', 2D Darcy flow.
    * Generation Script: `PDE_category1/PC1_datageneration/generate_datasets_PDE_task1.py`
    * Description: See `PDE_category1/datasets_full/dataset.md`.

2.  **Category 2: Complex BCs with Internal-Boundary Coupling and Control**
    * PDEs: Reaction-Diffusion with Neumann Feedback (RDFNF), Heat Equation with Non-linear Gain Feedback (Heat\_NF), Convection-Diffusion with Integral Boundary Control (Convdiff).
    * Generation Script: `PDE_category2/generate_datasets_PDE_task2.py`
    * Description: See `PDE_category2/datasets_new_feedback/dataset.md`.

Detailed information on dataset generation parameters can be found in Appendix A of the paper and the respective `dataset.md` files.

---

## 🚀 Getting Started

### Prerequisites
* Python (3.8+ recommended)
* PyTorch (see scripts for version, e.g., 1.8+)
* NumPy
* Matplotlib
* Pickle
* Argparse
* Glob

It is recommended to set up a Python virtual environment.

### 1. Datasets
Pre-generated datasets are expected in the `PDE_category1/datasets_full/` and `PDE_category2/datasets_new_feedback/` directories. If you need to regenerate them, use the scripts mentioned in the "Datasets" section above.
Example:
```bash
python PDE_category2/generate_datasets_PDE_task2.py --datatype heat_nonlinear_feedback_gain

2. Training BAROM
The main training scripts for BAROM are:

PDE_category2/BAROM_ExpBC.py (for Equation 9 aligned model, typically for Category 2 PDEs)
PDE_category1/BAROM_ImpBC.py or PDE_category2/BAROM_ImpBC.py (for implicit BC handling)
Example Training Command (BAROM_ExpBC):

Bash

python PDE_category2/BAROM_ExpBC.py --datatype heat_nonlinear_feedback_gain \
                                   --basis_dim 32 \
                                   --d_model 512 \
                                   --num_heads 4 \
                                   --bc_processed_dim 64 \
                                   --hidden_bc_processor_dim 128 \
                                   --num_epochs 150 \
                                   --lr 5e-4 \
                                   --batch_size 32
Adjust hyperparameters as needed. Checkpoints are typically saved in subdirectories like New_ckpt_explicit_bc_eq9/. Refer to PDE_category1/command_line.md and PDE_category2/command_line.md for more examples.

3. Running Benchmarks
The primary benchmarking script for Category 2 PDEs (which often includes BAROM_ExpBC) is PDE_category2/Benchmarking4GPu.py. A similar script exists for Category 1: PDE_category1/Benchmarking.py.

Example Benchmarking Command:

Bash

python PDE_category2/Benchmarking4GPu.py --datasets heat_nonlinear_feedback_gain --models BAROM SPFNO # Add other models
Ensure that the model configurations within the benchmark script (e.g., MODEL_CONFIGS, MODEL_TO_BENCHMARK_CONFIG) correctly point to your trained model checkpoints and use the appropriate model class name (e.g., MultiVarAttentionROMEq9 or the name used in your training script).

4. Ablation Studies
Scripts for ablation studies are located in the Ablation_study/ directory. Refer to Ablation_study/ablation.md for descriptions and corresponding scripts (e.g., BAROM_Non_attention.py). These can be run similarly to the training scripts.

📊 Results
Quantitative results comparing BAROM with baselines are presented in Tables 1, 2, 3, and 4 of the paper. Visual comparisons can be found in Figures 2-7 in the Appendix. Efficiency metrics are detailed in Appendix F.

✍️ Citation
If you find this work useful, please cite our paper:
