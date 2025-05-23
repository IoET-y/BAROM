# Latent Space Learning for PDE Systems with Complex Boundary Conditions (BAROM)

## Abstract
Latent space Reduced Order Models (ROMs) in Scientific Machine Learning (SciML) can enhance and accelerate Partial Differential Equation (PDE) simulations. However, they often struggle with complex boundary conditions (BCs) such as time-varying, nonlinear, or state-dependent ones. Current methods for handling BCs in latent space have limitations due to representation mismatch and projection difficulty, impacting predictive accuracy and physical consistency. To address this, we introduce BAROM (Boundary-Aware Attention ROM). BAROM integrates: (1) explicit, Reduced Basis Methods-inspired boundary treatment using a modified ansatz and a learnable lifting network for complex BCs; and (2) a non-intrusive, attention-based mechanism, inspired by Galerkin Neural Operators, to learn internal field dynamics within a POD-initialized latent space. Evaluations show BAROM achieves superior accuracy and robustness on benchmark PDEs with diverse complex BCs compared to established SciML approaches.

---

## 📜 Overview
This repository contains the source code and experimental setup for the BAROM model, as presented in the paper "Latent Space Learning for PDE Systems with Complex Boundary." BAROM is a novel framework designed to accurately and efficiently simulate PDE systems, particularly those with complex, dynamic, and feedback-dependent boundary conditions.

The core contributions include:
* An explicit boundary-internal field decomposition inspired by Reduced Basis Methods (RBM).
* A learnable lifting network to handle non-homogeneous boundary conditions.
* A Boundary-Aware Attention (BAA) mechanism for evolving latent coefficients, making the model responsive to boundary changes.
* Two main variants:
    * **BAROM\_ImpBC**: Handles boundary conditions implicitly during the latent dynamics update. (Implemented in `PDE_category1/BAROM_ImpBC.py` and `PDE_category2/BAROM_ImpBC.py`)
    * **BAROM\_ExpBC**: Explicitly processes boundary parameters to inform the latent dynamics update, aligning with Equation (9) in the paper. (Implemented in `PDE_category2/BAROM_ExpBC.py`)

---

## 📂 Repository Structure

The repository is organized as follows:

* **`PDE_category1/`**: Contains code and datasets for PDE systems with externally prescribed complex boundary conditions (no internal feedback).
    * `PC1_datageneration/`: Scripts for generating these datasets.
        * `generate_datasets_PDE_task1.py`: Main generation script.
    * `datasets_full/`: Location for storing or describing Category 1 datasets.
    * Individual Python files (e.g., `BAROM_ImpBC.py`, `FNO.py`, `SPFNO.py`, etc.): Implementations of BAROM (implicit BC version for these simpler cases) and baseline models.
* **`PDE_category2/`**: Contains code and datasets for PDE systems with complex boundary conditions involving internal-boundary coupling and/or external control signals.
    * `generate_datasets_PDE_task2.py`: Script for generating these more complex datasets.
    * `datasets_new_feedback/`: Location for storing or describing Category 2 datasets.
    * `BAROM_ExpBC.py`: Main implementation of BAROM with explicit boundary processing (aligned with Eq. 9).
    * `BAROM_ImpBC.py`: Implementation of BAROM with implicit boundary processing.
    * Other Python files: Implementations of baseline models for Category 2.
    * `Benchmarking4GPU.py`: Script for benchmarking various models, including BAROM.
* **`Ablation_study/`**: Contains scripts for ablation studies on BAROM's components and hyperparameters.
    * Includes variants like `BAROM_Non_attention.py`, `BAROM_Random_pod_initialize.py`, `BAROM_fixedlifting.py`, `BAROM_pod_dim_ablation.py`.
* **Root Directory**:
    * `README.md`: This file.

---

## ⚙️ The BAROM Model (Equation 9 Aligned Version)

The primary version of BAROM, referred to as **BAROM\_ExpBC** in the paper and implemented primarily in `PDE_category2/BAROM_ExpBC.py`, uses an explicit boundary treatment mechanism. The key evolution step for the latent coefficients $\mathbf{a}$ aligns with Equation (9) from the paper:

$\mathbf{a}^{n+1} = \underbrace{\mathbf{a}^{n} + \Delta\mathbf{a}_{\text{attn}} + \Delta\mathbf{a}_{\text{ffn}}}_{\text{Internal Dynamics (Learned)}} + \underbrace{\Delta\mathbf{a}_{\text{bc}}}_{\text{Learned from } P_{BC}(t_{k+1})} \quad (*)$
*(Note: The original paper's Equation (9) is $a^{n+1} = \hat{A}_{a}a^{n} + \hat{B}\hat{w}^{n} + \hat{A}_{BC}U_{B}^{n} - \Phi^{T}U_{B}^{n+1}$. The $\Delta\mathbf{a}_{\text{bc}}$ term in the code and paper description aims to encapsulate the influence of boundary parameters $P_{BC}(t_{k+1})$ on the latent state, which conceptually covers terms like $\hat{B}\hat{w}^{n} + \hat{A}_{BC}U_{B}^{n} - \Phi^{T}U_{B}^{n+1}$ in a learned, potentially more direct manner when $P_{BC}$ is used to derive features that directly update $a^n$. The code provided has been updated to more directly reflect the structure of this equation, particularly the inclusion of $U_B^{n+1}$ effects).*

The model reconstructs the physical solution using:
$\hat{U}^{n+1} = U_B(P_{BC}(t_{k+1})) + \mathbf{\Phi} \mathbf{a}^{n+1}$

Key components:
* **Lifting Network (`UniversalLifting` class)**: Maps boundary parameters $P_{BC}(t)$ to the boundary field $U_B(x,t)$.
* **Learnable Basis Functions ($\mathbf{\Phi}$)**: Initialized with POD and refined during training. Stored as `model.Phi`.
* **Latent Dynamics (`MultiVarAttentionROMEq9.forward_step`)**:
    * Calculates $U_B^n = \mathcal{L}_{\text{lift}}(P_{BC}(t_n))$.
    * Calculates $U_B^{n+1} = \mathcal{L}_{\text{lift}}(P_{BC}(t_{n+1}))$.
    * $\Delta\mathbf{a}_{\text{attn}}$: From multi-head attention.
    * $\Delta\mathbf{a}_{\text{ffn}}$: From a feed-forward network.
    * $\Delta\mathbf{a}_{\text{bc}}$: Explicitly processes $P_{BC}(t_{n+1})$ to update latent coefficients. The implementation details of how $P_{BC}(t_n)$ and $P_{BC}(t_{n+1})$ map to the terms in the original Eq. (9) ($\hat{B}\hat{w}^{n}$, $\hat{A}_{BC}U_{B}^{n}$, and $-\Phi^{T}U_{B}^{n+1}$) are embedded within the `bc_driven_a_update` and the direct subtraction of the $\Phi^{T}U_B^{n+1}$ projection.

---

## 📊 Datasets

The paper evaluates BAROM on two categories of PDE problems:

1.  **Category 1: Externally Prescribed Complex BCs (No Internal Feedback)**
    * PDEs: 1D Advection, 1D Euler, 1D Burgers', 2D Darcy flow.
    * Characteristics: Time-varying, non-linear boundary conditions.
    * Generation: `PDE_category1/PC1_datageneration/generate_datasets_PDE_task1.py`
    * Description: `PDE_category1/datasets_full/dataset.md`

2.  **Category 2: Complex BCs with Internal-Boundary Coupling and Control**
    * PDEs: Reaction-Diffusion with Neumann Feedback (RDFNF), Heat Equation with Non-linear Gain Feedback (Heat\_NF), Convection-Diffusion with Integral Boundary Control (Convdiff).
    * Characteristics: Boundary conditions dynamically coupled with the internal system state, may include external control signals.
    * Generation: `PDE_category2/generate_datasets_PDE_task2.py`
    * Description: `PDE_category2/datasets_new_feedback/dataset.md`

Refer to Appendix A of the paper for more details on dataset generation.

---

## 🚀 Experiments and Benchmarking
### Training model:

### Training BAROM
The main training script for BAROM (especially the Eq. 9 aligned version, BAROM\_ExpBC) is located in `PDE_category2/BAROM_ExpBC.py`. The script `PDE_category1/BAROM_ImpBC.py` can be used for the implicit BC version, typically for Category 1 PDEs.

**Example Training Command (for BAROM\_ExpBC):**
```bash
python PDE_category2/BAROM_ExpBC.py --datatype heat_nonlinear_feedback_gain --basis_dim 32 --d_model 512 --num_heads 8 --bc_processed_dim 32 --num_epochs 150 --lr 5e-4 --batch_size 32
