# =============================================================================
#        COMPLETE CODE: Handling PDE
#     with Control Inputs, Modified Lifting, and Multi-Variable Support
# =============================================================================
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import time
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import argparse
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve # For Burgers' solver
# ---------------------
# 固定随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
# ---------------------

# =============================================================================
# 2. 通用化数据集定义
# =============================================================================
class UniversalPDEDataset(Dataset):
    def __init__(self, data_list, dataset_type, train_nt_limit=None): # Added train_nt_limit
        """

        Args:
            data_list: 包含样本字典的列表。
            train_nt_limit: If specified, truncate sequences to this length for training.
        """
        if not data_list:
            raise ValueError("data_list cannot be empty")
        self.data_list = data_list
        self.dataset_type = dataset_type
        self.train_nt_limit = train_nt_limit # Store the limit

        # --- 从第一个样本推断参数 ---
        first_sample = data_list[0]
        params = first_sample.get('params', {})

        # --- 获取样本中的原始nt, nx, ny ---
        # This 'self.nt_from_sample' will be the full length in the file


        # Determine effective nt for the dataset instance (used in __getitem__)


        if dataset_type == 'heat_delayed_feedback':
            self.nt_from_sample = first_sample['U'].shape[0]
            self.nx_from_sample = first_sample['U'].shape[1]
            self.ny_from_sample = 1
            self.state_keys = ['U']
            self.num_state_vars = 1
            self.nx = self.nx_from_sample
            self.ny = self.ny_from_sample
            self.expected_bc_state_dim = 2 # BC_State stores [g0_t_base, gL_t_base + Feedback_delayed]
        elif dataset_type == 'reaction_diffusion_neumann_feedback':
            self.nt_from_sample = first_sample['U'].shape[0]
            self.nx_from_sample = first_sample['U'].shape[1]
            self.ny_from_sample = 1
            self.state_keys = ['U']
            self.num_state_vars = 1
            self.nx = self.nx_from_sample
            self.ny = self.ny_from_sample
            self.expected_bc_state_dim = 2 # BC_State stores [g0_t_base, gL_flux_base + Feedback_flux]
        elif dataset_type == 'heat_nonlinear_feedback_gain':
            self.nt_from_sample = first_sample['U'].shape[0]
            self.nx_from_sample = first_sample['U'].shape[1]
            self.ny_from_sample = 1
            self.state_keys = ['U']
            self.num_state_vars = 1
            self.nx = self.nx_from_sample
            self.ny = self.ny_from_sample
            self.expected_bc_state_dim = 2 # BC_State stores [g0_t_base, gL_t_base + Nonlinear_Feedback_term]
            
        elif dataset_type == 'convdiff':
            # 新增：Convection–Diffusion 带积分反馈
            self.nt_from_sample    = first_sample['U'].shape[0]
            self.nx_from_sample    = first_sample['U'].shape[1]
            self.state_keys        = ['U']
            self.num_state_vars    = 1
            self.nx                = self.nx_from_sample
            self.ny                = 1
            # BC_State 存储左右两端的 u 值，BC_Control 存储左右两端的积分控制
            self.expected_bc_state_dim = 2
            
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")
        self.effective_nt = self.train_nt_limit if self.train_nt_limit is not None else self.nt_from_sample

        self.spatial_dim = self.nx * self.ny

        self.bc_state_key = 'BC_State'
        if self.bc_state_key not in first_sample:
            raise KeyError(f"'{self.bc_state_key}' not found in the first sample!")
        actual_bc_state_dim = first_sample[self.bc_state_key].shape[1]
        if actual_bc_state_dim != self.expected_bc_state_dim:
              print(f"Warning: BC_State dimension mismatch for {dataset_type}. "
                    f"Expected {self.expected_bc_state_dim}, got {actual_bc_state_dim}. "
                    f"Using actual dimension: {actual_bc_state_dim}")
              self.bc_state_dim = actual_bc_state_dim
        else:
              self.bc_state_dim = self.expected_bc_state_dim

        self.bc_control_key = 'BC_Control'
        if self.bc_control_key in first_sample:
            self.num_controls = first_sample[self.bc_control_key].shape[1]
        else:
            self.num_controls = 0
            print(f"Warning: '{self.bc_control_key}' not found in the first sample. Assuming num_controls = 0.")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        norm_factors = {}

        # --- Potentially truncate sequences ---
        current_nt = self.effective_nt # Use the effective_nt for slicing

        state_tensors_norm_list = []
        for key in self.state_keys:
            try:
                state_seq_full = sample[key] # Full sequence from file
                # Truncate if train_nt_limit is set
                state_seq = state_seq_full[:current_nt, ...]
            except KeyError:
                raise KeyError(f"State variable key '{key}' not found in sample {idx} for dataset type '{self.dataset_type}'")

            if state_seq.shape[0] != current_nt: # Should not happen if slicing is correct
                 raise ValueError(f"Time dimension mismatch after potential truncation for {key}. Expected {current_nt}, got {state_seq.shape[0]}")

            # Normalization (now happens on potentially truncated data)
            state_mean = np.mean(state_seq)
            state_std = np.std(state_seq) + 1e-8
            state_norm = (state_seq - state_mean) / state_std

            state_tensors_norm_list.append(torch.tensor(state_norm).float())
            norm_factors[f'{key}_mean'] = state_mean
            norm_factors[f'{key}_std'] = state_std

        # --- BC State ---
        bc_state_seq_full = sample[self.bc_state_key]
        bc_state_seq = bc_state_seq_full[:current_nt, :] # Truncate

        if bc_state_seq.shape[0] != current_nt:
            raise ValueError(f"Time dimension mismatch for BC_State. Expected {current_nt}, got {bc_state_seq.shape[0]}")

        bc_state_norm = np.zeros_like(bc_state_seq, dtype=np.float32)
        # ... (rest of BC_State normalization logic, applied to 'bc_state_seq')
        norm_factors[f'{self.bc_state_key}_means'] = np.zeros(self.bc_state_dim)
        norm_factors[f'{self.bc_state_key}_stds'] = np.ones(self.bc_state_dim)
        for k_dim in range(self.bc_state_dim): # Renamed loop variable
            col = bc_state_seq[:, k_dim]
            mean_k = np.mean(col)
            std_k = np.std(col)
            if std_k > 1e-8:
                bc_state_norm[:, k_dim] = (col - mean_k) / std_k
                norm_factors[f'{self.bc_state_key}_means'][k_dim] = mean_k
                norm_factors[f'{self.bc_state_key}_stds'][k_dim] = std_k
            else:
                bc_state_norm[:, k_dim] = 0.0
                norm_factors[f'{self.bc_state_key}_means'][k_dim] = mean_k
        bc_state_tensor_norm = torch.tensor(bc_state_norm).float()


        # --- BC Control ---
        if self.num_controls > 0:
            try:
                bc_control_seq_full = sample[self.bc_control_key]
                bc_control_seq = bc_control_seq_full[:current_nt, :] # Truncate

                if bc_control_seq.shape[0] != current_nt:
                    raise ValueError(f"Time dimension mismatch for BC_Control. Expected {current_nt}, got {bc_control_seq.shape[0]}.")
                # ... (rest of BC_Control normalization logic, applied to 'bc_control_seq')
                bc_control_norm = np.zeros_like(bc_control_seq, dtype=np.float32)
                norm_factors[f'{self.bc_control_key}_means'] = np.zeros(self.num_controls)
                norm_factors[f'{self.bc_control_key}_stds'] = np.ones(self.num_controls)
                for k_dim in range(self.num_controls): # Renamed loop variable
                    col = bc_control_seq[:, k_dim]
                    mean_k = np.mean(col)
                    std_k = np.std(col)
                    if std_k > 1e-8:
                        bc_control_norm[:, k_dim] = (col - mean_k) / std_k
                        norm_factors[f'{self.bc_control_key}_means'][k_dim] = mean_k
                        norm_factors[f'{self.bc_control_key}_stds'][k_dim] = std_k
                    else:
                        bc_control_norm[:, k_dim] = 0.0
                        norm_factors[f'{self.bc_control_key}_means'][k_dim] = mean_k
                bc_control_tensor_norm = torch.tensor(bc_control_norm).float()

            except KeyError:
                print(f"Warning: Sample {idx} missing '{self.bc_control_key}'. Using zeros.")
                bc_control_tensor_norm = torch.zeros((current_nt, self.num_controls), dtype=torch.float32)
                norm_factors[f'{self.bc_control_key}_means'] = np.zeros(self.num_controls)
                norm_factors[f'{self.bc_control_key}_stds'] = np.ones(self.num_controls)
        else:
            bc_control_tensor_norm = torch.empty((current_nt, 0), dtype=torch.float32)

        bc_ctrl_tensor_norm = torch.cat((bc_state_tensor_norm, bc_control_tensor_norm), dim=-1)
        return state_tensors_norm_list, bc_ctrl_tensor_norm, norm_factors



# =============================================================================
# 3. 通用化 POD 基计算 (优化 U_B 定义)
# =============================================================================
# In your model/training script
def compute_pod_basis_generic(data_list, dataset_type, state_variable_key,
                              nx, nt, basis_dim, # nt here is the TRUNCATED length
                              max_snapshots_pod=100):
    """
    Computes POD basis for a specific state variable from the data list.
    Uses linear interpolation of *actual* snapshot boundary values for U_B.
    Data is truncated to 'nt' timesteps.
    """
    snapshots = []
    count = 0
    current_nx = nx
    lin_interp = np.linspace(0, 1, current_nx)[np.newaxis, :]
    print(f"  Computing POD for '{state_variable_key}' using {nt} timesteps, linear interp U_B...")

    for sample_idx, sample in enumerate(data_list):
        if count >= max_snapshots_pod:
            break
        if state_variable_key not in sample:
            print(f"Warning: Key '{state_variable_key}' not found in sample {sample_idx}. Skipping.")
            continue

        U_seq_full = sample[state_variable_key]  # shape: [full_file_nt, current_nx]
        U_seq = U_seq_full[:nt, :] # <<< TRUNCATE TO THE REQUIRED nt FOR POD >>>

        if U_seq.shape[0] != nt: # Should match the 'nt' passed to the function
             print(f"Warning: U_seq actual timesteps {U_seq.shape[0]} != requested nt {nt} for POD in sample {sample_idx}. Skipping.")
             continue
        if U_seq.shape[1] != current_nx:
            print(f"Warning: Mismatch nx in sample {sample_idx} for {state_variable_key}. Expected {current_nx}, got {U_seq.shape[1]}. Skipping.")
            continue

        bc_left_val = U_seq[:, 0:1]
        bc_right_val = U_seq[:, -1:]

        if np.isnan(bc_left_val).any() or np.isinf(bc_left_val).any() or \
           np.isnan(bc_right_val).any() or np.isinf(bc_right_val).any():
            print(f"Warning: NaN/Inf in boundary values for sample {sample_idx}, key '{state_variable_key}'. Skipping sample for POD.")
            continue

        U_B = bc_left_val * (1 - lin_interp) + bc_right_val * lin_interp
        U_star = U_seq - U_B
        snapshots.append(U_star)
        count += 1
    # ... (rest of the POD function remains the same) ...
    if not snapshots:
        print(f"Error: No valid snapshots collected for POD for '{state_variable_key}'. Ensure 'nt' ({nt}) is appropriate.")
        return None
    # ... (rest of SVD and basis calculation)
    try:
        snapshots = np.concatenate(snapshots, axis=0) # [total_nt_steps, nx]
    except ValueError as e:
        # ... (error handling as before) ...
        print(f"Error concatenating snapshots for '{state_variable_key}': {e}. Check snapshot shapes. Collected {len(snapshots)} lists of snapshots.")
        # Attempt to filter potentially problematic shapes (simple check)
        valid_snapshots_for_concat = []
        for s_idx, s_item in enumerate(snapshots):
            if s_item.shape[0] == nt and s_item.shape[1] == current_nx : # Each item should have nt rows
                valid_snapshots_for_concat.append(s_item)
            else:
                print(f"    Snapshot {s_idx} has shape {s_item.shape}, expected ({nt}, {current_nx}). Discarding.")

        if not valid_snapshots_for_concat:
             print(f"  No valid snapshots remained after filtering for {state_variable_key}. Aborting POD.")
             return None
        try:
             snapshots = np.concatenate(valid_snapshots_for_concat, axis=0)
             print(f"  Successfully concatenated {len(valid_snapshots_for_concat)} filtered snapshots, total rows: {snapshots.shape[0]}.")
        except ValueError as e2:
             print(f"  Error even after filtering: {e2}. Aborting POD for '{state_variable_key}'.")
             return None
    # ... (the rest of the POD logic: mean centering, SVD, normalization, padding)
    if np.isnan(snapshots).any() or np.isinf(snapshots).any():
        print(f"Warning: NaN/Inf found in snapshots for '{state_variable_key}' before POD. Clamping.")
        snapshots = np.nan_to_num(snapshots, nan=0.0, posinf=1e6, neginf=-1e6)
        if np.all(np.abs(snapshots) < 1e-12):
            print(f"Error: All snapshots became zero after clamping for '{state_variable_key}'.")
            return None

    U_mean = np.mean(snapshots, axis=0, keepdims=True)
    U_centered = snapshots - U_mean

    try:
        U_data_svd, S_data_svd, Vh_data_svd = np.linalg.svd(U_centered, full_matrices=False)
        rank = np.sum(S_data_svd > 1e-10)
        actual_basis_dim = min(basis_dim, rank, current_nx)
        if actual_basis_dim == 0:
            print(f"Error: Data rank is zero for '{state_variable_key}'.")
            return None
        if actual_basis_dim < basis_dim:
            print(f"Warning: Requested basis_dim {basis_dim} but data rank is ~{rank} for '{state_variable_key}'. Using {actual_basis_dim}.")
        basis = Vh_data_svd[:actual_basis_dim, :].T
    except np.linalg.LinAlgError as e:
        print(f"SVD failed for '{state_variable_key}': {e}.")
        return None
    except Exception as e:
        print(f"Error during SVD for '{state_variable_key}': {e}.")
        return None

    basis_norms = np.linalg.norm(basis, axis=0)
    basis_norms[basis_norms < 1e-10] = 1.0
    basis = basis / basis_norms[np.newaxis, :]

    if actual_basis_dim < basis_dim:
        print(f"Padding POD basis for '{state_variable_key}' from dim {actual_basis_dim} to {basis_dim}")
        padding = np.zeros((current_nx, basis_dim - actual_basis_dim))
        basis = np.hstack((basis, padding))

    print(f"  Successfully computed POD basis for '{state_variable_key}' with shape {basis.shape}.")
    return basis.astype(np.float32)




# 4.1. Feedforward 更新网络 (不变)
class ImprovedUpdateFFN(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=3, dropout=0.1):
        super().__init__()
        layers = []
        current_dim = input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, input_dim))
        self.mlp = nn.Sequential(*layers)
        self.layernorm = nn.LayerNorm(input_dim)

    def forward(self, x):
        residual = x
        out = self.mlp(x)
        out = self.layernorm(out + residual)
        return out


# 4.2. Lifting 模块: 支持多变量状态输入和多变量状态输出 (增强容量)
class UniversalLifting(nn.Module):
    def __init__(self, num_state_vars, bc_state_dim, num_controls, output_dim_per_var, nx,
        # Increased default hidden dimensions
        hidden_dims_state_branch=64,
        hidden_dims_control=[64, 128],
        hidden_dims_fusion=[256, 512, 256], # Added a layer, increased dims
        dropout=0.1):
        """
        Args:
            num_state_vars: Number of state variables (e.g., 1 for Adv, 2 for Euler).
            bc_state_dim: Total dimension of the boundary state input vector
                          (e.g., 2 for Adv/Burgers/Darcy, 4 for Euler).
            num_controls: Number of control inputs.
            output_dim_per_var: Should be nx (spatial dimension). Lifting produces a field.
            nx: Spatial dimension (output_dim_per_var).
            hidden_dims_state_branch: Hidden dim for processing *each* state boundary value.
            hidden_dims_control: Hidden layer sizes for control MLP.
            hidden_dims_fusion: Hidden layer sizes for fusion network.
            dropout: Dropout rate.
        """
        super().__init__()
        self.num_state_vars = num_state_vars
        self.bc_state_dim = bc_state_dim
        self.num_controls = num_controls
        self.nx = nx
        assert output_dim_per_var == nx, "output_dim_per_var must equal nx"

        # --- State Branches ---
        # Create branches for each boundary state input component
        self.state_branches = nn.ModuleList()
        if self.bc_state_dim > 0:
             for _ in range(bc_state_dim):
                 # Process each BC state value individually
                 self.state_branches.append(nn.Sequential(
                     nn.Linear(1, hidden_dims_state_branch),
                     nn.GELU(),
                     # Optional: Add another layer if needed
                     # nn.Linear(hidden_dims_state_branch, hidden_dims_state_branch),
                     # nn.GELU()
                 ))
             state_feature_dim = bc_state_dim * hidden_dims_state_branch
        else:
             state_feature_dim = 0


        # --- Control MLP ---
        control_feature_dim = 0
        if self.num_controls > 0:
            control_layers = []
            current_dim = num_controls
            for h_dim in hidden_dims_control:
                control_layers.append(nn.Linear(current_dim, h_dim))
                control_layers.append(nn.GELU())
                control_layers.append(nn.Dropout(dropout))
                current_dim = h_dim
            self.control_mlp = nn.Sequential(*control_layers)
            control_feature_dim = current_dim # Output dim of control MLP
        else:
             # Define an empty sequential if no controls, avoids errors later
             self.control_mlp = nn.Sequential()


        # --- Fusion Network ---
        # Takes concatenated state features and control features
        fusion_input_dim = state_feature_dim + control_feature_dim
        fusion_layers = []
        current_dim = fusion_input_dim
        # Build the fusion MLP dynamically based on hidden_dims_fusion
        for h_dim in hidden_dims_fusion:
             fusion_layers.append(nn.Linear(current_dim, h_dim))
             fusion_layers.append(nn.GELU())
             fusion_layers.append(nn.Dropout(dropout))
             current_dim = h_dim

        # Final layer maps to the required output size: num_state_vars * nx
        fusion_layers.append(nn.Linear(current_dim, num_state_vars * nx))
        self.fusion = nn.Sequential(*fusion_layers)

    def forward(self, BC_Ctrl):
        """
        Args:
            BC_Ctrl: Tensor shape [batch, bc_state_dim + num_controls]
                     Assumes BC_State comes first, then BC_Control.
        Returns:
            U_B_stacked: Lifted boundary fields for all state variables,
                         shape [batch, num_state_vars, nx]
        """
        features_to_concat = []

        # Process State BCs if they exist
        if self.bc_state_dim > 0:
            BC_state = BC_Ctrl[:, :self.bc_state_dim]
            state_features_list = []
            for i in range(self.bc_state_dim):
                # Input shape [batch, 1] to each branch
                branch_out = self.state_branches[i](BC_state[:, i:i+1])
                state_features_list.append(branch_out)
            state_features = torch.cat(state_features_list, dim=-1) # [batch, state_feature_dim]
            features_to_concat.append(state_features)

        # Process Control BCs if they exist
        if self.num_controls > 0:
            # Ensure slicing is correct even if bc_state_dim is 0
            BC_control = BC_Ctrl[:, self.bc_state_dim:]
            control_features = self.control_mlp(BC_control) # [batch, control_feature_dim]
            features_to_concat.append(control_features)

        # Concatenate features
        if not features_to_concat:
             raise ValueError("Lifting network received no state or control inputs.")

        if len(features_to_concat) == 1:
            concat_features = features_to_concat[0]
        else:
            concat_features = torch.cat(features_to_concat, dim=-1)

        # Fusion produces combined output
        fused_output = self.fusion(concat_features) # [batch, num_state_vars * nx]

        # Reshape to separate variables
        # Use -1 for batch size to handle varying batch sizes correctly
        U_B_stacked = fused_output.view(-1, self.num_state_vars, self.nx) # [batch, num_vars, nx]

        return U_B_stacked


# 4.3. 多头注意力模块 (不变)
class MultiHeadAttentionROM(nn.Module):
    # ... (keep implementation from previous response) ...
    def __init__(self, basis_dim, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V):
        batch_size, basis_dim_q, d_model_q = Q.size()
        _, basis_dim_kv, d_model_kv = K.size()
        assert basis_dim_q == basis_dim_kv
        assert d_model_q == d_model_kv
        basis_dim = basis_dim_q
        d_model = d_model_q
        num_heads = self.num_heads
        head_dim = self.head_dim
        Q = Q.view(batch_size, basis_dim, num_heads, head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, basis_dim, num_heads, head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, basis_dim, num_heads, head_dim).permute(0, 2, 1, 3)
        KV = torch.matmul(K.transpose(-2, -1), V)
        z = torch.matmul(Q, KV)
        z = z.permute(0, 2, 1, 3).contiguous().view(batch_size, basis_dim, d_model)
        z = self.out_proj(z)
        return z


# 4.4. Attention-Based ROM: 支持多变量 (修正 W_Q 和 Q 计算)
class MultiVarAttentionROM(nn.Module):
    # <<< MODIFIED: Added use_fixed_lifting flag and related logic
    def __init__(self, state_variable_keys, nx, basis_dim, d_model,
                 bc_state_dim, num_controls, num_heads=8,
                 add_error_estimator=False, shared_attention=False,
                 dropout_lifting=0.1, dropout_ffn=0.1, initial_alpha=0.1,
                 use_fixed_lifting=False): # Added use_fixed_lifting
        """ Args: ... use_fixed_lifting: If True, use fixed linear interpolation for U_B. ... """
        super().__init__()
        self.state_keys = state_variable_keys
        self.num_state_vars = len(state_variable_keys)
        self.nx = nx
        self.basis_dim = basis_dim # Per variable
        self.d_model = d_model
        self.num_heads = num_heads
        self.bc_state_dim = bc_state_dim # Storing for fixed lifting
        self.num_controls = num_controls
        self.add_error_estimator = add_error_estimator
        self.shared_attention = shared_attention
        self.use_fixed_lifting = use_fixed_lifting

        # --- Learnable Bases ---
        self.Phi = nn.ParameterDict()
        for key in self.state_keys:
            phi_param = nn.Parameter(torch.randn(nx, basis_dim))
            nn.init.orthogonal_(phi_param)
            self.Phi[key] = phi_param

        # --- Lifting Network ---
        if not self.use_fixed_lifting:
            self.lifting = UniversalLifting(
                num_state_vars=self.num_state_vars,
                bc_state_dim=bc_state_dim,
                num_controls=num_controls,
                output_dim_per_var=nx,
                nx=nx,
                dropout=dropout_lifting
            )
        else:
            self.lifting = None # No learnable lifting network
            # For fixed lifting, precompute linear interpolation coefficients
            # Shape: [1, 1, nx] for broadcasting with [batch, 1, 1] boundary values
            lin_interp_coeffs = torch.linspace(0, 1, self.nx, dtype=torch.float32)
            self.register_buffer('lin_interp_coeffs', lin_interp_coeffs.view(1, 1, -1))


        # --- Attention & FFN Components ---
        self.W_Q = nn.ModuleDict()
        self.W_K = nn.ModuleDict()
        self.W_V = nn.ModuleDict()
        self.multihead_attn = nn.ModuleDict()
        self.proj_to_coef = nn.ModuleDict()
        self.update_ffn = nn.ModuleDict()
        self.a0_mapping = nn.ModuleDict()
        self.alphas = nn.ParameterDict() # <<< MODIFIED: Added ParameterDict for alpha

        if shared_attention:
            self.W_Q['shared'] = nn.Linear(1, d_model)
            self.W_K['shared'] = nn.Linear(nx, d_model)
            self.W_V['shared'] = nn.Linear(nx, d_model)
            self.multihead_attn['shared'] = MultiHeadAttentionROM(basis_dim, d_model, num_heads)
            self.proj_to_coef['shared'] = nn.Linear(d_model, 1)
            self.update_ffn['shared'] = ImprovedUpdateFFN(basis_dim, hidden_dim=d_model, dropout=dropout_ffn)
            self.a0_mapping['shared'] = nn.Sequential(
                nn.Linear(basis_dim, basis_dim), nn.ReLU(), nn.LayerNorm(basis_dim)
            )
            self.alphas['shared'] = nn.Parameter(torch.tensor(initial_alpha))
        else:
            for key in self.state_keys:
                self.W_Q[key] = nn.Linear(1, d_model)
                self.W_K[key] = nn.Linear(nx, d_model)
                self.W_V[key] = nn.Linear(nx, d_model)
                self.multihead_attn[key] = MultiHeadAttentionROM(basis_dim, d_model, num_heads)
                self.proj_to_coef[key] = nn.Linear(d_model, 1)
                self.update_ffn[key] = ImprovedUpdateFFN(basis_dim, hidden_dim=d_model, dropout=dropout_ffn)
                self.a0_mapping[key] = nn.Sequential(
                     nn.Linear(basis_dim, basis_dim), nn.ReLU(), nn.LayerNorm(basis_dim)
                 )
                self.alphas[key] = nn.Parameter(torch.tensor(initial_alpha))


        # --- Error Estimator ---
        if self.add_error_estimator:
            total_basis_dim = self.num_state_vars * basis_dim
            self.error_estimator = nn.Linear(total_basis_dim, 1)

    def _get_layer(self, module_dict, key):
        return module_dict['shared'] if self.shared_attention else module_dict[key]

    def _get_alpha(self, key):
        return self.alphas['shared'] if self.shared_attention else self.alphas[key]

    def _compute_U_B(self, BC_Ctrl_n):
        """ Computes U_B either via learned lifting or fixed interpolation. """
        if not self.use_fixed_lifting:
            if self.lifting is None: # Should not happen if properly initialized
                raise ValueError("Lifting network is None but use_fixed_lifting is False.")
            return self.lifting(BC_Ctrl_n)
        else:
            # Fixed linear interpolation of boundary values from BC_State
            # Assumes BC_State is the first part of BC_Ctrl_n
            # Assumes for num_state_vars=1, BC_State[:,0] is left_val, BC_State[:,1] is right_val
            # BC_Ctrl_n has shape [batch, bc_state_dim + num_controls]
            # The BC_State part is BC_Ctrl_n[:, :self.bc_state_dim]

            if self.num_state_vars == 1:
                # Ensure bc_state_dim is at least 2 for left and right boundaries
                if self.bc_state_dim < 2:
                    raise ValueError(f"Fixed lifting requires bc_state_dim >= 2 for 1 state var, got {self.bc_state_dim}")

                bc_left_val = BC_Ctrl_n[:, 0:1].unsqueeze(-1) # [batch, 1, 1]
                bc_right_val = BC_Ctrl_n[:, 1:2].unsqueeze(-1) # [batch, 1, 1]
                
                # lin_interp_coeffs is [1, 1, nx]
                U_B_var = bc_left_val * (1 - self.lin_interp_coeffs) + \
                          bc_right_val * self.lin_interp_coeffs
                # U_B_var will have shape [batch, 1, nx]
                return U_B_var # which is U_B_stacked for num_state_vars=1
            else:
                # Generalize for multiple state variables if needed in the future
                # This would require a defined mapping from bc_state_dim components
                # to each variable's boundaries. For now, error out.
                raise NotImplementedError("Fixed lifting for num_state_vars > 1 is not implemented. Assumed num_state_vars = 1 for this ablation.")

    def forward_step(self, a_n_dict, BC_Ctrl_n, params=None):
        batch_size = list(a_n_dict.values())[0].size(0)
        a_next_dict = {}
        U_hat_dict = {}

        # Lifting (Compute once) - NOW USES THE HELPER METHOD
        U_B_stacked = self._compute_U_B(BC_Ctrl_n) # [batch, num_vars, nx]

        for i, key in enumerate(self.state_keys):
            a_n_var = a_n_dict[key] # [batch, basis_dim]
            Phi_var = self.Phi[key] # [nx, basis_dim]

            # Get layers and alpha
            W_Q_var = self._get_layer(self.W_Q, key)
            W_K_var = self._get_layer(self.W_K, key)
            W_V_var = self._get_layer(self.W_V, key)
            attn_var = self._get_layer(self.multihead_attn, key)
            proj_var = self._get_layer(self.proj_to_coef, key)
            ffn_var = self._get_layer(self.update_ffn, key)
            alpha_var = self._get_alpha(key) # <<< MODIFIED: Get alpha

            # --- Prepare K, V (same as before) ---
            Phi_basis = Phi_var.transpose(0, 1).unsqueeze(0).expand(batch_size, -1, -1) # [batch, basis_dim, nx]
            Phi_basis_flat = Phi_basis.reshape(-1, self.nx) # [batch * basis_dim, nx]
            K_flat = W_K_var(Phi_basis_flat); V_flat = W_V_var(Phi_basis_flat)
            K = K_flat.view(batch_size, self.basis_dim, self.d_model)
            V = V_flat.view(batch_size, self.basis_dim, self.d_model)

            # --- Prepare Q (using original method) ---
            a_n_unsq = a_n_var.unsqueeze(-1) # [batch, basis_dim, 1]
            Q_base = W_Q_var(a_n_unsq)      # [batch, basis_dim, d_model]
            ffn_update = ffn_var(a_n_var)          # [batch, basis_dim]
            ffn_term = alpha_var * ffn_update.unsqueeze(-1) # [batch, basis_dim, 1]
            Q = Q_base + ffn_term # [batch, basis_dim, d_model]

            # --- Attention & Update ---
            z = attn_var(Q, K, V)
            z = z / np.sqrt(float(self.d_model))
            z_reshaped = z.reshape(-1, self.d_model)
            a_update_attn = proj_var(z_reshaped).view(batch_size, self.basis_dim)
            a_update = a_update_attn + ffn_update
            a_next_var = a_n_var + a_update
            a_next_dict[key] = a_next_var

            # --- Reconstruction ---
            U_B_var = U_B_stacked[:, i, :].unsqueeze(-1) # U_B_stacked is [batch, num_vars, nx]
            Phi_exp = Phi_var.unsqueeze(0).expand(batch_size, -1, -1)
            a_next_unsq = a_next_var.unsqueeze(-1)
            U_recon = torch.bmm(Phi_exp, a_next_unsq)
            U_hat_dict[key] = U_B_var + U_recon

        # Error Estimation (if enabled)
        err_est = None
        if self.add_error_estimator:
            a_next_combined = torch.cat(list(a_next_dict.values()), dim=-1)
            err_est = self.error_estimator(a_next_combined)

        return a_next_dict, U_hat_dict, err_est

    def forward(self, a0_dict, BC_Ctrl_seq, T, params=None):
        a_current_dict = {}
        for key in self.state_keys:
            a0_map = self._get_layer(self.a0_mapping, key)
            a_current_dict[key] = a0_map(a0_dict[key])
        U_hat_seq_dict = {key: [] for key in self.state_keys}
        err_seq = [] if self.add_error_estimator else None
        for t in range(T):
            BC_Ctrl_n = BC_Ctrl_seq[:, t, :] 
            a_next_dict, U_hat_dict, err_est = self.forward_step(a_current_dict, BC_Ctrl_n, params)
            for key in self.state_keys:
                U_hat_seq_dict[key].append(U_hat_dict[key])
            if self.add_error_estimator:
                err_seq.append(err_est)
            a_current_dict = a_next_dict
        return U_hat_seq_dict, err_seq

    def get_basis(self, key):
        return self.Phi[key]
        

# =============================================================================
# 5. 训练与验证函数 - 支持多变量
# =============================================================================

# =============================================================================
# 5. 训练与验证函数 - 支持多变量
# =============================================================================

def train_multivar_model(model, data_loader, dataset_type, train_nt_target, # Added train_nt_target
                        lr=1e-3, num_epochs=50, device='cuda',
                        checkpoint_path='rom_checkpoint.pt', lambda_res=0.05,
                        lambda_orth=0.001, lambda_bc_penalty=0.01,
                        clip_grad_norm=1.0):
    # ... (optimizer, scheduler, checkpoint loading - no changes here) ...
    model.to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
    mse_loss = nn.MSELoss()
    best_val_loss = float('inf')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    start_epoch = 0
    # --- Checkpoint Loading (simplified) ---
    if os.path.exists(checkpoint_path):
        print(f"Loading existing checkpoint from {checkpoint_path} ...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # Check if use_fixed_lifting matches
            # For simplicity, we'll assume it does for now, or it's a fresh train.
            # A more robust way would be to save args.use_fixed_lifting in checkpoint
            # and error out if model's current use_fixed_lifting doesn't match.
            model.load_state_dict(checkpoint['model_state_dict'], strict=False) # strict=False if ablation changes model structure slightly (e.g. self.lifting is None)
            if 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except:
                    print("Warning: Could not load optimizer state. Reinitializing optimizer.")
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_loss = checkpoint.get('loss', float('inf'))
            print(f"Resuming training from epoch {start_epoch}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting fresh.")


    state_keys = model.state_keys

    # --- Training Loop ---
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        count = 0
        batch_start_time = time.time()

        for i, (state_data, BC_Ctrl_tensor, norm_factors) in enumerate(data_loader):
            if isinstance(state_data, list): 
                state_tensors = [s.to(device) for s in state_data]
                batch_size, nt, nx_data = state_tensors[0].shape 
            else: 
                state_tensors = [state_data.to(device)]
                batch_size, nt, nx_data = state_tensors[0].shape
            BC_Ctrl_tensor = BC_Ctrl_tensor.to(device)
            if nt != train_nt_target:
                raise ValueError(f"Mismatch: nt from DataLoader ({nt}) != train_nt_target ({train_nt_target})")

            optimizer.zero_grad()

            # --- Initial State Projection (per variable) ---
            a0_dict = {}
            BC_ctrl_combined_t0 = BC_Ctrl_tensor[:, 0, :]
            with torch.no_grad():
                 # <<< MODIFIED HERE >>>
                 U_B_stacked_t0 = model._compute_U_B(BC_ctrl_combined_t0)

            for k, key in enumerate(state_keys):
                U0_var = state_tensors[k][:, 0, :].unsqueeze(-1) 
                U_B_t0_var = U_B_stacked_t0[:, k, :].unsqueeze(-1) 
                U0_star_var = U0_var - U_B_t0_var 

                Phi_var = model.get_basis(key) 
                Phi_T_var = Phi_var.transpose(0, 1).unsqueeze(0).expand(batch_size, -1, -1)
                a0_dict[key] = torch.bmm(Phi_T_var, U0_star_var).squeeze(-1) 

            # --- Forward Pass ---
            U_hat_seq_dict, _ = model(a0_dict, BC_Ctrl_tensor, T=train_nt_target, params=None)

            # --- Calculate Loss (sum over variables) ---
            # ... (rest of loss calculation - no changes here) ...
            total_batch_loss = 0.0
            mse_recon_loss = 0.0
            residual_orth_loss = 0.0
            orth_loss = 0.0
            boundary_penalty = 0.0

            for k_var_idx, key_var in enumerate(state_keys):
                Phi_var = model.get_basis(key_var) 
                Phi_T_var = Phi_var.transpose(0, 1).unsqueeze(0).expand(batch_size, -1, -1)
                U_hat_seq_var = U_hat_seq_dict[key_var] 
                U_target_seq_var = state_tensors[k_var_idx] 

                var_mse_loss = 0.0
                var_res_orth_loss = 0.0
                for t in range(train_nt_target):
                    pred = U_hat_seq_var[t] 
                    target = U_target_seq_var[:, t, :].unsqueeze(-1)
                    var_mse_loss += mse_loss(pred, target)

                    if lambda_res > 0:
                        r = target - pred
                        r_proj = torch.bmm(Phi_T_var, r) 
                        var_res_orth_loss += mse_loss(r_proj, torch.zeros_like(r_proj))
                
                mse_recon_loss += (var_mse_loss / train_nt_target) # Corrected: use train_nt_target
                residual_orth_loss += (var_res_orth_loss / train_nt_target) # Corrected: use train_nt_target

                if lambda_orth > 0:
                    PhiT_Phi = torch.matmul(Phi_var.transpose(0, 1), Phi_var)
                    I = torch.eye(model.basis_dim, device=device)
                    orth_loss += torch.norm(PhiT_Phi - I, p='fro')**2
                
                if lambda_bc_penalty > 0:
                    if Phi_var.shape[0] > 1:
                        boundary_penalty += mse_loss(Phi_var[0, :], torch.zeros_like(Phi_var[0, :])) + \
                                            mse_loss(Phi_var[-1, :], torch.zeros_like(Phi_var[-1, :]))
                    else: # Should not happen for nx > 1
                        boundary_penalty += mse_loss(Phi_var[0, :], torch.zeros_like(Phi_var[0, :]))
            
            if len(state_keys) > 0:
                orth_loss /= len(state_keys)
                boundary_penalty /= len(state_keys)

            total_batch_loss = mse_recon_loss + \
                               lambda_res * residual_orth_loss + \
                               lambda_orth * orth_loss + \
                               lambda_bc_penalty * boundary_penalty

            # Backward and optimize
            total_batch_loss.backward()
            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            optimizer.step()

            epoch_loss += total_batch_loss.item()
            count += 1
        # ... (rest of epoch printing, validation, checkpoint saving - no changes here regarding logic) ...
        # ... but checkpoint save_dict should include model's use_fixed_lifting if needed for robust reloading
        # (currently not implemented for brevity, but important for real use)
        avg_epoch_loss = epoch_loss / count if count > 0 else 0.0
        print(f"Epoch {epoch+1}/{num_epochs} finished. Average Training Loss: {avg_epoch_loss:.6f}")

        scheduler.step(avg_epoch_loss)
        if avg_epoch_loss < best_val_loss:
            best_val_loss = avg_epoch_loss
            print(f"Saving checkpoint with loss {best_val_loss:.6f} to {checkpoint_path}")
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'dataset_type': dataset_type, 
                'state_keys': model.state_keys,
                'nx': model.nx,
                'basis_dim': model.basis_dim,
                'd_model': model.d_model,
                'bc_state_dim': model.bc_state_dim, # Use stored bc_state_dim
                'num_controls': model.num_controls,
                'num_heads': model.num_heads,
                'shared_attention': model.shared_attention,
                'use_fixed_lifting': model.use_fixed_lifting # Save the flag
            }
            torch.save(save_dict, checkpoint_path)

    print("Training finished.")
    if os.path.exists(checkpoint_path):
        print(f"Loading best model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Re-instantiate model with correct use_fixed_lifting if loading into a new script run
        # Here, we assume model is already the correct one from the training session
        model.load_state_dict(checkpoint['model_state_dict'])
    return model

# In your model/training script
def validate_multivar_model(model, data_loader, dataset_type, 
                            train_nt_for_model_training: int, 
                            T_value_for_model_training: float, 
                            full_T_in_datafile: float, 
                            full_nt_in_datafile: int, device='cuda',
                            save_fig_path='rom_result.png'):
    # ... (initial setup, test_horizons_T_values - no changes) ...
    model.eval()
    results = {key: {'mse': [], 'rmse': [], 'relative_error': [], 'max_error': []}
               for key in model.state_keys}
    overall_rel_err_T_train_horizon = [] # Renamed for clarity

    test_horizons_T_values = [T_value_for_model_training,
                              T_value_for_model_training + 0.5 * (full_T_in_datafile - T_value_for_model_training),
                              full_T_in_datafile]
    test_horizons_T_values = sorted(list(set(h for h in test_horizons_T_values if h <= full_T_in_datafile and h > 0)))


    print(f"Validation Horizons (T values): {test_horizons_T_values}")
    print(f"Model was trained with nt={train_nt_for_model_training} for T={T_value_for_model_training}")
    print(f"Datafile contains nt={full_nt_in_datafile} for T={full_T_in_datafile}")
    print(f"Model using fixed lifting: {model.use_fixed_lifting}")


    try:
        state_data_full, BC_Ctrl_tensor_full, norm_factors_batch = next(iter(data_loader))
    except StopIteration:
        print("No validation data. Skipping validation.")
        return

    if isinstance(state_data_full, list):
        state_tensors_full = [s[0:1].to(device) for s in state_data_full] 
        batch_size_check, nt_loaded, nx_loaded = state_tensors_full[0].shape
    else:
        state_tensors_full = [state_data_full[0:1].to(device)] 
        batch_size_check, nt_loaded, nx_loaded = state_tensors_full[0].shape
    
    BC_Ctrl_tensor_full_sample = BC_Ctrl_tensor_full[0:1].to(device)

    if nt_loaded != full_nt_in_datafile:
        print(f"Warning: nt from val_loader ({nt_loaded}) != full_nt_in_datafile ({full_nt_in_datafile}). Will use nt_loaded as max.")
        # This can happen if val_dataset was also truncated, or if pkl files are inconsistent.
        # For safety, we should use the minimum of the two as the actual full length.
        # full_nt_in_datafile = min(nt_loaded, full_nt_in_datafile) # Or just use nt_loaded
        # However, UniversalPDEDataset for validation is set to train_nt_limit=None, so it should load the full sequence from file.
        # So, if there's a mismatch, it implies full_nt_in_datafile parameter to this function might be wrong.
        # Let's assume nt_loaded is the ground truth available from this sample.
        # For calculation of nt_for_this_horizon, we still need a reference full_T and full_nt.

    norm_factors_sample = {}
    # ... (norm_factors_sample extraction - no changes) ...
    for key_nf, val_tensor_nf in norm_factors_batch.items():
        if isinstance(val_tensor_nf, torch.Tensor) or isinstance(val_tensor_nf, np.ndarray):
            if val_tensor_nf.ndim > 0 : 
                norm_factors_sample[key_nf] = val_tensor_nf[0] 
            else: 
                norm_factors_sample[key_nf] = val_tensor_nf
        else: 
            norm_factors_sample[key_nf] = val_tensor_nf


    state_keys = model.state_keys
    current_batch_size = 1 

    # --- Calculate initial coefficients a0_dict ---
    a0_dict = {}
    BC0_full = BC_Ctrl_tensor_full_sample[:, 0, :]
    # <<< MODIFIED HERE >>>
    U_B0_lifted = model._compute_U_B(BC0_full) 
    for k_var_idx, key in enumerate(state_keys):
        U0_full_var = state_tensors_full[k_var_idx][:, 0, :].unsqueeze(-1) 
        U_B0_var = U_B0_lifted[:, k_var_idx, :].unsqueeze(-1)
        Phi = model.get_basis(key).to(device)
        Phi_T = Phi.transpose(0, 1).unsqueeze(0)
        a0 = torch.bmm(Phi_T, U0_full_var - U_B0_var).squeeze(-1) 
        a0_dict[key] = a0

    # --- Loop through each test horizon ---
    # ... (rest of validation loop, plotting, metrics - no changes in core logic) ...
    # ... but save_fig_path should be unique for the ablation run (handled in main) ...
    for T_test_horizon in test_horizons_T_values:
        # nt_for_this_horizon = int((T_test_horizon / full_T_in_datafile) * (full_nt_in_datafile - 1)) + 1
        # nt_for_this_horizon = min(nt_for_this_horizon, nt_loaded) # Cap at available data from loader
        
        # Use the nt_loaded which is the actual number of time points in the validation sample
        # if UniversalPDEDataset (val) is set with train_nt_limit=None, nt_loaded is the full sequence length from file
        # So, full_nt_in_datafile passed to this function should ideally match nt_loaded.
        if full_nt_in_datafile <= 1 : # Avoid division by zero or meaningless ratio
             nt_for_this_horizon = nt_loaded
        else:
             nt_for_this_horizon = int((T_test_horizon / full_T_in_datafile) * (nt_loaded -1)) +1
        nt_for_this_horizon = min(nt_for_this_horizon, nt_loaded) # Ensure it doesn't exceed loaded data

        print(f"\n--- Validating for T_horizon = {T_test_horizon:.2f} (nt = {nt_for_this_horizon}) ---")

        BC_seq_for_pred = BC_Ctrl_tensor_full_sample[:, :nt_for_this_horizon, :]
        U_hat_seq_dict, _ = model(a0_dict, BC_seq_for_pred, T=nt_for_this_horizon) # Pass nt_for_this_horizon

        combined_pred_denorm_list = [] # Use list for appending full sequences
        combined_gt_denorm_list = []

        num_vars_plot = len(state_keys)
        # Ensure at least one row of subplots even if num_vars_plot is 0 (though it shouldn't be)
        fig_rows = max(1, num_vars_plot)
        fig, axs = plt.subplots(fig_rows, 3, figsize=(18, 5 * fig_rows), squeeze=False)
        
        # Determine L for plotting from dataset_params if available, else default
        # This part was missing in the original, add it for completeness
        # L_vis = dataset_params_for_plot.get('L', 1.0) if 'dataset_params_for_plot' in globals() else 1.0
        L_vis = 1.0 # Placeholder, should get from actual data params if varied

        for k_var_idx, key_var in enumerate(state_keys):
            pred_norm_stacked = torch.cat(U_hat_seq_dict[key_var], dim=0) 
            pred_norm_reshaped = pred_norm_stacked.view(nt_for_this_horizon, current_batch_size, nx_loaded)
            pred_norm_final = pred_norm_reshaped.squeeze(1).detach().cpu().numpy()

            mean_k_val = norm_factors_sample.get(f'{key_var}_mean', 0.0) # Default to 0 if not found
            std_k_val = norm_factors_sample.get(f'{key_var}_std', 1.0)   # Default to 1 if not found
            
            mean_k = mean_k_val.item() if hasattr(mean_k_val, 'item') else mean_k_val
            std_k = std_k_val.item() if hasattr(std_k_val, 'item') else std_k_val
            pred_denorm = pred_norm_final * std_k + mean_k

            gt_norm_full_var = state_tensors_full[k_var_idx].squeeze(0).cpu().numpy()
            gt_norm_sliced = gt_norm_full_var[:nt_for_this_horizon, :]
            gt_denorm = gt_norm_sliced * std_k + mean_k
            
            combined_pred_denorm_list.append(pred_denorm) # Store the 2D array
            combined_gt_denorm_list.append(gt_denorm)   # Store the 2D array

            mse_k = np.mean((pred_denorm - gt_denorm)**2)
            rmse_k = np.sqrt(mse_k)
            rel_err_k = np.linalg.norm(pred_denorm - gt_denorm, 'fro') / (np.linalg.norm(gt_denorm, 'fro') + 1e-10)
            max_err_k = np.max(np.abs(pred_denorm - gt_denorm))

            print(f"  '{key_var}': MSE={mse_k:.3e}, RMSE={rmse_k:.3e}, RelErr={rel_err_k:.3e}, MaxErr={max_err_k:.3e}")

            if abs(T_test_horizon - T_value_for_model_training) < 1e-5: # If it's the training T
                results[key_var]['mse'].append(mse_k)
                results[key_var]['rmse'].append(rmse_k)
                results[key_var]['relative_error'].append(rel_err_k)
                results[key_var]['max_error'].append(max_err_k)
            
            # Plotting
            diff_plot = np.abs(pred_denorm - gt_denorm)
            vmin_plot = min(gt_denorm.min(), pred_denorm.min())
            vmax_plot = max(gt_denorm.max(), pred_denorm.max())

            im0 = axs[k_var_idx,0].imshow(gt_denorm, aspect='auto', origin='lower', vmin=vmin_plot, vmax=vmax_plot, extent=[0, L_vis, 0, T_test_horizon], cmap='viridis')
            axs[k_var_idx,0].set_title(f"GT ({key_var}) T={T_test_horizon:.1f}"); axs[k_var_idx,0].set_ylabel("t")
            plt.colorbar(im0, ax=axs[k_var_idx,0])

            im1 = axs[k_var_idx,1].imshow(pred_denorm, aspect='auto', origin='lower', vmin=vmin_plot, vmax=vmax_plot, extent=[0, L_vis, 0, T_test_horizon], cmap='viridis')
            axs[k_var_idx,1].set_title(f"Pred ({key_var}) T={T_test_horizon:.1f}")
            plt.colorbar(im1, ax=axs[k_var_idx,1])

            im2 = axs[k_var_idx,2].imshow(diff_plot, aspect='auto', origin='lower', extent=[0, L_vis, 0, T_test_horizon], cmap='magma')
            axs[k_var_idx,2].set_title(f"Error ({key_var}) (Max {max_err_k:.2e})")
            plt.colorbar(im2, ax=axs[k_var_idx,2])
            for j_plot in range(3): axs[k_var_idx,j_plot].set_xlabel("x")

        # Calculate overall relative error for this horizon by concatenating all variables' data
        # Ensure all elements in the list are numpy arrays before concatenating
        if combined_pred_denorm_list and combined_gt_denorm_list:
            all_preds_flat = np.concatenate([p.flatten() for p in combined_pred_denorm_list])
            all_gts_flat = np.concatenate([g.flatten() for g in combined_gt_denorm_list])
            
            overall_rel_err_horizon = np.linalg.norm(all_preds_flat - all_gts_flat) / \
                                      (np.linalg.norm(all_gts_flat) + 1e-10)
            print(f"  Overall RelErr for T={T_test_horizon:.1f}: {overall_rel_err_horizon:.3e}")
            if abs(T_test_horizon - T_value_for_model_training) < 1e-5:
                overall_rel_err_T_train_horizon.append(overall_rel_err_horizon)
        else:
            print(f"  Could not compute Overall RelErr for T={T_test_horizon:.1f} (no data).")


        fig.suptitle(f"Validation @ T={T_test_horizon:.1f} ({dataset_type.upper()}) — basis={model.basis_dim}, d_model={model.d_model}, FixedLift={model.use_fixed_lifting}")
        fig.tight_layout(rect=[0,0.03,1,0.95])
        horizon_fig_path = save_fig_path.replace('.png', f'_T{str(T_test_horizon).replace(".","p")}.png')
        plt.savefig(horizon_fig_path)
        print(f"Saved validation figure to: {horizon_fig_path}")
        plt.close(fig) # Close the figure to free memory


    print(f"\n--- Validation Summary (Metrics for T={T_value_for_model_training:.1f}) ---")
    # ... (rest of summary printing - no changes) ...
    for key_sum in state_keys:
        if results[key_sum]['mse']: # Check if results were appended
            avg_mse = np.mean(results[key_sum]['mse'])
            avg_rmse = np.mean(results[key_sum]['rmse'])
            avg_rel = np.mean(results[key_sum]['relative_error'])
            avg_max = np.mean(results[key_sum]['max_error'])
            print(f"  {key_sum}: MSE={avg_mse:.4e}, RMSE={avg_rmse:.4e}, RelErr={avg_rel:.4e}, MaxErr={avg_max:.4e}")
    if overall_rel_err_T_train_horizon: # Use the renamed variable
        print(f"Overall Avg RelErr for T={T_value_for_model_training:.1f}: {np.mean(overall_rel_err_T_train_horizon):.4e}")
        
# =============================================================================
# 6. 主流程 - 可选择数据集类型
# =============================================================================
# =============================================================================
# 6. 主流程 - 可选择数据集类型
# =============================================================================
import pickle
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate PDE datasets with complex BCs.")

    parser.add_argument('--datatype', type=str, required=True,
                        choices=['heat_delayed_feedback',
                                 'reaction_diffusion_neumann_feedback',
                                 'heat_nonlinear_feedback_gain', 'convdiff'], 
                        help='Type of dataset to use.')
    # <<< ADDED ARGUMENT FOR ABLATION >>>
    parser.add_argument('--use_fixed_lifting', action='store_true',
                        help='Use fixed linear interpolation for U_B instead of learned lifting.')
    args = parser.parse_args()

    DATASET_TYPE = args.datatype 
    USE_FIXED_LIFTING_ABLATION = args.use_fixed_lifting # Store the flag
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Selected Dataset Type: {DATASET_TYPE.upper()}")
    if USE_FIXED_LIFTING_ABLATION:
        print("ABLATION STUDY: Using FIXED Lifting (Linear Interpolation for U_B)")


    # --- Key Time Parameters ---
    # ... (no changes here) ...
    if DATASET_TYPE in ['heat_delayed_feedback', 'reaction_diffusion_neumann_feedback', 'heat_nonlinear_feedback_gain', 'convdiff']:
        FULL_T_IN_DATAFILE = 2.0
        FULL_NT_IN_DATAFILE = 300 
    else: 
        FULL_T_IN_DATAFILE = 2.0
        FULL_NT_IN_DATAFILE = 600
        
    if DATASET_TYPE in ['heat_delayed_feedback', 'reaction_diffusion_neumann_feedback', 'heat_nonlinear_feedback_gain', 'convdiff']:
        TRAIN_T_TARGET = 1.5 
    else:
        TRAIN_T_TARGET = 1.0 
    
    if FULL_NT_IN_DATAFILE <= 1: # Avoid issues if nt is 0 or 1
        TRAIN_NT_FOR_MODEL = FULL_NT_IN_DATAFILE
    else:
        TRAIN_NT_FOR_MODEL = int((TRAIN_T_TARGET / FULL_T_IN_DATAFILE) * (FULL_NT_IN_DATAFILE - 1)) + 1


    print(f"Full data T={FULL_T_IN_DATAFILE}, nt={FULL_NT_IN_DATAFILE}")
    print(f"Training will use T={TRAIN_T_TARGET}, nt={TRAIN_NT_FOR_MODEL}")

    # --- Common Parameters ---
    # ... (no changes here) ...
    basis_dim = 32
    d_model = 512
    num_heads = 8
    add_error_estimator = True 
    shared_attention = False
    batch_size = 32 # Adjusted for potentially larger models/memory
    num_epochs = 150
    learning_rate = 1e-4
    lambda_res = 0.05
    lambda_orth = 0.001
    lambda_bc_penalty = 0.01
    clip_grad_norm = 1.0

    # Paths
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # <<< MODIFIED run_name for ablation >>>
    run_name_suffix = "_fixedlift" if USE_FIXED_LIFTING_ABLATION else ""
    run_name = f"{DATASET_TYPE}_b{basis_dim}_d{d_model}{run_name_suffix}"
    
    checkpoint_dir = f"./New_ckpt_2/_checkpoints_{DATASET_TYPE}"
    results_dir = f"./result_all_2/results_{DATASET_TYPE}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'barom_{run_name}.pt')
    save_fig_path = os.path.join(results_dir, f'barom_result_{run_name}.png')
    basis_dir = os.path.join(checkpoint_dir, 'pod_bases') # Bases are independent of lifting type
    os.makedirs(basis_dir, exist_ok=True)


    # --- Dataset Specific Parameters & Loading ---
    # ... (dataset path definitions - no changes here, but ensure num_controls matches dataset) ...
    if DATASET_TYPE == 'heat_delayed_feedback':
        dataset_path = "./datasets_new_feedback/heat_delayed_feedback_v1_5000s_64nx_300nt.pkl" 
        nx_data = 64; L_data = 1.0 
        state_keys = ['U']; num_state_vars = 1
        num_controls_param = 2 # This refers to BC_Control inputs, not directly to bc_state_dim for UniversalLifting
    elif DATASET_TYPE == 'reaction_diffusion_neumann_feedback':
        dataset_path = "./datasets_new_feedback/reaction_diffusion_neumann_feedback_v1_5000s_64nx_300nt.pkl"
        nx_data = 64; L_data = 1.0 
        state_keys = ['U']; num_state_vars = 1
        num_controls_param = 2
    elif DATASET_TYPE == 'heat_nonlinear_feedback_gain':
        dataset_path = "./datasets_new_feedback/heat_nonlinear_feedback_gain_v1_5000s_64nx_300nt.pkl"
        nx_data = 64; L_data = 1.0
        state_keys = ['U']; num_state_vars = 1
        num_controls_param = 2 # To be confirmed by dataset.num_controls later
    elif DATASET_TYPE == 'convdiff':
        dataset_path = "./datasets_new_feedback/convdiff_v1_5000s_64nx_300nt.pkl"
        nx_data = 64; L_data = 1.0
        state_keys = ['U']
        num_state_vars = 1
        num_controls_param = 2
    else:
        raise ValueError(f"Unknown dataset_type: {DATASET_TYPE}")

    dataset_params_for_plot = {'nx': nx_data, 'ny': 1, 'L': L_data, 'T': FULL_T_IN_DATAFILE}
    print(f"Loading dataset for {DATASET_TYPE} from {dataset_path}")
    try:
        with open(dataset_path, 'rb') as f:
            data_list = pickle.load(f)
        print(f"Loaded {len(data_list)} samples.")
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_path}")
        exit()
        
    if not data_list: print("No data generated, exiting."); exit()

    # This bc_state_dim is specific to how UniversalPDEDataset is structured.
    # It is the dimension of the 'BC_State' part of the concatenated BC_Ctrl tensor.
    # For the fixed lifting, we assume the first two components of this are left/right boundary values.
    bc_state_dim_from_dataset = data_list[0]['BC_State'].shape[1] # Infer from data
    print(f"Inferred bc_state_dim from data: {bc_state_dim_from_dataset}")
    # If your fixed lifting always assumes 2 (left/right), ensure this is compatible.
    # For learnable lifting, UniversalLifting will use this bc_state_dim_from_dataset.


    # --- Data Splitting & Loading ---
    random.shuffle(data_list)
    n_total = len(data_list); n_train = int(0.8 * n_total)
    train_data_list = data_list[:n_train]; val_data_list = data_list[n_train:]
    print(f"Train samples: {len(train_data_list)}, Validation samples: {len(val_data_list)}")

    train_dataset = UniversalPDEDataset(train_data_list, dataset_type=DATASET_TYPE,
                                        train_nt_limit=TRAIN_NT_FOR_MODEL)
    val_dataset = UniversalPDEDataset(val_data_list, dataset_type=DATASET_TYPE,
                                      train_nt_limit=None) 
    num_workers = 1 # Set to 0 if issues with multiprocessing, else 1 or more
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers) # val_loader batch_size=1 for plotting

    # Use nx and num_controls from the instantiated dataset object for model creation
    current_nx_model = train_dataset.nx 
    model_num_controls = train_dataset.num_controls # This is the number of external control signals
    model_bc_state_dim = train_dataset.bc_state_dim # This is dim of BC_State (e.g., left/right values)

    print(f"Model Configuration: nx={current_nx_model}, basis_dim={basis_dim}, d_model={d_model}")
    print(f"Model BC_State dim: {model_bc_state_dim}, Model Num Controls: {model_num_controls}")


    # --- Model Initialization ---
    online_model = MultiVarAttentionROM(
        state_variable_keys=state_keys, nx=current_nx_model, basis_dim=basis_dim,
        d_model=d_model, 
        bc_state_dim=model_bc_state_dim, # Pass the dimension of the state part of BC
        num_controls=model_num_controls,    # Pass the dimension of the control part of BC
        num_heads=num_heads, add_error_estimator=add_error_estimator,
        shared_attention=shared_attention,
        use_fixed_lifting=USE_FIXED_LIFTING_ABLATION # <<< Pass the ablation flag
    )

    # --- POD Basis Initialization (per variable) ---
    # ... (POD basis loading/computation - no changes needed here as it's independent of lifting type) ...
    print("\nInitializing POD bases...")
    pod_bases = {}
    if not train_dataset:
        print("Error: Training dataset is empty. Cannot compute POD.")
        exit()

    try:
        first_sample_data = train_dataset[0]
        first_state_tensor = first_sample_data[0][0] 
        actual_nt_for_pod = first_state_tensor.shape[0] # This is TRAIN_NT_FOR_MODEL
    except IndexError:
        print("Error: Could not access shape from the first sample in train_dataset.")
        exit()

    for key_pod in state_keys:
        # Basis path uses current_nx_model which is train_dataset.nx
        basis_path = os.path.join(basis_dir, f'pod_basis_{key_pod}_nx{current_nx_model}_nt{actual_nt_for_pod}_bdim{basis_dim}_fixedlifting.npy') 
        loaded_basis = None
        if os.path.exists(basis_path):
            print(f"  Loading existing POD basis for '{key_pod}' from {basis_path}...")
            try:
                loaded_basis = np.load(basis_path)
            except Exception as e:
                print(f"  Error loading {basis_path}: {e}. Will recompute.")
                loaded_basis = None
            if loaded_basis is not None and loaded_basis.shape != (current_nx_model, basis_dim):
                print(f"  Shape mismatch for loaded basis '{key_pod}'. Expected ({current_nx_model}, {basis_dim}), got {loaded_basis.shape}. Recomputing.")
                loaded_basis = None

        if loaded_basis is None:
            print(f"  Computing POD basis for '{key_pod}' (using nt={actual_nt_for_pod} from train_dataset)...")
            computed_basis = compute_pod_basis_generic(
                data_list=train_data_list, 
                dataset_type=DATASET_TYPE, state_variable_key=key_pod,
                nx=current_nx_model, nt=actual_nt_for_pod, # Use nt from truncated train_dataset
                basis_dim=basis_dim
            )
            if computed_basis is not None:
                pod_bases[key_pod] = computed_basis
                os.makedirs(os.path.dirname(basis_path), exist_ok=True)
                np.save(basis_path, computed_basis)
                print(f"  Saved computed POD basis for '{key_pod}' to {basis_path}")
            else:
                print(f"ERROR: Failed to compute POD basis for '{key_pod}'. Exiting.")
                exit()
        else:
            pod_bases[key_pod] = loaded_basis

    with torch.no_grad():
        for key_phi in state_keys:
            if key_phi in pod_bases and key_phi in online_model.Phi:
                model_phi = online_model.Phi[key_phi]; pod_phi_np = pod_bases[key_phi]
                # Ensure pod_phi_np is float32 before converting to tensor
                pod_phi = torch.tensor(pod_phi_np.astype(np.float32))
                if model_phi.shape == pod_phi.shape: 
                    model_phi.copy_(pod_phi)
                    print(f"  Initialized Phi for '{key_phi}' with POD.")
                else: 
                    print(f"  WARNING: Shape mismatch for Phi '{key_phi}'. Expected {model_phi.shape}, got {pod_phi.shape}. Using random init.")
            else: 
                print(f"  WARNING: No POD basis found or Phi module not present for '{key_phi}'. Using random init.")


    # --- Training ---
    print(f"\nStarting training for {DATASET_TYPE.upper()}{' (Fixed Lifting)' if USE_FIXED_LIFTING_ABLATION else ''}...")
    start_train_time = time.time()
    online_model = train_multivar_model(
        online_model, train_loader, dataset_type=DATASET_TYPE,
        train_nt_target=TRAIN_NT_FOR_MODEL, 
        lr=learning_rate, num_epochs=num_epochs, device=device,
        checkpoint_path=checkpoint_path, lambda_res=lambda_res,
        lambda_orth=lambda_orth, lambda_bc_penalty=lambda_bc_penalty,
        clip_grad_norm=clip_grad_norm
    )
    end_train_time = time.time()
    print(f"Training took {end_train_time - start_train_time:.2f} seconds.")

    # --- Validation ---
    if val_data_list: 
        print(f"\nStarting validation for {DATASET_TYPE.upper()}{' (Fixed Lifting)' if USE_FIXED_LIFTING_ABLATION else ''}...")
        validate_multivar_model(
            online_model, val_loader, dataset_type=DATASET_TYPE, device=device,
            save_fig_path=save_fig_path,
            train_nt_for_model_training=TRAIN_NT_FOR_MODEL,
            T_value_for_model_training=TRAIN_T_TARGET,
            full_T_in_datafile=FULL_T_IN_DATAFILE,
            full_nt_in_datafile=FULL_NT_IN_DATAFILE 
        )
    else:
        print("\nNo validation data. Skipping validation.")

    print("="*60)
    print(f"Run finished for dataset: {DATASET_TYPE.upper()} - {run_name}")
    print(f"Final checkpoint saved to: {checkpoint_path}")
    if val_data_list: print(f"Validation figure(s) saved with prefix: {save_fig_path.replace('.png','')}")
    print("="*60)