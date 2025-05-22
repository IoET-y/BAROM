
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
# fixed seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

class UniversalPDEDataset(Dataset):
    def __init__(self, data_list, dataset_type, train_nt_limit=None): # Added train_nt_limit
        """

        Args:
            data_list: 。
            train_nt_limit: If specified, truncate sequences to this length for training.
        """
        if not data_list:
            raise ValueError("data_list cannot be empty")
        self.data_list = data_list
        self.dataset_type = dataset_type
        self.train_nt_limit = train_nt_limit # Store the limit

        first_sample = data_list[0]
        params = first_sample.get('params', {})


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
            self.nt_from_sample    = first_sample['U'].shape[0]
            self.nx_from_sample    = first_sample['U'].shape[1]
            self.state_keys        = ['U']
            self.num_state_vars    = 1
            self.nx                = self.nx_from_sample
            self.ny                = 1
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

            state_mean = np.mean(state_seq)
            state_std = np.std(state_seq) + 1e-8
            state_norm = (state_seq - state_mean) / state_std

            state_tensors_norm_list.append(torch.tensor(state_norm).float())
            norm_factors[f'{key}_mean'] = state_mean
            norm_factors[f'{key}_std'] = state_std

        bc_state_seq_full = sample[self.bc_state_key]
        bc_state_seq = bc_state_seq_full[:current_nt, :] # Truncate

        if bc_state_seq.shape[0] != current_nt:
            raise ValueError(f"Time dimension mismatch for BC_State. Expected {current_nt}, got {bc_state_seq.shape[0]}")

        bc_state_norm = np.zeros_like(bc_state_seq, dtype=np.float32)
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


# 4.1. Feedforward
class ImprovedUpdateFFN(nn.Module):
     # ... (keep implementation from previous response) ...
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

# 4.2. Lifting
class UniversalLifting(nn.Module):
    # <<< MODIFIED: Increased default capacity and added fusion layer
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
             # Handle case with no BCs/Controls? Should likely not happen based on problem.
             # Maybe raise error or return zeros? Assuming at least one input type exists.
             # For safety, create a zero tensor of expected fusion input dim if needed
             # This part needs careful consideration based on expected use cases.
             # Let's assume fusion_input_dim > 0
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


# 4.3. MultiHeadAttentionROM
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


# 4.4. Attention-Based ROM
class MultiVarAttentionROM(nn.Module):
    # <<< MODIFIED: Reverted W_Q, added alpha, corrected Q calculation
    def __init__(self, state_variable_keys, nx, basis_dim, d_model,
                 bc_state_dim, num_controls, num_heads=8,
                 add_error_estimator=False, shared_attention=False,
                 dropout_lifting=0.1, dropout_ffn=0.1, initial_alpha=0.1): # Added initial_alpha
        """ Args: ... (omitted for brevity, same as before) ... """
        super().__init__()
        self.state_keys = state_variable_keys
        self.num_state_vars = len(state_variable_keys)
        self.nx = nx
        self.basis_dim = basis_dim # Per variable
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_controls = num_controls
        self.add_error_estimator = add_error_estimator
        self.shared_attention = shared_attention

        # --- Learnable Bases ---
        self.Phi = nn.ParameterDict()
        for key in self.state_keys:
            phi_param = nn.Parameter(torch.randn(nx, basis_dim))
            nn.init.orthogonal_(phi_param)
            self.Phi[key] = phi_param

        # --- Lifting Network ---
        self.lifting = UniversalLifting(
            num_state_vars=self.num_state_vars,
            bc_state_dim=bc_state_dim,
            num_controls=num_controls,
            output_dim_per_var=nx,
            nx=nx,
            dropout=dropout_lifting
        )

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
            # <<< MODIFIED: W_Q input dim is 1
            self.W_Q['shared'] = nn.Linear(1, d_model)
            self.W_K['shared'] = nn.Linear(nx, d_model)
            self.W_V['shared'] = nn.Linear(nx, d_model)
            self.multihead_attn['shared'] = MultiHeadAttentionROM(basis_dim, d_model, num_heads)
            self.proj_to_coef['shared'] = nn.Linear(d_model, 1)
            self.update_ffn['shared'] = ImprovedUpdateFFN(basis_dim, hidden_dim=d_model, dropout=dropout_ffn)
            self.a0_mapping['shared'] = nn.Sequential(
                nn.Linear(basis_dim, basis_dim), nn.ReLU(), nn.LayerNorm(basis_dim)
            )
            # <<< MODIFIED: Add shared alpha
            self.alphas['shared'] = nn.Parameter(torch.tensor(initial_alpha))
        else:
            for key in self.state_keys:
                # <<< MODIFIED: W_Q input dim is 1
                self.W_Q[key] = nn.Linear(1, d_model)
                self.W_K[key] = nn.Linear(nx, d_model)
                self.W_V[key] = nn.Linear(nx, d_model)
                self.multihead_attn[key] = MultiHeadAttentionROM(basis_dim, d_model, num_heads)
                self.proj_to_coef[key] = nn.Linear(d_model, 1)
                self.update_ffn[key] = ImprovedUpdateFFN(basis_dim, hidden_dim=d_model, dropout=dropout_ffn)
                self.a0_mapping[key] = nn.Sequential(
                     nn.Linear(basis_dim, basis_dim), nn.ReLU(), nn.LayerNorm(basis_dim)
                 )
                # <<< MODIFIED: Add alpha per variable
                self.alphas[key] = nn.Parameter(torch.tensor(initial_alpha))


        # --- Error Estimator ---
        if self.add_error_estimator:
            total_basis_dim = self.num_state_vars * basis_dim
            self.error_estimator = nn.Linear(total_basis_dim, 1)

    def _get_layer(self, module_dict, key):
        return module_dict['shared'] if self.shared_attention else module_dict[key]

    def _get_alpha(self, key):
        """ Helper to get shared or specific alpha parameter. """
        return self.alphas['shared'] if self.shared_attention else self.alphas[key]


    def forward_step(self, a_n_dict, BC_Ctrl_n, params=None):
        batch_size = list(a_n_dict.values())[0].size(0)
        a_next_dict = {}
        U_hat_dict = {}

        # Lifting (Compute once)
        U_B_stacked = self.lifting(BC_Ctrl_n) # [batch, num_vars, nx]

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
            # <<< MODIFIED: Reverted Q calculation
            # Apply W_Q (Linear(1, d_model)) to each coefficient independently
            a_n_unsq = a_n_var.unsqueeze(-1) # [batch, basis_dim, 1]
            Q_base = W_Q_var(a_n_unsq)      # [batch, basis_dim, d_model]

            # Calculate FFN update term
            ffn_update = ffn_var(a_n_var)          # [batch, basis_dim]
            ffn_term = alpha_var * ffn_update.unsqueeze(-1) # [batch, basis_dim, 1]

            # Add FFN term to base Q (broadcasting the last dim of ffn_term)
            Q = Q_base + ffn_term # [batch, basis_dim, d_model]

            # --- Attention & Update ---
            z = attn_var(Q, K, V)
            z = z / np.sqrt(float(self.d_model))
            z_reshaped = z.reshape(-1, self.d_model)
            a_update_attn = proj_var(z_reshaped).view(batch_size, self.basis_dim)

            # Combine updates (Attention result + FFN result directly - as per original logic)
            # Note: ffn_update was already computed for Q. Re-use it.
            a_update = a_update_attn + ffn_update
            a_next_var = a_n_var + a_update
            a_next_dict[key] = a_next_var

            # --- Reconstruction ---
            U_B_var = U_B_stacked[:, i, :].unsqueeze(-1)
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
        # Store sequence of predictions (dictionaries)
        U_hat_seq_dict = {key: [] for key in self.state_keys}
        err_seq = [] if self.add_error_estimator else None
        for t in range(T):
            BC_Ctrl_n = BC_Ctrl_seq[:, t, :] # Get current BC + Control
            a_next_dict, U_hat_dict, err_est = self.forward_step(a_current_dict, BC_Ctrl_n, params)
            # Store results
            for key in self.state_keys:
                U_hat_seq_dict[key].append(U_hat_dict[key])
            if self.add_error_estimator:
                err_seq.append(err_est)
            # Update state
            a_current_dict = a_next_dict
        return U_hat_seq_dict, err_seq

    def get_basis(self, key):
        return self.Phi[key]
 

def train_multivar_model(model, data_loader, dataset_type, train_nt_target, # Added train_nt_target
                        lr=1e-3, num_epochs=50, device='cuda',
                        checkpoint_path='rom_checkpoint.pt', lambda_res=0.05,
                        lambda_orth=0.001, lambda_bc_penalty=0.01,
                        clip_grad_norm=1.0):
    # <<< MODIFIED: Handles multi-variable training
    model.to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
    mse_loss = nn.MSELoss()
    best_val_loss = float('inf')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"Loading existing checkpoint from {checkpoint_path} ...")
        # Basic loading, assumes compatibility
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            # Try loading optimizer state, ignore if error
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

            # Move data to device
            if isinstance(state_data, list): # Euler returns list [rho, u]
                state_tensors = [s.to(device) for s in state_data]
                batch_size, nt, nx = state_tensors[0].shape # Get dims from first var
            else: # Advection returns single tensor U
                state_tensors = [state_data.to(device)]
                batch_size, nt, nx = state_tensors[0].shape
            BC_Ctrl_tensor = BC_Ctrl_tensor.to(device)
            # Ensure nt_from_loader matches train_nt_target
            if nt != train_nt_target:
                raise ValueError(f"Mismatch: nt from DataLoader ({nt}) != train_nt_target ({train_nt_target})")

            optimizer.zero_grad()

            # --- Initial State Projection (per variable) ---
            a0_dict = {}
            # Get BC+Ctrl at t=0
            BC_ctrl_combined_t0 = BC_Ctrl_tensor[:, 0, :]
            with torch.no_grad():
                 # U_B_stacked_t0: [batch, num_vars, nx]
                 U_B_stacked_t0 = model.lifting(BC_ctrl_combined_t0)

            for k, key in enumerate(state_keys):
                U0_var = state_tensors[k][:, 0, :].unsqueeze(-1) # [batch, nx, 1]
                U_B_t0_var = U_B_stacked_t0[:, k, :].unsqueeze(-1) # [batch, nx, 1]
                U0_star_var = U0_var - U_B_t0_var # Modified initial snapshot

                Phi_var = model.get_basis(key) # [nx, basis_dim]
                Phi_T_var = Phi_var.transpose(0, 1).unsqueeze(0).expand(batch_size, -1, -1)
                a0_dict[key] = torch.bmm(Phi_T_var, U0_star_var).squeeze(-1) # [batch, basis_dim]

            # --- Forward Pass ---
            U_hat_seq_dict, _ = model(a0_dict, BC_Ctrl_tensor, T=train_nt_target, params=None)

            total_batch_loss = 0.0
            mse_recon_loss = 0.0
            residual_orth_loss = 0.0
            orth_loss = 0.0
            boundary_penalty = 0.0

            # Loop through each variable
            for k, key in enumerate(state_keys):
                Phi_var = model.get_basis(key) # Basis for this variable
                Phi_T_var = Phi_var.transpose(0, 1).unsqueeze(0).expand(batch_size, -1, -1)
                U_hat_seq_var = U_hat_seq_dict[key] # List of [batch, nx, 1]
                U_target_seq_var = state_tensors[k] # [batch, nt, nx]

                # Accumulate losses per variable over time
                var_mse_loss = 0.0
                var_res_orth_loss = 0.0
                for t in range(train_nt_target):
                    pred = U_hat_seq_var[t] # [batch, nx, 1]
                    target = U_target_seq_var[:, t, :].unsqueeze(-1) # [batch, nx, 1]
                    var_mse_loss += mse_loss(pred, target)

                    if lambda_res > 0:
                        r = target - pred
                        r_proj = torch.bmm(Phi_T_var, r) # [batch, basis_dim, 1]
                        var_res_orth_loss += mse_loss(r_proj, torch.zeros_like(r_proj))

                mse_recon_loss += (var_mse_loss / nt)
                residual_orth_loss += (var_res_orth_loss / nt)

                # Basis constraints (calculated once per variable)
                if lambda_orth > 0:
                    PhiT_Phi = torch.matmul(Phi_var.transpose(0, 1), Phi_var)
                    I = torch.eye(model.basis_dim, device=device)
                    orth_loss += torch.norm(PhiT_Phi - I, p='fro')**2

                if lambda_bc_penalty > 0:
                    if Phi_var.shape[0] > 1:
                        boundary_penalty += mse_loss(Phi_var[0, :], torch.zeros_like(Phi_var[0, :])) + \
                                            mse_loss(Phi_var[-1, :], torch.zeros_like(Phi_var[-1, :]))
                    else:
                        boundary_penalty += mse_loss(Phi_var[0, :], torch.zeros_like(Phi_var[0, :]))

            # Average the penalties over number of variables
            orth_loss /= len(state_keys)
            boundary_penalty /= len(state_keys)


            # Combine all losses
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

            # Print progress
            # if (i + 1) % 50 == 0:
            #      batch_time = time.time() - batch_start_time
            #      print(f"  Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(data_loader)}, "
            #            f"Batch Loss: {total_batch_loss.item():.4e}, Time/Batch: {batch_time/50:.3f}s")
            #      batch_start_time = time.time()


        avg_epoch_loss = epoch_loss / count
        print(f"Epoch {epoch+1}/{num_epochs} finished. Average Training Loss: {avg_epoch_loss:.6f}")

        # --- Validation & Checkpointing (Simplified: use train loss) ---
        scheduler.step(avg_epoch_loss)
        if avg_epoch_loss < best_val_loss:
            best_val_loss = avg_epoch_loss
            print(f"Saving checkpoint with loss {best_val_loss:.6f} to {checkpoint_path}")
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'dataset_type': dataset_type, # Store info needed to reload model
                'state_keys': model.state_keys,
                'nx': model.nx,
                'basis_dim': model.basis_dim,
                'd_model': model.d_model,
                'bc_state_dim': model.lifting.bc_state_dim, # Get from lifting
                'num_controls': model.num_controls,
                'num_heads': model.num_heads,
                'shared_attention': model.shared_attention
            }
            torch.save(save_dict, checkpoint_path)

    print("Training finished.")
    # Load best model
    if os.path.exists(checkpoint_path):
        print(f"Loading best model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Re-create model using saved params? Or assume current model structure matches?
        # For simplicity, assume current model structure matches checkpoint.
        model.load_state_dict(checkpoint['model_state_dict'])
    return model
    
def validate_multivar_model(model, data_loader, dataset_type, 
                            train_nt_for_model_training: int, # Timesteps used during model.forward() in training
                            T_value_for_model_training: float, # T value corresponding to train_nt_for_model_training
                            full_T_in_datafile: float, # Max T value in the raw .pkl file (e.g., 2.0)
                            full_nt_in_datafile: int,device='cuda',
                            save_fig_path='rom_result.png',): # Max nt in the raw .pkl file (e.g., 601 if 0-600)
    model.eval()
    results = {key: {'mse': [], 'rmse': [], 'relative_error': [], 'max_error': []}
               for key in model.state_keys}
    overall_rel_err_T1 = [] # For T=1.0 specifically

    # test_horizons are T values (e.g., [1.0, 1.5, 2.0])
    test_horizons_T_values = [T_value_for_model_training,
                              T_value_for_model_training + 0.5 * (full_T_in_datafile - T_value_for_model_training), # Mid-point
                              full_T_in_datafile] # Full duration
    test_horizons_T_values = sorted(list(set(h for h in test_horizons_T_values if h <= full_T_in_datafile))) # Ensure unique and within bounds

    print(f"Validation Horizons (T values): {test_horizons_T_values}")
    print(f"Model was trained with nt={train_nt_for_model_training} for T={T_value_for_model_training}")
    print(f"Datafile contains nt={full_nt_in_datafile} for T={full_T_in_datafile}")


    try:
        state_data_full, BC_Ctrl_tensor_full, norm_factors_batch = next(iter(data_loader))
        # data_loader for validation should give FULL sequences from UniversalPDEDataset
    except StopIteration:
        print("No validation data. Skipping validation.")
        return

    # --- Prepare data from the first sample of the batch ---
    if isinstance(state_data_full, list):
        state_tensors_full = [s[0:1].to(device) for s in state_data_full] # Take 1st sample, full nt
        batch_size_check, nt_loaded, nx_loaded = state_tensors_full[0].shape
    else:
        state_tensors_full = [state_data_full[0:1].to(device)] # Take 1st sample, full nt
        batch_size_check, nt_loaded, nx_loaded = state_tensors_full[0].shape

    BC_Ctrl_tensor_full_sample = BC_Ctrl_tensor_full[0:1].to(device) # [1, full_nt, bc_dim]

    if nt_loaded != full_nt_in_datafile:
        print(f"Warning: nt from val_loader ({nt_loaded}) != full_nt_in_datafile ({full_nt_in_datafile}). Check val_dataset setup.")
        # Potentially adjust full_nt_in_datafile if loader provides less, but ideally they match.
        # For now, we proceed assuming nt_loaded is the true full length available for this sample.
        # full_nt_in_datafile = nt_loaded # Or raise error

    norm_factors_sample = {}
    for key, val_tensor in norm_factors_batch.items():
        if isinstance(val_tensor, torch.Tensor) or isinstance(val_tensor, np.ndarray):
            if val_tensor.ndim > 0 : # Check if it's an array/tensor with elements
                 norm_factors_sample[key] = val_tensor[0] # Take first item if it's batched
            else: # It's a scalar already (e.g. from .item())
                 norm_factors_sample[key] = val_tensor
        else: # It's a scalar Python number
             norm_factors_sample[key] = val_tensor


    state_keys = model.state_keys
    current_batch_size = 1 # We are processing one sample

    # --- Calculate initial coefficients a0_dict (from t=0 of the full data) ---
    a0_dict = {}
    BC0_full = BC_Ctrl_tensor_full_sample[:, 0, :]       # [1, bc_state+ctrl_dim]
    U_B0_lifted = model.lifting(BC0_full)                # [1, num_vars, nx]
    for k_var_idx, key in enumerate(state_keys):
        U0_full_var = state_tensors_full[k_var_idx][:, 0, :].unsqueeze(-1) # [1, nx, 1]
        U_B0_var = U_B0_lifted[:, k_var_idx, :].unsqueeze(-1)               # [1, nx, 1]
        Phi = model.get_basis(key).to(device)
        Phi_T = Phi.transpose(0, 1).unsqueeze(0)
        a0 = torch.bmm(Phi_T, U0_full_var - U_B0_var).squeeze(-1) # [1, basis_dim]
        a0_dict[key] = a0

    for T_test_horizon in test_horizons_T_values:
        # Calculate number of timesteps for this test horizon
        # (nt_file_points - 1) is the number of intervals for T_file
        nt_for_this_horizon = int((T_test_horizon / full_T_in_datafile) * (full_nt_in_datafile - 1)) + 1
        nt_for_this_horizon = min(nt_for_this_horizon, full_nt_in_datafile) # Cap at available data

        print(f"\n--- Validating for T_horizon = {T_test_horizon:.2f} (nt = {nt_for_this_horizon}) ---")

        # Prepare BC_sequence for this horizon by slicing the full BC_Ctrl_tensor_full_sample
        BC_seq_for_pred = BC_Ctrl_tensor_full_sample[:, :nt_for_this_horizon, :]

        # Forward pass for the current horizon
        U_hat_seq_dict, _ = model(a0_dict, BC_seq_for_pred, T=nt_for_this_horizon)

        # --- Quantitative Evaluation (only if T_test_horizon matches training T) ---
        # Or, always evaluate against available ground truth up to nt_for_this_horizon
        # For your request, T=1.0 is special, others are for visual/qualitative.
        # Let's calculate metrics for ALL horizons against their corresponding ground truth slices.

        combined_pred_denorm = []
        combined_gt_denorm = []

        num_vars_plot = len(state_keys)
        fig, axs = plt.subplots(num_vars_plot, 3, figsize=(18, 5 * num_vars_plot), squeeze=False)
        L_vis = 1.0 # Assuming L=1.0 for plotting extent, get from params if dynamic

        for k_var_idx, key in enumerate(state_keys):
            # Predictions (denormalized)
            pred_norm_stacked = torch.cat(U_hat_seq_dict[key], dim=0) # [nt_horizon*batch, nx, 1]
            pred_norm_reshaped = pred_norm_stacked.view(nt_for_this_horizon, current_batch_size, nx_loaded)
            pred_norm_final = pred_norm_reshaped.squeeze(1).detach().cpu().numpy() # [nt_horizon, nx]

            mean_k_val = norm_factors_sample[f'{key}_mean']
            std_k_val = norm_factors_sample[f'{key}_std']
            # Handle scalar vs array for norm_factors (if they were stored differently)
            mean_k = mean_k_val.item() if hasattr(mean_k_val, 'item') else mean_k_val
            std_k = std_k_val.item() if hasattr(std_k_val, 'item') else std_k_val
            pred_denorm = pred_norm_final * std_k + mean_k

            # Ground Truth (denormalized, sliced for this horizon)
            gt_norm_full_var = state_tensors_full[k_var_idx].squeeze(0).cpu().numpy() # [full_nt, nx]
            gt_norm_sliced = gt_norm_full_var[:nt_for_this_horizon, :] # Slice to current horizon
            gt_denorm = gt_norm_sliced * std_k + mean_k

            combined_pred_denorm.append(pred_denorm.flatten())
            combined_gt_denorm.append(gt_denorm.flatten())

            # Calculate metrics for this variable at this horizon
            mse_k = np.mean((pred_denorm - gt_denorm)**2)
            rmse_k = np.sqrt(mse_k)
            rel_err_k = np.linalg.norm(pred_denorm - gt_denorm, 'fro') / (np.linalg.norm(gt_denorm, 'fro') + 1e-10)
            max_err_k = np.max(np.abs(pred_denorm - gt_denorm))

            print(f"  '{key}': MSE={mse_k:.3e}, RMSE={rmse_k:.3e}, RelErr={rel_err_k:.3e}, MaxErr={max_err_k:.3e}")

            # Store results if this is the primary evaluation horizon (e.g., T=1.0)
            if abs(T_test_horizon - T_value_for_model_training) < 1e-5: # If it's the training T
                results[key]['mse'].append(mse_k)
                results[key]['rmse'].append(rmse_k)
                results[key]['relative_error'].append(rel_err_k)
                results[key]['max_error'].append(max_err_k)

            # Plotting
    
            diff_plot = np.abs(pred_denorm - gt_denorm)
            vmin_plot = min(gt_denorm.min(), pred_denorm.min())
            vmax_plot = max(gt_denorm.max(), pred_denorm.max())

            im0 = axs[k_var_idx,0].imshow(gt_denorm, aspect='auto', origin='lower', vmin=vmin_plot, vmax=vmax_plot, extent=[0, L_vis, 0, T_test_horizon], cmap='viridis')
            axs[k_var_idx,0].set_title(f"GT ({key}) T={T_test_horizon:.1f}"); axs[k_var_idx,0].set_ylabel("t")
            plt.colorbar(im0, ax=axs[k_var_idx,0])

            im1 = axs[k_var_idx,1].imshow(pred_denorm, aspect='auto', origin='lower', vmin=vmin_plot, vmax=vmax_plot, extent=[0, L_vis, 0, T_test_horizon], cmap='viridis')
            axs[k_var_idx,1].set_title(f"Pred ({key}) T={T_test_horizon:.1f}")
            plt.colorbar(im1, ax=axs[k_var_idx,1])

            im2 = axs[k_var_idx,2].imshow(diff_plot, aspect='auto', origin='lower', extent=[0, L_vis, 0, T_test_horizon], cmap='magma')
            axs[k_var_idx,2].set_title(f"Error ({key}) (Max {max_err_k:.2e})")
            plt.colorbar(im2, ax=axs[k_var_idx,2])
            for j_plot in range(3): axs[k_var_idx,j_plot].set_xlabel("x")

        overall_rel_err_horizon = np.linalg.norm(np.concatenate(combined_pred_denorm) - np.concatenate(combined_gt_denorm)) / \
                                  (np.linalg.norm(np.concatenate(combined_gt_denorm)) + 1e-10)
        print(f"  Overall RelErr for T={T_test_horizon:.1f}: {overall_rel_err_horizon:.3e}")
        if abs(T_test_horizon - T_value_for_model_training) < 1e-5:
            overall_rel_err_T1.append(overall_rel_err_horizon)


        fig.suptitle(f"Validation @ T={T_test_horizon:.1f} ({dataset_type.upper()}) — basis={model.basis_dim}, d_model={model.d_model}")
        fig.tight_layout(rect=[0,0.03,1,0.95])
        horizon_fig_path = save_fig_path.replace('.png', f'_T{str(T_test_horizon).replace(".","p")}.png')
        plt.savefig(horizon_fig_path)
        print(f"Saved validation figure to: {horizon_fig_path}")
        plt.show()


    # --- Print Summary for T matching training T ---
    print(f"\n--- Validation Summary (Metrics for T={T_value_for_model_training:.1f}) ---")
    for key in state_keys:
        if results[key]['mse']: # Check if results were appended
            avg_mse = np.mean(results[key]['mse'])
            avg_rmse = np.mean(results[key]['rmse'])
            avg_rel = np.mean(results[key]['relative_error'])
            avg_max = np.mean(results[key]['max_error'])
            print(f"  {key}: MSE={avg_mse:.4e}, RMSE={avg_rmse:.4e}, RelErr={avg_rel:.4e}, MaxErr={avg_max:.4e}")
    if overall_rel_err_T1:
        print(f"Overall Avg RelErr for T={T_value_for_model_training:.1f}: {np.mean(overall_rel_err_T1):.4e}")

# main script
import pickle
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate PDE datasets with complex BCs.")
    
    parser.add_argument('--datatype', type=str, required=True,
                        choices=['heat_delayed_feedback',
                                 'reaction_diffusion_neumann_feedback',
                                 'heat_nonlinear_feedback_gain', 'convdiff'], # Added new types
                        help='Type of dataset to use.')
    parser.add_argument('--poddim', type=int, required=True,
                            choices=[8,16,24], # Added new types
                            help='Type of dataset to use.')
    args = parser.parse_args()

    DATASET_TYPE = args.datatype # 'advection', 'euler', 'burgers', 'darcy'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Selected Dataset Type: {DATASET_TYPE.upper()}")

    # --- Key Time Parameters ---
    # These should align with your dataset generation commands
    # e.g., python generate_datasets.py --T 2 --nt 600

    FULL_T_IN_DATAFILE = 2.0
    FULL_NT_IN_DATAFILE = 300 # Number of time points (e.g., if nt=200, indices 0 to 199)

        

    TRAIN_T_TARGET = 1.5 # Example: train on 80% of the new dataset's T

    # Calculate the number of timesteps for training (from 0 to TRAIN_T_TARGET)
    # (FULL_NT_IN_DATAFILE - 1) is the number of intervals for FULL_T_IN_DATAFILE
    TRAIN_NT_FOR_MODEL = int((TRAIN_T_TARGET / FULL_T_IN_DATAFILE) * (FULL_NT_IN_DATAFILE - 1)) + 1

    print(f"Full data T={FULL_T_IN_DATAFILE}, nt={FULL_NT_IN_DATAFILE}")
    print(f"Training will use T={TRAIN_T_TARGET}, nt={TRAIN_NT_FOR_MODEL}")


    # --- Common Parameters ---
    basis_dim = args.poddim
    d_model = 512
    num_heads = 8
    add_error_estimator = True # Set to True if you want to use it
    shared_attention = False

    batch_size = 32
    num_epochs = 150
    learning_rate = 1e-4
    lambda_res = 0.05
    lambda_orth = 0.001
    lambda_bc_penalty = 0.01
    clip_grad_norm = 1.0


    # Paths
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{DATASET_TYPE}_b{basis_dim}_d{d_model}"
    checkpoint_dir = f"./New_ckpt_2/_checkpoints_{DATASET_TYPE}"
    results_dir = f"./result_all_2/results_{DATASET_TYPE}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'barom_{run_name}.pt')
    save_fig_path = os.path.join(results_dir, f'barom_result_{run_name}.png')
    basis_dir = os.path.join(checkpoint_dir, 'pod_bases')
    os.makedirs(basis_dir, exist_ok=True)


    # --- Dataset Specific Parameters & Loading ---
    if DATASET_TYPE == 'heat_delayed_feedback':
        # === IMPORTANT: Update with your actual path and generation parameters ===
        dataset_path = "./datasets_new_feedback/heat_delayed_feedback_v1_5000s_64nx_300nt.pkl" # EXAMPLE PATH
        nx = 64; L = 1.0 # Must match generation
        state_keys = ['U']; num_state_vars = 1
        num_controls = 2
    elif DATASET_TYPE == 'reaction_diffusion_neumann_feedback':
        # === IMPORTANT: Update with your actual path and generation parameters ===
        dataset_path = "./datasets_new_feedback/reaction_diffusion_neumann_feedback_v1_5000s_64nx_300nt.pkl" # EXAMPLE PATH
        nx = 64; L = 1.0 # Must match generation
        state_keys = ['U']; num_state_vars = 1
        num_controls = 2
    elif DATASET_TYPE == 'heat_nonlinear_feedback_gain':
        #num_controls = train_dataset.num_controls
        # === IMPORTANT: Update with your actual path and generation parameters ===
        dataset_path = "./datasets_new_feedback/heat_nonlinear_feedback_gain_v1_5000s_64nx_300nt.pkl" # EXAMPLE PATH
        nx = 64; L = 1.0 # Must match generation
        state_keys = ['U']; num_state_vars = 1
        num_controls = 2
         
    elif DATASET_TYPE == 'convdiff':
        # 你生成的 Convection–Diffusion 数据集
        dataset_path    = "./datasets_new_feedback/convdiff_v1_5000s_64nx_300nt.pkl"
        nx               = 64; L = 1.0
        state_keys       = ['U']
        num_state_vars   = 1
        num_controls     = 2
    else:
        raise ValueError(f"Unknown dataset_type: {DATASET_TYPE}")
    dataset_params_for_plot = {'nx': nx, 'ny': 1, 'L': L, 'T': FULL_T_IN_DATAFILE}
    print(f"Loading dataset for {DATASET_TYPE} from {dataset_path}")
    try:
        with open(dataset_path, 'rb') as f:
            data_list = pickle.load(f)
        print(f"Loaded {len(data_list)} samples.")
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_path}")
        exit()
        
    # --- Data Generation ---
    if not data_list: print("No data generated, exiting."); exit()

    bc_state_dim = 2
    # --- Data Splitting & Loading ---
    random.shuffle(data_list)
    n_total = len(data_list); n_train = int(0.8 * n_total)
    train_data_list = data_list[:n_train]; val_data_list = data_list[n_train:]
    print(f"Train samples: {len(train_data_list)}, Validation samples: {len(val_data_list)}")

    # Create training dataset with truncation
    train_dataset = UniversalPDEDataset(train_data_list, dataset_type=DATASET_TYPE,
                                        train_nt_limit=TRAIN_NT_FOR_MODEL)
    # Create validation dataset *without* truncation (it will load full sequences)
    val_dataset = UniversalPDEDataset(val_data_list, dataset_type=DATASET_TYPE,
                                      train_nt_limit=None) # or pass full_nt_in_datafile
    num_workers = 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    # --- Determine spatial dimension for model ---
    current_nx = nx if DATASET_TYPE != 'darcy' else nx * ny

    current_nx_model = train_dataset.nx #* train_dataset.ny # spatial_dim from dataset
    if DATASET_TYPE == 'heat_nonlinear_feedback_gain':
        num_controls = train_dataset.num_controls

    # --- Model Initialization ---
    online_model = MultiVarAttentionROM(
        state_variable_keys=state_keys, nx=current_nx_model, basis_dim=basis_dim,
        d_model=d_model, bc_state_dim=bc_state_dim, num_controls=num_controls,
        num_heads=num_heads, add_error_estimator=add_error_estimator,
        shared_attention=shared_attention
    )

    # --- POD Basis Initialization (per variable) ---
    print("\nInitializing POD bases...")
    pod_bases = {}

    if not train_dataset:
        print("Error: Training dataset is empty. Cannot compute POD.")
        exit()


    try:
        # Get the first sample's data to determine nt
        first_sample_data = train_dataset[0]
        first_state_tensor = first_sample_data[0][0] # Access the first tensor in the state list
        actual_nt = first_state_tensor.shape[0]
    except IndexError:
        print("Error: Could not access shape from the first sample in train_dataset.")
        exit()


    for key in state_keys:
        basis_path = os.path.join(basis_dir, f'pod_basis_{key}_nx{current_nx}_bd{basis_dim}.npy') # current_nx should be correctly set earlier (nx or nx*ny)



        loaded_basis = None
        if os.path.exists(basis_path):
            print(f"  Loading existing POD basis for '{key}' from {basis_path}...")
            try:
                loaded_basis = np.load(basis_path)
            except Exception as e:
                print(f"  Error loading {basis_path}: {e}. Will recompute.")
                loaded_basis = None
            # Validate shape after loading
            if loaded_basis is not None and loaded_basis.shape != (current_nx, basis_dim):
                print(f"  Shape mismatch for loaded basis '{key}'. Expected ({current_nx}, {basis_dim}), got {loaded_basis.shape}. Recomputing.")
                loaded_basis = None

        if loaded_basis is None:
            print(f"  Computing POD basis for '{key}' (using nt={actual_nt})...")

            # <<< CORRECTED CALL to compute_pod_basis_generic >>>
            computed_basis = compute_pod_basis_generic(
                data_list=train_data_list, # Raw list of dicts for POD function to truncate
                dataset_type=DATASET_TYPE, state_variable_key=key,
                nx=current_nx_model, nt=TRAIN_NT_FOR_MODEL, # Pass truncated nt
                basis_dim=basis_dim
            )
            # <<< END CORRECTION >>>

            if computed_basis is not None:
                 pod_bases[key] = computed_basis
                 # Ensure directory exists before saving
                 os.makedirs(os.path.dirname(basis_path), exist_ok=True)
                 np.save(basis_path, computed_basis)
                 print(f"  Saved computed POD basis for '{key}' to {basis_path}")
            else:
                 print(f"ERROR: Failed to compute POD basis for '{key}'. Exiting.")
                 exit()
        else:
             pod_bases[key] = loaded_basis

    # Initialize model Phi parameters
    with torch.no_grad():
        for key in state_keys:
            if key in pod_bases and key in online_model.Phi:
                 model_phi = online_model.Phi[key]; pod_phi = torch.tensor(pod_bases[key])
                 if model_phi.shape == pod_phi.shape: model_phi.copy_(pod_phi); print(f"  Initialized Phi for '{key}' with POD.")
                 else: print(f"  WARNING: Shape mismatch for Phi '{key}'. Using random init.")
            else: print(f"  WARNING: No POD basis found/needed for '{key}'. Using random init.")


    # --- Training ---
    print(f"\nStarting training for {DATASET_TYPE.upper()}...")
    start_train_time = time.time()
    online_model = train_multivar_model(
        online_model, train_loader, dataset_type=DATASET_TYPE,
        train_nt_target=TRAIN_NT_FOR_MODEL, # Pass the training horizon
        lr=learning_rate, num_epochs=num_epochs, device=device,
        checkpoint_path=checkpoint_path, lambda_res=lambda_res,
        lambda_orth=lambda_orth, lambda_bc_penalty=lambda_bc_penalty,
        clip_grad_norm=clip_grad_norm
    )
    end_train_time = time.time()
    print(f"Training took {end_train_time - start_train_time:.2f} seconds.")

    # --- Validation ---
    if val_data_list: # Only validate if there is validation data
        print(f"\nStarting validation for {DATASET_TYPE.upper()}...")
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
    if val_data_list: print(f"Validation figure saved to: {save_fig_path}")
    print("="*60)
