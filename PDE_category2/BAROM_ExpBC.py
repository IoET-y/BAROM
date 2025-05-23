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
import argparse
import pickle

# Fixed random seed)
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

class UniversalPDEDataset(Dataset):
    def __init__(self, data_list, dataset_type, train_nt_limit=None):
        if not data_list:
            raise ValueError("data_list cannot be empty")
        self.data_list = data_list
        self.dataset_type = dataset_type
        self.train_nt_limit = train_nt_limit

        first_sample = data_list[0]

        if dataset_type == 'heat_delayed_feedback':
            self.nt_from_sample = first_sample['U'].shape[0]
            self.nx_from_sample = first_sample['U'].shape[1]
            self.ny_from_sample = 1
            self.state_keys = ['U']
            self.num_state_vars = 1
            self.nx = self.nx_from_sample
            self.ny = self.ny_from_sample
            self.expected_bc_state_dim = 2
        elif dataset_type == 'reaction_diffusion_neumann_feedback':
            self.nt_from_sample = first_sample['U'].shape[0]
            self.nx_from_sample = first_sample['U'].shape[1]
            self.ny_from_sample = 1
            self.state_keys = ['U']
            self.num_state_vars = 1
            self.nx = self.nx_from_sample
            self.ny = self.ny_from_sample
            self.expected_bc_state_dim = 2
        elif dataset_type == 'heat_nonlinear_feedback_gain':
            self.nt_from_sample = first_sample['U'].shape[0]
            self.nx_from_sample = first_sample['U'].shape[1]
            self.ny_from_sample = 1
            self.state_keys = ['U']
            self.num_state_vars = 1
            self.nx = self.nx_from_sample
            self.ny = self.ny_from_sample
            self.expected_bc_state_dim = 2
        elif dataset_type == 'convdiff':
            self.nt_from_sample = first_sample['U'].shape[0]
            self.nx_from_sample = first_sample['U'].shape[1]
            self.state_keys = ['U']
            self.num_state_vars = 1
            self.nx = self.nx_from_sample
            self.ny = 1
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
        if self.bc_control_key in first_sample and first_sample[self.bc_control_key] is not None:
            if first_sample[self.bc_control_key].ndim == 2:
                 self.num_controls = first_sample[self.bc_control_key].shape[1]
            elif first_sample[self.bc_control_key].ndim == 1 and first_sample[self.bc_control_key].shape[0] == self.effective_nt:
                 print(f"Warning: '{self.bc_control_key}' is 1D. Assuming num_controls = 0 or check data format.")
                 self.num_controls = 0
            else:
                 self.num_controls = 0
        else:
            self.num_controls = 0

        if self.num_controls == 0 and self.bc_control_key in first_sample and first_sample[self.bc_control_key] is not None:
            print(f"Info: '{self.bc_control_key}' found but num_controls is 0 for sample 0. Shape: {first_sample[self.bc_control_key].shape if hasattr(first_sample[self.bc_control_key], 'shape') else 'N/A'}")
        elif self.num_controls == 0:
            print(f"Info: '{self.bc_control_key}' not found or is None in the first sample. Assuming num_controls = 0.")


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        norm_factors = {}
        current_nt = self.effective_nt

        state_tensors_norm_list = []
        for key in self.state_keys:
            try:
                state_seq_full = sample[key]
                state_seq = state_seq_full[:current_nt, ...]
            except KeyError:
                raise KeyError(f"State variable key '{key}' not found in sample {idx} for dataset type '{self.dataset_type}'")

            if state_seq.shape[0] != current_nt:
                raise ValueError(f"Time dimension mismatch after potential truncation for {key}. Expected {current_nt}, got {state_seq.shape[0]}")

            state_mean = np.mean(state_seq)
            state_std = np.std(state_seq) + 1e-8
            state_norm = (state_seq - state_mean) / state_std
            state_tensors_norm_list.append(torch.tensor(state_norm).float())
            norm_factors[f'{key}_mean'] = state_mean
            norm_factors[f'{key}_std'] = state_std

        bc_state_seq_full = sample[self.bc_state_key]
        bc_state_seq = bc_state_seq_full[:current_nt, :]

        if bc_state_seq.shape[0] != current_nt:
            raise ValueError(f"Time dimension mismatch for BC_State. Expected {current_nt}, got {bc_state_seq.shape[0]}")

        bc_state_norm = np.zeros_like(bc_state_seq, dtype=np.float32)
        norm_factors[f'{self.bc_state_key}_means'] = np.zeros(self.bc_state_dim)
        norm_factors[f'{self.bc_state_key}_stds'] = np.ones(self.bc_state_dim)
        if self.bc_state_dim > 0:
            for k_dim in range(self.bc_state_dim):
                col = bc_state_seq[:, k_dim]
                mean_k = np.mean(col)
                std_k = np.std(col)
                if std_k > 1e-8:
                    bc_state_norm[:, k_dim] = (col - mean_k) / std_k
                    norm_factors[f'{self.bc_state_key}_means'][k_dim] = mean_k
                    norm_factors[f'{self.bc_state_key}_stds'][k_dim] = std_k
                else:
                    bc_state_norm[:, k_dim] = col - mean_k
                    norm_factors[f'{self.bc_state_key}_means'][k_dim] = mean_k
        bc_state_tensor_norm = torch.tensor(bc_state_norm).float()

        if self.num_controls > 0:
            try:
                bc_control_seq_full = sample[self.bc_control_key]
                bc_control_seq = bc_control_seq_full[:current_nt, :]

                if bc_control_seq.shape[0] != current_nt:
                    raise ValueError(f"Time dimension mismatch for BC_Control. Expected {current_nt}, got {bc_control_seq.shape[0]}.")
                if bc_control_seq.shape[1] != self.num_controls:
                     raise ValueError(f"Control dimension mismatch for sample {idx}. Expected {self.num_controls}, got {bc_control_seq.shape[1]}")

                bc_control_norm = np.zeros_like(bc_control_seq, dtype=np.float32)
                norm_factors[f'{self.bc_control_key}_means'] = np.zeros(self.num_controls)
                norm_factors[f'{self.bc_control_key}_stds'] = np.ones(self.num_controls)
                for k_dim in range(self.num_controls):
                    col = bc_control_seq[:, k_dim]
                    mean_k = np.mean(col)
                    std_k = np.std(col)
                    if std_k > 1e-8:
                        bc_control_norm[:, k_dim] = (col - mean_k) / std_k
                        norm_factors[f'{self.bc_control_key}_means'][k_dim] = mean_k
                        norm_factors[f'{self.bc_control_key}_stds'][k_dim] = std_k
                    else:
                        bc_control_norm[:, k_dim] = col - mean_k
                        norm_factors[f'{self.bc_control_key}_means'][k_dim] = mean_k
                bc_control_tensor_norm = torch.tensor(bc_control_norm).float()
            except KeyError:
                print(f"Warning: Sample {idx} missing '{self.bc_control_key}' despite num_controls > 0. Using zeros.")
                bc_control_tensor_norm = torch.zeros((current_nt, self.num_controls), dtype=torch.float32)
                norm_factors[f'{self.bc_control_key}_means'] = np.zeros(self.num_controls)
                norm_factors[f'{self.bc_control_key}_stds'] = np.ones(self.num_controls)
        else:
            bc_control_tensor_norm = torch.empty((current_nt, 0), dtype=torch.float32)

        bc_ctrl_tensor_norm = torch.cat((bc_state_tensor_norm, bc_control_tensor_norm), dim=-1)
        return state_tensors_norm_list, bc_ctrl_tensor_norm, norm_factors

# =============================================================================
# 3. 通用化 POD 基计算 (compute_pod_basis_generic)
# =============================================================================
def compute_pod_basis_generic(data_list, dataset_type, state_variable_key,
                              nx, nt, basis_dim,
                              max_snapshots_pod=100):
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

        U_seq_full = sample[state_variable_key]
        U_seq = U_seq_full[:nt, :]

        if U_seq.shape[0] != nt:
            print(f"Warning: U_seq actual timesteps {U_seq.shape[0]} != requested nt {nt} for POD in sample {sample_idx}. Skipping.")
            continue
        if U_seq.shape[1] != current_nx:
            print(f"Warning: Mismatch nx in sample {sample_idx} for {state_variable_key}. Expected {current_nx}, got {U_seq.shape[1]}. Skipping.")
            continue

        # Paper Eq (8): U^{n*} = U^n - U_B(U_BC^n)
        # Here U_B is linear interpolation for POD snapshot modification
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
    try:
        all_snapshots_np = np.concatenate(snapshots, axis=0)
    except ValueError as e:
        print(f"Error concatenating snapshots for '{state_variable_key}': {e}. Check snapshot shapes.")
        return None

    if np.isnan(all_snapshots_np).any() or np.isinf(all_snapshots_np).any():
        print(f"Warning: NaN/Inf found in snapshots for '{state_variable_key}' before POD. Clamping.")
        all_snapshots_np = np.nan_to_num(all_snapshots_np, nan=0.0, posinf=1e6, neginf=-1e6)
        if np.all(np.abs(all_snapshots_np) < 1e-12):
            print(f"Error: All snapshots became zero after clamping for '{state_variable_key}'.")
            return None

    U_mean = np.mean(all_snapshots_np, axis=0, keepdims=True)
    U_centered = all_snapshots_np - U_mean

    try:
        U_data_svd, S_data_svd, Vh_data_svd = np.linalg.svd(U_centered, full_matrices=False)
        rank = np.sum(S_data_svd > 1e-10)
        actual_basis_dim = min(basis_dim, rank, current_nx)
        if actual_basis_dim == 0:
            print(f"Error: Data rank is zero or too low for '{state_variable_key}' after SVD. Effective rank: {rank}")
            return None
        if actual_basis_dim < basis_dim:
            print(f"Warning: Requested basis_dim {basis_dim} but effective data rank is ~{rank} for '{state_variable_key}'. Using {actual_basis_dim}.")
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
    def __init__(self, input_dim, hidden_dim=256, num_layers=3, dropout=0.1, output_dim=None):
        super().__init__()
        layers = []
        current_dim = input_dim
        if output_dim is None:
            output_dim = input_dim

        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        self.layernorm = nn.LayerNorm(output_dim)
        self.input_dim_for_residual = input_dim
        self.output_dim_for_residual = output_dim

    def forward(self, x):
        mlp_out = self.mlp(x)
        if self.input_dim_for_residual == self.output_dim_for_residual:
            out = self.layernorm(mlp_out + x)
        else:
            out = self.layernorm(mlp_out)
        return out

# 4.2. Lifting 
class UniversalLifting(nn.Module):
    def __init__(self, num_state_vars, bc_state_dim, num_controls, output_dim_per_var, nx,
                 hidden_dims_state_branch=64,
                 hidden_dims_control=[64, 128],
                 hidden_dims_fusion=[256, 512, 256],
                 dropout=0.1):
        super().__init__()
        self.num_state_vars = num_state_vars
        self.bc_state_dim = bc_state_dim
        self.num_controls = num_controls
        self.nx = nx
        assert output_dim_per_var == nx, "output_dim_per_var must equal nx"

        self.state_branches = nn.ModuleList()
        if self.bc_state_dim > 0:
            for _ in range(bc_state_dim):
                self.state_branches.append(nn.Sequential(
                    nn.Linear(1, hidden_dims_state_branch),
                    nn.GELU(),
                ))
            state_feature_dim = bc_state_dim * hidden_dims_state_branch
        else:
            state_feature_dim = 0

        control_feature_dim = 0
        if self.num_controls > 0:
            control_layers = []
            current_dim_ctrl = num_controls
            for h_dim in hidden_dims_control:
                control_layers.append(nn.Linear(current_dim_ctrl, h_dim))
                control_layers.append(nn.GELU())
                control_layers.append(nn.Dropout(dropout))
                current_dim_ctrl = h_dim
            self.control_mlp = nn.Sequential(*control_layers)
            control_feature_dim = current_dim_ctrl
        else:
            self.control_mlp = nn.Sequential()

        fusion_input_dim = state_feature_dim + control_feature_dim
        fusion_layers = []

        if fusion_input_dim > 0:
            current_dim_fusion = fusion_input_dim
            for h_dim in hidden_dims_fusion:
                fusion_layers.append(nn.Linear(current_dim_fusion, h_dim))
                fusion_layers.append(nn.GELU())
                fusion_layers.append(nn.Dropout(dropout))
                current_dim_fusion = h_dim
            fusion_layers.append(nn.Linear(current_dim_fusion, num_state_vars * nx))
            self.fusion = nn.Sequential(*fusion_layers)
        else:
             self.fusion = None


    def forward(self, BC_Ctrl):
        if self.fusion is None:
            batch_size = BC_Ctrl.shape[0] if BC_Ctrl is not None and BC_Ctrl.nelement() > 0 else 1
            return torch.zeros(batch_size, self.num_state_vars, self.nx, device=BC_Ctrl.device if BC_Ctrl is not None else 'cpu')

        features_to_concat = []
        if self.bc_state_dim > 0:
            BC_state = BC_Ctrl[:, :self.bc_state_dim]
            state_features_list = []
            for i in range(self.bc_state_dim):
                branch_out = self.state_branches[i](BC_state[:, i:i+1])
                state_features_list.append(branch_out)
            state_features = torch.cat(state_features_list, dim=-1)
            features_to_concat.append(state_features)

        if self.num_controls > 0:
            BC_control = BC_Ctrl[:, self.bc_state_dim:]
            control_features = self.control_mlp(BC_control)
            features_to_concat.append(control_features)

        if not features_to_concat:
            batch_size = BC_Ctrl.shape[0] if BC_Ctrl is not None and BC_Ctrl.nelement() > 0 else 1
            return torch.zeros(batch_size, self.num_state_vars, self.nx, device=BC_Ctrl.device if BC_Ctrl is not None else 'cpu')

        if len(features_to_concat) == 1:
            concat_features = features_to_concat[0]
        else:
            concat_features = torch.cat(features_to_concat, dim=-1)

        fused_output = self.fusion(concat_features)
        U_B_stacked = fused_output.view(-1, self.num_state_vars, self.nx)
        return U_B_stacked

# 4.3.MultiHeadAttentionROM base class
class MultiHeadAttentionROM(nn.Module):
    def __init__(self, basis_dim, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V):
        batch_size, seq_len_q, d_model_q = Q.size()
        _, seq_len_kv, d_model_kv = K.size()

        assert seq_len_q == seq_len_kv, "Sequence lengths of Q and K/V must match"
        assert d_model_q == d_model_kv, "d_model of Q and K/V must match"

        seq_len = seq_len_q
        d_model = d_model_q

        Q_reshaped = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K_reshaped = K.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V_reshaped = V.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        KV = torch.matmul(K_reshaped.transpose(-2, -1), V_reshaped)
        z = torch.matmul(Q_reshaped, KV)

        z = z.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, d_model)
        z = self.out_proj(z)
        return z


# 4.4. Attention-Based ROM: 支持多变量 (UPGRADED MultiVarAttentionROM)
class MultiVarAttentionROM(nn.Module):
    def __init__(self, state_variable_keys, nx, basis_dim, d_model,
                 bc_state_dim, num_controls, num_heads=8,
                 add_error_estimator=False, shared_attention=False,
                 dropout_lifting=0.1, dropout_ffn=0.1, initial_alpha=0.1,
                 use_fixed_lifting=False,
                 bc_processed_dim=64,
                 hidden_bc_processor_dim=128
                 ):
        super().__init__()
        self.state_keys = state_variable_keys
        self.num_state_vars = len(state_variable_keys)
        self.nx = nx
        self.basis_dim = basis_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.bc_state_dim = bc_state_dim
        self.num_controls = num_controls
        self.add_error_estimator = add_error_estimator
        self.shared_attention = shared_attention
        self.use_fixed_lifting = use_fixed_lifting
        self.bc_processed_dim = bc_processed_dim

        # --- Learnable Bases $\Phi$ ---
        # Corresponds to $\Phi$ in Eq. (9)
        self.Phi = nn.ParameterDict()
        for key in self.state_keys:
            phi_param = nn.Parameter(torch.randn(nx, basis_dim))
            nn.init.orthogonal_(phi_param)
            self.Phi[key] = phi_param

        # --- Lifting Network ---
        # Computes U_B(...) in Eq. (9) based on BC_State and BC_Control
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
            self.lifting = None
            lin_interp_coeffs = torch.linspace(0, 1, self.nx, dtype=torch.float32)
            self.register_buffer('lin_interp_coeffs', lin_interp_coeffs.view(1, 1, -1))

        # --- Attention & FFN Components ---
        self.W_Q = nn.ModuleDict()
        self.W_K = nn.ModuleDict()
        self.W_V = nn.ModuleDict()
        self.multihead_attn = nn.ModuleDict()
        self.proj_to_coef = nn.ModuleDict()
        self.update_ffn = nn.ModuleDict() # Learns part of internal dynamics updates
        self.a0_mapping = nn.ModuleDict()
        self.alphas = nn.ParameterDict()

        # --- BC Feature Processing Components ---
        self.bc_feature_processor = nn.ModuleDict() # Processes BC_Ctrl_n
        # Maps processed BC features to an update for a_n,
        # implicitly learning terms like $\hat{B}\hat{w}^n + \hat{A}_{BC}U_B^n$ from Eq. (9)
        self.bc_to_a_update = nn.ModuleDict()

        total_bc_ctrl_dim = self.bc_state_dim + self.num_controls

        if shared_attention:
            self.W_Q['shared'] = nn.Linear(1, d_model)
            self.W_K['shared'] = nn.Linear(nx, d_model)
            self.W_V['shared'] = nn.Linear(nx, d_model)
            self.multihead_attn['shared'] = MultiHeadAttentionROM(basis_dim, d_model, num_heads)
            self.proj_to_coef['shared'] = nn.Linear(d_model, 1)
            self.update_ffn['shared'] = ImprovedUpdateFFN(input_dim=basis_dim, output_dim=basis_dim,
                                                          hidden_dim=d_model, dropout=dropout_ffn)
            self.a0_mapping['shared'] = nn.Sequential(
                nn.Linear(basis_dim, basis_dim), nn.ReLU(), nn.LayerNorm(basis_dim)
            )
            self.alphas['shared'] = nn.Parameter(torch.tensor(initial_alpha))
            if total_bc_ctrl_dim > 0:
                self.bc_feature_processor['shared'] = nn.Sequential(
                    nn.Linear(total_bc_ctrl_dim, hidden_bc_processor_dim), nn.GELU(),
                    nn.Linear(hidden_bc_processor_dim, self.bc_processed_dim)
                )
                self.bc_to_a_update['shared'] = nn.Linear(self.bc_processed_dim, basis_dim)
            else:
                self.bc_feature_processor['shared'] = nn.Sequential()
                self.bc_to_a_update['shared'] = nn.Linear(0, basis_dim) if self.bc_processed_dim == 0 else nn.Linear(self.bc_processed_dim, basis_dim)
        else:
            for key in self.state_keys:
                self.W_Q[key] = nn.Linear(1, d_model)
                self.W_K[key] = nn.Linear(nx, d_model)
                self.W_V[key] = nn.Linear(nx, d_model)
                self.multihead_attn[key] = MultiHeadAttentionROM(basis_dim, d_model, num_heads)
                self.proj_to_coef[key] = nn.Linear(d_model, 1)
                self.update_ffn[key] = ImprovedUpdateFFN(input_dim=basis_dim, output_dim=basis_dim,
                                                         hidden_dim=d_model, dropout=dropout_ffn)
                self.a0_mapping[key] = nn.Sequential(
                    nn.Linear(basis_dim, basis_dim), nn.ReLU(), nn.LayerNorm(basis_dim)
                )
                self.alphas[key] = nn.Parameter(torch.tensor(initial_alpha))
                if total_bc_ctrl_dim > 0:
                    self.bc_feature_processor[key] = nn.Sequential(
                        nn.Linear(total_bc_ctrl_dim, hidden_bc_processor_dim), nn.GELU(),
                        nn.Linear(hidden_bc_processor_dim, self.bc_processed_dim)
                    )
                    self.bc_to_a_update[key] = nn.Linear(self.bc_processed_dim, basis_dim)
                else:
                    self.bc_feature_processor[key] = nn.Sequential()
                    self.bc_to_a_update[key] = nn.Linear(0, basis_dim) if self.bc_processed_dim == 0 else nn.Linear(self.bc_processed_dim, basis_dim)

        if self.add_error_estimator:
            total_basis_dim_all_vars = self.num_state_vars * basis_dim
            self.error_estimator = nn.Linear(total_basis_dim_all_vars, 1)

    def _get_layer(self, module_dict, key):
        return module_dict['shared'] if self.shared_attention else module_dict[key]

    def _get_alpha(self, key):
        return self.alphas['shared'] if self.shared_attention else self.alphas[key]

    def _compute_U_B(self, BC_Ctrl_step):
        # This computes U_B(BC_State, BC_Control) at a given time step.
        # Corresponds to U_B^n or U_B^{n+1} in Eq. (9) depending on input BC_Ctrl_step.
        if not self.use_fixed_lifting:
            if self.lifting is None:
                raise ValueError("Lifting network is None but use_fixed_lifting is False.")
            return self.lifting(BC_Ctrl_step)
        else:
            if self.num_state_vars == 1:
                if self.bc_state_dim < 2 :
                    print(f"Warning: Fixed lifting for 1 state var expects bc_state_dim >= 2, got {self.bc_state_dim}. Returning zeros for U_B.")
                    return torch.zeros(BC_Ctrl_step.shape[0], 1, self.nx, device=BC_Ctrl_step.device)
                bc_left_val = BC_Ctrl_step[:, 0:1].unsqueeze(-1)
                bc_right_val = BC_Ctrl_step[:, 1:2].unsqueeze(-1)
                U_B_var = bc_left_val * (1 - self.lin_interp_coeffs) + \
                          bc_right_val * self.lin_interp_coeffs
                return U_B_var
            else:
                U_B_list = []
                if self.bc_state_dim < 2 * self.num_state_vars and self.num_state_vars > 0:
                    print(f"Warning: Fixed lifting for {self.num_state_vars} vars expects bc_state_dim >= {2*self.num_state_vars}, got {self.bc_state_dim}. Returning zeros for U_B.")
                    return torch.zeros(BC_Ctrl_step.shape[0], self.num_state_vars, self.nx, device=BC_Ctrl_step.device)
                for i_var in range(self.num_state_vars):
                    idx_left = i_var * 2
                    idx_right = i_var * 2 + 1
                    if idx_right >= self.bc_state_dim :
                        print(f"Warning: Not enough BC_State values for fixed lifting of var {i_var}. Using zeros.")
                        U_B_single_var = torch.zeros(BC_Ctrl_step.shape[0], 1, self.nx, device=BC_Ctrl_step.device)
                    else:
                        bc_left_val = BC_Ctrl_step[:, idx_left:idx_left+1].unsqueeze(-1)
                        bc_right_val = BC_Ctrl_step[:, idx_right:idx_right+1].unsqueeze(-1)
                        U_B_single_var = bc_left_val * (1 - self.lin_interp_coeffs) + \
                                         bc_right_val * self.lin_interp_coeffs
                    U_B_list.append(U_B_single_var)
                return torch.cat(U_B_list, dim=1)

    # MODIFIED forward_step
    def forward_step(self, a_n_dict, BC_Ctrl_n, U_B_np1_stacked, params=None):
        """
        Computes one step of the ROM: a^{n+1} and U_hat^{n+1}.
        Args:
            a_n_dict (dict): Dict of modal coefficients at time n, {key: [batch, basis_dim]}. (Corresponds to a^n in Eq. (9))
            BC_Ctrl_n (torch.Tensor): Boundary and control inputs at time n. [batch, bc_state_dim + num_controls]
                                      Used to compute U_B^n and terms like hat(B)w^n + hat(A)_BC U_B^n.
            U_B_np1_stacked (torch.Tensor): Lifting function at time n+1. [batch, num_vars, nx] (Corresponds to U_B^{n+1} in Eq. (9))
        """
        batch_size = list(a_n_dict.values())[0].size(0)
        a_next_dict = {}  # To store a^{n+1}
        U_hat_dict = {}   # To store U_hat^{n+1}

        # 1. Compute U_B^n from BC_Ctrl_n (for terms in Eq. (9) depending on U_B^n)
        # This is U_B^n in Eq. (9)
        U_B_n_stacked = self._compute_U_B(BC_Ctrl_n)  # [batch, num_vars, nx]

        # 2. Process BC_Ctrl_n to get features for bc_driven_a_update.
        # This part implicitly learns terms like $\hat{B}\hat{w}^n + \hat{A}_{BC}U_B^n$ from Eq. (9)
        bc_features_processed_dict = {}
        total_bc_ctrl_dim = self.bc_state_dim + self.num_controls
        if total_bc_ctrl_dim > 0:
            if self.shared_attention:
                bc_proc_layer = self._get_layer(self.bc_feature_processor, 'shared')
                if hasattr(bc_proc_layer, 'weight') or len(list(bc_proc_layer.parameters())) > 0 :
                    shared_bc_features = bc_proc_layer(BC_Ctrl_n)
                    for key_loop in self.state_keys: bc_features_processed_dict[key_loop] = shared_bc_features
                else:
                    for key_loop in self.state_keys: bc_features_processed_dict[key_loop] = None
            else:
                for key_loop in self.state_keys:
                    bc_proc_layer = self._get_layer(self.bc_feature_processor, key_loop)
                    if hasattr(bc_proc_layer, 'weight') or len(list(bc_proc_layer.parameters())) > 0 :
                        bc_features_processed_dict[key_loop] = bc_proc_layer(BC_Ctrl_n)
                    else:
                        bc_features_processed_dict[key_loop] = None
        else:
            for key_loop in self.state_keys: bc_features_processed_dict[key_loop] = None

        for i_var, key in enumerate(self.state_keys):
            a_n_var = a_n_dict[key]    # This is a^n for the current variable
            Phi_var = self.Phi[key]      # This is \Phi for the current variable

            W_Q_var = self._get_layer(self.W_Q, key)
            W_K_var = self._get_layer(self.W_K, key)
            W_V_var = self._get_layer(self.W_V, key)
            attn_module_var = self._get_layer(self.multihead_attn, key)
            proj_var = self._get_layer(self.proj_to_coef, key)
            ffn_var = self._get_layer(self.update_ffn, key)
            alpha_var = self._get_alpha(key) # Use self._get_alpha
            bc_to_a_update_layer = self._get_layer(self.bc_to_a_update, key)

            # --- Attention Update Component (Learns part of internal dynamics like \hat{A}_a a^n) ---
            Phi_basis_vectors = Phi_var.transpose(0, 1).unsqueeze(0).expand(batch_size, -1, -1)
            K_flat = W_K_var(Phi_basis_vectors.reshape(-1, self.nx))
            V_flat = W_V_var(Phi_basis_vectors.reshape(-1, self.nx))
            K = K_flat.view(batch_size, self.basis_dim, self.d_model)
            V = V_flat.view(batch_size, self.basis_dim, self.d_model)
            a_n_unsq_for_Q = a_n_var.unsqueeze(-1)
            Q_base = W_Q_var(a_n_unsq_for_Q)
            Q_for_attention = Q_base # Simplified Q for attention; FFN effect added later

            z = attn_module_var(Q_for_attention, K, V)
            z = z / np.sqrt(float(self.d_model)) # Scaling
            # This part contributes to the learned equivalent of $\hat{A}_a a^n$ term in Eq. (9)
            a_update_attn_val = proj_var(z.reshape(-1, self.d_model)).view(batch_size, self.basis_dim)

            # --- FFN Update Component (Learns another part of internal dynamics) ---
            # This also contributes to the learned equivalent of $\hat{A}_a a^n$ in Eq. (9)
            ffn_update_intrinsic_val = ffn_var(a_n_var)

            # --- BC-driven update for a_n (Learns $\hat{B}\hat{w}^n + \hat{A}_{BC}U_B^n$) ---
            # This term depends on U_B^n (via BC_Ctrl_n)
            bc_driven_a_update_val = torch.zeros_like(a_n_var)
            current_bc_features = bc_features_processed_dict[key]
            if current_bc_features is not None and ( hasattr(bc_to_a_update_layer, 'weight') or len(list(bc_to_a_update_layer.parameters())) > 0 ):
                bc_driven_a_update_val = bc_to_a_update_layer(current_bc_features)

            # --- New Term: - \Phi^T U_B^{n+1} ---
            # This directly implements the $-\Phi^T U_B^{n+1}$ term from Eq. (9)
            Phi_T_var_expanded = Phi_var.transpose(0, 1).unsqueeze(0).expand(batch_size, -1, -1)
            # U_B_np1_stacked is U_B^{n+1}
            U_B_np1_current_var = U_B_np1_stacked[:, i_var, :].unsqueeze(-1) # [batch, nx, 1] for current variable
            term_to_subtract_phiT_UBnp1 = torch.bmm(Phi_T_var_expanded, U_B_np1_current_var).squeeze(-1) # [batch, basis_dim]

            # --- Total update for a^{n+1} (Paper Eq. (9), first line in \hat{\Sigma}_lin) ---
            # a^{n+1} = a^n + [(\hat{A}_a - I)a^n]_{learned} + [\hat{B}\hat{w}^n + \hat{A}_{BC}U_B^n]_{learned} - \Phi^T U_B^{n+1}
            # where `a_update_attn_val + alpha_var * ffn_update_intrinsic_val` learns `(\hat{A}_a - I)a^n` (approx.)
            # and `bc_driven_a_update_val` learns `\hat{B}\hat{w}^n + \hat{A}_{BC}U_B^n` (approx.)
            a_next_var = a_n_var + \
                         a_update_attn_val + \
                         alpha_var * ffn_update_intrinsic_val + \
                         bc_driven_a_update_val - \
                         term_to_subtract_phiT_UBnp1
            a_next_dict[key] = a_next_var # This is a^{n+1}

            # --- Reconstruction of \hat{U}^{n+1} (Paper Eq. (9), second line in \hat{\Sigma}_lin, for time n+1) ---
            # $\hat{U}^{n+1} = U_B^{n+1} + \Phi a^{n+1}$
            U_B_np1_for_recon_var = U_B_np1_stacked[:, i_var, :].unsqueeze(-1) # This is U_B^{n+1} for the current variable
            Phi_expanded = Phi_var.unsqueeze(0).expand(batch_size, -1, -1)
            a_next_unsq = a_next_var.unsqueeze(-1) # This is a^{n+1}
            Phi_a_next_term = torch.bmm(Phi_expanded, a_next_unsq) # This is \Phi a^{n+1}
            U_hat_dict[key] = U_B_np1_for_recon_var + Phi_a_next_term # This is \hat{U}^{n+1}

        err_est = None
        if self.add_error_estimator:
            # Concatenate all a_next_var from a_next_dict for error estimation
            if a_next_dict: # Ensure dict is not empty
                a_next_combined_for_err_est = torch.cat(list(a_next_dict.values()), dim=-1)
                if hasattr(self, 'error_estimator') and self.error_estimator is not None : # Check if estimator exists
                    err_est = self.error_estimator(a_next_combined_for_err_est)


        return a_next_dict, U_hat_dict, err_est

    # MODIFIED forward
    def forward(self, a0_dict, BC_Ctrl_seq, T, params=None):
        """
        Main forward pass for the ROM over T time steps.
        Args:
            a0_dict (dict): Initial modal coefficients {key: [batch, basis_dim]}. (Corresponds to a^0 in Eq. (9))
            BC_Ctrl_seq (torch.Tensor): Sequence of boundary and control inputs. [batch, SequenceLength, bc_state_dim + num_controls]
                                      SequenceLength must be at least T.
            T (int): Number of time steps to simulate (i.e., number of a^{n+1} to compute).
        """
        a_current_dict = {} # This will hold a^n, initialized with mapped a^0
        for key_init in self.state_keys:
            a0_map_layer = self._get_layer(self.a0_mapping, key_init)
            a_current_dict[key_init] = a0_map_layer(a0_dict[key_init])

        U_hat_seq_dict = {key: [] for key in self.state_keys} # To store sequence of U_hat^{n+1}
        err_seq = [] if self.add_error_estimator else None

        # Check if BC_Ctrl_seq is long enough
        if BC_Ctrl_seq.shape[1] < T + 1 and T > 0 : # Need T+1 elements in BC_Ctrl_seq if we need BC_Ctrl^T for U_B^T
             print(f"Warning: BC_Ctrl_seq length ({BC_Ctrl_seq.shape[1]}) might be too short for T={T} steps if U_B^T is needed from BC_Ctrl^T.")
             # For computing a^T (which is a_next at t_step=T-1), we need U_B^T.
             # This implies BC_Ctrl_seq should have data up to index T.

        for t_step in range(T): # t_step from 0 to T-1; represents current time index 'n'
            # BC_Ctrl_n is for the current time step 'n'
            BC_Ctrl_n = BC_Ctrl_seq[:, t_step, :] # Input for U_B^n and w^n related terms (Eq. (9))

            # Prepare U_B^{n+1} for the current step's calculation (Eq. (9))
            if t_step + 1 < BC_Ctrl_seq.shape[1]: # Check if index t_step+1 is valid for BC_Ctrl_seq
                BC_Ctrl_np1 = BC_Ctrl_seq[:, t_step + 1, :]
                U_B_np1_stacked = self._compute_U_B(BC_Ctrl_np1) # This is U_B^{n+1}
            else:
                # Handle case where BC_Ctrl for n+1 is not available (e.g., at the very end of given BC_Ctrl_seq)
                print(f"Warning: BC_Ctrl data for t_step+1={t_step+1} not available in BC_Ctrl_seq (length {BC_Ctrl_seq.shape[1]}). "
                      f"Using zeros for U_B^{t_step+1}. This may impact accuracy for the last step(s).")
                # Fallback: create a zero tensor. The shape reference should be robust.
                ref_device = BC_Ctrl_n.device
                ref_dtype = BC_Ctrl_n.dtype
                current_batch_size = BC_Ctrl_n.shape[0]
                U_B_np1_stacked = torch.zeros(current_batch_size, self.num_state_vars, self.nx,
                                              device=ref_device, dtype=ref_dtype)

            # a_current_dict is a^n
            # BC_Ctrl_n is input for U_B^n and w^n related terms
            # U_B_np1_stacked is U_B^{n+1}
            a_next_dict_step, U_hat_dict_step, err_est_step = self.forward_step(
                a_current_dict,
                BC_Ctrl_n,
                U_B_np1_stacked,
                params
            )
            # a_next_dict_step is a^{n+1}
            # U_hat_dict_step is U_hat^{n+1} (reconstructed using U_B^{n+1} and a^{n+1})

            for key_store in self.state_keys:
                U_hat_seq_dict[key_store].append(U_hat_dict_step[key_store]) # Appending U_hat^{n+1}

            if self.add_error_estimator and err_est_step is not None:
                err_seq.append(err_est_step)

            a_current_dict = a_next_dict_step # Update a^n to a^{n+1} for the next loop iteration

        # Convert lists of tensors to stacked tensors for output
        for key_out in self.state_keys:
            if U_hat_seq_dict[key_out]:
                # Each element is [batch, nx, 1], stack to [T, batch, nx, 1]
                stacked_tensor = torch.stack(U_hat_seq_dict[key_out], dim=0)
                # Permute and squeeze to [batch, T, nx]
                U_hat_seq_dict[key_out] = stacked_tensor.squeeze(-1).permute(1, 0, 2)
            else: # Handle T=0 case
                batch_size_fallback = BC_Ctrl_seq.shape[0] if BC_Ctrl_seq is not None and BC_Ctrl_seq.nelement() > 0 else 1
                device_fallback = BC_Ctrl_seq.device if BC_Ctrl_seq is not None and BC_Ctrl_seq.nelement() > 0 else 'cpu'
                U_hat_seq_dict[key_out] = torch.empty(batch_size_fallback, 0, self.nx, device=device_fallback)


        if self.add_error_estimator and err_seq and err_seq[0] is not None: # Check if err_seq is populated
            err_seq = torch.stack(err_seq, dim=1) # [batch_size, T, 1] or [batch_size, T] if err_est is scalar per step

        return U_hat_seq_dict, err_seq

    def get_basis(self, key):
        return self.Phi[key]


def train_multivar_model(model, data_loader, dataset_type, train_model_steps_T_arg, # Renamed for clarity
                         lr=1e-3, num_epochs=50, device='cuda',
                         checkpoint_path='rom_checkpoint.pt', lambda_res=0.05,
                         lambda_orth=0.001, lambda_bc_penalty=0.01,
                         clip_grad_norm=1.0):
    model.to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
    mse_loss = nn.MSELoss()
    best_val_loss = float('inf')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    start_epoch = 0
    # ... (checkpoint loading logic remains the same) ...
    if os.path.exists(checkpoint_path):
        print(f"Loading existing checkpoint from {checkpoint_path} ...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
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

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        count = 0

        for i, (state_data, BC_Ctrl_tensor, norm_factors) in enumerate(data_loader):
            if isinstance(state_data, list):
                state_tensors = [s.to(device) for s in state_data] # state_tensors[k] is U^0...U^{N_pts-1}
                batch_size, nt_data_loader_points, nx = state_tensors[0].shape
            else:
                state_tensors = [state_data.to(device)]
                batch_size, nt_data_loader_points, nx = state_tensors[0].shape

            BC_Ctrl_tensor = BC_Ctrl_tensor.to(device) # Shape [batch, N_pts, bc_dim]

            if nt_data_loader_points != train_model_steps_T_arg + 1:
                raise ValueError(
                    f"Data loader provided {nt_data_loader_points} points. "
                    f"For {train_model_steps_T_arg} model steps, expected {train_model_steps_T_arg + 1} data points "
                    f"(U^0 to U^{{train_model_steps_T_arg}} and BC_Ctrl^0 to BC_Ctrl^{{train_model_steps_T_arg}})."
                )

            optimizer.zero_grad()

            a0_dict = {}
            BC_ctrl_combined_t0 = BC_Ctrl_tensor[:, 0, :] # BC_Ctrl^0
            U_B_stacked_t0 = model._compute_U_B(BC_ctrl_combined_t0) # U_B^0

            for k, key in enumerate(state_keys):
                U0_var = state_tensors[k][:, 0, :].unsqueeze(-1) # U^0
                U_B_t0_var = U_B_stacked_t0[:, k, :].unsqueeze(-1)
                U0_star_var = U0_var - U_B_t0_var

                Phi_var = model.get_basis(key)
                Phi_T_var = Phi_var.transpose(0, 1).unsqueeze(0).expand(batch_size, -1, -1)
                a0_dict[key] = torch.bmm(Phi_T_var, U0_star_var).squeeze(-1) # a^0


            U_hat_seq_dict, _ = model(a0_dict, BC_Ctrl_tensor, T=train_model_steps_T_arg, params=None)
            # U_hat_seq_dict[key] contains U_hat^1, ..., U_hat^{train_model_steps_T_arg}

            total_batch_loss = 0.0
            mse_recon_loss_val = 0.0
            residual_orth_loss_val = 0.0
            orth_loss_val = 0.0
            boundary_penalty_val = 0.0

            for k, key in enumerate(state_keys):
                Phi_var = model.get_basis(key)
                Phi_T_var = Phi_var.transpose(0, 1).unsqueeze(0).expand(batch_size, -1, -1)


                pred_seq_var = U_hat_seq_dict[key]


                targets_for_loss = state_tensors[k][:, 1 : train_model_steps_T_arg + 1, :]

                if pred_seq_var.shape[1] != targets_for_loss.shape[1]:
                    raise ValueError(f"Shape mismatch for loss: preds {pred_seq_var.shape}, targets {targets_for_loss.shape}")

                current_var_mse_loss = mse_loss(pred_seq_var, targets_for_loss)

                current_var_res_orth_loss = 0
                if lambda_res > 0 and pred_seq_var.shape[1] > 0: # If there are steps to compare
                    for t_loss in range(train_model_steps_T_arg): # Iterate through each predicted step
                        # r_t = U_true^{t+1} - U_hat^{t+1}
                        r_t = targets_for_loss[:, t_loss, :].unsqueeze(-1) - pred_seq_var[:, t_loss, :].unsqueeze(-1) #[B,Nx,1]
                        r_proj_t = torch.bmm(Phi_T_var, r_t) # [B, basis_dim, 1]
                        current_var_res_orth_loss += mse_loss(r_proj_t, torch.zeros_like(r_proj_t))
                    current_var_res_orth_loss /= train_model_steps_T_arg

                mse_recon_loss_val += current_var_mse_loss
                residual_orth_loss_val += current_var_res_orth_loss

                if lambda_orth > 0:
                    PhiT_Phi = torch.matmul(Phi_var.transpose(0, 1), Phi_var)
                    I = torch.eye(model.basis_dim, device=device)
                    orth_loss_val += torch.norm(PhiT_Phi - I, p='fro')**2
                if lambda_bc_penalty > 0: # Penalty for \phi_i|_{x_boundary} = 0 from Eq (7c)
                    if Phi_var.shape[0] > 1:
                        boundary_penalty_val += mse_loss(Phi_var[0, :], torch.zeros_like(Phi_var[0, :])) + \
                                             mse_loss(Phi_var[-1, :], torch.zeros_like(Phi_var[-1, :]))
                    elif Phi_var.shape[0] == 1:
                         boundary_penalty_val += mse_loss(Phi_var[0, :], torch.zeros_like(Phi_var[0, :]))

            mse_recon_loss_val /= len(state_keys)
            residual_orth_loss_val /= len(state_keys) # not used in experiment
            orth_loss_val /= len(state_keys)
            boundary_penalty_val /= len(state_keys)

            total_batch_loss = mse_recon_loss_val + \
                               lambda_res * residual_orth_loss_val + \
                               lambda_orth * orth_loss_val + \
                               lambda_bc_penalty * boundary_penalty_val
            if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
                print(f"NaN/Inf loss detected at epoch {epoch+1}, batch {i+1}. Skipping batch.")
                optimizer.zero_grad()
                continue

            total_batch_loss.backward()
            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            optimizer.step()

            epoch_loss += total_batch_loss.item()
            count += 1

        if count > 0:
            avg_epoch_loss = epoch_loss / count
            print(f"Epoch {epoch+1}/{num_epochs} finished. Average Training Loss: {avg_epoch_loss:.6f}")
            scheduler.step(avg_epoch_loss)
            if avg_epoch_loss < best_val_loss:
                best_val_loss = avg_epoch_loss
                print(f"Saving checkpoint with loss {best_val_loss:.6f} to {checkpoint_path}")
                save_dict = {
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'loss': best_val_loss,
                    'dataset_type': dataset_type, 'state_keys': model.state_keys, 'nx': model.nx,
                    'basis_dim': model.basis_dim, 'd_model': model.d_model,
                    'bc_state_dim': model.bc_state_dim,
                    'num_controls': model.num_controls, 'num_heads': model.num_heads,
                    'shared_attention': model.shared_attention,
                    'use_fixed_lifting': model.use_fixed_lifting,
                    'bc_processed_dim': model.bc_processed_dim,
                    'hidden_bc_processor_dim': getattr(model, 'hidden_bc_processor_dim', 128) # Add if exists
                }
                torch.save(save_dict, checkpoint_path)
        else:
            print(f"Epoch {epoch+1}/{num_epochs} finished. No batches processed successfully.")

    print("Training finished.")
    if os.path.exists(checkpoint_path):
        print(f"Loading best model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    return model

def validate_multivar_model(model, data_loader, dataset_type,
                            train_nt_for_model_training: int,
                            T_value_for_model_training: float,
                            full_T_in_datafile: float,
                            full_nt_in_datafile: int, device='cuda',
                            save_fig_path='barom_result.png'):
    model.eval()
    results = {key: {'mse': [], 'rmse': [], 'relative_error': [], 'max_error': []}
               for key in model.state_keys}
    overall_rel_err_at_train_T = []

    test_horizons_T_values = [T_value_for_model_training]
    if full_T_in_datafile > T_value_for_model_training:
         mid_T = T_value_for_model_training + 0.5 * (full_T_in_datafile - T_value_for_model_training)
         if mid_T < full_T_in_datafile : test_horizons_T_values.append(mid_T)
         test_horizons_T_values.append(full_T_in_datafile)
    test_horizons_T_values = sorted(list(set(h for h in test_horizons_T_values if h <= full_T_in_datafile)))

    print(f"Validation Horizons (T values): {test_horizons_T_values}")
    print(f"Model was trained with T_model_steps={train_nt_for_model_training} for physical T={T_value_for_model_training}")
    print(f"Datafile contains nt={full_nt_in_datafile} for physical T={full_T_in_datafile}")

    try:
        state_data_full_batch, BC_Ctrl_tensor_full_batch, norm_factors_batch = next(iter(data_loader))
    except StopIteration:
        print("No validation data. Skipping validation.")
        return

    if isinstance(state_data_full_batch, list):
        state_tensors_full_sample = [s[0:1].to(device) for s in state_data_full_batch]
        _, nt_loaded_from_val_loader, nx_loaded = state_tensors_full_sample[0].shape
    else:
        state_tensors_full_sample = [state_data_full_batch[0:1].to(device)]
        _, nt_loaded_from_val_loader, nx_loaded = state_tensors_full_sample[0].shape

    BC_Ctrl_tensor_full_sample = BC_Ctrl_tensor_full_batch[0:1].to(device) # [1, full_loader_nt, bc_dim]


    if nt_loaded_from_val_loader != full_nt_in_datafile:
        print(f"Warning: nt from val_loader ({nt_loaded_from_val_loader}) != full_nt_in_datafile ({full_nt_in_datafile}). Adjusting.")

        full_nt_in_datafile = nt_loaded_from_val_loader


    norm_factors_sample = {}
    for key_nf, val_tensor_nf in norm_factors_batch.items():
        if isinstance(val_tensor_nf, (torch.Tensor, np.ndarray)) and val_tensor_nf.ndim > 0:
            norm_factors_sample[key_nf] = val_tensor_nf[0]
        else:
            norm_factors_sample[key_nf] = val_tensor_nf

    state_keys = model.state_keys
    current_batch_size = 1 # Processing one sample from the batch

    a0_dict = {}
    BC0_full = BC_Ctrl_tensor_full_sample[:, 0, :]
    U_B0_lifted = model._compute_U_B(BC0_full)
    for k_var_idx, key_a0 in enumerate(state_keys):
        U0_full_var = state_tensors_full_sample[k_var_idx][:, 0, :].unsqueeze(-1)
        U_B0_var = U_B0_lifted[:, k_var_idx, :].unsqueeze(-1)
        Phi = model.get_basis(key_a0).to(device)
        Phi_T = Phi.transpose(0, 1).unsqueeze(0)
        a0 = torch.bmm(Phi_T, U0_full_var - U_B0_var).squeeze(-1)
        a0_dict[key_a0] = a0

    for T_test_horizon_physical in test_horizons_T_values:
        if full_T_in_datafile > 1e-6 : # Avoid division by zero if full_T is very small
            dt_approx = full_T_in_datafile / (full_nt_in_datafile -1) if full_nt_in_datafile > 1 else full_T_in_datafile
            nt_steps_for_this_horizon = int(round(T_test_horizon_physical / dt_approx))
        else:
            nt_steps_for_this_horizon = full_nt_in_datafile -1 if T_test_horizon_physical > 0 else 0

        nt_steps_for_this_horizon = min(nt_steps_for_this_horizon, full_nt_in_datafile - 1) # Cap at available steps
        nt_points_for_this_horizon = nt_steps_for_this_horizon + 1 # Number of points including t=0

        print(f"\n--- Validating for Physical T_horizon = {T_test_horizon_physical:.2f} (Model steps = {nt_steps_for_this_horizon}, Points = {nt_points_for_this_horizon}) ---")

        if nt_steps_for_this_horizon <= 0:
            print("  Skipping this horizon as number of model steps is zero or less.")
            continue

        # BC_Ctrl_seq for model needs to cover up to nt_steps_for_this_horizon for U_B^{n+1} term
        # If T_model = nt_steps_for_this_horizon, it predicts U_hat^1...U_hat^{nt_steps_for_this_horizon}
        required_bc_ctrl_len = nt_steps_for_this_horizon + 1
        if BC_Ctrl_tensor_full_sample.shape[1] < required_bc_ctrl_len:
            print(f"  Warning: Not enough BC_Ctrl data ({BC_Ctrl_tensor_full_sample.shape[1]}) for {required_bc_ctrl_len} required points. Validation might be inaccurate for this horizon.")
            # Use available data, model.forward has a fallback for missing U_B^{n+1}
            sliced_bc_ctrl_len = BC_Ctrl_tensor_full_sample.shape[1]
        else:
            sliced_bc_ctrl_len = required_bc_ctrl_len

        BC_seq_for_pred = BC_Ctrl_tensor_full_sample[:, :sliced_bc_ctrl_len, :]

        with torch.no_grad():
             U_hat_seq_dict, _ = model(a0_dict, BC_seq_for_pred, T=nt_steps_for_this_horizon)

        combined_pred_denorm_list = []
        combined_gt_denorm_list = []

        num_vars_plot = len(state_keys)
        fig, axs = plt.subplots(num_vars_plot, 3, figsize=(18, 5 * num_vars_plot), squeeze=False)
        L_vis = 1.0 # Assuming L=1.0, adjust if dynamic

        for k_var_idx, key_val in enumerate(state_keys):
            # U_hat_seq_dict[key_val] is [batch, nt_steps_for_this_horizon, nx]
            # These are predictions for U_hat^1, ..., U_hat^{nt_steps_for_this_horizon}
            pred_norm_seq = U_hat_seq_dict[key_val].squeeze(0).cpu().numpy() # [nt_steps, nx]

            mean_k_val = norm_factors_sample[f'{key_val}_mean']
            std_k_val = norm_factors_sample[f'{key_val}_std']
            mean_k = mean_k_val.item() if hasattr(mean_k_val, 'item') else mean_k_val
            std_k = std_k_val.item() if hasattr(std_k_val, 'item') else std_k_val
            pred_denorm = pred_norm_seq * std_k + mean_k # Denormalized U_hat^1 ... U_hat^{nt_steps}

            # Ground Truth: U_true^1, ..., U_true^{nt_steps_for_this_horizon}

            gt_norm_full_var = state_tensors_full_sample[k_var_idx].squeeze(0).cpu().numpy()
            # Slice for U_true^1 to U_true^{nt_steps_for_this_horizon}
            gt_norm_sliced_for_comp = gt_norm_full_var[1 : nt_steps_for_this_horizon + 1, :]
            gt_denorm = gt_norm_sliced_for_comp * std_k + mean_k

            if pred_denorm.shape[0] != gt_denorm.shape[0]:
                print(f"  Warning: Shape mismatch for '{key_val}' at T={T_test_horizon_physical:.2f}. "
                      f"Pred len: {pred_denorm.shape[0]}, GT len: {gt_denorm.shape[0]}. Truncating for metrics.")
                min_len = min(pred_denorm.shape[0], gt_denorm.shape[0])
                pred_denorm = pred_denorm[:min_len, :]
                gt_denorm = gt_denorm[:min_len, :]
                if min_len == 0:
                    print(f"  Skipping metrics for '{key_val}' due to zero length after truncation.")
                    continue


            combined_pred_denorm_list.append(pred_denorm.flatten())
            combined_gt_denorm_list.append(gt_denorm.flatten())

            mse_k = np.mean((pred_denorm - gt_denorm)**2) if pred_denorm.size > 0 else 0
            rmse_k = np.sqrt(mse_k) if pred_denorm.size > 0 else 0
            norm_gt = np.linalg.norm(gt_denorm, 'fro')
            rel_err_k = np.linalg.norm(pred_denorm - gt_denorm, 'fro') / (norm_gt + 1e-10) if pred_denorm.size > 0 else 0
            max_err_k = np.max(np.abs(pred_denorm - gt_denorm)) if pred_denorm.size > 0 else 0

            print(f"  '{key_val}': MSE={mse_k:.3e}, RMSE={rmse_k:.3e}, RelErr={rel_err_k:.3e}, MaxErr={max_err_k:.3e}")

            if abs(T_test_horizon_physical - T_value_for_model_training) < 1e-5: # If it's the training T
                results[key_val]['mse'].append(mse_k)
                results[key_val]['rmse'].append(rmse_k)
                results[key_val]['relative_error'].append(rel_err_k)
                results[key_val]['max_error'].append(max_err_k)

            diff_plot = np.abs(pred_denorm - gt_denorm) if pred_denorm.size > 0 else np.array([[]])
            vmin_plot = min(gt_denorm.min(), pred_denorm.min()) if pred_denorm.size > 0 else 0
            vmax_plot = max(gt_denorm.max(), pred_denorm.max()) if pred_denorm.size > 0 else 1
            current_plot_T = pred_denorm.shape[0] * dt_approx if pred_denorm.size > 0 else 0


            im0 = axs[k_var_idx,0].imshow(gt_denorm, aspect='auto', origin='lower', vmin=vmin_plot, vmax=vmax_plot, extent=[0, L_vis, 0, current_plot_T], cmap='viridis')
            axs[k_var_idx,0].set_title(f"GT ({key_val}) T={T_test_horizon_physical:.1f}"); axs[k_var_idx,0].set_ylabel("t (physical)")
            plt.colorbar(im0, ax=axs[k_var_idx,0])

            im1 = axs[k_var_idx,1].imshow(pred_denorm, aspect='auto', origin='lower', vmin=vmin_plot, vmax=vmax_plot, extent=[0, L_vis, 0, current_plot_T], cmap='viridis')
            axs[k_var_idx,1].set_title(f"Pred ({key_val}) T={T_test_horizon_physical:.1f}")
            plt.colorbar(im1, ax=axs[k_var_idx,1])

            im2 = axs[k_var_idx,2].imshow(diff_plot, aspect='auto', origin='lower', extent=[0, L_vis, 0, current_plot_T], cmap='magma')
            axs[k_var_idx,2].set_title(f"Error ({key_val}) (Max {max_err_k:.2e})")
            plt.colorbar(im2, ax=axs[k_var_idx,2])
            for j_plot in range(3): axs[k_var_idx,j_plot].set_xlabel("x")

        if combined_pred_denorm_list and combined_gt_denorm_list: # Ensure lists are not empty
            cat_preds = np.concatenate(combined_pred_denorm_list)
            cat_gts = np.concatenate(combined_gt_denorm_list)
            if cat_gts.size > 0 : # Ensure not dividing by zero norm
                overall_rel_err_horizon = np.linalg.norm(cat_preds - cat_gts) / (np.linalg.norm(cat_gts) + 1e-10)
                print(f"  Overall RelErr for T={T_test_horizon_physical:.1f}: {overall_rel_err_horizon:.3e}")
                if abs(T_test_horizon_physical - T_value_for_model_training) < 1e-5:
                    overall_rel_err_at_train_T.append(overall_rel_err_horizon)
            else:
                print(f"  Could not compute overall RelErr for T={T_test_horizon_physical:.1f} due to empty GT data.")


        fig.suptitle(f"Validation @ Physical T={T_test_horizon_physical:.1f} ({dataset_type.upper()}) — basis={model.basis_dim}, d_model={model.d_model}")
        fig.tight_layout(rect=[0,0.03,1,0.95])
        horizon_fig_path = save_fig_path.replace('.png', f'_PhysT{str(T_test_horizon_physical).replace(".","p")}.png')
        plt.savefig(horizon_fig_path)
        print(f"Saved validation figure to: {horizon_fig_path}")
        plt.show()

    print(f"\n--- Validation Summary (Metrics for Physical T={T_value_for_model_training:.1f}) ---")
    for key_sum in state_keys:
        if results[key_sum]['mse']:
            avg_mse = np.mean(results[key_sum]['mse'])
            avg_rmse = np.mean(results[key_sum]['rmse'])
            avg_rel = np.mean(results[key_sum]['relative_error'])
            avg_max = np.mean(results[key_sum]['max_error'])
            print(f"  {key_sum}: MSE={avg_mse:.4e}, RMSE={avg_rmse:.4e}, RelErr={avg_rel:.4e}, MaxErr={avg_max:.4e}")
    if overall_rel_err_at_train_T:
        print(f"Overall Avg RelErr for Physical T={T_value_for_model_training:.1f}: {np.mean(overall_rel_err_at_train_T):.4e}")


# main script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ROM models with various configurations.")
    parser.add_argument('--datatype', type=str, required=True,
                        choices=['heat_delayed_feedback', 'reaction_diffusion_neumann_feedback',
                                 'heat_nonlinear_feedback_gain', 'convdiff'],
                        help='Type of dataset to use.')
    parser.add_argument('--use_fixed_lifting', action='store_true', help='Use fixed linear interpolation for U_B.')
    parser.add_argument('--random_phi_init', action='store_true', help='Use random orthogonal initialization for Phi.')
    parser.add_argument('--basis_dim', type=int, default=32, help='Dimension of the reduced basis.')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension for Attention ROM.')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of heads for Attention ROM.')
    parser.add_argument('--bc_processed_dim', type=int, default=64, help='Dimension of processed BC features.')
    parser.add_argument('--hidden_bc_processor_dim', type=int, default=128, help='Hidden dimension for BC feature processor MLP.')
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    args = parser.parse_args()
    DATASET_TYPE = args.datatype
    USE_FIXED_LIFTING_ABLATION = args.use_fixed_lifting
    RANDOM_PHI_INIT_ABLATION = args.random_phi_init

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learning_rate = args.lr
    num_epochs = args.num_epochs
    lambda_res = 0
    lambda_orth = 0.001
    lambda_bc_penalty = 0.01
    clip_grad_norm = 1.0

    print(f"Using device: {device}")
    print(f"Selected Dataset Type: {DATASET_TYPE.upper()}")
    print(f"Using fixed lifting: {USE_FIXED_LIFTING_ABLATION}")
    print(f"Using random Phi init: {RANDOM_PHI_INIT_ABLATION}")

    if DATASET_TYPE in ['heat_delayed_feedback', 'reaction_diffusion_neumann_feedback', 'heat_nonlinear_feedback_gain', 'convdiff']:
        FULL_T_IN_DATAFILE = 2.0
        FULL_NT_IN_DATAFILE = 300 # Number of points t=0 to t=299 (total 300 points)
        TRAIN_T_TARGET_PHYSICAL = 1.5
    else: # Default, adjust if other datasets are used
        FULL_T_IN_DATAFILE = 2.0
        FULL_NT_IN_DATAFILE = 600
        TRAIN_T_TARGET_PHYSICAL = 1.0

    # TRAIN_NT_DATA_LOADER_POINTS: Number of data points (U^0, ..., U^{N_pts-1}) loaded by DataLoader for training.
    # This should correspond to the physical training horizon TRAIN_T_TARGET_PHYSICAL.
    if FULL_NT_IN_DATAFILE <= 1:
        TRAIN_NT_DATA_LOADER_POINTS = FULL_NT_IN_DATAFILE
    else:

        TRAIN_NT_DATA_LOADER_POINTS = int(round((TRAIN_T_TARGET_PHYSICAL / FULL_T_IN_DATAFILE) * (FULL_NT_IN_DATAFILE - 1))) + 1

    # TRAIN_MODEL_STEPS_T: Number of steps a^n -> a^{n+1} the model will perform.

    TRAIN_MODEL_STEPS_T = TRAIN_NT_DATA_LOADER_POINTS - 1

    if TRAIN_MODEL_STEPS_T <= 0: # Ensure at least one step
        raise ValueError(
            f"Calculated TRAIN_MODEL_STEPS_T is {TRAIN_MODEL_STEPS_T}. "
            f"This must be positive. Check time parameter definitions. "
            f"TRAIN_NT_DATA_LOADER_POINTS was {TRAIN_NT_DATA_LOADER_POINTS}."
        )

    print(f"Full data physical T={FULL_T_IN_DATAFILE}, total data points nt={FULL_NT_IN_DATAFILE}")
    print(f"Training will use physical T={TRAIN_T_TARGET_PHYSICAL}, "
          f"loading {TRAIN_NT_DATA_LOADER_POINTS} data points (U^0 to U^{TRAIN_NT_DATA_LOADER_POINTS-1}).")
    print(f"Model will run for T_model_steps={TRAIN_MODEL_STEPS_T} steps, predicting U_hat^1 to U_hat^{TRAIN_MODEL_STEPS_T}.")



    timestamp = time.strftime("%Y%m%d-%H%M%S")
    suffix_lift = "_fixedlift" if USE_FIXED_LIFTING_ABLATION else ""
    suffix_phi = "_randphi" if RANDOM_PHI_INIT_ABLATION else ""
    suffix_rom = f"_attn_d{args.d_model}_bcp{args.bc_processed_dim}"
    run_name = f"{DATASET_TYPE}_b{args.basis_dim}{suffix_rom}{suffix_lift}{suffix_phi}_h{args.num_heads}_eq9_v2"

    checkpoint_dir = f"./New_ckpt_explicit_bc_eq9/_checkpoints_{DATASET_TYPE}"
    results_dir = f"./result_all_explicit_bc_eq9/results_{DATASET_TYPE}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'barom_{run_name}.pt')
    save_fig_path = os.path.join(results_dir, f'barom_result_{run_name}.png')
    basis_dir = os.path.join(checkpoint_dir, 'pod_bases')
    if not RANDOM_PHI_INIT_ABLATION:
        os.makedirs(basis_dir, exist_ok=True)

    # --- Dataset specific parameters and loading ---
    # ... (Dataset path logic remains the same) ...
    if DATASET_TYPE == 'heat_delayed_feedback':
        dataset_path = "./datasets_new_feedback/heat_delayed_feedback_v1_5000s_64nx_300nt.pkl"
        nx_data_param = 64; state_keys = ['U']
    elif DATASET_TYPE == 'reaction_diffusion_neumann_feedback':
        dataset_path = "./datasets_new_feedback/reaction_diffusion_neumann_feedback_v1_5000s_64nx_300nt.pkl"
        nx_data_param = 64; state_keys = ['U']
    elif DATASET_TYPE == 'heat_nonlinear_feedback_gain':
        dataset_path = "./datasets_new_feedback/heat_nonlinear_feedback_gain_v1_5000s_64nx_300nt.pkl"
        nx_data_param = 64; state_keys = ['U']
    elif DATASET_TYPE == 'convdiff':
        dataset_path = "./datasets_new_feedback/convdiff_v1_5000s_64nx_300nt.pkl"
        nx_data_param = 64; state_keys = ['U']
    else:
        raise ValueError(f"Unknown dataset_type: {DATASET_TYPE}")

    print(f"Loading dataset for {DATASET_TYPE} from {dataset_path}")
    try:
        with open(dataset_path, 'rb') as f: data_list = pickle.load(f)
        print(f"Loaded {len(data_list)} samples.")
    except FileNotFoundError: print(f"Error: Dataset file not found at {dataset_path}"); exit()
    if not data_list: print("No data generated, exiting."); exit()


    # --- Data splitting and loading ---
    # keep same seed for data split in both training and evaluation!!
    random.shuffle(data_list)
    n_total = len(data_list); n_train = int(0.8 * n_total)
    train_data_list = data_list[:n_train]; val_data_list = data_list[n_train:]
    print(f"Train samples: {len(train_data_list)}, Validation samples: {len(val_data_list)}")


    train_dataset = UniversalPDEDataset(train_data_list, dataset_type=DATASET_TYPE, train_nt_limit=TRAIN_NT_DATA_LOADER_POINTS)
    val_dataset = UniversalPDEDataset(val_data_list, dataset_type=DATASET_TYPE, train_nt_limit=None) # Val loads full sequence
    num_workers = 2
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)



    current_nx_model = train_dataset.nx
    model_num_controls = train_dataset.num_controls
    model_bc_state_dim = train_dataset.bc_state_dim

    print(f"Model Configuration: nx={current_nx_model}, basis_dim={args.basis_dim}")
    print(f"Model BC_State dim: {model_bc_state_dim}, Model Num Controls: {model_num_controls}")

    online_model = MultiVarAttentionROM(
        state_variable_keys=state_keys,
        nx=current_nx_model,
        basis_dim=args.basis_dim,
        d_model=args.d_model,
        bc_state_dim=model_bc_state_dim,
        num_controls=model_num_controls,
        num_heads=args.num_heads,
        add_error_estimator=False,
        shared_attention=False,
        use_fixed_lifting=USE_FIXED_LIFTING_ABLATION,
        bc_processed_dim=args.bc_processed_dim,
        hidden_bc_processor_dim=args.hidden_bc_processor_dim
    ).to(device)

    if not RANDOM_PHI_INIT_ABLATION:
        print("\nInitializing Phi with POD bases...")
        pod_bases = {}
        actual_nt_for_pod_computation = TRAIN_NT_DATA_LOADER_POINTS
        print(f"Computing POD using nt={actual_nt_for_pod_computation} points from training data.")
        for key_pod_loop in state_keys:
            basis_filename = f'pod_basis_{key_pod_loop}_nx{current_nx_model}_nt{actual_nt_for_pod_computation}_bdim{args.basis_dim}.npy'
            basis_path = os.path.join(basis_dir, basis_filename)
            loaded_basis = None
            if os.path.exists(basis_path):
                print(f"  Loading existing POD basis for '{key_pod_loop}' from {basis_path}...")
                try: loaded_basis = np.load(basis_path)
                except Exception as e: print(f"  Error loading {basis_path}: {e}. Will recompute."); loaded_basis = None
                if loaded_basis is not None and loaded_basis.shape != (current_nx_model, args.basis_dim):
                    print(f"  Shape mismatch for loaded basis '{key_pod_loop}'. Expected ({current_nx_model}, {args.basis_dim}), got {loaded_basis.shape}. Recomputing."); loaded_basis = None
            if loaded_basis is None:
                print(f"  Computing POD basis for '{key_pod_loop}' (using nt={actual_nt_for_pod_computation}, basis_dim={args.basis_dim})...")
                computed_basis = compute_pod_basis_generic(
                    data_list=train_data_list, dataset_type=DATASET_TYPE, state_variable_key=key_pod_loop,
                    nx=current_nx_model, nt=actual_nt_for_pod_computation, basis_dim=args.basis_dim)
                if computed_basis is not None:
                    pod_bases[key_pod_loop] = computed_basis
                    os.makedirs(os.path.dirname(basis_path), exist_ok=True)
                    np.save(basis_path, computed_basis)
                    print(f"  Saved computed POD basis for '{key_pod_loop}' to {basis_path}")
                else: print(f"ERROR: Failed to compute POD basis for '{key_pod_loop}'. Exiting."); exit()
            else: pod_bases[key_pod_loop] = loaded_basis
        with torch.no_grad():
            for key_phi_init in state_keys:
                if key_phi_init in pod_bases and hasattr(online_model, 'Phi') and key_phi_init in online_model.Phi:
                    model_phi_param = online_model.Phi[key_phi_init]
                    pod_phi_tensor = torch.tensor(pod_bases[key_phi_init].astype(np.float32), device=model_phi_param.device)
                    if model_phi_param.shape == pod_phi_tensor.shape:
                        model_phi_param.copy_(pod_phi_tensor); print(f"  Initialized Phi for '{key_phi_init}' with POD.")
                    else: print(f"  WARNING: Shape mismatch for Phi '{key_phi_init}'. Expected {model_phi_param.shape}, got {pod_phi_tensor.shape}. Using random init.")
                else: print(f"  WARNING: No POD basis found or Phi module not present for '{key_phi_init}'. Using random init for Phi.")

    else:
        print("\nSkipping POD basis initialization for Phi (using random orthogonal initialization).")


    print(f"\nStarting training for {DATASET_TYPE.upper()}...")
    start_train_time = time.time()
    online_model = train_multivar_model(
        online_model, train_loader, dataset_type=DATASET_TYPE,
        train_model_steps_T_arg=TRAIN_MODEL_STEPS_T, # Pass number of model steps
        lr=learning_rate, num_epochs=num_epochs, device=device,
        checkpoint_path=checkpoint_path, lambda_res=lambda_res,
        lambda_orth=lambda_orth, lambda_bc_penalty=lambda_bc_penalty,
        clip_grad_norm=clip_grad_norm
    )
    end_train_time = time.time()
    print(f"Training took {end_train_time - start_train_time:.2f} seconds.")

    if val_data_list:
        print(f"\nStarting validation for {DATASET_TYPE.upper()}...")
        validate_multivar_model(
            online_model, val_loader, dataset_type=DATASET_TYPE, device=device,
            save_fig_path=save_fig_path,
            train_nt_for_model_training=TRAIN_MODEL_STEPS_T,
            T_value_for_model_training=TRAIN_T_TARGET_PHYSICAL,
            full_T_in_datafile=FULL_T_IN_DATAFILE,
            full_nt_in_datafile=FULL_NT_IN_DATAFILE
        )
    else:
        print("\nNo validation data. Skipping validation.")

    print(f"\n--- Script setup complete for run: {run_name} ---")
    print(f"--- Model: {online_model.__class__.__name__} ---")
    print("="*60)
    print(f"Run configuration finished for dataset: {DATASET_TYPE.upper()} - {run_name}")
    print(f"Target checkpoint path: {checkpoint_path}")
    if val_data_list: print(f"Target validation figure path: {save_fig_path}")
    print("="*60)
