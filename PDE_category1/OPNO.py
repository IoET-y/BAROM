# =============================================================================
#       COMPLETE CODE: OPNO Baseline Adapted for Task
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
import pickle
import torch.fft
import pickle
import argparse
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve # For Burgers' solver
# ---------------------
# import functools # Not strictly needed for this version

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
# 1. DATASET CLASS (UniversalPDEDataset - Corrected Version)
# =============================================================================
# class UniversalPDEDataset(Dataset):
#     def __init__(self, data_list, dataset_type, train_nt_limit=None): # Added train_nt_limit
#         if not data_list:
#             raise ValueError("data_list cannot be empty")
#         self.data_list = data_list
#         self.dataset_type = dataset_type.lower()
#         self.train_nt_limit = train_nt_limit

#         first_sample = data_list[0]
#         params = first_sample.get('params', {})

#         self.nt_from_sample_file = 0
#         self.nx_from_sample_file = 0
#         self.ny_from_sample_file = 1

#         if self.dataset_type == 'advection' or self.dataset_type == 'burgers':
#             self.nt_from_sample_file = first_sample['U'].shape[0]
#             self.nx_from_sample_file = first_sample['U'].shape[1]
#             self.state_keys = ['U']; self.num_state_vars = 1
#             self.expected_bc_state_dim = 2
#         elif self.dataset_type == 'euler':
#             self.nt_from_sample_file = first_sample['rho'].shape[0]
#             self.nx_from_sample_file = first_sample['rho'].shape[1]
#             self.state_keys = ['rho', 'u']; self.num_state_vars = 2
#             self.expected_bc_state_dim = 4
#         elif self.dataset_type == 'darcy':
#             self.nt_from_sample_file = first_sample['P'].shape[0]
#             self.nx_from_sample_file = params.get('nx', first_sample['P'].shape[1])
#             self.ny_from_sample_file = params.get('ny', 1)
#             self.state_keys = ['P']; self.num_state_vars = 1
#             self.expected_bc_state_dim = 2
#         else:
#             raise ValueError(f"Unknown dataset_type: {self.dataset_type}")

#         self.effective_nt_for_loader = self.train_nt_limit if self.train_nt_limit is not None else self.nt_from_sample_file

#         self.nx = self.nx_from_sample_file
#         self.ny = self.ny_from_sample_file
#         self.spatial_dim = self.nx * self.ny

#         self.bc_state_key = 'BC_State'
#         if self.bc_state_key not in first_sample:
#             raise KeyError(f"'{self.bc_state_key}' not found in the first sample!")
#         actual_bc_state_dim = first_sample[self.bc_state_key].shape[1]
#         if actual_bc_state_dim != self.expected_bc_state_dim:
#               print(f"Warning: BC_State dimension mismatch for {self.dataset_type}. "
#                     f"Expected {self.expected_bc_state_dim}, got {actual_bc_state_dim}. "
#                     f"Using actual dimension: {actual_bc_state_dim}")
#               self.bc_state_dim = actual_bc_state_dim
#         else:
#               self.bc_state_dim = self.expected_bc_state_dim

#         self.bc_control_key = 'BC_Control'
#         if self.bc_control_key in first_sample and first_sample[self.bc_control_key] is not None and first_sample[self.bc_control_key].size > 0 :
#             self.num_controls = first_sample[self.bc_control_key].shape[1]
#         else:
#             self.num_controls = 0

#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, idx):
#         sample = self.data_list[idx]
#         norm_factors = {}
#         current_nt_for_item = self.effective_nt_for_loader
#         state_tensors_norm_list = []

#         for key in self.state_keys:
#             try:
#                 state_seq_full = sample[key]
#                 state_seq = state_seq_full[:current_nt_for_item, ...]
#             except KeyError:
#                 raise KeyError(f"State variable key '{key}' not found in sample {idx} for dataset type '{self.dataset_type}'")
#             if state_seq.shape[0] != current_nt_for_item:
#                  raise ValueError(f"Time dimension mismatch for {key}. Expected {current_nt_for_item}, got {state_seq.shape[0]}")

#             state_mean = np.mean(state_seq)
#             state_std = np.std(state_seq) + 1e-8
#             state_norm = (state_seq - state_mean) / state_std
#             state_tensors_norm_list.append(torch.tensor(state_norm).float())
#             norm_factors[f'{key}_mean'] = state_mean
#             norm_factors[f'{key}_std'] = state_std

#         bc_state_seq_full = sample[self.bc_state_key]
#         bc_state_seq = bc_state_seq_full[:current_nt_for_item, :]
#         if bc_state_seq.shape[0] != current_nt_for_item:
#             raise ValueError(f"Time dim mismatch for BC_State. Expected {current_nt_for_item}, got {bc_state_seq.shape[0]}")

#         bc_state_norm = np.zeros_like(bc_state_seq, dtype=np.float32)
#         norm_factors[f'{self.bc_state_key}_means'] = np.zeros(self.bc_state_dim)
#         norm_factors[f'{self.bc_state_key}_stds'] = np.ones(self.bc_state_dim)
#         for k_dim in range(self.bc_state_dim):
#             col = bc_state_seq[:, k_dim]
#             mean_k = np.mean(col)
#             std_k = np.std(col)
#             if std_k > 1e-8:
#                 bc_state_norm[:, k_dim] = (col - mean_k) / std_k
#                 norm_factors[f'{self.bc_state_key}_means'][k_dim] = mean_k
#                 norm_factors[f'{self.bc_state_key}_stds'][k_dim] = std_k
#             else:
#                 bc_state_norm[:, k_dim] = 0.0
#                 norm_factors[f'{self.bc_state_key}_means'][k_dim] = mean_k
#         bc_state_tensor_norm = torch.tensor(bc_state_norm).float()

#         if self.num_controls > 0:
#             try:
#                 bc_control_seq_full = sample[self.bc_control_key]
#                 bc_control_seq = bc_control_seq_full[:current_nt_for_item, :]
#                 if bc_control_seq.shape[0] != current_nt_for_item:
#                     raise ValueError(f"Time dim mismatch for BC_Control. Expected {current_nt_for_item}, got {bc_control_seq.shape[0]}")
#                 if bc_control_seq.shape[1] != self.num_controls:
#                      raise ValueError(f"Control dim mismatch in sample {idx}. Expected {self.num_controls}, got {bc_control_seq.shape[1]}.")
#                 bc_control_norm = np.zeros_like(bc_control_seq, dtype=np.float32)
#                 norm_factors[f'{self.bc_control_key}_means'] = np.zeros(self.num_controls)
#                 norm_factors[f'{self.bc_control_key}_stds'] = np.ones(self.num_controls)
#                 for k_dim in range(self.num_controls):
#                     col = bc_control_seq[:, k_dim]
#                     mean_k = np.mean(col)
#                     std_k = np.std(col)
#                     if std_k > 1e-8:
#                         bc_control_norm[:, k_dim] = (col - mean_k) / std_k
#                         norm_factors[f'{self.bc_control_key}_means'][k_dim] = mean_k
#                         norm_factors[f'{self.bc_control_key}_stds'][k_dim] = std_k
#                     else:
#                         bc_control_norm[:, k_dim] = 0.0
#                         norm_factors[f'{self.bc_control_key}_means'][k_dim] = mean_k
#                 bc_control_tensor_norm = torch.tensor(bc_control_norm).float()
#             except KeyError:
#                 bc_control_tensor_norm = torch.zeros((current_nt_for_item, self.num_controls), dtype=torch.float32)
#                 norm_factors[f'{self.bc_control_key}_means'] = np.zeros(self.num_controls)
#                 norm_factors[f'{self.bc_control_key}_stds'] = np.ones(self.num_controls)
#         else:
#             bc_control_tensor_norm = torch.empty((current_nt_for_item, 0), dtype=torch.float32)

#         bc_ctrl_tensor_norm = torch.cat((bc_state_tensor_norm, bc_control_tensor_norm), dim=-1)
#         output_state_tensors = state_tensors_norm_list[0] if self.num_state_vars == 1 else state_tensors_norm_list
#         return output_state_tensors, bc_ctrl_tensor_norm, norm_factors
class UniversalPDEDataset(Dataset):
    def __init__(self, data_list, dataset_type, stats_calculation_length, sequence_length_to_return=None):
        if not data_list:
            raise ValueError("data_list cannot be empty")
        self.data_list = data_list
        self.dataset_type = dataset_type.lower()

        first_sample = data_list[0]
        params=first_sample.get('params', {})
        self.nt_from_sample_file=0
        self.nx_from_sample_file=0
        self.ny_from_sample_file=1
        # ... (设置 self.nt_from_sample_file, self.nx_from_sample_file, etc. 如前) ...
        # (例如，如果dataset_type是 'advection', self.state_keys = ['U'], self.num_state_vars = 1 等)
        if self.dataset_type == 'advection' or self.dataset_type == 'burgers':
            self.nt_from_sample_file = first_sample['U'].shape[0]
            self.nx_from_sample_file = first_sample['U'].shape[1]
            self.state_keys = ['U']; self.num_state_vars = 1
            self.expected_bc_state_dim = 2 # 示例，具体根据您的数据
        elif self.dataset_type == 'euler':
            self.nt_from_sample_file = first_sample['rho'].shape[0]
            self.nx_from_sample_file = first_sample['rho'].shape[1]
            self.state_keys = ['rho', 'u']; self.num_state_vars = 2
            self.expected_bc_state_dim = 4 # 示例
        elif self.dataset_type == 'darcy':
            self.nt_from_sample_file=first_sample['P'].shape[0]
            self.nx_from_sample_file = params.get('nx', first_sample['P'].shape[1])
            self.ny_from_sample_file = params.get('ny', 1)
            self.state_keys = ['P'] 
            self.num_state_vars = 1
            self.expected_bc_state_dim = 2
        else:
            raise ValueError(f"Unknown dataset_type: {self.dataset_type}")

        self.nx = self.nx_from_sample_file # 确保 nx, ny 等属性被正确设置

        self.stats_calc_len = stats_calculation_length
        self.seq_len_to_return = sequence_length_to_return if sequence_length_to_return is not None else self.nt_from_sample_file

        self.bc_state_key = 'BC_State' # 确保这些key存在于您的数据中
        # ... (获取 bc_state_dim 和 num_controls 如前)
        # 在__init__中获取正确的bc_state_dim 和 num_controls
        if self.bc_state_key not in first_sample:
            raise KeyError(f"'{self.bc_state_key}' not found in the first sample!")
        self.bc_state_dim = first_sample[self.bc_state_key].shape[1] # 或者使用expected_bc_state_dim并做检查

        self.bc_control_key = 'BC_Control'
        if self.bc_control_key in first_sample and first_sample[self.bc_control_key] is not None and first_sample[self.bc_control_key].size > 0 :
            self.num_controls = first_sample[self.bc_control_key].shape[1]
        else:
            self.num_controls = 0

    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        norm_factors = {}
    
        # 安全地确定用于计算统计数据的实际长度
        # (确保 self.stats_calc_len 不超过样本的实际nt长度)
        current_sample_nt = sample[self.state_keys[0]].shape[0]
        effective_stats_len = min(self.stats_calc_len, current_sample_nt)
    
        # --- 1. 计算 norm_factors (基于 [0:effective_stats_len] 段) ---
        # 状态变量的 norm_factors
        temp_state_means = {}
        temp_state_stds = {}
        for key in self.state_keys:
            state_seq_for_stats = sample[key][:effective_stats_len, ...]
            mean_val = np.mean(state_seq_for_stats)
            std_val = np.std(state_seq_for_stats) + 1e-8
            temp_state_means[key] = mean_val
            temp_state_stds[key] = std_val
            norm_factors[f'{key}_mean'] = mean_val
            norm_factors[f'{key}_std'] = std_val
    
        # BC_State 的 norm_factors
        bc_state_seq_for_stats = sample[self.bc_state_key][:effective_stats_len, :]
        norm_factors[f'{self.bc_state_key}_means'] = np.zeros(self.bc_state_dim)
        norm_factors[f'{self.bc_state_key}_stds'] = np.ones(self.bc_state_dim)
        temp_bc_state_means = np.zeros(self.bc_state_dim)
        temp_bc_state_stds = np.ones(self.bc_state_dim)
        for k_dim in range(self.bc_state_dim):
            col_for_stats = bc_state_seq_for_stats[:, k_dim]
            mean_k = np.mean(col_for_stats)
            std_k = np.std(col_for_stats)
            temp_bc_state_means[k_dim] = mean_k
            norm_factors[f'{self.bc_state_key}_means'][k_dim] = mean_k
            if std_k > 1e-8:
                temp_bc_state_stds[k_dim] = std_k
                norm_factors[f'{self.bc_state_key}_stds'][k_dim] = std_k
            # else std_k 保持为1 (来自初始化)
    
        # BC_Control 的 norm_factors (如果存在)
        temp_bc_control_means = None
        temp_bc_control_stds = None
        if self.num_controls > 0:
            # 确保 sample[self.bc_control_key] 存在且有效
            if self.bc_control_key not in sample or sample[self.bc_control_key] is None or sample[self.bc_control_key].size == 0:
                 # 根据您的数据特性决定如何处理，这里假设如果声明了num_controls > 0，数据就应该存在
                raise ValueError(f"BC_Control data missing or empty in sample {idx} but num_controls={self.num_controls}")
    
            bc_control_seq_for_stats = sample[self.bc_control_key][:effective_stats_len, :]
            norm_factors[f'{self.bc_control_key}_means'] = np.zeros(self.num_controls)
            norm_factors[f'{self.bc_control_key}_stds'] = np.ones(self.num_controls)
            temp_bc_control_means = np.zeros(self.num_controls)
            temp_bc_control_stds = np.ones(self.num_controls)
    
            for k_dim in range(self.num_controls):
                col_for_stats = bc_control_seq_for_stats[:, k_dim]
                mean_k = np.mean(col_for_stats)
                std_k = np.std(col_for_stats)
                temp_bc_control_means[k_dim] = mean_k
                norm_factors[f'{self.bc_control_key}_means'][k_dim] = mean_k
                if std_k > 1e-8:
                    temp_bc_control_stds[k_dim] = std_k
                    norm_factors[f'{self.bc_control_key}_stds'][k_dim] = std_k
                # else std_k 保持为1
    
        # --- 2. 使用计算出的 norm_factors 归一化完整序列 ---
        normalized_full_state_tensors_list = []
        for key in self.state_keys:
            full_state_seq_original = sample[key]
            mean_val = temp_state_means[key] # 使用从 stats_len 计算的均值
            std_val = temp_state_stds[key]   # 使用从 stats_len 计算的标准差
            normalized_full_seq = (full_state_seq_original - mean_val) / std_val
            normalized_full_state_tensors_list.append(torch.tensor(normalized_full_seq).float())
    
        full_bc_state_original = sample[self.bc_state_key]
        bc_state_norm_full = np.zeros_like(full_bc_state_original, dtype=np.float32)
        for k_dim in range(self.bc_state_dim):
            mean_k = temp_bc_state_means[k_dim]
            std_k = temp_bc_state_stds[k_dim]
            bc_state_norm_full[:, k_dim] = (full_bc_state_original[:, k_dim] - mean_k) / std_k
        bc_state_tensor_norm_full = torch.tensor(bc_state_norm_full).float()
    
        bc_control_tensor_norm_full = torch.empty((current_sample_nt, 0), dtype=torch.float32)
        if self.num_controls > 0:
            full_bc_control_original = sample[self.bc_control_key]
            bc_control_norm_full_np = np.zeros_like(full_bc_control_original, dtype=np.float32)
            for k_dim in range(self.num_controls):
                mean_k = temp_bc_control_means[k_dim]
                std_k = temp_bc_control_stds[k_dim]
                bc_control_norm_full_np[:, k_dim] = (full_bc_control_original[:, k_dim] - mean_k) / std_k
            bc_control_tensor_norm_full = torch.tensor(bc_control_norm_full_np).float()
    
    
        # --- 3. 截取归一化后的序列到 self.seq_len_to_return ---
        effective_output_len = min(self.seq_len_to_return, current_sample_nt)
    
        output_state_tensors_list = []
        for norm_full_tensor in normalized_full_state_tensors_list:
            output_state_tensors_list.append(norm_full_tensor[:effective_output_len, ...])
    
        final_bc_state_tensor = bc_state_tensor_norm_full[:effective_output_len, :]
        final_bc_control_tensor = bc_control_tensor_norm_full[:effective_output_len, :]
    
        final_bc_ctrl_tensor_norm = torch.cat((final_bc_state_tensor, final_bc_control_tensor), dim=-1)
        output_state_tensors = output_state_tensors_list[0] if self.num_state_vars == 1 else output_state_tensors_list
    
        return output_state_tensors, final_bc_ctrl_tensor_norm, norm_factors
# =============================================================================
# 2. Chebypack Functions (Retained from your OPNO code)
# =============================================================================
def cheb_dct(u):
    Nx = u.shape[-1]
    if Nx <= 1: return u.clone()
    first = u[..., :-1]
    second = torch.flip(u[..., 1:], dims=[-1])
    V = torch.cat([first, second], dim=-1)
    a = torch.fft.rfft(V, dim=-1).real
    a[..., 0] /= 2; a[..., -1] /= 2
    return a

def cheb_idct(a):
    Nx = a.shape[-1]
    if Nx <= 1: return a.clone()
    a2 = a.clone(); a2[..., 0] /= 2; a2[..., -1] /= 2
    first = a2[..., :-1]; second = torch.flip(a2[..., 1:], dims=[-1])
    V = torch.cat([first, second], dim=-1)
    y = torch.fft.fft(V, dim=-1).real
    u = y[..., :Nx] / 2.0
    return u

def cheb_compact_dirichlet(a):
    Nx = a.shape[-1]
    if Nx <= 2: return a
    s = torch.zeros_like(a); s[..., ::2] = 1.0
    fft_len = 2 * Nx
    b = torch.fft.irfft(torch.fft.rfft(s, n=fft_len, dim=-1) * torch.fft.rfft(a, n=fft_len, dim=-1), n=fft_len, dim=-1)[..., :Nx]
    return b

# def cheb_inverse_compact_dirichlet(b):
#     Nx = b.shape[-1]
#     if Nx <= 2: return b
#     a = torch.zeros_like(b)
#     a[..., :2] = b[..., :2]
#     if Nx > 2 : a[..., 2:] = b[..., 2:] - b[..., :Nx-2] # Main recursion
#     # The original snippet's special handling for last two might be for specific matrix structures not used here.
#     # If a simpler N-2 length output from compact was assumed, then this inverse makes more sense.
#     # For now, this covers beta_j for j=0,1 and the recursion for j >= 2.
#     return a
def cheb_inverse_compact_dirichlet(b):
    Nx = b.shape[-1]
    if Nx <= 2: return b
    a = torch.zeros_like(b)
    a[..., :2] = b[..., :2]
    a[..., 2:] = b[..., 2:] - b[..., :Nx-2]
    a[..., -2:] = -b[..., Nx-4:Nx-2]
    return a
# Other compact/inverse compact (Neumann, Robin) retained as in your code if needed,
# but default is Dirichlet for OPNO example.

# =============================================================================
# Interpolation Placeholders & CGL Grid Generation
# =============================================================================
def get_cgl_points(N_intervals_plus_1): # N_intervals_plus_1 is total points (nx_cgl)
    """ Generates N_intervals_plus_1 Chebyshev-Gauss-Lobatto points in [-1, 1]. """
    if N_intervals_plus_1 <= 1: return np.array([0.0], dtype=np.float32) # Or [-1.0] if N=1
    return -np.cos(np.pi * np.arange(N_intervals_plus_1) / (N_intervals_plus_1 -1) ).astype(np.float32)

def interpolate_uniform_to_cgl(data_uniform, x_uniform_np, x_cgl_np):
    if isinstance(x_uniform_np, torch.Tensor): x_uniform_np = x_uniform_np.cpu().numpy().squeeze()
    if isinstance(x_cgl_np, torch.Tensor): x_cgl_np = x_cgl_np.cpu().numpy().squeeze()
    data_uniform_np = data_uniform.cpu().numpy() # No detach needed if not from graph

    B, Nx_uniform, C = data_uniform_np.shape
    Nx_cgl = x_cgl_np.shape[0]
    data_cgl_np = np.zeros((B, Nx_cgl, C), dtype=data_uniform_np.dtype)
    for b_idx in range(B):
        for c_idx in range(C):
            data_cgl_np[b_idx, :, c_idx] = np.interp(x_cgl_np, x_uniform_np, data_uniform_np[b_idx, :, c_idx])
    return torch.from_numpy(data_cgl_np).to(data_uniform.device)

def interpolate_cgl_to_uniform(data_cgl, x_cgl_np, x_uniform_np):
    if isinstance(x_uniform_np, torch.Tensor): x_uniform_np = x_uniform_np.cpu().numpy().squeeze()
    if isinstance(x_cgl_np, torch.Tensor): x_cgl_np = x_cgl_np.cpu().numpy().squeeze()
    data_cgl_np = data_cgl.cpu().detach().numpy() # Detach if from graph

    B, Nx_cgl, C = data_cgl_np.shape
    Nx_uniform = x_uniform_np.shape[0]
    data_uniform_np = np.zeros((B, Nx_uniform, C), dtype=data_cgl_np.dtype)
    for b_idx in range(B):
        for c_idx in range(C):
            data_uniform_np[b_idx, :, c_idx] = np.interp(x_uniform_np, x_cgl_np, data_cgl_np[b_idx, :, c_idx])
    return torch.from_numpy(data_uniform_np).to(data_cgl.device)

# =============================================================================
# OPNO Specific Components
# =============================================================================
class ChebyshevSpectralConv1d_OPNO(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, bc_type='dirichlet'): # Default bc_type
        super().__init__()
        self.in_channels = in_channels; self.out_channels = out_channels
        self.modes1 = modes1; self.bc_type = bc_type

        if bc_type == 'dirichlet': self.compact_inv_func = cheb_inverse_compact_dirichlet
        # Add other BC types if needed, or default to identity
        else: self.compact_inv_func = lambda x: x # Identity if not Dirichlet

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1))

    def cheb_mul1d(self, input_cheb_coeffs, weights):
        B, C_in, N = input_cheb_coeffs.shape; C_out = weights.shape[1]
        output = torch.zeros(B, C_out, N, device=input_cheb_coeffs.device, dtype=input_cheb_coeffs.dtype)
        modes_to_use = min(self.modes1, N)
        output_modes = torch.einsum("bim,iom->bom", input_cheb_coeffs[:, :, :modes_to_use], weights[:, :, :modes_to_use])
        output[:, :, :modes_to_use] = output_modes
        return output

    def forward(self, x): # x shape: [B, C, N_cgl]
        x_cheb_coeffs = cheb_dct(x)
        weighted_cheb_coeffs = self.cheb_mul1d(x_cheb_coeffs, self.weights1)
        if self.bc_type == 'dirichlet': # Only apply compact_inv if dirichlet for now
             out_cheb_coeffs_compact_inv = self.compact_inv_func(weighted_cheb_coeffs)
        else:
             out_cheb_coeffs_compact_inv = weighted_cheb_coeffs # No compacting for other BCs in this simplified version
        x_out = cheb_idct(out_cheb_coeffs_compact_inv)
        return x_out

class OPNO1d_Stepper(nn.Module):
    def __init__(self, modes, width, input_channels, output_channels, nx_cgl, num_layers=4, bc_type='dirichlet'):
        super().__init__()
        self.modes1 = modes; self.width = width
        self.input_channels = input_channels; self.output_channels = output_channels
        self.num_layers = num_layers; self.bc_type = bc_type
        self.nx_cgl = nx_cgl # Number of CGL points, also stored as self.nx for convenience

        self.fc0 = nn.Linear(input_channels, self.width)
        self.convs = nn.ModuleList()
        self.ws = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(ChebyshevSpectralConv1d_OPNO(self.width, self.width, self.modes1, self.bc_type))
            self.ws.append(nn.Conv1d(self.width, self.width, 1))
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, output_channels)
        self.nx = nx_cgl # For compatibility with plotting

    def forward(self, x_cgl_input): # x_cgl_input shape: [B, N_cgl, C_in]
        # C_in = num_state_vars_on_cgl + bc_ctrl_dim
        x_lifted = self.fc0(x_cgl_input)
        x_permuted = x_lifted.permute(0, 2, 1) # [B, W, N_cgl]
        x_proc = x_permuted
        for i in range(self.num_layers):
            x1 = self.convs[i](x_proc)
            x2 = self.ws[i](x_proc)
            x_proc = x1 + x2
            x_proc = F.gelu(x_proc)
        x_out_perm = x_proc.permute(0, 2, 1) # [B, N_cgl, W]
        x_out = self.fc1(x_out_perm); x_out = F.gelu(x_out)
        x_out = self.fc2(x_out) # [B, N_cgl, C_out] (C_out is num_output_state_vars)
        return x_out

# =============================================================================
# OPNO Training Function (Adapted for TRAIN_NT_FOR_MODEL)
# =============================================================================
def train_opno_stepper(model, data_loader, dataset_type, train_nt_for_model, # Added train_nt_for_model
                       nx_uniform_data: int, domain_L_data: float, no_interp_if_match: bool = True, device='cuda',
                       checkpoint_path='opno_checkpoint.pt',lr=1e-3, num_epochs=50, clip_grad_norm=1.0):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    mse_loss = nn.MSELoss(reduction='mean')
    start_epoch = 0; best_loss = float('inf')

    if os.path.exists(checkpoint_path): # Checkpoint loading
        print(f"Loading OPNO checkpoint from {checkpoint_path} ...")
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                try: optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                except: print("Warning: OPNO Optimizer state mismatch.")
            start_epoch = ckpt.get('epoch', 0) + 1
            best_loss = ckpt.get('loss', float('inf'))
            print(f"Resuming OPNO training from epoch {start_epoch}")
        except Exception as e: print(f"Error loading OPNO checkpoint: {e}. Starting fresh.")

    nx_cgl_model = model.nx_cgl # From OPNO1d_Stepper
    skip_interpolation = no_interp_if_match and (nx_uniform_data == nx_cgl_model)
    if skip_interpolation: print("OPNO Training: Uniform grid matches CGL grid. Skipping interpolation.")

    x_uniform_np = np.linspace(0, domain_L_data, nx_uniform_data).astype(np.float32)
    x_cgl_np = (get_cgl_points(nx_cgl_model) + 1.0) * domain_L_data / 2.0 # Scale CGL to [0, L]

    num_output_vars = model.output_channels # num_state_vars

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_train_loss = 0.0; num_batches = 0; batch_start_time = time.time()

        for i, (state_data_loaded, BC_Ctrl_tensor_loaded, _) in enumerate(data_loader):
            if isinstance(state_data_loaded, list):
                state_seq_uni_true = torch.stack(state_data_loaded, dim=-1).to(device)
            else:
                state_seq_uni_true = state_data_loaded.unsqueeze(-1).to(device)
            BC_Ctrl_seq_train = BC_Ctrl_tensor_loaded.to(device)
            batch_size, nt_loaded, _, _ = state_seq_uni_true.shape

            if nt_loaded != train_nt_for_model:
                raise ValueError(f"Mismatch: nt from DataLoader ({nt_loaded}) != train_nt_for_model ({train_nt_for_model})")

            optimizer.zero_grad()
            total_sequence_loss = torch.tensor(0.0, device=device, requires_grad=True)

            for t in range(train_nt_for_model - 1):
                u_n_uni_true = state_seq_uni_true[:, t, :, :]       # [B, nx_uni, num_vars]
                bc_ctrl_n = BC_Ctrl_seq_train[:, t, :]            # [B, bc_ctrl_dim]

                u_n_cgl_true = interpolate_uniform_to_cgl(u_n_uni_true, x_uniform_np, x_cgl_np) if not skip_interpolation else u_n_uni_true

                # Expand bc_ctrl to match spatial dim of CGL grid for concat
                bc_ctrl_n_expanded = bc_ctrl_n.unsqueeze(1).repeat(1, nx_cgl_model, 1) # [B, nx_cgl, bc_ctrl_dim]
                opno_input = torch.cat((u_n_cgl_true, bc_ctrl_n_expanded), dim=-1) # [B, nx_cgl, num_vars + bc_ctrl_dim]

                u_np1_cgl_pred = model(opno_input) # [B, nx_cgl, num_output_vars]
                u_np1_uni_pred = interpolate_cgl_to_uniform(u_np1_cgl_pred, x_cgl_np, x_uniform_np) if not skip_interpolation else u_np1_cgl_pred
                u_np1_uni_true = state_seq_uni_true[:, t+1, :, :] # Target is on uniform grid

                step_loss = mse_loss(u_np1_uni_pred, u_np1_uni_true)
                total_sequence_loss = total_sequence_loss + step_loss

            current_batch_loss = total_sequence_loss / (train_nt_for_model - 1)
            epoch_train_loss += current_batch_loss.item()
            num_batches += 1
            current_batch_loss.backward()
            if clip_grad_norm > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            optimizer.step()

            if (i + 1) % 50 == 0:
                batch_time_elapsed = time.time() - batch_start_time
                print(f"  OPNO Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(data_loader)}, "
                      f"Loss: {current_batch_loss.item():.4e}, Time/50Batch: {batch_time_elapsed:.2f}s")
                batch_start_time = time.time()

        avg_epoch_loss = epoch_train_loss / max(num_batches,1)
        print(f"OPNO Epoch {epoch+1}/{num_epochs} finished. Avg Training Loss: {avg_epoch_loss:.6f}")
        scheduler.step(avg_epoch_loss)
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            print(f"Saving OPNO checkpoint with loss {best_loss:.6f} to {checkpoint_path}")
            save_dict = {
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'loss': best_loss,
                'dataset_type': dataset_type,
                'modes': model.modes1, 'width': model.width, 'nx_cgl': model.nx_cgl,
                'input_channels': model.input_channels, 'output_channels': model.output_channels,
                'bc_type': model.bc_type
            }
            torch.save(save_dict, checkpoint_path)
    print("OPNO Training finished.")
    if os.path.exists(checkpoint_path):
        print(f"Loading best OPNO model from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    return model


# =============================================================================
# OPNO Validation Function (Autoregressive Rollout for Multiple Horizons)
# =============================================================================
def validate_opno_stepper(model, data_loader, dataset_type,
                          train_nt_for_model_training: int, T_value_for_model_training: float,
                          full_T_in_datafile: float, full_nt_in_datafile: int,
                          dataset_params_for_plot: dict, no_interp_if_match: bool = True, device='cuda',
                          save_fig_path_prefix='opno_result'):
    model.eval()
    if dataset_type == 'advection' or dataset_type == 'burgers': state_keys_val = ['U']
    elif dataset_type == 'euler': state_keys_val = ['rho', 'u']
    elif dataset_type == 'darcy': state_keys_val = ['P']
    else: raise ValueError(f"Unknown dataset_type '{dataset_type}' in OPNO validation")
    num_state_vars_val = len(state_keys_val)

    test_horizons_T_values = [T_value_for_model_training]
    if full_T_in_datafile > T_value_for_model_training:
        test_horizons_T_values.append(T_value_for_model_training + 0.5 * (full_T_in_datafile - T_value_for_model_training))
        test_horizons_T_values.append(full_T_in_datafile)
    test_horizons_T_values = sorted(list(set(h for h in test_horizons_T_values if h <= full_T_in_datafile + 1e-6)))
    print(f"OPNO Validation for T_horizons: {test_horizons_T_values}")

    results_primary_horizon = {key: {'mse': [], 'rmse': [], 'relative_error': [], 'max_error': []} for key in state_keys_val}
    overall_rel_err_primary_horizon = []

    # Grid setup for interpolation
    nx_uniform_data = dataset_params_for_plot.get('nx', 128) # From data file
    domain_L_data = dataset_params_for_plot.get('L', 1.0)
    nx_cgl_model = model.nx_cgl # From OPNO model
    skip_interpolation = no_interp_if_match and (nx_uniform_data == nx_cgl_model)
    if skip_interpolation: print("OPNO Validation: Uniform grid matches CGL. Skipping interpolation.")
    x_uniform_np = np.linspace(0, domain_L_data, nx_uniform_data).astype(np.float32)
    x_cgl_np = (get_cgl_points(nx_cgl_model) + 1.0) * domain_L_data / 2.0

    with torch.no_grad():
        try:
            state_data_full_loaded, BC_Ctrl_tensor_full_loaded, norm_factors_batch = next(iter(data_loader))
        except StopIteration: print("Val data_loader empty. Skipping OPNO validation."); return

        if isinstance(state_data_full_loaded, list):
            state_seq_uni_true_norm_full = torch.stack(state_data_full_loaded, dim=-1)[0].to(device)
        else:
            state_seq_uni_true_norm_full = state_data_full_loaded.unsqueeze(-1)[0].to(device)
        BC_Ctrl_seq_norm_full = BC_Ctrl_tensor_full_loaded[0].to(device)
        nt_file_check, _, num_vars_check_val = state_seq_uni_true_norm_full.shape

        norm_factors_sample = {}
        for key, val_tensor in norm_factors_batch.items():
            norm_factors_sample[key] = val_tensor[0].cpu().numpy() if isinstance(val_tensor, torch.Tensor) and val_tensor.ndim > 0 else (val_tensor.cpu().numpy() if isinstance(val_tensor, torch.Tensor) else val_tensor)


        u_initial_uni_norm = state_seq_uni_true_norm_full[0:1, :, :] # [1, nx_uni, num_vars]
        u_current_cgl_norm = interpolate_uniform_to_cgl(u_initial_uni_norm, x_uniform_np, x_cgl_np) if not skip_interpolation else u_initial_uni_norm.clone()


        for T_horizon_current in test_horizons_T_values:
            nt_for_rollout = int((T_horizon_current / full_T_in_datafile) * (full_nt_in_datafile - 1)) + 1
            nt_for_rollout = min(nt_for_rollout, full_nt_in_datafile)
            print(f"\n  Rollout for T_horizon = {T_horizon_current:.2f} (nt = {nt_for_rollout})")

            u_pred_seq_uni_norm_horizon = torch.zeros(nt_for_rollout, nx_uniform_data, num_vars_check_val, device=device)
            u_pred_seq_uni_norm_horizon[0, :, :] = u_initial_uni_norm.squeeze(0) # Store initial condition (on uniform grid)

            # Current state for rollout loop starts from the CGL version of initial condition
            u_roll_cgl_current = u_current_cgl_norm.clone() # [1, nx_cgl, num_vars]

            BC_Ctrl_for_rollout = BC_Ctrl_seq_norm_full[:nt_for_rollout, :]

            for t_step in range(nt_for_rollout - 1):
                bc_ctrl_n_step = BC_Ctrl_for_rollout[t_step:t_step+1, :]
                bc_ctrl_n_expanded = bc_ctrl_n_step.unsqueeze(1).repeat(1, nx_cgl_model, 1)
                opno_input_step = torch.cat((u_roll_cgl_current, bc_ctrl_n_expanded), dim=-1)
                u_next_pred_cgl_norm = model(opno_input_step) # [1, nx_cgl, num_vars]

                u_next_pred_uni_norm = interpolate_cgl_to_uniform(u_next_pred_cgl_norm, x_cgl_np, x_uniform_np) if not skip_interpolation else u_next_pred_cgl_norm
                u_pred_seq_uni_norm_horizon[t_step+1, :, :] = u_next_pred_uni_norm.squeeze(0)
                u_roll_cgl_current = u_next_pred_cgl_norm # Update for next step (on CGL grid)

            # Denormalize & Calculate Metrics (using uniform grid results)
            # (Identical to FNO/DeepONet validation's metric part)
            U_pred_denorm_horizon={}; U_gt_denorm_horizon={}
            combined_pred_list_horizon=[]; combined_gt_list_horizon=[]
            state_seq_true_uni_norm_sliced = state_seq_uni_true_norm_full[:nt_for_rollout, :, :]
            pred_seq_np_horizon = u_pred_seq_uni_norm_horizon.cpu().numpy()
            gt_seq_np_horizon = state_seq_true_uni_norm_sliced.cpu().numpy()

            for k_idx, key_val in enumerate(state_keys_val):
                mean_k_val = norm_factors_sample[f'{key_val}_mean']; std_k_val = norm_factors_sample[f'{key_val}_std']
                mean_k = mean_k_val.item() if hasattr(mean_k_val, 'item') else mean_k_val
                std_k = std_k_val.item() if hasattr(std_k_val, 'item') else std_k_val
                pred_norm_var = pred_seq_np_horizon[:, :, k_idx]; gt_norm_var = gt_seq_np_horizon[:, :, k_idx]
                pred_denorm_var = pred_norm_var * std_k + mean_k; gt_denorm_var = gt_norm_var * std_k + mean_k
                U_pred_denorm_horizon[key_val] = pred_denorm_var; U_gt_denorm_horizon[key_val] = gt_denorm_var
                combined_pred_list_horizon.append(pred_denorm_var.flatten()); combined_gt_list_horizon.append(gt_denorm_var.flatten())
                mse_k_hor = np.mean((pred_denorm_var - gt_denorm_var)**2)
                rel_err_k_hor = np.linalg.norm(pred_denorm_var - gt_denorm_var, 'fro') / (np.linalg.norm(gt_denorm_var, 'fro') + 1e-10)
                print(f"    Metrics for '{key_val}' @ T={T_horizon_current:.1f}: MSE={mse_k_hor:.3e}, RelErr={rel_err_k_hor:.3e}")
                if abs(T_horizon_current - T_value_for_model_training) < 1e-6:
                    results_primary_horizon[key_val]['mse'].append(mse_k_hor)
                    results_primary_horizon[key_val]['relative_error'].append(rel_err_k_hor)
            # ... (overall rel err calculation)
            U_pred_vec_hor = np.concatenate(combined_pred_list_horizon); U_gt_vec_hor = np.concatenate(combined_gt_list_horizon)
            overall_rel_err_val = np.linalg.norm(U_pred_vec_hor - U_gt_vec_hor) / (np.linalg.norm(U_gt_vec_hor) + 1e-10)
            print(f"    Overall RelErr @ T={T_horizon_current:.1f}: {overall_rel_err_val:.3e}")
            if abs(T_horizon_current - T_value_for_model_training) < 1e-6:
                overall_rel_err_primary_horizon.append(overall_rel_err_val)


            # Visualization (Identical to FNO/DeepONet, ensure title is "OPNO Pred")
            # ... (Plotting logic as in FNO/DeepONet, ensure title says "OPNO Pred") ...
            fig, axs = plt.subplots(num_state_vars_val, 3, figsize=(18, 5 * num_state_vars_val), squeeze=False)
            # ... (Copy plotting code from validate_fno_stepper, change "FNO Pred" to "OPNO Pred")
            fig_L = dataset_params_for_plot.get('L', 1.0); fig_nx = dataset_params_for_plot.get('nx', nx_uniform_data); fig_ny = dataset_params_for_plot.get('ny', 1)
            for k_idx, key_val in enumerate(state_keys_val):
                gt_plot = U_gt_denorm_horizon[key_val]; pred_plot = U_pred_denorm_horizon[key_val]; diff_plot = np.abs(pred_plot - gt_plot)
                max_err_plot = np.max(diff_plot) if diff_plot.size > 0 else 0
                vmin_plot = min(np.min(gt_plot), np.min(pred_plot)) if gt_plot.size > 0 else 0; vmax_plot = max(np.max(gt_plot), np.max(pred_plot)) if gt_plot.size > 0 else 1
                is_1d_plot = (fig_ny == 1); plot_extent = [0, fig_L, 0, T_horizon_current]
                if is_1d_plot:
                    im0=axs[k_idx,0].imshow(gt_plot, aspect='auto', origin='lower', vmin=vmin_plot, vmax=vmax_plot, extent=plot_extent, cmap='viridis'); axs[k_idx,0].set_title(f"Truth ({key_val})")
                    im1=axs[k_idx,1].imshow(pred_plot, aspect='auto', origin='lower', vmin=vmin_plot, vmax=vmax_plot, extent=plot_extent, cmap='viridis'); axs[k_idx,1].set_title(f"OPNO Pred ({key_val})") # OPNO Title
                    im2=axs[k_idx,2].imshow(diff_plot, aspect='auto', origin='lower', extent=plot_extent, cmap='magma'); axs[k_idx,2].set_title(f"Abs Error (Max:{max_err_plot:.2e})")
                    for j_plot in range(3): axs[k_idx,j_plot].set_xlabel("x"); axs[k_idx,0].set_ylabel("t")
                    plt.colorbar(im0, ax=axs[k_idx,0]); plt.colorbar(im1, ax=axs[k_idx,1]); plt.colorbar(im2, ax=axs[k_idx,2])
                else: axs[k_idx,0].text(0.5,0.5, "2D Plot Placeholder", ha='center')
            fig.suptitle(f"OPNO Validation ({dataset_type.capitalize()}) @ T={T_horizon_current:.1f}") # OPNO Title
            fig.tight_layout(rect=[0,0.03,1,0.95])
            current_fig_path = save_fig_path_prefix + f"_T{str(T_horizon_current).replace('.','p')}.png"
            plt.savefig(current_fig_path); print(f"  Saved OPNO validation plot to {current_fig_path}"); plt.show()


    print(f"\n--- OPNO Validation Summary (Metrics for T={T_value_for_model_training:.1f}) ---")
    # ... (Summary printing from FNO/DeepONet validation) ...
    for key_val in state_keys_val:
        if results_primary_horizon[key_val]['mse']:
            avg_mse = np.mean(results_primary_horizon[key_val]['mse']); avg_rmse = np.sqrt(avg_mse)
            avg_rel_err = np.mean(results_primary_horizon[key_val]['relative_error'])
            print(f"  Var '{key_val}': Avg MSE={avg_mse:.3e}, RMSE={avg_rmse:.3e}, RelErr={avg_rel_err:.3e}")
    if overall_rel_err_primary_horizon:
        print(f"  Overall Avg RelErr @ T={T_value_for_model_training:.1f}: {np.mean(overall_rel_err_primary_horizon):.3e}")
    print("------------------------")

# =============================================================================
# Main Block - Adapted for OPNO
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate PDE datasets with complex BCs.")
    parser.add_argument('--datatype', type=str, required=True, choices=['advection', 'euler', 'burgers', 'darcy'], help='Type of dataset to generate.')
    args = parser.parse_args()

    DATASET_TYPE = args.datatype # 'advection', 'euler', 'burgers', 'darcy'
    MODEL_TYPE = 'OPNO'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}") # ... (print dataset/model type)

    FULL_T_IN_DATAFILE = 2.0; FULL_NT_IN_DATAFILE = 600
    TRAIN_T_TARGET = 1.0
    TRAIN_NT_FOR_MODEL = int((TRAIN_T_TARGET / FULL_T_IN_DATAFILE) * (FULL_NT_IN_DATAFILE - 1)) + 1
    print(f"Datafile T={FULL_T_IN_DATAFILE}, nt={FULL_NT_IN_DATAFILE}")
    print(f"Training with T={TRAIN_T_TARGET}, nt={TRAIN_NT_FOR_MODEL}")

    # OPNO Specific Hyperparameters
    opno_modes = 32       # Modes for ChebyshevSpectralConv1d_OPNO
    opno_width = 64       # Feature width in OPNO layers
    opno_layers = 4
    opno_bc_type = 'dirichlet' # Default for OPNO example; 'mixed' or specific for others
    # Number of CGL points for OPNO's internal representation.
    # Can be same as data's uniform grid size, or different.
    # If different, interpolation is used. If same, can be skipped.
    opno_nx_cgl = 128 # Example: OPNO might work on a different grid size internally
    # If opno_nx_cgl is same as data's nx, set no_interp_if_match=True

    learning_rate = 1e-3; batch_size = 32; num_epochs = 80; clip_grad_norm = 1.0

    dataset_params_for_plot = {}
    if DATASET_TYPE == 'advection':
        dataset_path = "./datasets_full/advection_data_10000s_128nx_600nt.pkl"
        main_state_keys = ['U']; main_num_state_vars = 1
        dataset_params_for_plot = {'nx': 128, 'ny': 1, 'L': 1.0, 'T': FULL_T_IN_DATAFILE}
        opno_nx_cgl = 128 # Match data grid to potentially skip interpolation
    elif DATASET_TYPE == 'euler':
        dataset_path = "./datasets_full/euler_data_10000s_128nx_600nt.pkl"
        main_state_keys = ['rho', 'u']; main_num_state_vars = 2
        dataset_params_for_plot = {'nx': 128, 'ny': 1, 'L': 1.0, 'T': FULL_T_IN_DATAFILE}
        opno_nx_cgl = 128
        opno_bc_type = 'mixed' # Euler often has mixed BCs conceptually
    elif DATASET_TYPE == 'burgers':
        dataset_path = "./datasets_full/burgers_data_10000s_128nx_600nt.pkl"
        main_state_keys = ['U']; main_num_state_vars = 1
        dataset_params_for_plot = {'nx': 128, 'ny': 1, 'L': 1.0, 'T': FULL_T_IN_DATAFILE}
        opno_nx_cgl = 128
    elif DATASET_TYPE == 'darcy':
        dataset_path = "./datasets_full/darcy_data_10000s_128nx_600nt.pkl"
        main_state_keys = ['P']; main_num_state_vars = 1
        dataset_params_for_plot = {'nx': 128, 'ny': 1, 'L': 1.0, 'T': FULL_T_IN_DATAFILE}
        opno_nx_cgl = 128
    else: raise ValueError(f"Unknown dataset_type: {DATASET_TYPE}")

    print(f"Loading dataset for {DATASET_TYPE} from {dataset_path}...")
    try:
        with open(dataset_path, 'rb') as f: data_list_all = pickle.load(f)
        print(f"Loaded {len(data_list_all)} samples.")
        # ... (File parameter verification from previous responses) ...
    except FileNotFoundError: print(f"Error: Dataset file not found: {dataset_path}"); exit()
    if not data_list_all: print("No data loaded, exiting."); exit()

    # Update dataset_params_for_plot with actual file values if they were read
    if data_list_all:
        sample_params_check = data_list_all[0].get('params', {})
        file_L_check = sample_params_check.get('L')
        file_nx_check = sample_params_check.get('nx')
        if file_L_check is not None: dataset_params_for_plot['L'] = file_L_check
        if file_nx_check is not None: dataset_params_for_plot['nx'] = file_nx_check


    random.shuffle(data_list_all)
    n_total = len(data_list_all); n_train = int(0.8 * n_total)
    train_data_list_split = data_list_all[:n_train]; val_data_list_split = data_list_all[n_train:]
    # ... (Print train/val split size)

    # train_dataset = UniversalPDEDataset(train_data_list_split, dataset_type=DATASET_TYPE, train_nt_limit=TRAIN_NT_FOR_MODEL)
    # val_dataset = UniversalPDEDataset(val_data_list_split, dataset_type=DATASET_TYPE, train_nt_limit=None)
    train_dataset = UniversalPDEDataset(
        data_list=train_data_list_split,
        dataset_type=DATASET_TYPE,
        stats_calculation_length=TRAIN_NT_FOR_MODEL,
        sequence_length_to_return=TRAIN_NT_FOR_MODEL
    )
    
    val_dataset = UniversalPDEDataset(
        data_list=val_data_list_split,
        dataset_type=DATASET_TYPE,
        stats_calculation_length=TRAIN_NT_FOR_MODEL,  # 关键：使用与训练相同的统计长度
        sequence_length_to_return=FULL_NT_IN_DATAFILE # 或者您希望验证的最大步数
    )
    num_workers = 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    actual_bc_state_dim_ds = train_dataset.bc_state_dim
    actual_num_controls_ds = train_dataset.num_controls
    opno_input_channels = main_num_state_vars + actual_bc_state_dim_ds + actual_num_controls_ds
    opno_output_channels = main_num_state_vars

    print(f"\nInitializing OPNO model: Modes={opno_modes}, Width={opno_width}, Nx_CGL={opno_nx_cgl}")
    print(f"  InputChannels={opno_input_channels}, OutputChannels={opno_output_channels}, BC_Type='{opno_bc_type}'")

    online_opno_model = OPNO1d_Stepper( # Renamed
        modes=opno_modes, width=opno_width,
        input_channels=opno_input_channels, output_channels=opno_output_channels,
        nx_cgl=opno_nx_cgl, num_layers=opno_layers, bc_type=opno_bc_type
    )

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{DATASET_TYPE}_{MODEL_TYPE}_trainT{TRAIN_T_TARGET}_m{opno_modes}_w{opno_width}_cgl{opno_nx_cgl}"
    checkpoint_dir = f"./New_ckpt_2/checkpoints_{DATASET_TYPE}_{MODEL_TYPE}" # Simplified path
    results_dir = f"./result_all_2/results_{DATASET_TYPE}_{MODEL_TYPE}"      # Simplified path
    os.makedirs(checkpoint_dir, exist_ok=True); os.makedirs(results_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'model_{run_name}.pt')
    save_fig_path_prefix = os.path.join(results_dir, f'result_{run_name}')

    no_interp_flag = (dataset_params_for_plot['nx'] == opno_nx_cgl)

    print(f"\nStarting training for OPNO on {DATASET_TYPE}...")
    start_train_time = time.time()
    online_opno_model = train_opno_stepper(
        online_opno_model, train_loader, dataset_type=DATASET_TYPE,
        train_nt_for_model=TRAIN_NT_FOR_MODEL,
        nx_uniform_data=dataset_params_for_plot['nx'],
        domain_L_data=dataset_params_for_plot['L'],
        no_interp_if_match=no_interp_flag, device=device,
        checkpoint_path=checkpoint_path,
        lr=learning_rate, num_epochs=num_epochs, clip_grad_norm=clip_grad_norm
    )
    # ... (End training time print)

    if val_data_list_split:
        print(f"\nStarting validation for OPNO on {DATASET_TYPE}...")
        validate_opno_stepper(
            online_opno_model, val_loader, dataset_type=DATASET_TYPE,
            train_nt_for_model_training=TRAIN_NT_FOR_MODEL,
            T_value_for_model_training=TRAIN_T_TARGET,
            full_T_in_datafile=FULL_T_IN_DATAFILE,
            full_nt_in_datafile=FULL_NT_IN_DATAFILE,
            dataset_params_for_plot=dataset_params_for_plot,
            no_interp_if_match=no_interp_flag, device=device,
            save_fig_path_prefix=save_fig_path_prefix
        )
    # ... (Final print statements) ...
    print(f"Run finished: {run_name}") # etc.