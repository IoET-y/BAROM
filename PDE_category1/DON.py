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
# import scipy.sparse as sp # Not used by DeepONet training/validation
# from scipy.sparse.linalg import spsolve # Not used by DeepONet
import pickle
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
# 2. 通用化数据集定义 (Corrected version with train_nt_limit)
# =============================================================================
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
# DeepONet Components
# =============================================================================
class BranchNet(nn.Module):
    def __init__(self, num_state_vars, nx, bc_ctrl_dim, cnn_out_dim=64, fnn_out_dim=64, combined_dim=128, output_size=128):
        super().__init__()
        self.conv1 = nn.Conv1d(num_state_vars, cnn_out_dim // 2, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(cnn_out_dim // 2, cnn_out_dim, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fnn1 = nn.Linear(bc_ctrl_dim, fnn_out_dim // 2)
        self.fnn2 = nn.Linear(fnn_out_dim // 2, fnn_out_dim)
        self.fc_combined1 = nn.Linear(cnn_out_dim + fnn_out_dim, combined_dim)
        self.fc_combined2 = nn.Linear(combined_dim, output_size)

    def forward(self, u_n, bc_ctrl_n):
        c = F.gelu(self.conv1(u_n))
        c = F.gelu(self.conv2(c))
        c = self.pool(c).squeeze(-1)
        f = F.gelu(self.fnn1(bc_ctrl_n))
        f = F.gelu(self.fnn2(f))
        combined = torch.cat((c, f), dim=-1)
        out = F.gelu(self.fc_combined1(combined))
        out = self.fc_combined2(out)
        return out

class TrunkNet(nn.Module):
    def __init__(self, coord_dim=1, embed_dim=128, output_size=128):
        super().__init__()
        self.fc1 = nn.Linear(coord_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.fc3 = nn.Linear(embed_dim, output_size)

    def forward(self, grid_coords):
        t = F.gelu(self.fc1(grid_coords))
        t = F.gelu(self.fc2(t))
        t = self.fc3(t)
        return t

class DeepONetStepper(nn.Module):
    def __init__(self, num_state_vars, nx, bc_ctrl_dim, num_output_vars,
                 branch_p, trunk_p, coord_dim=1, domain_L=1.0): # Added domain_L
        super().__init__()
        self.num_state_vars = num_state_vars
        self.num_output_vars = num_output_vars # Should be same as num_state_vars for stepper
        self.nx = nx
        self.bc_ctrl_dim = bc_ctrl_dim
        assert branch_p == trunk_p, "Branch 'p' and Trunk 'p' must match"
        self.p = trunk_p

        self.branch = BranchNet(num_state_vars, nx, bc_ctrl_dim, output_size=self.p * num_output_vars)
        self.trunk = TrunkNet(coord_dim=coord_dim, output_size=self.p)
        grid = torch.linspace(0, domain_L, nx).view(-1, coord_dim) # Use domain_L
        self.register_buffer('grid', grid)

    def forward(self, u_n, bc_ctrl_n):
        u_n_permuted = u_n.permute(0, 2, 1) # [B, num_state_vars, nx]
        branch_out = self.branch(u_n_permuted, bc_ctrl_n) # [B, p * num_output_vars]
        branch_out_reshaped = branch_out.view(-1, self.p, self.num_output_vars) # [B, p, num_output_vars]
        grid_batched = self.grid.unsqueeze(0).repeat(u_n.shape[0], 1, 1) # [B, nx, coord_dim]
        trunk_out = self.trunk(grid_batched) # [B, nx, p]
        u_np1_pred = torch.matmul(trunk_out, branch_out_reshaped) # [B, nx, num_output_vars]
        return u_np1_pred

# =============================================================================
# DeepONet Training Function (Adapted for TRAIN_NT_FOR_MODEL)
# =============================================================================
def train_deeponet_stepper(model, data_loader, dataset_type, train_nt_for_model, # Added train_nt_for_model
                           lr=1e-3, num_epochs=50, device='cuda',
                           checkpoint_path='don_checkpoint.pt', clip_grad_norm=1.0):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    mse_loss = nn.MSELoss(reduction='mean')

    start_epoch = 0; best_loss = float('inf')
    if os.path.exists(checkpoint_path):
        print(f"Loading DeepONet checkpoint from {checkpoint_path} ...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                try: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except: print("Warning: DON Optimizer state mismatch.")
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_loss = checkpoint.get('loss', float('inf'))
            print(f"Resuming DON training from epoch {start_epoch}")
        except Exception as e: print(f"Error loading DON checkpoint: {e}. Starting fresh.")

    num_state_vars = model.num_output_vars # from DeepONetStepper init

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_train_loss = 0.0; num_batches = 0; batch_start_time = time.time()

        for i, (state_data_loaded, BC_Ctrl_tensor_loaded, _) in enumerate(data_loader):
            # Data from loader is already truncated to train_nt_for_model
            if isinstance(state_data_loaded, list):
                state_seq_true_train = torch.stack(state_data_loaded, dim=-1).to(device)
            else:
                state_seq_true_train = state_data_loaded.unsqueeze(-1).to(device)
            BC_Ctrl_seq_train = BC_Ctrl_tensor_loaded.to(device)
            batch_size, nt_loaded, nx_or_N, _ = state_seq_true_train.shape

            if nt_loaded != train_nt_for_model:
                raise ValueError(f"Mismatch: nt from DataLoader ({nt_loaded}) != train_nt_for_model ({train_nt_for_model})")

            optimizer.zero_grad()
            total_sequence_loss = 0.0
            for t in range(train_nt_for_model - 1): # Iterate up to train_nt_for_model-1
                u_n_true = state_seq_true_train[:, t, :, :]      # [B, nx, num_vars]
                bc_ctrl_n = BC_Ctrl_seq_train[:, t, :]           # [B, bc_ctrl_dim]
                u_np1_pred = model(u_n_true, bc_ctrl_n)          # [B, nx, num_vars]
                u_np1_true = state_seq_true_train[:, t+1, :, :]  # [B, nx, num_vars]
                step_loss = mse_loss(u_np1_pred, u_np1_true)
                total_sequence_loss += step_loss

            current_batch_loss = total_sequence_loss / (train_nt_for_model - 1)
            epoch_train_loss += current_batch_loss.item()
            num_batches += 1
            current_batch_loss.backward()
            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            optimizer.step()

            if (i + 1) % 50 == 0:
                batch_time_elapsed = time.time() - batch_start_time
                print(f"  DON Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(data_loader)}, "
                      f"Batch Loss: {current_batch_loss.item():.4e}, Time/50Batch: {batch_time_elapsed:.2f}s")
                batch_start_time = time.time()

        avg_epoch_loss = epoch_train_loss / num_batches
        print(f"DON Epoch {epoch+1}/{num_epochs} finished. Avg Training Loss: {avg_epoch_loss:.6f}")
        scheduler.step(avg_epoch_loss)
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            print(f"Saving DON checkpoint with loss {best_loss:.6f} to {checkpoint_path}")
            save_dict = {
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'loss': best_loss,
                'dataset_type': dataset_type,
                # Add relevant DeepONet arch params if needed for reloading
                'num_state_vars': model.num_state_vars, 'nx': model.nx,
                'bc_ctrl_dim': model.bc_ctrl_dim, 'num_output_vars': model.num_output_vars,
                'branch_p': model.p, 'trunk_p': model.p
            }
            torch.save(save_dict, checkpoint_path)
    print("DeepONet Training finished.")
    if os.path.exists(checkpoint_path):
        print(f"Loading best DeepONet model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    return model

# =============================================================================
# DeepONet Validation Function (Autoregressive Rollout for Multiple Horizons)
# =============================================================================
def validate_deeponet_stepper(model, data_loader, dataset_type,
                              train_nt_for_model_training: int,
                              T_value_for_model_training: float,
                              full_T_in_datafile: float,
                              full_nt_in_datafile: int,
                              dataset_params_for_plot: dict, device='cuda',
                              save_fig_path_prefix='don_result'):
    model.eval()

    if dataset_type == 'advection' or dataset_type == 'burgers': state_keys_val = ['U']
    elif dataset_type == 'euler': state_keys_val = ['rho', 'u']
    elif dataset_type == 'darcy': state_keys_val = ['P']
    else: raise ValueError(f"Unknown dataset_type '{dataset_type}' in DON validation")
    num_state_vars_val = len(state_keys_val)
    if model.num_output_vars != num_state_vars_val:
        print(f"Warning: DeepONet model output vars ({model.num_output_vars}) != num_state_vars from dataset_type ({num_state_vars_val})")

    test_horizons_T_values = [T_value_for_model_training]
    if full_T_in_datafile > T_value_for_model_training:
        test_horizons_T_values.append(T_value_for_model_training + 0.5 * (full_T_in_datafile - T_value_for_model_training))
        test_horizons_T_values.append(full_T_in_datafile)
    test_horizons_T_values = sorted(list(set(h for h in test_horizons_T_values if h <= full_T_in_datafile + 1e-6)))

    print(f"DeepONet Validation for T_horizons: {test_horizons_T_values}")
    # ... (rest of print statements from FNO validation) ...

    results_primary_horizon = {key: {'mse': [], 'rmse': [], 'relative_error': [], 'max_error': []} for key in state_keys_val}
    overall_rel_err_primary_horizon = []

    with torch.no_grad():
        try:
            state_data_full_loaded, BC_Ctrl_tensor_full_loaded, norm_factors_batch = next(iter(data_loader))
        except StopIteration:
            print("Validation data_loader is empty. Skipping DeepONet validation.")
            return

        if isinstance(state_data_full_loaded, list):
            state_seq_true_norm_full = torch.stack(state_data_full_loaded, dim=-1)[0].to(device)
        else:
            state_seq_true_norm_full = state_data_full_loaded.unsqueeze(-1)[0].to(device)
        BC_Ctrl_seq_norm_full = BC_Ctrl_tensor_full_loaded[0].to(device)
        nt_file_check, nx_or_N_file_check, num_vars_check = state_seq_true_norm_full.shape

        norm_factors_sample = {}
        for key, val_tensor in norm_factors_batch.items():
            if isinstance(val_tensor, torch.Tensor) or isinstance(val_tensor, np.ndarray):
                norm_factors_sample[key] = val_tensor[0].cpu().numpy() if val_tensor.ndim > 0 else val_tensor.cpu().numpy()
            else: norm_factors_sample[key] = val_tensor

        u_initial_norm = state_seq_true_norm_full[0:1, :, :]

        for T_horizon_current in test_horizons_T_values:
            nt_for_rollout = int((T_horizon_current / full_T_in_datafile) * (full_nt_in_datafile - 1)) + 1
            nt_for_rollout = min(nt_for_rollout, full_nt_in_datafile)
            print(f"\n  Rollout for T_horizon = {T_horizon_current:.2f} (nt = {nt_for_rollout})")

            u_pred_seq_norm_current_horizon = torch.zeros(nt_for_rollout, nx_or_N_file_check, num_vars_check, device=device)
            u_current_pred_step = u_initial_norm.clone()
            u_pred_seq_norm_current_horizon[0, :, :] = u_current_pred_step.squeeze(0)
            BC_Ctrl_for_rollout = BC_Ctrl_seq_norm_full[:nt_for_rollout, :]

            for t_step in range(nt_for_rollout - 1):
                bc_ctrl_n_step = BC_Ctrl_for_rollout[t_step:t_step+1, :]
                # Ensure u_current_pred_step is [1, nx, num_vars] for model input
                if u_current_pred_step.shape[0] != 1: u_current_pred_step = u_current_pred_step.unsqueeze(0)

                u_next_pred_norm_step = model(u_current_pred_step, bc_ctrl_n_step) # Input u: [1, nx, n_vars]
                u_pred_seq_norm_current_horizon[t_step+1, :, :] = u_next_pred_norm_step.squeeze(0)
                u_current_pred_step = u_next_pred_norm_step

            # Denormalize & Calculate Metrics (Identical to FNO validation section)
            U_pred_denorm_horizon = {}; U_gt_denorm_horizon = {}
            combined_pred_list_horizon = []; combined_gt_list_horizon = []
            state_seq_true_norm_sliced_horizon = state_seq_true_norm_full[:nt_for_rollout, :, :]
            pred_seq_np_horizon = u_pred_seq_norm_current_horizon.cpu().numpy()
            gt_seq_np_horizon = state_seq_true_norm_sliced_horizon.cpu().numpy()

            for k_idx, key_val in enumerate(state_keys_val):
                mean_k_val = norm_factors_sample[f'{key_val}_mean']
                std_k_val = norm_factors_sample[f'{key_val}_std']
                mean_k = mean_k_val.item() if hasattr(mean_k_val, 'item') else mean_k_val
                std_k = std_k_val.item() if hasattr(std_k_val, 'item') else std_k_val
                pred_norm_var = pred_seq_np_horizon[:, :, k_idx]
                gt_norm_var = gt_seq_np_horizon[:, :, k_idx]
                pred_denorm_var = pred_norm_var * std_k + mean_k
                gt_denorm_var = gt_norm_var * std_k + mean_k
                U_pred_denorm_horizon[key_val] = pred_denorm_var
                U_gt_denorm_horizon[key_val] = gt_denorm_var
                combined_pred_list_horizon.append(pred_denorm_var.flatten())
                combined_gt_list_horizon.append(gt_denorm_var.flatten())
                mse_k_hor = np.mean((pred_denorm_var - gt_denorm_var)**2)
                # ... (rmse, rel_err, max_err calculation)
                rel_err_k_hor = np.linalg.norm(pred_denorm_var - gt_denorm_var, 'fro') / (np.linalg.norm(gt_denorm_var, 'fro') + 1e-10)
                print(f"    Metrics for '{key_val}' @ T={T_horizon_current:.1f}: MSE={mse_k_hor:.3e}, RelErr={rel_err_k_hor:.3e}")
                if abs(T_horizon_current - T_value_for_model_training) < 1e-6:
                    results_primary_horizon[key_val]['mse'].append(mse_k_hor)
                    # ... (append other metrics)
                    results_primary_horizon[key_val]['relative_error'].append(rel_err_k_hor)


            U_pred_vec_hor = np.concatenate(combined_pred_list_horizon)
            U_gt_vec_hor = np.concatenate(combined_gt_list_horizon)
            overall_rel_err_val = np.linalg.norm(U_pred_vec_hor - U_gt_vec_hor) / (np.linalg.norm(U_gt_vec_hor) + 1e-10)
            print(f"    Overall RelErr @ T={T_horizon_current:.1f}: {overall_rel_err_val:.3e}")
            if abs(T_horizon_current - T_value_for_model_training) < 1e-6:
                overall_rel_err_primary_horizon.append(overall_rel_err_val)

            # Visualization (Identical to FNO validation, just change title prefix)
            fig, axs = plt.subplots(num_state_vars_val, 3, figsize=(18, 5 * num_state_vars_val), squeeze=False)
            fig_L = dataset_params_for_plot.get('L', 1.0)
            fig_nx = dataset_params_for_plot.get('nx', nx_or_N_file_check)
            fig_ny = dataset_params_for_plot.get('ny', 1)

            for k_idx, key_val in enumerate(state_keys_val):
                # ... (plotting logic from FNO, ensure titles reflect "DeepONet Pred")
                gt_plot = U_gt_denorm_horizon[key_val]
                pred_plot = U_pred_denorm_horizon[key_val]
                diff_plot = np.abs(pred_plot - gt_plot)
                max_err_plot = np.max(diff_plot) if diff_plot.size > 0 else 0
                vmin_plot = min(np.min(gt_plot), np.min(pred_plot)) if gt_plot.size > 0 else 0
                vmax_plot = max(np.max(gt_plot), np.max(pred_plot)) if gt_plot.size > 0 else 1
                is_1d_plot = (fig_ny == 1)
                plot_extent = [0, fig_L, 0, T_horizon_current]
                if is_1d_plot:
                    im0=axs[k_idx,0].imshow(gt_plot, aspect='auto', origin='lower', vmin=vmin_plot, vmax=vmax_plot, extent=plot_extent, cmap='viridis'); axs[k_idx,0].set_title(f"Truth ({key_val})")
                    im1=axs[k_idx,1].imshow(pred_plot, aspect='auto', origin='lower', vmin=vmin_plot, vmax=vmax_plot, extent=plot_extent, cmap='viridis'); axs[k_idx,1].set_title(f"DeepONet Pred ({key_val})") # Changed title
                    im2=axs[k_idx,2].imshow(diff_plot, aspect='auto', origin='lower', extent=plot_extent, cmap='magma'); axs[k_idx,2].set_title(f"Abs Error (Max:{max_err_plot:.2e})")
                    for j_plot in range(3): axs[k_idx,j_plot].set_xlabel("x"); axs[k_idx,0].set_ylabel("t")
                    plt.colorbar(im0, ax=axs[k_idx,0]); plt.colorbar(im1, ax=axs[k_idx,1]); plt.colorbar(im2, ax=axs[k_idx,2])
                else: # 2D plot (Darcy)
                    # ... (2D plotting logic from FNO, ensure titles are correct)
                    axs[k_idx,0].text(0.5,0.5, "2D Plot Placeholder (Darcy)", ha='center')


            fig.suptitle(f"DeepONet Validation ({dataset_type.capitalize()}) @ T={T_horizon_current:.1f}") # Changed title
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            current_fig_path = save_fig_path_prefix + f"_T{str(T_horizon_current).replace('.', 'p')}.png"
            plt.savefig(current_fig_path)
            print(f"  Saved DeepONet validation visualization to {current_fig_path}")
            plt.show()

    print(f"\n--- DeepONet Validation Summary (Metrics for T={T_value_for_model_training:.1f}) ---")
    # ... (Summary printing from FNO validation)
    for key_val in state_keys_val:
        if results_primary_horizon[key_val]['mse']:
            avg_mse = np.mean(results_primary_horizon[key_val]['mse']); avg_rmse = np.sqrt(avg_mse) # Corrected RMSE
            avg_rel_err = np.mean(results_primary_horizon[key_val]['relative_error'])
            # avg_max_err = np.mean(results_primary_horizon[key_val]['max_error']) # Max error doesn't average well
            print(f"  Var '{key_val}': Avg MSE={avg_mse:.3e}, RMSE={avg_rmse:.3e}, RelErr={avg_rel_err:.3e}")
    if overall_rel_err_primary_horizon:
        print(f"  Overall Avg RelErr @ T={T_value_for_model_training:.1f}: {np.mean(overall_rel_err_primary_horizon):.3e}")
    print("------------------------")


# =============================================================================
# Main Block - Adapted for DeepONet
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate PDE datasets with complex BCs.")
    parser.add_argument('--datatype', type=str, required=True, choices=['advection', 'euler', 'burgers', 'darcy'], help='Type of dataset to generate.')
    args = parser.parse_args()

    DATASET_TYPE = args.datatype # 'advection', 'euler', 'burgers', 'darcy'
    MODEL_TYPE = 'DeepONet'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Selected Dataset Type: {DATASET_TYPE.upper()}")
    print(f"Selected Model Type: {MODEL_TYPE.upper()}")

    FULL_T_IN_DATAFILE = 2.0
    FULL_NT_IN_DATAFILE = 600
    TRAIN_T_TARGET = 1.0
    TRAIN_NT_FOR_MODEL = int((TRAIN_T_TARGET / FULL_T_IN_DATAFILE) * (FULL_NT_IN_DATAFILE - 1)) + 1
    print(f"Datafile T={FULL_T_IN_DATAFILE}, nt={FULL_NT_IN_DATAFILE}")
    print(f"Training with T={TRAIN_T_TARGET}, nt={TRAIN_NT_FOR_MODEL}")

    # DeepONet Specific Hyperparameters
    branch_p_dim = 256  # Renamed to avoid conflict with state_keys 'P'
    trunk_p_dim = 256   # Renamed
    learning_rate = 3e-4
    batch_size = 32
    num_epochs = 80 # DeepONet might need more epochs or larger batches
    clip_grad_norm = 1.0

    dataset_params_for_plot = {}
    expected_num_controls_main = 0 # From your generate_dataset commands

    if DATASET_TYPE == 'advection':
        dataset_path = "./datasets_full/advection_data_10000s_128nx_600nt.pkl"
        main_state_keys = ['U']; main_num_state_vars = 1; main_bc_state_dim = 2
        dataset_params_for_plot = {'nx': 128, 'ny': 1, 'L': 1.0, 'T': FULL_T_IN_DATAFILE}
    elif DATASET_TYPE == 'euler':
        dataset_path = "./datasets_full/euler_data_10000s_128nx_600nt.pkl"
        main_state_keys = ['rho', 'u']; main_num_state_vars = 2; main_bc_state_dim = 4
        dataset_params_for_plot = {'nx': 128, 'ny': 1, 'L': 1.0, 'T': FULL_T_IN_DATAFILE}
    elif DATASET_TYPE == 'burgers':
        dataset_path = "./datasets_full/burgers_data_10000s_128nx_600nt.pkl"
        main_state_keys = ['U']; main_num_state_vars = 1; main_bc_state_dim = 2
        dataset_params_for_plot = {'nx': 128, 'ny': 1, 'L': 1.0, 'T': FULL_T_IN_DATAFILE}
    elif DATASET_TYPE == 'darcy':
        dataset_path = "./datasets_full/darcy_data_10000s_128nx_600nt.pkl"
        main_state_keys = ['P']; main_num_state_vars = 1; main_bc_state_dim = 2
        dataset_params_for_plot = {'nx': 128, 'ny': 1, 'L': 1.0, 'T': FULL_T_IN_DATAFILE} # Assuming 1D solve as per generator
    else:
        raise ValueError(f"Unknown dataset_type: {DATASET_TYPE}")

    print(f"Loading dataset for {DATASET_TYPE} from {dataset_path}...")
    try:
        with open(dataset_path, 'rb') as f: data_list_all = pickle.load(f)
        print(f"Loaded {len(data_list_all)} samples.")
        # ... (File parameter verification from FNO main block)
    except FileNotFoundError: print(f"Error: Dataset file not found: {dataset_path}"); exit()
    if not data_list_all: print("No data loaded, exiting."); exit()

    random.shuffle(data_list_all)
    n_total = len(data_list_all); n_train = int(0.8 * n_total)
    train_data_list_split = data_list_all[:n_train]; val_data_list_split = data_list_all[n_train:]
    print(f"Train samples: {len(train_data_list_split)}, Validation samples: {len(val_data_list_split)}")

    train_dataset = UniversalPDEDataset(train_data_list_split, 
                                        dataset_type=DATASET_TYPE,
                                        stats_calculation_length=TRAIN_NT_FOR_MODEL,
                                        sequence_length_to_return=TRAIN_NT_FOR_MODEL)
    
    val_dataset = UniversalPDEDataset(val_data_list_split, 
                                      dataset_type=DATASET_TYPE,
                                      stats_calculation_length=TRAIN_NT_FOR_MODEL, # 关键：使用与训练相同的统计长度
                                      sequence_length_to_return=FULL_NT_IN_DATAFILE) # 返回完整序列用于验证评估

    num_workers = 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    # Get actual dimensions from dataset for model init
    actual_nx_from_dataset = train_dataset.nx # This is nx_from_sample_file
    actual_bc_state_dim_ds = train_dataset.bc_state_dim
    actual_num_controls_ds = train_dataset.num_controls
    don_input_bc_ctrl_dim = actual_bc_state_dim_ds + actual_num_controls_ds

    print(f"\nInitializing DeepONet model...")
    print(f"  num_state_vars (branch input): {main_num_state_vars}")
    print(f"  nx (for grid & branch CNN): {actual_nx_from_dataset}")
    print(f"  bc_ctrl_dim (branch input): {don_input_bc_ctrl_dim}")
    print(f"  num_output_vars (model output): {main_num_state_vars}")
    print(f"  branch_p / trunk_p: {branch_p_dim}")


    online_deeponet_model = DeepONetStepper( # Renamed model variable
        num_state_vars=main_num_state_vars,
        nx=actual_nx_from_dataset,
        bc_ctrl_dim=don_input_bc_ctrl_dim,
        num_output_vars=main_num_state_vars, # Stepper predicts same variables
        branch_p=branch_p_dim,
        trunk_p=trunk_p_dim,
        coord_dim=1, # Assuming 1D spatial coordinates
        domain_L=dataset_params_for_plot.get('L', 1.0) # Pass domain length for grid
    )

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{DATASET_TYPE}_{MODEL_TYPE}_trainT{TRAIN_T_TARGET}_p{branch_p_dim}"
    checkpoint_dir = f"./New_ckpt_2/checkpoints_{DATASET_TYPE}_{MODEL_TYPE}" # Simplified path
    results_dir = f"./result_all_2/results_{DATASET_TYPE}_{MODEL_TYPE}"      # Simplified path
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'model_{run_name}.pt')
    save_fig_path_prefix = os.path.join(results_dir, f'result_{run_name}')

    print(f"\nStarting training for DeepONet on {DATASET_TYPE}...")
    start_train_time = time.time()
    online_deeponet_model = train_deeponet_stepper(
        online_deeponet_model, train_loader, dataset_type=DATASET_TYPE,
        train_nt_for_model=TRAIN_NT_FOR_MODEL,
        lr=learning_rate, num_epochs=num_epochs, device=device,
        checkpoint_path=checkpoint_path, clip_grad_norm=clip_grad_norm
    )
    end_train_time = time.time()
    print(f"Training took {end_train_time - start_train_time:.2f} seconds.")

    if val_data_list_split:
        print(f"\nStarting validation for DeepONet on {DATASET_TYPE}...")
        validate_deeponet_stepper(
            online_deeponet_model, val_loader, dataset_type=DATASET_TYPE,
            train_nt_for_model_training=TRAIN_NT_FOR_MODEL,
            T_value_for_model_training=TRAIN_T_TARGET,
            full_T_in_datafile=FULL_T_IN_DATAFILE,
            full_nt_in_datafile=FULL_NT_IN_DATAFILE,
            dataset_params_for_plot=dataset_params_for_plot, device=device,
            save_fig_path_prefix=save_fig_path_prefix
        )
    else:
        print("\nNo validation data. Skipping validation.")

    print("="*60); print(f"Run finished: {run_name}"); # ... (rest of final prints)
    print(f"Final checkpoint: {checkpoint_path}")
    if val_data_list_split: print(f"Validation figures saved with prefix: {save_fig_path_prefix}")
    print("="*60)