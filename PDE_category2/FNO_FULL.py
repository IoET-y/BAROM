import torch.fft
# =============================================================================
#     COMPLETE CODE: FNO Baseline Adapted for Task
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
# import scipy.sparse as sp # Not directly used by FNO training/validation
# from scipy.sparse.linalg import spsolve # Not directly used by FNO
import pickle
import argparse # Added to allow command-line dataset type selection

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
# 2. 通用化数据集定义 (WITH train_nt_limit)
# =============================================================================
class UniversalPDEDataset(Dataset):
    def __init__(self, data_list, dataset_type, train_nt_limit=None): # Added train_nt_limit
        """
        通用化数据集类，适配 Advection, Euler, Burgers, Darcy。

        Args:
            data_list: 包含样本字典的列表。
            dataset_type: 'advection', 'euler', 'burgers', 或 'darcy'。
            train_nt_limit: If specified, truncate sequences to this length.
        """
        if not data_list:
            raise ValueError("data_list cannot be empty")
        self.data_list = data_list
        self.dataset_type = dataset_type.lower()
        self.train_nt_limit = train_nt_limit # Store the limit

        first_sample = data_list[0]
        params = first_sample.get('params', {})

        # --- 获取样本中的原始nt, nx, ny ---
        self.nt_from_sample_file = 0 # Full length in the file for this sample
        self.nx_from_sample_file = 0
        self.ny_from_sample_file = 1 # Default for 1D

        if self.dataset_type == 'advection' or self.dataset_type == 'burgers':
            self.nt_from_sample_file = first_sample['U'].shape[0]
            self.nx_from_sample_file = first_sample['U'].shape[1]
            self.state_keys = ['U']; self.num_state_vars = 1
            self.expected_bc_state_dim = 2
        elif self.dataset_type == 'euler':
            self.nt_from_sample_file = first_sample['rho'].shape[0]
            self.nx_from_sample_file = first_sample['rho'].shape[1]
            self.state_keys = ['rho', 'u']; self.num_state_vars = 2
            self.expected_bc_state_dim = 4
        elif self.dataset_type == 'darcy':
            self.nt_from_sample_file = first_sample['P'].shape[0]
            # For Darcy, nx might be from params or inferred if P is flattened
            self.nx_from_sample_file = params.get('nx', first_sample['P'].shape[1])
            self.ny_from_sample_file = params.get('ny', 1) # Default to 1 if not in params
            self.state_keys = ['P']; self.num_state_vars = 1
            self.expected_bc_state_dim = 2
        elif self.dataset_type == 'heat_delayed_feedback' or \
             self.dataset_type == 'reaction_diffusion_neumann_feedback' or \
             self.dataset_type == 'heat_nonlinear_feedback_gain' or \
             self.dataset_type == 'convdiff': # Added convdiff
            # All these new types have 'U' as the state variable
            self.nt_from_sample_file = first_sample['U'].shape[0]
            self.nx_from_sample_file = first_sample['U'].shape[1]
            self.state_keys = ['U']; self.num_state_vars = 1
            self.expected_bc_state_dim = 2
        else:
            raise ValueError(f"Unknown dataset_type: {self.dataset_type}")

        # Determine effective nt for this dataset instance (used in __getitem__)
        self.effective_nt_for_loader = self.train_nt_limit if self.train_nt_limit is not None else self.nt_from_sample_file

        self.nx = self.nx_from_sample_file
        self.ny = self.ny_from_sample_file
        self.spatial_dim = self.nx * self.ny # ny=1 for 1D cases, product for 2D

        self.bc_state_key = 'BC_State'
        if self.bc_state_key not in first_sample:
            raise KeyError(f"'{self.bc_state_key}' not found in the first sample!")
        actual_bc_state_dim = first_sample[self.bc_state_key].shape[1]
        if actual_bc_state_dim != self.expected_bc_state_dim:
              print(f"Warning: BC_State dimension mismatch for {self.dataset_type}. "
                    f"Expected {self.expected_bc_state_dim}, got {actual_bc_state_dim}. "
                    f"Using actual dimension: {actual_bc_state_dim}")
              self.bc_state_dim = actual_bc_state_dim
        else:
              self.bc_state_dim = self.expected_bc_state_dim

        self.bc_control_key = 'BC_Control'
        if self.bc_control_key in first_sample and first_sample[self.bc_control_key] is not None and first_sample[self.bc_control_key].size > 0 :
            self.num_controls = first_sample[self.bc_control_key].shape[1]
        else:
            self.num_controls = 0
            # print(f"Warning: '{self.bc_control_key}' not found or empty in the first sample. Assuming num_controls = 0.")


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        norm_factors = {}

        current_nt_for_item = self.effective_nt_for_loader

        state_tensors_norm_list = []
        for key in self.state_keys:
            try:
                state_seq_full = sample[key]
                state_seq = state_seq_full[:current_nt_for_item, ...] # Truncate
            except KeyError:
                raise KeyError(f"State variable key '{key}' not found in sample {idx} for dataset type '{self.dataset_type}'")

            if state_seq.shape[0] != current_nt_for_item:
                 raise ValueError(f"Time dimension mismatch for {key} after slice. Expected {current_nt_for_item}, got {state_seq.shape[0]}")

            state_mean = np.mean(state_seq) # Normalize on the (potentially truncated) slice
            state_std = np.std(state_seq) + 1e-8
            state_norm = (state_seq - state_mean) / state_std

            state_tensors_norm_list.append(torch.tensor(state_norm).float())
            norm_factors[f'{key}_mean'] = state_mean
            norm_factors[f'{key}_std'] = state_std

        # --- BC State ---
        bc_state_seq_full = sample[self.bc_state_key]
        bc_state_seq = bc_state_seq_full[:current_nt_for_item, :] # Truncate

        if bc_state_seq.shape[0] != current_nt_for_item:
            raise ValueError(f"Time dim mismatch for BC_State. Expected {current_nt_for_item}, got {bc_state_seq.shape[0]}")

        bc_state_norm = np.zeros_like(bc_state_seq, dtype=np.float32)
        norm_factors[f'{self.bc_state_key}_means'] = np.zeros(self.bc_state_dim)
        norm_factors[f'{self.bc_state_key}_stds'] = np.ones(self.bc_state_dim) # Init stds to 1
        for k_dim in range(self.bc_state_dim):
            col = bc_state_seq[:, k_dim]
            mean_k = np.mean(col)
            std_k = np.std(col)
            if std_k > 1e-8:
                bc_state_norm[:, k_dim] = (col - mean_k) / std_k
                norm_factors[f'{self.bc_state_key}_means'][k_dim] = mean_k
                norm_factors[f'{self.bc_state_key}_stds'][k_dim] = std_k
            else:
                bc_state_norm[:, k_dim] = 0.0 # Or col - mean_k which should be close to 0
                norm_factors[f'{self.bc_state_key}_means'][k_dim] = mean_k
                # norm_factors[f'{self.bc_state_key}_stds'][k_dim] remains 1.0
        bc_state_tensor_norm = torch.tensor(bc_state_norm).float()

        # --- BC Control ---
        if self.num_controls > 0:
            try:
                bc_control_seq_full = sample[self.bc_control_key]
                bc_control_seq = bc_control_seq_full[:current_nt_for_item, :] # Truncate
                if bc_control_seq.shape[0] != current_nt_for_item:
                    raise ValueError(f"Time dim mismatch for BC_Control. Expected {current_nt_for_item}, got {bc_control_seq.shape[0]}")
                if bc_control_seq.shape[1] != self.num_controls:
                     raise ValueError(f"Control dim mismatch in sample {idx}. Expected {self.num_controls}, got {bc_control_seq.shape[1]}.")

                bc_control_norm = np.zeros_like(bc_control_seq, dtype=np.float32)
                norm_factors[f'{self.bc_control_key}_means'] = np.zeros(self.num_controls)
                norm_factors[f'{self.bc_control_key}_stds'] = np.ones(self.num_controls) # Init stds to 1
                for k_dim in range(self.num_controls):
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
                # print(f"Warning: Sample {idx} missing '{self.bc_control_key}'. Using zeros.")
                bc_control_tensor_norm = torch.zeros((current_nt_for_item, self.num_controls), dtype=torch.float32)
                norm_factors[f'{self.bc_control_key}_means'] = np.zeros(self.num_controls)
                norm_factors[f'{self.bc_control_key}_stds'] = np.ones(self.num_controls)
        else: # No controls
            bc_control_tensor_norm = torch.empty((current_nt_for_item, 0), dtype=torch.float32)


        bc_ctrl_tensor_norm = torch.cat((bc_state_tensor_norm, bc_control_tensor_norm), dim=-1)

        if self.num_state_vars == 1:
            output_state_tensors = state_tensors_norm_list[0] # Return tensor directly
        else:
            output_state_tensors = state_tensors_norm_list # Return list of tensors

        return output_state_tensors, bc_ctrl_tensor_norm, norm_factors

# =============================================================================
# FNO Components (Helper Classes)
# =============================================================================
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def compl_mul1d(self, input_tensor, weights): # Renamed input to input_tensor
        return torch.einsum("bix,iox->box", input_tensor, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width, input_channels, output_channels, num_layers=4):
        super(FNO1d, self).__init__()
        self.modes1 = modes
        self.width = width
        self.input_channels = input_channels
        self.output_channels = output_channels # This should be num_state_vars
        self.num_layers = num_layers

        self.fc0 = nn.Linear(input_channels, self.width)
        self.convs = nn.ModuleList()
        self.ws = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SpectralConv1d(self.width, self.width, self.modes1))
            self.ws.append(nn.Conv1d(self.width, self.width, 1))
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, output_channels) # Output matches num_state_vars

    def forward(self, x):
        # x shape: [B, N, C_in] (N is spatial dim, C_in is input_channels)
        try:
            x_lifted = self.fc0(x) # [B, N, W]
        except RuntimeError as e:
            print(f"ERROR FNO1d fc0: Input shape={x.shape}, Expected C_in={self.input_channels}, Weight shape={self.fc0.weight.shape}")
            raise e
        x_permuted = x_lifted.permute(0, 2, 1) # [B, W, N]
        x_proc = x_permuted
        for i in range(self.num_layers):
            x1 = self.convs[i](x_proc)
            x2 = self.ws[i](x_proc)
            x_proc = x1 + x2
            x_proc = F.gelu(x_proc)
        x_out_perm = x_proc.permute(0, 2, 1) # [B, N, W]
        x_out = self.fc1(x_out_perm)
        x_out = F.gelu(x_out)
        x_out = self.fc2(x_out) # [B, N, C_out] (C_out is output_channels, i.e. num_state_vars)
        return x_out

# =============================================================================
# FNO Training Function (One-Step Prediction Loss, adapted for TRAIN_NT_FOR_MODEL)
# =============================================================================
def train_fno_stepper(model, data_loader, dataset_type, train_nt_for_model, # Added train_nt_for_model
                      lr=1e-3, num_epochs=50, device='cuda',
                      checkpoint_path='fno_checkpoint.pt', clip_grad_norm=1.0):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    mse_loss = nn.MSELoss(reduction='mean')

    start_epoch = 0
    best_loss = float('inf')
    if os.path.exists(checkpoint_path):
        print(f"Loading FNO checkpoint from {checkpoint_path} ...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                try: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except: print("Warning: FNO Optimizer state mismatch. Reinitializing.")
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_loss = checkpoint.get('loss', float('inf'))
            print(f"Resuming FNO training from epoch {start_epoch}")
        except Exception as e:
            print(f"Error loading FNO checkpoint: {e}. Starting fresh.")

    # num_state_vars from model output channels
    num_state_vars = model.output_channels

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_train_loss = 0.0 # Renamed to avoid conflict
        num_batches = 0
        batch_start_time = time.time()

        for i, (state_data_loaded, BC_Ctrl_tensor_loaded, _) in enumerate(data_loader):
            # Data from loader is already truncated to train_nt_for_model by UniversalPDEDataset
            if isinstance(state_data_loaded, list): # Euler
                state_seq_true_train = torch.stack(state_data_loaded, dim=-1).to(device)
            else: # Advection, Burgers, Darcy
                state_seq_true_train = state_data_loaded.unsqueeze(-1).to(device)

            BC_Ctrl_seq_train = BC_Ctrl_tensor_loaded.to(device)
            batch_size, nt_loaded, nx_or_N, _ = state_seq_true_train.shape
            _, _, bc_ctrl_dim = BC_Ctrl_seq_train.shape

            if nt_loaded != train_nt_for_model:
                raise ValueError(f"Mismatch: nt from DataLoader ({nt_loaded}) != train_nt_for_model ({train_nt_for_model})")

            optimizer.zero_grad()
            total_sequence_loss = 0.0

            # Loop over the truncated time steps for one-step prediction loss
            for t in range(train_nt_for_model - 1): # Iterate up to train_nt_for_model-1
                u_n_true = state_seq_true_train[:, t, :, :]       # [B, nx, num_vars]
                bc_ctrl_n = BC_Ctrl_seq_train[:, t, :]            # [B, bc_ctrl_dim]
                bc_ctrl_n_expanded = bc_ctrl_n.unsqueeze(1)       # [B, 1, bc_ctrl_dim]
                bc_ctrl_n_spatial = bc_ctrl_n_expanded.repeat(1, nx_or_N, 1) # [B, nx, bc_ctrl_dim]

                fno_input = torch.cat((u_n_true, bc_ctrl_n_spatial), dim=-1) # [B, nx, num_vars + bc_ctrl_dim]
                u_np1_pred = model(fno_input)                     # [B, nx, num_vars]
                u_np1_true = state_seq_true_train[:, t+1, :, :]   # [B, nx, num_vars]

                step_loss = mse_loss(u_np1_pred, u_np1_true)
                total_sequence_loss += step_loss

            current_batch_loss = total_sequence_loss / (train_nt_for_model - 1) # Renamed variable
            epoch_train_loss += current_batch_loss.item()
            num_batches += 1

            current_batch_loss.backward()
            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            optimizer.step()

            if (i + 1) % 50 == 0: # Print more often
                batch_time_elapsed = time.time() - batch_start_time
                print(f"  FNO Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(data_loader)}, "
                      f"Batch Loss: {current_batch_loss.item():.4e}, Time/50Batch: {batch_time_elapsed:.2f}s")
                batch_start_time = time.time() # Reset timer

        avg_epoch_loss = epoch_train_loss / num_batches
        print(f"FNO Epoch {epoch+1}/{num_epochs} finished. Avg Training Loss: {avg_epoch_loss:.6f}")

        scheduler.step(avg_epoch_loss)
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            print(f"Saving FNO checkpoint with loss {best_loss:.6f} to {checkpoint_path}")
            save_dict = {
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'loss': best_loss,
                'modes': model.modes1, 'width': model.width,
                'input_channels': model.input_channels, 'output_channels': model.output_channels,
                'num_layers': model.num_layers, 'dataset_type': dataset_type
            }
            torch.save(save_dict, checkpoint_path)

    print("FNO Training finished.")
    if os.path.exists(checkpoint_path):
        print(f"Loading best FNO model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    return model

# =============================================================================
# FNO Validation Function (Autoregressive Rollout for Multiple Horizons)
# =============================================================================
def validate_fno_stepper(model, data_loader, dataset_type,
                         # Prefix for figure paths
                         # Parameters for defining horizons and data structure:
                         train_nt_for_model_training: int,
                         T_value_for_model_training: float,
                         full_T_in_datafile: float,
                         full_nt_in_datafile: int,
                         dataset_params_for_plot: dict, device='cuda', save_fig_path_prefix='fno_result'): # Contains nx, ny, L for plotting
    model.eval()
    mse = nn.MSELoss(reduction='mean')

    if dataset_type == 'advection' or dataset_type == 'burgers': state_keys_val = ['U']
    elif dataset_type == 'euler': state_keys_val = ['rho', 'u']
    elif dataset_type == 'darcy': state_keys_val = ['P']
    else: raise ValueError(f"Unknown dataset_type '{dataset_type}' in FNO validation")
    num_state_vars_val = len(state_keys_val)
    if model.output_channels != num_state_vars_val:
        print(f"Warning: FNO model output channels ({model.output_channels}) != num_state_vars from dataset_type ({num_state_vars_val})")


    # Define test horizons (T values)
    test_horizons_T_values = [T_value_for_model_training]
    if full_T_in_datafile > T_value_for_model_training:
        test_horizons_T_values.append(T_value_for_model_training + 0.5 * (full_T_in_datafile - T_value_for_model_training)) # Mid-point
        test_horizons_T_values.append(full_T_in_datafile)
    test_horizons_T_values = sorted(list(set(h for h in test_horizons_T_values if h <= full_T_in_datafile + 1e-6)))


    print(f"FNO Validation for T_horizons: {test_horizons_T_values}")
    print(f"  Model trained with nt={train_nt_for_model_training} for T={T_value_for_model_training}")
    print(f"  Datafile has nt={full_nt_in_datafile} for T={full_T_in_datafile}")

    # Metrics storage for the primary horizon (T_value_for_model_training)
    results_primary_horizon = {key: {'mse': [], 'rmse': [], 'relative_error': [], 'max_error': []} for key in state_keys_val}
    overall_rel_err_primary_horizon = []

    with torch.no_grad():
        # Use only the first sample from the validation loader for detailed plotting
        try:
            state_data_full_loaded, BC_Ctrl_tensor_full_loaded, norm_factors_batch = next(iter(data_loader))
        except StopIteration:
            print("Validation data_loader is empty. Skipping FNO validation.")
            return

        # Prepare full sequences for the first sample
        if isinstance(state_data_full_loaded, list): # Euler
            state_seq_true_norm_full = torch.stack(state_data_full_loaded, dim=-1)[0].to(device) # [nt_full, nx, num_vars]
        else: # Advection, Burgers, Darcy
            state_seq_true_norm_full = state_data_full_loaded.unsqueeze(-1)[0].to(device) # [nt_full, nx, num_vars]

        BC_Ctrl_seq_norm_full = BC_Ctrl_tensor_full_loaded[0].to(device) # [nt_full, bc_ctrl_dim]

        nt_file_check, nx_or_N_file_check, num_vars_check = state_seq_true_norm_full.shape
        if nt_file_check != full_nt_in_datafile:
            print(f"Warning: nt from val_loader ({nt_file_check}) != expected full_nt_in_datafile ({full_nt_in_datafile}). Adjusting.")
            # This can happen if the last batch of data_list was smaller than train_nt_limit during UniversalPDEDataset init for val_dataset
            # For safety, use the actual loaded full_nt for this sample.
            # However, UniversalPDEDataset with train_nt_limit=None should always return full length from file.
            # This warning might indicate an issue in UniversalPDEDataset's handling of train_nt_limit=None
            # OR that full_nt_in_datafile param passed here is not matching the actual file content.

        # Extract norm_factors for this single sample (index 0 of the batch)
        norm_factors_sample = {}
        for key, val_tensor in norm_factors_batch.items():
            if isinstance(val_tensor, torch.Tensor) or isinstance(val_tensor, np.ndarray):
                norm_factors_sample[key] = val_tensor[0].cpu().numpy() if val_tensor.ndim > 0 else val_tensor.cpu().numpy() # Get item for sample 0
            else: # scalar
                norm_factors_sample[key] = val_tensor


        # --- Autoregressive Rollout for each horizon ---
        u_initial_norm = state_seq_true_norm_full[0:1, :, :] # Initial condition [1, nx, num_vars]

        for T_horizon_current in test_horizons_T_values:
            nt_for_rollout = int((T_horizon_current / full_T_in_datafile) * (full_nt_in_datafile - 1)) + 1
            nt_for_rollout = min(nt_for_rollout, full_nt_in_datafile) # Cap at available data points

            print(f"\n  Rollout for T_horizon = {T_horizon_current:.2f} (nt = {nt_for_rollout})")

            u_pred_seq_norm_current_horizon = torch.zeros(nt_for_rollout, nx_or_N_file_check, num_vars_check, device=device)
            u_current_pred_step = u_initial_norm.clone() # Start with the true initial condition [1, nx, num_vars]
            u_pred_seq_norm_current_horizon[0, :, :] = u_current_pred_step.squeeze(0)

            # Slice BCs for this rollout duration
            BC_Ctrl_for_rollout = BC_Ctrl_seq_norm_full[:nt_for_rollout, :]

            for t_step in range(nt_for_rollout - 1):
                bc_ctrl_n_step = BC_Ctrl_for_rollout[t_step:t_step+1, :] # [1, bc_ctrl_dim]
                bc_ctrl_n_expanded_step = bc_ctrl_n_step.unsqueeze(1)    # [1, 1, bc_ctrl_dim]
                bc_ctrl_n_spatial_step = bc_ctrl_n_expanded_step.repeat(1, nx_or_N_file_check, 1) # [1, nx, bc_ctrl_dim]

                # Ensure u_current_pred_step is [1, nx, num_vars]
                if u_current_pred_step.shape[0] != 1: u_current_pred_step = u_current_pred_step.unsqueeze(0)

                fno_input_step = torch.cat((u_current_pred_step, bc_ctrl_n_spatial_step), dim=-1)
                u_next_pred_norm_step = model(fno_input_step) # Output: [1, nx, num_vars]

                u_pred_seq_norm_current_horizon[t_step+1, :, :] = u_next_pred_norm_step.squeeze(0)
                u_current_pred_step = u_next_pred_norm_step # Update for next step

            # Denormalize & Calculate Metrics for this horizon
            U_pred_denorm_horizon = {}
            U_gt_denorm_horizon = {}
            combined_pred_list_horizon = []
            combined_gt_list_horizon = []

            # Ground truth sliced for this horizon
            state_seq_true_norm_sliced_horizon = state_seq_true_norm_full[:nt_for_rollout, :, :]

            pred_seq_np_horizon = u_pred_seq_norm_current_horizon.cpu().numpy()
            gt_seq_np_horizon = state_seq_true_norm_sliced_horizon.cpu().numpy()

            for k_idx, key_val in enumerate(state_keys_val):
                mean_k_val = norm_factors_sample[f'{key_val}_mean']
                std_k_val = norm_factors_sample[f'{key_val}_std']
                # Ensure mean_k and std_k are scalars
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
                rmse_k_hor = np.sqrt(mse_k_hor)
                rel_err_k_hor = np.linalg.norm(pred_denorm_var - gt_denorm_var, 'fro') / (np.linalg.norm(gt_denorm_var, 'fro') + 1e-10)
                max_err_k_hor = np.max(np.abs(pred_denorm_var - gt_denorm_var))
                print(f"    Metrics for '{key_val}' @ T={T_horizon_current:.1f}: MSE={mse_k_hor:.3e}, RelErr={rel_err_k_hor:.3e}")

                # Store results if this is the primary evaluation horizon
                if abs(T_horizon_current - T_value_for_model_training) < 1e-6:
                    results_primary_horizon[key_val]['mse'].append(mse_k_hor)
                    results_primary_horizon[key_val]['rmse'].append(rmse_k_hor)
                    results_primary_horizon[key_val]['relative_error'].append(rel_err_k_hor)
                    results_primary_horizon[key_val]['max_error'].append(max_err_k_hor)

            U_pred_vec_hor = np.concatenate(combined_pred_list_horizon)
            U_gt_vec_hor = np.concatenate(combined_gt_list_horizon)
            overall_rel_err_val = np.linalg.norm(U_pred_vec_hor - U_gt_vec_hor) / (np.linalg.norm(U_gt_vec_hor) + 1e-10)
            print(f"    Overall RelErr @ T={T_horizon_current:.1f}: {overall_rel_err_val:.3e}")
            if abs(T_horizon_current - T_value_for_model_training) < 1e-6:
                overall_rel_err_primary_horizon.append(overall_rel_err_val)

            # Visualization for this horizon
            fig, axs = plt.subplots(num_state_vars_val, 3, figsize=(18, 5 * num_state_vars_val), squeeze=False)
            fig_L = dataset_params_for_plot.get('L', 1.0)
            fig_nx = dataset_params_for_plot.get('nx', nx_or_N_file_check) # Use loaded if not in params
            fig_ny = dataset_params_for_plot.get('ny', 1)


            for k_idx, key_val in enumerate(state_keys_val):
                gt_plot = U_gt_denorm_horizon[key_val]
                pred_plot = U_pred_denorm_horizon[key_val]
                diff_plot = np.abs(pred_plot - gt_plot)
                max_err_plot = np.max(diff_plot) if diff_plot.size > 0 else 0

                vmin_plot = min(np.min(gt_plot), np.min(pred_plot)) if gt_plot.size > 0 else 0
                vmax_plot = max(np.max(gt_plot), np.max(pred_plot)) if gt_plot.size > 0 else 1

                is_1d_plot = (fig_ny == 1)
                plot_extent = [0, fig_L, 0, T_horizon_current]

                if is_1d_plot:
                    im0 = axs[k_idx,0].imshow(gt_plot, aspect='auto', origin='lower', vmin=vmin_plot, vmax=vmax_plot, extent=plot_extent, cmap='viridis')
                    axs[k_idx,0].set_title(f"Truth ({key_val})")
                    im1 = axs[k_idx,1].imshow(pred_plot, aspect='auto', origin='lower', vmin=vmin_plot, vmax=vmax_plot, extent=plot_extent, cmap='viridis')
                    axs[k_idx,1].set_title(f"FNO Pred ({key_val})")
                    im2 = axs[k_idx,2].imshow(diff_plot, aspect='auto', origin='lower', extent=plot_extent, cmap='magma')
                    axs[k_idx,2].set_title(f"Abs Error (Max:{max_err_plot:.2e})")
                    for j_plot in range(3): axs[k_idx,j_plot].set_xlabel("x"); axs[k_idx,0].set_ylabel("t")
                    plt.colorbar(im0, ax=axs[k_idx,0]); plt.colorbar(im1, ax=axs[k_idx,1]); plt.colorbar(im2, ax=axs[k_idx,2])
                else: # Basic 2D plot (last time step)
                    t_idx_2d_plot = -1
                    if gt_plot.shape[0] > 0 and gt_plot.shape[1] == fig_nx * fig_ny:
                        gt_2d = gt_plot[t_idx_2d_plot,:].reshape(fig_nx, fig_ny)
                        pred_2d = pred_plot[t_idx_2d_plot,:].reshape(fig_nx, fig_ny)
                        diff_2d = diff_plot[t_idx_2d_plot,:].reshape(fig_nx, fig_ny)
                        vmin_2d = min(np.min(gt_2d), np.min(pred_2d))
                        vmax_2d = max(np.max(gt_2d), np.max(pred_2d))
                        plot_extent_2d = [0, fig_L, 0, fig_L] # Assume square domain LxL for 2D plot

                        im0 = axs[k_idx,0].imshow(gt_2d, aspect='auto', origin='lower', vmin=vmin_2d, vmax=vmax_2d, extent=plot_extent_2d, cmap='viridis')
                        axs[k_idx,0].set_title(f"Truth ({key_val}) @ T={T_horizon_current:.1f}")
                        im1 = axs[k_idx,1].imshow(pred_2d, aspect='auto', origin='lower', vmin=vmin_2d, vmax=vmax_2d, extent=plot_extent_2d, cmap='viridis')
                        axs[k_idx,1].set_title(f"FNO Pred ({key_val})")
                        im2 = axs[k_idx,2].imshow(diff_2d, aspect='auto', origin='lower', extent=plot_extent_2d, cmap='magma')
                        axs[k_idx,2].set_title(f"Abs Error (Max:{np.max(diff_2d):.2e})")
                        plt.colorbar(im0, ax=axs[k_idx,0]); plt.colorbar(im1, ax=axs[k_idx,1]); plt.colorbar(im2, ax=axs[k_idx,2])
                    else:
                        axs[k_idx,0].text(0.5, 0.5, "2D Plot Error (Shape)", ha='center', va='center')
                        axs[k_idx,1].text(0.5, 0.5, "Data Check doc", ha='center', va='center')
                        axs[k_idx,2].text(0.5, 0.5, "Shape mismatch for 2D", ha='center', va='center')


            model_modes_plot = getattr(model, 'modes1', '?')
            model_width_plot = getattr(model, 'width', '?')
            fig.suptitle(f"FNO Validation ({dataset_type.capitalize()}) @ T={T_horizon_current:.1f} - Modes:{model_modes_plot}, Width:{model_width_plot}")
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            current_fig_path = save_fig_path_prefix + f"_T{str(T_horizon_current).replace('.', 'p')}.png"
            plt.savefig(current_fig_path)
            print(f"  Saved FNO validation visualization to {current_fig_path}")
            plt.show()

    # Print Summary for the primary horizon
    print(f"\n--- FNO Validation Summary (Metrics for T={T_value_for_model_training:.1f}) ---")
    for key_val in state_keys_val:
        if results_primary_horizon[key_val]['mse']: # Check if any results were appended
            avg_mse = np.mean(results_primary_horizon[key_val]['mse'])
            avg_rmse = np.mean(results_primary_horizon[key_val]['rmse'])
            avg_rel_err = np.mean(results_primary_horizon[key_val]['relative_error'])
            avg_max_err = np.mean(results_primary_horizon[key_val]['max_error'])
            print(f"  Variable '{key_val}': Avg MSE={avg_mse:.4e}, RMSE={avg_rmse:.4e}, RelErr={avg_rel_err:.4e}, MaxErr={avg_max_err:.4e}")
    if overall_rel_err_primary_horizon:
        print(f"  Overall Avg Trajectory Relative Error @ T={T_value_for_model_training:.1f}: {np.mean(overall_rel_err_primary_horizon):.4e}")
    print("------------------------")

# =============================================================================
# Main Block - Modified to Run FNO Baseline with Task Definition
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run FNO baseline for PDE datasets.") # Added argparse
    parser.add_argument('--datatype', type=str, required=True,
                        choices=['advection', 'euler', 'burgers', 'darcy',
                                 'heat_delayed_feedback',
                                 'reaction_diffusion_neumann_feedback',
                                 'heat_nonlinear_feedback_gain',
                                 'convdiff'], # Added convdiff
                        help='Type of dataset to run FNO on.')
    args = parser.parse_args()
    DATASET_TYPE = args.datatype
    MODEL_TYPE = 'FNO'    # This script is now tailored for FNO

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Selected Dataset Type: {DATASET_TYPE.upper()}")
    print(f"Selected Model Type: {MODEL_TYPE.upper()}")

    # --- Key Time Parameters (MUST MATCH YOUR generate_dataset.py CALLS) ---
    FULL_T_IN_DATAFILE = 2.0  # T value your .pkl files were generated with
    FULL_NT_IN_DATAFILE = 600 # nt value (number of points) your .pkl files were generated with

    TRAIN_T_TARGET = 1.0      # Target T for training
    # Calculate the number of timesteps for training (indices 0 to TRAIN_NT_FOR_MODEL-1)
    TRAIN_NT_FOR_MODEL = int((TRAIN_T_TARGET / FULL_T_IN_DATAFILE) * (FULL_NT_IN_DATAFILE - 1)) + 1
    print(f"Datafile T={FULL_T_IN_DATAFILE}, nt={FULL_NT_IN_DATAFILE}")
    print(f"Training with T={TRAIN_T_TARGET}, nt={TRAIN_NT_FOR_MODEL}")

    # --- FNO Specific Hyperparameters ---
    fno_modes = 16
    fno_width = 64
    fno_layers = 4
    learning_rate = 1e-3 # FNO can often use higher LR
    batch_size = 32
    num_epochs = 80 # Adjust as needed
    clip_grad_norm = 1.0

    # --- Dataset Paths and Parameters (update paths to your ../datasets_full) ---
    dataset_params_for_plot = {} # To store L, nx, ny for plotting in validation

    if DATASET_TYPE == 'advection':
        dataset_path = "./datasets_full/advection_data_10000s_128nx_600nt.pkl"
        # From generate_dataset call: nx=128, nt=600, T=2.0, L=1.0 (default)
        state_keys = ['U']; num_state_vars = 1; bc_state_dim = 2; expected_num_controls = 0
        dataset_params_for_plot = {'nx': 128, 'ny': 1, 'L': 1.0, 'T': FULL_T_IN_DATAFILE}
    elif DATASET_TYPE == 'euler':
        dataset_path = "./datasets_full/euler_data_10000s_128nx_600nt.pkl"
        state_keys = ['rho', 'u']; num_state_vars = 2; bc_state_dim = 4; expected_num_controls = 0
        dataset_params_for_plot = {'nx': 128, 'ny': 1, 'L': 1.0, 'T': FULL_T_IN_DATAFILE}
    elif DATASET_TYPE == 'burgers':
        dataset_path = "./datasets_full/burgers_data_10000s_128nx_600nt.pkl"
        state_keys = ['U']; num_state_vars = 1; bc_state_dim = 2; expected_num_controls = 0
        dataset_params_for_plot = {'nx': 128, 'ny': 1, 'L': 1.0, 'T': FULL_T_IN_DATAFILE}
    elif DATASET_TYPE == 'darcy':
        # IMPORTANT: The Darcy generator in your script hardcodes nt=50.
        # If your darcy files reflect --nt 600 --T 2, this section is fine.
        # If they reflect the internal nt=50, T=1 (approx), then FULL_NT_IN_DATAFILE and FULL_T_IN_DATAFILE
        # must be changed here for Darcy (e.g., 50 and 1.0 respectively).
        dataset_path = "./datasets_full/darcy_data_10000s_128nx_600nt.pkl"
        state_keys = ['P']; num_state_vars = 1; bc_state_dim = 2; expected_num_controls = 0
        # For Darcy, nx_file is 128 (total spatial points if 1D solver, or one dim if 2D)
        # The UniversalPDEDataset tries to infer nx, ny. For plotting, we pass explicitly.
        # If Darcy was solved on 1D grid:
        dataset_params_for_plot = {'nx': 128, 'ny': 1, 'L': 1.0, 'T': FULL_T_IN_DATAFILE}
        # If Darcy was solved on e.g. 32x32 (if nx was 32 and ny was also 32 from generator):
        # dataset_params_for_plot = {'nx': 32, 'ny': 32, 'L': 1.0, 'T': FULL_T_IN_DATAFILE}
        # Your Darcy generator uses nx for 1D solve, so nx=128, ny=1 is appropriate here.
    elif DATASET_TYPE == 'heat_delayed_feedback':
        dataset_path = "./datasets_new_feedback/heat_delayed_feedback_v1_5000s_64nx_300nt.pkl" # UPDATE PATH
        state_keys = ['U']; num_state_vars = 1
        dataset_params_for_plot = {'nx': 64, 'ny': 1, 'L': 1.0, 'T': FULL_T_IN_DATAFILE}
    elif DATASET_TYPE == 'reaction_diffusion_neumann_feedback':
        dataset_path = "./datasets_new_feedback/reaction_diffusion_neumann_feedback_v1_5000s_64nx_300nt.pkl" # UPDATE PATH
        state_keys = ['U']; num_state_vars = 1
        dataset_params_for_plot = {'nx': 64, 'ny': 1, 'L': 1.0, 'T': FULL_T_IN_DATAFILE}
    elif DATASET_TYPE == 'heat_nonlinear_feedback_gain':

        dataset_path = "./datasets_new_feedback/heat_nonlinear_feedback_gain_v1_5000s_64nx_300nt.pkl" # UPDATE PATH
        state_keys = ['U']; num_state_vars = 1
        dataset_params_for_plot = {'nx': 64, 'ny': 1, 'L': 1.0, 'T': FULL_T_IN_DATAFILE}
    elif DATASET_TYPE == 'convdiff': # Added convdiff
        dataset_path = "./datasets_new_feedback/convdiff_v1_5000s_64nx_300nt.pkl" # UPDATE PATH
        state_keys = ['U']; num_state_vars = 1
        dataset_params_for_plot = {'nx': 64, 'ny': 1, 'L': 1.0, 'T': FULL_T_IN_DATAFILE}
    else:
        raise ValueError(f"Unknown dataset_type: {DATASET_TYPE}")
        
    FULL_T_IN_DATAFILE = 2.0  # MUST MATCH YOUR GENERATION SCRIPT
    FULL_NT_IN_DATAFILE = 300 # MUST MATCH YOUR GENERATION SCRIPT
    TRAIN_T_TARGET = 1.5      # Example
    TRAIN_NT_FOR_MODEL = int((TRAIN_T_TARGET / FULL_T_IN_DATAFILE) * (FULL_NT_IN_DATAFILE - 1)) + 1

    print(f"Dataset: {DATASET_TYPE}")
    print(f"Datafile T={FULL_T_IN_DATAFILE}, nt_points={FULL_NT_IN_DATAFILE}")
    print(f"Training with T_duration={TRAIN_T_TARGET}, nt_points_in_sequence={TRAIN_NT_FOR_MODEL}")

    print(f"Loading dataset for {DATASET_TYPE} from {dataset_path}...")
    try:
        with open(dataset_path, 'rb') as f: data_list_all = pickle.load(f)
        print(f"Loaded {len(data_list_all)} samples.")
        if data_list_all:
            sample_params_check = data_list_all[0].get('params', {})
            file_nt_check = data_list_all[0][state_keys[0]].shape[0]
            file_T_check_val = sample_params_check.get('T', 'N/A') # Use a different var name
            file_L_check_val = sample_params_check.get('L', 'N/A')
            file_nx_check_val = sample_params_check.get('nx', data_list_all[0][state_keys[0]].shape[1])
            print(f"  Sample 0 from file: nt={file_nt_check}, T={file_T_check_val}, nx={file_nx_check_val}, L={file_L_check_val}")
            if file_nt_check != FULL_NT_IN_DATAFILE:
                print(f"  CRITICAL WARNING: FULL_NT_IN_DATAFILE ({FULL_NT_IN_DATAFILE}) in script "
                      f"differs from actual nt in file ({file_nt_check}) for {DATASET_TYPE}. "
                      f"Ensure FULL_NT_IN_DATAFILE is correctly set for this dataset type in the script.")
            # Update plot params directly from file if available and more accurate
            if isinstance(file_T_check_val, (int,float)): dataset_params_for_plot['T'] = file_T_check_val
            if isinstance(file_L_check_val, (int,float)): dataset_params_for_plot['L'] = file_L_check_val
            dataset_params_for_plot['nx'] = file_nx_check_val # Always update nx from file for safety
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_path}"); exit()
    if not data_list_all: print("No data loaded, exiting."); exit()

    random.shuffle(data_list_all)
    n_total = len(data_list_all); n_train = int(0.8 * n_total)
    train_data_list_split = data_list_all[:n_train]; val_data_list_split = data_list_all[n_train:]
    print(f"Train samples: {len(train_data_list_split)}, Validation samples: {len(val_data_list_split)}")

    train_dataset = UniversalPDEDataset(train_data_list_split, dataset_type=DATASET_TYPE,
                                        train_nt_limit=TRAIN_NT_FOR_MODEL)
    val_dataset = UniversalPDEDataset(val_data_list_split, dataset_type=DATASET_TYPE,
                                      train_nt_limit=None)

    num_workers = 1 # Adjusted from 2 for potentially easier debugging
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    actual_bc_state_dim = train_dataset.bc_state_dim
    actual_num_controls = train_dataset.num_controls
    fno_input_channels = num_state_vars + actual_bc_state_dim + actual_num_controls
    fno_output_channels = num_state_vars
    
    # --- Initialize FNO Model ---
    print(f"\nInitializing FNO model: InputChannels={fno_input_channels}, OutputChannels={fno_output_channels}")
    online_fno_model = FNO1d( # Renamed
        modes=fno_modes,
        width=fno_width,
        input_channels=fno_input_channels,
        output_channels=fno_output_channels,
        num_layers=fno_layers
    )

    # Paths
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # More descriptive run name
    run_name = f"{DATASET_TYPE}_{MODEL_TYPE}_trainT{TRAIN_T_TARGET}_m{fno_modes}_w{fno_width}"
    checkpoint_dir = f"./New_ckpt_2/checkpoints_{DATASET_TYPE}_{MODEL_TYPE}" # Simplified path
    results_dir = f"./result_all_2/results_{DATASET_TYPE}_{MODEL_TYPE}"      # Simplified path
    os.makedirs(checkpoint_dir, exist_ok=True); os.makedirs(results_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'model_{run_name}.pt')
    save_fig_path_prefix = os.path.join(results_dir, f'result_{run_name}') # Prefix for multiple horizon plots

    # --- Training ---
    print(f"\nStarting training for FNO on {DATASET_TYPE}...")
    start_train_time = time.time()
    online_fno_model = train_fno_stepper(
        online_fno_model, train_loader, dataset_type=DATASET_TYPE,
        train_nt_for_model=TRAIN_NT_FOR_MODEL, # Pass the training horizon
        lr=learning_rate, num_epochs=num_epochs, device=device,
        checkpoint_path=checkpoint_path, clip_grad_norm=clip_grad_norm
    )
    end_train_time = time.time()
    print(f"Training took {end_train_time - start_train_time:.2f} seconds.")

    # --- Validation ---
    if val_data_list_split:
        print(f"\nStarting validation for FNO on {DATASET_TYPE}...")
        validate_fno_stepper(
            online_fno_model, val_loader, dataset_type=DATASET_TYPE, device=device,
            train_nt_for_model_training=TRAIN_NT_FOR_MODEL,
            T_value_for_model_training=TRAIN_T_TARGET,
            full_T_in_datafile=FULL_T_IN_DATAFILE,
            full_nt_in_datafile=FULL_NT_IN_DATAFILE,
            dataset_params_for_plot=dataset_params_for_plot,  save_fig_path_prefix=save_fig_path_prefix
        )
    else:
        print("\nNo validation data. Skipping validation.")

    print("="*60)
    print(f"Run finished: {run_name}")
    print(f"Final checkpoint: {checkpoint_path}")
    if val_data_list_split: print(f"Validation figures saved with prefix: {save_fig_path_prefix}")
    print("="*60)