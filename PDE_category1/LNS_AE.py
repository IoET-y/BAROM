# LNS_AE
# =============================================================================
# latent deepOnet (Adapted for Task1) ref:https://github.com/katiana22/latent-deeponet
# Learning nonlinear operators in latent spaces for real-time predictions of complex dynamics in physical systems.Nature Communications,2024
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
    torch.backends.cudnn.deterministic = True # Potentially slower, but good for reproducibility
# ---------------------

print(f"LatentDNO-Stepper Script (Task Adapted) started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

class UniversalPDEDataset(Dataset):
    def __init__(self, data_list, dataset_type,
                 global_norm_stats=None, # Pass pre-computed normalization statistics
                 train_nt_limit=None):
        if not data_list: raise ValueError("data_list cannot be empty")
        self.data_list = data_list
        self.dataset_type = dataset_type.lower()
        self.train_nt_limit = train_nt_limit
        self.global_norm_stats = global_norm_stats

        first_sample = data_list[0]
        params = first_sample.get('params', {})

        self.nt_from_sample_file=0; self.nx_from_sample_file=0; self.ny_from_sample_file=1
        if self.dataset_type in ['advection', 'burgers']:
            self.nt_from_sample_file=first_sample['U'].shape[0]; self.nx_from_sample_file=first_sample['U'].shape[1]
            self.state_keys=['U']; self.num_state_vars=1; self.expected_bc_state_dim=2
        elif self.dataset_type == 'euler':
            self.nt_from_sample_file=first_sample['rho'].shape[0]; self.nx_from_sample_file=first_sample['rho'].shape[1]
            self.state_keys=['rho','u']; self.num_state_vars=2; self.expected_bc_state_dim=4
        elif self.dataset_type == 'darcy':
            self.nt_from_sample_file=first_sample['P'].shape[0]; self.nx_from_sample_file=params.get('nx', first_sample['P'].shape[1])
            self.ny_from_sample_file=params.get('ny',1); self.state_keys=['P']; self.num_state_vars=1; self.expected_bc_state_dim=2
        else: raise ValueError(f"Unknown type: {self.dataset_type}")

        self.effective_nt_for_loader = self.train_nt_limit if self.train_nt_limit is not None else self.nt_from_sample_file
        self.nx=self.nx_from_sample_file; self.ny=self.ny_from_sample_file; self.spatial_dim=self.nx*self.ny

        self.bc_state_key='BC_State'; actual_bc_dim = first_sample[self.bc_state_key].shape[1]
        if actual_bc_dim != self.expected_bc_state_dim: print(f"Warn: BC_State dim mismatch for {self.dataset_type}. Exp {self.expected_bc_state_dim}, got {actual_bc_dim}")
        self.bc_state_dim = actual_bc_dim

        self.bc_control_key='BC_Control'
        if self.bc_control_key in first_sample and first_sample[self.bc_control_key] is not None and first_sample[self.bc_control_key].size>0:
            self.num_controls = first_sample[self.bc_control_key].shape[1]
        else: self.num_controls=0

        if self.global_norm_stats is None:
            print("Warning: Global normalization stats not provided to UniversalPDEDataset. Normalization will be per-sample which might be inconsistent for validation if not intended.")


    def __len__(self): return len(self.data_list)

    def _normalize_data(self, data, key_prefix, var_idx=None):
        full_key = f"{key_prefix}_{self.state_keys[var_idx]}" if var_idx is not None and key_prefix == "state" else key_prefix
        
        if self.global_norm_stats:
            mean = self.global_norm_stats[f"{full_key}_mean"]
            std = self.global_norm_stats[f"{full_key}_std"]
            if isinstance(mean, np.ndarray) and var_idx is not None and key_prefix != "state": # For BC_State, BC_Control which are multi-dim
                 mean_val = mean[var_idx]
                 std_val = std[var_idx]
            else: # For state (scalar mean/std) or cases where var_idx is not for indexing mean/std array
                 mean_val = mean
                 std_val = std
            return (data - mean_val) / (std_val + 1e-8)
        else: # Fallback to per-sample normalization if no global stats
            return (data - np.mean(data)) / (np.std(data) + 1e-8)

    def __getitem__(self, idx):
        sample=self.data_list[idx]; current_nt=self.effective_nt_for_loader; slist=[]
        # This is for returning the *raw* data from the sample, normalization happens later if global_stats are used
        # Or, if global_stats=None, it normalizes here per sample.
        # For the task, we want to use global_stats, so this part mostly just slices.

        for i_sk, key in enumerate(self.state_keys):
            s_full=sample[key]; s_seq=s_full[:current_nt,...]
            if self.global_norm_stats:
                norm_s_seq = self._normalize_data(s_seq, "state", i_sk)
            else: # Per-sample norm if no global stats
                s_mean=np.mean(s_seq); s_std=np.std(s_seq)+1e-8; norm_s_seq = (s_seq-s_mean)/s_std
            slist.append(torch.tensor(norm_s_seq).float())

        bcs_full=sample[self.bc_state_key]; bcs_seq=bcs_full[:current_nt,:]
        bcs_norm_tensor = torch.zeros_like(torch.tensor(bcs_seq), dtype=torch.float32)
        if bcs_seq.size > 0:
            if self.global_norm_stats:
                for d in range(self.bc_state_dim):
                    bcs_norm_tensor[:,d] = torch.tensor(self._normalize_data(bcs_seq[:,d], "BC_State", d)).float()
            else: # Per-column, per-sample norm
                for d in range(self.bc_state_dim):
                    col=bcs_seq[:,d]; m=np.mean(col); s=np.std(col)+1e-8; bcs_norm_tensor[:,d]=torch.tensor((col-m)/s).float()
        
        bcc_norm_tensor = torch.empty((current_nt,0),dtype=torch.float32)
        if self.num_controls > 0:
            try:
                bcc_full=sample[self.bc_control_key]; bcc_seq=bcc_full[:current_nt,:]
                bcc_norm_tensor_temp = torch.zeros_like(torch.tensor(bcc_seq), dtype=torch.float32)
                if bcc_seq.size > 0:
                    if self.global_norm_stats:
                        for d in range(self.num_controls):
                            bcc_norm_tensor_temp[:,d] = torch.tensor(self._normalize_data(bcc_seq[:,d], "BC_Control", d)).float()
                    else: # Per-column, per-sample norm
                        for d in range(self.num_controls):
                            col=bcc_seq[:,d];m=np.mean(col);s=np.std(col)+1e-8; bcc_norm_tensor_temp[:,d]=torch.tensor((col-m)/s).float()
                bcc_norm_tensor = bcc_norm_tensor_temp
            except KeyError: pass # Handled by empty tensor init

        bc_ctrl_tensor=torch.cat((bcs_norm_tensor,bcc_norm_tensor),dim=-1)
        out_state = slist[0] if self.num_state_vars==1 else slist
        

        return out_state, bc_ctrl_tensor, {} # Empty dict for norm_factors


class MLP(nn.Module): # Moved MLP definition here
    def __init__(self, input_dim, output_dim, hidden_dims, activation=nn.GELU, dropout=0.0):
        super().__init__(); layers = []; current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim)); layers.append(activation())
            if dropout > 0: layers.append(nn.Dropout(dropout))
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, output_dim)); self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)
        
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_norm=True, activation=nn.GELU):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, padding_mode='replicate')
        self.norm = nn.GroupNorm(min(32, out_channels), out_channels) if use_norm else nn.Identity() # Fig 8 uses GroupNorm
        self.act = activation() if activation is not None else nn.Identity()
    def forward(self, x): return self.act(self.norm(self.conv(x)))

class ResConvBlock(nn.Module): # Fig 8 "Residual Block"
    def __init__(self, channels, kernel_size=3, dilation=1, activation=nn.GELU):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, padding_mode='replicate')
        self.act1 = activation()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, padding_mode='replicate')
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.act2 = activation()
    def forward(self, x): # x: [B, C, N_latent]
        residual = x
        out = self.act1(self.conv1(x))
        out = self.conv2(out)
        out = self.act2(self.norm(out + residual)) # Norm and Activation after skip
        return out

class LNS_Encoder(nn.Module):
    def __init__(self, input_channels, initial_width=64, num_downsampling_blocks=3, latent_channels=16, final_latent_nx=8):
        super().__init__()
        self.final_latent_nx = final_latent_nx
        layers = [ConvBlock(input_channels, initial_width, use_norm=False)] # No norm on first layer often
        current_channels = initial_width
        for _ in range(num_downsampling_blocks):
            out_channels = current_channels * 2
            layers.append(ConvBlock(current_channels, current_channels)) # "Res block like" or just conv
            layers.append(ConvBlock(current_channels, out_channels, stride=2)) # Downsample
            current_channels = out_channels
        layers.append(nn.Conv1d(current_channels, latent_channels, kernel_size=1)) # Bottleneck projection
        layers.append(nn.AdaptiveAvgPool1d(final_latent_nx)) # Ensure final spatial dim
        self.encoder_net = nn.Sequential(*layers)
    def forward(self, x): # x: [B, input_channels, Nx_full]
        return self.encoder_net(x) # Output: [B, latent_channels, final_latent_nx]

class LNS_Decoder(nn.Module):
    def __init__(self, latent_channels, output_channels, initial_width_encoder=64, num_upsampling_blocks=3, final_latent_nx=8, target_nx_full=128):
        super().__init__()
        self.target_nx_full = target_nx_full
        layers = []
        
        # Calculate channel progression in reverse
        # Example: initial_width=64, blocks=3. Encoder channels: C_in->64->128->256 (output of last downsample block). Then latent_channels.
        # Decoder: Latent -> 256 -> 128 -> 64 -> C_out
        current_channels = initial_width_encoder * (2**(num_upsampling_blocks-1)) # Channel dim before bottleneck in encoder
        
        layers.append(nn.Upsample(scale_factor=(target_nx_full // final_latent_nx) / (2**num_upsampling_blocks))) # Initial upsample to match first ConvT input size
        layers.append(nn.Conv1d(latent_channels, current_channels, kernel_size=1)) # Project to this width
        layers.append(nn.GELU())

        for i in range(num_upsampling_blocks):
            out_channels = initial_width_encoder * (2**(num_upsampling_blocks-1-i-1)) if i < num_upsampling_blocks-1 else initial_width_encoder
            layers.append(ConvBlock(current_channels, current_channels, use_norm=True)) # "Res block like"
            layers.append(nn.ConvTranspose1d(current_channels, out_channels, kernel_size=4, stride=2, padding=1)) # Upsample by 2
            layers.append(nn.GroupNorm(min(32, out_channels), out_channels))
            layers.append(nn.GELU())
            current_channels = out_channels
        
        layers.append(nn.Conv1d(current_channels, output_channels, kernel_size=3, padding=1)) # Final projection
        self.decoder_net = nn.Sequential(*layers)

    def forward(self, z): # z: [B, latent_channels, latent_Nx]
        decoded = self.decoder_net(z)
        # Ensure final output matches target_nx_full, e.g. via adaptive pooling or final interpolation if ConvT output size is tricky
        if decoded.shape[-1] != self.target_nx_full:
            decoded = F.interpolate(decoded, size=self.target_nx_full, mode='linear', align_corners=False)
        return decoded # Output: [B, output_channels, Nx_full]

class LNS_Autoencoder(nn.Module):
    def __init__(self, num_state_vars, target_nx_full, ae_initial_width=64, ae_downsample_blocks=3, ae_latent_channels=16, final_latent_nx=8):
        super().__init__()
        self.encoder = LNS_Encoder(num_state_vars, ae_initial_width, ae_downsample_blocks, ae_latent_channels, final_latent_nx)
        self.decoder = LNS_Decoder(ae_latent_channels, num_state_vars, ae_initial_width, ae_downsample_blocks, final_latent_nx, target_nx_full)
    def forward(self, u_t): # u_t: [B, num_state_vars, Nx_full]
        z_t = self.encoder(u_t)
        u_t_reconstructed = self.decoder(z_t)
        return u_t_reconstructed, z_t


class LatentStepperNet(nn.Module): # This is the "Propagator" in LNS, adapted for DeepONet-like inputs
    def __init__(self, latent_dim_c, latent_dim_x, # Latent channels and spatial size
                 bc_ctrl_input_dim, # Total dimension of concatenated BC_State and BC_Control from data
                 branch_hidden_dims=[128, 128],
                 trunk_hidden_dims=[64, 64],
                 combined_output_p=64): # 'p' from DeepONet paper, number of basis for output
        super().__init__()
        self.latent_dim_c = latent_dim_c
        self.latent_dim_x = latent_dim_x
        
        # Branch net processes the flattened latent state z_t
        self.branch_net = MLP(latent_dim_c * latent_dim_x, combined_output_p, branch_hidden_dims)
        
        # Trunk net processes the BC/Control signals
        self.trunk_net = MLP(bc_ctrl_input_dim, combined_output_p, trunk_hidden_dims)
        
        # No explicit bias b_0 here, can be learned by MLPs if needed
        self.output_projection = nn.Linear(combined_output_p, latent_dim_c * latent_dim_x) # Predicts next flattened latent state

    def forward(self, z_t, bc_ctrl_next_t):
        # z_t: [B, latent_C, latent_Nx]
        # bc_ctrl_next_t: [B, total_bc_ctrl_dim]
        
        B = z_t.shape[0]
        z_t_flat = z_t.view(B, -1) # Flatten latent state: [B, latent_C * latent_Nx]
        
        branch_out = self.branch_net(z_t_flat) # [B, p]
        trunk_out = self.trunk_net(bc_ctrl_next_t) # [B, p]
        
        combined_latent_features = branch_out + trunk_out # [B, p] (simple sum)
        
        z_next_flat_pred = self.output_projection(combined_latent_features) # [B, latent_C * latent_Nx]
        z_next_pred = z_next_flat_pred.view(B, self.latent_dim_c, self.latent_dim_x) # [B, latent_C, latent_Nx]
        
        return z_next_pred


# train_lns_autoencoder 
def train_lns_autoencoder(autoencoder, data_loader, train_nt_for_model,
                          lr=3e-5, num_epochs=100, device='cuda',
                          checkpoint_path='lns_ae_ckpt.pt', clip_grad_norm=1.0):
    autoencoder.to(device); optimizer = optim.AdamW(autoencoder.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.5,patience=5,verbose=True) # Added patience
    mse_loss = nn.MSELoss(); start_epoch=0; best_loss=float('inf')
    if os.path.exists(checkpoint_path):
        print(f"Loading LNS AE ckpt from {checkpoint_path}...");
        try:
            ckpt=torch.load(checkpoint_path,map_location=device); autoencoder.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt: optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch=ckpt.get('epoch',0)+1; best_loss=ckpt.get('loss',float('inf'))
            print(f"Resuming LNS AE training from epoch {start_epoch}")
        except Exception as e: print(f"Error loading LNS AE ckpt: {e}. Fresh start.")

    for epoch in range(start_epoch, num_epochs):
        autoencoder.train(); epoch_loss_val=0.0; num_batches=0; batch_start_time=time.time()
        for i,(state_data_loaded, _, _) in enumerate(data_loader):
            if isinstance(state_data_loaded,list): u_t_batch=torch.stack(state_data_loaded,dim=-1).to(device)
            else: u_t_batch=state_data_loaded.unsqueeze(-1).to(device)
            B,nt_actual,nx_actual,num_vars_actual = u_t_batch.shape
            if nt_actual!=train_nt_for_model: raise ValueError(f"Data nt {nt_actual} != train_nt {train_nt_for_model} for AE")
            u_snapshots = u_t_batch.permute(0,1,3,2).reshape(B*nt_actual, num_vars_actual, nx_actual)
            optimizer.zero_grad(); u_reconstructed, _ = autoencoder(u_snapshots)
            loss = mse_loss(u_reconstructed, u_snapshots)
            epoch_loss_val+=loss.item(); num_batches+=1; loss.backward()
            if clip_grad_norm>0: torch.nn.utils.clip_grad_norm_(autoencoder.parameters(),max_norm=clip_grad_norm)
            optimizer.step()
            if (i+1)%100==0:
                elapsed=time.time()-batch_start_time
                print(f" LNS-AE Ep {epoch+1} B {i+1}/{len(data_loader)}, Loss {loss.item():.4e}, Time/100B {elapsed:.2f}s")
                batch_start_time=time.time()
        avg_epoch_loss = epoch_loss_val/max(num_batches,1)
        print(f"LNS-AE Epoch {epoch+1}/{num_epochs} Avg Loss: {avg_epoch_loss:.6f}")
        scheduler.step(avg_epoch_loss)
        if avg_epoch_loss<best_loss:
            best_loss=avg_epoch_loss; print(f"Saving LNS-AE ckpt loss {best_loss:.6f}")
            torch.save({'epoch':epoch,'model_state_dict':autoencoder.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'loss':best_loss},checkpoint_path) # Removed dataset_type for simplicity
    print("LNS Autoencoder Training finished.")
    if os.path.exists(checkpoint_path): print(f"Loading best LNS-AE model"); ckpt=torch.load(checkpoint_path,map_location=device); autoencoder.load_state_dict(ckpt['model_state_dict'])
    return autoencoder


def train_latent_don_stepper(latent_stepper, encoder, # Pass trained frozen encoder
                             data_loader, train_nt_for_model,
                             lr=5e-4, num_epochs=150, device='cuda',
                             checkpoint_path='latent_don_stepper_ckpt.pt', clip_grad_norm=1.0,
                             train_rollout_steps=1):
    latent_stepper.to(device); encoder.to(device); encoder.eval()
    optimizer = optim.AdamW(latent_stepper.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs*len(data_loader))
    mse_loss = nn.MSELoss()

    start_epoch=0; best_loss=float('inf')
    if os.path.exists(checkpoint_path):
        print(f"Loading LatentDON Stepper ckpt from {checkpoint_path}...");
        try:
            ckpt=torch.load(checkpoint_path,map_location=device); latent_stepper.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt: optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt: scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch=ckpt.get('epoch',0)+1; best_loss=ckpt.get('loss',float('inf'))
            print(f"Resuming LatentDON Stepper training from epoch {start_epoch}")
        except Exception as e: print(f"Error loading LatentDON Stepper ckpt: {e}. Fresh start.")

    for epoch in range(start_epoch, num_epochs):
        latent_stepper.train(); epoch_loss_val=0.0; num_batches=0; batch_start_time=time.time()
        for i,(state_data_loaded,BC_Ctrl_tensor_loaded,_) in enumerate(data_loader):
            if isinstance(state_data_loaded,list): state_seq_true_train=torch.stack(state_data_loaded,dim=-1).to(device)
            else: state_seq_true_train=state_data_loaded.unsqueeze(-1).to(device)
            BC_Ctrl_seq_train=BC_Ctrl_tensor_loaded.to(device)
            B,nt_actual,nx_actual,num_vars_actual = state_seq_true_train.shape
            if nt_actual!=train_nt_for_model: raise ValueError("Data nt != train_nt for Stepper")

            optimizer.zero_grad(); current_batch_loss_val=0.0 # Renamed

            u_snapshots_all = state_seq_true_train.permute(0,1,3,2).reshape(B*nt_actual, num_vars_actual, nx_actual)
            with torch.no_grad(): z_all_encoded = encoder(u_snapshots_all)
            latent_C, latent_Nx_enc = z_all_encoded.shape[-2:]
            z_seq_true_train = z_all_encoded.view(B, nt_actual, latent_C, latent_Nx_enc)

            current_rollout = train_rollout_steps # Use fixed rollout for now, can be made random
            if train_nt_for_model - current_rollout <=0 : current_rollout = 1
            
            rollout_start_idx = random.randint(0, train_nt_for_model - 1 - current_rollout)
            z_current_rollout = z_seq_true_train[:, rollout_start_idx, :, :]
            
            rollout_loss = 0.0
            for k_roll in range(current_rollout):
                t_idx_for_bc = rollout_start_idx + k_roll + 1 # BC for predicting state at t+1
                bc_ctrl_input_roll = BC_Ctrl_seq_train[:, t_idx_for_bc, :]
                z_next_pred_rollout = latent_stepper(z_current_rollout, bc_ctrl_input_roll)
                z_next_target_rollout = z_seq_true_train[:, rollout_start_idx + k_roll + 1, :, :]
                rollout_loss += mse_loss(z_next_pred_rollout, z_next_target_rollout)
                z_current_rollout = z_next_pred_rollout # Autoregressive
            current_batch_loss_val = rollout_loss / current_rollout
            
            epoch_loss_val+=current_batch_loss_val.item(); num_batches+=1
            current_batch_loss_val.backward()
            if clip_grad_norm>0: torch.nn.utils.clip_grad_norm_(latent_stepper.parameters(),max_norm=clip_grad_norm)
            optimizer.step(); scheduler.step()

            if (i+1)%50==0:
                elapsed=time.time()-batch_start_time
                print(f" LatentDON Ep {epoch+1} B {i+1}/{len(data_loader)}, Loss {current_batch_loss_val.item():.3e}, LR {optimizer.param_groups[0]['lr']:.3e}, Time/50B {elapsed:.2f}s")
                batch_start_time=time.time()
        avg_epoch_loss = epoch_loss_val/max(num_batches,1)
        print(f"LatentDON Stepper Epoch {epoch+1}/{num_epochs} Avg Loss: {avg_epoch_loss:.6f}")
        if avg_epoch_loss<best_loss:
            best_loss=avg_epoch_loss; print(f"Saving LatentDON Stepper ckpt loss {best_loss:.6f}")
            torch.save({'epoch':epoch,'model_state_dict':latent_stepper.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'scheduler_state_dict':scheduler.state_dict(),'loss':best_loss},checkpoint_path)
    print("LatentDON Stepper Training finished.")
    if os.path.exists(checkpoint_path): print(f"Loading best LatentDON Stepper model"); ckpt=torch.load(checkpoint_path,map_location=device); latent_stepper.load_state_dict(ckpt['model_state_dict'])
    return latent_stepper

def validate_latent_don_stepper(encoder, decoder, latent_stepper,
                                data_loader, dataset_type,
                                train_nt_for_model_training: int, T_value_for_model_training: float,
                                full_T_in_datafile: float, full_nt_in_datafile: int,
                                dataset_params_for_plot: dict, global_norm_stats_val: dict, device='cuda',
                                save_fig_path_prefix='latent_don_result'): # Added global_norm_stats_val
    encoder.eval(); decoder.eval(); latent_stepper.eval()

    if dataset_type in ['advection','burgers']: state_keys_val=['U']; num_state_vars_val=1
    elif dataset_type == 'euler': state_keys_val=['rho','u']; num_state_vars_val=2
    elif dataset_type == 'darcy': state_keys_val=['P']; num_state_vars_val=1
    else: raise ValueError(f"Unknown type '{dataset_type}' in validation")

    test_horizons_T_values = [T_value_for_model_training]
    if full_T_in_datafile > T_value_for_model_training + 1e-5 :
        test_horizons_T_values.append(T_value_for_model_training + 0.5 * (full_T_in_datafile - T_value_for_model_training))
        test_horizons_T_values.append(full_T_in_datafile)
    test_horizons_T_values = sorted(list(set(h for h in test_horizons_T_values if h <= full_T_in_datafile + 1e-6)))
    print(f"LatentDON Stepper Validation for T_horizons: {test_horizons_T_values}")

    results_primary_horizon = {key: {'mse':[],'rmse':[],'relative_error':[],'max_error':[]} for key in state_keys_val}
    overall_rel_err_primary_horizon = []
    
    nx_plot = dataset_params_for_plot.get('nx',128)

    with torch.no_grad():
        try: state_data_full_loaded, BC_Ctrl_tensor_full_loaded, _ = next(iter(data_loader)) # Norm factors from loader ignored, use global
        except StopIteration: print("Val data_loader empty. Skipping validation."); return

        if isinstance(state_data_full_loaded,list): state_seq_true_norm_full_sample = torch.stack(state_data_full_loaded,dim=-1)[0].to(device)
        else: state_seq_true_norm_full_sample = state_data_full_loaded.unsqueeze(-1)[0].to(device)
        BC_Ctrl_seq_norm_full_sample = BC_Ctrl_tensor_full_loaded[0].to(device)
        
        u_t0_physical_norm = state_seq_true_norm_full_sample[0:1,:,:].permute(0,2,1) # [1, C_state, Nx]
        z_current_latent = encoder(u_t0_physical_norm) # Initial latent state

        for T_horizon_current in test_horizons_T_values:
            nt_for_rollout = int((T_horizon_current/full_T_in_datafile)*(full_nt_in_datafile-1))+1
            nt_for_rollout = min(nt_for_rollout, full_nt_in_datafile)
            print(f"\n  Rollout for T_horizon = {T_horizon_current:.2f} (nt = {nt_for_rollout})")

            u_pred_seq_physical_norm_horizon = torch.zeros(nt_for_rollout, nx_plot, num_state_vars_val, device=device)
            u_pred_seq_physical_norm_horizon[0,:,:] = state_seq_true_norm_full_sample[0,:,:]
            z_rollout_current = z_current_latent.clone()
            BC_Ctrl_for_rollout = BC_Ctrl_seq_norm_full_sample[:nt_for_rollout,:]

            for t_step in range(nt_for_rollout - 1):
                bc_ctrl_input_roll = BC_Ctrl_for_rollout[t_step+1:t_step+2,:]
                z_next_pred_latent = latent_stepper(z_rollout_current, bc_ctrl_input_roll)
                u_next_pred_physical_norm = decoder(z_next_pred_latent).permute(0,2,1).squeeze(0)
                u_pred_seq_physical_norm_horizon[t_step+1,:,:] = u_next_pred_physical_norm
                z_rollout_current = z_next_pred_latent
            
            # Denormalize & Metrics
            U_pred_denorm_h={}; U_gt_denorm_h={}
            pred_list_h=[]; gt_list_h=[]
            state_true_norm_sliced_h = state_seq_true_norm_full_sample[:nt_for_rollout,:,:]
            pred_np_h = u_pred_seq_physical_norm_horizon.cpu().numpy(); gt_np_h = state_true_norm_sliced_h.cpu().numpy()

            for k_idx, key_val in enumerate(state_keys_val):
                # Use global_norm_stats_val for denormalization
                mean_k = global_norm_stats_val[f'state_{key_val}_mean']
                std_k = global_norm_stats_val[f'state_{key_val}_std']
                pred_norm_v=pred_np_h[:,:,k_idx]; gt_norm_v=gt_np_h[:,:,k_idx]
                pred_denorm_v=pred_norm_v*std_k+mean_k; gt_denorm_v=gt_norm_v*std_k+mean_k
                U_pred_denorm_h[key_val]=pred_denorm_v; U_gt_denorm_h[key_val]=gt_denorm_v
                pred_list_h.append(pred_denorm_v.flatten()); gt_list_h.append(gt_denorm_v.flatten())
                mse_k_h=np.mean((pred_denorm_v-gt_denorm_v)**2)
                rel_err_k_h=np.linalg.norm(pred_denorm_v-gt_denorm_v,'fro')/(np.linalg.norm(gt_denorm_v,'fro')+1e-10)
                print(f"    Metrics '{key_val}' @ T={T_horizon_current:.1f}: MSE={mse_k_h:.3e}, RelErr={rel_err_k_h:.3e}")
                if abs(T_horizon_current-T_value_for_model_training)<1e-6:
                    results_primary_horizon[key_val]['mse'].append(mse_k_h)
                    results_primary_horizon[key_val]['relative_error'].append(rel_err_k_h)
            
            pred_vec_h=np.concatenate(pred_list_h); gt_vec_h=np.concatenate(gt_list_h)
            overall_rel_h=np.linalg.norm(pred_vec_h-gt_vec_h)/(np.linalg.norm(gt_vec_h)+1e-10)
            print(f"    Overall RelErr @ T={T_horizon_current:.1f}: {overall_rel_h:.3e}")
            if abs(T_horizon_current-T_value_for_model_training)<1e-6: overall_rel_err_primary_horizon.append(overall_rel_h)

            # Visualization
            fig,axs=plt.subplots(num_state_vars_val,3,figsize=(18,5*num_state_vars_val),squeeze=False)
            fig_L_plot=dataset_params_for_plot.get('L',1.0); fig_nx_plot=nx_plot; fig_ny_plot=dataset_params_for_plot.get('ny',1)
            for k_idx, key_val in enumerate(state_keys_val):
                gt_p=U_gt_denorm_h[key_val]; pred_p=U_pred_denorm_h[key_val]; diff_p=np.abs(pred_p-gt_p)
                max_err_p=np.max(diff_p) if diff_p.size>0 else 0
                vmin_p=min(np.min(gt_p),np.min(pred_p)) if gt_p.size>0 else 0; vmax_p=max(np.max(gt_p),np.max(pred_p)) if gt_p.size>0 else 1
                is_1d_p=(fig_ny_plot==1); plot_ext=[0,fig_L_plot,0,T_horizon_current]
                if is_1d_p:
                    im0=axs[k_idx,0].imshow(gt_p,aspect='auto',origin='lower',vmin=vmin_p,vmax=vmax_p,extent=plot_ext,cmap='viridis'); axs[k_idx,0].set_title(f"Truth ({key_val})")
                    im1=axs[k_idx,1].imshow(pred_p,aspect='auto',origin='lower',vmin=vmin_p,vmax=vmax_p,extent=plot_ext,cmap='viridis'); axs[k_idx,1].set_title(f"LatentDON Pred ({key_val})")
                    im2=axs[k_idx,2].imshow(diff_p,aspect='auto',origin='lower',extent=plot_ext,cmap='magma'); axs[k_idx,2].set_title(f"Abs Error (Max:{max_err_p:.2e})")
                    for j_p in range(3): axs[k_idx,j_p].set_xlabel("x"); axs[k_idx,0].set_ylabel("t")
                    plt.colorbar(im0,ax=axs[k_idx,0]); plt.colorbar(im1,ax=axs[k_idx,1]); plt.colorbar(im2,ax=axs[k_idx,2])
                else: axs[k_idx,0].text(0.5,0.5,"2D Plot Placeholder",ha='center')
            fig.suptitle(f"LatentDON Stepper Validation ({dataset_type.capitalize()}) @ T={T_horizon_current:.1f}")
            fig.tight_layout(rect=[0,0.03,1,0.95]); curr_fig_path=save_fig_path_prefix+f"_T{str(T_horizon_current).replace('.','p')}.png"
            plt.savefig(curr_fig_path); print(f"  Saved LatentDON validation plot to {curr_fig_path}"); plt.show()
    
    print(f"\n--- LatentDON Stepper Validation Summary (Metrics for T={T_value_for_model_training:.1f}) ---")
    for key_val in state_keys_val:
        if results_primary_horizon[key_val]['mse']:
            avg_mse=np.mean(results_primary_horizon[key_val]['mse']); avg_rmse=np.sqrt(avg_mse)
            avg_rel=np.mean(results_primary_horizon[key_val]['relative_error'])
            print(f"  Var '{key_val}': Avg MSE={avg_mse:.3e}, RMSE={avg_rmse:.3e}, RelErr={avg_rel:.3e}")
    if overall_rel_err_primary_horizon: print(f"  Overall RelErr @ T={T_value_for_model_training:.1f}: {np.mean(overall_rel_err_primary_horizon):.3e}")
    print("------------------------")

# main script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate PDE datasets with complex BCs.")
    parser.add_argument('--datatype', type=str, required=True, choices=['advection', 'euler', 'burgers', 'darcy'], help='Type of dataset to generate.')
    args = parser.parse_args()

    DATASET_TYPE = args.datatype # 'advection', 'euler', 'burgers', 'darcy'    
    MODEL_TYPE = 'LatentDON_Stepper' # To distinguish from a direct LatentDeepONet

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Running {MODEL_TYPE} Baseline for {DATASET_TYPE} on {device} ---")

    FULL_T_IN_DATAFILE = 2.0; FULL_NT_IN_DATAFILE = 600
    TRAIN_T_TARGET = 1.0
    TRAIN_NT_FOR_MODEL = int((TRAIN_T_TARGET/FULL_T_IN_DATAFILE)*(FULL_NT_IN_DATAFILE-1))+1
    print(f"Datafile T={FULL_T_IN_DATAFILE}, nt={FULL_NT_IN_DATAFILE}")
    print(f"Training T={TRAIN_T_TARGET}, nt={TRAIN_NT_FOR_MODEL}")


    dataset_params_for_plot = {}
    if DATASET_TYPE == 'advection':
        dataset_path = "./datasets_full/advection_data_10000s_128nx_600nt.pkl"
        main_state_keys=['U']; main_num_state_vars=1
        dataset_params_for_plot={'nx':128,'ny':1,'L':1.0,'T':FULL_T_IN_DATAFILE}
    elif DATASET_TYPE == 'euler':
        dataset_path = "./datasets_full/euler_data_10000s_128nx_600nt.pkl"
        main_state_keys=['rho','u']; main_num_state_vars=2
        dataset_params_for_plot={'nx':128,'ny':1,'L':1.0,'T':FULL_T_IN_DATAFILE}
    elif DATASET_TYPE == 'burgers':
        dataset_path = "./datasets_full/burgers_data_10000s_128nx_600nt.pkl"
        main_state_keys=['U']; main_num_state_vars=1
        dataset_params_for_plot={'nx':128,'ny':1,'L':1.0,'T':FULL_T_IN_DATAFILE}
    elif DATASET_TYPE == 'darcy':
        dataset_path = "./datasets_full/darcy_data_10000s_128nx_600nt.pkl"
        main_state_keys=['P']; main_num_state_vars=1
        dataset_params_for_plot={'nx':128,'ny':1,'L':1.0,'T':FULL_T_IN_DATAFILE} # Assuming 1D spatial from generator
    else: raise ValueError(f"Unknown dataset type: {DATASET_TYPE}")

    # Model Hyperparameters (Inspired by ae2.pdf Fig8 details & tables)
    AE_INITIAL_WIDTH = 64; AE_DOWNSAMPLE_BLOCKS = 3; AE_LATENT_CHANNELS = 16
    AE_FINAL_LATENT_NX = dataset_params_for_plot.get('nx',128) // (2**AE_DOWNSAMPLE_BLOCKS) # e.g. 128/8 = 16
    
    LATENT_STEPPER_BRANCH_HIDDEN = [128, 128] # For z_t
    LATENT_STEPPER_TRUNK_HIDDEN = [64, 64]   # For BC_Ctrl
    LATENT_STEPPER_COMBINED_P = 128         # Output dim 'p' for branch/trunk before final projection

    AE_LR = 3e-4; AE_EPOCHS = 100 # Adjust AE training
    PROP_LR = 1e-3; PROP_EPOCHS = 150 # Adjust Stepper training
    PROP_TRAIN_ROLLOUT_STEPS = 1 # As per user request for one-step learning

    BATCH_SIZE = 32; CLIP_GRAD_NORM = 1.0
    
    # ... Add other dataset types if needed

    print(f"Loading dataset: {dataset_path}")
    try:
        with open(dataset_path,'rb') as f: data_list_all=pickle.load(f)
        print(f"Loaded {len(data_list_all)} samples.")
        if data_list_all and 'params' in data_list_all[0]:
            file_params = data_list_all[0]['params']
            dataset_params_for_plot['L'] = file_params.get('L', dataset_params_for_plot.get('L',1.0))
            dataset_params_for_plot['nx'] = file_params.get('nx', dataset_params_for_plot.get('nx',128))
            actual_file_nt = data_list_all[0][main_state_keys[0]].shape[0]
            if actual_file_nt != FULL_NT_IN_DATAFILE:
                print(f"WARN: Config FULL_NT ({FULL_NT_IN_DATAFILE}) vs file nt ({actual_file_nt}). Using file nt."); FULL_NT_IN_DATAFILE = actual_file_nt
                TRAIN_NT_FOR_MODEL = int((TRAIN_T_TARGET/FULL_T_IN_DATAFILE)*(FULL_NT_IN_DATAFILE-1))+1
                print(f"Recalculated TRAIN_NT_FOR_MODEL = {TRAIN_NT_FOR_MODEL}")
        AE_FINAL_LATENT_NX = dataset_params_for_plot.get('nx',128) // (2**AE_DOWNSAMPLE_BLOCKS)

    except FileNotFoundError: print(f"Error: File not found {dataset_path}"); exit()
    if not data_list_all: print("No data. Exiting."); exit()

    random.shuffle(data_list_all); n_total=len(data_list_all); n_train=int(0.8*n_total)
    train_data_list_split=data_list_all[:n_train]; val_data_list_split=data_list_all[n_train:]

    # --- Compute Global Normalization Statistics from Training Data (T=0 to 1.0 portion) ---
    print("Computing global normalization statistics from training data (T=0 to 1.0)...")
    global_norm_stats = {f"state_{key}_mean": [] for key in main_state_keys}
    global_norm_stats.update({f"state_{key}_std": [] for key in main_state_keys})
    temp_ds_for_stats = UniversalPDEDataset(train_data_list_split, DATASET_TYPE, train_nt_limit=TRAIN_NT_FOR_MODEL, global_norm_stats=None) # Use per-sample norm here to collect raw data
    
    # Collect all data points from the training set (T=0 to 1 part)
    all_state_data_for_stats = {key: [] for key in main_state_keys}
    all_bc_state_data_for_stats = []
    all_bc_control_data_for_stats = []

    for i_sample in range(len(temp_ds_for_stats)):
        sample_data_raw = temp_ds_for_stats.data_list[i_sample] # Get raw sample
        
        for i_sk_stat, sk_stat in enumerate(main_state_keys):
            all_state_data_for_stats[sk_stat].append(sample_data_raw[sk_stat][:TRAIN_NT_FOR_MODEL,...])

        all_bc_state_data_for_stats.append(sample_data_raw[temp_ds_for_stats.bc_state_key][:TRAIN_NT_FOR_MODEL,...])
        if temp_ds_for_stats.num_controls > 0 and temp_ds_for_stats.bc_control_key in sample_data_raw:
            all_bc_control_data_for_stats.append(sample_data_raw[temp_ds_for_stats.bc_control_key][:TRAIN_NT_FOR_MODEL,...])

    for i_sk_stat, sk_stat in enumerate(main_state_keys):
        concatenated_state = np.concatenate(all_state_data_for_stats[sk_stat], axis=0)
        global_norm_stats[f"state_{sk_stat}_mean"] = np.mean(concatenated_state)
        global_norm_stats[f"state_{sk_stat}_std"] = np.std(concatenated_state) + 1e-8
        print(f"  Global norm for {sk_stat}: mean={global_norm_stats[f'state_{sk_stat}_mean']:.4f}, std={global_norm_stats[f'state_{sk_stat}_std']:.4f}")

    concatenated_bc_state = np.concatenate(all_bc_state_data_for_stats, axis=0)
    global_norm_stats["BC_State_mean"] = np.mean(concatenated_bc_state, axis=0)
    global_norm_stats["BC_State_std"] = np.std(concatenated_bc_state, axis=0) + 1e-8
    
    if temp_ds_for_stats.num_controls > 0 and all_bc_control_data_for_stats:
        concatenated_bc_control = np.concatenate(all_bc_control_data_for_stats, axis=0)
        global_norm_stats["BC_Control_mean"] = np.mean(concatenated_bc_control, axis=0)
        global_norm_stats["BC_Control_std"] = np.std(concatenated_bc_control, axis=0) + 1e-8
    else: # Handle case with no controls
        global_norm_stats["BC_Control_mean"] = np.array([]) 
        global_norm_stats["BC_Control_std"] = np.array([])
    print("Global normalization stats computed.")


    train_dataset = UniversalPDEDataset(train_data_list_split, dataset_type=DATASET_TYPE, global_norm_stats=global_norm_stats, train_nt_limit=TRAIN_NT_FOR_MODEL)
    val_dataset = UniversalPDEDataset(val_data_list_split, dataset_type=DATASET_TYPE, global_norm_stats=global_norm_stats, train_nt_limit=None)

    num_workers=1
    train_loader_ae = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=num_workers,pin_memory=True,drop_last=True)
    train_loader_prop = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=num_workers,pin_memory=True,drop_last=True)
    val_loader = DataLoader(val_dataset,batch_size=1,shuffle=False,num_workers=num_workers)

    actual_bc_ctrl_dim_from_dataset = train_dataset.bc_state_dim + train_dataset.num_controls
    current_nx_from_dataset = train_dataset.nx

    # --- Stage 1: Train Autoencoder ---
    print(f"\nInitializing LNS Autoencoder: input_channels={main_num_state_vars}, target_nx_full={current_nx_from_dataset}, latent_channels={AE_LATENT_CHANNELS}, final_latent_nx={AE_FINAL_LATENT_NX}")
    autoencoder = LNS_Autoencoder(num_state_vars=main_num_state_vars, target_nx_full=current_nx_from_dataset, ae_initial_width=AE_INITIAL_WIDTH, ae_downsample_blocks=AE_DOWNSAMPLE_BLOCKS, ae_latent_channels=AE_LATENT_CHANNELS, final_latent_nx=AE_FINAL_LATENT_NX).to(device)
    ae_checkpoint_path = f"./New_ckpt_2/checkpoints_{DATASET_TYPE}_{MODEL_TYPE}/lns_ae_stage1.pt"; os.makedirs(os.path.dirname(ae_checkpoint_path), exist_ok=True)
    print(f"Starting LNS Autoencoder training...")
    
    autoencoder = train_lns_autoencoder(autoencoder, train_loader_ae,  TRAIN_NT_FOR_MODEL, lr=AE_LR, num_epochs=AE_EPOCHS, device=device, checkpoint_path=ae_checkpoint_path, clip_grad_norm=CLIP_GRAD_NORM)

    # --- Stage 2: Train Latent Stepper (Propagator) ---
    print(f"\nInitializing LatentDON Stepper: latent_C={AE_LATENT_CHANNELS}, latent_Nx={AE_FINAL_LATENT_NX}, bc_ctrl_input_dim={actual_bc_ctrl_dim_from_dataset}")
    latent_don_stepper = LatentStepperNet(
        latent_dim_c=AE_LATENT_CHANNELS, latent_dim_x=AE_FINAL_LATENT_NX,
        bc_ctrl_input_dim=actual_bc_ctrl_dim_from_dataset,
        branch_hidden_dims=LATENT_STEPPER_BRANCH_HIDDEN,
        trunk_hidden_dims=LATENT_STEPPER_TRUNK_HIDDEN,
        combined_output_p=LATENT_STEPPER_COMBINED_P
    ).to(device)
    prop_checkpoint_path = f"./New_ckpt_2/checkpoints_{DATASET_TYPE}_{MODEL_TYPE}/latent_don_stepper_stage2.pt"
    print(f"Starting LatentDON Stepper training...")
    latent_don_stepper = train_latent_don_stepper(latent_don_stepper, autoencoder.encoder, train_loader_prop, TRAIN_NT_FOR_MODEL, lr=PROP_LR, num_epochs=PROP_EPOCHS, device=device, checkpoint_path=prop_checkpoint_path, clip_grad_norm=CLIP_GRAD_NORM, train_rollout_steps=PROP_TRAIN_ROLLOUT_STEPS)

    # --- Validation ---
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{DATASET_TYPE}_{MODEL_TYPE}_LB128_LT64_LC128"
    checkpoint_dir = f"./New_ckpt_2/_checkpoints_{DATASET_TYPE}"
    results_dir = f"./result_all_2/results_{DATASET_TYPE}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'model_{run_name}.pt')
    save_fig_path = os.path.join(results_dir, f'result_{run_name}.png')
    basis_dir = os.path.join(checkpoint_dir, 'pod_bases')
    os.makedirs(basis_dir, exist_ok=True)

    if val_data_list_split:
        print(f"\nStarting validation for {MODEL_TYPE} on {DATASET_TYPE}...")
        validate_latent_don_stepper(
            autoencoder.encoder, autoencoder.decoder, latent_don_stepper,
            val_loader, dataset_type=DATASET_TYPE,
            train_nt_for_model_training=TRAIN_NT_FOR_MODEL,
            T_value_for_model_training=TRAIN_T_TARGET,
            full_T_in_datafile=FULL_T_IN_DATAFILE,
            full_nt_in_datafile=FULL_NT_IN_DATAFILE,
            dataset_params_for_plot=dataset_params_for_plot,
            global_norm_stats_val=global_norm_stats, device=device,
            save_fig_path_prefix=save_fig_path # Pass the computed training stats for validation denorm
        )
    else: print("\nNo validation data. Skipping validation.")

    print("="*60); print(f"Run finished: {run_name}");
    print(f"AE checkpoint: {ae_checkpoint_path}"); print(f"Stepper checkpoint: {prop_checkpoint_path}");
    if val_data_list_split: print(f"Validation figures saved with prefix: {save_fig_path}"); print("="*60)
