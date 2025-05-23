# LNS_AE
# =============================================================================
# latent deepOnet (Adapted for Task2) ref:https://github.com/katiana22/latent-deeponet
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
import argparse # Added argparse
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve 


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
        
        if self.dataset_type in ['heat_delayed_feedback', 
                                   'reaction_diffusion_neumann_feedback', 
                                   'heat_nonlinear_feedback_gain', 
                                   'convdiff']: # Added convdiff
            self.nt_from_sample_file=first_sample['U'].shape[0]; self.nx_from_sample_file=first_sample['U'].shape[1]
            self.state_keys=['U']; self.num_state_vars=1; self.expected_bc_state_dim=2
        else: 
            raise ValueError(f"Unknown dataset_type in UniversalPDEDataset: {self.dataset_type}")

        self.effective_nt_for_loader = self.train_nt_limit if self.train_nt_limit is not None else self.nt_from_sample_file
        self.nx=self.nx_from_sample_file; self.ny=self.ny_from_sample_file; self.spatial_dim=self.nx*self.ny

        self.bc_state_key='BC_State'
        if self.bc_state_key not in first_sample: # Check if key exists
            raise KeyError(f"'{self.bc_state_key}' not found in the first sample of dataset type '{self.dataset_type}'!")
            
        actual_bc_dim = first_sample[self.bc_state_key].shape[1]
        if actual_bc_dim != self.expected_bc_state_dim: 
            print(f"Warning: BC_State dimension mismatch for {self.dataset_type}. Exp {self.expected_bc_state_dim}, got {actual_bc_dim}. Using actual: {actual_bc_dim}")
        self.bc_state_dim = actual_bc_dim # Use actual dimension from data

        self.bc_control_key='BC_Control'
        if self.bc_control_key in first_sample and first_sample[self.bc_control_key] is not None and first_sample[self.bc_control_key].size>0:
            self.num_controls = first_sample[self.bc_control_key].shape[1]
        else: self.num_controls=0

        if self.global_norm_stats is None:
            print("Warning: Global normalization stats not provided to UniversalPDEDataset. Normalization will be per-sample.")

    def __len__(self): return len(self.data_list)

    def _normalize_data(self, data, key_prefix, var_idx=None):
        # Construct the full key for accessing stats, handling multi-dim BC/Control
        if key_prefix == "state":
            stat_key_name = f"{key_prefix}_{self.state_keys[var_idx]}"
        elif key_prefix == "BC_State" or key_prefix == "BC_Control": # These are multi-dimensional
            # For BC_State and BC_Control, mean/std are arrays. We normalize column by column.
            # The var_idx here refers to the column index of the BC/Control data.
            mean_arr = self.global_norm_stats.get(f"{key_prefix}_mean")
            std_arr = self.global_norm_stats.get(f"{key_prefix}_std")
            if mean_arr is not None and std_arr is not None and var_idx < len(mean_arr):
                mean_val = mean_arr[var_idx]
                std_val = std_arr[var_idx]
                return (data - mean_val) / (std_val + 1e-8)
            else: # Fallback or error if stats are missing for a specific dimension
                print(f"Warning: Missing global norm stats for {key_prefix} dimension {var_idx}. Using per-sample norm for this column.")
                return (data - np.mean(data)) / (np.std(data) + 1e-8)
        else: # Should not happen
            stat_key_name = key_prefix 

        # For 'state' variables (single mean/std per variable)
        if self.global_norm_stats and stat_key_name in self.global_norm_stats:
            mean = self.global_norm_stats[f"{stat_key_name}_mean"]
            std = self.global_norm_stats[f"{stat_key_name}_std"]
            return (data - mean) / (std + 1e-8)
        else: # Fallback to per-sample normalization
            return (data - np.mean(data)) / (np.std(data) + 1e-8)


    def __getitem__(self, idx):
        sample=self.data_list[idx]; current_nt=self.effective_nt_for_loader; slist=[]
        
        for i_sk, key in enumerate(self.state_keys):
            s_full=sample[key]; s_seq=s_full[:current_nt,...]
            if self.global_norm_stats:
                norm_s_seq = self._normalize_data(s_seq, "state", i_sk)
            else: 
                s_mean=np.mean(s_seq); s_std=np.std(s_seq)+1e-8; norm_s_seq = (s_seq-s_mean)/s_std
            slist.append(torch.tensor(norm_s_seq).float())

        bcs_full=sample[self.bc_state_key]; bcs_seq=bcs_full[:current_nt,:]
        bcs_norm_tensor_list = []
        if bcs_seq.size > 0:
            for d_idx in range(self.bc_state_dim):
                col_data = bcs_seq[:,d_idx]
                if self.global_norm_stats:
                    norm_col = self._normalize_data(col_data, "BC_State", d_idx)
                else:
                    m=np.mean(col_data); s=np.std(col_data)+1e-8; norm_col = (col_data-m)/s
                bcs_norm_tensor_list.append(torch.tensor(norm_col).float())
        bcs_norm_tensor = torch.stack(bcs_norm_tensor_list, dim=1) if bcs_norm_tensor_list else torch.empty((current_nt,0),dtype=torch.float32)
        
        bcc_norm_tensor_list = []
        if self.num_controls > 0:
            try:
                bcc_full=sample[self.bc_control_key]; bcc_seq=bcc_full[:current_nt,:]
                if bcc_seq.size > 0:
                    for d_idx in range(self.num_controls):
                        col_data = bcc_seq[:,d_idx]
                        if self.global_norm_stats:
                            norm_col = self._normalize_data(col_data, "BC_Control", d_idx)
                        else:
                            m=np.mean(col_data);s=np.std(col_data)+1e-8; norm_col = (col_data-m)/s
                        bcc_norm_tensor_list.append(torch.tensor(norm_col).float())
            except KeyError: pass 
        bcc_norm_tensor = torch.stack(bcc_norm_tensor_list, dim=1) if bcc_norm_tensor_list else torch.empty((current_nt,0),dtype=torch.float32)


        bc_ctrl_tensor=torch.cat((bcs_norm_tensor,bcc_norm_tensor),dim=-1)
        out_state = slist[0] if self.num_state_vars==1 else slist
        
        return out_state, bc_ctrl_tensor, {}


class MLP(nn.Module):
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
        self.norm = nn.GroupNorm(min(32, out_channels), out_channels) if use_norm else nn.Identity()
        self.act = activation() if activation is not None else nn.Identity()
    def forward(self, x): return self.act(self.norm(self.conv(x)))

class ResConvBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, activation=nn.GELU):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, padding_mode='replicate')
        self.act1 = activation()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, padding_mode='replicate')
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.act2 = activation()
    def forward(self, x):
        residual = x
        out = self.act1(self.conv1(x))
        out = self.conv2(out)
        out = self.act2(self.norm(out + residual))
        return out

class LNS_Encoder(nn.Module):
    def __init__(self, input_channels, initial_width=64, num_downsampling_blocks=3, latent_channels=16, final_latent_nx=8):
        super().__init__()
        self.final_latent_nx = final_latent_nx
        layers = [ConvBlock(input_channels, initial_width, use_norm=False)]
        current_channels = initial_width
        for _ in range(num_downsampling_blocks):
            out_channels = current_channels * 2
            layers.append(ConvBlock(current_channels, current_channels))
            layers.append(ConvBlock(current_channels, out_channels, stride=2))
            current_channels = out_channels
        layers.append(nn.Conv1d(current_channels, latent_channels, kernel_size=1))
        layers.append(nn.AdaptiveAvgPool1d(final_latent_nx))
        self.encoder_net = nn.Sequential(*layers)
    def forward(self, x):
        return self.encoder_net(x)

class LNS_Decoder(nn.Module):
    def __init__(self, latent_channels, output_channels, initial_width_encoder=64, num_upsampling_blocks=3, final_latent_nx=8, target_nx_full=128):
        super().__init__()
        self.target_nx_full = target_nx_full
        layers = []
        current_channels = initial_width_encoder * (2**(num_upsampling_blocks-1))
        

        initial_upsample_factor = (target_nx_full / (2**num_upsampling_blocks) ) / final_latent_nx
        if initial_upsample_factor > 1.0 + 1e-3: # Only add if significant upsampling is needed here
             layers.append(nn.Upsample(scale_factor=initial_upsample_factor, mode='linear', align_corners=False))
        
        layers.append(nn.Conv1d(latent_channels, current_channels, kernel_size=1))
        layers.append(nn.GELU())

        for i in range(num_upsampling_blocks):
            out_channels = initial_width_encoder * (2**(num_upsampling_blocks-1-i-1)) if i < num_upsampling_blocks-1 else initial_width_encoder
            layers.append(ConvBlock(current_channels, current_channels, use_norm=True))
            layers.append(nn.ConvTranspose1d(current_channels, out_channels, kernel_size=4, stride=2, padding=1))
            layers.append(nn.GroupNorm(min(32, out_channels), out_channels))
            layers.append(nn.GELU())
            current_channels = out_channels
        
        layers.append(nn.Conv1d(current_channels, output_channels, kernel_size=3, padding=1))
        self.decoder_net = nn.Sequential(*layers)

    def forward(self, z):
        decoded = self.decoder_net(z)
        if decoded.shape[-1] != self.target_nx_full:
            decoded = F.interpolate(decoded, size=self.target_nx_full, mode='linear', align_corners=False)
        return decoded

class LNS_Autoencoder(nn.Module):
    def __init__(self, num_state_vars, target_nx_full, ae_initial_width=64, ae_downsample_blocks=3, ae_latent_channels=16, final_latent_nx=8):
        super().__init__()
        self.encoder = LNS_Encoder(num_state_vars, ae_initial_width, ae_downsample_blocks, ae_latent_channels, final_latent_nx)
        self.decoder = LNS_Decoder(ae_latent_channels, num_state_vars, ae_initial_width, ae_downsample_blocks, final_latent_nx, target_nx_full)
    def forward(self, u_t):
        z_t = self.encoder(u_t)
        u_t_reconstructed = self.decoder(z_t)
        return u_t_reconstructed, z_t


class LatentStepperNet(nn.Module):
    def __init__(self, latent_dim_c, latent_dim_x,
                 bc_ctrl_input_dim,
                 branch_hidden_dims=[128, 128],
                 trunk_hidden_dims=[64, 64],
                 combined_output_p=64):
        super().__init__()
        self.latent_dim_c = latent_dim_c
        self.latent_dim_x = latent_dim_x
        self.branch_net = MLP(latent_dim_c * latent_dim_x, combined_output_p, branch_hidden_dims)
        self.trunk_net = MLP(bc_ctrl_input_dim, combined_output_p, trunk_hidden_dims)
        self.output_projection = nn.Linear(combined_output_p, latent_dim_c * latent_dim_x)

    def forward(self, z_t, bc_ctrl_next_t):
        B = z_t.shape[0]
        z_t_flat = z_t.view(B, -1)
        branch_out = self.branch_net(z_t_flat)
        trunk_out = self.trunk_net(bc_ctrl_next_t)
        combined_latent_features = branch_out + trunk_out
        z_next_flat_pred = self.output_projection(combined_latent_features)
        z_next_pred = z_next_flat_pred.view(B, self.latent_dim_c, self.latent_dim_x)
        return z_next_pred


def train_lns_autoencoder(autoencoder, data_loader, train_nt_for_model,
                          lr=3e-5, num_epochs=100, device='cuda',
                          checkpoint_path='lns_ae_ckpt.pt', clip_grad_norm=1.0):
    autoencoder.to(device); optimizer = optim.AdamW(autoencoder.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.5,patience=10,verbose=True) # Increased patience
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
            if (i+1)%100==0: # Print less frequently
                elapsed=time.time()-batch_start_time
                print(f" LNS-AE Ep {epoch+1} B {i+1}/{len(data_loader)}, Loss {loss.item():.4e}, Time/100B {elapsed:.2f}s")
                batch_start_time=time.time()
        avg_epoch_loss = epoch_loss_val/max(num_batches,1)
        print(f"LNS-AE Epoch {epoch+1}/{num_epochs} Avg Loss: {avg_epoch_loss:.6f}")
        scheduler.step(avg_epoch_loss)
        if avg_epoch_loss<best_loss:
            best_loss=avg_epoch_loss; print(f"Saving LNS-AE ckpt loss {best_loss:.6f}")
            # Save architecture params with AE checkpoint
            save_dict_ae = {
                'epoch':epoch,'model_state_dict':autoencoder.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),'loss':best_loss,
                'num_state_vars': autoencoder.encoder.encoder_net[0].conv.in_channels, # Infer from first layer
                'target_nx_full': autoencoder.decoder.target_nx_full,
                'ae_initial_width': autoencoder.encoder.encoder_net[0].conv.out_channels, # Infer
                'ae_downsample_blocks': sum(isinstance(m, ConvBlock) and m.conv.stride[0]==2 for m in autoencoder.encoder.encoder_net),
                'ae_latent_channels': autoencoder.encoder.encoder_net[-2].out_channels, # Infer from layer before pool
                'final_latent_nx': autoencoder.encoder.final_latent_nx
            }
            torch.save(save_dict_ae,checkpoint_path)
    print("LNS Autoencoder Training finished.")
    if os.path.exists(checkpoint_path): 
        print(f"Loading best LNS-AE model from {checkpoint_path}")
        ckpt=torch.load(checkpoint_path,map_location=device)
        # Re-instantiate model with saved params if architecture changed, or load directly if arch is fixed
        autoencoder.load_state_dict(ckpt['model_state_dict'])
    return autoencoder

def train_latent_don_stepper(latent_stepper, encoder, 
                             data_loader, train_nt_for_model,
                             lr=5e-4, num_epochs=150, device='cuda',
                             checkpoint_path='latent_don_stepper_ckpt.pt', clip_grad_norm=1.0,
                             train_rollout_steps=1):
    latent_stepper.to(device); encoder.to(device); encoder.eval() # Freeze encoder
    optimizer = optim.AdamW(latent_stepper.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs*len(data_loader)) # Example scheduler
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

            u_snapshots_all = state_seq_true_train.permute(0,1,3,2).reshape(B*nt_actual, num_vars_actual, nx_actual)
            with torch.no_grad(): z_all_encoded = encoder(u_snapshots_all)
            latent_C, latent_Nx_enc = z_all_encoded.shape[-2:]
            z_seq_true_train = z_all_encoded.view(B, nt_actual, latent_C, latent_Nx_enc)

            current_rollout = train_rollout_steps
            if train_nt_for_model - current_rollout <=0 : current_rollout = 1 # Ensure at least 1 step
            
            optimizer.zero_grad()
            tensor_rollout_loss = torch.tensor(0.0, device=device, requires_grad=True) # For backward

            for _ in range(B): # Simulate per-sample rollout for batch diversity, or average over batch
                rollout_start_idx = random.randint(0, train_nt_for_model - 1 - current_rollout)
                z_current_rollout_sample = z_seq_true_train[_, rollout_start_idx, :, :].unsqueeze(0) # [1, C, Nx_lat]
                
                sample_rollout_loss = torch.tensor(0.0, device=device, requires_grad=False)

                for k_roll in range(current_rollout):
                    t_idx_for_bc = rollout_start_idx + k_roll + 1
                    bc_ctrl_input_roll_sample = BC_Ctrl_seq_train[_, t_idx_for_bc, :].unsqueeze(0) # [1, bc_dim]
                    z_next_pred_rollout_sample = latent_stepper(z_current_rollout_sample, bc_ctrl_input_roll_sample)
                    z_next_target_rollout_sample = z_seq_true_train[_, rollout_start_idx + k_roll + 1, :, :].unsqueeze(0)
                    
                    step_loss_tensor = mse_loss(z_next_pred_rollout_sample, z_next_target_rollout_sample)
                    tensor_rollout_loss = tensor_rollout_loss + step_loss_tensor # Accumulate tensor loss for backward
                    sample_rollout_loss += step_loss_tensor.item() # For reporting
                    z_current_rollout_sample = z_next_pred_rollout_sample # Autoregressive
            
            avg_rollout_loss_for_backward = tensor_rollout_loss / (B * current_rollout) if (B * current_rollout) > 0 else tensor_rollout_loss
            avg_rollout_loss_for_backward.backward()
            
            current_batch_loss_item = (tensor_rollout_loss.item()) / (B * current_rollout) if (B * current_rollout) > 0 else 0.0


            epoch_loss_val+=current_batch_loss_item; num_batches+=1
            if clip_grad_norm>0: torch.nn.utils.clip_grad_norm_(latent_stepper.parameters(),max_norm=clip_grad_norm)
            optimizer.step(); scheduler.step()

            if (i+1)%50==0:
                elapsed=time.time()-batch_start_time
                print(f" LatentDON Ep {epoch+1} B {i+1}/{len(data_loader)}, Loss {current_batch_loss_item:.3e}, LR {optimizer.param_groups[0]['lr']:.3e}, Time/50B {elapsed:.2f}s")
                batch_start_time=time.time()
        avg_epoch_loss = epoch_loss_val/max(num_batches,1)
        print(f"LatentDON Stepper Epoch {epoch+1}/{num_epochs} Avg Loss: {avg_epoch_loss:.6f}")
        if avg_epoch_loss<best_loss:
            best_loss=avg_epoch_loss; print(f"Saving LatentDON Stepper ckpt loss {best_loss:.6f}")
            # Save stepper architecture params
            save_dict_stepper = {
                'epoch':epoch,'model_state_dict':latent_stepper.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict(),'loss':best_loss,
                'latent_dim_c': latent_stepper.latent_dim_c,
                'latent_dim_x': latent_stepper.latent_dim_x,
                'bc_ctrl_input_dim': latent_stepper.trunk_net.net[0].in_features, # Infer from trunk
                'branch_hidden_dims': [l.out_features for l in latent_stepper.branch_net.net if isinstance(l, nn.Linear)][:-1], # Approx
                'trunk_hidden_dims': [l.out_features for l in latent_stepper.trunk_net.net if isinstance(l, nn.Linear)][:-1], # Approx
                'combined_output_p': latent_stepper.branch_net.net[-1].out_features # Infer from branch output
            }
            torch.save(save_dict_stepper,checkpoint_path)
    print("LatentDON Stepper Training finished.")
    if os.path.exists(checkpoint_path): 
        print(f"Loading best LatentDON Stepper model from {checkpoint_path}")
        ckpt=torch.load(checkpoint_path,map_location=device)
        # Re-instantiate stepper with saved params if arch changed
        latent_stepper.load_state_dict(ckpt['model_state_dict'])
    return latent_stepper


def validate_latent_don_stepper(encoder, decoder, latent_stepper,
                                data_loader, dataset_type,
                                train_nt_for_model_training: int, T_value_for_model_training: float,
                                full_T_in_datafile: float, full_nt_in_datafile: int,
                                dataset_params_for_plot: dict, global_norm_stats_val: dict, device='cuda',
                                save_fig_path_prefix='latent_don_result'):
    encoder.eval(); decoder.eval(); latent_stepper.eval()

    if dataset_type in ['advection','burgers', 'heat_delayed_feedback', 
                        'reaction_diffusion_neumann_feedback', 
                        'heat_nonlinear_feedback_gain', 'convdiff']:
        state_keys_val=['U']
    elif dataset_type == 'euler': state_keys_val=['rho','u']
    elif dataset_type == 'darcy': state_keys_val=['P']
    else: raise ValueError(f"Unknown type '{dataset_type}' in validation for LNS-AE")
    num_state_vars_val=len(state_keys_val)


    test_horizons_T_values = [T_value_for_model_training]
    if full_T_in_datafile > T_value_for_model_training + 1e-5 :
        test_horizons_T_values.append(T_value_for_model_training + 0.5 * (full_T_in_datafile - T_value_for_model_training))
        test_horizons_T_values.append(full_T_in_datafile)
    test_horizons_T_values = sorted(list(set(h for h in test_horizons_T_values if h <= full_T_in_datafile + 1e-6 and h > 0)))
    print(f"LatentDON Stepper Validation for T_horizons: {test_horizons_T_values}")

    results_primary_horizon = {key: {'mse':[],'rmse':[],'relative_error':[],'max_error':[]} for key in state_keys_val}
    overall_rel_err_primary_horizon = []
    
    nx_plot = dataset_params_for_plot.get('nx',128)

    with torch.no_grad():
        try: state_data_full_loaded, BC_Ctrl_tensor_full_loaded, _ = next(iter(data_loader))
        except StopIteration: print("Val data_loader empty. Skipping validation."); return

        if isinstance(state_data_full_loaded,list): state_seq_true_norm_full_sample = torch.stack(state_data_full_loaded,dim=-1)[0].to(device)
        else: state_seq_true_norm_full_sample = state_data_full_loaded.unsqueeze(-1)[0].to(device)
        BC_Ctrl_seq_norm_full_sample = BC_Ctrl_tensor_full_loaded[0].to(device)
        
        current_sample_full_nt_val = state_seq_true_norm_full_sample.shape[0]


        u_t0_physical_norm = state_seq_true_norm_full_sample[0:1,:,:].permute(0,2,1)
        z_current_latent = encoder(u_t0_physical_norm)

        # Max rollout based on longest T_horizon and current sample's length
        max_T_rollout = max(test_horizons_T_values) if test_horizons_T_values else 0
        max_nt_rollout = int((max_T_rollout / full_T_in_datafile) * (current_sample_full_nt_val - 1)) + 1
        max_nt_rollout = min(max_nt_rollout, current_sample_full_nt_val)
        
        u_pred_seq_physical_norm_all_horizons = torch.zeros(max_nt_rollout, nx_plot, num_state_vars_val, device=device)
        if max_nt_rollout > 0:
            u_pred_seq_physical_norm_all_horizons[0,:,:] = state_seq_true_norm_full_sample[0,:,:]
        
        z_rollout_current_for_all = z_current_latent.clone()

        for t_step in range(max_nt_rollout - 1):
            bc_ctrl_input_roll = BC_Ctrl_seq_norm_full_sample[t_step+1:t_step+2,:] # Use BC at t+1 to predict state at t+1
            z_next_pred_latent = latent_stepper(z_rollout_current_for_all, bc_ctrl_input_roll)
            u_next_pred_physical_norm = decoder(z_next_pred_latent).permute(0,2,1).squeeze(0)
            u_pred_seq_physical_norm_all_horizons[t_step+1,:,:] = u_next_pred_physical_norm
            z_rollout_current_for_all = z_next_pred_latent


        for T_horizon_current in test_horizons_T_values:
            nt_for_rollout = int((T_horizon_current/full_T_in_datafile)*(current_sample_full_nt_val-1))+1
            nt_for_rollout = min(nt_for_rollout, max_nt_rollout) # Cannot exceed what was rolled out
            if nt_for_rollout <=0: continue

            print(f"\n  Rollout for T_horizon = {T_horizon_current:.2f} (nt = {nt_for_rollout})")
            
            u_pred_seq_physical_norm_horizon = u_pred_seq_physical_norm_all_horizons[:nt_for_rollout, :, :]
            
            U_pred_denorm_h={}; U_gt_denorm_h={}
            pred_list_h=[]; gt_list_h=[]
            state_true_norm_sliced_h = state_seq_true_norm_full_sample[:nt_for_rollout,:,:]
            pred_np_h = u_pred_seq_physical_norm_horizon.cpu().numpy(); gt_np_h = state_true_norm_sliced_h.cpu().numpy()

            for k_idx, key_val in enumerate(state_keys_val):
                mean_k = global_norm_stats_val[f'state_{key_val}_mean']
                std_k = global_norm_stats_val[f'state_{key_val}_std']
                if isinstance(mean_k, np.ndarray): mean_k = mean_k.item() # Ensure scalar if it's a 0-d array
                if isinstance(std_k, np.ndarray): std_k = std_k.item()
                if abs(std_k) < 1e-9: std_k = 1.0

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
            
            if pred_list_h and gt_list_h:
                pred_vec_h=np.concatenate(pred_list_h); gt_vec_h=np.concatenate(gt_list_h)
                if gt_vec_h.size > 0:
                    overall_rel_h=np.linalg.norm(pred_vec_h-gt_vec_h)/(np.linalg.norm(gt_vec_h)+1e-10)
                    print(f"    Overall RelErr @ T={T_horizon_current:.1f}: {overall_rel_h:.3e}")
                    if abs(T_horizon_current-T_value_for_model_training)<1e-6: overall_rel_err_primary_horizon.append(overall_rel_h)
                else: print(f"    Skipping Overall RelErr @ T={T_horizon_current:.1f} due to empty ground truth vector.")
            else: print(f"    Skipping Overall RelErr @ T={T_horizon_current:.1f} due to empty pred/gt lists.")


            fig,axs=plt.subplots(num_state_vars_val,3,figsize=(18,5*num_state_vars_val),squeeze=False)
            # ... (Plotting logic - same as FNO/BENO, ensure titles reflect "LatentDON Pred")
            fig_L_plot=dataset_params_for_plot.get('L',1.0); fig_nx_plot=nx_plot; fig_ny_plot=dataset_params_for_plot.get('ny',1)
            for k_idx, key_val in enumerate(state_keys_val):
                gt_p=U_gt_denorm_h[key_val]; pred_p=U_pred_denorm_h[key_val]
                if gt_p.shape[0] == 0 or pred_p.shape[0] == 0:
                    for ax_idx in range(3): axs[k_idx, ax_idx].text(0.5,0.5, "No data", ha="center", va="center"); axs[k_idx, ax_idx].set_xticks([]); axs[k_idx, ax_idx].set_yticks([])
                    axs[k_idx,0].set_title(f"GT ({key_val})"); axs[k_idx,1].set_title(f"Pred ({key_val})"); axs[k_idx,2].set_title("Error"); continue
                
                diff_p=np.abs(pred_p-gt_p)
                max_err_p=np.max(diff_p) if diff_p.size>0 else 0
                vmin_p=min(np.min(gt_p),np.min(pred_p)) if gt_p.size>0 and pred_p.size > 0 else 0
                vmax_p=max(np.max(gt_p),np.max(pred_p)) if gt_p.size>0 and pred_p.size > 0 else 1
                if abs(vmax_p - vmin_p) < 1e-9: vmax_p = vmin_p + 1.0

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
    # ... (Summary printing - same as FNO/BENO)
    for key_val in state_keys_val:
        if results_primary_horizon[key_val]['mse']:
            avg_mse=np.mean(results_primary_horizon[key_val]['mse']); avg_rmse=np.sqrt(avg_mse)
            avg_rel=np.mean(results_primary_horizon[key_val]['relative_error'])
            print(f"  Var '{key_val}': Avg MSE={avg_mse:.3e}, RMSE={avg_rmse:.3e}, RelErr={avg_rel:.3e}")
    if overall_rel_err_primary_horizon: print(f"  Overall RelErr @ T={T_value_for_model_training:.1f}: {np.mean(overall_rel_err_primary_horizon):.3e}")
    print("------------------------")


# main script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run LatentDON-Stepper for PDE datasets.") # Updated description
    parser.add_argument('--datatype', type=str, required=True,
                        choices=['advection', 'euler', 'burgers', 'darcy',
                                 'heat_delayed_feedback',
                                 'reaction_diffusion_neumann_feedback',
                                 'heat_nonlinear_feedback_gain',
                                 'convdiff'], # Added convdiff
                        help='Type of dataset to run LatentDON-Stepper on.')
    args = parser.parse_args()
    DATASET_TYPE = args.datatype
    MODEL_TYPE = 'LatentDON_Stepper'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Running {MODEL_TYPE} Baseline for {DATASET_TYPE} on {device} ---")

    # --- Model Hyperparameters ---
    AE_INITIAL_WIDTH = 64; AE_DOWNSAMPLE_BLOCKS = 3; AE_LATENT_CHANNELS = 16
    # AE_FINAL_LATENT_NX will be determined after loading dataset_params_for_plot
    LATENT_STEPPER_BRANCH_HIDDEN = [128, 128]
    LATENT_STEPPER_TRUNK_HIDDEN = [64, 64]
    LATENT_STEPPER_COMBINED_P = 128
    AE_LR = 3e-4
    AE_EPOCHS = 100 
    PROP_LR = 1e-3
    PROP_EPOCHS = 150 
    PROP_TRAIN_ROLLOUT_STEPS = 1
    BATCH_SIZE = 32; CLIP_GRAD_NORM = 1.0

    # --- Dataset Paths and Parameters ---
    dataset_params_for_plot = {}
    main_state_keys = []
    main_num_state_vars = 0
    FULL_T_IN_DATAFILE = 2.0; FULL_NT_IN_DATAFILE = 300 # MUST MATCH GENERATION
    TRAIN_T_TARGET = 1.5  
    if DATASET_TYPE == 'advection':
        FULL_T_IN_DATAFILE = 2.0; FULL_NT_IN_DATAFILE = 600
        TRAIN_T_TARGET = 1.0
        dataset_path = "./datasets_full/advection_data_10000s_128nx_600nt.pkl"
        main_state_keys=['U']; main_num_state_vars=1
        dataset_params_for_plot={'nx':128,'ny':1,'L':1.0,'T':FULL_T_IN_DATAFILE}
    elif DATASET_TYPE == 'euler':
        FULL_T_IN_DATAFILE = 2.0; FULL_NT_IN_DATAFILE = 600
        TRAIN_T_TARGET = 1.0
        dataset_path = "./datasets_full/euler_data_10000s_128nx_600nt.pkl"
        main_state_keys=['rho','u']; main_num_state_vars=2
        dataset_params_for_plot={'nx':128,'ny':1,'L':1.0,'T':FULL_T_IN_DATAFILE}
    elif DATASET_TYPE == 'burgers':
        FULL_T_IN_DATAFILE = 2.0; FULL_NT_IN_DATAFILE = 600
        TRAIN_T_TARGET = 1.0
        dataset_path = "./datasets_full/burgers_data_10000s_128nx_600nt.pkl"
        main_state_keys=['U']; main_num_state_vars=1
        dataset_params_for_plot={'nx':128,'ny':1,'L':1.0,'T':FULL_T_IN_DATAFILE}
    elif DATASET_TYPE == 'darcy':
        FULL_T_IN_DATAFILE = 2.0; FULL_NT_IN_DATAFILE = 600
        TRAIN_T_TARGET = 1.0
        dataset_path = "./datasets_full/darcy_data_10000s_128nx_600nt.pkl"
        main_state_keys=['P']; main_num_state_vars=1
        dataset_params_for_plot={'nx':128,'ny':1,'L':1.0,'T':FULL_T_IN_DATAFILE}
    elif DATASET_TYPE == 'heat_delayed_feedback':
        dataset_path = "./datasets_new_feedback/heat_delayed_feedback_v1_5000s_64nx_300nt.pkl" # UPDATE
        main_state_keys=['U']; main_num_state_vars=1
        dataset_params_for_plot={'nx':64,'ny':1,'L':1.0,'T':FULL_T_IN_DATAFILE}
    elif DATASET_TYPE == 'reaction_diffusion_neumann_feedback':
        dataset_path = "./datasets_new_feedback/reaction_diffusion_neumann_feedback_v1_5000s_64nx_300nt.pkl" # UPDATE
        main_state_keys=['U']; main_num_state_vars=1
        dataset_params_for_plot={'nx':64,'ny':1,'L':1.0,'T':FULL_T_IN_DATAFILE}
    elif DATASET_TYPE == 'heat_nonlinear_feedback_gain':
        dataset_path = "./datasets_new_feedback/heat_nonlinear_feedback_gain_v1_5000s_64nx_300nt.pkl" # UPDATE
        main_state_keys=['U']; main_num_state_vars=1
        dataset_params_for_plot={'nx':64,'ny':1,'L':1.0,'T':FULL_T_IN_DATAFILE}
    elif DATASET_TYPE == 'convdiff':
        dataset_path = "./datasets_new_feedback/convdiff_v1_5000s_64nx_300nt.pkl" # UPDATE PATH
        main_state_keys = ['U']; main_num_state_vars = 1
        dataset_params_for_plot = {'nx': 64, 'ny': 1, 'L': 1.0, 'T': FULL_T_IN_DATAFILE}
    else:
        raise ValueError(f"Unknown or unconfigured dataset_type: {DATASET_TYPE}")

    
    TRAIN_NT_FOR_MODEL = int((TRAIN_T_TARGET/FULL_T_IN_DATAFILE)*(FULL_NT_IN_DATAFILE-1))+1
    print(f"Dataset: {DATASET_TYPE}")
    print(f"Datafile T={FULL_T_IN_DATAFILE}, nt_points={FULL_NT_IN_DATAFILE}")
    print(f"Training with T_duration={TRAIN_T_TARGET}, nt_points_in_sequence={TRAIN_NT_FOR_MODEL}")
    
    # Determine AE_FINAL_LATENT_NX after dataset_params_for_plot['nx'] is set
    AE_FINAL_LATENT_NX = dataset_params_for_plot.get('nx',64) // (2**AE_DOWNSAMPLE_BLOCKS)
    print(f"Autoencoder final_latent_nx calculated as: {AE_FINAL_LATENT_NX}")


    print(f"Loading dataset: {dataset_path}")
    try:
        with open(dataset_path,'rb') as f: data_list_all=pickle.load(f)
        print(f"Loaded {len(data_list_all)} samples.")
        if data_list_all:
            sample_params_check = data_list_all[0].get('params', {})
            file_nt_check = data_list_all[0][main_state_keys[0]].shape[0]
            file_T_check_val = sample_params_check.get('T', 'N/A')
            file_L_check_val = sample_params_check.get('L', 'N/A')
            file_nx_check_val = sample_params_check.get('nx', data_list_all[0][main_state_keys[0]].shape[1])
            print(f"  Sample 0 from file: nt={file_nt_check}, T={file_T_check_val}, nx={file_nx_check_val}, L={file_L_check_val}")
            if file_nt_check != FULL_NT_IN_DATAFILE:
                print(f"  CRITICAL WARNING: FULL_NT_IN_DATAFILE ({FULL_NT_IN_DATAFILE}) in script "
                      f"differs from actual nt in file ({file_nt_check}) for {DATASET_TYPE}. "
                      f"Ensure FULL_NT_IN_DATAFILE is correctly set for this dataset type in the script.")
            if isinstance(file_T_check_val, (int,float)): dataset_params_for_plot['T'] = file_T_check_val # Override if in file
            if isinstance(file_L_check_val, (int,float)): dataset_params_for_plot['L'] = file_L_check_val
            dataset_params_for_plot['nx'] = file_nx_check_val # Always use nx from file for consistency
            # Recalculate AE_FINAL_LATENT_NX if nx changed
            AE_FINAL_LATENT_NX = dataset_params_for_plot.get('nx',128) // (2**AE_DOWNSAMPLE_BLOCKS)
            print(f"  After file check, AE_FINAL_LATENT_NX re-calculated to: {AE_FINAL_LATENT_NX}")


    except FileNotFoundError: print(f"Error: File not found {dataset_path}"); exit()
    if not data_list_all: print("No data. Exiting."); exit()

    random.shuffle(data_list_all); n_total=len(data_list_all); n_train=int(0.8*n_total)
    train_data_list_split=data_list_all[:n_train]; val_data_list_split=data_list_all[n_train:]
    print(f"Train samples: {len(train_data_list_split)}, Validation samples: {len(val_data_list_split)}")


    print("Computing global normalization statistics from training data...")
    global_norm_stats = {}
    # State variables
    for sk_stat in main_state_keys:
        all_state_data_for_stats_var = np.concatenate([
            sample[sk_stat][:TRAIN_NT_FOR_MODEL,...] for sample in train_data_list_split
        ], axis=0)
        global_norm_stats[f"state_{sk_stat}_mean"] = np.mean(all_state_data_for_stats_var)
        global_norm_stats[f"state_{sk_stat}_std"] = np.std(all_state_data_for_stats_var) + 1e-8
        print(f"  Global norm for {sk_stat}: mean={global_norm_stats[f'state_{sk_stat}_mean']:.4f}, std={global_norm_stats[f'state_{sk_stat}_std']:.4f}")
    
    # BC_State (column-wise)
    if train_data_list_split: # Ensure list is not empty
        temp_bc_state_dim = train_data_list_split[0]['BC_State'].shape[1]
        all_bc_state_data_for_stats = np.concatenate([
            sample['BC_State'][:TRAIN_NT_FOR_MODEL,...] for sample in train_data_list_split
        ], axis=0)
        global_norm_stats["BC_State_mean"] = np.mean(all_bc_state_data_for_stats, axis=0)
        global_norm_stats["BC_State_std"] = np.std(all_bc_state_data_for_stats, axis=0) + 1e-8
        print(f"  Global norm for BC_State: mean={global_norm_stats['BC_State_mean']}, std={global_norm_stats['BC_State_std']}")

        # BC_Control (column-wise, if exists)
        if train_data_list_split[0].get('BC_Control') is not None and train_data_list_split[0]['BC_Control'].size > 0:
            temp_bc_control_dim = train_data_list_split[0]['BC_Control'].shape[1]
            if temp_bc_control_dim > 0:
                all_bc_control_data_for_stats = np.concatenate([
                    sample['BC_Control'][:TRAIN_NT_FOR_MODEL,...] for sample in train_data_list_split if sample.get('BC_Control') is not None and sample['BC_Control'].size > 0
                ], axis=0)
                if all_bc_control_data_for_stats.size > 0:
                    global_norm_stats["BC_Control_mean"] = np.mean(all_bc_control_data_for_stats, axis=0)
                    global_norm_stats["BC_Control_std"] = np.std(all_bc_control_data_for_stats, axis=0) + 1e-8
                    print(f"  Global norm for BC_Control: mean={global_norm_stats['BC_Control_mean']}, std={global_norm_stats['BC_Control_std']}")
                else:
                    global_norm_stats["BC_Control_mean"] = np.array([]) 
                    global_norm_stats["BC_Control_std"] = np.array([])
            else: # num_controls was 0
                global_norm_stats["BC_Control_mean"] = np.array([]) 
                global_norm_stats["BC_Control_std"] = np.array([])
        else: # No BC_Control key or it's empty
            global_norm_stats["BC_Control_mean"] = np.array([]) 
            global_norm_stats["BC_Control_std"] = np.array([])
    print("Global normalization stats computed.")


    train_dataset = UniversalPDEDataset(train_data_list_split, dataset_type=DATASET_TYPE, global_norm_stats=global_norm_stats, train_nt_limit=TRAIN_NT_FOR_MODEL)
    val_dataset = UniversalPDEDataset(val_data_list_split, dataset_type=DATASET_TYPE, global_norm_stats=global_norm_stats, train_nt_limit=None)

    num_workers=1 # Adjusted for potential debugging ease
    train_loader_ae = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=num_workers,pin_memory=True,drop_last=True)
    train_loader_prop = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=num_workers,pin_memory=True,drop_last=True)
    val_loader = DataLoader(val_dataset,batch_size=1,shuffle=False,num_workers=num_workers)

    actual_bc_ctrl_dim_from_dataset = train_dataset.bc_state_dim + train_dataset.num_controls
    current_nx_from_dataset = train_dataset.nx # This is the actual nx for the AE's target_nx_full

    print(f"\nInitializing LNS Autoencoder: input_channels={main_num_state_vars}, target_nx_full={current_nx_from_dataset}, latent_channels={AE_LATENT_CHANNELS}, final_latent_nx={AE_FINAL_LATENT_NX}")
    autoencoder = LNS_Autoencoder(num_state_vars=main_num_state_vars, target_nx_full=current_nx_from_dataset, ae_initial_width=AE_INITIAL_WIDTH, ae_downsample_blocks=AE_DOWNSAMPLE_BLOCKS, ae_latent_channels=AE_LATENT_CHANNELS, final_latent_nx=AE_FINAL_LATENT_NX).to(device)
    
    timestamp_ae = time.strftime("%Y%m%d-%H%M%S") # Unique timestamp for AE
    run_name_ae = f"{DATASET_TYPE}_{MODEL_TYPE}_AE_w{AE_INITIAL_WIDTH}_lc{AE_LATENT_CHANNELS}"
    checkpoint_dir_ae = f"./New_ckpt_2/checkpoints_{DATASET_TYPE}_{MODEL_TYPE}"
    os.makedirs(checkpoint_dir_ae, exist_ok=True)
    ae_checkpoint_path = os.path.join(checkpoint_dir_ae, f'model_ae_{run_name_ae}.pt')
    
    print(f"Starting LNS Autoencoder training (checkpoint: {ae_checkpoint_path})...")
    autoencoder = train_lns_autoencoder(autoencoder, train_loader_ae, TRAIN_NT_FOR_MODEL, lr=AE_LR, num_epochs=AE_EPOCHS, device=device, checkpoint_path=ae_checkpoint_path, clip_grad_norm=CLIP_GRAD_NORM)

    print(f"\nInitializing LatentDON Stepper: latent_C={AE_LATENT_CHANNELS}, latent_Nx={AE_FINAL_LATENT_NX}, bc_ctrl_input_dim={actual_bc_ctrl_dim_from_dataset}")
    latent_don_stepper = LatentStepperNet(
        latent_dim_c=AE_LATENT_CHANNELS, latent_dim_x=AE_FINAL_LATENT_NX,
        bc_ctrl_input_dim=actual_bc_ctrl_dim_from_dataset,
        branch_hidden_dims=LATENT_STEPPER_BRANCH_HIDDEN,
        trunk_hidden_dims=LATENT_STEPPER_TRUNK_HIDDEN,
        combined_output_p=LATENT_STEPPER_COMBINED_P
    ).to(device)

    timestamp_prop = time.strftime("%Y%m%d-%H%M%S") # Unique timestamp for Propagator
    run_name_prop = f"{DATASET_TYPE}_{MODEL_TYPE}_Prop_p{LATENT_STEPPER_COMBINED_P}"
    checkpoint_dir_prop = f"./New_ckpt_2/checkpoints_{DATASET_TYPE}_{MODEL_TYPE}" # Can be same dir
    # os.makedirs(checkpoint_dir_prop, exist_ok=True) # Already created for AE
    prop_checkpoint_path = os.path.join(checkpoint_dir_prop, f'model_prop_{run_name_prop}.pt')

    print(f"Starting LatentDON Stepper training (checkpoint: {prop_checkpoint_path})...")
    latent_don_stepper = train_latent_don_stepper(latent_don_stepper, autoencoder.encoder, train_loader_prop, TRAIN_NT_FOR_MODEL, lr=PROP_LR, num_epochs=PROP_EPOCHS, device=device, checkpoint_path=prop_checkpoint_path, clip_grad_norm=CLIP_GRAD_NORM, train_rollout_steps=PROP_TRAIN_ROLLOUT_STEPS)

    # --- Validation ---
    # Use a combined run name for results if desired, or keep separate
    timestamp_val = time.strftime("%Y%m%d-%H%M%S")
    run_name_val = f"{DATASET_TYPE}_{MODEL_TYPE}_FullVal_{timestamp_val}"
    results_dir_val = f"./result_all_2/results_{DATASET_TYPE}_{MODEL_TYPE}"
    os.makedirs(results_dir_val, exist_ok=True)
    save_fig_path_prefix_val = os.path.join(results_dir_val, f'result_{run_name_val}')


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
            save_fig_path_prefix=save_fig_path_prefix_val
        )
    else: print("\nNo validation data. Skipping validation.")

    print("="*60); print(f"Run finished for {DATASET_TYPE} - {MODEL_TYPE}");
    print(f"AE checkpoint: {ae_checkpoint_path}"); print(f"Stepper checkpoint: {prop_checkpoint_path}");
    if val_data_list_split: print(f"Validation figures saved with prefix: {save_fig_path_prefix_val}"); print("="*60)

