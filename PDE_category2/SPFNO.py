# SPFNO
# =============================================================================
#  SPFNO: Spectral operator learning for PDEs with Dirichlet and Neumann boundary conditions (Adapted for Task2) ref:https://github.com/liu-ziyuan-math/SPFNO 
#
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
# import functools # Not strictly needed here
import argparse # Added argparse
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve # For Burgers' solver (though not directly used in SPFNO training)

# fixed seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


def sin_transform(u):
    Nx = u.shape[-1]
    V = torch.cat([u, -u.flip(dims=[-1])[..., 1:Nx-1]], dim=-1)
    a = -torch.fft.fft(V, dim=-1)[..., :Nx].imag / (Nx-1) # Added potential scaling
    return a

def isin_transform(a, n=None):
    Nx_coeffs = a.shape[-1]
    N_out = n if n is not None else Nx_coeffs
    fft_len = 2 * (N_out - 1)
    if fft_len <= 0: return torch.zeros(*a.shape[:-1], N_out, dtype=torch.float32, device=a.device)

    V_imag = torch.zeros(*a.shape[:-1], fft_len, device=a.device)
    if Nx_coeffs >= 1:
         limit = min(Nx_coeffs, N_out)
         V_imag[..., 1:limit] = a[..., 1:limit]
         if N_out > 1 and limit > 1: # Ensure there's something to flip
             V_imag[..., N_out:] = -a[..., 1:limit-1].flip(dims=[-1])


    V_complex = torch.zeros_like(V_imag, dtype=torch.complex64)
    V_complex.imag = -V_imag
    u = torch.fft.ifft(V_complex, dim=-1)[..., :N_out].real * (N_out - 1.0) * 0.5 # Matching forward scaling approximately
    return u


def cos_transform(u): # DCT-I (scaled)
    Nx = u.shape[-1]
    if Nx < 2 : return u.clone()
    V = torch.cat([u, u[..., 1:Nx-1].flip(dims=[-1])], dim=-1)
    a = torch.fft.rfft(V, dim=-1).real / (Nx-1.0 if Nx > 1 else 1.0)
    return a


def icos_transform(a, n=None): # Inverse DCT-I (scaled)
    Nx_coeffs = a.shape[-1]
    N_out = n if n is not None else Nx_coeffs

    if N_out < 1: return torch.zeros(*a.shape[:-1], N_out, dtype=a.dtype, device=a.device)
    if N_out == 1: return a[..., :1].clone() if Nx_coeffs >=1 else torch.zeros(*a.shape[:-1], 1, dtype=a.dtype, device=a.device)

    fft_len = 2 * (N_out - 1)
    if fft_len <=0: # Handles N_out = 1
        if N_out == 1 and Nx_coeffs >=1: return a[...,:1].clone() # DCT of 1 point is itself
        else: return torch.zeros(*a.shape[:-1], N_out, dtype=a.dtype, device=a.device)


    fft_input_real = torch.zeros(*a.shape[:-1], fft_len, device=a.device, dtype=a.dtype)
    coeffs_to_use = min(Nx_coeffs, N_out)

    fft_input_real[..., :coeffs_to_use] = a[..., :coeffs_to_use]
    if N_out > 1 and coeffs_to_use > 1: # Check if there's anything to flip
        # Correct indices for flipping: a[1] to a[N-2] (if coeffs_to_use covers this range)
        # The flip should be for elements a_1, ..., a_{N_out-2}
        # So, from a, we take elements 1 to min(coeffs_to_use, N_out-1)-1
        num_to_flip = min(coeffs_to_use, N_out -1) -1
        if num_to_flip > 0:
             fft_input_real[..., N_out-1 : N_out-1 + num_to_flip] = a[..., 1 : 1+num_to_flip].flip(dims=[-1])


    fft_input_complex = torch.zeros_like(fft_input_real, dtype=torch.complex64)
    fft_input_complex.real = fft_input_real
    u_full = torch.fft.irfft(fft_input_complex, n=fft_len) * (N_out - 1.0) # Apply inverse scaling from forward
    u = u_full[..., :N_out]
    return u


def WSWA(u):
    return torch.fft.rfft(u)

def iWSWA(a, n=None):
    N_out = n if n is not None else ( (a.shape[-1]-1)*2 if a.shape[-1]>0 else 0 )
    if N_out == 0 and a.shape[-1] == 1 : N_out = 1
    return torch.fft.irfft(a, n=N_out)


class UniversalPDEDataset(Dataset):
    def __init__(self, data_list, dataset_type, train_nt_limit=None):
        if not data_list: raise ValueError("data_list cannot be empty")
        self.data_list = data_list; self.dataset_type = dataset_type.lower(); self.train_nt_limit = train_nt_limit
        first_sample = data_list[0]; params = first_sample.get('params', {})
        self.nt_from_sample_file=0; self.nx_from_sample_file=0; self.ny_from_sample_file=1


        if self.dataset_type in ['heat_delayed_feedback', 
                                   'reaction_diffusion_neumann_feedback', 
                                   'heat_nonlinear_feedback_gain', 
                                   'convdiff']:
            self.nt_from_sample_file=first_sample['U'].shape[0]; self.nx_from_sample_file=first_sample['U'].shape[1]
            self.state_keys=['U']; self.num_state_vars=1; self.expected_bc_state_dim=2
        else: raise ValueError(f"Unknown dataset_type in UniversalPDEDataset: {self.dataset_type}")
        
        self.effective_nt_for_loader = self.train_nt_limit if self.train_nt_limit is not None else self.nt_from_sample_file
        self.nx=self.nx_from_sample_file; self.ny=self.ny_from_sample_file; self.spatial_dim=self.nx*self.ny
        
        self.bc_state_key='BC_State'
        if self.bc_state_key not in first_sample:
             raise KeyError(f"'{self.bc_state_key}' not found in the first sample of dataset type '{self.dataset_type}'!")
        actual_bc_dim = first_sample[self.bc_state_key].shape[1]
        if actual_bc_dim != self.expected_bc_state_dim: 
            print(f"Warn: BC_State dim mismatch for {self.dataset_type}. Exp {self.expected_bc_state_dim}, got {actual_bc_dim}. Using actual: {actual_bc_dim}")
        self.bc_state_dim = actual_bc_dim
        
        self.bc_control_key='BC_Control'
        if self.bc_control_key in first_sample and first_sample[self.bc_control_key] is not None and first_sample[self.bc_control_key].size>0: 
            self.num_controls = first_sample[self.bc_control_key].shape[1]
        else: self.num_controls=0

    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx):
        sample=self.data_list[idx]; norm_factors={}; current_nt=self.effective_nt_for_loader; slist=[]
        for key in self.state_keys:
            s_full=sample[key]; s_seq=s_full[:current_nt,...]; 
            s_mean=np.mean(s_seq); s_std=np.std(s_seq)+1e-8
            slist.append(torch.tensor((s_seq-s_mean)/s_std).float()); 
            norm_factors[f'{key}_mean']=s_mean; norm_factors[f'{key}_std']=s_std
        
        bcs_full=sample[self.bc_state_key]; bcs_seq=bcs_full[:current_nt,:]; bcs_norm=np.zeros_like(bcs_seq,dtype=np.float32)
        norm_factors[f'{self.bc_state_key}_means']=np.zeros(self.bc_state_dim); norm_factors[f'{self.bc_state_key}_stds']=np.ones(self.bc_state_dim)
        if bcs_seq.size > 0: # Ensure not empty before iterating
            for d in range(self.bc_state_dim):
                col=bcs_seq[:,d]; m=np.mean(col); s=np.std(col)
                if s>1e-8: bcs_norm[:,d]=(col-m)/s; norm_factors[f'{self.bc_state_key}_means'][d]=m; norm_factors[f'{self.bc_state_key}_stds'][d]=s
                else: bcs_norm[:,d]=col-m; norm_factors[f'{self.bc_state_key}_means'][d]=m # Store mean, std remains 1
        bcs_tensor=torch.tensor(bcs_norm).float()
        
        if self.num_controls>0:
            bcc_full=sample[self.bc_control_key]; bcc_seq=bcc_full[:current_nt,:]; bcc_norm=np.zeros_like(bcc_seq,dtype=np.float32)
            norm_factors[f'{self.bc_control_key}_means']=np.zeros(self.num_controls); norm_factors[f'{self.bc_control_key}_stds']=np.ones(self.num_controls)
            if bcc_seq.size > 0: # Ensure not empty
                for d in range(self.num_controls):
                    col=bcc_seq[:,d]; m=np.mean(col); s=np.std(col)
                    if s>1e-8: bcc_norm[:,d]=(col-m)/s; norm_factors[f'{self.bc_control_key}_means'][d]=m; norm_factors[f'{self.bc_control_key}_stds'][d]=s
                    else: bcc_norm[:,d]=col-m; norm_factors[f'{self.bc_control_key}_means'][d]=m
            bcc_tensor=torch.tensor(bcc_norm).float()
        else: bcc_tensor=torch.empty((current_nt,0),dtype=torch.float32)
        
        bc_ctrl_tensor=torch.cat((bcs_tensor,bcc_tensor),dim=-1)
        out_state = slist[0] if self.num_state_vars==1 else slist
        return out_state, bc_ctrl_tensor, norm_factors


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super().__init__(); self.in_channels=in_channels; self.out_channels=out_channels; self.modes1=modes1
        self.scale=(1/(in_channels*out_channels)); self.weights1=nn.Parameter(self.scale*torch.rand(in_channels,out_channels,self.modes1,dtype=torch.cfloat))
    def compl_mul1d(self, input_tensor, weights): return torch.einsum("bix,iox->box", input_tensor, weights)
    def forward(self, x):
        B=x.shape[0]; x_ft=torch.fft.rfft(x); out_ft=torch.zeros(B,self.out_channels,x.size(-1)//2+1,device=x.device,dtype=torch.cfloat)
        out_ft[:,:,:self.modes1]=self.compl_mul1d(x_ft[:,:,:self.modes1],self.weights1); return torch.fft.irfft(out_ft,n=x.size(-1))

class ProjectionFilter1d(nn.Module):
    def __init__(self, transform_type='dirichlet'):
        super().__init__()
        self.transform_type = transform_type
        self.fwd_func = None
        self.inv_func = None

        if transform_type == 'dirichlet':
            print("ProjectionFilter: Using sin_transform/isin_transform for Dirichlet BCs.")
            self.fwd_func = sin_transform
            self.inv_func = isin_transform
        elif transform_type == 'neumann':
            print("ProjectionFilter: Using cos_transform/icos_transform for Neumann BCs.")
            self.fwd_func = cos_transform
            self.inv_func = icos_transform
        elif transform_type == 'mixed':
            print("ProjectionFilter: Using WSWA/iWSWA for Mixed BCs.")
            self.fwd_func = WSWA
            self.inv_func = iWSWA
        else: 
            print(f"ProjectionFilter: No specific BC projection applied (Type: {transform_type}). Using Identity.")
            self.fwd_func = lambda x: x 
            self.inv_func = lambda x, n: x

    def forward(self, x):
        if not callable(self.fwd_func) or not callable(self.inv_func):
             print(f"Warning: Invalid transform functions for type '{self.transform_type}'. Skipping filter.")
             return x
        N = x.shape[-1]
        try:
            x_transformed = self.fwd_func(x)
            x_reconstructed = self.inv_func(x_transformed, n=N)
            if x_reconstructed.shape[-1] != N:
                 if x_reconstructed.shape[-1] > N:
                     x_reconstructed = x_reconstructed[..., :N]
                 else:
                     pad_width = N - x_reconstructed.shape[-1]
                     x_reconstructed = F.pad(x_reconstructed, (0, pad_width))
        except Exception as e:
             print(f"Error during ProjectionFilter transform (type={self.transform_type}): {e}, Input shape: {x.shape}. Returning input.")
             return x 
        return x_reconstructed

class SPFNO1d(nn.Module):
    def __init__(self, modes, width, input_channels, output_channels, num_layers=4, transform_type='dirichlet', use_projection_filter=True):
        super().__init__(); self.modes1=modes; self.width=width; self.input_channels=input_channels; self.output_channels=output_channels; self.num_layers=num_layers; self.transform_type=transform_type; self.use_projection_filter=use_projection_filter
        self.fc0=nn.Linear(input_channels,self.width); self.convs=nn.ModuleList(); self.ws=nn.ModuleList()
        for _ in range(num_layers): self.convs.append(SpectralConv1d(self.width,self.width,self.modes1)); self.ws.append(nn.Conv1d(self.width,self.width,1))
        self.fc1=nn.Linear(self.width,128); self.fc2=nn.Linear(128,output_channels)
        if self.use_projection_filter and self.transform_type not in ['periodic', None, 'identity']: # Added 'identity' as a no-op
            self.proj_filter=ProjectionFilter1d(transform_type=self.transform_type)
        else: 
            self.proj_filter=None;
            if self.use_projection_filter: print(f"SPFNO Info: Projection filter not used for transform_type='{self.transform_type}'.")
        self.nx = None

    def forward(self, x): 
        x_lifted=self.fc0(x); x_permuted=x_lifted.permute(0,2,1); x_proc=x_permuted
        for i in range(self.num_layers):
            x1=self.convs[i](x_proc); x2=self.ws[i](x_proc); x_proc=x1+x2; x_proc=F.gelu(x_proc)
        x_out_perm=x_proc.permute(0,2,1); x_out=self.fc1(x_out_perm); x_out=F.gelu(x_out); x_out=self.fc2(x_out)
        if self.proj_filter is not None:
            x_proj_in = x_out.permute(0,2,1)
            x_proj_out = self.proj_filter(x_proj_in)
            x_out = x_proj_out.permute(0,2,1)
        return x_out


def train_spfno_stepper(model, data_loader, dataset_type, train_nt_for_model,
                        lr=1e-3, num_epochs=50, device='cuda',
                        checkpoint_path='spfno_checkpoint.pt', clip_grad_norm=1.0):
    model.to(device); optimizer=optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=10,verbose=True)
    mse_loss=nn.MSELoss(reduction='mean'); start_epoch=0; best_loss=float('inf')
    if os.path.exists(checkpoint_path):
        print(f"Loading SPFNO checkpoint from {checkpoint_path} ...")
        try:
            ckpt=torch.load(checkpoint_path,map_location=device); model.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                try: optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                except: print("Warn: SPFNO Optimizer state mismatch.")
            start_epoch=ckpt.get('epoch',0)+1; best_loss=ckpt.get('loss',float('inf'))
            print(f"Resuming SPFNO training from epoch {start_epoch}")
        except Exception as e: print(f"Error loading SPFNO checkpoint: {e}. Starting fresh.")

    for epoch in range(start_epoch, num_epochs):
        model.train(); epoch_loss_val=0.0; num_batches=0; batch_start_time=time.time() # Renamed epoch_loss
        for i, (state_data_loaded, BC_Ctrl_tensor_loaded, _) in enumerate(data_loader):
            if isinstance(state_data_loaded, list): state_seq_true_train=torch.stack(state_data_loaded,dim=-1).to(device)
            else: state_seq_true_train=state_data_loaded.unsqueeze(-1).to(device)
            BC_Ctrl_seq_train=BC_Ctrl_tensor_loaded.to(device)
            B,nt_loaded,nx_or_N,_ = state_seq_true_train.shape
            if nt_loaded != train_nt_for_model: raise ValueError(f"Data nt {nt_loaded} != train_nt {train_nt_for_model}")

            optimizer.zero_grad()
            tensor_total_sequence_loss=torch.tensor(0.0,device=device,requires_grad=True) # For backward
            
            for t in range(train_nt_for_model - 1):
                u_n_true=state_seq_true_train[:,t,:,:]; bc_ctrl_n=BC_Ctrl_seq_train[:,t,:]
                bc_ctrl_n_expanded=bc_ctrl_n.unsqueeze(1).repeat(1,nx_or_N,1)
                spfno_input=torch.cat((u_n_true,bc_ctrl_n_expanded),dim=-1)
                u_np1_pred=model(spfno_input)
                u_np1_true=state_seq_true_train[:,t+1,:,:]
                step_loss_tensor=mse_loss(u_np1_pred,u_np1_true)
                tensor_total_sequence_loss = tensor_total_sequence_loss + step_loss_tensor
            
            if train_nt_for_model > 1:
                current_batch_loss_for_backward = tensor_total_sequence_loss/(train_nt_for_model-1)
            else: # Should not happen for stepper training
                current_batch_loss_for_backward = tensor_total_sequence_loss
            
            current_batch_loss_item = current_batch_loss_for_backward.item() # For reporting
            epoch_loss_val+=current_batch_loss_item; num_batches+=1
            current_batch_loss_for_backward.backward()

            if clip_grad_norm>0: torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=clip_grad_norm)
            optimizer.step()
            if (i+1)%50==0:
                elapsed=time.time()-batch_start_time
                print(f" SPFNO Ep {epoch+1} B {i+1}/{len(data_loader)}, Loss: {current_batch_loss_item:.3e}, Time/50B: {elapsed:.2f}s")
                batch_start_time=time.time()
        avg_epoch_loss = epoch_loss_val/max(num_batches,1)
        print(f"SPFNO Epoch {epoch+1}/{num_epochs} Avg Loss: {avg_epoch_loss:.6f}")
        scheduler.step(avg_epoch_loss)
        if avg_epoch_loss < best_loss:
            best_loss=avg_epoch_loss; print(f"Saving SPFNO ckpt loss {best_loss:.6f}")
            torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'loss':best_loss,'dataset_type':dataset_type, 'modes':model.modes1, 'width':model.width, 'transform_type':model.transform_type},checkpoint_path)
    print("SPFNO Training finished.")
    if os.path.exists(checkpoint_path): print(f"Loading best SPFNO model"); ckpt=torch.load(checkpoint_path,map_location=device); model.load_state_dict(ckpt['model_state_dict'])
    return model


def validate_spfno_stepper(model, data_loader, dataset_type,
                           train_nt_for_model_training: int, T_value_for_model_training: float,
                           full_T_in_datafile: float, full_nt_in_datafile: int,
                           dataset_params_for_plot: dict, device='cuda',
                           save_fig_path_prefix='spfno_result'):
    model.eval()
    if dataset_type in ['advection', 'burgers', 'heat_delayed_feedback', 
                        'reaction_diffusion_neumann_feedback', 
                        'heat_nonlinear_feedback_gain', 'convdiff']:
        state_keys_val = ['U']
    elif dataset_type == 'euler': state_keys_val = ['rho', 'u']
    elif dataset_type == 'darcy': state_keys_val = ['P']
    else: raise ValueError(f"Unknown type '{dataset_type}' in SPFNO validation")
    num_state_vars_val = len(state_keys_val)

    test_horizons_T_values = [T_value_for_model_training]
    if full_T_in_datafile > T_value_for_model_training + 1e-5:
        test_horizons_T_values.append(T_value_for_model_training + 0.5 * (full_T_in_datafile - T_value_for_model_training))
        test_horizons_T_values.append(full_T_in_datafile)
    test_horizons_T_values = sorted(list(set(h for h in test_horizons_T_values if h <= full_T_in_datafile + 1e-6 and h > 0)))
    print(f"SPFNO Validation for T_horizons: {test_horizons_T_values}")

    results_primary_horizon = {key: {'mse':[],'rmse':[],'relative_error':[],'max_error':[]} for key in state_keys_val}
    overall_rel_err_primary_horizon = []

    with torch.no_grad():
        try:
            state_data_full_loaded, BC_Ctrl_tensor_full_loaded, norm_factors_batch = next(iter(data_loader))
        except StopIteration: print("Val data_loader empty. Skipping SPFNO validation."); return

        if isinstance(state_data_full_loaded, list):
            state_seq_true_norm_full = torch.stack(state_data_full_loaded, dim=-1)[0].to(device)
        else:
            state_seq_true_norm_full = state_data_full_loaded.unsqueeze(-1)[0].to(device)
        BC_Ctrl_seq_norm_full = BC_Ctrl_tensor_full_loaded[0].to(device)
        
        current_sample_full_nt_val, nx_or_N_plot, num_vars_check_val = state_seq_true_norm_full.shape
        if current_sample_full_nt_val != full_nt_in_datafile:
             print(f"  Note: Validating with sample of nt={current_sample_full_nt_val} (param full_nt_in_datafile was {full_nt_in_datafile})")


        norm_factors_sample = {}
        for key, val_tensor in norm_factors_batch.items():
            norm_factors_sample[key] = val_tensor[0].cpu().numpy() if isinstance(val_tensor,torch.Tensor) and val_tensor.ndim>0 else (val_tensor.cpu().numpy() if isinstance(val_tensor,torch.Tensor) else val_tensor)

        u_initial_norm = state_seq_true_norm_full[0:1, :, :]

        for T_horizon_current in test_horizons_T_values:
            nt_for_rollout = int((T_horizon_current/full_T_in_datafile)*(current_sample_full_nt_val-1))+1
            nt_for_rollout = min(nt_for_rollout, current_sample_full_nt_val)
            if nt_for_rollout <=0: continue
            print(f"\n  Rollout for T_horizon = {T_horizon_current:.2f} (nt = {nt_for_rollout})")

            u_pred_seq_norm_horizon = torch.zeros(nt_for_rollout, nx_or_N_plot, num_vars_check_val, device=device)
            u_current_pred_step = u_initial_norm.clone()
            if nt_for_rollout > 0:
                u_pred_seq_norm_horizon[0,:,:] = u_current_pred_step.squeeze(0)
            
            BC_Ctrl_for_rollout = BC_Ctrl_seq_norm_full[:nt_for_rollout, :]

            for t_step in range(nt_for_rollout - 1):
                bc_ctrl_n_step = BC_Ctrl_for_rollout[t_step:t_step+1, :]
                bc_ctrl_n_expanded = bc_ctrl_n_step.unsqueeze(1).repeat(1,nx_or_N_plot,1)
                if u_current_pred_step.shape[0]!=1: u_current_pred_step=u_current_pred_step.unsqueeze(0)
                spfno_input_step = torch.cat((u_current_pred_step, bc_ctrl_n_expanded), dim=-1)
                u_next_pred_norm_step = model(spfno_input_step)
                u_pred_seq_norm_horizon[t_step+1,:,:] = u_next_pred_norm_step.squeeze(0)
                u_current_pred_step = u_next_pred_norm_step
            
            U_pred_denorm_h={}; U_gt_denorm_h={}
            pred_list_h=[]; gt_list_h=[]
            state_true_norm_sliced_h = state_seq_true_norm_full[:nt_for_rollout,:,:]
            pred_np_h = u_pred_seq_norm_horizon.cpu().numpy()
            gt_np_h = state_true_norm_sliced_h.cpu().numpy()

            for k_idx, key_val in enumerate(state_keys_val):
                mean_k=norm_factors_sample[f'{key_val}_mean']; std_k=norm_factors_sample[f'{key_val}_std']
                mean_k = mean_k.item() if hasattr(mean_k,'item') else mean_k
                std_k = std_k.item() if hasattr(std_k,'item') else std_k
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
            # ... (Plotting logic - same as FNO, ensure titles reflect "SPFNO Pred")
            fig_L=dataset_params_for_plot.get('L',1.0); fig_nx=dataset_params_for_plot.get('nx',nx_or_N_plot); fig_ny=dataset_params_for_plot.get('ny',1)
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

                is_1d_p=(fig_ny==1); plot_ext=[0,fig_L,0,T_horizon_current]
                if is_1d_p:
                    im0=axs[k_idx,0].imshow(gt_p,aspect='auto',origin='lower',vmin=vmin_p,vmax=vmax_p,extent=plot_ext,cmap='viridis'); axs[k_idx,0].set_title(f"Truth ({key_val})")
                    im1=axs[k_idx,1].imshow(pred_p,aspect='auto',origin='lower',vmin=vmin_p,vmax=vmax_p,extent=plot_ext,cmap='viridis'); axs[k_idx,1].set_title(f"SPFNO Pred ({key_val})")
                    im2=axs[k_idx,2].imshow(diff_p,aspect='auto',origin='lower',extent=plot_ext,cmap='magma'); axs[k_idx,2].set_title(f"Abs Error (Max:{max_err_p:.2e})")
                    for j_p in range(3): axs[k_idx,j_p].set_xlabel("x"); axs[k_idx,0].set_ylabel("t")
                    plt.colorbar(im0,ax=axs[k_idx,0]); plt.colorbar(im1,ax=axs[k_idx,1]); plt.colorbar(im2,ax=axs[k_idx,2])
                else: axs[k_idx,0].text(0.5,0.5,"2D Plot Placeholder",ha='center')
            fig.suptitle(f"SPFNO Validation ({dataset_type.capitalize()}) @ T={T_horizon_current:.1f} - Modes:{model.modes1}, Width:{model.width}")
            fig.tight_layout(rect=[0,0.03,1,0.95])
            curr_fig_path=save_fig_path_prefix+f"_T{str(T_horizon_current).replace('.','p')}.png"
            plt.savefig(curr_fig_path); print(f"  Saved SPFNO validation plot to {curr_fig_path}"); plt.show()
    
    print(f"\n--- SPFNO Validation Summary (Metrics for T={T_value_for_model_training:.1f}) ---")
    # ... (Summary printing - same as FNO)
    for key_val in state_keys_val:
        if results_primary_horizon[key_val]['mse']:
            avg_mse=np.mean(results_primary_horizon[key_val]['mse']); avg_rmse=np.sqrt(avg_mse)
            avg_rel=np.mean(results_primary_horizon[key_val]['relative_error'])
            print(f"  Var '{key_val}': Avg MSE={avg_mse:.3e}, RMSE={avg_rmse:.3e}, RelErr={avg_rel:.3e}")
    if overall_rel_err_primary_horizon: print(f"  Overall RelErr @ T={T_value_for_model_training:.1f}: {np.mean(overall_rel_err_primary_horizon):.3e}")
    print("------------------------")

# main script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run SPFNO baseline for PDE datasets.") # Corrected description
    parser.add_argument('--datatype', type=str, required=True,
                        choices=['advection', 'euler', 'burgers', 'darcy',
                                 'heat_delayed_feedback',
                                 'reaction_diffusion_neumann_feedback',
                                 'heat_nonlinear_feedback_gain',
                                 'convdiff'], # Added convdiff
                        help='Type of dataset to run SPFNO on.')
    args = parser.parse_args()
    DATASET_TYPE = args.datatype
    MODEL_TYPE = 'SPFNO'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Running {MODEL_TYPE} Baseline for {DATASET_TYPE} on {device} ---")

    # SPFNO Hyperparameters
    SPFNO_MODES = 16; SPFNO_WIDTH = 64; SPFNO_LAYERS = 4
    LEARNING_RATE = 1e-3; BATCH_SIZE = 32; NUM_EPOCHS = 80
    CLIP_GRAD_NORM = 1.0
    USE_PROJECTION_FILTER = True

    # --- Dataset Paths and Parameters ---
    dataset_params_for_plot = {}
    main_state_keys = []
    main_num_state_vars = 0
    main_transform_type = 'dirichlet' # Default, can be overridden
    FULL_T_IN_DATAFILE = 2.0; FULL_NT_IN_DATAFILE = 300 # MUST MATCH GENERATION
    TRAIN_T_TARGET = 1.5  

    if DATASET_TYPE == 'reaction_diffusion_neumann_feedback':
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
            if isinstance(file_T_check_val, (int,float)): dataset_params_for_plot['T'] = file_T_check_val
            if isinstance(file_L_check_val, (int,float)): dataset_params_for_plot['L'] = file_L_check_val
            dataset_params_for_plot['nx'] = file_nx_check_val
    except FileNotFoundError: print(f"Error: File not found {dataset_path}"); exit()
    if not data_list_all: print("No data. Exiting."); exit()

    random.shuffle(data_list_all)
    n_total = len(data_list_all); n_train = int(0.8 * n_total)
    train_data_list_split = data_list_all[:n_train]; val_data_list_split = data_list_all[n_train:]
    print(f"Train samples: {len(train_data_list_split)}, Validation samples: {len(val_data_list_split)}")


    train_dataset = UniversalPDEDataset(train_data_list_split, dataset_type=DATASET_TYPE, train_nt_limit=TRAIN_NT_FOR_MODEL)
    val_dataset = UniversalPDEDataset(val_data_list_split, dataset_type=DATASET_TYPE, train_nt_limit=None)

    num_workers=1 # Adjusted
    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=num_workers,pin_memory=True,drop_last=True)
    val_loader = DataLoader(val_dataset,batch_size=1,shuffle=False,num_workers=num_workers)

    actual_bc_state_dim_ds = train_dataset.bc_state_dim
    actual_num_controls_ds = train_dataset.num_controls
    spfno_input_channels = main_num_state_vars + actual_bc_state_dim_ds + actual_num_controls_ds
    spfno_output_channels = main_num_state_vars

    print(f"\nInitializing SPFNO model: Modes={SPFNO_MODES}, Width={SPFNO_WIDTH}, Transform='{main_transform_type}'")
    online_spfno_model = SPFNO1d(
        modes=SPFNO_MODES, width=SPFNO_WIDTH,
        input_channels=spfno_input_channels, output_channels=spfno_output_channels,
        num_layers=SPFNO_LAYERS, transform_type=main_transform_type,
        use_projection_filter=USE_PROJECTION_FILTER
    )
    online_spfno_model.nx = dataset_params_for_plot['nx']

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{DATASET_TYPE}_{MODEL_TYPE}_trainT{TRAIN_T_TARGET}_m{SPFNO_MODES}_w{SPFNO_WIDTH}_{timestamp}" # Added timestamp
    checkpoint_dir = f"./New_ckpt_2/checkpoints_{DATASET_TYPE}_{MODEL_TYPE}"
    results_dir = f"./result_all_2/results_{DATASET_TYPE}_{MODEL_TYPE}"
    os.makedirs(checkpoint_dir,exist_ok=True); os.makedirs(results_dir,exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'model_{run_name}.pt')
    save_fig_path_prefix = os.path.join(results_dir, f'result_{run_name}')

    print(f"\nStarting training for SPFNO on {DATASET_TYPE}...")
    start_train_time = time.time()
    online_spfno_model = train_spfno_stepper(
        online_spfno_model, train_loader, dataset_type=DATASET_TYPE,
        train_nt_for_model=TRAIN_NT_FOR_MODEL,
        lr=LEARNING_RATE, num_epochs=NUM_EPOCHS, device=device,
        checkpoint_path=checkpoint_path, clip_grad_norm=CLIP_GRAD_NORM
    )
    end_train_time = time.time(); print(f"Training took {end_train_time - start_train_time:.2f} seconds.")

    if val_data_list_split:
        print(f"\nStarting validation for SPFNO on {DATASET_TYPE}...")
        validate_spfno_stepper(
            online_spfno_model, val_loader, dataset_type=DATASET_TYPE,
            train_nt_for_model_training=TRAIN_NT_FOR_MODEL,
            T_value_for_model_training=TRAIN_T_TARGET,
            full_T_in_datafile=FULL_T_IN_DATAFILE,
            full_nt_in_datafile=FULL_NT_IN_DATAFILE,
            dataset_params_for_plot=dataset_params_for_plot, device=device,
            save_fig_path_prefix=save_fig_path_prefix
        )
    else: print("\nNo validation data. Skipping validation.")

    print("="*60); print(f"Run finished: {run_name}"); print(f"Final checkpoint: {checkpoint_path}");
    if val_data_list_split: print(f"Validation figures saved with prefix: {save_fig_path_prefix}"); print("="*60)

