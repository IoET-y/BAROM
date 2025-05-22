# =============================================================================
#       COMPLETE CODE: SPFNO Baseline Adapted for Task
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
# 0. Fourierpack Functions (Retained from your SPFNO code)
# =============================================================================
# def sin_transform(u): # DST-I (scaled)
#     Nx = u.shape[-1]
#     if Nx <= 1: return u.clone() # Or handle appropriately (e.g. return zeros of certain shape)
#     # DST-I is related to FFT of odd extension. For u of length N (0 to N-1), extend to 2N.
#     # Input to FFT for DST-I is typically [0, u_0, u_1, ..., u_{N-1}, 0, -u_{N-1}, ..., -u_1]
#     # The provided V construction is for FFT-based DCT/DST where N is number of points (Nx)
#     # For DST-I (sin_transform for Dirichlet), typically input function is 0 at boundaries.
#     # If u includes boundaries, they should be 0. If u is interior points of length N-2:
#     #   V = [0, u_0, ..., u_{N-3}, 0, -u_{N-3}, ..., -u_0] for FFT of length 2(N-1)
#     # The current implementation:
#     # u is [B, C, Nx_pts]. If Nx_pts includes boundaries which are zero, this might be okay.
#     # V = torch.cat([u, -u.flip(dims=[-1])[..., 1:Nx-1]], dim=-1) # Result is 2*Nx - 2
#     # This V is typically for DST-I of u[..., 1:-1] (interior points).
#     # If u is the full N points [u0, ..., uN-1] and u0=uN-1=0, then take u_int = u[..., 1:-1]
#     # V_int = torch.cat([u_int, -u_int.flip(dims=[-1])[..., :-1]], dim=-1) # if u_int has N-3 points
#     # For now, assume u is defined on N points, and DST-I is needed.
#     # Python scipy.fft.dst(type=1) expects N points.
#     # FFT based DST-I using `scipy.fftpack.dst`:
#     # `y = zeros(2*N); y[1:N+1] = x; imag(fft(y))[1:N+1]/(2*N)` -> Not quite matching
#     # A common way: x_odd = [0, x[0], ..., x[N-1], 0, -x[N-1], ..., -x[0]]. fft(x_odd).imag[:N]
#     # The V provided seems to be for a variant for DCT-II or DST-II
#     # Let's use a simplified RFFT for now as placeholder if issues arise,
#     # but ideally, this should match a standard DST type or be used consistently with its inverse.
#     # The scaling factor 1/(Nx-1) is also non-standard for typical DSTs.
#     # Using current implementation assuming it's consistent with isin_transform
#     if Nx < 2: return torch.zeros_like(u) # DST typically for N>=2
#     V = torch.cat([u[..., :Nx-1], -u[..., 1:].flip(dims=[-1])], dim=-1) # Trying a common form
#     a = -torch.fft.rfft(V, dim=-1)[...,1:Nx].imag / (Nx-1.0 if Nx >1 else 1.0) # Scaling to make it invertible with isin
#     return a


# def isin_transform(a, n=None): # Inverse DST-I (scaled)
#     # 'a' are the coefficients, n is the desired output length in physical space
#     Nx_coeffs = a.shape[-1] # This should be N-1 coeffs for DST-I of N points
#     N_out = n if n is not None else Nx_coeffs + 1 # If a has N-1 coeffs, output has N points

#     if N_out < 2: return torch.zeros(*a.shape[:-1], N_out, dtype=a.dtype, device=a.device)

#     # Construct the imaginary part of the sequence for IFFT
#     # For DST-I of N points, coeffs a_k for k=1...N-1
#     # Input to ifft should be [0, a_1, ..., a_{N-1}, 0, -a_{N-1}, ..., -a_1] (length 2N)
#     fft_input_imag = torch.zeros(*a.shape[:-1], 2 * (N_out-1), device=a.device, dtype=a.dtype)
#     coeffs_to_use = min(Nx_coeffs, N_out - 1)

#     fft_input_imag[..., 1:coeffs_to_use+1] = a[..., :coeffs_to_use]
#     if N_out - 1 > coeffs_to_use : # Zero padding if N_out-1 > Nx_coeffs
#         pass # Already zeros
#     if N_out > 1: # Avoid issues if N_out=1
#          fft_input_imag[..., N_out-1+1:] = -a[..., :coeffs_to_use].flip(dims=[-1])


#     fft_input_complex = torch.zeros_like(fft_input_imag, dtype=torch.complex64)
#     fft_input_complex.imag = -fft_input_imag * ( (N_out-1.0 if N_out > 1 else 1.0) / 2.0) # Invert scaling, add 0.5 for IFFT of DST

#     # Perform IFFT. The result should be real.
#     u_full = torch.fft.ifft(fft_input_complex, dim=-1).real
#     u = u_full[..., :N_out] # Take the first N_out points
#     return u


def sin_transform(u):
    Nx = u.shape[-1]
    V = torch.cat([u, -u.flip(dims=[-1])[..., 1:Nx-1]], dim=-1)
    a = -torch.fft.fft(V, dim=-1)[..., :Nx].imag / (Nx-1) # Added potential scaling
    return a

def isin_transform(a, n=None):
    Nx_coeffs = a.shape[-1]
    N_out = n if n is not None else Nx_coeffs
    # Reconstruct imaginary part of FFT input correctly for ifft
    # Need 2*(N_out-1) points for ifft based on FFT definition? Or 2*N_out?
    # Let's assume forward transform V size was 2*(N_out-1)
    fft_len = 2 * (N_out - 1)
    if fft_len <= 0: return torch.zeros(*a.shape[:-1], N_out, dtype=torch.float32, device=a.device) # Handle N=1 case

    V_imag = torch.zeros(*a.shape[:-1], fft_len, device=a.device)
    if Nx_coeffs >= 1: # Need at least one coeff (k=1)
         # Place coeffs for k=1 to N-1 (index 1 to N-1 in 'a')
         limit = min(Nx_coeffs, N_out)
         V_imag[..., 1:limit] = a[..., 1:limit]
         # Place flipped coeffs for k=N to 2N-3 (index N-1 downto 1 in 'a')
         if N_out > 1:
             V_imag[..., N_out:] = -a[..., 1:limit-1].flip(dims=[-1])

    V_complex = torch.zeros_like(V_imag, dtype=torch.complex64)
    V_complex.imag = -V_imag # Need negative imag part for ifft? Check DST-I def.
    u = torch.fft.ifft(V_complex, dim=-1)[..., :N_out].real * fft_len * 0.5 # Added potential scaling
    return u


def cos_transform(u): # DCT-I (scaled)
    Nx = u.shape[-1]
    if Nx < 2 : return u.clone() # DCT-I usually N>=2
    # V for DCT-I: [u0, u1, ..., uN-1, uN-2, ..., u1] of length 2N-2
    V = torch.cat([u, u[..., 1:Nx-1].flip(dims=[-1])], dim=-1)
    a = torch.fft.rfft(V, dim=-1).real / (Nx-1.0 if Nx > 1 else 1.0) # Scaling
    return a


def icos_transform(a, n=None): # Inverse DCT-I (scaled)
    Nx_coeffs = a.shape[-1] # Should be N coeffs for DCT-I of N points
    N_out = n if n is not None else Nx_coeffs

    if N_out < 1: return torch.zeros(*a.shape[:-1], N_out, dtype=a.dtype, device=a.device)
    if N_out == 1: return a[..., :1].clone() if Nx_coeffs >=1 else torch.zeros(*a.shape[:-1], 1, dtype=a.dtype, device=a.device)


    # Construct the real part of the sequence for IFFT
    # For DCT-I of N points, coeffs a_k for k=0...N-1
    # Input to ifft should be [a0, a1, ..., aN-1, aN-2, ..., a1] (length 2N-2)
    fft_input_real = torch.zeros(*a.shape[:-1], 2 * (N_out - 1), device=a.device, dtype=a.dtype)
    coeffs_to_use = min(Nx_coeffs, N_out)

    fft_input_real[..., :coeffs_to_use] = a[..., :coeffs_to_use]
    if N_out - 1 > coeffs_to_use: # Zero padding coeffs if N_out > Nx_coeffs
        pass
    if N_out > 1:
        fft_input_real[..., N_out:] = a[..., 1:min(coeffs_to_use, N_out-1)].flip(dims=[-1])


    fft_input_complex = torch.zeros_like(fft_input_real, dtype=torch.complex64)
    # Apply scaling inverse, factor 0.5 for ifft of DCT
    fft_input_complex.real = fft_input_real * ( (N_out-1.0 if N_out > 1 else 1.0) / 2.0)
    fft_input_complex.real[..., 0] *= 2.0 # a0 and aN-1 are not halved in some defs
    if N_out > 1: fft_input_complex.real[..., N_out-1] *=2.0


    u_full = torch.fft.ifft(fft_input_complex, dim=-1).real
    u = u_full[..., :N_out]
    return u


def WSWA(u):
    return torch.fft.rfft(u) # Placeholder

def iWSWA(a, n=None):
    N_out = n if n is not None else ( (a.shape[-1]-1)*2 if a.shape[-1]>0 else 0 )
    if N_out == 0 and a.shape[-1] == 1 : N_out = 1 # Handle single coeff case for rfft
    return torch.fft.irfft(a, n=N_out)


# =============================================================================
# 1. DATASET CLASS (UniversalPDEDataset - Corrected Version)
# =============================================================================
# Using the same corrected UniversalPDEDataset as for FNO/DeepONet
class UniversalPDEDataset(Dataset):
    def __init__(self, data_list, dataset_type, train_nt_limit=None):
        if not data_list: raise ValueError("data_list cannot be empty")
        self.data_list = data_list; self.dataset_type = dataset_type.lower(); self.train_nt_limit = train_nt_limit
        first_sample = data_list[0]; params = first_sample.get('params', {})
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
        if actual_bc_dim != self.expected_bc_state_dim: print(f"Warn: BC dim mismatch {self.dataset_type}. Exp {self.expected_bc_state_dim}, got {actual_bc_dim}")
        self.bc_state_dim = actual_bc_dim # Use actual
        self.bc_control_key='BC_Control'
        if self.bc_control_key in first_sample and first_sample[self.bc_control_key] is not None and first_sample[self.bc_control_key].size>0: self.num_controls = first_sample[self.bc_control_key].shape[1]
        else: self.num_controls=0
    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx):
        sample=self.data_list[idx]; norm_factors={}; current_nt=self.effective_nt_for_loader; slist=[]
        for key in self.state_keys:
            s_full=sample[key]; s_seq=s_full[:current_nt,...]; s_mean=np.mean(s_seq); s_std=np.std(s_seq)+1e-8
            slist.append(torch.tensor((s_seq-s_mean)/s_std).float()); norm_factors[f'{key}_mean']=s_mean; norm_factors[f'{key}_std']=s_std
        bcs_full=sample[self.bc_state_key]; bcs_seq=bcs_full[:current_nt,:]; bcs_norm=np.zeros_like(bcs_seq,dtype=np.float32)
        norm_factors[f'{self.bc_state_key}_means']=np.zeros(self.bc_state_dim); norm_factors[f'{self.bc_state_key}_stds']=np.ones(self.bc_state_dim)
        for d in range(self.bc_state_dim):
            col=bcs_seq[:,d]; m=np.mean(col); s=np.std(col)
            if s>1e-8: bcs_norm[:,d]=(col-m)/s; norm_factors[f'{self.bc_state_key}_means'][d]=m; norm_factors[f'{self.bc_state_key}_stds'][d]=s
            else: bcs_norm[:,d]=0.0; norm_factors[f'{self.bc_state_key}_means'][d]=m
        bcs_tensor=torch.tensor(bcs_norm).float()
        if self.num_controls>0:
            bcc_full=sample[self.bc_control_key]; bcc_seq=bcc_full[:current_nt,:]; bcc_norm=np.zeros_like(bcc_seq,dtype=np.float32)
            norm_factors[f'{self.bc_control_key}_means']=np.zeros(self.num_controls); norm_factors[f'{self.bc_control_key}_stds']=np.ones(self.num_controls)
            for d in range(self.num_controls):
                col=bcc_seq[:,d]; m=np.mean(col); s=np.std(col)
                if s>1e-8: bcc_norm[:,d]=(col-m)/s; norm_factors[f'{self.bc_control_key}_means'][d]=m; norm_factors[f'{self.bc_control_key}_stds'][d]=s
                else: bcc_norm[:,d]=0.0; norm_factors[f'{self.bc_control_key}_means'][d]=m
            bcc_tensor=torch.tensor(bcc_norm).float()
        else: bcc_tensor=torch.empty((current_nt,0),dtype=torch.float32)
        bc_ctrl_tensor=torch.cat((bcs_tensor,bcc_tensor),dim=-1)
        out_state = slist[0] if self.num_state_vars==1 else slist
        return out_state, bc_ctrl_tensor, norm_factors

# =============================================================================
# FNO Components (SpectralConv1d - unchanged from FNO/SPFNO code)
# =============================================================================
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super().__init__(); self.in_channels=in_channels; self.out_channels=out_channels; self.modes1=modes1
        self.scale=(1/(in_channels*out_channels)); self.weights1=nn.Parameter(self.scale*torch.rand(in_channels,out_channels,self.modes1,dtype=torch.cfloat))
    def compl_mul1d(self, input_tensor, weights): return torch.einsum("bix,iox->box", input_tensor, weights)
    def forward(self, x):
        B=x.shape[0]; x_ft=torch.fft.rfft(x); out_ft=torch.zeros(B,self.out_channels,x.size(-1)//2+1,device=x.device,dtype=torch.cfloat)
        out_ft[:,:,:self.modes1]=self.compl_mul1d(x_ft[:,:,:self.modes1],self.weights1); return torch.fft.irfft(out_ft,n=x.size(-1))

# =============================================================================
# SPFNO Specific Components
# =============================================================================
# class ProjectionFilter1d(nn.Module):
#     def __init__(self, transform_type='dirichlet'):
#         super().__init__()
#         self.transform_type = transform_type
#         if transform_type == 'dirichlet': self.fwd_func=sin_transform; self.inv_func=isin_transform
#         elif transform_type == 'neumann': self.fwd_func=cos_transform; self.inv_func=icos_transform
#         elif transform_type == 'mixed': self.fwd_func=WSWA; self.inv_func=iWSWA
#         else: self.fwd_func=lambda x:torch.fft.rfft(x); self.inv_func=lambda x,n:torch.fft.irfft(x,n=n) # Default to RFFT/IRFFT for periodic/None
#         if transform_type not in ['dirichlet', 'neumann', 'mixed', 'periodic', None]:
#             print(f"ProjectionFilter: Unknown type '{transform_type}'. Using RFFT/IRFFT.")

#     def forward(self, x): # x shape: [B, C, N]
#         N = x.shape[-1]
#         try:
#             x_transformed = self.fwd_func(x)
#             x_reconstructed = self.inv_func(x_transformed, n=N)
#             if x_reconstructed.shape[-1] != N: # Fallback padding/truncating
#                 if x_reconstructed.shape[-1] > N: x_reconstructed = x_reconstructed[..., :N]
#                 else: x_reconstructed = F.pad(x_reconstructed, (0, N - x_reconstructed.shape[-1]))
#         except Exception as e:
#             print(f"Error in ProjectionFilter (type={self.transform_type}): {e}. Input shape: {x.shape}. Returning input."); return x
#         return x_reconstructed

class ProjectionFilter1d(nn.Module):
    """ Applies specific transform and inverse to project onto space satisfying BCs. """
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
            self.inv_func = iWSWA # Still needs correct implementation
        else: # Includes 'periodic' or None
            print("ProjectionFilter: No specific BC projection applied (Periodic/None).")
            # Use identity functions if no projection needed
            self.fwd_func = lambda x: x # Or RFFT if preferred for consistency
            self.inv_func = lambda x, n: x # Or IRFFT

    def forward(self, x):
        # x shape: [B, C, N]
        if not callable(self.fwd_func) or not callable(self.inv_func):
             print(f"Warning: Invalid transform functions for type '{self.transform_type}'. Skipping filter.")
             return x

        N = x.shape[-1]
        try:
            # Assume forward transform outputs compatible shape for inverse
            x_transformed = self.fwd_func(x)
            # Inverse might need original size 'n'
            x_reconstructed = self.inv_func(x_transformed, n=N)
            # Ensure output size matches input size N
            if x_reconstructed.shape[-1] != N:
                 print(f"Warning: Projection filter output size {x_reconstructed.shape[-1]} != input size {N}. Padding/Truncating.")
                 # Simple padding/truncating - might not be correct
                 if x_reconstructed.shape[-1] > N:
                     x_reconstructed = x_reconstructed[..., :N]
                 else:
                     pad_width = N - x_reconstructed.shape[-1]
                     x_reconstructed = F.pad(x_reconstructed, (0, pad_width))

        except Exception as e:
             print(f"Error during ProjectionFilter transform (type={self.transform_type}): {e}")
             print(f"Input shape: {x.shape}")
             return x # Return input as fallback

        return x_reconstructed



class SPFNO1d(nn.Module):
    def __init__(self, modes, width, input_channels, output_channels, num_layers=4, transform_type='dirichlet', use_projection_filter=True):
        super().__init__(); self.modes1=modes; self.width=width; self.input_channels=input_channels; self.output_channels=output_channels; self.num_layers=num_layers; self.transform_type=transform_type; self.use_projection_filter=use_projection_filter
        self.fc0=nn.Linear(input_channels,self.width); self.convs=nn.ModuleList(); self.ws=nn.ModuleList()
        for _ in range(num_layers): self.convs.append(SpectralConv1d(self.width,self.width,self.modes1)); self.ws.append(nn.Conv1d(self.width,self.width,1))
        self.fc1=nn.Linear(self.width,128); self.fc2=nn.Linear(128,output_channels)
        if self.use_projection_filter and self.transform_type not in ['periodic', None]: self.proj_filter=ProjectionFilter1d(transform_type=self.transform_type)
        else: self.proj_filter=None;
        if self.use_projection_filter and self.transform_type in ['periodic', None]: print("SPFNO Info: Projection filter not used for periodic/None.")
        self.nx = None # Will be set in main based on data

    def forward(self, x): # x shape: [B, N, C_in]
        x_lifted=self.fc0(x); x_permuted=x_lifted.permute(0,2,1); x_proc=x_permuted # x_proc is [B, W, N]
        for i in range(self.num_layers):
            x1=self.convs[i](x_proc); x2=self.ws[i](x_proc); x_proc=x1+x2; x_proc=F.gelu(x_proc)
        x_out_perm=x_proc.permute(0,2,1); x_out=self.fc1(x_out_perm); x_out=F.gelu(x_out); x_out=self.fc2(x_out) # [B, N, C_out]
        if self.proj_filter is not None:
            x_proj_in = x_out.permute(0,2,1) # [B, C_out, N]
            x_proj_out = self.proj_filter(x_proj_in)
            x_out = x_proj_out.permute(0,2,1) # [B, N, C_out]
        return x_out

# =============================================================================
# SPFNO Training Function (Adapted for TRAIN_NT_FOR_MODEL)
# =============================================================================
def train_spfno_stepper(model, data_loader, dataset_type, train_nt_for_model, # Added
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
        model.train(); epoch_loss=0.0; num_batches=0; batch_start_time=time.time()
        for i, (state_data_loaded, BC_Ctrl_tensor_loaded, _) in enumerate(data_loader):
            if isinstance(state_data_loaded, list): state_seq_true_train=torch.stack(state_data_loaded,dim=-1).to(device)
            else: state_seq_true_train=state_data_loaded.unsqueeze(-1).to(device)
            BC_Ctrl_seq_train=BC_Ctrl_tensor_loaded.to(device)
            B,nt_loaded,nx_or_N,_ = state_seq_true_train.shape
            if nt_loaded != train_nt_for_model: raise ValueError(f"Data nt {nt_loaded} != train_nt {train_nt_for_model}")

            optimizer.zero_grad(); total_sequence_loss=torch.tensor(0.0,device=device,requires_grad=True)
            for t in range(train_nt_for_model - 1):
                u_n_true=state_seq_true_train[:,t,:,:]; bc_ctrl_n=BC_Ctrl_seq_train[:,t,:]
                bc_ctrl_n_expanded=bc_ctrl_n.unsqueeze(1).repeat(1,nx_or_N,1)
                spfno_input=torch.cat((u_n_true,bc_ctrl_n_expanded),dim=-1)
                u_np1_pred=model(spfno_input)
                u_np1_true=state_seq_true_train[:,t+1,:,:]
                step_loss=mse_loss(u_np1_pred,u_np1_true); total_sequence_loss=total_sequence_loss+step_loss
            
            current_batch_loss=total_sequence_loss/(train_nt_for_model-1)
            epoch_loss+=current_batch_loss.item(); num_batches+=1
            current_batch_loss.backward()
            if clip_grad_norm>0: torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=clip_grad_norm)
            optimizer.step()
            if (i+1)%50==0:
                elapsed=time.time()-batch_start_time
                print(f" SPFNO Ep {epoch+1} B {i+1}/{len(data_loader)}, Loss: {current_batch_loss.item():.3e}, Time/50B: {elapsed:.2f}s")
                batch_start_time=time.time()
        avg_epoch_loss = epoch_loss/max(num_batches,1)
        print(f"SPFNO Epoch {epoch+1}/{num_epochs} Avg Loss: {avg_epoch_loss:.6f}")
        scheduler.step(avg_epoch_loss)
        if avg_epoch_loss < best_loss:
            best_loss=avg_epoch_loss; print(f"Saving SPFNO ckpt loss {best_loss:.6f}")
            torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'loss':best_loss,'dataset_type':dataset_type, 'modes':model.modes1, 'width':model.width, 'transform_type':model.transform_type},checkpoint_path)
    print("SPFNO Training finished.")
    if os.path.exists(checkpoint_path): print(f"Loading best SPFNO model"); ckpt=torch.load(checkpoint_path,map_location=device); model.load_state_dict(ckpt['model_state_dict'])
    return model

# =============================================================================
# SPFNO Validation Function (Autoregressive Rollout for Multiple Horizons)
# =============================================================================
def validate_spfno_stepper(model, data_loader, dataset_type,
                           train_nt_for_model_training: int, T_value_for_model_training: float,
                           full_T_in_datafile: float, full_nt_in_datafile: int,
                           dataset_params_for_plot: dict, device='cuda',
                           save_fig_path_prefix='spfno_result'):
    model.eval()
    if dataset_type in ['advection', 'burgers']: state_keys_val = ['U']
    elif dataset_type == 'euler': state_keys_val = ['rho', 'u']
    elif dataset_type == 'darcy': state_keys_val = ['P']
    else: raise ValueError(f"Unknown type '{dataset_type}' in SPFNO validation")
    num_state_vars_val = len(state_keys_val)

    test_horizons_T_values = [T_value_for_model_training]
    if full_T_in_datafile > T_value_for_model_training:
        test_horizons_T_values.append(T_value_for_model_training + 0.5 * (full_T_in_datafile - T_value_for_model_training))
        test_horizons_T_values.append(full_T_in_datafile)
    test_horizons_T_values = sorted(list(set(h for h in test_horizons_T_values if h <= full_T_in_datafile + 1e-6)))
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
        nt_file_check, nx_or_N_plot, num_vars_check_val = state_seq_true_norm_full.shape

        norm_factors_sample = {}
        for key, val_tensor in norm_factors_batch.items():
            norm_factors_sample[key] = val_tensor[0].cpu().numpy() if isinstance(val_tensor,torch.Tensor) and val_tensor.ndim>0 else (val_tensor.cpu().numpy() if isinstance(val_tensor,torch.Tensor) else val_tensor)

        u_initial_norm = state_seq_true_norm_full[0:1, :, :]

        for T_horizon_current in test_horizons_T_values:
            nt_for_rollout = int((T_horizon_current/full_T_in_datafile)*(full_nt_in_datafile-1))+1
            nt_for_rollout = min(nt_for_rollout, full_nt_in_datafile)
            print(f"\n  Rollout for T_horizon = {T_horizon_current:.2f} (nt = {nt_for_rollout})")

            u_pred_seq_norm_horizon = torch.zeros(nt_for_rollout, nx_or_N_plot, num_vars_check_val, device=device)
            u_current_pred_step = u_initial_norm.clone()
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
            
            # Denormalize & Metrics
            U_pred_denorm_h={}; U_gt_denorm_h={}
            pred_list_h=[]; gt_list_h=[]
            state_true_norm_sliced_h = state_seq_true_norm_full[:nt_for_rollout,:,:]
            pred_np_h = u_pred_seq_norm_horizon.cpu().numpy()
            gt_np_h = state_true_norm_sliced_h.cpu().numpy()

            for k_idx, key_val in enumerate(state_keys_val):
                mean_k=norm_factors_sample[f'{key_val}_mean']; std_k=norm_factors_sample[f'{key_val}_std']
                mean_k = mean_k.item() if hasattr(mean_k,'item') else mean_k
                std_k = std_k.item() if hasattr(std_k,'item') else std_k
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
            fig_L=dataset_params_for_plot.get('L',1.0); fig_nx=dataset_params_for_plot.get('nx',nx_or_N_plot); fig_ny=dataset_params_for_plot.get('ny',1)
            for k_idx, key_val in enumerate(state_keys_val):
                gt_p=U_gt_denorm_h[key_val]; pred_p=U_pred_denorm_h[key_val]; diff_p=np.abs(pred_p-gt_p)
                max_err_p=np.max(diff_p) if diff_p.size>0 else 0
                vmin_p=min(np.min(gt_p),np.min(pred_p)) if gt_p.size>0 else 0; vmax_p=max(np.max(gt_p),np.max(pred_p)) if gt_p.size>0 else 1
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
    for key_val in state_keys_val:
        if results_primary_horizon[key_val]['mse']:
            avg_mse=np.mean(results_primary_horizon[key_val]['mse']); avg_rmse=np.sqrt(avg_mse)
            avg_rel=np.mean(results_primary_horizon[key_val]['relative_error'])
            print(f"  Var '{key_val}': Avg MSE={avg_mse:.3e}, RMSE={avg_rmse:.3e}, RelErr={avg_rel:.3e}")
    if overall_rel_err_primary_horizon: print(f"  Overall RelErr @ T={T_value_for_model_training:.1f}: {np.mean(overall_rel_err_primary_horizon):.3e}")
    print("------------------------")


# =============================================================================
# Main Block - SPFNO
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate PDE datasets with complex BCs.")
    parser.add_argument('--datatype', type=str, required=True, choices=['advection', 'euler', 'burgers', 'darcy'], help='Type of dataset to generate.')
    args = parser.parse_args()

    DATASET_TYPE = args.datatype # 'advection', 'euler', 'burgers', 'darcy'
    MODEL_TYPE = 'SPFNO'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Running SPFNO Baseline for {DATASET_TYPE} on {device} ---")

    FULL_T_IN_DATAFILE = 2.0; FULL_NT_IN_DATAFILE = 600
    TRAIN_T_TARGET = 1.0
    TRAIN_NT_FOR_MODEL = int((TRAIN_T_TARGET/FULL_T_IN_DATAFILE)*(FULL_NT_IN_DATAFILE-1))+1
    print(f"Datafile T={FULL_T_IN_DATAFILE}, nt={FULL_NT_IN_DATAFILE}")
    print(f"Training T={TRAIN_T_TARGET}, nt={TRAIN_NT_FOR_MODEL}")

    # SPFNO Hyperparameters
    SPFNO_MODES = 16; SPFNO_WIDTH = 64; SPFNO_LAYERS = 4
    LEARNING_RATE = 1e-3; BATCH_SIZE = 32; NUM_EPOCHS = 80 # Adjusted for potentially faster SPFNO training
    CLIP_GRAD_NORM = 1.0
    USE_PROJECTION_FILTER = True # Key SPFNO feature

    dataset_params_for_plot = {}
    if DATASET_TYPE == 'advection':
        dataset_path = "./datasets_full/advection_data_10000s_128nx_600nt.pkl"
        main_state_keys=['U']; main_num_state_vars=1; main_bc_state_dim=2; main_transform_type='dirichlet'
        dataset_params_for_plot={'nx':128,'ny':1,'L':1.0,'T':FULL_T_IN_DATAFILE}
    elif DATASET_TYPE == 'euler':
        dataset_path = "./datasets_full/euler_data_10000s_128nx_600nt.pkl"
        main_state_keys=['rho','u']; main_num_state_vars=2; main_bc_state_dim=4; main_transform_type='dirichlet' # Or 'mixed' if more appropriate for Euler BCs in your setup
        dataset_params_for_plot={'nx':128,'ny':1,'L':1.0,'T':FULL_T_IN_DATAFILE}
    elif DATASET_TYPE == 'burgers':
        dataset_path = "./datasets_full/burgers_data_10000s_128nx_600nt.pkl"
        main_state_keys=['U']; main_num_state_vars=1; main_bc_state_dim=2; main_transform_type='mixed'
        dataset_params_for_plot={'nx':128,'ny':1,'L':1.0,'T':FULL_T_IN_DATAFILE}
    elif DATASET_TYPE == 'darcy':
        dataset_path = "./datasets_full/darcy_data_10000s_128nx_600nt.pkl"
        main_state_keys=['P']; main_num_state_vars=1; main_bc_state_dim=2; main_transform_type='dirichlet'
        dataset_params_for_plot={'nx':128,'ny':1,'L':1.0,'T':FULL_T_IN_DATAFILE}
    else: raise ValueError(f"Unknown dataset type: {DATASET_TYPE}")

    print(f"Loading dataset: {dataset_path}")
    try:
        with open(dataset_path,'rb') as f: data_list_all=pickle.load(f)
        print(f"Loaded {len(data_list_all)} samples.")
        # File param verification can be added here
    except FileNotFoundError: print(f"Error: File not found {dataset_path}"); exit()
    if not data_list_all: print("No data. Exiting."); exit()

    # Update dataset_params_for_plot with actual file values if available
    if data_list_all and 'params' in data_list_all[0]:
        file_params = data_list_all[0]['params']
        dataset_params_for_plot['L'] = file_params.get('L', dataset_params_for_plot['L'])
        dataset_params_for_plot['nx'] = file_params.get('nx', dataset_params_for_plot['nx'])
        # Ensure FULL_NT_IN_DATAFILE matches actual data if not hardcoded correctly
        actual_file_nt = data_list_all[0][main_state_keys[0]].shape[0]
        if actual_file_nt != FULL_NT_IN_DATAFILE:
            print(f"WARNING: FULL_NT_IN_DATAFILE ({FULL_NT_IN_DATAFILE}) differs from actual file nt ({actual_file_nt}). Using actual file nt.")
            FULL_NT_IN_DATAFILE = actual_file_nt
            TRAIN_NT_FOR_MODEL = int((TRAIN_T_TARGET/FULL_T_IN_DATAFILE)*(FULL_NT_IN_DATAFILE-1))+1 # Recalculate


    random.shuffle(data_list_all)
    n_total = len(data_list_all); n_train = int(0.8 * n_total)
    train_data_list_split = data_list_all[:n_train]; val_data_list_split = data_list_all[n_train:]

    train_dataset = UniversalPDEDataset(train_data_list_split, dataset_type=DATASET_TYPE, train_nt_limit=TRAIN_NT_FOR_MODEL)
    val_dataset = UniversalPDEDataset(val_data_list_split, dataset_type=DATASET_TYPE, train_nt_limit=None) # Full length for validation

    num_workers=1
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
    online_spfno_model.nx = dataset_params_for_plot['nx'] # Set nx for SPFNO from data params

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{DATASET_TYPE}_{MODEL_TYPE}_trainT{TRAIN_T_TARGET}_m{SPFNO_MODES}_w{SPFNO_WIDTH}"
    checkpoint_dir = f"./New_ckpt_2/checkpoints_{DATASET_TYPE}_{MODEL_TYPE}" # Simplified path
    results_dir = f"./result_all_2/results_{DATASET_TYPE}_{MODEL_TYPE}"      # Simplified path
    
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