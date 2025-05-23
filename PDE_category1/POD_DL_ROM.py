# POD_DL_ROM
# =============================================================================
#    POD_DL_ROM (Adapted for Task1) ref:https://github.com/stefaniafresca/POD-DL-ROM
# POD-DL-ROM: Enhancing deep learning-based reduced order models for nonlinear parametrized PDEs by proper orthogonal decomposition, Computer Methods in Applied Mechanics and Engineering
# =============================================================================
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
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
torch.backends.cudnn.deterministic = True

print(f"POD-DL-ROM Script (Task Adapted) started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")


class UniversalPDEDataset(Dataset):
    def __init__(self, data_list, dataset_type, train_nt_limit=None): # Added train_nt_limit
        if not data_list:
            raise ValueError("data_list cannot be empty")
        self.data_list = data_list
        self.dataset_type = dataset_type.lower()
        self.train_nt_limit = train_nt_limit

        first_sample = data_list[0]
        params = first_sample.get('params', {})

        self.nt_from_sample_file = 0
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
            self.nx_from_sample_file = params.get('nx', first_sample['P'].shape[1])
            self.ny_from_sample_file = params.get('ny', 1)
            self.state_keys = ['P']; self.num_state_vars = 1
            self.expected_bc_state_dim = 2
        else:
            raise ValueError(f"Unknown dataset_type: {self.dataset_type}")

        self.effective_nt_for_loader = self.train_nt_limit if self.train_nt_limit is not None else self.nt_from_sample_file

        self.nx = self.nx_from_sample_file
        self.ny = self.ny_from_sample_file
        self.spatial_dim = self.nx * self.ny

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

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]; norm_factors = {}
        current_nt_for_item = self.effective_nt_for_loader; state_tensors_norm_list = []
        for key in self.state_keys:
            try: state_seq_full = sample[key]; state_seq = state_seq_full[:current_nt_for_item, ...]
            except KeyError: raise KeyError(f"State key '{key}' not found for {self.dataset_type}")
            if state_seq.shape[0]!=current_nt_for_item: raise ValueError(f"Time dim mismatch for {key}")
            state_mean = np.mean(state_seq); state_std = np.std(state_seq)+1e-8
            state_tensors_norm_list.append(torch.tensor((state_seq-state_mean)/state_std).float())
            norm_factors[f'{key}_mean']=state_mean; norm_factors[f'{key}_std']=state_std
        bc_state_seq_full=sample[self.bc_state_key]; bc_state_seq=bc_state_seq_full[:current_nt_for_item,:]
        if bc_state_seq.shape[0]!=current_nt_for_item: raise ValueError("Time dim mismatch for BC_State")
        bc_state_norm=np.zeros_like(bc_state_seq,dtype=np.float32)
        norm_factors[f'{self.bc_state_key}_means']=np.mean(bc_state_seq,axis=0,keepdims=True).squeeze() if bc_state_seq.size>0 else np.zeros(self.bc_state_dim)
        norm_factors[f'{self.bc_state_key}_stds']=np.ones(self.bc_state_dim)
        if bc_state_seq.size>0:
            for d in range(self.bc_state_dim):
                col=bc_state_seq[:,d]; m=np.mean(col); s=np.std(col)
                if s>1e-8: bc_state_norm[:,d]=(col-m)/s; norm_factors[f'{self.bc_state_key}_stds'][d]=s
                else: bc_state_norm[:,d]=col-m
                norm_factors[f'{self.bc_state_key}_means'][d]=m
        bcs_tensor=torch.tensor(bc_state_norm).float()
        if self.num_controls > 0:
            try:
                bcc_full=sample[self.bc_control_key]; bcc_seq=bcc_full[:current_nt_for_item,:]
                if bcc_seq.shape[0]!=current_nt_for_item: raise ValueError("Time dim mismatch for BC_Control")
                if bcc_seq.shape[1]!=self.num_controls: raise ValueError("Control dim mismatch")
                bcc_norm=np.zeros_like(bcc_seq,dtype=np.float32)
                norm_factors[f'{self.bc_control_key}_means']=np.mean(bcc_seq,axis=0,keepdims=True).squeeze() if bcc_seq.size>0 else np.zeros(self.num_controls)
                norm_factors[f'{self.bc_control_key}_stds']=np.ones(self.num_controls)
                if bcc_seq.size>0:
                    for d in range(self.num_controls):
                        col=bcc_seq[:,d];m=np.mean(col);s=np.std(col)
                        if s>1e-8: bcc_norm[:,d]=(col-m)/s; norm_factors[f'{self.bc_control_key}_stds'][d]=s
                        else: bcc_norm[:,d]=col-m
                        norm_factors[f'{self.bc_control_key}_means'][d]=m
                bcc_tensor=torch.tensor(bcc_norm).float()
            except KeyError: bcc_tensor=torch.zeros((current_nt_for_item,self.num_controls),dtype=torch.float32); norm_factors[f'{self.bc_control_key}_means']=np.zeros(self.num_controls); norm_factors[f'{self.bc_control_key}_stds']=np.ones(self.num_controls)
        else: bcc_tensor=torch.empty((current_nt_for_item,0),dtype=torch.float32)
        bc_ctrl_tensor=torch.cat((bcs_tensor,bcc_tensor),dim=-1)
        out_state = state_tensors_norm_list[0] if self.num_state_vars==1 else state_tensors_norm_list
        return out_state, bc_ctrl_tensor, norm_factors


def compute_pod_basis_generic(data_list, dataset_type, state_variable_key,
                              nx, nt, basis_dim, max_snapshots_pod=100):
    snapshots = []; count = 0
    current_nx = nx; lin_interp = np.linspace(0,1,current_nx)[np.newaxis,:]
    print(f"  Computing POD for '{state_variable_key}' using {nt} timesteps, linear interp U_B...")
    for sample_idx, sample in enumerate(data_list):
        if count >= max_snapshots_pod: break
        if state_variable_key not in sample: continue
        U_seq_full = sample[state_variable_key]
        U_seq = U_seq_full[:nt, :] # <<< Ensured truncation for POD data collection


        if U_seq.shape[0] != nt or U_seq.shape[1] != current_nx: continue
        bc_left_val = U_seq[:,0:1]; bc_right_val = U_seq[:,-1:]
        if np.isnan(bc_left_val).any() or np.isinf(bc_left_val).any() or \
           np.isnan(bc_right_val).any() or np.isinf(bc_right_val).any(): continue
        U_B = bc_left_val*(1-lin_interp)+bc_right_val*lin_interp
        U_star = U_seq - U_B; snapshots.append(U_star); count+=1
    if not snapshots: print(f"Error: No valid POD snapshots for '{state_variable_key}'."); return None
    try: snapshots = np.concatenate(snapshots,axis=0)
    except ValueError: # Simplified error handling for brevity
        print(f"Error concatenating POD snapshots for '{state_variable_key}'."); return None
    if np.isnan(snapshots).any() or np.isinf(snapshots).any(): snapshots = np.nan_to_num(snapshots)
    if np.all(np.abs(snapshots)<1e-12): print(f"Error: All POD snapshots zero for '{state_variable_key}'."); return None
    U_mean = np.mean(snapshots,axis=0,keepdims=True); U_centered = snapshots-U_mean
    try:
        _, S, Vh = np.linalg.svd(U_centered,full_matrices=False)
        rank = np.sum(S>1e-10); actual_basis_dim = min(basis_dim,rank,current_nx)
        if actual_basis_dim==0: print(f"Error: Data rank zero for '{state_variable_key}'."); return None
        if actual_basis_dim<basis_dim: print(f"Warn: POD basis_dim {basis_dim} > rank ~{rank} for '{state_variable_key}'. Using {actual_basis_dim}.")
        basis = Vh[:actual_basis_dim,:].T
    except Exception as e: print(f"SVD failed for '{state_variable_key}': {e}"); return None
    basis_norms=np.linalg.norm(basis,axis=0); basis_norms[basis_norms<1e-10]=1.0; basis/=basis_norms[np.newaxis,:]
    if actual_basis_dim<basis_dim:
        padding=np.zeros((current_nx,basis_dim-actual_basis_dim)); basis=np.hstack((basis,padding))
    print(f"  POD basis computed for '{state_variable_key}', shape {basis.shape}.")
    return basis.astype(np.float32)

# 1. Convolutional Autoencoder Components

class Encoder(nn.Module):
    def __init__(self, input_channels, N_pod, n_latent, conv_channels=[16,32,64], fc_layers=[128]):
        super().__init__(); self.N_pod=N_pod; self.input_channels=input_channels; self.n_latent=n_latent
        if int(np.sqrt(N_pod))**2 != N_pod: raise ValueError(f"N_pod ({N_pod}) must be perfect square.")
        self.spatial_dim_sqrt=int(np.sqrt(N_pod)); layers=[]; in_ch=input_channels; current_spatial_dim=self.spatial_dim_sqrt
        for out_ch in conv_channels:
            layers.append(nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=2,padding=1)); layers.append(nn.ReLU(inplace=True))
            in_ch=out_ch; current_spatial_dim=(current_spatial_dim+1)//2
        self.start_spatial_dim=current_spatial_dim; self.final_conv_channels=in_ch; layers.append(nn.Flatten())
        fc_input_dim=in_ch*current_spatial_dim*current_spatial_dim; current_dim=fc_input_dim
        for hidden_dim in fc_layers: layers.append(nn.Linear(current_dim,hidden_dim)); layers.append(nn.ReLU(inplace=True)); current_dim=hidden_dim
        layers.append(nn.Linear(current_dim,n_latent)); self.encoder_net=nn.Sequential(*layers); self._fc_input_dim=fc_input_dim
    def forward(self,x_N): # x_N: [B, d, N_pod]
        B=x_N.shape[0]
        if x_N.shape[1]!=self.input_channels or x_N.shape[2]!=self.N_pod: raise ValueError(f"Encoder input. Expected [B,{self.input_channels},{self.N_pod}], got {x_N.shape}")
        x_reshaped=x_N.view(B,self.input_channels,self.spatial_dim_sqrt,self.spatial_dim_sqrt)
        return self.encoder_net(x_reshaped)

class Decoder(nn.Module):
    def __init__(self,n_latent,N_pod,output_channels,encoder_fc_input_dim,encoder_conv_channels=[16,32,64],fc_layers=[128],encoder_spatial_dim_start=None):
        super().__init__(); self.n_latent=n_latent; self.N_pod=N_pod; self.output_channels=output_channels
        if int(np.sqrt(N_pod))**2!=N_pod: raise ValueError("N_pod must be perfect square.")
        self.spatial_dim_sqrt=int(np.sqrt(N_pod)); self.start_spatial_dim=encoder_spatial_dim_start
        self.decoder_fc_input_dim=encoder_fc_input_dim; self.start_channels=encoder_conv_channels[-1]
        fc_decoder_layers=[]; current_dim=n_latent
        for hidden_dim in reversed(fc_layers): fc_decoder_layers.append(nn.Linear(current_dim,hidden_dim)); fc_decoder_layers.append(nn.ReLU(inplace=True)); current_dim=hidden_dim
        fc_decoder_layers.append(nn.Linear(current_dim,self.decoder_fc_input_dim)); fc_decoder_layers.append(nn.ReLU(inplace=True)); self.decoder_fc=nn.Sequential(*fc_decoder_layers)
        conv_decoder_layers=[]; source_channels_for_layers=list(reversed(encoder_conv_channels)); target_channels_for_layers=list(reversed(encoder_conv_channels[:-1]))+[self.output_channels]
        for i in range(len(source_channels_for_layers)):
            in_ch=source_channels_for_layers[i]; out_ch=target_channels_for_layers[i]
            conv_decoder_layers.append(nn.ConvTranspose2d(in_ch,out_ch,kernel_size=3,stride=2,padding=1,output_padding=1))
            if i<len(source_channels_for_layers)-1: conv_decoder_layers.append(nn.ReLU(inplace=True))
        self.decoder_conv=nn.Sequential(*conv_decoder_layers)
    def forward(self,latent):
        B=latent.shape[0]; x=self.decoder_fc(latent)
        x=x.view(B,self.start_channels,self.start_spatial_dim,self.start_spatial_dim)
        x=self.decoder_conv(x)
        if x.shape[2]!=self.spatial_dim_sqrt or x.shape[3]!=self.spatial_dim_sqrt: x=F.interpolate(x,size=(self.spatial_dim_sqrt,self.spatial_dim_sqrt),mode='bilinear',align_corners=False)
        return x.view(B,self.output_channels,self.N_pod)


class DFNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=[256,512,256], activation=nn.GELU):
        super().__init__(); layers=[]; current_dim=input_dim
        for hidden_dim in hidden_layers: layers.append(nn.Linear(current_dim,hidden_dim)); layers.append(activation()); current_dim=hidden_dim
        layers.append(nn.Linear(current_dim,output_dim)); self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)


class POD_DL_ROM(nn.Module):
    def __init__(self,V_N_dict,n_latent,dfnn_input_dim,N_pod,num_state_vars,encoder_conv_channels=[16,32,64],encoder_fc_layers=[128],decoder_fc_layers=[128],dfnn_hidden_layers=[256,512,256]):
        super().__init__(); self.state_keys=list(V_N_dict.keys()); self.num_state_vars=num_state_vars; self.N_pod=N_pod; self.n_latent=n_latent
        first_key=self.state_keys[0]; self.Nx=V_N_dict[first_key].shape[0]
        for key,V_N_val in V_N_dict.items(): # Renamed V_N to V_N_val
            if V_N_val.shape!=(self.Nx,N_pod): raise ValueError(f"Basis shape mismatch for {key}.")
            self.register_buffer(f'V_N_{key}',V_N_val)
        self.dfnn=DFNN(input_dim=dfnn_input_dim,output_dim=n_latent,hidden_layers=dfnn_hidden_layers)
        self.encoder=Encoder(input_channels=num_state_vars,N_pod=N_pod,n_latent=n_latent,conv_channels=encoder_conv_channels,fc_layers=encoder_fc_layers)
        self.decoder=Decoder(n_latent=n_latent,N_pod=N_pod,output_channels=num_state_vars,encoder_fc_input_dim=self.encoder._fc_input_dim,encoder_conv_channels=encoder_conv_channels,fc_layers=decoder_fc_layers,encoder_spatial_dim_start=self.encoder.start_spatial_dim)
    def get_VN(self,key): return getattr(self,f'V_N_{key}')
    def encode(self,u_h_dict):
        u_N_list=[]
        for key in self.state_keys:
            V_N=self.get_VN(key); u_h=u_h_dict[key];
            if u_h.dim()==3:u_h=u_h.squeeze(-1)
            u_N=torch.matmul(V_N.T,u_h.T).T; u_N_list.append(u_N)
        u_N_stacked=torch.stack(u_N_list,dim=1)
        return self.encoder(u_N_stacked)
    def decode_and_reconstruct(self,u_n):
        u_N_decoded_stacked=self.decoder(u_n); u_h_reconstructed_dict={}
        for i,key in enumerate(self.state_keys):
            V_N=self.get_VN(key); u_N_var=u_N_decoded_stacked[:,i,:]
            u_h_rec=torch.matmul(V_N,u_N_var.T).T
            u_h_reconstructed_dict[key]=u_h_rec.unsqueeze(-1)
        return u_h_reconstructed_dict,u_N_decoded_stacked
    def forward(self,time_param_ctrl_input):
        u_n_predicted=self.dfnn(time_param_ctrl_input)
        u_h_predicted_dict,u_N_predicted_stacked=self.decode_and_reconstruct(u_n_predicted)
        return u_h_predicted_dict,u_N_predicted_stacked


def train_pod_dl_rom(model, data_loader, dataset_type, train_nt_for_model, # Added
                     lr=1e-3, num_epochs=100, device='cuda',
                     checkpoint_path='pod_dl_rom_ckpt.pt', clip_grad_norm=1.0, omega_h=0.5):
    model.to(device); optimizer=optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=10,verbose=True)
    mse_loss=nn.MSELoss(); start_epoch=0; best_loss=float('inf')
    if os.path.exists(checkpoint_path):
        print(f"Loading POD-DL-ROM ckpt from {checkpoint_path}...");
        try:
            ckpt=torch.load(checkpoint_path,map_location=device); model.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                try: optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                except: print("Warn: POD-DL-ROM Optimizer state mismatch.")
            start_epoch=ckpt.get('epoch',0)+1; best_loss=ckpt.get('loss',float('inf'))
            print(f"Resuming POD-DL-ROM training from epoch {start_epoch}")
        except Exception as e: print(f"Error loading POD-DL-ROM ckpt: {e}. Starting fresh.")
    
    state_keys=model.state_keys

    for epoch in range(start_epoch, num_epochs):
        model.train(); epoch_loss=0.0; num_batches=0; batch_start_time=time.time()
        for i,(state_data_loaded,BC_Ctrl_tensor_loaded,_) in enumerate(data_loader):
            if isinstance(state_data_loaded,list): state_tensors_train={key:s.to(device) for key,s in zip(state_keys,state_data_loaded)}
            else: state_tensors_train={state_keys[0]:state_data_loaded.to(device)}
            BC_Ctrl_seq_train=BC_Ctrl_tensor_loaded.to(device)
            batch_size,nt_loaded,_=state_tensors_train[state_keys[0]].shape
            if nt_loaded!=train_nt_for_model: raise ValueError(f"Data nt {nt_loaded} != train_nt {train_nt_for_model}")

            optimizer.zero_grad(); total_loss_for_batch=torch.tensor(0.0,device=device,requires_grad=True) # For accumulation
            
            for t in range(train_nt_for_model): # Iterate over truncated time steps
                u_h_true_dict_t={key:state_tensors_train[key][:,t,:].unsqueeze(-1) for key in state_keys}
                BC_Ctrl_t=BC_Ctrl_seq_train[:,t,:]
                u_N_true_list=[]
                for key in state_keys:
                    V_N=model.get_VN(key); u_h_true_t_var=u_h_true_dict_t[key].squeeze(-1)
                    u_N_true_var=torch.matmul(V_N.T,u_h_true_t_var.T).T; u_N_true_list.append(u_N_true_var)
                u_N_true_stacked=torch.stack(u_N_true_list,dim=1)
                u_n_target=model.encoder(u_N_true_stacked)
                time_normalized=torch.tensor([t/(train_nt_for_model-1.0 if train_nt_for_model > 1 else 1.0)],device=device).float().unsqueeze(0).repeat(batch_size,1)
                dfnn_input=torch.cat((time_normalized,BC_Ctrl_t),dim=-1)
                if dfnn_input.shape[1]!=model.dfnn.net[0].in_features: raise ValueError(f"DFNN input dim mismatch. Expected {model.dfnn.net[0].in_features}, got {dfnn_input.shape[1]}")
                u_n_predicted=model.dfnn(dfnn_input)
                u_N_predicted_stacked=model.decoder(u_n_predicted)
                loss_rec=mse_loss(u_N_predicted_stacked,u_N_true_stacked)
                loss_int=mse_loss(u_n_predicted,u_n_target)
                step_loss=omega_h*loss_rec+(1.0-omega_h)*loss_int
                total_loss_for_batch = total_loss_for_batch + step_loss # Accumulate
            
            current_batch_loss = total_loss_for_batch / train_nt_for_model # Avg over time steps
            epoch_loss+=current_batch_loss.item(); num_batches+=1
            current_batch_loss.backward()
            if clip_grad_norm>0: torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=clip_grad_norm)
            optimizer.step()
            if (i+1)%50==0:
                elapsed=time.time()-batch_start_time
                print(f" POD-DL-ROM Ep {epoch+1} B {i+1}/{len(data_loader)}, Loss {current_batch_loss.item():.3e}, Lrec {loss_rec.item():.3e}, Lint {loss_int.item():.3e}, Time/50B {elapsed:.2f}s")
                batch_start_time=time.time()
        avg_epoch_loss=epoch_loss/max(num_batches,1)
        print(f"POD-DL-ROM Epoch {epoch+1}/{num_epochs} Avg Loss: {avg_epoch_loss:.6f}")
        scheduler.step(avg_epoch_loss)
        if avg_epoch_loss<best_loss:
            best_loss=avg_epoch_loss; print(f"Saving POD-DL-ROM ckpt loss {best_loss:.6f}")
            torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'loss':best_loss,'dataset_type':dataset_type,'N_pod':model.N_pod,'n_latent':model.n_latent},checkpoint_path)
    print("POD-DL-ROM Training finished.")
    if os.path.exists(checkpoint_path): print(f"Loading best POD-DL-ROM model"); ckpt=torch.load(checkpoint_path,map_location=device); model.load_state_dict(ckpt['model_state_dict'])
    return model


def validate_pod_dl_rom(model, data_loader, dataset_type, # Changed for multiple plots
                        train_nt_for_model_training: int, T_value_for_model_training: float,
                        full_T_in_datafile: float, full_nt_in_datafile: int,
                        dataset_params_for_plot: dict, device='cuda',
                        save_fig_path_prefix='pod_dl_rom_result'):
    model.eval()
    state_keys_val = model.state_keys # Renamed

    test_horizons_T_values = [T_value_for_model_training]
    if full_T_in_datafile > T_value_for_model_training:
        test_horizons_T_values.append(T_value_for_model_training + 0.5 * (full_T_in_datafile - T_value_for_model_training))
        test_horizons_T_values.append(full_T_in_datafile)
    test_horizons_T_values = sorted(list(set(h for h in test_horizons_T_values if h <= full_T_in_datafile + 1e-6)))
    print(f"POD-DL-ROM Validation for T_horizons: {test_horizons_T_values}")

    results_primary_horizon = {key: {'mse':[],'rmse':[],'relative_error':[],'max_error':[]} for key in state_keys_val}
    overall_rel_err_primary_horizon = []

    with torch.no_grad():
        try:
            state_data_full_loaded, BC_Ctrl_tensor_full_loaded, norm_factors_batch = next(iter(data_loader))
        except StopIteration: print("Val data_loader empty. Skipping POD-DL-ROM validation."); return

        if isinstance(state_data_full_loaded, list):
            state_tensors_true_norm_full = {key:s[0].to(device) for key,s in zip(state_keys_val,state_data_full_loaded)} # [nt_full, Nx] per key
        else:
            state_tensors_true_norm_full = {state_keys_val[0]: state_data_full_loaded[0].to(device)}
        BC_Ctrl_seq_norm_full = BC_Ctrl_tensor_full_loaded[0].to(device) # [nt_full, bc_ctrl_dim]
        
        # nt_file_check = state_tensors_true_norm_full[state_keys_val[0]].shape[0]
        # nx_plot = state_tensors_true_norm_full[state_keys_val[0]].shape[1]

        norm_factors_sample = {}
        for key, val_tensor in norm_factors_batch.items():
            norm_factors_sample[key] = val_tensor[0].cpu().numpy() if isinstance(val_tensor,torch.Tensor) and val_tensor.ndim>0 else (val_tensor.cpu().numpy() if isinstance(val_tensor,torch.Tensor) else val_tensor)

        for T_horizon_current in test_horizons_T_values:
            nt_for_this_horizon = int((T_horizon_current/full_T_in_datafile)*(full_nt_in_datafile-1))+1
            nt_for_this_horizon = min(nt_for_this_horizon, full_nt_in_datafile)
            print(f"\n  Predicting for T_horizon = {T_horizon_current:.2f} (nt = {nt_for_this_horizon})")

            u_h_pred_seq_norm_dict_horizon = {key: [] for key in state_keys_val}

            for t_eval_idx in range(nt_for_this_horizon):
                # Physical time for this evaluation step in the current horizon
                current_physical_time = (t_eval_idx / (nt_for_this_horizon -1 if nt_for_this_horizon > 1 else 1)) * T_horizon_current
                

                time_norm_for_dfnn = torch.tensor([current_physical_time / T_value_for_model_training], device=device).float().unsqueeze(0) # [1,1]
                
                # BC/Control at the actual physical time step from full sequence
                actual_t_idx_in_file = min(int(round((current_physical_time / full_T_in_datafile) * (full_nt_in_datafile - 1))), full_nt_in_datafile -1)
                BC_Ctrl_t_actual = BC_Ctrl_seq_norm_full[actual_t_idx_in_file:actual_t_idx_in_file+1, :] # [1, bc_ctrl_dim]

                dfnn_input_val = torch.cat((time_norm_for_dfnn, BC_Ctrl_t_actual), dim=-1)
                if dfnn_input_val.shape[1] != model.dfnn.net[0].in_features: raise ValueError("DFNN input dim mismatch during validation.")

                u_h_predicted_dict_t, _ = model(dfnn_input_val) # {key: [1, Nx, 1]}
                for key in state_keys_val:
                    u_h_pred_seq_norm_dict_horizon[key].append(u_h_predicted_dict_t[key])
            
            # Process predictions for this horizon
            U_pred_denorm_h={}; U_gt_denorm_h={}
            pred_list_h=[]; gt_list_h=[]

            for k_idx, key_val in enumerate(state_keys_val):
                pred_seq_norm_h_cat = torch.cat(u_h_pred_seq_norm_dict_horizon[key_val],dim=0).squeeze(1).squeeze(-1) # [nt_horizon, Nx]
                pred_np_h_var = pred_seq_norm_h_cat.cpu().numpy()
                
                gt_true_norm_sliced_h_var = state_tensors_true_norm_full[key_val][:nt_for_this_horizon, :] # [nt_horizon, Nx]
                gt_np_h_var = gt_true_norm_sliced_h_var.cpu().numpy()

                mean_k=norm_factors_sample[f'{key_val}_mean']; std_k=norm_factors_sample[f'{key_val}_std']
                mean_k=mean_k.item() if hasattr(mean_k,'item') else mean_k; std_k=std_k.item() if hasattr(std_k,'item') else std_k
                
                pred_denorm_v=pred_np_h_var*std_k+mean_k; gt_denorm_v=gt_np_h_var*std_k+mean_k
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
            fig,axs=plt.subplots(len(state_keys_val),3,figsize=(18,5*len(state_keys_val)),squeeze=False)
            fig_L=dataset_params_for_plot.get('L',1.0); fig_nx=dataset_params_for_plot.get('nx',model.Nx); fig_ny=dataset_params_for_plot.get('ny',1) # model.Nx is full spatial dim
            for k_idx,key_val in enumerate(state_keys_val):
                gt_p=U_gt_denorm_h[key_val]; pred_p=U_pred_denorm_h[key_val]; diff_p=np.abs(pred_p-gt_p)
                max_err_p=np.max(diff_p) if diff_p.size>0 else 0
                vmin_p=min(np.min(gt_p),np.min(pred_p)) if gt_p.size>0 else 0; vmax_p=max(np.max(gt_p),np.max(pred_p)) if gt_p.size>0 else 1
                is_1d_p=(fig_ny==1); plot_ext=[0,fig_L,0,T_horizon_current]
                if is_1d_p:
                    im0=axs[k_idx,0].imshow(gt_p,aspect='auto',origin='lower',vmin=vmin_p,vmax=vmax_p,extent=plot_ext,cmap='viridis'); axs[k_idx,0].set_title(f"Truth ({key_val})")
                    im1=axs[k_idx,1].imshow(pred_p,aspect='auto',origin='lower',vmin=vmin_p,vmax=vmax_p,extent=plot_ext,cmap='viridis'); axs[k_idx,1].set_title(f"POD-DL-ROM Pred ({key_val})")
                    im2=axs[k_idx,2].imshow(diff_p,aspect='auto',origin='lower',extent=plot_ext,cmap='magma'); axs[k_idx,2].set_title(f"Abs Error (Max:{max_err_p:.2e})")
                    for j_p in range(3): axs[k_idx,j_p].set_xlabel("x"); axs[k_idx,0].set_ylabel("t")
                    plt.colorbar(im0,ax=axs[k_idx,0]); plt.colorbar(im1,ax=axs[k_idx,1]); plt.colorbar(im2,ax=axs[k_idx,2])
                else: axs[k_idx,0].text(0.5,0.5,"2D Plot Placeholder",ha='center')
            fig.suptitle(f"POD-DL-ROM Validation ({dataset_type.capitalize()}) @ T={T_horizon_current:.1f} - N={model.N_pod}, n={model.n_latent}")
            fig.tight_layout(rect=[0,0.03,1,0.95]); curr_fig_path=save_fig_path_prefix+f"_T{str(T_horizon_current).replace('.','p')}.png"
            plt.savefig(curr_fig_path); print(f"  Saved POD-DL-ROM validation plot to {curr_fig_path}"); plt.show()
    
    print(f"\n--- POD-DL-ROM Validation Summary (Metrics for T={T_value_for_model_training:.1f}) ---")
    for key_val in state_keys_val:
        if results_primary_horizon[key_val]['mse']:
            avg_mse=np.mean(results_primary_horizon[key_val]['mse']); avg_rmse=np.sqrt(avg_mse)
            avg_rel=np.mean(results_primary_horizon[key_val]['relative_error'])
            print(f"  Var '{key_val}': Avg MSE={avg_mse:.3e}, RMSE={avg_rmse:.3e}, RelErr={avg_rel:.3e}")
    if overall_rel_err_primary_horizon: print(f"  Overall RelErr @ T={T_value_for_model_training:.1f}: {np.mean(overall_rel_err_primary_horizon):.3e}")


# main script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate PDE datasets with complex BCs.")
    parser.add_argument('--datatype', type=str, required=True, choices=['advection', 'euler', 'burgers', 'darcy'], help='Type of dataset to generate.')
    args = parser.parse_args()

    DATASET_TYPE = args.datatype # 'advection', 'euler', 'burgers', 'darcy'
    MODEL_TYPE = 'POD_DL_ROM'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Running {MODEL_TYPE} Baseline for {DATASET_TYPE} on {device} ---")

    FULL_T_IN_DATAFILE = 2.0; FULL_NT_IN_DATAFILE = 600
    TRAIN_T_TARGET = 1.0
    TRAIN_NT_FOR_MODEL = int((TRAIN_T_TARGET/FULL_T_IN_DATAFILE)*(FULL_NT_IN_DATAFILE-1))+1
    print(f"Datafile T={FULL_T_IN_DATAFILE}, nt={FULL_NT_IN_DATAFILE}")
    print(f"Training T={TRAIN_T_TARGET}, nt={TRAIN_NT_FOR_MODEL}")

    N_pod_dim = 32 
    n_latent_dim = 8 
    LEARNING_RATE = 1e-4; BATCH_SIZE = 32
    NUM_EPOCHS = 150 # Adjusted epochs
    CLIP_GRAD_NORM = 1.0; OMEGA_H = 0.5

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
        dataset_params_for_plot={'nx':128,'ny':1,'L':1.0,'T':FULL_T_IN_DATAFILE}
    else: raise ValueError(f"Unknown dataset type: {DATASET_TYPE}")

    print(f"Loading dataset: {dataset_path}")
    try:
        with open(dataset_path,'rb') as f: data_list_all=pickle.load(f)
        print(f"Loaded {len(data_list_all)} samples.")
        if data_list_all and 'params' in data_list_all[0]:
            file_params = data_list_all[0]['params']
            dataset_params_for_plot['L'] = file_params.get('L', dataset_params_for_plot['L'])
            dataset_params_for_plot['nx'] = file_params.get('nx', dataset_params_for_plot['nx'])
            actual_file_nt = data_list_all[0][main_state_keys[0]].shape[0]
            if actual_file_nt != FULL_NT_IN_DATAFILE:
                print(f"WARN: FULL_NT ({FULL_NT_IN_DATAFILE}) vs file nt ({actual_file_nt}). Using file nt."); FULL_NT_IN_DATAFILE = actual_file_nt
                TRAIN_NT_FOR_MODEL = int((TRAIN_T_TARGET/FULL_T_IN_DATAFILE)*(FULL_NT_IN_DATAFILE-1))+1
    except FileNotFoundError: print(f"Error: File not found {dataset_path}"); exit()
    if not data_list_all: print("No data. Exiting."); exit()

    random.shuffle(data_list_all); n_total=len(data_list_all); n_train=int(0.8*n_total)
    train_data_list_split=data_list_all[:n_train]; val_data_list_split=data_list_all[n_train:]

    train_dataset = UniversalPDEDataset(train_data_list_split, dataset_type=DATASET_TYPE, train_nt_limit=TRAIN_NT_FOR_MODEL)
    val_dataset = UniversalPDEDataset(val_data_list_split, dataset_type=DATASET_TYPE, train_nt_limit=None) # Full length for validation

    num_workers=1
    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=num_workers,pin_memory=True,drop_last=True)
    val_loader = DataLoader(val_dataset,batch_size=1,shuffle=False,num_workers=num_workers)

    current_nx_from_dataset = train_dataset.nx #* train_dataset.ny # Full spatial dimension for POD

    print(f"\nInitializing POD bases (N_pod={N_pod_dim})...")
    V_N_basis_dict = {} # Renamed
    # actual_nt for POD basis is TRAIN_NT_FOR_MODEL
    for key in main_state_keys:
        basis_path_main = os.path.join(f"./pod_bases_cache_{DATASET_TYPE}", f'pod_basis_{key}_nx{current_nx_from_dataset}_N{N_pod_dim}_trainNT{TRAIN_NT_FOR_MODEL}.npy') # More specific path
        os.makedirs(os.path.dirname(basis_path_main), exist_ok=True)
        loaded_b = None
        if os.path.exists(basis_path_main):
            try:
                loaded_b = np.load(basis_path_main)
                if loaded_b.shape!=(current_nx_from_dataset,N_pod_dim): loaded_b=None; print("Shape mismatch, recomputing POD.")
                else: print(f" Loaded POD for '{key}' from {basis_path_main}")
            except: loaded_b=None; print(f"Error loading POD for '{key}', recomputing.")
        if loaded_b is None:
            print(f" Computing POD for '{key}' (using {TRAIN_NT_FOR_MODEL} train timesteps)...")
            computed_b = compute_pod_basis_generic(train_data_list_split,DATASET_TYPE,key,current_nx_from_dataset,TRAIN_NT_FOR_MODEL,N_pod_dim)
            if computed_b is not None and computed_b.shape==(current_nx_from_dataset,N_pod_dim):
                V_N_basis_dict[key]=torch.tensor(computed_b).float(); np.save(basis_path_main,computed_b)
                print(f" Saved POD for '{key}' to {basis_path_main}")
            else: print(f"ERROR: POD failed for '{key}'. Exiting."); exit()
        else: V_N_basis_dict[key]=torch.tensor(loaded_b).float()

    actual_bc_ctrl_dim_from_dataset = train_dataset.bc_state_dim + train_dataset.num_controls
    dfnn_input_dim_main = 1 + actual_bc_ctrl_dim_from_dataset # time + bc_state + controls

    print(f"\nInitializing {MODEL_TYPE} model...")
    print(f"  N_pod={N_pod_dim}, n_latent={n_latent_dim}, dfnn_input_dim={dfnn_input_dim_main}")
    online_poddlrom_model = POD_DL_ROM( # Renamed
        V_N_dict=V_N_basis_dict, n_latent=n_latent_dim, dfnn_input_dim=dfnn_input_dim_main,
        N_pod=N_pod_dim, num_state_vars=main_num_state_vars
    )

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{DATASET_TYPE}_{MODEL_TYPE}_trainT{TRAIN_T_TARGET}_N{N_pod_dim}_n{n_latent_dim}"
    checkpoint_dir = f"./New_ckpt_2/checkpoints_{DATASET_TYPE}_{MODEL_TYPE}" # Simplified path
    results_dir = f"./result_all_2/results_{DATASET_TYPE}_{MODEL_TYPE}"      # Simplified path
    
    os.makedirs(checkpoint_dir,exist_ok=True); os.makedirs(results_dir,exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'model_{run_name}.pt')
    save_fig_path_prefix = os.path.join(results_dir, f'result_{run_name}')

    print(f"\nStarting training for {MODEL_TYPE} on {DATASET_TYPE}...")
    start_train_time = time.time()
    online_poddlrom_model = train_pod_dl_rom(
        online_poddlrom_model, train_loader, dataset_type=DATASET_TYPE,
        train_nt_for_model=TRAIN_NT_FOR_MODEL,
        lr=LEARNING_RATE, num_epochs=NUM_EPOCHS, device=device,
        checkpoint_path=checkpoint_path, clip_grad_norm=CLIP_GRAD_NORM, omega_h=OMEGA_H
    )
    end_train_time = time.time(); print(f"Training took {end_train_time - start_train_time:.2f} seconds.")

    if val_data_list_split:
        print(f"\nStarting validation for {MODEL_TYPE} on {DATASET_TYPE}...")
        validate_pod_dl_rom( # Modified to take new params
            online_poddlrom_model, val_loader, dataset_type=DATASET_TYPE, # Pass prefix
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
