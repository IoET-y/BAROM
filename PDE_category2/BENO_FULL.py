# beno
# =============================================================================
#     BENO-inspired Time-Stepping Neural Operator Baseline (Adapted for Task)
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
# ---------------------
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

print(f"BENO-Stepper Script (Task Adapted) started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# =============================================================================
# 0. UniversalPDEDataset (Corrected Version)
# =============================================================================
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
        self.ny_from_sample_file = 1

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
              self.bc_state_dim = actual_bc_state_dim # Use actual dimension from data
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
        sample = self.data_list[idx]
        norm_factors = {}
        current_nt_for_item = self.effective_nt_for_loader
        state_tensors_norm_list = []

        for key in self.state_keys:
            try:
                state_seq_full = sample[key]
                state_seq = state_seq_full[:current_nt_for_item, ...]
            except KeyError:
                raise KeyError(f"State variable key '{key}' not found in sample {idx} for dataset type '{self.dataset_type}'")
            if state_seq.shape[0] != current_nt_for_item:
                 raise ValueError(f"Time dimension mismatch for {key}. Expected {current_nt_for_item}, got {state_seq.shape[0]}")

            state_mean = np.mean(state_seq)
            state_std = np.std(state_seq) + 1e-8
            state_norm = (state_seq - state_mean) / state_std
            state_tensors_norm_list.append(torch.tensor(state_norm).float())
            norm_factors[f'{key}_mean'] = state_mean
            norm_factors[f'{key}_std'] = state_std

        bc_state_seq_full = sample[self.bc_state_key]
        bc_state_seq = bc_state_seq_full[:current_nt_for_item, :]
        if bc_state_seq.shape[0] != current_nt_for_item:
            raise ValueError(f"Time dim mismatch for BC_State. Expected {current_nt_for_item}, got {bc_state_seq.shape[0]}")

        bc_state_norm = np.zeros_like(bc_state_seq, dtype=np.float32)
        # Initialize with actual means and stds=1 to prevent issues if std is zero
        norm_factors[f'{self.bc_state_key}_means'] = np.mean(bc_state_seq, axis=0, keepdims=True).squeeze() if bc_state_seq.size > 0 else np.zeros(self.bc_state_dim)
        norm_factors[f'{self.bc_state_key}_stds'] = np.ones(self.bc_state_dim)

        if bc_state_seq.size > 0: # Ensure not empty
            for k_dim in range(self.bc_state_dim):
                col = bc_state_seq[:, k_dim]
                mean_k = np.mean(col) # Already stored in norm_factors if needed
                std_k = np.std(col)
                if std_k > 1e-8:
                    bc_state_norm[:, k_dim] = (col - mean_k) / std_k
                    norm_factors[f'{self.bc_state_key}_stds'][k_dim] = std_k # Update std
                else: # if std is zero, normalized is zero
                    bc_state_norm[:, k_dim] = col - mean_k # effectively zero
                norm_factors[f'{self.bc_state_key}_means'][k_dim] = mean_k # Store/update mean
        bc_state_tensor_norm = torch.tensor(bc_state_norm).float()


        if self.num_controls > 0:
            try:
                bc_control_seq_full = sample[self.bc_control_key]
                bc_control_seq = bc_control_seq_full[:current_nt_for_item, :]
                if bc_control_seq.shape[0] != current_nt_for_item:
                    raise ValueError(f"Time dim mismatch for BC_Control. Expected {current_nt_for_item}, got {bc_control_seq.shape[0]}")
                if bc_control_seq.shape[1] != self.num_controls:
                     raise ValueError(f"Control dim mismatch. Expected {self.num_controls}, got {bc_control_seq.shape[1]}.")

                bc_control_norm = np.zeros_like(bc_control_seq, dtype=np.float32)
                # norm_factors[f'{self.bc_control_key}_means'] = np.mean(bc_control_seq, axis=0, keepdims=True).squeeze() if bc_control_seq.size > 0 else np.zeros(self.num_controls)
                # norm_factors[f'{self.bc_control_key}_stds'] = np.ones(self.num_controls)
                # before normalization loop:
                if bc_control_seq.size > 0:
                    # returns shape (num_controls,)
                    norm_factors[f'{self.bc_control_key}_means'] = np.mean(bc_control_seq, axis=0)
                else:
                    norm_factors[f'{self.bc_control_key}_means'] = np.zeros(self.num_controls)
                norm_factors[f'{self.bc_control_key}_stds'] = np.ones(self.num_controls)

                if bc_control_seq.size > 0:
                    for k_dim in range(self.num_controls):
                        col = bc_control_seq[:, k_dim]
                        mean_k = np.mean(col)
                        std_k = np.std(col)
                        if std_k > 1e-8:
                            bc_control_norm[:, k_dim] = (col - mean_k) / std_k
                            norm_factors[f'{self.bc_control_key}_stds'][k_dim] = std_k
                        else:
                            bc_control_norm[:, k_dim] = col - mean_k
                        norm_factors[f'{self.bc_control_key}_means'][k_dim] = mean_k
                bc_control_tensor_norm = torch.tensor(bc_control_norm).float()
            except KeyError:
                bc_control_tensor_norm = torch.zeros((current_nt_for_item, self.num_controls), dtype=torch.float32)
                norm_factors[f'{self.bc_control_key}_means'] = np.zeros(self.num_controls)
                norm_factors[f'{self.bc_control_key}_stds'] = np.ones(self.num_controls)
        else:
            bc_control_tensor_norm = torch.empty((current_nt_for_item, 0), dtype=torch.float32)

        bc_ctrl_tensor_norm = torch.cat((bc_state_tensor_norm, bc_control_tensor_norm), dim=-1)
        output_state_tensors = state_tensors_norm_list[0] if self.num_state_vars == 1 else state_tensors_norm_list
        return output_state_tensors, bc_ctrl_tensor_norm, norm_factors

# =============================================================================
# 1. Helper Modules (MLP, Basic Transformer Encoder Layer)
# =============================================================================
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[], activation=nn.GELU, dropout=0.1):
        super().__init__(); layers=[]; current_dim=input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim,h_dim)); layers.append(activation())
            if dropout>0: layers.append(nn.Dropout(dropout))
            current_dim=h_dim
        layers.append(nn.Linear(current_dim,output_dim)); self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1, activation=F.gelu):
        super().__init__()
        self.self_attn=nn.MultiheadAttention(d_model,nhead,dropout=dropout,batch_first=True)
        self.linear1=nn.Linear(d_model,dim_feedforward); self.dropout=nn.Dropout(dropout)
        self.linear2=nn.Linear(dim_feedforward,d_model); self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model); self.dropout1=nn.Dropout(dropout)
        self.dropout2=nn.Dropout(dropout); self.activation=activation
    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False): # Compatibility
        # src_mask, is_causal typically not used in this type of encoder if no look-ahead mask needed
        src2, _ = self.self_attn(src,src,src,key_padding_mask=src_key_padding_mask, attn_mask=src_mask) # Pass through
        src=src+self.dropout1(src2); src=self.norm1(src)
        src2=self.linear2(self.dropout(self.activation(self.linear1(src))))
        src=src+self.dropout2(src2); src=self.norm2(src)
        return src

# =============================================================================
# 2. BENO-Stepper Architecture Components
# =============================================================================
class BoundaryEmbedder(nn.Module):
    def __init__(self, num_bc_points, input_feat_dim, d_model, nhead, num_encoder_layers, output_dim):
        super().__init__()
        self.num_bc_points=num_bc_points; self.input_feat_dim=input_feat_dim; self.d_model=d_model
        self.input_proj=nn.Linear(input_feat_dim,d_model)
        self.pos_encoder=nn.Parameter(torch.randn(1,num_bc_points,d_model)*0.02)
        encoder_layer=TransformerEncoderLayer(d_model,nhead)
        self.transformer_encoder=nn.TransformerEncoder(encoder_layer,num_layers=num_encoder_layers)
        self.output_mlp=MLP(d_model,output_dim,hidden_dims=[d_model//2])
    def forward(self,boundary_features):
        if boundary_features.shape[1]!=self.num_bc_points or boundary_features.shape[2]!=self.input_feat_dim:
            raise ValueError(f"BoundaryEmbedder input shape. Expected [B,{self.num_bc_points},{self.input_feat_dim}], got {boundary_features.shape}")
        src=self.input_proj(boundary_features)+self.pos_encoder; memory=self.transformer_encoder(src)
        pooled_memory=memory.mean(dim=1); global_embedding=self.output_mlp(pooled_memory)
        return global_embedding

class GNNLikeProcessor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, global_embed_dim):
        super().__init__()
        self.num_layers=num_layers; self.input_dim=input_dim; self.output_dim=output_dim
        self.hidden_dim=hidden_dim; self.global_embed_dim=global_embed_dim
        self.input_layer=nn.Linear(input_dim,hidden_dim); self.layers=nn.ModuleList()
        for _ in range(num_layers):
            layer=nn.ModuleDict({'conv':nn.Conv1d(hidden_dim,hidden_dim,kernel_size=5,padding=2,padding_mode='replicate'),
                                 'norm':nn.LayerNorm(hidden_dim),'act':nn.GELU(),
                                 'node_mlp':MLP(hidden_dim+global_embed_dim,hidden_dim,[hidden_dim])})
            self.layers.append(layer)
        self.output_layer=nn.Linear(hidden_dim,output_dim)
    def forward(self,x,global_boundary_embed): # x: [B, nx, C_in]
        x=self.input_layer(x) # [B, nx, hidden_dim]
        for layer in self.layers:
            x_res=x; x_perm=x.permute(0,2,1) # [B, hidden_dim, nx]
            aggregated=layer['conv'](x_perm).permute(0,2,1) # [B, nx, hidden_dim]
            aggregated=layer['norm'](aggregated); aggregated=layer['act'](aggregated)
            B,N,_=aggregated.shape; embed_expanded=global_boundary_embed.unsqueeze(1).repeat(1,N,1)
            mlp_input=torch.cat([aggregated,embed_expanded],dim=-1)
            x_updated=layer['node_mlp'](mlp_input); x=x_res+x_updated
        return self.output_layer(x)

class BENOStepper(nn.Module):
    def __init__(self, nx, num_state_vars, bc_ctrl_dim_input, state_keys, # Use bc_ctrl_dim_input
                 embed_dim=128, hidden_dim=128, gnn_layers=3, transformer_layers=2, nhead=4):
        super().__init__()
        self.nx=nx; self.num_state_vars=num_state_vars; self.state_keys=state_keys
        self.bc_ctrl_dim_input = bc_ctrl_dim_input # Total dimension of bc_ctrl_t from data loader

        self.num_bc_points = 2 # Hardcoded for left and right boundaries
        # Calculate feature dimension PER boundary point for the BoundaryEmbedder
        # Each boundary point gets its own state u(0) or u(L) AND its share of bc_ctrl_t
        # bc_ctrl_dim_input = (bc_state_dim + num_controls)
        # If bc_ctrl_dim_input represents features for ALL points, we need to split it.
        # Assuming bc_ctrl_t already contains [BCState_L, BCState_R, BCCtrl_L, BCCtrl_R] potentially
        # A robust way: bc_feat_dim = num_state_vars (from u_t at boundary) + (bc_ctrl_dim_input / num_bc_points)
        # This assumes bc_ctrl_dim_input is evenly divisible and structured for this split.
        if bc_ctrl_dim_input % self.num_bc_points != 0:
            print(f"Warning: bc_ctrl_dim_input ({bc_ctrl_dim_input}) not evenly divisible by num_bc_points ({self.num_bc_points}). Adjust feature extraction logic.")
            # Fallback or specific logic might be needed here
            # For now, let's assume it's okay, or user has specific _extract_boundary_features
            self.ctrl_feat_per_bc_point = bc_ctrl_dim_input // self.num_bc_points # Approx
        else:
            self.ctrl_feat_per_bc_point = bc_ctrl_dim_input // self.num_bc_points

        self.bc_feat_dim = num_state_vars + self.ctrl_feat_per_bc_point
        print(f"BENO Initializing: num_state_vars={num_state_vars}, total_bc_ctrl_dim={bc_ctrl_dim_input}")
        print(f"  Derived ctrl_feat_per_bc_point={self.ctrl_feat_per_bc_point}, leading to bc_feat_dim (for embedder)={self.bc_feat_dim}")


        self.global_embed_dim = embed_dim//2
        self.boundary_embedder = BoundaryEmbedder(self.num_bc_points,self.bc_feat_dim,embed_dim,nhead,transformer_layers,self.global_embed_dim)
        
        processor_input_dim = num_state_vars; processor_output_dim = hidden_dim
        self.processor1 = GNNLikeProcessor(processor_input_dim,processor_output_dim,hidden_dim,gnn_layers,self.global_embed_dim)
        
        # Input to boundary_processor_input_mlp is the flattened boundary_features
        # boundary_features has shape [B, num_bc_points, bc_feat_dim]
        # Flattened it's [B, num_bc_points * bc_feat_dim]
        # Output should be [B, nx * processor_input_dim] to be reshaped
        self.boundary_processor_input_mlp = MLP(self.num_bc_points * self.bc_feat_dim,
                                                processor_input_dim * nx,
                                                [hidden_dim * nx // 4, hidden_dim * nx // 2])
        self.processor2 = GNNLikeProcessor(processor_input_dim,processor_output_dim,hidden_dim,gnn_layers,self.global_embed_dim)
        self.decoder_mlp = MLP(processor_output_dim*2, num_state_vars, [hidden_dim, hidden_dim])

    def _extract_boundary_features(self, u_t, bc_ctrl_t):
        B = u_t.shape[0]
        u_left = u_t[:,0,:]; u_right = u_t[:,-1,:] # [B, C_state]

        # Split bc_ctrl_t for left and right points
        # bc_ctrl_t has shape [B, total_bc_ctrl_dim_input]
        # ctrl_feat_per_bc_point was calculated in __init__
        # This assumes bc_ctrl_t is structured as [left_ctrl_feats, right_ctrl_feats]
        # If not, this part needs dataset-specific logic.
        # Example: if bc_ctrl_t is [StateL_rho, StateL_u, CtrlL, StateR_rho, StateR_u, CtrlR]
        # For Euler: C_state=2. bc_ctrl_dim_input might be 6 (2 for state_l, 1 for ctrl_l, 2 for state_r, 1 for ctrl_r)
        # Then ctrl_feat_per_bc_point should be 3. bc_feat_dim = 2 (from u_t) + 3 = 5.

        # Simplified general split:
        # This assumes bc_ctrl_t structure is [left_features, right_features]
        # where each part has dimension self.ctrl_feat_per_bc_point
        expected_split_len = self.ctrl_feat_per_bc_point
        
        # Handle cases where bc_ctrl_dim_input might not be perfectly splittable
        # or where the structure is more complex (e.g. Euler's specific 4 state + 2 control layout)
        # For now, using the earlier generic split strategy. More robust logic would be needed for complex bc_ctrl_t structures.
        if self.num_state_vars == 2 and self.bc_ctrl_dim_input == (self.state_keys.index('u')*2 + 2 + self.state_keys.index('u')*2 + 2): # Very specific Euler check based on old example
             # This part is from the original BENO Euler logic where bc_ctrl_t was (state_L, state_R, ctrl_L, ctrl_R)
             # bc_ctrl_dim for Euler is typically 4 (BC_State from UniversalDataset) + N_controls.
             # If num_controls=0 for your runs, bc_ctrl_dim_input from data loader = 4
             # ctrl_feat_per_bc_point = 4/2 = 2. bc_feat_dim = 2(u_t) + 2 = 4.
             # bc_ctrl_t: [rho_L, u_L, rho_R, u_R]
            bc_info_left = bc_ctrl_t[:, :self.num_state_vars] # e.g., rho_L, u_L
            bc_info_right = bc_ctrl_t[:, self.num_state_vars : 2*self.num_state_vars] # e.g., rho_R, u_R
            # If controls exist and are appended after state:
            if self.bc_ctrl_dim_input > 2*self.num_state_vars: # Controls are present
                num_total_controls = self.bc_ctrl_dim_input - 2*self.num_state_vars
                num_ctrl_per_side = num_total_controls // 2
                ctrl_left = bc_ctrl_t[:, 2*self.num_state_vars : 2*self.num_state_vars + num_ctrl_per_side]
                ctrl_right = bc_ctrl_t[:, 2*self.num_state_vars + num_ctrl_per_side:]
                bc_info_left = torch.cat([bc_info_left, ctrl_left], dim=-1)
                bc_info_right = torch.cat([bc_info_right, ctrl_right], dim=-1)
        else: # General split assuming [left_ctrl_data, right_ctrl_data]
            bc_info_left = bc_ctrl_t[:, :self.ctrl_feat_per_bc_point]
            bc_info_right = bc_ctrl_t[:, self.ctrl_feat_per_bc_point : 2*self.ctrl_feat_per_bc_point] # Ensure slice is correct

        feat_left = torch.cat([u_left, bc_info_left], dim=-1)
        feat_right = torch.cat([u_right, bc_info_right], dim=-1)

        if feat_left.shape[-1] != self.bc_feat_dim:
            raise ValueError(f"Boundary feature dim mismatch after concat. Expected {self.bc_feat_dim}, got {feat_left.shape[-1]}. u_left: {u_left.shape}, bc_info_left: {bc_info_left.shape}")

        return torch.stack([feat_left, feat_right], dim=1) # [B, 2, bc_feat_dim]

    def forward(self, u_t, bc_ctrl_t):
        B,N,C_in=u_t.shape
        boundary_features=self._extract_boundary_features(u_t,bc_ctrl_t)
        global_boundary_embed=self.boundary_embedder(boundary_features)
        out1=self.processor1(u_t,global_boundary_embed)
        flat_boundary_feat=boundary_features.view(B,-1)
        boundary_influence_field=self.boundary_processor_input_mlp(flat_boundary_feat).view(B,N,C_in)
        out2=self.processor2(boundary_influence_field,global_boundary_embed)
        combined_features=torch.cat([out1,out2],dim=-1)
        u_tp1_pred=self.decoder_mlp(combined_features)
        return u_tp1_pred

# =============================================================================
# 4. Training and Validation Functions (Adapted for Task)
# =============================================================================
def train_beno_stepper(model, data_loader, dataset_type, train_nt_for_model, # Added
                       lr=1e-3, num_epochs=50, device='cuda',
                       checkpoint_path='beno_ckpt.pt', clip_grad_norm=1.0):
    model.to(device); optimizer=optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=10,verbose=True)
    mse_loss=nn.MSELoss(reduction='mean'); start_epoch=0; best_loss=float('inf')
    if os.path.exists(checkpoint_path):
        print(f"Loading BENO ckpt from {checkpoint_path}...")
        try:
            ckpt=torch.load(checkpoint_path,map_location=device); model.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                try: optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                except: print("Warn: BENO Optimizer state mismatch.")
            start_epoch=ckpt.get('epoch',0)+1; best_loss=ckpt.get('loss',float('inf'))
            print(f"Resuming BENO training from epoch {start_epoch}")
        except Exception as e: print(f"Error loading BENO ckpt: {e}. Starting fresh.")

    for epoch in range(start_epoch, num_epochs):
        model.train(); epoch_loss=0.0; num_batches=0; batch_start_time=time.time()
        for i, (state_data_loaded, BC_Ctrl_tensor_loaded, _) in enumerate(data_loader):
            if isinstance(state_data_loaded, list): state_seq_true_train=torch.stack(state_data_loaded,dim=-1).to(device)
            else: state_seq_true_train=state_data_loaded.unsqueeze(-1).to(device)
            BC_Ctrl_seq_train=BC_Ctrl_tensor_loaded.to(device)
            B,nt_loaded,nx,_=state_seq_true_train.shape
            if nt_loaded!=train_nt_for_model: raise ValueError(f"Data nt {nt_loaded} != train_nt {train_nt_for_model}")

            optimizer.zero_grad(); total_seq_loss=torch.tensor(0.0,device=device,requires_grad=True)
            for t in range(train_nt_for_model-1):
                u_n_true=state_seq_true_train[:,t,:,:]; bc_ctrl_n=BC_Ctrl_seq_train[:,t,:]
                u_np1_true=state_seq_true_train[:,t+1,:,:]
                u_np1_pred=model(u_n_true,bc_ctrl_n)
                step_loss=mse_loss(u_np1_pred,u_np1_true); total_seq_loss=total_seq_loss+step_loss
            
            batch_loss=total_seq_loss/(train_nt_for_model-1) # Use current_batch_loss if prefer
            epoch_loss+=batch_loss.item(); num_batches+=1
            batch_loss.backward()
            if clip_grad_norm>0: torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=clip_grad_norm)
            optimizer.step()
            if (i+1)%50==0:
                elapsed=time.time()-batch_start_time
                print(f" BENO Ep {epoch+1} B {i+1}/{len(data_loader)}, Loss {batch_loss.item():.3e}, Time/50B {elapsed:.2f}s")
                batch_start_time=time.time()
        avg_epoch_loss=epoch_loss/max(num_batches,1)
        print(f"BENO Epoch {epoch+1}/{num_epochs} Avg Loss: {avg_epoch_loss:.6f}")
        scheduler.step(avg_epoch_loss)
        if avg_epoch_loss<best_loss:
            best_loss=avg_epoch_loss; print(f"Saving BENO ckpt loss {best_loss:.6f}")
            torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'loss':best_loss,'dataset_type':dataset_type,'state_keys':model.state_keys},checkpoint_path) # Removed .item() from loss
    print("BENO Training finished.")
    if os.path.exists(checkpoint_path): print(f"Loading best BENO model"); ckpt=torch.load(checkpoint_path,map_location=device); model.load_state_dict(ckpt['model_state_dict'])
    return model

def validate_beno_stepper(model, data_loader, dataset_type,
                          train_nt_for_model_training: int, T_value_for_model_training: float,
                          full_T_in_datafile: float, full_nt_in_datafile: int,
                          dataset_params_for_plot: dict, device='cuda',
                          save_fig_path_prefix='beno_result'):
    model.eval()
    state_keys_val = model.state_keys
    num_state_vars_val = model.num_state_vars

    test_horizons_T_values = [T_value_for_model_training]
    if full_T_in_datafile > T_value_for_model_training:
        test_horizons_T_values.append(T_value_for_model_training + 0.5 * (full_T_in_datafile - T_value_for_model_training))
        test_horizons_T_values.append(full_T_in_datafile)
    test_horizons_T_values = sorted(list(set(h for h in test_horizons_T_values if h <= full_T_in_datafile + 1e-6)))
    print(f"BENO Validation for T_horizons: {test_horizons_T_values}")

    results_primary_horizon = {key: {'mse':[],'rmse':[],'relative_error':[],'max_error':[]} for key in state_keys_val}
    overall_rel_err_primary_horizon = []

    with torch.no_grad():
        try:
            state_data_full_loaded, BC_Ctrl_tensor_full_loaded, norm_factors_batch = next(iter(data_loader))
        except StopIteration: print("Val data_loader empty. Skipping BENO validation."); return

        if isinstance(state_data_full_loaded, list): state_seq_true_norm_full = torch.stack(state_data_full_loaded, dim=-1)[0].to(device)
        else: state_seq_true_norm_full = state_data_full_loaded.unsqueeze(-1)[0].to(device)
        BC_Ctrl_seq_norm_full = BC_Ctrl_tensor_full_loaded[0].to(device)
        _, nx_plot, _ = state_seq_true_norm_full.shape # nt_file_check, nx_plot, num_vars_check_val

        norm_factors_sample = {}
        for key, val_tensor in norm_factors_batch.items():
            norm_factors_sample[key] = val_tensor[0].cpu().numpy() if isinstance(val_tensor,torch.Tensor) and val_tensor.ndim>0 else (val_tensor.cpu().numpy() if isinstance(val_tensor,torch.Tensor) else val_tensor)

        u_initial_norm = state_seq_true_norm_full[0:1, :, :]

        for T_horizon_current in test_horizons_T_values:
            nt_for_rollout = int((T_horizon_current/full_T_in_datafile)*(full_nt_in_datafile-1))+1
            nt_for_rollout = min(nt_for_rollout, full_nt_in_datafile)
            print(f"\n  Rollout for T_horizon = {T_horizon_current:.2f} (nt = {nt_for_rollout})")

            u_pred_seq_norm_horizon = torch.zeros(nt_for_rollout, nx_plot, num_state_vars_val, device=device)
            u_current_pred_step = u_initial_norm.clone()
            u_pred_seq_norm_horizon[0,:,:] = u_current_pred_step.squeeze(0)
            BC_Ctrl_for_rollout = BC_Ctrl_seq_norm_full[:nt_for_rollout, :]

            for t_step in range(nt_for_rollout - 1):
                bc_ctrl_n_step = BC_Ctrl_for_rollout[t_step:t_step+1, :]
                if u_current_pred_step.shape[0]!=1: u_current_pred_step=u_current_pred_step.unsqueeze(0)
                u_next_pred_norm_step = model(u_current_pred_step, bc_ctrl_n_step)
                u_pred_seq_norm_horizon[t_step+1,:,:] = u_next_pred_norm_step.squeeze(0)
                u_current_pred_step = u_next_pred_norm_step
            
            # Denormalize & Metrics
            U_pred_denorm_h={}; U_gt_denorm_h={}
            pred_list_h=[]; gt_list_h=[]
            state_true_norm_sliced_h = state_seq_true_norm_full[:nt_for_rollout,:,:]
            pred_np_h = u_pred_seq_norm_horizon.cpu().numpy(); gt_np_h = state_true_norm_sliced_h.cpu().numpy()

            for k_idx, key_val in enumerate(state_keys_val):
                mean_k=norm_factors_sample[f'{key_val}_mean']; std_k=norm_factors_sample[f'{key_val}_std']
                mean_k=mean_k.item() if hasattr(mean_k,'item') else mean_k; std_k=std_k.item() if hasattr(std_k,'item') else std_k
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
            fig_L=dataset_params_for_plot.get('L',1.0); fig_nx=dataset_params_for_plot.get('nx',nx_plot); fig_ny=dataset_params_for_plot.get('ny',1)
            for k_idx, key_val in enumerate(state_keys_val):
                gt_p=U_gt_denorm_h[key_val]; pred_p=U_pred_denorm_h[key_val]; diff_p=np.abs(pred_p-gt_p)
                max_err_p=np.max(diff_p) if diff_p.size>0 else 0
                vmin_p=min(np.min(gt_p),np.min(pred_p)) if gt_p.size>0 else 0; vmax_p=max(np.max(gt_p),np.max(pred_p)) if gt_p.size>0 else 1
                is_1d_p=(fig_ny==1); plot_ext=[0,fig_L,0,T_horizon_current]
                if is_1d_p:
                    im0=axs[k_idx,0].imshow(gt_p,aspect='auto',origin='lower',vmin=vmin_p,vmax=vmax_p,extent=plot_ext,cmap='viridis'); axs[k_idx,0].set_title(f"Truth ({key_val})")
                    im1=axs[k_idx,1].imshow(pred_p,aspect='auto',origin='lower',vmin=vmin_p,vmax=vmax_p,extent=plot_ext,cmap='viridis'); axs[k_idx,1].set_title(f"BENO Pred ({key_val})") # BENO Title
                    im2=axs[k_idx,2].imshow(diff_p,aspect='auto',origin='lower',extent=plot_ext,cmap='magma'); axs[k_idx,2].set_title(f"Abs Error (Max:{max_err_p:.2e})")
                    for j_p in range(3): axs[k_idx,j_p].set_xlabel("x"); axs[k_idx,0].set_ylabel("t")
                    plt.colorbar(im0,ax=axs[k_idx,0]); plt.colorbar(im1,ax=axs[k_idx,1]); plt.colorbar(im2,ax=axs[k_idx,2])
                else: axs[k_idx,0].text(0.5,0.5,"2D Plot Placeholder",ha='center')
            fig.suptitle(f"BENO Validation ({dataset_type.capitalize()}) @ T={T_horizon_current:.1f}") # BENO Title
            fig.tight_layout(rect=[0,0.03,1,0.95])
            curr_fig_path=save_fig_path_prefix+f"_T{str(T_horizon_current).replace('.','p')}.png"
            plt.savefig(curr_fig_path); print(f"  Saved BENO validation plot to {curr_fig_path}"); plt.show()
    
    print(f"\n--- BENO Validation Summary (Metrics for T={T_value_for_model_training:.1f}) ---")
    for key_val in state_keys_val:
        if results_primary_horizon[key_val]['mse']:
            avg_mse=np.mean(results_primary_horizon[key_val]['mse']); avg_rmse=np.sqrt(avg_mse)
            avg_rel=np.mean(results_primary_horizon[key_val]['relative_error'])
            print(f"  Var '{key_val}': Avg MSE={avg_mse:.3e}, RMSE={avg_rmse:.3e}, RelErr={avg_rel:.3e}")
    if overall_rel_err_primary_horizon: print(f"  Overall RelErr @ T={T_value_for_model_training:.1f}: {np.mean(overall_rel_err_primary_horizon):.3e}")
    print("------------------------")


# =============================================================================
# 5. Main Execution Block
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate PDE datasets with complex BCs.")
    parser.add_argument('--datatype', type=str, required=True, choices=['advection', 'euler', 'burgers', 'darcy','heat_delayed_feedback','reaction_diffusion_neumann_feedback','heat_nonlinear_feedback_gain','convdiff']) # Added convdiff, help='Type of dataset to generate.')
    args = parser.parse_args()

    DATASET_TYPE = args.datatype # 'advection', 'euler', 'burgers', 'darcy'
    MODEL_TYPE = 'BENO_Stepper'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Running {MODEL_TYPE} Baseline for {DATASET_TYPE} on {device} ---")

    # BENO Model Hyperparameters from original script
    EMBED_DIM = 64; HIDDEN_DIM = 64; GNN_LAYERS = 4
    TRANSFORMER_LAYERS = 2; NHEAD = 4
    LEARNING_RATE = 5e-4; BATCH_SIZE = 32
    NUM_EPOCHS = 80
    CLIP_GRAD_NORM = 1.0
    FULL_T_IN_DATAFILE = 2.0  # MUST MATCH YOUR GENERATION SCRIPT
    FULL_NT_IN_DATAFILE = 300
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
    elif DATASET_TYPE == 'heat_delayed_feedback':

        dataset_path = "./datasets_new_feedback/heat_delayed_feedback_v1_5000s_64nx_300nt.pkl" # UPDATE PATH
        main_state_keys = ['U']; main_num_state_vars = 1
        dataset_params_for_plot = {'nx': 64, 'ny': 1, 'L': 1.0, 'T': FULL_T_IN_DATAFILE}
    elif DATASET_TYPE == 'reaction_diffusion_neumann_feedback':

        dataset_path = "./datasets_new_feedback/reaction_diffusion_neumann_feedback_v1_5000s_64nx_300nt.pkl" # UPDATE PATH
        main_state_keys = ['U']; main_num_state_vars = 1
        dataset_params_for_plot = {'nx': 64, 'ny': 1, 'L': 1.0, 'T': FULL_T_IN_DATAFILE}
    elif DATASET_TYPE == 'heat_nonlinear_feedback_gain':

        dataset_path = "./datasets_new_feedback/heat_nonlinear_feedback_gain_v1_5000s_64nx_300nt.pkl" # UPDATE PATH
        main_state_keys = ['U']; main_num_state_vars = 1
        dataset_params_for_plot = {'nx': 64, 'ny': 1, 'L': 1.0, 'T': FULL_T_IN_DATAFILE}
    elif DATASET_TYPE == 'convdiff':
     # Example
        dataset_path = "./datasets_new_feedback/convdiff_v1_5000s_64nx_300nt.pkl" # UPDATE PATH
        main_state_keys = ['U']; main_num_state_vars = 1
        dataset_params_for_plot = {'nx': 64, 'ny': 1, 'L': 1.0, 'T': FULL_T_IN_DATAFILE} 
    else: raise ValueError(f"Unknown dataset type: {DATASET_TYPE}")
    FULL_T_IN_DATAFILE = 2.0  # MUST MATCH YOUR GENERATION SCRIPT
    FULL_NT_IN_DATAFILE = 300 # MUST MATCH YOUR GENERATION SCRIPT
    TRAIN_T_TARGET = 1.5     # Example
    TRAIN_NT_FOR_MODEL = int((TRAIN_T_TARGET / FULL_T_IN_DATAFILE) * (FULL_NT_IN_DATAFILE - 1)) + 1

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

    random.shuffle(data_list_all); n_total=len(data_list_all); n_train=int(0.8*n_total)
    train_data_list_split=data_list_all[:n_train]; val_data_list_split=data_list_all[n_train:]
    print(f"Train samples: {len(train_data_list_split)}, Validation samples: {len(val_data_list_split)}")


    train_dataset = UniversalPDEDataset(train_data_list_split, dataset_type=DATASET_TYPE, train_nt_limit=TRAIN_NT_FOR_MODEL)
    val_dataset = UniversalPDEDataset(val_data_list_split, dataset_type=DATASET_TYPE, train_nt_limit=None)

    num_workers=1
    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=num_workers,pin_memory=True,drop_last=True)
    val_loader = DataLoader(val_dataset,batch_size=1,shuffle=False,num_workers=num_workers)

    actual_bc_ctrl_dim_from_dataset = train_dataset.bc_state_dim + train_dataset.num_controls
    current_nx_from_dataset = train_dataset.nx

    print(f"\nInitializing {MODEL_TYPE} model...")
    print(f"  nx={current_nx_from_dataset}, num_state_vars={main_num_state_vars}, bc_ctrl_dim_input={actual_bc_ctrl_dim_from_dataset}")
    online_beno_model = BENOStepper(
        nx=current_nx_from_dataset,
        num_state_vars=main_num_state_vars,
        bc_ctrl_dim_input=actual_bc_ctrl_dim_from_dataset,
        state_keys=main_state_keys, # Pass state_keys
        embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, gnn_layers=GNN_LAYERS,
        transformer_layers=TRANSFORMER_LAYERS, nhead=NHEAD
    )
    # Add dataset_type to model if needed by _extract_boundary_features, or pass it if it becomes a method arg
    online_beno_model.dataset_type = DATASET_TYPE # Store for _extract_boundary_features if needed
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{DATASET_TYPE}_{MODEL_TYPE}_trainT{TRAIN_T_TARGET}_emb{EMBED_DIM}"
    checkpoint_dir = f"./New_ckpt_2/checkpoints_{DATASET_TYPE}_{MODEL_TYPE}" # Simplified path
    results_dir = f"./result_all_2/results_{DATASET_TYPE}_{MODEL_TYPE}"      # Simplified path
    os.makedirs(checkpoint_dir,exist_ok=True); os.makedirs(results_dir,exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'model_{run_name}.pt')
    save_fig_path_prefix = os.path.join(results_dir, f'result_{run_name}')

    print(f"\nStarting training for {MODEL_TYPE} on {DATASET_TYPE}...")
    start_train_time = time.time()
    online_beno_model = train_beno_stepper(
        online_beno_model, train_loader, dataset_type=DATASET_TYPE,
        train_nt_for_model=TRAIN_NT_FOR_MODEL,
        lr=LEARNING_RATE, num_epochs=NUM_EPOCHS, device=device,
        checkpoint_path=checkpoint_path, clip_grad_norm=CLIP_GRAD_NORM
    )
    end_train_time = time.time(); print(f"Training took {end_train_time - start_train_time:.2f} seconds.")

    if val_data_list_split:
        print(f"\nStarting validation for {MODEL_TYPE} on {DATASET_TYPE}...")
        validate_beno_stepper(
            online_beno_model, val_loader, dataset_type=DATASET_TYPE,
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
