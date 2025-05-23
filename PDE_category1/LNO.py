# LNO
# =============================================================================
#     Latent Neural Operator (Adapted for Task1) ref:https://github.com/L-I-M-I-T/LatentNeuralOperator
#Tian Wang and Chuang Wang. Latent Neural Operator for Solving Forward and Inverse PDE Problem. In Conference on Neural Information Processing Systems (NeurIPS), 2024
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
# scipy.sparse and spsolve are not directly used by LNO but were in the original file.
# Keeping them in case they are part of the broader project context.
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


# fixed seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

print(f"Latent Neural Operator (Autoregressive) Script started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")


class UniversalPDEDataset(Dataset):
    def __init__(self, data_list, dataset_type, train_nt_limit=None):
        if not data_list: raise ValueError("data_list cannot be empty")
        self.data_list = data_list; self.dataset_type = dataset_type.lower(); self.train_nt_limit = train_nt_limit
        first_sample = data_list[0]; params = first_sample.get('params', {})
        self.nt_from_sample_file=0; self.nx_from_sample_file=0; self.ny_from_sample_file=1 # ny for potential 2D spatial
        if self.dataset_type in ['advection', 'burgers']:
            self.nt_from_sample_file=first_sample['U'].shape[0]; self.nx_from_sample_file=first_sample['U'].shape[1]
            self.state_keys=['U']; self.num_state_vars=1; self.expected_bc_state_dim=2
        elif self.dataset_type == 'euler':
            self.nt_from_sample_file=first_sample['rho'].shape[0]; self.nx_from_sample_file=first_sample['rho'].shape[1]
            self.state_keys=['rho','u']; self.num_state_vars=2; self.expected_bc_state_dim=4
        elif self.dataset_type == 'darcy': # Darcy is typically steady-state, but if data has pseudo-time
            self.nt_from_sample_file=first_sample['P'].shape[0]; self.nx_from_sample_file=params.get('nx', first_sample['P'].shape[1])
            self.ny_from_sample_file=params.get('ny',1); self.state_keys=['P']; self.num_state_vars=1; self.expected_bc_state_dim=2
        else: raise ValueError(f"Unknown type: {self.dataset_type}")

        self.effective_nt_for_loader = self.train_nt_limit if self.train_nt_limit is not None else self.nt_from_sample_file
        self.nx=self.nx_from_sample_file; self.ny=self.ny_from_sample_file;
        # For LNO, spatial_dim is typically the number of points, nx, if 1D spatial.
        # If you have 2D spatial data (nx, ny), nx_from_sample_file might be nx*ny or you handle it differently.
        # Assuming 1D spatial for now based on ROM context (nx points).
        self.spatial_dim=self.nx # Or self.nx * self.ny if 2D spatial

        self.bc_state_key='BC_State'
        if self.bc_state_key not in first_sample:  # Ensure BC_State key exists
            raise KeyError(f"'{self.bc_state_key}' not found in the first sample!")
        actual_bc_dim = first_sample[self.bc_state_key].shape[1]

        if actual_bc_dim != self.expected_bc_state_dim:
             print(f"Warning: BC_State dimension mismatch for {self.dataset_type}. "
                   f"Expected {self.expected_bc_state_dim}, got {actual_bc_dim}. "
                   f"Using actual dimension: {actual_bc_dim}")
        self.bc_state_dim = actual_bc_dim


        self.bc_control_key='BC_Control'
        # Check if BC_Control exists and is not empty or None
        if self.bc_control_key in first_sample and \
           first_sample[self.bc_control_key] is not None and \
           first_sample[self.bc_control_key].size > 0:
            self.num_controls = first_sample[self.bc_control_key].shape[1]
        else:
            self.num_controls=0
            # print(f"Warning: '{self.bc_control_key}' not found or is empty in the first sample for {self.dataset_type}. Assuming num_controls = 0.")


    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx):
        sample=self.data_list[idx]; norm_factors={}; current_nt=self.effective_nt_for_loader; slist=[]
        for key in self.state_keys:
            s_full=sample[key]; s_seq=s_full[:current_nt,...];
            # Ensure s_seq is 2D (nt, nx) for 1D spatial or 3D (nt, nx, ny) for 2D spatial before reshaping
            if s_seq.ndim == 1 and self.nx == 1: # scalar field at multiple times e.g. shape (nt,)
                s_seq = s_seq.reshape(current_nt, 1)
            elif s_seq.ndim == 2 and self.ny > 1 : # (nt, nx*ny) needs to be (nt,nx,ny) for per-dim norm.
                 # This case might require more specific handling based on how data is stored.
                 # For now, assume global norm if it's already flattened.
                 pass

            s_mean=np.mean(s_seq); s_std=np.std(s_seq)+1e-8 # Global normalization for the variable's sequence
            slist.append(torch.tensor((s_seq-s_mean)/s_std).float());
            norm_factors[f'{key}_mean']=s_mean; norm_factors[f'{key}_std']=s_std

        # BC State (remains, though LNO stepper won't use it directly per step)
        bcs_full=sample[self.bc_state_key]; bcs_seq=bcs_full[:current_nt,:]; bcs_norm=np.zeros_like(bcs_seq,dtype=np.float32)
        norm_factors[f'{self.bc_state_key}_means']=np.mean(bcs_seq,axis=0,keepdims=True).squeeze(0) if bcs_seq.size>0 else np.zeros(self.bc_state_dim) # Squeeze(0) for 2D mean
        norm_factors[f'{self.bc_state_key}_stds']=np.ones(self.bc_state_dim)
        if bcs_seq.size > 0 and self.bc_state_dim > 0: # ensure bc_state_dim is positive
            for d in range(self.bc_state_dim):
                col=bcs_seq[:,d]; m=np.mean(col); s=np.std(col)
                if s>1e-8: bcs_norm[:,d]=(col-m)/s; norm_factors[f'{self.bc_state_key}_stds'][d]=s
                else: bcs_norm[:,d]=col-m # effectively zero if std is tiny
                norm_factors[f'{self.bc_state_key}_means'][d]=m # update mean for this column
        bcs_tensor=torch.tensor(bcs_norm).float()

        # BC Control (remains)
        if self.num_controls>0:
            try:
                bcc_full=sample[self.bc_control_key]; bcc_seq=bcc_full[:current_nt,:];
                if bcc_seq.shape[0]!=current_nt: raise ValueError(f"Time dim mismatch BC_Control. Expected {current_nt}, got {bcc_seq.shape[0]}")
                if bcc_seq.shape[1]!=self.num_controls: raise ValueError(f"Control dim mismatch. Expected {self.num_controls}, got {bcc_seq.shape[1]}")
                bcc_norm=np.zeros_like(bcc_seq,dtype=np.float32)
                norm_factors[f'{self.bc_control_key}_means']=np.mean(bcc_seq,axis=0,keepdims=True).squeeze(0) if bcc_seq.size>0 else np.zeros(self.num_controls)
                norm_factors[f'{self.bc_control_key}_stds']=np.ones(self.num_controls)
                if bcc_seq.size > 0:
                    for d in range(self.num_controls):
                        col=bcc_seq[:,d];m=np.mean(col);s=np.std(col)
                        if s>1e-8: bcc_norm[:,d]=(col-m)/s; norm_factors[f'{self.bc_control_key}_stds'][d]=s
                        else: bcc_norm[:,d]=col-m
                        norm_factors[f'{self.bc_control_key}_means'][d]=m
                bcc_tensor=torch.tensor(bcc_norm).float()
            except KeyError:
                # print(f"Warning: Sample {idx} missing '{self.bc_control_key}'. Using zeros.")
                bcc_tensor=torch.zeros((current_nt,self.num_controls),dtype=torch.float32)
                norm_factors[f'{self.bc_control_key}_means']=np.zeros(self.num_controls)
                norm_factors[f'{self.bc_control_key}_stds']=np.ones(self.num_controls)
        else: bcc_tensor=torch.empty((current_nt,0),dtype=torch.float32)

        bc_ctrl_tensor=torch.cat((bcs_tensor,bcc_tensor),dim=-1)
        out_state = slist[0] if self.num_state_vars==1 else slist # slist contains normalized tensors
        return out_state, bc_ctrl_tensor, norm_factors


# 1. LNO Components
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, activation=nn.GELU):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), activation()]
            d = h
        layers.append(nn.Linear(d, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class PhysicsCrossAttention(nn.Module):
    def __init__(self, embed_dim, M_latent, projector_hidden_dims=[128,128]):
        super().__init__()
        self.attn_proj = MLP(input_dim=embed_dim,
                             output_dim=M_latent,
                             hidden_dims=projector_hidden_dims)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

    def encode(self, X_hat, Y_hat):
        scores = self.attn_proj(X_hat)
        scores = scores.transpose(1,2)
        attn  = F.softmax(scores, dim=-1)
        V = self.value_proj(Y_hat)
        return attn @ V

    def decode(self, P_embed, ZL):
        scores = self.attn_proj(P_embed)
        attn  = F.softmax(scores, dim=-1)
        V = self.value_proj(ZL)
        return attn @ V

class LatentNeuralOperator(nn.Module):
    def __init__(self, num_state_vars, coord_dim, # coord_dim will be 2 (spatial_x, pseudo_t)
                 M_latent_points, D_embedding, L_transformer_blocks,
                 transformer_nhead, transformer_dim_feedforward,
                 projector_hidden_dims=[128, 128], final_mlp_hidden_dims=[128,128]):
        super().__init__()
        self.num_state_vars = num_state_vars
        self.coord_dim = coord_dim
        self.M_latent_points = M_latent_points
        self.D_embedding = D_embedding
        # Store hyperparameters as instance attributes
        self.L_transformer_blocks = L_transformer_blocks
        self.transformer_nhead = transformer_nhead
        self.transformer_dim_feedforward = transformer_dim_feedforward
        self.projector_hidden_dims = projector_hidden_dims
        self.final_mlp_hidden_dims = final_mlp_hidden_dims

        
        self.trunk_projector = MLP(coord_dim, D_embedding, projector_hidden_dims)
        # Branch projector takes concat(pos, val_at_pos) -> D_embedding
        # For a single time step, val_in has num_state_vars channels.
        # Input to branch projector is (spatial_coord, pseudo_t_input, U_k_vars)
        # So, its input dimension is coord_dim + num_state_vars
        self.branch_projector = MLP(coord_dim + num_state_vars, D_embedding, projector_hidden_dims)

        self.H_latent_queries = nn.Parameter(torch.randn(1, M_latent_points, D_embedding) * 0.02)

        self.phca = PhysicsCrossAttention(
            embed_dim=D_embedding,
            M_latent=M_latent_points,
            projector_hidden_dims=projector_hidden_dims
        )
        # Using the same phca instance for encoder and decoder implies shared weights for W1/W2 in the paper's context
        self.phca_encoder = self.phca
        self.phca_decoder = self.phca

        encoder_layer = nn.TransformerEncoderLayer(d_model=D_embedding, nhead=transformer_nhead,
                                                   dim_feedforward=transformer_dim_feedforward,
                                                   batch_first=True, activation=F.gelu, dropout=0.0)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=L_transformer_blocks)

        self.final_mlp = MLP(D_embedding, num_state_vars, final_mlp_hidden_dims)

        # Attributes for convenience, to be set after instantiation
        self.state_keys = None
        self.nx = None # Spatial dimension (e.g., number of grid points in x)
        self.domain_L = None # Physical length of the domain

    def forward(self, pos_in, val_in, pos_out):
        # Autoregressive context:
        # pos_in: [B, N_points (e.g. nx), coord_dim (e.g. 2 for x, pseudo_t_in)]
        # val_in: [B, N_points (e.g. nx), num_state_vars] (U_k at these points)
        # pos_out: [B, N_points (e.g. nx), coord_dim (e.g. 2 for x, pseudo_t_out)] (query points for U_{k+1})
        B = pos_in.shape[0]

        X_hat = self.trunk_projector(pos_in) # [B, N_points, D_embedding]

        branch_input = torch.cat([pos_in, val_in], dim=-1) # [B, N_points, coord_dim + num_state_vars]
        Y_hat = self.branch_projector(branch_input) # [B, N_points, D_embedding]


        Z0 = self.phca_encoder.encode(X_hat, Y_hat) # [B, M_latent_points, D_embedding]

        ZL = self.transformer_encoder(Z0) # [B, M_latent_points, D_embedding]

        # Decode: Map from latent space to physical space (at t_{k+1})
        P_out_embed = self.trunk_projector(pos_out) # [B, N_points, D_embedding]
        U_embed = self.phca_decoder.decode(P_out_embed, ZL) # [B, N_points, D_embedding]

        val_out_raw = self.final_mlp(U_embed) # [B, N_points, num_state_vars]

        # Residual connection: U_{k+1} = U_k + Update
        # This implies the LNO learns the *change* in U.
        val_out = val_in + val_out_raw # val_in is U_k

        return val_out

def train_lno_autoregressive(model, data_loader, dataset_type, train_nt_for_model, # train_nt_for_model is the number of timesteps in input sequences
                             lr=1e-3, num_epochs=100, device='cuda',
                             checkpoint_path='lno_ar_checkpoint.pt', clip_grad_norm=1.0):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    mse_loss = nn.MSELoss()

    start_epoch=0; best_loss=float('inf')
    if os.path.exists(checkpoint_path):
        print(f"Loading LNO AR checkpoint from {checkpoint_path}...")
        try:
            ckpt=torch.load(checkpoint_path,map_location=device); model.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                try: optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                except: print("Warn: LNO AR Optimizer state mismatch.")
            if 'scheduler_state_dict' in ckpt:
                try: scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                except: print("Warn: LNO AR Scheduler state mismatch.")
            start_epoch=ckpt.get('epoch',0)+1; best_loss=ckpt.get('loss',float('inf'))
            print(f"Resuming LNO AR training from epoch {start_epoch}")
        except Exception as e: print(f"Error loading LNO AR ckpt: {e}. Starting fresh.")

    # Spatial coordinates (fixed for each step)
    # model.nx and model.domain_L must be set before calling train
    if model.nx is None or model.domain_L is None:
        raise ValueError("model.nx and model.domain_L must be set before training.")

    x_coords_spatial = torch.linspace(0, model.domain_L, model.nx, device=device) # [nx]
    t_pseudo_input = torch.zeros_like(x_coords_spatial)    # [nx]
    t_pseudo_output = torch.ones_like(x_coords_spatial)   # [nx]

    # pos_in_step_template: [nx, 2] for (x, pseudo_t_input)
    pos_in_step_template = torch.stack([x_coords_spatial, t_pseudo_input], dim=-1).unsqueeze(0) # [1, nx, 2]
    # pos_out_step_template: [nx, 2] for (x, pseudo_t_output)
    pos_out_step_template = torch.stack([x_coords_spatial, t_pseudo_output], dim=-1).unsqueeze(0) # [1, nx, 2]


    for epoch in range(start_epoch, num_epochs):
        model.train(); epoch_loss_sum=0.0; num_steps_processed=0; batch_print_time_start=time.time()

        for i_batch,(state_data_loaded, _, _) in enumerate(data_loader): # BC_Ctrl not used by LNO stepper for now
            if isinstance(state_data_loaded,list):
                # Assuming each tensor in list is [B, train_nt, nx]
                state_seq_train=torch.stack(state_data_loaded,dim=-1).to(device)
            else: # Single variable like Advection/Burgers
                # state_data_loaded is [B, train_nt, nx]
                state_seq_train=state_data_loaded.unsqueeze(-1).to(device)

            B, nt_from_data, nx_from_data, num_vars_from_data = state_seq_train.shape
            if nx_from_data != model.nx:
                raise ValueError(f"Data nx {nx_from_data} != model.nx {model.nx}")
            if nt_from_data != train_nt_for_model: # Should match due to train_nt_limit in Dataset
                print(f"Warning: Data nt {nt_from_data} != train_nt_for_model {train_nt_for_model}. Using {nt_from_data}.")

            # Expand templates for batch
            pos_in_batch = pos_in_step_template.repeat(B, 1, 1)   # [B, nx, 2]
            pos_out_batch = pos_out_step_template.repeat(B, 1, 1) # [B, nx, 2]

            current_batch_total_loss = 0.0
            optimizer.zero_grad()


            for k_step in range(nt_from_data - 1):
                val_in_k_step = state_seq_train[:, k_step, :, :]      # U_k: [B, nx, num_vars]
                val_out_target_k_step = state_seq_train[:, k_step+1, :, :] # U_{k+1}: [B, nx, num_vars]

                # Predict U_{k+1} from U_k
                # LNO input: pos_in=[B,nx,2], val_in=[B,nx,num_vars], pos_out=[B,nx,2]
                val_out_pred_k_step = model(pos_in_batch, val_in_k_step, pos_out_batch) # [B, nx, num_vars]

                loss_k_step = mse_loss(val_out_pred_k_step, val_out_target_k_step)
                current_batch_total_loss += loss_k_step
                num_steps_processed += B # Count total individual step predictions

            if (nt_from_data -1) > 0:
                avg_loss_for_batch_seq = current_batch_total_loss / (nt_from_data - 1)
                avg_loss_for_batch_seq.backward() # Backpropagate average loss for the sequence

                if clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                optimizer.step()
                epoch_loss_sum += current_batch_total_loss.item() # Sum of losses over all steps in batch

            if (i_batch+1)%50==0: # Print frequency
                elapsed_batch_time = time.time() - batch_print_time_start
                avg_step_loss_in_batch = current_batch_total_loss.item() / max(1, (nt_from_data -1)*B)
                print(f" LNO_AR Ep {epoch+1} B {i_batch+1}/{len(data_loader)}, AvgStepLoss {avg_step_loss_in_batch:.3e}, LR {optimizer.param_groups[0]['lr']:.3e}, Time/20B {elapsed_batch_time:.2f}s")
                batch_print_time_start = time.time()

        avg_epoch_loss = epoch_loss_sum / max(1, num_steps_processed / B) # Avg loss per sequence in epoch
        scheduler.step(avg_epoch_loss) # For ReduceLROnPlateau

        print(f"LNO_AR Epoch {epoch+1}/{num_epochs} Avg Seq Loss: {avg_epoch_loss:.6f}")
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            print(f"Saving LNO_AR AR checkpoint with loss {best_loss:.6f}")
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
                'dataset_type': dataset_type,
                'model_nx': model.nx,
                'model_domain_L': model.domain_L,
                'model_state_keys': model.state_keys,
                'coord_dim': model.coord_dim,
                'M_latent_points': model.M_latent_points,
                'D_embedding': model.D_embedding,
                'L_transformer_blocks': model.L_transformer_blocks, # Now an attribute
                'transformer_nhead': model.transformer_nhead,       # Now an attribute
                'transformer_dim_feedforward': model.transformer_dim_feedforward, # Now an attribute
                'projector_hidden_dims': model.projector_hidden_dims, # Save architecture params
                'final_mlp_hidden_dims': model.final_mlp_hidden_dims, # Save architecture params
            }
            torch.save(save_dict, checkpoint_path)

    print("LNO_AR Training finished.")
    if os.path.exists(checkpoint_path):
        print(f"Loading best LNO_AR model from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    return model

def validate_lno_autoregressive(model, data_loader, dataset_type,
                                train_nt_for_model_training: int,
                                T_value_for_model_training: float,
                                full_T_in_datafile: float, full_nt_in_datafile: int,
                                dataset_params_for_plot: dict, device='cuda',
                                save_fig_path_prefix='lno_ar_result'):
    model.eval()
    state_keys_val = model.state_keys
    num_state_vars_val = model.num_state_vars
    nx_val = model.nx # nx from the model/dataset_params
    domain_L_val = model.domain_L

    test_horizons_T_values = [T_value_for_model_training]
    if full_T_in_datafile > T_value_for_model_training + 1e-5 :
        test_horizons_T_values.append(T_value_for_model_training + 0.5 * (full_T_in_datafile - T_value_for_model_training))
        test_horizons_T_values.append(full_T_in_datafile)
    test_horizons_T_values = sorted(list(set(h for h in test_horizons_T_values if h <= full_T_in_datafile + 1e-6)))

    print(f"LNO_AR Validation for T_horizons: {test_horizons_T_values}")
    print(f"  (Model was trained to predict one step ahead, using sequences up to T={T_value_for_model_training})")
    print(f"  (Datafile contains T={full_T_in_datafile}, nt={full_nt_in_datafile})")

    results_at_T_train = {key: {'mse':[],'rmse':[],'relative_error':[],'max_error':[]} for key in state_keys_val}
    overall_rel_err_at_T_train = []

    x_coords_spatial = torch.linspace(0, domain_L_val, nx_val, device=device)
    t_pseudo_input = torch.zeros_like(x_coords_spatial)
    t_pseudo_output = torch.ones_like(x_coords_spatial)
    pos_in_step_template = torch.stack([x_coords_spatial, t_pseudo_input], dim=-1).unsqueeze(0)
    pos_out_step_template = torch.stack([x_coords_spatial, t_pseudo_output], dim=-1).unsqueeze(0)

    with torch.no_grad():
        try:
            state_data_full_loaded, _, norm_factors_batch = next(iter(data_loader))
        except StopIteration: print("Val data_loader empty. Skipping LNO_AR validation."); return

        if isinstance(state_data_full_loaded, list):
            # state_data_full_loaded is a list of tensors, each [1, full_file_nt, nx_file]
            # Stack them to get [1, full_file_nt, nx_file, num_vars]
            stacked_data = torch.stack(state_data_full_loaded, dim=-1)
            # Squeeze batch dim to get [full_file_nt, nx_file, num_vars]
            gt_state_seq_norm_full = stacked_data.squeeze(0).to(device)
        else:
            # state_data_full_loaded is a tensor [1, full_file_nt, nx_file]
            # Add num_vars dim: [1, full_file_nt, nx_file, 1]
            # Squeeze batch dim: [full_file_nt, nx_file, 1]
            gt_state_seq_norm_full = state_data_full_loaded.unsqueeze(-1).squeeze(0).to(device)

        nt_file_actual_val, nx_file_actual_val, num_vars_actual_val = gt_state_seq_norm_full.shape
        # B_val is implicitly 1 for this single validation sample processing.

        if nx_file_actual_val != nx_val: # nx_val is model.nx set from dataset_params_for_plot
            raise ValueError(f"Validation data nx from file ({nx_file_actual_val}) != model.nx ({nx_val}) used for coordinate generation.")
        if num_vars_actual_val != num_state_vars_val: # num_state_vars_val is model.num_state_vars
             raise ValueError(f"Validation data num_vars ({num_vars_actual_val}) != model.num_state_vars ({num_state_vars_val}).")


        norm_factors_sample = {}
        for key_nf, val_tensor_nf in norm_factors_batch.items():
            if isinstance(val_tensor_nf, torch.Tensor) and val_tensor_nf.ndim > 0 and val_tensor_nf.shape[0] == 1:
                 norm_factors_sample[key_nf] = val_tensor_nf[0].cpu().numpy() if val_tensor_nf.is_cuda else val_tensor_nf[0].numpy()
            elif isinstance(val_tensor_nf, torch.Tensor):
                 norm_factors_sample[key_nf] = val_tensor_nf.cpu().numpy() if val_tensor_nf.is_cuda else val_tensor_nf.numpy()
            else:
                 norm_factors_sample[key_nf] = val_tensor_nf

        current_u_norm = gt_state_seq_norm_full[0, :, :].unsqueeze(0) # Shape [1, nx, num_vars]

        max_nt_horizon = int((max(test_horizons_T_values) / full_T_in_datafile) * (full_nt_in_datafile - 1)) + 1
        max_nt_horizon = min(max_nt_horizon, full_nt_in_datafile) # Cap by total nt in a full datafile sample
        max_nt_horizon = min(max_nt_horizon, nt_file_actual_val) # Cap by nt of the current loaded validation sample

        predictions_all_steps_norm = [current_u_norm.squeeze(0).clone()] # Store U_0 (squeezed to [nx, num_vars])

        print(f"  Performing autoregressive rollout for {max_nt_horizon-1} steps...")
        for k_rollout_step in range(max_nt_horizon - 1):
            # current_u_norm is U_k_norm, should be [1, nx, num_vars]
            pred_next_u_norm = model(pos_in_step_template, current_u_norm, pos_out_step_template) # Output [1, nx, num_vars]
            predictions_all_steps_norm.append(pred_next_u_norm.squeeze(0).clone()) # Store squeezed [nx, num_vars]
            current_u_norm = pred_next_u_norm # Update for next step, keep as [1, nx, num_vars]

        predictions_seq_norm_tensor = torch.stack(predictions_all_steps_norm, dim=0) # [max_nt_horizon, nx, num_vars]

        for T_horizon_current in test_horizons_T_values:
            nt_for_this_horizon = int((T_horizon_current / full_T_in_datafile) * (full_nt_in_datafile - 1)) + 1
            nt_for_this_horizon = min(nt_for_this_horizon, max_nt_horizon)
            nt_for_this_horizon = min(nt_for_this_horizon, nt_file_actual_val)

            print(f"\n  --- LNO_AR Validating for T_horizon = {T_horizon_current:.2f} (nt = {nt_for_this_horizon}) ---")

            pred_norm_sliced_h = predictions_seq_norm_tensor[:nt_for_this_horizon, :, :]
            gt_norm_sliced_h = gt_state_seq_norm_full[:nt_for_this_horizon, :, :]

            U_pred_denorm_h_dict = {}; U_gt_denorm_h_dict = {}
            pred_flat_list_h = []; gt_flat_list_h = []

            pred_np_h = pred_norm_sliced_h.cpu().numpy()
            gt_np_h = gt_norm_sliced_h.cpu().numpy()

            for k_var_idx, key_val_plot in enumerate(state_keys_val):
                mean_k_plot_val = norm_factors_sample.get(f'{key_val_plot}_mean', 0.0)
                std_k_plot_val = norm_factors_sample.get(f'{key_val_plot}_std', 1.0)

                mean_k_plot = float(mean_k_plot_val.item() if hasattr(mean_k_plot_val, 'item') and np.size(mean_k_plot_val)==1 else mean_k_plot_val)
                std_k_plot = float(std_k_plot_val.item() if hasattr(std_k_plot_val, 'item') and np.size(std_k_plot_val)==1 else std_k_plot_val)
                if abs(std_k_plot) < 1e-10: std_k_plot = 1e-8 if std_k_plot >=0 else -1e-8


                pred_norm_var_h = pred_np_h[:, :, k_var_idx]
                gt_norm_var_h = gt_np_h[:, :, k_var_idx]

                pred_denorm_var_h = pred_norm_var_h * std_k_plot + mean_k_plot
                gt_denorm_var_h = gt_norm_var_h * std_k_plot + mean_k_plot

                U_pred_denorm_h_dict[key_val_plot] = pred_denorm_var_h
                U_gt_denorm_h_dict[key_val_plot] = gt_denorm_var_h

                pred_flat_list_h.append(pred_denorm_var_h.flatten())
                gt_flat_list_h.append(gt_denorm_var_h.flatten())

                mse_k_h = np.mean((pred_denorm_var_h - gt_denorm_var_h)**2)
                rmse_k_h = np.sqrt(mse_k_h)
                rel_err_k_h = np.linalg.norm(pred_denorm_var_h - gt_denorm_var_h, 'fro') / (np.linalg.norm(gt_denorm_var_h, 'fro') + 1e-10)
                max_err_k_h = np.max(np.abs(pred_denorm_var_h - gt_denorm_var_h)) if pred_denorm_var_h.size > 0 else 0

                print(f"    '{key_val_plot}': MSE={mse_k_h:.3e}, RMSE={rmse_k_h:.3e}, RelErr={rel_err_k_h:.3e}, MaxErr={max_err_k_h:.3e}")

                if abs(T_horizon_current - T_value_for_model_training) < 1e-5:
                    results_at_T_train[key_val_plot]['mse'].append(mse_k_h)
                    results_at_T_train[key_val_plot]['rmse'].append(rmse_k_h)
                    results_at_T_train[key_val_plot]['relative_error'].append(rel_err_k_h)
                    results_at_T_train[key_val_plot]['max_error'].append(max_err_k_h)
            
            if not pred_flat_list_h or not gt_flat_list_h: 
                print(f"    Not enough steps to compute overall relative error for T={T_horizon_current:.1f}")
            else:
                pred_vec_h = np.concatenate(pred_flat_list_h)
                gt_vec_h = np.concatenate(gt_flat_list_h)
                overall_rel_err_h = np.linalg.norm(pred_vec_h - gt_vec_h) / (np.linalg.norm(gt_vec_h) + 1e-10)
                print(f"    Overall RelErr for T={T_horizon_current:.1f}: {overall_rel_err_h:.3e}")
                if abs(T_horizon_current - T_value_for_model_training) < 1e-5:
                    overall_rel_err_at_T_train.append(overall_rel_err_h)

            fig_L_plot = dataset_params_for_plot.get('L', 1.0)
            # fig_nx_plot = nx_val # Already have nx_val
            # fig_ny_plot = dataset_params_for_plot.get('ny', 1) # Already have ny_val

            fig, axs = plt.subplots(num_state_vars_val, 3, figsize=(18, 5 * num_state_vars_val), squeeze=False)
            for k_var_idx_plot, key_val_p in enumerate(state_keys_val):
                gt_plot_data = U_gt_denorm_h_dict[key_val_p]
                pred_plot_data = U_pred_denorm_h_dict[key_val_p]
                
                if gt_plot_data.shape[0] == 0 or pred_plot_data.shape[0] == 0 : 
                    for ax_idx in range(3):
                        axs[k_var_idx_plot, ax_idx].text(0.5, 0.5, "No data for this horizon", ha="center", va="center")
                        axs[k_var_idx_plot, ax_idx].set_xticks([])
                        axs[k_var_idx_plot, ax_idx].set_yticks([])
                    axs[k_var_idx_plot, 0].set_title(f"GT ({key_val_p})")
                    axs[k_var_idx_plot, 1].set_title(f"LNO_AR Pred ({key_val_p})")
                    axs[k_var_idx_plot, 2].set_title(f"Abs Error")
                    continue

                diff_plot_data = np.abs(pred_plot_data - gt_plot_data)
                max_err_plot = np.max(diff_plot_data) if diff_plot_data.size > 0 else 0.0

                vmin_plot = min(np.min(gt_plot_data) if gt_plot_data.size > 0 else 0,
                                np.min(pred_plot_data) if pred_plot_data.size > 0 else 0)
                vmax_plot = max(np.max(gt_plot_data) if gt_plot_data.size > 0 else 1,
                                np.max(pred_plot_data) if pred_plot_data.size > 0 else 1)
                if abs(vmin_plot - vmax_plot) < 1e-9 : vmax_plot = vmin_plot + 1e-5


                plot_extent = [0, fig_L_plot, 0, T_horizon_current]

                im0 = axs[k_var_idx_plot, 0].imshow(gt_plot_data, aspect='auto', origin='lower', vmin=vmin_plot, vmax=vmax_plot, extent=plot_extent, cmap='viridis')
                axs[k_var_idx_plot, 0].set_title(f"GT ({key_val_p})")
                plt.colorbar(im0, ax=axs[k_var_idx_plot, 0])

                im1 = axs[k_var_idx_plot, 1].imshow(pred_plot_data, aspect='auto', origin='lower', vmin=vmin_plot, vmax=vmax_plot, extent=plot_extent, cmap='viridis')
                axs[k_var_idx_plot, 1].set_title(f"LNO_AR Pred ({key_val_p})")
                plt.colorbar(im1, ax=axs[k_var_idx_plot, 1])

                im2 = axs[k_var_idx_plot, 2].imshow(diff_plot_data, aspect='auto', origin='lower', extent=plot_extent, cmap='magma')
                axs[k_var_idx_plot, 2].set_title(f"Abs Error (Max:{max_err_plot:.2e})")
                plt.colorbar(im2, ax=axs[k_var_idx_plot, 2])

                for j_plot in range(3): axs[k_var_idx_plot, j_plot].set_xlabel("x")
                axs[k_var_idx_plot, 0].set_ylabel("t (physical)")

            fig.suptitle(f"LNO_AR Autoregressive Validation ({dataset_type.capitalize()}) @ T={T_horizon_current:.1f}\nM_latent={model.M_latent_points}, D_embed={model.D_embedding}, Trained T_seq={T_value_for_model_training:.1f}")
            fig.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjusted rect to prevent title overlap
            current_horizon_fig_path = save_fig_path_prefix + f"_T{str(T_horizon_current).replace('.', 'p')}.png"
            plt.savefig(current_horizon_fig_path)
            print(f"    Saved LNO_AR validation plot to {current_horizon_fig_path}")
            plt.close(fig)

    print(f"\n--- LNO_AR Validation Summary (Metrics for Autoregressive Rollout up to T={T_value_for_model_training:.1f}) ---")
    for key_s in state_keys_val:
        if results_at_T_train[key_s]['mse']:
            avg_mse_s = np.mean(results_at_T_train[key_s]['mse'])
            avg_rmse_s = np.mean(results_at_T_train[key_s]['rmse']) # or np.sqrt(avg_mse_s)
            avg_rel_s = np.mean(results_at_T_train[key_s]['relative_error'])
            avg_max_s = np.mean(results_at_T_train[key_s]['max_error'])
            print(f"  Var '{key_s}': Avg MSE={avg_mse_s:.4e}, RMSE={avg_rmse_s:.4e}, RelErr={avg_rel_s:.4e}, MaxErr={avg_max_s:.4e}")
    if overall_rel_err_at_T_train:
        print(f"  Overall Avg RelErr for T={T_value_for_model_training:.1f}: {np.mean(overall_rel_err_at_T_train):.4e}")
    print("--------------------------------------")

# main script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Autoregressive LNO for PDE datasets.")
    parser.add_argument('--datatype', type=str, required=True, choices=['advection', 'euler', 'burgers', 'darcy'], help='Type of dataset.')
    args = parser.parse_args()

    DATASET_TYPE = args.datatype
    MODEL_TYPE = 'LNO_AR' # Indicate Autoregressive LNO

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Running {MODEL_TYPE} Baseline for {DATASET_TYPE} on {device} ---")

    # --- Time Parameters ---
    FULL_T_IN_DATAFILE = 2.0
    FULL_NT_IN_DATAFILE = 600 # Number of points (e.g., if dt=0.01, T=2, nt=201; if T=2, nt=600 => dt smaller)
    TRAIN_T_TARGET = 1.0 # The LNO will be trained on sequences of this physical time duration
    # TRAIN_NT_FOR_MODEL is the number of time *points* in the training sequences
    TRAIN_NT_FOR_MODEL = int((TRAIN_T_TARGET / FULL_T_IN_DATAFILE) * (FULL_NT_IN_DATAFILE - 1)) + 1

    print(f"Datafile T_max={FULL_T_IN_DATAFILE}, nt_max_points={FULL_NT_IN_DATAFILE}")
    print(f"Training sequences will have T_duration={TRAIN_T_TARGET}, nt_points={TRAIN_NT_FOR_MODEL}")

    # LNO Hyperparameters
    M_LATENT_POINTS = 64
    D_EMBEDDING = 64
    L_TRANSFORMER_BLOCKS = 3
    TRANSFORMER_NHEAD = 4
    TRANSFORMER_DIM_FF = D_EMBEDDING * 2
    PROJECTOR_HIDDEN_DIMS = [64, 128] # Smaller projectors
    FINAL_MLP_HIDDEN_DIMS = [128, 64]   # Smaller final MLP

    LEARNING_RATE = 3e-4 # Might need adjustment
    BATCH_SIZE = 16      # Sequences per batch
    NUM_EPOCHS = 80     # Autoregressive training might need more epochs or careful LR scheduling
    CLIP_GRAD_NORM = 1.0

    # --- Dataset Specific Parameters ---
    dataset_params_for_plot = {} # Store L, nx, ny etc. for plotting and model setup
    main_state_keys = []
    main_num_state_vars = 0
    COORD_DIM_LNO = 2 # (x_spatial, t_pseudo)

    if DATASET_TYPE == 'advection':
        dataset_path = "./datasets_full/advection_data_10000s_128nx_600nt.pkl"
        main_state_keys=['U']; main_num_state_vars=1
        dataset_params_for_plot={'nx':128,'ny':1,'L':1.0,'T_file':FULL_T_IN_DATAFILE, 'nt_file':FULL_NT_IN_DATAFILE}
    elif DATASET_TYPE == 'euler':
        dataset_path = "./datasets_full/euler_data_10000s_128nx_600nt.pkl"
        main_state_keys=['rho','u']; main_num_state_vars=2
        dataset_params_for_plot={'nx':128,'ny':1,'L':1.0,'T_file':FULL_T_IN_DATAFILE, 'nt_file':FULL_NT_IN_DATAFILE}
    elif DATASET_TYPE == 'burgers':
        dataset_path = "./datasets_full/burgers_data_10000s_128nx_600nt.pkl"
        main_state_keys=['U']; main_num_state_vars=1
        dataset_params_for_plot={'nx':128,'ny':1,'L':1.0,'T_file':FULL_T_IN_DATAFILE, 'nt_file':FULL_NT_IN_DATAFILE}
    elif DATASET_TYPE == 'darcy': # Darcy needs careful thought for "time-stepping" if it's pseudo-time
        dataset_path = "./datasets_full/darcy_data_10000s_128nx_600nt.pkl"
        main_state_keys=['P']; main_num_state_vars=1
        dataset_params_for_plot={'nx':128,'ny':1,'L':1.0,'T_file':FULL_T_IN_DATAFILE, 'nt_file':FULL_NT_IN_DATAFILE}
    else: raise ValueError(f"Unknown dataset type: {DATASET_TYPE}")

    print(f"Loading dataset: {dataset_path}")
    data_list_all = []
    try:
        with open(dataset_path,'rb') as f: data_list_all=pickle.load(f)
        print(f"Loaded {len(data_list_all)} samples.")
        if data_list_all:
            # Verify and potentially update FULL_NT_IN_DATAFILE and TRAIN_NT_FOR_MODEL based on actual file data
            first_sample_data = data_list_all[0][main_state_keys[0]]
            actual_file_nt = first_sample_data.shape[0]
            actual_file_nx = first_sample_data.shape[1]

            if actual_file_nt != FULL_NT_IN_DATAFILE:
                print(f"WARNING: Configured FULL_NT_IN_DATAFILE ({FULL_NT_IN_DATAFILE}) "
                      f"differs from actual nt in file ({actual_file_nt}). Using nt from file.")
                FULL_NT_IN_DATAFILE = actual_file_nt
                dataset_params_for_plot['nt_file'] = actual_file_nt
                TRAIN_NT_FOR_MODEL = int((TRAIN_T_TARGET / FULL_T_IN_DATAFILE) * (FULL_NT_IN_DATAFILE - 1)) + 1
                print(f" Adjusted TRAIN_NT_FOR_MODEL (points in sequence) to: {TRAIN_NT_FOR_MODEL}")

            if actual_file_nx != dataset_params_for_plot['nx']:
                 print(f"WARNING: Configured nx ({dataset_params_for_plot['nx']}) "
                      f"differs from actual nx in file ({actual_file_nx}). Using nx from file.")
                 dataset_params_for_plot['nx'] = actual_file_nx

            # Check for 'params' in sample to get L if not hardcoded
            if 'params' in data_list_all[0] and 'L' in data_list_all[0]['params']:
                dataset_params_for_plot['L'] = data_list_all[0]['params']['L']
                print(f"  Domain L set from file params: {dataset_params_for_plot['L']}")


    except FileNotFoundError: print(f"Error: File not found {dataset_path}"); exit()
    if not data_list_all: print("No data loaded. Exiting."); exit()

    random.shuffle(data_list_all); n_total=len(data_list_all); n_train=int(0.8*n_total)
    train_data_list_split=data_list_all[:n_train]; val_data_list_split=data_list_all[n_train:]
    print(f"Train samples: {len(train_data_list_split)}, Validation samples: {len(val_data_list_split)}")


    train_dataset = UniversalPDEDataset(train_data_list_split, dataset_type=DATASET_TYPE, train_nt_limit=TRAIN_NT_FOR_MODEL)
    val_dataset = UniversalPDEDataset(val_data_list_split, dataset_type=DATASET_TYPE, train_nt_limit=None) # Load full sequence for validation

    num_workers=1
    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=num_workers,pin_memory=True,drop_last=True)
    val_loader = DataLoader(val_dataset,batch_size=1,shuffle=False,num_workers=num_workers) # B=1 for validation plotting

    print(f"\nInitializing {MODEL_TYPE} model...")
    lno_model_autoregressive = LatentNeuralOperator(
        num_state_vars=main_num_state_vars,
        coord_dim=COORD_DIM_LNO, # (x, pseudo_t)
        M_latent_points=M_LATENT_POINTS,
        D_embedding=D_EMBEDDING,
        L_transformer_blocks=L_TRANSFORMER_BLOCKS,
        transformer_nhead=TRANSFORMER_NHEAD,
        transformer_dim_feedforward=TRANSFORMER_DIM_FF,
        projector_hidden_dims=PROJECTOR_HIDDEN_DIMS,
        final_mlp_hidden_dims=FINAL_MLP_HIDDEN_DIMS
    )
    # Set necessary attributes on the model instance
    lno_model_autoregressive.state_keys = main_state_keys
    lno_model_autoregressive.nx = dataset_params_for_plot['nx']
    lno_model_autoregressive.domain_L = dataset_params_for_plot['L']


    # --- File Paths ---
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{DATASET_TYPE}_{MODEL_TYPE}_trainT{TRAIN_T_TARGET}_M{M_LATENT_POINTS}_D{D_EMBEDDING}"
    checkpoint_dir = f"./New_ckpt_2/checkpoints_{DATASET_TYPE}_{MODEL_TYPE}" # Simplified path
    results_dir = f"./result_all_2/results_{DATASET_TYPE}_{MODEL_TYPE}"      # Simplified path
    os.makedirs(checkpoint_dir,exist_ok=True); os.makedirs(results_dir,exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'model_{run_name}.pt')
    save_fig_path_prefix = os.path.join(results_dir, f'result_{run_name}')


    print(f"\nStarting training for {MODEL_TYPE} on {DATASET_TYPE}...")
    print(f" Checkpoint will be saved to: {checkpoint_path}")
    start_train_time = time.time()

    lno_model_autoregressive = train_lno_autoregressive(
        lno_model_autoregressive, train_loader, dataset_type=DATASET_TYPE,
        train_nt_for_model=TRAIN_NT_FOR_MODEL, # num points in training sequences
        lr=LEARNING_RATE, num_epochs=NUM_EPOCHS, device=device,
        checkpoint_path=checkpoint_path, clip_grad_norm=CLIP_GRAD_NORM
    )
    end_train_time = time.time(); print(f"Training took {end_train_time - start_train_time:.2f} seconds.")

    if val_data_list_split:
        print(f"\nStarting validation for {MODEL_TYPE} (Autoregressive Rollout) on {DATASET_TYPE}...")
        validate_lno_autoregressive(
            lno_model_autoregressive, val_loader, dataset_type=DATASET_TYPE,
            train_nt_for_model_training=TRAIN_NT_FOR_MODEL, # Num points in sequences model saw during training
            T_value_for_model_training=TRAIN_T_TARGET,    # Physical time duration of these sequences
            full_T_in_datafile=FULL_T_IN_DATAFILE,
            full_nt_in_datafile=FULL_NT_IN_DATAFILE,
            dataset_params_for_plot=dataset_params_for_plot, # Contains L, nx, ny etc.
            device=device,
            save_fig_path_prefix=save_fig_path_prefix
        )
    else: print("\nNo validation data. Skipping validation.")

    print("="*60);
    print(f"Run finished: {run_name}");
    print(f"Final checkpoint: {checkpoint_path}");
    if val_data_list_split: print(f"Validation figures saved with prefix: {save_fig_path_prefix}");
    print("="*60)
