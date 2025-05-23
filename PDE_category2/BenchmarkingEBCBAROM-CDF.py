import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim # Not strictly needed for benchmark, but often in model files
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F # Not strictly needed for benchmark, but often in model files
import matplotlib.pyplot as plt
import random
import time
import pickle
import argparse
import glob
import traceback

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

# ---------------------
# UniversalPDEDataset (Copied from your reference)
# ---------------------
class UniversalPDEDataset(Dataset):
    def __init__(self, data_list, dataset_type, train_nt_limit=None):
        if not data_list: raise ValueError("data_list cannot be empty")
        self.data_list = data_list; self.dataset_type = dataset_type; self.train_nt_limit = train_nt_limit
        first_sample = data_list[0]

        if dataset_type == 'heat_delayed_feedback':
            self.nt_from_sample = first_sample['U'].shape[0]; self.nx_from_sample = first_sample['U'].shape[1]; self.ny_from_sample = 1
            self.state_keys = ['U']; self.num_state_vars = 1; self.nx = self.nx_from_sample; self.ny = self.ny_from_sample
            self.expected_bc_state_dim = 2
        elif dataset_type == 'reaction_diffusion_neumann_feedback':
            self.nt_from_sample = first_sample['U'].shape[0]; self.nx_from_sample = first_sample['U'].shape[1]; self.ny_from_sample = 1
            self.state_keys = ['U']; self.num_state_vars = 1; self.nx = self.nx_from_sample; self.ny = self.ny_from_sample
            self.expected_bc_state_dim = 2
        elif dataset_type == 'heat_nonlinear_feedback_gain':
            self.nt_from_sample = first_sample['U'].shape[0]; self.nx_from_sample = first_sample['U'].shape[1]; self.ny_from_sample = 1
            self.state_keys = ['U']; self.num_state_vars = 1; self.nx = self.nx_from_sample; self.ny = self.ny_from_sample
            self.expected_bc_state_dim = 2
        elif dataset_type == 'convdiff':
            self.nt_from_sample = first_sample['U'].shape[0]; self.nx_from_sample = first_sample['U'].shape[1]; self.state_keys = ['U']
            self.num_state_vars = 1; self.nx = self.nx_from_sample; self.ny = 1; self.expected_bc_state_dim = 2
        else: raise ValueError(f"Unknown dataset_type: {dataset_type}")

        self.effective_nt = self.train_nt_limit if self.train_nt_limit is not None else self.nt_from_sample
        self.spatial_dim = self.nx * self.ny
        self.bc_state_key = 'BC_State'
        if self.bc_state_key not in first_sample: raise KeyError(f"'{self.bc_state_key}' not found!")
        actual_bc_state_dim = first_sample[self.bc_state_key].shape[1]
        self.bc_state_dim = actual_bc_state_dim
        if actual_bc_state_dim != self.expected_bc_state_dim:
                 print(f"Info: For {dataset_type}, expected BC_State dim {self.expected_bc_state_dim}, got {actual_bc_state_dim}.")

        self.bc_control_key = 'BC_Control'
        if self.bc_control_key in first_sample and first_sample[self.bc_control_key] is not None and \
           hasattr(first_sample[self.bc_control_key], 'ndim') and first_sample[self.bc_control_key].ndim == 2:
            self.num_controls = first_sample[self.bc_control_key].shape[1]
        else: self.num_controls = 0
        if self.num_controls == 0 :
            print(f"Info: num_controls is 0 for dataset {dataset_type} based on first sample (key: '{self.bc_control_key}').")

    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx):
        sample = self.data_list[idx]; norm_factors = {}; current_nt = self.effective_nt
        state_tensors_norm_list = []
        for key in self.state_keys:
            state_seq_full = sample[key]; state_seq = state_seq_full[:current_nt, ...]
            state_mean = np.mean(state_seq); state_std = np.std(state_seq) + 1e-8
            state_norm = (state_seq - state_mean) / state_std
            state_tensors_norm_list.append(torch.tensor(state_norm).float())
            norm_factors[f'{key}_mean'] = state_mean; norm_factors[f'{key}_std'] = state_std

        bc_state_seq_full = sample[self.bc_state_key]; bc_state_seq = bc_state_seq_full[:current_nt, :]
        bc_state_norm = np.zeros_like(bc_state_seq, dtype=np.float32)
        norm_factors[f'{self.bc_state_key}_means'] = np.zeros(self.bc_state_dim); norm_factors[f'{self.bc_state_key}_stds'] = np.ones(self.bc_state_dim)
        if self.bc_state_dim > 0:
            for k_dim in range(self.bc_state_dim):
                col = bc_state_seq[:, k_dim]; mean_k = np.mean(col); std_k = np.std(col)
                if std_k > 1e-8:
                    bc_state_norm[:, k_dim] = (col - mean_k) / std_k
                    norm_factors[f'{self.bc_state_key}_means'][k_dim] = mean_k; norm_factors[f'{self.bc_state_key}_stds'][k_dim] = std_k
                else:
                    bc_state_norm[:, k_dim] = col - mean_k; norm_factors[f'{self.bc_state_key}_means'][k_dim] = mean_k
        bc_state_tensor_norm = torch.tensor(bc_state_norm).float()

        if self.num_controls > 0:
            bc_control_seq_full = sample[self.bc_control_key]
            if bc_control_seq_full.ndim == 1 and self.num_controls == 1: # Handle if control is 1D but num_controls is 1
                bc_control_seq_full = bc_control_seq_full[:, np.newaxis]
            bc_control_seq = bc_control_seq_full[:current_nt, :]
            bc_control_norm = np.zeros_like(bc_control_seq, dtype=np.float32)
            norm_factors[f'{self.bc_control_key}_means'] = np.zeros(self.num_controls); norm_factors[f'{self.bc_control_key}_stds'] = np.ones(self.num_controls)
            for k_dim in range(self.num_controls):
                col = bc_control_seq[:, k_dim]; mean_k = np.mean(col); std_k = np.std(col)
                if std_k > 1e-8:
                    bc_control_norm[:, k_dim] = (col - mean_k) / std_k
                    norm_factors[f'{self.bc_control_key}_means'][k_dim] = mean_k; norm_factors[f'{self.bc_control_key}_stds'][k_dim] = std_k
                else:
                    bc_control_norm[:, k_dim] = col - mean_k; norm_factors[f'{self.bc_control_key}_means'][k_dim] = mean_k
            bc_control_tensor_norm = torch.tensor(bc_control_norm).float()
        else: bc_control_tensor_norm = torch.empty((current_nt, 0), dtype=torch.float32)

        bc_ctrl_tensor_norm = torch.cat((bc_state_tensor_norm, bc_control_tensor_norm), dim=-1)
        return state_tensors_norm_list, bc_ctrl_tensor_norm, norm_factors

# ---------------------
# ImprovedUpdateFFN (Copied from your reference)
# ---------------------
class ImprovedUpdateFFN(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=3, dropout=0.1, output_dim=None):
        super().__init__(); layers = []; current_dim = input_dim
        if output_dim is None: output_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim)); layers.append(nn.GELU()); layers.append(nn.Dropout(dropout)); current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim)); self.mlp = nn.Sequential(*layers)
        self.layernorm = nn.LayerNorm(output_dim); self.input_dim_for_residual = input_dim; self.output_dim_for_residual = output_dim
    def forward(self, x):
        mlp_out = self.mlp(x)
        if self.input_dim_for_residual == self.output_dim_for_residual: return self.layernorm(mlp_out + x)
        else: return self.layernorm(mlp_out)

# ---------------------
# UniversalLifting (Copied from your reference)
# ---------------------
class UniversalLifting(nn.Module):
    def __init__(self, num_state_vars, bc_state_dim, num_controls, output_dim_per_var, nx, hidden_dims_state_branch=64, hidden_dims_control=[64,128], hidden_dims_fusion=[256,512,256], dropout=0.1):
        super().__init__(); self.num_state_vars=num_state_vars; self.bc_state_dim=bc_state_dim; self.num_controls=num_controls; self.nx=nx
        assert output_dim_per_var == nx
        self.state_branches=nn.ModuleList(); state_feature_dim=0
        if self.bc_state_dim > 0:
            for _ in range(bc_state_dim): self.state_branches.append(nn.Sequential(nn.Linear(1,hidden_dims_state_branch), nn.GELU()))
            state_feature_dim=bc_state_dim*hidden_dims_state_branch
        control_feature_dim=0; self.control_mlp=nn.Sequential()
        if self.num_controls > 0:
            control_layers=[]; current_dim_ctrl=num_controls
            for h_dim in hidden_dims_control: control_layers.append(nn.Linear(current_dim_ctrl,h_dim)); control_layers.append(nn.GELU()); control_layers.append(nn.Dropout(dropout)); current_dim_ctrl=h_dim
            self.control_mlp=nn.Sequential(*control_layers); control_feature_dim=current_dim_ctrl
        fusion_input_dim=state_feature_dim+control_feature_dim; fusion_layers=[]
        if fusion_input_dim > 0:
            current_dim_fusion=fusion_input_dim
            for h_dim in hidden_dims_fusion: fusion_layers.append(nn.Linear(current_dim_fusion,h_dim)); fusion_layers.append(nn.GELU()); fusion_layers.append(nn.Dropout(dropout)); current_dim_fusion=h_dim
            fusion_layers.append(nn.Linear(current_dim_fusion, num_state_vars*nx)); self.fusion=nn.Sequential(*fusion_layers)
        else: self.fusion=None
    def forward(self, BC_Ctrl):
        if self.fusion is None: batch_size=BC_Ctrl.shape[0] if BC_Ctrl is not None and BC_Ctrl.nelement()>0 else 1; return torch.zeros(batch_size,self.num_state_vars,self.nx,device=BC_Ctrl.device if BC_Ctrl is not None else 'cpu')
        features_to_concat=[];
        if self.bc_state_dim > 0:
            BC_state=BC_Ctrl[:,:self.bc_state_dim]; state_features_list=[]
            for i in range(self.bc_state_dim):branch_out=self.state_branches[i](BC_state[:,i:i+1]); state_features_list.append(branch_out)
            features_to_concat.append(torch.cat(state_features_list,dim=-1))
        if self.num_controls > 0: features_to_concat.append(self.control_mlp(BC_Ctrl[:,self.bc_state_dim:]))
        if not features_to_concat: batch_size=BC_Ctrl.shape[0] if BC_Ctrl is not None and BC_Ctrl.nelement()>0 else 1; return torch.zeros(batch_size,self.num_state_vars,self.nx,device=BC_Ctrl.device if BC_Ctrl is not None else 'cpu')
        concat_features = features_to_concat[0] if len(features_to_concat)==1 else torch.cat(features_to_concat,dim=-1)
        return self.fusion(concat_features).view(-1,self.num_state_vars,self.nx)

# ---------------------
# MultiHeadAttentionROM (Base class, copied from your reference)
# ---------------------
class MultiHeadAttentionROM(nn.Module):
    def __init__(self, basis_dim, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads
        self.out_proj = nn.Linear(d_model, d_model)
    def forward(self, Q, K, V):
        batch_size, seq_len_q, d_model_q = Q.size()
        _, seq_len_kv, d_model_kv = K.size()
        assert seq_len_q == seq_len_kv, "Sequence lengths of Q and K/V must match for this attention"
        assert d_model_q == d_model_kv, "d_model of Q and K/V must match"
        seq_len = seq_len_q; d_model = d_model_q
        Q_reshaped = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K_reshaped = K.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V_reshaped = V.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        KV = torch.matmul(K_reshaped.transpose(-2, -1), V_reshaped)
        z = torch.matmul(Q_reshaped, KV)
        z = z.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, d_model)
        z = self.out_proj(z)
        return z

# ----------------------------------------------------------------------------------
# MultiVarAttentionROMEq9: This is your ROM model with Equation 9 modifications
# ----------------------------------------------------------------------------------
class MultiVarAttentionROMEq9(nn.Module): # Renamed for clarity in this benchmark script
    def __init__(self, state_variable_keys, nx, basis_dim, d_model,
                 bc_state_dim, num_controls, num_heads=8,
                 add_error_estimator=False, shared_attention=False,
                 dropout_lifting=0.1, dropout_ffn=0.1, initial_alpha=0.1,
                 use_fixed_lifting=False,
                 bc_processed_dim=64,
                 hidden_bc_processor_dim=128
                 ):
        super().__init__()
        self.state_keys = state_variable_keys
        self.num_state_vars = len(state_variable_keys)
        self.nx = nx
        self.basis_dim = basis_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.bc_state_dim = bc_state_dim
        self.num_controls = num_controls
        self.add_error_estimator = add_error_estimator
        self.shared_attention = shared_attention
        self.use_fixed_lifting = use_fixed_lifting
        self.bc_processed_dim = bc_processed_dim
        self.hidden_bc_processor_dim = hidden_bc_processor_dim # Store this

        self.Phi = nn.ParameterDict()
        for key in self.state_keys:
            phi_param = nn.Parameter(torch.randn(nx, basis_dim))
            nn.init.orthogonal_(phi_param)
            self.Phi[key] = phi_param

        if not self.use_fixed_lifting:
            self.lifting = UniversalLifting(
                num_state_vars=self.num_state_vars, bc_state_dim=bc_state_dim,
                num_controls=num_controls, output_dim_per_var=nx, nx=nx,
                dropout=dropout_lifting
            )
        else:
            self.lifting = None
            lin_interp_coeffs = torch.linspace(0, 1, self.nx, dtype=torch.float32)
            self.register_buffer('lin_interp_coeffs', lin_interp_coeffs.view(1, 1, -1))

        self.W_Q = nn.ModuleDict(); self.W_K = nn.ModuleDict(); self.W_V = nn.ModuleDict()
        self.multihead_attn = nn.ModuleDict(); self.proj_to_coef = nn.ModuleDict()
        self.update_ffn = nn.ModuleDict(); self.a0_mapping = nn.ModuleDict()
        self.alphas = nn.ParameterDict()
        self.bc_feature_processor = nn.ModuleDict(); self.bc_to_a_update = nn.ModuleDict()
        total_bc_ctrl_dim = self.bc_state_dim + self.num_controls

        if shared_attention:
            self.W_Q['shared'] = nn.Linear(1, d_model)
            self.W_K['shared'] = nn.Linear(nx, d_model)
            self.W_V['shared'] = nn.Linear(nx, d_model)
            self.multihead_attn['shared'] = MultiHeadAttentionROM(basis_dim, d_model, num_heads)
            self.proj_to_coef['shared'] = nn.Linear(d_model, 1)
            self.update_ffn['shared'] = ImprovedUpdateFFN(input_dim=basis_dim, output_dim=basis_dim, hidden_dim=d_model, dropout=dropout_ffn)
            self.a0_mapping['shared'] = nn.Sequential(nn.Linear(basis_dim, basis_dim), nn.ReLU(), nn.LayerNorm(basis_dim))
            self.alphas['shared'] = nn.Parameter(torch.tensor(initial_alpha))
            if total_bc_ctrl_dim > 0:
                self.bc_feature_processor['shared'] = nn.Sequential(nn.Linear(total_bc_ctrl_dim, hidden_bc_processor_dim), nn.GELU(), nn.Linear(hidden_bc_processor_dim, self.bc_processed_dim))
                self.bc_to_a_update['shared'] = nn.Linear(self.bc_processed_dim, basis_dim)
            else:
                self.bc_feature_processor['shared'] = nn.Sequential()
                self.bc_to_a_update['shared'] = nn.Linear(0, basis_dim) if self.bc_processed_dim == 0 else nn.Linear(self.bc_processed_dim, basis_dim)
        else:
            for key in self.state_keys:
                self.W_Q[key] = nn.Linear(1, d_model); self.W_K[key] = nn.Linear(nx, d_model); self.W_V[key] = nn.Linear(nx, d_model)
                self.multihead_attn[key] = MultiHeadAttentionROM(basis_dim, d_model, num_heads); self.proj_to_coef[key] = nn.Linear(d_model, 1)
                self.update_ffn[key] = ImprovedUpdateFFN(input_dim=basis_dim, output_dim=basis_dim, hidden_dim=d_model, dropout=dropout_ffn)
                self.a0_mapping[key] = nn.Sequential(nn.Linear(basis_dim, basis_dim), nn.ReLU(), nn.LayerNorm(basis_dim))
                self.alphas[key] = nn.Parameter(torch.tensor(initial_alpha))
                if total_bc_ctrl_dim > 0:
                    self.bc_feature_processor[key] = nn.Sequential(nn.Linear(total_bc_ctrl_dim, hidden_bc_processor_dim), nn.GELU(), nn.Linear(hidden_bc_processor_dim, self.bc_processed_dim))
                    self.bc_to_a_update[key] = nn.Linear(self.bc_processed_dim, basis_dim)
                else:
                    self.bc_feature_processor[key] = nn.Sequential()
                    self.bc_to_a_update[key] = nn.Linear(0, basis_dim) if self.bc_processed_dim == 0 else nn.Linear(self.bc_processed_dim, basis_dim)

        if self.add_error_estimator:
            self.error_estimator = nn.Linear(self.num_state_vars * basis_dim, 1)

    def _get_layer(self, module_dict, key):
        return module_dict['shared'] if self.shared_attention else module_dict[key]
    def _get_alpha(self, key):
        return self.alphas['shared'] if self.shared_attention else self.alphas[key]
    def _compute_U_B(self, BC_Ctrl_step):
        if not self.use_fixed_lifting: return self.lifting(BC_Ctrl_step)
        else:
            if self.num_state_vars == 1:
                if self.bc_state_dim < 2: return torch.zeros(BC_Ctrl_step.shape[0],1,self.nx,device=BC_Ctrl_step.device)
                bc_l=BC_Ctrl_step[:,0:1].unsqueeze(-1); bc_r=BC_Ctrl_step[:,1:2].unsqueeze(-1)
                return bc_l*(1-self.lin_interp_coeffs)+bc_r*self.lin_interp_coeffs
            else:
                U_B_l=[];
                if self.bc_state_dim<2*self.num_state_vars and self.num_state_vars > 0: return torch.zeros(BC_Ctrl_step.shape[0],self.num_state_vars,self.nx,device=BC_Ctrl_step.device)
                for i in range(self.num_state_vars):
                    idl=i*2; idr=i*2+1
                    if idr>=self.bc_state_dim: U_B_s=torch.zeros(BC_Ctrl_step.shape[0],1,self.nx,device=BC_Ctrl_step.device)
                    else: bc_l=BC_Ctrl_step[:,idl:idl+1].unsqueeze(-1); bc_r=BC_Ctrl_step[:,idr:idr+1].unsqueeze(-1); U_B_s=bc_l*(1-self.lin_interp_coeffs)+bc_r*self.lin_interp_coeffs
                    U_B_l.append(U_B_s)
                return torch.cat(U_B_l,dim=1)

    def forward_step(self, a_n_dict, BC_Ctrl_n, U_B_np1_stacked, params=None):
        batch_size = list(a_n_dict.values())[0].size(0); a_next_dict={}; U_hat_dict={}
        U_B_n_stacked = self._compute_U_B(BC_Ctrl_n) # This is U_B^n from Eq. (9)
        bc_features_processed_dict = {}
        total_bc_ctrl_dim = self.bc_state_dim + self.num_controls
        if total_bc_ctrl_dim > 0:
            if self.shared_attention:
                bc_proc_layer=self._get_layer(self.bc_feature_processor,'shared')
                if hasattr(bc_proc_layer,'weight') or len(list(bc_proc_layer.parameters()))>0: shared_bc_f=bc_proc_layer(BC_Ctrl_n); [bc_features_processed_dict.update({k:shared_bc_f}) for k in self.state_keys]
                else: [bc_features_processed_dict.update({k:None}) for k in self.state_keys]
            else:
                for k_loop in self.state_keys:
                    bc_proc_layer=self._get_layer(self.bc_feature_processor,k_loop)
                    if hasattr(bc_proc_layer,'weight') or len(list(bc_proc_layer.parameters()))>0: bc_features_processed_dict[k_loop]=bc_proc_layer(BC_Ctrl_n)
                    else: bc_features_processed_dict[k_loop]=None
        else: [bc_features_processed_dict.update({k:None}) for k in self.state_keys]

        for i_var, key in enumerate(self.state_keys):
            a_n_var=a_n_dict[key]; Phi_var=self.Phi[key]
            W_Q_v=self._get_layer(self.W_Q,key); W_K_v=self._get_layer(self.W_K,key); W_V_v=self._get_layer(self.W_V,key)
            attn_mod_v=self._get_layer(self.multihead_attn,key); proj_v=self._get_layer(self.proj_to_coef,key)
            ffn_v=self._get_layer(self.update_ffn,key); alpha_v=self._get_alpha(key)
            bc_to_a_upd_l=self._get_layer(self.bc_to_a_update,key)

            Phi_basis_vecs=Phi_var.transpose(0,1).unsqueeze(0).expand(batch_size,-1,-1)
            K_f=W_K_v(Phi_basis_vecs.reshape(-1,self.nx)); V_f=W_V_v(Phi_basis_vecs.reshape(-1,self.nx))
            K=K_f.view(batch_size,self.basis_dim,self.d_model); V=V_f.view(batch_size,self.basis_dim,self.d_model)
            a_n_usq_Q=a_n_var.unsqueeze(-1); Q_b=W_Q_v(a_n_usq_Q)
            ffn_upd_int_val=ffn_v(a_n_var); Q_for_attn=Q_b # Simpler Q
            z=attn_mod_v(Q_for_attn,K,V)/np.sqrt(float(self.d_model))
            # a_update_attn_val: part of learned equivalent of (\hat{A}_a - I)a^n
            a_update_attn_val=proj_v(z.reshape(-1,self.d_model)).view(batch_size,self.basis_dim)
            # bc_driven_a_update_val: learned equivalent of \hat{B}\hat{w}^n + \hat{A}_{BC}U_B^n
            bc_driven_a_update_val=torch.zeros_like(a_n_var)
            curr_bc_feat=bc_features_processed_dict[key]
            if curr_bc_feat is not None and (hasattr(bc_to_a_upd_l,'weight')or len(list(bc_to_a_upd_l.parameters()))>0): bc_driven_a_update_val=bc_to_a_upd_l(curr_bc_feat)

            # Term: \Phi^T U_B^{n+1} from Eq. (9)
            Phi_T_exp=Phi_var.transpose(0,1).unsqueeze(0).expand(batch_size,-1,-1)
            U_B_np1_curr_var=U_B_np1_stacked[:,i_var,:].unsqueeze(-1)
            term_sub_phiT_UBnp1=torch.bmm(Phi_T_exp,U_B_np1_curr_var).squeeze(-1)

            # Total update for a^{n+1} (Paper Eq. (9), first line in \hat{\Sigma}_lin)
            a_next_var=a_n_var + a_update_attn_val + alpha_v*ffn_upd_int_val + bc_driven_a_update_val - term_sub_phiT_UBnp1
            a_next_dict[key]=a_next_var # This is a^{n+1}

            # Reconstruction of \hat{U}^{n+1} (Paper Eq. (9), second line in \hat{\Sigma}_lin, for time n+1)
            # $\hat{U}^{n+1} = U_B^{n+1} + \Phi a^{n+1}$
            U_B_np1_recon_var=U_B_np1_stacked[:,i_var,:].unsqueeze(-1) # This is U_B^{n+1}
            Phi_exp=Phi_var.unsqueeze(0).expand(batch_size,-1,-1); a_next_usq=a_next_var.unsqueeze(-1)
            Phi_a_next=torch.bmm(Phi_exp,a_next_usq) # This is \Phi a^{n+1}
            U_hat_dict[key]=U_B_np1_recon_var + Phi_a_next # This is \hat{U}^{n+1}
        err_est=None
        if self.add_error_estimator and a_next_dict:
            a_next_comb=torch.cat(list(a_next_dict.values()),dim=-1)
            if hasattr(self,'error_estimator') and self.error_estimator is not None: err_est=self.error_estimator(a_next_comb)
        return a_next_dict,U_hat_dict,err_est

    def forward(self, a0_dict, BC_Ctrl_seq, T, params=None):
        a_curr_dict={}; [a_curr_dict.update({k_init:self._get_layer(self.a0_mapping,k_init)(a0_dict[k_init])}) for k_init in self.state_keys]
        U_hat_seq_dict={k:[] for k in self.state_keys}; err_seq=[] if self.add_error_estimator else None
        # BC_Ctrl_seq needs to be long enough to provide BC_Ctrl^{n+1} for each step n=0..T-1
        # So, BC_Ctrl_seq should have T+1 elements if T is number of steps (indices 0 to T)
        if T > 0 and BC_Ctrl_seq.shape[1] < T + 1:
            print(f"Warning in model.forward: BC_Ctrl_seq length ({BC_Ctrl_seq.shape[1]}) is less than required T+1 ({T+1}) for {T} steps. "
                  "U_B^{n+1} might be zero-padded for the last step(s).")

        for t_s in range(T): # t_s is current time index 'n'
            BC_Ctrl_n=BC_Ctrl_seq[:,t_s,:] # BC_Ctrl^n
            if t_s+1 < BC_Ctrl_seq.shape[1]: # Check if BC_Ctrl^{n+1} is available
                BC_Ctrl_np1=BC_Ctrl_seq[:,t_s+1,:]
                U_B_np1_stack=self._compute_U_B(BC_Ctrl_np1) # This is U_B^{n+1}
            else: # Fallback for U_B^{n+1} if BC_Ctrl^{n+1} is not in sequence
                # print(f"Warning: Forward step {t_s+1}/{T}: BC_Ctrl for t_s+1 not in BC_Ctrl_seq. Using zeros for U_B^{t_s+1}.")
                ref_dev=BC_Ctrl_n.device; ref_dtype=BC_Ctrl_n.dtype; curr_bs=BC_Ctrl_n.shape[0]
                U_B_np1_stack=torch.zeros(curr_bs,self.num_state_vars,self.nx,device=ref_dev,dtype=ref_dtype)
            a_next_d_s,U_hat_d_s,err_e_s=self.forward_step(a_curr_dict,BC_Ctrl_n,U_B_np1_stack,params)
            [U_hat_seq_dict[k_store].append(U_hat_d_s[k_store]) for k_store in self.state_keys]
            if self.add_error_estimator and err_e_s is not None: err_seq.append(err_e_s)
            a_curr_dict=a_next_d_s
        for k_out in self.state_keys:
            if U_hat_seq_dict[k_out]: U_hat_seq_dict[k_out]=torch.stack(U_hat_seq_dict[k_out],dim=0).squeeze(-1).permute(1,0,2) # [B, T, Nx]
            else: U_hat_seq_dict[k_out]=torch.empty(BC_Ctrl_seq.shape[0] if BC_Ctrl_seq is not None else 1,0,self.nx,device=BC_Ctrl_seq.device if BC_Ctrl_seq is not None else 'cpu')
        if self.add_error_estimator and err_seq and err_seq[0] is not None: err_seq=torch.stack(err_seq,dim=1)
        return U_hat_seq_dict,err_seq

    def get_basis(self,key): return self.Phi[key]

# =============================================================================
# BENCHMARKING SCRIPT SPECIFIC CODE
# =============================================================================
# --- Configuration ---
SEED = 42; DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DATASET_PATH = "./datasets_new_feedback/"
# Use a different checkpoint path for your new model
BASE_CHECKPOINT_PATH = "./New_ckpt_explicit_bc_eq9/" # Adjusted for Eq9 model
BASE_RESULTS_PATH = "./benchmark_MultiVarROMEq9_results/"
os.makedirs(BASE_RESULTS_PATH, exist_ok=True)
MAX_VISUALIZATION_SAMPLES = 3 # Reduced for quicker benchmark runs

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# Define the target dataset and model configuration for *this* benchmark run
# This should match one of the datasets your MultiVarAttentionROMEq9 was trained on.
TARGET_DATASET_KEY_BENCHMARK = 'convdiff' # Example

DATASET_CONFIGS = {
    'convdiff': { # Ensure this key matches TARGET_DATASET_KEY_BENCHMARK
        'path': os.path.join(BASE_DATASET_PATH, "convdiff_v1_5000s_64nx_300nt.pkl"),
        'T_file': 2.0, 'NT_file': 300, # NT_file is number of points (0 to 299)
        'nx': 64, 'L': 1.0, 'state_keys': ['U'], 'num_state_vars': 1,
        'dataset_type_arg': 'convdiff' # For UniversalPDEDataset
    }
    # Add other dataset configs if you plan to benchmark on them
}
if TARGET_DATASET_KEY_BENCHMARK not in DATASET_CONFIGS:
    raise ValueError(f"TARGET_DATASET_KEY_BENCHMARK '{TARGET_DATASET_KEY_BENCHMARK}' not found in DATASET_CONFIGS.")


# Model configuration for YOUR MultiVarAttentionROMEq9 model
# The checkpoint_filename should match how your training script saves it.
# Example: run_name = f"{DATASET_TYPE}_b{args.basis_dim}{suffix_rom}{suffix_lift}{suffix_phi}_h{args.num_heads}_eq9_{timestamp}"
# For this benchmark, you need to provide the *exact* trained checkpoint filename.
# Let's assume some default training parameters for constructing the expected filename.
ASSUMED_TRAIN_BASIS_DIM = 32
ASSUMED_TRAIN_D_MODEL = 512 # Example, match your training
ASSUMED_TRAIN_NUM_HEADS = 8   # Example
ASSUMED_TRAIN_BC_PROC_DIM = 32 # Example
ASSUMED_TRAIN_SHARED_ATTN = False # Example
ASSUMED_TRAIN_FIXED_LIFT = False # Example
ASSUMED_TRAIN_RAND_PHI = False # Example

suffix_rom_bench = f"_attn_d{ASSUMED_TRAIN_D_MODEL}_bcp{ASSUMED_TRAIN_BC_PROC_DIM}"
suffix_lift_bench = "_fixedlift" if ASSUMED_TRAIN_FIXED_LIFT else ""
suffix_phi_bench = "_randphi" if ASSUMED_TRAIN_RAND_PHI else ""
# This is a *template* for the filename. You might need to find the exact one.
# The timestamp part will make it tricky, so ideally load the LATEST trained model for that config.
# For simplicity, let's assume you can find the exact filename or use a wildcard.
# Constructing part of the expected filename:
EXPECTED_CKPT_PREFIX = f"barom_{TARGET_DATASET_KEY_BENCHMARK}_b{ASSUMED_TRAIN_BASIS_DIM}{suffix_rom_bench}{suffix_lift_bench}{suffix_phi_bench}_h{ASSUMED_TRAIN_NUM_HEADS}_eq9_v2"


MODEL_TO_BENCHMARK_CONFIG = {
    "MultiVarROMEq9_Default": { # A unique name for this benchmark configuration
        "model_class_name": "MultiVarAttentionROMEq9",
        "checkpoint_filename_prefix": EXPECTED_CKPT_PREFIX, # Will use glob to find the latest
        "params": { # These are primarily for re-instantiation if not in checkpoint, or for reference
            "basis_dim": ASSUMED_TRAIN_BASIS_DIM,
            "d_model": ASSUMED_TRAIN_D_MODEL,
            "num_heads": ASSUMED_TRAIN_NUM_HEADS,
            "use_fixed_lifting": ASSUMED_TRAIN_FIXED_LIFT,
            "bc_processed_dim": ASSUMED_TRAIN_BC_PROC_DIM,
            "hidden_bc_processor_dim": 128, # Default from your class
            "initial_alpha": 0.1,           # Default
            "add_error_estimator": False,
            "shared_attention": ASSUMED_TRAIN_SHARED_ATTN
        },
        "prediction_fn_name": "predict_multivar_rom_eq9" # Specific prediction function
    }
}
MODEL_TO_BENCHMARK_KEY = "MultiVarROMEq9_Default"


# --- Utility Functions (Adapted from your reference) ---
def load_data(dataset_name_key_load):
    config = DATASET_CONFIGS[dataset_name_key_load]
    dataset_path = config['path']
    if not os.path.exists(dataset_path): print(f"Dataset file not found: {dataset_path}"); return None, None, None
    with open(dataset_path, 'rb') as f: data_list_all = pickle.load(f)
    random.shuffle(data_list_all)
    n_total = len(data_list_all); n_val_start_idx = int(0.8 * n_total); val_data_list = data_list_all[n_val_start_idx:]
    if not val_data_list: print(f"No validation data for {dataset_name_key_load}."); return None, None, None
    print(f"Using {len(val_data_list)} samples for validation from {dataset_name_key_load}.")
    # For benchmark, load full sequences (train_nt_limit=None)
    val_dataset = UniversalPDEDataset(val_data_list, dataset_type=config['dataset_type_arg'], train_nt_limit=None)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    return val_data_list, val_loader, config

def calculate_metrics(pred_denorm_calc, gt_denorm_calc):
    if pred_denorm_calc.shape != gt_denorm_calc.shape:
        min_len_t = min(pred_denorm_calc.shape[0], gt_denorm_calc.shape[0]); min_len_x = min(pred_denorm_calc.shape[1], gt_denorm_calc.shape[1])
        pred_denorm_calc = pred_denorm_calc[:min_len_t, :min_len_x]; gt_denorm_calc = gt_denorm_calc[:min_len_t, :min_len_x]
        if pred_denorm_calc.shape != gt_denorm_calc.shape or pred_denorm_calc.size == 0:
            print(f"Warning: Shape mismatch or empty array after truncation. Pred: {pred_denorm_calc.shape}, GT: {gt_denorm_calc.shape}")
            return {'mse': float('nan'), 'rmse': float('nan'), 'rel_err': float('nan'), 'max_err': float('nan')}
    if pred_denorm_calc.size == 0: return {'mse': float('nan'), 'rmse': float('nan'), 'rel_err': float('nan'), 'max_err': float('nan')}
    mse = np.mean((pred_denorm_calc - gt_denorm_calc)**2); rmse = np.sqrt(mse)
    norm_gt = np.linalg.norm(gt_denorm_calc, 'fro')
    rel_err = np.linalg.norm(pred_denorm_calc - gt_denorm_calc, 'fro') / (norm_gt + 1e-10) if norm_gt > 1e-9 else np.linalg.norm(pred_denorm_calc - gt_denorm_calc, 'fro')
    max_err = np.max(np.abs(pred_denorm_calc - gt_denorm_calc))
    return {'mse': mse, 'rmse': rmse, 'rel_err': rel_err, 'max_err': max_err}

def plot_comparison(gt_seq_denorm_dict_plot, predictions_denorm_dict_plot, dataset_name_plot, sample_idx_str_plot, state_keys_to_plot_plot, L_domain_plot, T_horizon_plot, save_path_base_plot):
    num_models_to_plot = len(predictions_denorm_dict_plot) + 1; num_vars = len(state_keys_to_plot_plot)
    if num_vars == 0: return
    fig, axs = plt.subplots(num_vars, num_models_to_plot, figsize=(6 * num_models_to_plot, 5 * num_vars), squeeze=False)
    plot_model_names = ["Ground Truth"] + list(predictions_denorm_dict_plot.keys())
    for i_skey, skey_plot in enumerate(state_keys_to_plot_plot):
        gt_data_var_plot = gt_seq_denorm_dict_plot.get(skey_plot)
        if gt_data_var_plot is None: axs[i_skey,0].set_ylabel(f"{skey_plot}\n(GT Missing)"); continue
        all_series_for_var = [gt_data_var_plot] + [predictions_denorm_dict_plot.get(model_name, {}).get(skey_plot) for model_name in predictions_denorm_dict_plot.keys()]
        valid_series = [s for s in all_series_for_var if s is not None and s.size > 0 and s.ndim == 2]
        if not valid_series: axs[i_skey,0].set_ylabel(f"{skey_plot}\n(No valid data)"); continue
        vmin = min(s.min() for s in valid_series); vmax = max(s.max() for s in valid_series)
        if abs(vmax - vmin) < 1e-9: vmax = vmin + 1.0 if vmax == vmin else vmax # Adjust if all values are identical
        for j_model, model_name_plot_iter in enumerate(plot_model_names):
            ax = axs[i_skey, j_model]; data_to_plot = gt_data_var_plot if model_name_plot_iter == "Ground Truth" else predictions_denorm_dict_plot.get(model_name_plot_iter, {}).get(skey_plot, None)
            if data_to_plot is None or data_to_plot.size == 0 or data_to_plot.ndim != 2: ax.text(0.5, 0.5, "No/Invalid data", ha="center", va="center", fontsize=9); ax.set_title(f"{model_name_plot_iter}\n({skey_plot})", fontsize=10)
            else:
                im = ax.imshow(data_to_plot, aspect='auto', origin='lower', extent=[0, L_domain_plot, 0, T_horizon_plot], cmap='viridis', vmin=vmin, vmax=vmax)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(f"{model_name_plot_iter}\n({skey_plot})", fontsize=10)
            if i_skey == num_vars -1 : ax.set_xlabel("x", fontsize=10)
            if j_model == 0: ax.set_ylabel(f"{skey_plot}\nt (physical)", fontsize=10)
            else: ax.set_yticklabels([])
            ax.tick_params(axis='both', which='major', labelsize=8)
    fig.suptitle(f"Benchmark: {dataset_name_plot} (Sample {sample_idx_str_plot}) @ T={T_horizon_plot:.2f}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); final_save_path = os.path.join(save_path_base_plot, f"comparison_{dataset_name_plot}_sample{sample_idx_str_plot}_T{T_horizon_plot:.2f}.png")
    os.makedirs(save_path_base_plot, exist_ok=True); plt.savefig(final_save_path); print(f"Saved comparison plot to {final_save_path}"); plt.close(fig)


# --- Model Loading Function for MultiVarAttentionROMEq9 ---
def load_multivar_rom_eq9_model(model_name_key_load, model_arch_config_load, dataset_config_load, device_load):
    ckpt_filename_prefix = model_arch_config_load["checkpoint_filename_prefix"]
    # Adjust checkpoint directory to where your Eq9 models are saved
    ckpt_dir_load = os.path.join(BASE_CHECKPOINT_PATH, f"_checkpoints_{dataset_config_load['dataset_type_arg']}")

    # Find the latest checkpoint matching the prefix
    search_pattern = os.path.join(ckpt_dir_load, f"{ckpt_filename_prefix}*.pt")
    list_of_files = glob.glob(search_pattern)
    if not list_of_files:
        print(f"No checkpoint found matching prefix '{ckpt_filename_prefix}' in {ckpt_dir_load}"); return None
    latest_file = max(list_of_files, key=os.path.getctime)
    checkpoint_path_load = latest_file
    print(f"Loading latest checkpoint for {model_name_key_load}: {checkpoint_path_load}")

    try:
        ckpt = torch.load(checkpoint_path_load, map_location=device_load, weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint file {checkpoint_path_load}: {e}"); return None

    state_keys_load=dataset_config_load['state_keys']; nx_val_load=dataset_config_load['nx']
    try:
        with open(dataset_config_load['path'], 'rb') as f_dummy_load: dummy_data_sample_load = pickle.load(f_dummy_load)[0]
        dummy_dataset_load = UniversalPDEDataset([dummy_data_sample_load], dataset_type=dataset_config_load['dataset_type_arg'])
        bc_state_dim_actual_load = dummy_dataset_load.bc_state_dim; num_controls_actual_load = dummy_dataset_load.num_controls
    except Exception as e:
        print(f"Error creating dummy dataset for {model_name_key_load}: {e}. Using defaults from ckpt/config.")
        bc_state_dim_actual_load = ckpt.get('bc_state_dim', model_arch_config_load["params"].get('bc_state_dim', 2))
        num_controls_actual_load = ckpt.get('num_controls', model_arch_config_load["params"].get('num_controls', 0)) # Default to 0 if not found

    cfg_params_load = model_arch_config_load["params"]
    model_load = None
    try:
        model_load = MultiVarAttentionROMEq9( # Use the correct class name here
            state_variable_keys=state_keys_load, nx=nx_val_load,
            basis_dim=ckpt.get('basis_dim', cfg_params_load['basis_dim']),
            d_model=ckpt.get('d_model', cfg_params_load['d_model']),
            bc_state_dim=bc_state_dim_actual_load,
            num_controls=num_controls_actual_load,
            num_heads=ckpt.get('num_heads', cfg_params_load['num_heads']),
            use_fixed_lifting=ckpt.get('use_fixed_lifting', cfg_params_load.get('use_fixed_lifting', False)),
            bc_processed_dim=ckpt.get('bc_processed_dim', cfg_params_load.get('bc_processed_dim', 64)),
            hidden_bc_processor_dim=ckpt.get('hidden_bc_processor_dim', cfg_params_load.get('hidden_bc_processor_dim', 128)),
            initial_alpha=ckpt.get('initial_alpha', cfg_params_load.get('initial_alpha',0.1)),
            add_error_estimator=ckpt.get('add_error_estimator', cfg_params_load.get('add_error_estimator',False)),
            shared_attention=ckpt.get('shared_attention', cfg_params_load.get('shared_attention',False))
        )
    except Exception as e:
        print(f"Error during model instantiation for {model_name_key_load}: {e}"); traceback.print_exc(); return None

    try: model_load.load_state_dict(ckpt['model_state_dict'], strict=False)
    except Exception as e: print(f"Error loading state_dict for {model_name_key_load}: {e}"); traceback.print_exc(); return None
    model_load.to(device_load); model_load.eval(); print(f"Successfully loaded {model_name_key_load}."); return model_load


# --- Prediction Function for MultiVarAttentionROMEq9 ---
@torch.no_grad()
def predict_multivar_rom_eq9(model_pred, initial_state_dict_norm_pred, bc_ctrl_seq_norm_pred, num_model_steps_pred, dataset_config_pred):
    a0_dict_pred = {}; batch_size_pred = list(initial_state_dict_norm_pred.values())[0].shape[0]
    device_model_pred = next(model_pred.parameters()).device
    initial_state_dict_norm_pred = {k: v.to(device_model_pred) for k, v in initial_state_dict_norm_pred.items()}
    bc_ctrl_seq_norm_pred = bc_ctrl_seq_norm_pred.to(device_model_pred) # Shape [B, SeqLen_BC, bc_dim]

    # Ensure bc_ctrl_seq_norm_pred is long enough for T_model_steps + 1 points (indices 0 to T_model_steps)
    required_bc_len = num_model_steps_pred + 1
    if bc_ctrl_seq_norm_pred.shape[1] < required_bc_len:
        print(f"  predict_multivar_rom_eq9: Warning: BC_Ctrl sequence length {bc_ctrl_seq_norm_pred.shape[1]} is less than required {required_bc_len} for {num_model_steps_pred} steps. Padding with last value.")
        padding_needed = required_bc_len - bc_ctrl_seq_norm_pred.shape[1]
        last_bc_slice = bc_ctrl_seq_norm_pred[:, -1:, :] # Shape [B, 1, bc_dim]
        padding = last_bc_slice.repeat(1, padding_needed, 1)
        bc_ctrl_seq_norm_pred = torch.cat([bc_ctrl_seq_norm_pred, padding], dim=1)

    BC_ctrl_t0_norm_pred = bc_ctrl_seq_norm_pred[:, 0, :] # BC_Ctrl^0
    U_B0_lifted_norm_pred = model_pred._compute_U_B(BC_ctrl_t0_norm_pred) # U_B^0

    for i_key_pred, key_pred in enumerate(model_pred.state_keys):
        U0_norm_var_pred = initial_state_dict_norm_pred[key_pred] # U^0_norm
        if U0_norm_var_pred.dim() == 2: U0_norm_var_pred = U0_norm_var_pred.unsqueeze(-1) # [B, Nx, 1]
        U_B0_norm_var_pred = U_B0_lifted_norm_pred[:, i_key_pred, :].unsqueeze(-1)
        U0_star_norm_var_pred = U0_norm_var_pred - U_B0_norm_var_pred # (U^0 - U_B^0)_norm

        Phi_var_pred = model_pred.get_basis(key_pred).to(device_model_pred)
        Phi_T_var_pred = Phi_var_pred.transpose(0,1).unsqueeze(0).expand(batch_size_pred, -1, -1)
        a0_norm_var_pred = torch.bmm(Phi_T_var_pred, U0_star_norm_var_pred).squeeze(-1) # a^0_norm
        a0_dict_pred[key_pred] = a0_norm_var_pred

    # model.forward T is number of steps a^n -> a^{n+1}
    # It requires BC_Ctrl_seq to have length T+1 (indices 0 to T)
    pred_seq_norm_dict_of_tensors, _ = model_pred(a0_dict_pred, bc_ctrl_seq_norm_pred[:, :num_model_steps_pred + 1, :], T=num_model_steps_pred)
    # Output U_hat_seq_dict[key] is [batch, num_model_steps_pred, nx]
    # These are U_hat^1_norm, ..., U_hat^{num_model_steps_pred}_norm

    # Squeeze batch dimension if batch_size_pred was 1
    if batch_size_pred == 1:
        for key_sqz in pred_seq_norm_dict_of_tensors:
            pred_seq_norm_dict_of_tensors[key_sqz] = pred_seq_norm_dict_of_tensors[key_sqz].squeeze(0)

    return pred_seq_norm_dict_of_tensors


# --- Main Benchmarking Logic ---
def main(target_model_key_main):
    print(f"Device: {DEVICE}")
    dataset_name_key_main = TARGET_DATASET_KEY_BENCHMARK # Use the globally defined target dataset
    print(f"Benchmarking model: {target_model_key_main} on dataset {dataset_name_key_main}")

    # Physical time horizons for benchmarking
    BENCHMARK_T_HORIZONS_PHYSICAL = [0.5, 1.0, 1.5, 2.0] # Example physical times

    ds_config_main = DATASET_CONFIGS[dataset_name_key_main]
    # Filter horizons to be within the dataset's max time
    valid_benchmark_T_horizons = [T_h for T_h in BENCHMARK_T_HORIZONS_PHYSICAL if T_h <= ds_config_main['T_file']]
    if not valid_benchmark_T_horizons:
        print(f"No valid benchmark horizons for dataset {dataset_name_key_main} (max T_file: {ds_config_main['T_file']}). Exiting.")
        return

    print(f"Effective benchmark physical T_horizons: {valid_benchmark_T_horizons}")

    overall_aggregated_metrics = {dataset_name_key_main: {target_model_key_main: {
        T_h: {skey: {'mse': [], 'rel_err': [], 'rmse': [], 'max_err': []}
              for skey in ds_config_main['state_keys']}
        for T_h in valid_benchmark_T_horizons
    }}}

    val_data_list_main, val_loader_main, _ = load_data(dataset_name_key_main) # ds_config_main already has config
    if val_loader_main is None: print(f"Failed to load data for {dataset_name_key_main}. Exiting."); return

    num_val_samples_main = len(val_data_list_main); vis_sample_count_main = min(MAX_VISUALIZATION_SAMPLES, num_val_samples_main)
    visualization_indices_main = random.sample(range(num_val_samples_main), vis_sample_count_main) if num_val_samples_main > 0 else []
    print(f"Will visualize {len(visualization_indices_main)} random samples: {visualization_indices_main}")

    gt_data_for_visualization_main = {vis_idx: None for vis_idx in visualization_indices_main}
    predictions_for_visualization_main = {vis_idx: {target_model_key_main: {}} for vis_idx in visualization_indices_main} # Model key is fixed

    print(f"\n  -- Pre-loading Model: {target_model_key_main} for dataset {dataset_name_key_main} --")
    if target_model_key_main not in MODEL_TO_BENCHMARK_CONFIG:
        print(f"  Skipping model {target_model_key_main}: Not configured in MODEL_TO_BENCHMARK_CONFIG."); return

    model_arch_config_main = MODEL_TO_BENCHMARK_CONFIG[target_model_key_main]
    model_instance_main = load_multivar_rom_eq9_model(target_model_key_main, model_arch_config_main, ds_config_main, DEVICE)

    if model_instance_main is None:
        print(f"  Failed to pre-load model {target_model_key_main}. Exiting."); return

    for val_idx_main, (sample_state_list_norm_main, sample_bc_ctrl_seq_norm_main, sample_norm_factors_main) in enumerate(val_loader_main):
        print(f"  Processing validation sample {val_idx_main+1}/{num_val_samples_main} for dataset {dataset_name_key_main}...")
        initial_state_norm_dict_main = {}; gt_full_seq_denorm_dict_sample_main = {}
        # sample_state_list_norm_main is a list of tensors [1, NT_file, nx]
        # sample_bc_ctrl_seq_norm_main is a tensor [1, NT_file, bc_dim]
        for idx_skey_main, skey_main in enumerate(ds_config_main['state_keys']):
            initial_state_norm_dict_main[skey_main] = sample_state_list_norm_main[idx_skey_main][:, 0, :].to(DEVICE) # U^0_norm
            gt_seq_norm_var_main = sample_state_list_norm_main[idx_skey_main].squeeze(0).to(DEVICE) # [NT_file, nx] U^0_norm to U^{NT_file-1}_norm
            mean_val_main = sample_norm_factors_main[f'{skey_main}_mean'].item(); std_val_main = sample_norm_factors_main[f'{skey_main}_std'].item()
            gt_full_seq_denorm_dict_sample_main[skey_main] = gt_seq_norm_var_main.cpu().numpy() * std_val_main + mean_val_main # Denormalized U^0 to U^{NT_file-1}
        if val_idx_main in visualization_indices_main: gt_data_for_visualization_main[val_idx_main] = gt_full_seq_denorm_dict_sample_main

        for T_current_horizon_physical_main in valid_benchmark_T_horizons:
            # Calculate number of model steps to reach this physical horizon
            # num_model_steps = number of a^n -> a^{n+1} transitions
            # Predictions will be U_hat^1 ... U_hat^{num_model_steps}
            if ds_config_main['T_file'] > 1e-9 and ds_config_main['NT_file'] > 1:
                 dt_data_approx = ds_config_main['T_file'] / (ds_config_main['NT_file'] - 1)
                 num_model_steps_horizon = int(round(T_current_horizon_physical_main / dt_data_approx))
            elif ds_config_main['NT_file'] == 1 and T_current_horizon_physical_main <= ds_config_main['T_file']: # Only one data point
                 num_model_steps_horizon = 0 # No steps to take if only U0 is available
            else: # Should not happen with valid config
                 num_model_steps_horizon = ds_config_main['NT_file'] -1


            num_model_steps_horizon = min(num_model_steps_horizon, ds_config_main['NT_file'] - 1) # Cap at max possible steps
            num_model_steps_horizon = max(0, num_model_steps_horizon) # Ensure non-negative

            # Ground truth for comparison: U_true^1 ... U_true^{num_model_steps_horizon}
            # These correspond to time points dt, 2*dt, ..., num_model_steps_horizon*dt
            gt_horizon_denorm_dict_main = {}
            for skey_gt_slice in ds_config_main['state_keys']:
                # gt_full_seq_denorm_dict_sample_main[skey_gt_slice] is U^0 ... U^{NT_file-1}
                # We need indices 1 to num_model_steps_horizon
                if num_model_steps_horizon > 0:
                     gt_horizon_denorm_dict_main[skey_gt_slice] = gt_full_seq_denorm_dict_sample_main[skey_gt_slice][1 : num_model_steps_horizon + 1, :]
                else: # No steps, no GT to compare against for predictions
                     gt_horizon_denorm_dict_main[skey_gt_slice] = np.empty((0, ds_config_main['nx']))


            # BC_Ctrl sequence for prediction needs to cover up to time index num_model_steps_horizon
            # So, length num_model_steps_horizon + 1
            bc_ctrl_for_pred_len = num_model_steps_horizon + 1
            # sample_bc_ctrl_seq_norm_main has shape [1, NT_file, bc_dim]
            bc_ctrl_for_pred = sample_bc_ctrl_seq_norm_main[:, :bc_ctrl_for_pred_len, :].to(DEVICE)

            pred_seq_norm_dict_model_main = {}
            if num_model_steps_horizon > 0: # Only predict if there are steps to take
                try:
                    prediction_fn = globals()[model_arch_config_main["prediction_fn_name"]]
                    pred_seq_norm_dict_model_main = prediction_fn(model_instance_main, initial_state_norm_dict_main, bc_ctrl_for_pred, num_model_steps_horizon, ds_config_main)
                except Exception as e:
                    print(f"    ERROR during prediction for {target_model_key_main} on sample {val_idx_main}, T_phys={T_current_horizon_physical_main} (steps={num_model_steps_horizon}): {e}"); traceback.print_exc(); continue
            else: # No steps to predict
                 for skey_empty in ds_config_main['state_keys']: pred_seq_norm_dict_model_main[skey_empty] = torch.empty(0, ds_config_main['nx'], device=DEVICE)


            model_preds_denorm_sample_horizon_main = {}
            for skey_metrics in ds_config_main['state_keys']:
                if skey_metrics not in pred_seq_norm_dict_model_main or \
                   pred_seq_norm_dict_model_main[skey_metrics] is None or \
                   pred_seq_norm_dict_model_main[skey_metrics].numel() == 0:
                    # This case handles num_model_steps_horizon == 0 correctly
                    model_preds_denorm_sample_horizon_main[skey_metrics] = np.empty((0, ds_config_main['nx']))
                    # For metrics, if no prediction, record NaNs or skip
                    if num_model_steps_horizon == 0:
                        metrics = {'mse': float('nan'), 'rmse': float('nan'), 'rel_err': float('nan'), 'max_err': float('nan')}
                    # else: # error in prediction logic if steps > 0 but no pred data
                    # continue # or handle as error
                else:
                    pred_norm_var_main = pred_seq_norm_dict_model_main[skey_metrics].cpu().numpy() # Should be [num_model_steps, nx]
                    gt_denorm_var_main = gt_horizon_denorm_dict_main[skey_metrics] # Also [num_model_steps, nx]
                    mean_val_main_metric = sample_norm_factors_main[f'{skey_metrics}_mean'].item(); std_val_main_metric = sample_norm_factors_main[f'{skey_metrics}_std'].item()
                    pred_denorm_var_main = pred_norm_var_main * std_val_main_metric + mean_val_main_metric
                    model_preds_denorm_sample_horizon_main[skey_metrics] = pred_denorm_var_main
                    metrics = calculate_metrics(pred_denorm_var_main, gt_denorm_var_main)

                # Store metrics
                if T_current_horizon_physical_main in overall_aggregated_metrics[dataset_name_key_main][target_model_key_main]:
                    for metric_name, metric_val in metrics.items():
                        overall_aggregated_metrics[dataset_name_key_main][target_model_key_main][T_current_horizon_physical_main][skey_metrics][metric_name].append(metric_val)

            if val_idx_main in visualization_indices_main:
                # Store predictions for this model and this horizon
                predictions_for_visualization_main[val_idx_main][target_model_key_main][T_current_horizon_physical_main] = model_preds_denorm_sample_horizon_main
        torch.cuda.empty_cache()


    print(f"\n  Generating visualizations for {dataset_name_key_main}...")
    for vis_idx_main in visualization_indices_main:
        if vis_idx_main not in gt_data_for_visualization_main or gt_data_for_visualization_main[vis_idx_main] is None: continue
        for T_plot_horizon_physical_main in valid_benchmark_T_horizons:
            # Calculate num_plot_steps again for slicing GT and ensuring consistency with plot T
            if ds_config_main['T_file'] > 1e-9 and ds_config_main['NT_file'] > 1:
                 dt_data_approx_plot = ds_config_main['T_file'] / (ds_config_main['NT_file'] - 1)
                 num_plot_steps_main = int(round(T_plot_horizon_physical_main / dt_data_approx_plot))
            elif ds_config_main['NT_file'] == 1 and T_plot_horizon_physical_main <= ds_config_main['T_file']:
                 num_plot_steps_main = 0
            else: num_plot_steps_main = ds_config_main['NT_file'] -1
            num_plot_steps_main = min(num_plot_steps_main, ds_config_main['NT_file'] - 1); num_plot_steps_main = max(0, num_plot_steps_main)

            # GT for plotting U_true^1 ... U_true^{num_plot_steps}
            current_gt_denorm_sliced_plot_main = {}
            for skey_plot_gt in ds_config_main['state_keys']:
                if num_plot_steps_main > 0:
                    current_gt_denorm_sliced_plot_main[skey_plot_gt] = gt_data_for_visualization_main[vis_idx_main][skey_plot_gt][1 : num_plot_steps_main + 1, :]
                else:
                    current_gt_denorm_sliced_plot_main[skey_plot_gt] = np.empty((0, ds_config_main['nx']))


            # Predictions for plotting (already U_hat^1 ... U_hat^{num_plot_steps})
            predictions_for_single_plot_main = {}
            if target_model_key_main in predictions_for_visualization_main[vis_idx_main] and \
               T_plot_horizon_physical_main in predictions_for_visualization_main[vis_idx_main][target_model_key_main]:
                # Ensure predictions are also sliced to num_plot_steps if they were longer due to rounding differences
                raw_preds_for_plot = predictions_for_visualization_main[vis_idx_main][target_model_key_main][T_plot_horizon_physical_main]
                sliced_preds_for_plot = {skey: data[:num_plot_steps_main,:] for skey, data in raw_preds_for_plot.items() if data.ndim==2 and data.shape[0]>=num_plot_steps_main }
                predictions_for_single_plot_main[target_model_key_main] = sliced_preds_for_plot


            if not predictions_for_single_plot_main.get(target_model_key_main):
                print(f"No predictions to plot for sample {vis_idx_main}, T={T_plot_horizon_physical_main}. Skipping plot.")
                continue

            plot_comparison(current_gt_denorm_sliced_plot_main, predictions_for_single_plot_main,
                            dataset_name_key_main, f"{vis_idx_main}", ds_config_main['state_keys'],
                            ds_config_main['L'], T_plot_horizon_physical_main, # Use physical T for plot extent
                            os.path.join(BASE_RESULTS_PATH, dataset_name_key_main, "plots", target_model_key_main))

    print("\n\n===== BENCHMARKING COMPLETE - AGGREGATED METRICS =====")
    for ds_name_print_main, model_data_agg_main in overall_aggregated_metrics.items():
        print(f"\n--- Aggregated Results for Dataset: {ds_name_print_main.upper()} ---")
        for model_name_print_main, horizon_metrics_data_main in model_data_agg_main.items():
            print(f"  Model: {model_name_print_main}")
            if not horizon_metrics_data_main: print("    No metrics recorded for this model."); continue
            for T_h_print_main, var_metrics_lists_main in sorted(horizon_metrics_data_main.items()):
                print(f"    Horizon Physical T={T_h_print_main:.2f}:")
                for skey_print_main, metrics_lists_main in var_metrics_lists_main.items():
                    if not metrics_lists_main['mse'] or all(np.isnan(m) for m in metrics_lists_main['mse']):
                        print(f"      {skey_print_main}: No valid metrics recorded for this horizon."); continue
                    avg_mse=np.nanmean(metrics_lists_main['mse']); avg_rmse=np.nanmean(metrics_lists_main['rmse']);
                    avg_rel_err=np.nanmean(metrics_lists_main['rel_err']); avg_max_err=np.nanmean(metrics_lists_main['max_err'])
                    num_valid_samples = len([m for m in metrics_lists_main['mse'] if not np.isnan(m)])
                    print(f"      {skey_print_main}: Avg MSE={avg_mse:.3e}, Avg RMSE={avg_rmse:.3e}, Avg RelErr={avg_rel_err:.3e}, Avg MaxErr={avg_max_err:.3e} (from {num_valid_samples} valid samples)")


if __name__ == '__main__':
    # This script is designed to benchmark ONE specific model configuration defined in MODEL_TO_BENCHMARK_CONFIG
    # The key for this configuration is MODEL_TO_BENCHMARK_KEY
    print(f"Starting benchmark for model key: {MODEL_TO_BENCHMARK_KEY}")
    main(MODEL_TO_BENCHMARK_KEY)
