# BENCHMARK_NO_ATTENTION_VARIANTS.PY
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
import glob 
import traceback
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
# --- Model Class Definitions ---
# Copied and adapted from your provided training scripts.
# Ensure these are the exact versions used for training the target checkpoints.

# =============================================================================
# 2. 通用化数据集定义 (UniversalPDEDataset)
# =============================================================================
class UniversalPDEDataset(Dataset):
    def __init__(self, data_list, dataset_type, train_nt_limit=None):
        if not data_list: raise ValueError("data_list cannot be empty")
        self.data_list = data_list; self.dataset_type = dataset_type; self.train_nt_limit = train_nt_limit
        first_sample = data_list[0]
        
        # Specific to 'reaction_diffusion_neumann_feedback' for this benchmark
        self.nt_from_sample = first_sample['U'].shape[0]
        self.nx_from_sample = first_sample['U'].shape[1]
        self.ny_from_sample = 1
        self.state_keys = ['U']; self.num_state_vars = 1
        self.nx = self.nx_from_sample; self.ny = self.ny_from_sample
        self.expected_bc_state_dim = 2 
        
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
        if self.num_controls == 0:
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
            if bc_control_seq_full.ndim == 1 and self.num_controls == 1:
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

# =============================================================================
# 4. 模型定义 (Model Definitions)
# =============================================================================
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

# NoAttention_ExplicitBC_ROM (from user's training script)
class NoAttention_ExplicitBC_ROM(nn.Module):
    def __init__(self, state_variable_keys, nx, basis_dim, d_model, 
                 bc_state_dim, num_controls, 
                 add_error_estimator=False, shared_components=False, 
                 dropout_lifting=0.1, dropout_ffn=0.1, initial_alpha=0.1,
                 use_fixed_lifting=False,
                 bc_processed_dim=64, 
                 hidden_bc_processor_dim=128
                 ):
        super().__init__()
        self.state_keys = state_variable_keys; self.num_state_vars = len(state_variable_keys)
        self.nx = nx; self.basis_dim = basis_dim; self.d_model = d_model 
        self.bc_state_dim = bc_state_dim; self.num_controls = num_controls
        self.add_error_estimator = add_error_estimator; self.shared_components = shared_components
        self.use_fixed_lifting = use_fixed_lifting; self.bc_processed_dim = bc_processed_dim
        self.hidden_bc_processor_dim = hidden_bc_processor_dim

        self.Phi = nn.ParameterDict()
        for key in self.state_keys:
            phi_param = nn.Parameter(torch.randn(nx, basis_dim)); nn.init.orthogonal_(phi_param); self.Phi[key] = phi_param

        if not self.use_fixed_lifting:
            self.lifting = UniversalLifting(self.num_state_vars, bc_state_dim, num_controls, nx, nx, dropout=dropout_lifting)
        else:
            self.lifting = None; self.register_buffer('lin_interp_coeffs', torch.linspace(0, 1, self.nx, dtype=torch.float32).view(1, 1, -1))

        self.update_ffn = nn.ModuleDict(); self.a0_mapping = nn.ModuleDict(); self.alphas = nn.ParameterDict()
        self.bc_feature_processor = nn.ModuleDict(); self.bc_to_a_update = nn.ModuleDict()
        total_bc_ctrl_dim = self.bc_state_dim + self.num_controls

        loop_keys = ['shared'] if self.shared_components else self.state_keys
        for k_loop in loop_keys:
            self.update_ffn[k_loop] = ImprovedUpdateFFN(input_dim=basis_dim, output_dim=basis_dim, hidden_dim=d_model, dropout=dropout_ffn)
            self.a0_mapping[k_loop] = nn.Sequential(nn.Linear(basis_dim, basis_dim), nn.ReLU(), nn.LayerNorm(basis_dim))
            self.alphas[k_loop] = nn.Parameter(torch.tensor(initial_alpha))
            if total_bc_ctrl_dim > 0 and self.bc_processed_dim > 0:
                self.bc_feature_processor[k_loop] = nn.Sequential(
                    nn.Linear(total_bc_ctrl_dim, self.hidden_bc_processor_dim), nn.GELU(),
                    nn.Linear(self.hidden_bc_processor_dim, self.bc_processed_dim))
                self.bc_to_a_update[k_loop] = nn.Linear(self.bc_processed_dim, basis_dim)
            else: 
                self.bc_feature_processor[k_loop] = nn.Sequential()
                self.bc_to_a_update[k_loop] = nn.Sequential() 
        
        if self.add_error_estimator: self.error_estimator = nn.Linear(self.num_state_vars * basis_dim, 1)

    def _get_layer(self, module_dict, key): return module_dict['shared'] if self.shared_components else module_dict[key]
    def _get_alpha(self, key): return self.alphas['shared'] if self.shared_components else self.alphas[key]
    def _compute_U_B(self, BC_Ctrl_n): 
        if not self.use_fixed_lifting: 
            if self.lifting is None: raise ValueError("Lifting network is None but use_fixed_lifting is False.")
            return self.lifting(BC_Ctrl_n)
        else:
            if self.num_state_vars == 1:
                if self.bc_state_dim < 2: return torch.zeros(BC_Ctrl_n.shape[0], 1, self.nx, device=BC_Ctrl_n.device)
                l=BC_Ctrl_n[:,0:1].unsqueeze(-1);r=BC_Ctrl_n[:,1:2].unsqueeze(-1);return l*(1-self.lin_interp_coeffs)+r*self.lin_interp_coeffs
            else:
                if self.bc_state_dim < 2*self.num_state_vars: return torch.zeros(BC_Ctrl_n.shape[0],self.num_state_vars,self.nx,device=BC_Ctrl_n.device)
                ul=[];
                for i in range(self.num_state_vars):
                    il=i*2;ir=i*2+1
                    if ir>=self.bc_state_dim: usv=torch.zeros(BC_Ctrl_n.shape[0],1,self.nx,device=BC_Ctrl_n.device)
                    else: l=BC_Ctrl_n[:,il:il+1].unsqueeze(-1);r=BC_Ctrl_n[:,ir:ir+1].unsqueeze(-1);usv=l*(1-self.lin_interp_coeffs)+r*self.lin_interp_coeffs
                    ul.append(usv)
                return torch.cat(ul,dim=1)

    def forward_step(self, a_n_dict, BC_Ctrl_n, params=None):
        batch_size = list(a_n_dict.values())[0].size(0); a_next_dict = {}; U_hat_dict = {}
        U_B_stacked = self._compute_U_B(BC_Ctrl_n)
        bc_features_processed_dict = {}
        total_bc_ctrl_dim = self.bc_state_dim + self.num_controls
        if total_bc_ctrl_dim > 0 and self.bc_processed_dim > 0:
            loop_key_bc = 'shared' if self.shared_components else None
            for key in self.state_keys:
                actual_key_bc = loop_key_bc if loop_key_bc else key
                bc_proc_layer = self.bc_feature_processor[actual_key_bc]
                if len(list(bc_proc_layer.parameters())) > 0 : 
                    bc_features_processed_dict[key] = bc_proc_layer(BC_Ctrl_n)
                else:
                    bc_features_processed_dict[key] = None
        else:
            for key in self.state_keys: bc_features_processed_dict[key] = None
            
        for i_var, key in enumerate(self.state_keys):
            a_n_var = a_n_dict[key]; Phi_var = self.Phi[key]
            ffn_var = self._get_layer(self.update_ffn, key); alpha_var = self._get_alpha(key)
            bc_to_a_update_layer = self._get_layer(self.bc_to_a_update, key)
            ffn_update_intrinsic = ffn_var(a_n_var)
            bc_driven_a_update = torch.zeros_like(a_n_var)
            current_bc_features = bc_features_processed_dict[key]
            if current_bc_features is not None and (not isinstance(bc_to_a_update_layer, nn.Sequential) or len(list(bc_to_a_update_layer.parameters())) > 0):
                bc_driven_a_update = bc_to_a_update_layer(current_bc_features)
            a_next_var = a_n_var + alpha_var * ffn_update_intrinsic + bc_driven_a_update
            a_next_dict[key] = a_next_var
            U_B_var = U_B_stacked[:, i_var, :].unsqueeze(-1)
            Phi_expanded = Phi_var.unsqueeze(0).expand(batch_size, -1, -1)
            a_next_unsq = a_next_var.unsqueeze(-1)
            U_recon_star = torch.bmm(Phi_expanded, a_next_unsq)
            U_hat_dict[key] = U_B_var + U_recon_star
        err_est = None
        if self.add_error_estimator: err_est = self.error_estimator(torch.cat(list(a_next_dict.values()), dim=-1))
        return a_next_dict, U_hat_dict, err_est

    def forward(self, a0_dict, BC_Ctrl_seq, T, params=None):
        a_current_dict = {}
        for key in self.state_keys: a_current_dict[key] = self._get_layer(self.a0_mapping, key)(a0_dict[key])
        U_hat_seq_dict = {key: [] for key in self.state_keys}; err_seq = [] if self.add_error_estimator else None
        for t_step in range(T):
            BC_Ctrl_n_step = BC_Ctrl_seq[:, t_step, :]
            a_next_dict_step, U_hat_dict_step, err_est_step = self.forward_step(a_current_dict, BC_Ctrl_n_step, params)
            for key in self.state_keys: U_hat_seq_dict[key].append(U_hat_dict_step[key])
            if self.add_error_estimator and err_est_step is not None: err_seq.append(err_est_step)
            a_current_dict = a_next_dict_step
        return U_hat_seq_dict, err_seq
    def get_basis(self, key): return self.Phi[key]

# No-Attention ROM with IMPLICIT BC handling (from user's training script)
class NoAttention_ImplicitBC_ROM(nn.Module):
    def __init__(self, state_variable_keys, nx, basis_dim, d_model, 
                 bc_state_dim, num_controls,
                 add_error_estimator=False, shared_components=False, 
                 dropout_lifting=0.1, dropout_ffn=0.1, initial_alpha=0.1,
                 use_fixed_lifting=False):
        super().__init__()
        self.state_keys = state_variable_keys; self.num_state_vars = len(state_variable_keys)
        self.nx = nx; self.basis_dim = basis_dim; self.d_model = d_model
        self.bc_state_dim = bc_state_dim; self.num_controls = num_controls
        self.add_error_estimator = add_error_estimator; self.shared_components = shared_components
        self.use_fixed_lifting = use_fixed_lifting

        self.Phi = nn.ParameterDict()
        for key in self.state_keys:
            phi_param = nn.Parameter(torch.randn(nx, basis_dim)); nn.init.orthogonal_(phi_param); self.Phi[key] = phi_param

        if not self.use_fixed_lifting:
            self.lifting = UniversalLifting(self.num_state_vars, bc_state_dim, num_controls, nx, nx, dropout=dropout_lifting)
        else:
            self.lifting = None; self.register_buffer('lin_interp_coeffs', torch.linspace(0, 1, self.nx, dtype=torch.float32).view(1, 1, -1))

        self.update_ffn = nn.ModuleDict(); self.a0_mapping = nn.ModuleDict(); self.alphas = nn.ParameterDict()
        
        loop_keys = ['shared'] if self.shared_components else self.state_keys
        for k_loop in loop_keys:
            self.update_ffn[k_loop] = ImprovedUpdateFFN(input_dim=basis_dim, output_dim=basis_dim, hidden_dim=d_model, dropout=dropout_ffn)
            self.a0_mapping[k_loop] = nn.Sequential(nn.Linear(basis_dim, basis_dim), nn.ReLU(), nn.LayerNorm(basis_dim))
            self.alphas[k_loop] = nn.Parameter(torch.tensor(initial_alpha))

        if self.add_error_estimator: self.error_estimator = nn.Linear(self.num_state_vars * basis_dim, 1)

    def _get_layer(self, module_dict, key): return module_dict['shared'] if self.shared_components else module_dict[key]
    def _get_alpha(self, key): return self.alphas['shared'] if self.shared_components else self.alphas[key]
    def _compute_U_B(self, BC_Ctrl_n): # Copied from MultiVarAttentionROM
        if not self.use_fixed_lifting: 
            if self.lifting is None: raise ValueError("Lifting network is None but use_fixed_lifting is False.")
            return self.lifting(BC_Ctrl_n)
        else:
            if self.num_state_vars == 1:
                if self.bc_state_dim < 2: return torch.zeros(BC_Ctrl_n.shape[0], 1, self.nx, device=BC_Ctrl_n.device)
                l=BC_Ctrl_n[:,0:1].unsqueeze(-1);r=BC_Ctrl_n[:,1:2].unsqueeze(-1);return l*(1-self.lin_interp_coeffs)+r*self.lin_interp_coeffs
            else:
                if self.bc_state_dim < 2*self.num_state_vars: return torch.zeros(BC_Ctrl_n.shape[0],self.num_state_vars,self.nx,device=BC_Ctrl_n.device)
                ul=[];
                for i in range(self.num_state_vars):
                    il=i*2;ir=i*2+1
                    if ir>=self.bc_state_dim: usv=torch.zeros(BC_Ctrl_n.shape[0],1,self.nx,device=BC_Ctrl_n.device)
                    else: l=BC_Ctrl_n[:,il:il+1].unsqueeze(-1);r=BC_Ctrl_n[:,ir:ir+1].unsqueeze(-1);usv=l*(1-self.lin_interp_coeffs)+r*self.lin_interp_coeffs
                    ul.append(usv)
                return torch.cat(ul,dim=1)

    def forward_step(self, a_n_dict, BC_Ctrl_n, params=None):
        batch_size = list(a_n_dict.values())[0].size(0); a_next_dict = {}; U_hat_dict = {}
        U_B_stacked = self._compute_U_B(BC_Ctrl_n) 
        for i, key in enumerate(self.state_keys):
            a_n_var = a_n_dict[key]; Phi_var = self.Phi[key]
            ffn_var = self._get_layer(self.update_ffn, key); alpha_var = self._get_alpha(key)
            ffn_output = ffn_var(a_n_var) 
            a_next_var = a_n_var + alpha_var * ffn_output # Update only with FFN
            a_next_dict[key] = a_next_var
            U_B_var = U_B_stacked[:, i, :].unsqueeze(-1)
            Phi_exp = Phi_var.unsqueeze(0).expand(batch_size, -1, -1)
            a_next_unsq = a_next_var.unsqueeze(-1)
            U_recon = torch.bmm(Phi_exp, a_next_unsq)
            U_hat_dict[key] = U_B_var + U_recon
        err_est = None
        if self.add_error_estimator: err_est = self.error_estimator(torch.cat(list(a_next_dict.values()), dim=-1))
        return a_next_dict, U_hat_dict, err_est

    def forward(self, a0_dict, BC_Ctrl_seq, T, params=None): 
        a_current_dict = {}
        for key in self.state_keys: a_current_dict[key] = self._get_layer(self.a0_mapping, key)(a0_dict[key])
        U_hat_seq_dict = {key: [] for key in self.state_keys}; err_seq = [] if self.add_error_estimator else None
        for t in range(T):
            BC_Ctrl_n = BC_Ctrl_seq[:, t, :]
            a_next_dict, U_hat_dict, err_est = self.forward_step(a_current_dict, BC_Ctrl_n, params)
            for key in self.state_keys: U_hat_seq_dict[key].append(U_hat_dict[key])
            if self.add_error_estimator and err_est is not None: err_seq.append(err_est)
            a_current_dict = a_next_dict
        return U_hat_seq_dict, err_seq
    def get_basis(self, key): return self.Phi[key]

print("No-Attention ROM model classes defined.")


# --- Configuration ---
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DATASET_PATH = "./datasets_new_feedback/"
# Assuming your NoAttention ablation checkpoints are in New_ckpt_NoAttention as per your training script
BASE_CHECKPOINT_PATH = "./New_ckpt_NoAttention/" 
BASE_RESULTS_PATH = "./benchmark_no_attention_focused/" 
MAX_VISUALIZATION_SAMPLES = 5 
os.makedirs(BASE_RESULTS_PATH, exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

TARGET_DATASET_KEY = 'reaction_diffusion_neumann_feedback'
DATASET_CONFIGS = {
    TARGET_DATASET_KEY: {
        'path': os.path.join(BASE_DATASET_PATH, "reaction_diffusion_neumann_feedback_v1_5000s_64nx_300nt.pkl"),
        'T_file': 2.0, 'NT_file': 300, 'T_train': 1.5, 'NT_train': 225,
        'nx': 64, 'L': 1.0, 'state_keys': ['U'], 'num_state_vars': 1,
        'dataset_type_arg': 'reaction_diffusion_neumann_feedback'
    }
}

# Default Hyperparameters (match your training script defaults for these specific ablations)
BASIS_DIM_DEFAULT = 32
D_MODEL_FFN_DEFAULT = 512 # d_model is used as FFN hidden dim for NoAttention models

# For NoAttention_ExplicitBC_ROM
BC_PROCESSED_DIM_DEFAULT = 32 
HIDDEN_BC_PROCESSOR_DIM_DEFAULT = 128

# --- Model Configurations for the TWO No-Attention ablations ---
# We will assume shared_components=False (_pervar) and no other ablation flags (_fixedlift, _randphi)
# unless you specify checkpoints that were trained with those flags.
ABLATION_MODEL_CONFIGS = {
    "NoAttn_ImpBC_b32_pervar": { 
        "model_class_name": "NoAttention_ImplicitBC_ROM",
        # Constructed checkpoint name based on training script's run_name logic
        "checkpoint_filename": f"rom_{TARGET_DATASET_KEY}_b{BASIS_DIM_DEFAULT}_NoAttnImpBC_ffn{D_MODEL_FFN_DEFAULT}_pervar.pt", 
        "params": {
            "basis_dim": BASIS_DIM_DEFAULT, 
            "d_model": D_MODEL_FFN_DEFAULT, # For FFN hidden dim
            "use_fixed_lifting": False, 
            "shared_components": False, # Corresponds to _pervar
            "add_error_estimator": False
        },
        "prediction_fn_name": "predict_rom_model" # Generic prediction function
    },
    "NoAttn_ExpBC_b32_pervar": { 
        "model_class_name": "NoAttention_ExplicitBC_ROM",
        # Constructed checkpoint name
        "checkpoint_filename": f"rom_{TARGET_DATASET_KEY}_b{BASIS_DIM_DEFAULT}_NoAttnExpBC_ffn{D_MODEL_FFN_DEFAULT}_bcp{BC_PROCESSED_DIM_DEFAULT}_pervar.pt",
        "params": {
            "basis_dim": BASIS_DIM_DEFAULT, 
            "d_model": D_MODEL_FFN_DEFAULT, # For FFN hidden dim
            "use_fixed_lifting": False, 
            "shared_components": False, # Corresponds to _pervar
            "bc_processed_dim": BC_PROCESSED_DIM_DEFAULT, 
            "hidden_bc_processor_dim": HIDDEN_BC_PROCESSOR_DIM_DEFAULT,
            "add_error_estimator": False
        },
        "prediction_fn_name": "predict_rom_model"
    }
    # You can add entries for "_shared" versions if you trained them, e.g.:
    # "NoAttn_ImpBC_b32_shared": { ... "checkpoint_filename": "..._shared.pt", "params": {..., "shared_components": True} ...}
}

# --- Utility Functions (load_data, calculate_metrics, plot_comparison) ---
# These are assumed to be the same as in the previous benchmark script and are omitted for brevity.
# Make sure they are defined in your actual script.
def load_data(dataset_name_key):
    config = DATASET_CONFIGS[dataset_name_key]
    dataset_path = config['path']
    if not os.path.exists(dataset_path): print(f"Dataset file not found: {dataset_path}"); return None, None, None
    with open(dataset_path, 'rb') as f: data_list_all = pickle.load(f)
    n_total = len(data_list_all); n_val_start_idx = int(0.8 * n_total); val_data_list = data_list_all[n_val_start_idx:]
    if not val_data_list: print(f"No validation data for {dataset_name_key}."); return None, None, None
    print(f"Using {len(val_data_list)} samples for validation from {dataset_name_key}.")
    val_dataset = UniversalPDEDataset(val_data_list, dataset_type=config['dataset_type_arg'], train_nt_limit=None)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    return val_data_list, val_loader, config

def calculate_metrics(pred_denorm, gt_denorm):
    if pred_denorm.shape != gt_denorm.shape:
        min_len_t = min(pred_denorm.shape[0], gt_denorm.shape[0]); min_len_x = min(pred_denorm.shape[1], gt_denorm.shape[1])
        pred_denorm = pred_denorm[:min_len_t, :min_len_x]; gt_denorm = gt_denorm[:min_len_t, :min_len_x]
        if pred_denorm.shape != gt_denorm.shape: return {'mse': float('nan'), 'rmse': float('nan'), 'rel_err': float('nan'), 'max_err': float('nan')}
    mse = np.mean((pred_denorm - gt_denorm)**2); rmse = np.sqrt(mse)
    rel_err = np.linalg.norm(pred_denorm - gt_denorm, 'fro') / (np.linalg.norm(gt_denorm, 'fro') + 1e-10)
    max_err = np.max(np.abs(pred_denorm - gt_denorm)) if pred_denorm.size > 0 else 0.0
    return {'mse': mse, 'rmse': rmse, 'rel_err': rel_err, 'max_err': max_err}

def plot_comparison(gt_seq_denorm_dict, predictions_denorm_dict, dataset_name, sample_idx_str, state_keys_to_plot, L_domain, T_horizon, save_path_base):
    num_models_to_plot = len(predictions_denorm_dict) + 1; num_vars = len(state_keys_to_plot)
    if num_vars == 0: return
    fig, axs = plt.subplots(num_vars, num_models_to_plot, figsize=(6 * num_models_to_plot, 5 * num_vars), squeeze=False)
    plot_model_names = ["Ground Truth"] + list(predictions_denorm_dict.keys())
    for i_skey, skey in enumerate(state_keys_to_plot):
        gt_data_var = gt_seq_denorm_dict.get(skey)
        if gt_data_var is None: axs[i_skey,0].set_ylabel(f"{skey}\n(GT Missing)"); continue
        all_series_for_var = [gt_data_var] + [predictions_denorm_dict.get(model_name, {}).get(skey) for model_name in predictions_denorm_dict.keys()]
        valid_series = [s for s in all_series_for_var if s is not None and s.size > 0 and s.ndim == 2]
        if not valid_series: axs[i_skey,0].set_ylabel(f"{skey}\n(No valid data)"); continue
        vmin = min(s.min() for s in valid_series); vmax = max(s.max() for s in valid_series)
        if abs(vmax - vmin) < 1e-9: vmax = vmin + 1.0
        for j_model, model_name_plot in enumerate(plot_model_names):
            ax = axs[i_skey, j_model]; data_to_plot = gt_data_var if model_name_plot == "Ground Truth" else predictions_denorm_dict.get(model_name_plot, {}).get(skey, None)
            if data_to_plot is None or data_to_plot.size == 0 or data_to_plot.ndim != 2: ax.text(0.5, 0.5, "No/Invalid data", ha="center", va="center", fontsize=9)
            else:
                im = ax.imshow(data_to_plot, aspect='auto', origin='lower', extent=[0, L_domain, 0, T_horizon], cmap='viridis', vmin=vmin, vmax=vmax)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{model_name_plot}\n({skey})", fontsize=10)
            if i_skey == num_vars -1 : ax.set_xlabel("x", fontsize=10)
            if j_model == 0: ax.set_ylabel(f"{skey}\nt (physical)", fontsize=10)
            else: ax.set_yticklabels([])
            ax.tick_params(axis='both', which='major', labelsize=8)
    fig.suptitle(f"Benchmark: {dataset_name} (Sample {sample_idx_str}) @ T={T_horizon:.2f}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); final_save_path = os.path.join(save_path_base, f"comparison_{dataset_name}_sample{sample_idx_str}_T{T_horizon:.2f}.png")
    os.makedirs(save_path_base, exist_ok=True); plt.savefig(final_save_path); print(f"Saved comparison plot to {final_save_path}"); plt.close(fig)

# --- Model Loading Function for No-Attention Ablations ---
def load_no_attention_model(model_name_key, model_config, dataset_config, device):
    checkpoint_filename = model_config["checkpoint_filename"]
    # Checkpoint path from your training script for NoAttention models
    ckpt_dir = os.path.join(BASE_CHECKPOINT_PATH, f"_checkpoints_{dataset_config['dataset_type_arg']}")
    checkpoint_path = os.path.join(ckpt_dir, checkpoint_filename)

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found for {model_name_key}: {checkpoint_path}"); return None
    print(f"Loading checkpoint for {model_name_key} from {checkpoint_path}")
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False) 
    except Exception as e:
        print(f"Error loading checkpoint file {checkpoint_path}: {e}"); return None
    
    state_keys=dataset_config['state_keys']; nx_val=dataset_config['nx'] # Renamed nx to nx_val
    try: 
        with open(dataset_config['path'], 'rb') as f_dummy: dummy_data_sample = pickle.load(f_dummy)[0]
        dummy_dataset = UniversalPDEDataset([dummy_data_sample], dataset_type=dataset_config['dataset_type_arg'])
        bc_state_dim_actual = dummy_dataset.bc_state_dim; num_controls_actual = dummy_dataset.num_controls
    except Exception as e:
        print(f"Error creating dummy dataset for {model_name_key}: {e}. Using defaults from ckpt/config.")
        bc_state_dim_actual = ckpt.get('bc_state_dim', model_config["params"].get('bc_state_dim', 2))
        num_controls_actual = ckpt.get('num_controls', model_config["params"].get('num_controls', 2))

    cfg_params = model_config["params"]
    model_class_name = model_config["model_class_name"]
    
    basis_dim_final = ckpt.get('basis_dim', cfg_params.get('basis_dim'))
    use_fixed_lifting_final = ckpt.get('use_fixed_lifting', cfg_params.get('use_fixed_lifting', False))
    shared_components_final = ckpt.get('shared_components', cfg_params.get('shared_components', False))

    d_model_final = ckpt.get('d_model', cfg_params.get('d_model')) # For FFN hidden dim
    initial_alpha_final = ckpt.get('initial_alpha', cfg_params.get('initial_alpha', 0.1))

    if model_class_name == "NoAttention_ExplicitBC_ROM":
        # --- START MODIFICATION ---
        # Robustly get bc_processed_dim
        bc_processed_dim_ckpt = ckpt.get('bc_processed_dim')
        final_bc_processed_dim = bc_processed_dim_ckpt if bc_processed_dim_ckpt is not None \
                                 else cfg_params.get('bc_processed_dim', BC_PROCESSED_DIM_DEFAULT)
        if final_bc_processed_dim is None: # Fallback if key somehow missing from cfg_params too
            final_bc_processed_dim = BC_PROCESSED_DIM_DEFAULT

        # Robustly get hidden_bc_processor_dim
        hidden_bc_processor_dim_ckpt = ckpt.get('hidden_bc_processor_dim')
        final_hidden_bc_processor_dim = hidden_bc_processor_dim_ckpt if hidden_bc_processor_dim_ckpt is not None \
                                        else cfg_params.get('hidden_bc_processor_dim', HIDDEN_BC_PROCESSOR_DIM_DEFAULT)
        if final_hidden_bc_processor_dim is None: # Fallback
            final_hidden_bc_processor_dim = HIDDEN_BC_PROCESSOR_DIM_DEFAULT
        # --- END MODIFICATION ---

        model = NoAttention_ExplicitBC_ROM(
            state_variable_keys=state_keys, nx=nx_val, basis_dim=basis_dim_final,
            d_model=d_model_final,
            bc_state_dim=bc_state_dim_actual, num_controls=num_controls_actual,
            use_fixed_lifting=use_fixed_lifting_final,
            shared_components=shared_components_final,
            # Use the robustly loaded values
            bc_processed_dim=final_bc_processed_dim,
            hidden_bc_processor_dim=final_hidden_bc_processor_dim,
            initial_alpha=initial_alpha_final,
            add_error_estimator=False # Assuming not used for these benchmarks
        )
    elif model_class_name == "NoAttention_ImplicitBC_ROM":
        model = NoAttention_ImplicitBC_ROM(
            state_variable_keys=state_keys, nx=nx_val, basis_dim=basis_dim_final,
            d_model=d_model_final,
            bc_state_dim=bc_state_dim_actual, num_controls=num_controls_actual,
            use_fixed_lifting=use_fixed_lifting_final,
            shared_components=shared_components_final,
            initial_alpha=initial_alpha_final,
            add_error_estimator=False
        )
    else: print(f"Unknown model class for NoAttention benchmark: {model_class_name}"); return None
    
    try: model.load_state_dict(ckpt['model_state_dict'], strict=False)
    except Exception as e: print(f"Error loading state_dict for {model_name_key}: {e}"); traceback.print_exc(); return None
    model.to(device); model.eval(); print(f"Successfully loaded {model_name_key}."); return model

# --- Prediction Function for No-Attention ROMs ---
@torch.no_grad()
def predict_no_attention_rom_model(model, initial_state_dict_norm, bc_ctrl_seq_norm, num_steps, dataset_config):
    a0_dict = {}; batch_size = list(initial_state_dict_norm.values())[0].shape[0] 
    device_model = next(model.parameters()).device 
    initial_state_dict_norm = {k: v.to(device_model) for k, v in initial_state_dict_norm.items()}
    bc_ctrl_seq_norm = bc_ctrl_seq_norm.to(device_model)
    BC_ctrl_t0_norm = bc_ctrl_seq_norm[:, 0, :]
    U_B0_lifted_norm = model._compute_U_B(BC_ctrl_t0_norm) 
    for i_key, key in enumerate(model.state_keys):
        U0_norm_var = initial_state_dict_norm[key]
        if U0_norm_var.dim() == 2: U0_norm_var = U0_norm_var.unsqueeze(-1)
        U_B0_norm_var = U_B0_lifted_norm[:, i_key, :].unsqueeze(-1)
        U0_star_norm_var = U0_norm_var - U_B0_norm_var
        Phi_var = model.get_basis(key).to(device_model)
        Phi_T_var = Phi_var.transpose(0,1).unsqueeze(0).expand(batch_size, -1, -1)
        a0_norm_var = torch.bmm(Phi_T_var, U0_star_norm_var).squeeze(-1)
        a0_dict[key] = a0_norm_var
    
    pred_seq_norm_dict_of_lists, _ = model(a0_dict, bc_ctrl_seq_norm[:, :num_steps, :], T=num_steps)
    
    pred_seq_norm_dict_of_tensors = {}
    for key in model.state_keys:
        if pred_seq_norm_dict_of_lists.get(key) and len(pred_seq_norm_dict_of_lists[key]) > 0 :
            stacked_preds = torch.stack(pred_seq_norm_dict_of_lists[key], dim=1) 
            pred_seq_norm_dict_of_tensors[key] = stacked_preds.squeeze(0).squeeze(-1)
        else: 
            pred_seq_norm_dict_of_tensors[key] = torch.empty(0, dataset_config['nx'], device=device_model)
    return pred_seq_norm_dict_of_tensors

# --- Main Benchmarking Logic ---
def main(models_to_benchmark_keys):
    print(f"Device: {DEVICE}")
    dataset_name_key = TARGET_DATASET_KEY
    print(f"Benchmarking models: {models_to_benchmark_keys} on dataset {dataset_name_key}")
    
    BENCHMARK_T_HORIZONS = [1.75] # Updated T horizons

    overall_aggregated_metrics = {dataset_name_key: { model_name: { T_h: {skey: {'mse': [], 'rel_err': [], 'rmse': [], 'max_err': []} for skey in DATASET_CONFIGS[dataset_name_key]['state_keys']} for T_h in BENCHMARK_T_HORIZONS if T_h <= DATASET_CONFIGS[dataset_name_key]['T_file'] } for model_name in models_to_benchmark_keys}}
            
    val_data_list, val_loader, ds_config = load_data(dataset_name_key)
    if val_loader is None: print(f"Failed to load data for {dataset_name_key}. Exiting."); return

    num_val_samples = len(val_data_list); vis_sample_count = min(MAX_VISUALIZATION_SAMPLES, num_val_samples)
    visualization_indices = random.sample(range(num_val_samples), vis_sample_count) if num_val_samples > 0 else []
    print(f"Will visualize {len(visualization_indices)} random samples: {visualization_indices}")
    
    gt_data_for_visualization = {vis_idx: None for vis_idx in visualization_indices}
    predictions_for_visualization_all_models_all_horizons = {vis_idx: {model_name: {} for model_name in models_to_benchmark_keys} for vis_idx in visualization_indices}

    loaded_models_cache = {}
    for model_name_key in models_to_benchmark_keys:
        print(f"\n  -- Pre-loading Model: {model_name_key} for dataset {dataset_name_key} --")
        if model_name_key not in ABLATION_MODEL_CONFIGS: print(f"  Skipping model {model_name_key}: Not configured."); loaded_models_cache[model_name_key] = None; continue
        model_arch_config = ABLATION_MODEL_CONFIGS[model_name_key]
        # Using the specific loader for NoAttention models
        model_instance = load_no_attention_model(model_name_key, model_arch_config, ds_config, DEVICE)
        loaded_models_cache[model_name_key] = model_instance
        if model_instance is None: print(f"  Failed to pre-load model {model_name_key}.")

    for val_idx, (sample_state_list_norm, sample_bc_ctrl_seq_norm, sample_norm_factors) in enumerate(val_loader):
        print(f"  Processing validation sample {val_idx+1}/{num_val_samples} for dataset {dataset_name_key}...")
        initial_state_norm_dict = {}; gt_full_seq_denorm_dict_current_sample = {}
        for idx_skey, skey in enumerate(ds_config['state_keys']):
            initial_state_norm_dict[skey] = sample_state_list_norm[idx_skey][:, 0, :].to(DEVICE)
            gt_seq_norm_var = sample_state_list_norm[idx_skey].squeeze(0).to(DEVICE)
            mean_val = sample_norm_factors[f'{skey}_mean'].item(); std_val = sample_norm_factors[f'{skey}_std'].item()
            gt_full_seq_denorm_dict_current_sample[skey] = gt_seq_norm_var.cpu().numpy() * std_val + mean_val
        if val_idx in visualization_indices: gt_data_for_visualization[val_idx] = gt_full_seq_denorm_dict_current_sample
        
        for T_current_horizon in BENCHMARK_T_HORIZONS:
            if T_current_horizon > ds_config['T_file']: continue
            num_benchmark_steps = min(int((T_current_horizon / ds_config['T_file']) * (ds_config['NT_file'] - 1)) + 1 if ds_config['T_file'] > 1e-6 else ds_config['NT_file'], ds_config['NT_file'])
            num_benchmark_steps = max(1, num_benchmark_steps)
            gt_benchmark_horizon_denorm_dict = {skey: gt_full_seq_denorm_dict_current_sample[skey][:num_benchmark_steps, :] for skey in ds_config['state_keys']}
            sample_bc_ctrl_seq_norm_benchmark = sample_bc_ctrl_seq_norm[:, :num_benchmark_steps, :].to(DEVICE)

            for model_name_key in models_to_benchmark_keys:
                current_model_instance = loaded_models_cache.get(model_name_key)
                if current_model_instance is None: continue
                model_arch_config = ABLATION_MODEL_CONFIGS[model_name_key]
                prediction_fn_name = model_arch_config["prediction_fn_name"] # Should be "predict_no_attention_rom_model"
                pred_seq_norm_dict_model = {}
                try:
                    if prediction_fn_name == "predict_rom_model": # Using generic name for consistency
                        pred_seq_norm_dict_model = predict_no_attention_rom_model(current_model_instance, initial_state_norm_dict, sample_bc_ctrl_seq_norm_benchmark, num_benchmark_steps, ds_config)
                    else: print(f"Unknown prediction function: {prediction_fn_name} for model {model_name_key}"); continue
                except Exception as e: print(f"    ERROR during prediction for {model_name_key} on sample {val_idx}, T={T_current_horizon}: {e}"); traceback.print_exc(); continue
                
                model_preds_denorm_this_sample_this_horizon = {}
                for skey in ds_config['state_keys']:
                    if skey not in pred_seq_norm_dict_model or pred_seq_norm_dict_model[skey] is None or pred_seq_norm_dict_model[skey].numel() == 0: continue
                    pred_norm_var = pred_seq_norm_dict_model[skey].cpu().numpy()
                    gt_denorm_var = gt_benchmark_horizon_denorm_dict[skey]
                    mean_val = sample_norm_factors[f'{skey}_mean'].item(); std_val = sample_norm_factors[f'{skey}_std'].item()
                    pred_denorm_var = pred_norm_var * std_val + mean_val
                    model_preds_denorm_this_sample_this_horizon[skey] = pred_denorm_var
                    metrics = calculate_metrics(pred_denorm_var, gt_denorm_var)
                    for metric_name, metric_val in metrics.items():
                        if T_current_horizon in overall_aggregated_metrics[dataset_name_key][model_name_key]:
                            overall_aggregated_metrics[dataset_name_key][model_name_key][T_current_horizon][skey][metric_name].append(metric_val)
                if val_idx in visualization_indices:
                    predictions_for_visualization_all_models_all_horizons[val_idx][model_name_key][T_current_horizon] = model_preds_denorm_this_sample_this_horizon
            torch.cuda.empty_cache()
            
    print(f"\n  Generating visualizations for {dataset_name_key}...")
    for vis_idx in visualization_indices:
        if vis_idx not in gt_data_for_visualization or gt_data_for_visualization[vis_idx] is None: continue
        for T_current_horizon_plot in BENCHMARK_T_HORIZONS:
            if T_current_horizon_plot > ds_config['T_file']: continue
            num_plot_steps = min(int((T_current_horizon_plot / ds_config['T_file']) * (ds_config['NT_file'] - 1)) + 1 if ds_config['T_file'] > 1e-6 else ds_config['NT_file'], ds_config['NT_file'])
            num_plot_steps = max(1, num_plot_steps)
            current_gt_denorm_sliced_for_plot = {skey: data[:num_plot_steps, :] for skey, data in gt_data_for_visualization[vis_idx].items()}
            current_preds_for_plot_this_horizon = {model_name_plot_key: predictions_for_visualization_all_models_all_horizons[vis_idx][model_name_plot_key].get(T_current_horizon_plot) for model_name_plot_key in models_to_benchmark_keys if predictions_for_visualization_all_models_all_horizons[vis_idx][model_name_plot_key].get(T_current_horizon_plot) is not None}
            if not current_preds_for_plot_this_horizon: continue
            plot_comparison(current_gt_denorm_sliced_for_plot, current_preds_for_plot_this_horizon, dataset_name_key, f"{val_idx}", ds_config['state_keys'], ds_config['L'], T_current_horizon_plot, os.path.join(BASE_RESULTS_PATH, dataset_name_key, "plots"))

    print("\n\n===== BENCHMARKING COMPLETE - AGGREGATED METRICS =====")
    for ds_name_print, model_data_agg_per_horizon in overall_aggregated_metrics.items():
        print(f"\n--- Aggregated Results for Dataset: {ds_name_print.upper()} ---")
        for model_name_print, horizon_metrics_data in model_data_agg_per_horizon.items():
            print(f"  Model: {model_name_print}")
            if not horizon_metrics_data: print("    No metrics recorded for this model."); continue
            for T_h_print, var_metrics_lists in sorted(horizon_metrics_data.items()):
                print(f"    Horizon T={T_h_print:.2f}:")
                for skey_print, metrics_lists in var_metrics_lists.items():
                    if not metrics_lists['mse'] : print(f"      {skey_print}: No metrics recorded for this horizon."); continue
                    avg_mse=np.mean(metrics_lists['mse']) if metrics_lists['mse'] else float('nan'); avg_rmse=np.mean(metrics_lists['rmse']) if metrics_lists['rmse'] else float('nan'); avg_rel_err=np.mean(metrics_lists['rel_err']) if metrics_lists['rel_err'] else float('nan'); avg_max_err=np.mean(metrics_lists['max_err']) if metrics_lists['max_err'] else float('nan')
                    print(f"      {skey_print}: Avg MSE={avg_mse:.3e}, Avg RMSE={avg_rmse:.3e}, Avg RelErr={avg_rel_err:.3e}, Avg MaxErr={avg_max_err:.3e} (from {len(metrics_lists['mse'])} samples)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark No-Attention ROM Ablation Study Models.")
    # Default to the two specific No-Attention models
    default_models_to_benchmark = ["NoAttn_ImpBC_b32_pervar", "NoAttn_ExpBC_b32_pervar"]
    parser.add_argument('--models', nargs='+', 
                        default=default_models_to_benchmark,
                        choices=list(ABLATION_MODEL_CONFIGS.keys()), 
                        help='Which specific No-Attention ablation models to benchmark.')
    args = parser.parse_args()
    
    # Filter ABLATION_MODEL_CONFIGS to only include models specified by args.models
    # This ensures that if the user provides a subset, only those are processed.
    # The main loop already iterates through `args.models`.
    
    main(args.models)
