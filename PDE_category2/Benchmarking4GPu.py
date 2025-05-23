# BENCHMARK_PDE_MODELS for there inference performance regarding GPU memories usage, inference efficiency times / sample
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import time
import pickle
import argparse
from torch.utils.data import Dataset, DataLoader
import glob # For finding files with wildcards
import traceback
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

# --- Import Model Definitions and UniversalPDEDataset ---
class UniversalPDEDataset(Dataset):
    def __init__(self, data_list, dataset_type, train_nt_limit=None):
        if not data_list:
            raise ValueError("data_list cannot be empty")
        self.data_list = data_list
        self.dataset_type = dataset_type 
        self.train_nt_limit = train_nt_limit
        first_sample = data_list[0]
        params = first_sample.get('params', {})
        self.nt_from_sample = 0
        self.nx_from_sample = 0
        self.ny_from_sample = 1
        self.state_keys = []
        self.num_state_vars = 0
        self.expected_bc_state_dim = 0
        dataset_type_lower = dataset_type.lower()

        if dataset_type_lower == 'advection' or dataset_type_lower == 'burgers':
            self.nt_from_sample = first_sample['U'].shape[0]
            self.nx_from_sample = first_sample['U'].shape[1]
            self.state_keys = ['U']; self.num_state_vars = 1
            self.expected_bc_state_dim = 2
        elif dataset_type_lower == 'euler':
            self.nt_from_sample = first_sample['rho'].shape[0]
            self.nx_from_sample = first_sample['rho'].shape[1]
            self.state_keys = ['rho', 'u']; self.num_state_vars = 2
            self.expected_bc_state_dim = 4
        elif dataset_type_lower == 'darcy':
            self.nt_from_sample = first_sample['P'].shape[0]
            self.nx_from_sample = params.get('nx', first_sample['P'].shape[1])
            self.ny_from_sample = params.get('ny', 1)
            self.state_keys = ['P']; self.num_state_vars = 1
            self.expected_bc_state_dim = 2
        elif dataset_type_lower in ['heat_delayed_feedback',
                               'reaction_diffusion_neumann_feedback',
                               'heat_nonlinear_feedback_gain',
                               'convdiff', 'convdiff_feedback']:
            self.nt_from_sample = first_sample['U'].shape[0]
            self.nx_from_sample = first_sample['U'].shape[1]
            self.state_keys = ['U']; self.num_state_vars = 1
            self.expected_bc_state_dim = 2
        else:
            raise ValueError(f"Unknown dataset_type in UniversalPDEDataset: {dataset_type}")

        self.effective_nt = self.train_nt_limit if self.train_nt_limit is not None else self.nt_from_sample
        self.nx = self.nx_from_sample
        self.ny = self.ny_from_sample
        self.spatial_dim = self.nx * self.ny
        self.bc_state_key = 'BC_State'
        if self.bc_state_key not in first_sample:
            raise KeyError(f"'{self.bc_state_key}' not found in the first sample for {dataset_type}!")
        actual_bc_state_dim = first_sample[self.bc_state_key].shape[1]

        if dataset_type_lower in ['heat_delayed_feedback', 'reaction_diffusion_neumann_feedback', 'heat_nonlinear_feedback_gain', 'convdiff', 'convdiff_feedback'] and actual_bc_state_dim != self.expected_bc_state_dim:
            print(f"Info: For {dataset_type}, expected BC_State dim {self.expected_bc_state_dim}, got {actual_bc_state_dim}. Using actual: {actual_bc_state_dim}")
            self.bc_state_dim = actual_bc_state_dim
        elif actual_bc_state_dim != self.expected_bc_state_dim :
             print(f"Warning: BC_State dimension mismatch for {dataset_type}. "
                    f"Expected {self.expected_bc_state_dim}, got {actual_bc_state_dim}. "
                    f"Using actual dimension: {actual_bc_state_dim}")
             self.bc_state_dim = actual_bc_state_dim
        else:
            self.bc_state_dim = self.expected_bc_state_dim

        self.bc_control_key = 'BC_Control'
        if self.bc_control_key in first_sample and first_sample[self.bc_control_key] is not None and first_sample[self.bc_control_key].size > 0:
            self.num_controls = first_sample[self.bc_control_key].shape[1]
        else:
            self.num_controls = 0

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]; norm_factors = {}; current_nt = self.effective_nt
        state_tensors_norm_list = []
        for key in self.state_keys:
            state_seq_full = sample[key]; state_seq = state_seq_full[:current_nt, ...]
            if state_seq.shape[0] != current_nt: raise ValueError(f"Time dim mismatch for {key}. Expected {current_nt}, got {state_seq.shape[0]}")
            state_mean = np.mean(state_seq); state_std = np.std(state_seq) + 1e-8
            state_norm = (state_seq - state_mean) / state_std
            state_tensors_norm_list.append(torch.tensor(state_norm).float())
            norm_factors[f'{key}_mean'] = state_mean; norm_factors[f'{key}_std'] = state_std

        bc_state_seq_full = sample[self.bc_state_key]; bc_state_seq = bc_state_seq_full[:current_nt, :]
        if bc_state_seq.shape[0] != current_nt: raise ValueError(f"Time dim mismatch for BC_State. Expected {current_nt}, got {bc_state_seq.shape[0]}")
        bc_state_norm = np.zeros_like(bc_state_seq, dtype=np.float32)
        norm_factors[f'{self.bc_state_key}_means'] = np.zeros(self.bc_state_dim); norm_factors[f'{self.bc_state_key}_stds'] = np.ones(self.bc_state_dim)
        if self.bc_state_dim > 0 :
            for k_dim in range(self.bc_state_dim):
                col = bc_state_seq[:, k_dim]; mean_k = np.mean(col); std_k = np.std(col)
                if std_k > 1e-8:
                    bc_state_norm[:, k_dim] = (col - mean_k) / std_k
                    norm_factors[f'{self.bc_state_key}_means'][k_dim] = mean_k; norm_factors[f'{self.bc_state_key}_stds'][k_dim] = std_k
                else:
                    bc_state_norm[:, k_dim] = col - mean_k; norm_factors[f'{self.bc_state_key}_means'][k_dim] = mean_k
        bc_state_tensor_norm = torch.tensor(bc_state_norm).float()

        if self.num_controls > 0:
            bc_control_seq_full = sample[self.bc_control_key]; bc_control_seq = bc_control_seq_full[:current_nt, :]
            if bc_control_seq.shape[0] != current_nt: raise ValueError(f"Time dim mismatch for BC_Control. Expected {current_nt}, got {bc_control_seq.shape[0]}.")
            if bc_control_seq.shape[1] != self.num_controls: raise ValueError(f"Control dim mismatch in sample {idx}. Expected {self.num_controls}, got {bc_control_seq.shape[1]}.")
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
        else:
            bc_control_tensor_norm = torch.empty((current_nt, 0), dtype=torch.float32)
        bc_ctrl_tensor_norm = torch.cat((bc_state_tensor_norm, bc_control_tensor_norm), dim=-1)
        return state_tensors_norm_list, bc_ctrl_tensor_norm, norm_factors


from FNO_FULL import FNO1d, SpectralConv1d
from SPFNO_FULL import SPFNO1d as SPFNO_SPFNO1d, ProjectionFilter1d, sin_transform, isin_transform, cos_transform, icos_transform, WSWA, iWSWA
from BENO_FULL import BENOStepper, BoundaryEmbedder, GNNLikeProcessor as BENO_GNNProcessor, MLP as BENO_MLP, TransformerEncoderLayer as BENO_TransformerEncoderLayer
from LNO_FULL import LatentNeuralOperator as LNO_LatentNeuralOperator, MLP as LNO_MLP, PhysicsCrossAttention as LNO_PhysicsCrossAttention
from LNS_AE_FULL import LNS_Autoencoder, LatentStepperNet as LNS_LatentStepperNet, MLP as LNS_MLP, ConvBlock as LNS_ConvBlock, ResConvBlock as LNS_ResConvBlock, LNS_Encoder, LNS_Decoder
from POD_DL_ROM_FULL import POD_DL_ROM, Encoder as POD_Encoder, Decoder as POD_Decoder, DFNN as POD_DFNN, compute_pod_basis_generic as compute_pod_basis_pod_dl_rom
from ROM_FULL import MultiVarAttentionROM, UniversalLifting as ROM_UniversalLifting, ImprovedUpdateFFN as ROM_ImprovedUpdateFFN, MultiHeadAttentionROM as ROM_MultiHeadAttentionROM

# --- Configuration ---
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DATASET_PATH = "./datasets_new_feedback/"
BASE_CHECKPOINT_PATH = "./New_ckpt_2/"
BASE_RESULTS_PATH = "./benchmark_result_t1p75/"
MAX_VISUALIZATION_SAMPLES = 500 # Max number of samples to visualize per dataset
os.makedirs(BASE_RESULTS_PATH, exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

DATASET_CONFIGS = {
    'heat_delayed_feedback': {'path': BASE_DATASET_PATH + "heat_delayed_feedback_v1_5000s_64nx_300nt.pkl",
                               'T_file': 2.0, 'NT_file': 300, 'T_train': 1.5, 'NT_train': 225,
                               'nx': 64, 'L': 1.0, 'state_keys': ['U'], 'num_state_vars': 1, 'dataset_type_arg': 'heat_delayed_feedback'},
    'reaction_diffusion_neumann_feedback': {'path': BASE_DATASET_PATH + "reaction_diffusion_neumann_feedback_v1_5000s_64nx_300nt.pkl",
                                            'T_file': 2.0, 'NT_file': 300, 'T_train': 1.5, 'NT_train': 225,
                                            'nx': 64, 'L': 1.0, 'state_keys': ['U'], 'num_state_vars': 1, 'dataset_type_arg': 'reaction_diffusion_neumann_feedback'},
    'heat_nonlinear_feedback_gain': {'path': BASE_DATASET_PATH + "heat_nonlinear_feedback_gain_v1_5000s_64nx_300nt.pkl",
                                    'T_file': 2.0, 'NT_file': 300, 'T_train': 1.5, 'NT_train': 225,
                                    'nx': 64, 'L': 1.0, 'state_keys': ['U'], 'num_state_vars': 1, 'dataset_type_arg': 'heat_nonlinear_feedback_gain'},
    'convdiff_feedback': {'path': BASE_DATASET_PATH + "convdiff_v1_5000s_64nx_300nt.pkl",
                           'T_file': 2.0, 'NT_file': 300, 'T_train': 1.5, 'NT_train': 225,
                           'nx': 64, 'L': 1.0, 'state_keys': ['U'], 'num_state_vars': 1, 'dataset_type_arg': 'convdiff_feedback'},
    'convdiff': {'path': BASE_DATASET_PATH + "convdiff_v1_5000s_64nx_300nt.pkl",
                 'T_file': 2.0, 'NT_file': 300, 'T_train': 1.5, 'NT_train': 225,
                 'nx': 64, 'L': 1.0, 'state_keys': ['U'], 'num_state_vars': 1, 'dataset_type_arg': 'convdiff'},
}

MODEL_CONFIGS = {
    "BAROM": {
        "loader_fn": "load_rom_full_model",
        "checkpoint_pattern": "_checkpoints_{dataset_type_arg}/barom_{dataset_type_arg}_b{basis_dim}_d{d_model}.pt",
        "params": {"basis_dim": 32, "d_model": 512, "num_heads": 8, "add_error_estimator": True, "shared_attention": False}
    },
    "FNO": {
        "loader_fn": "load_fno_model",
        "checkpoint_pattern": "checkpoints_{dataset_type_arg}_FNO/model_{dataset_type_arg}_FNO_trainT{T_train_str}_m{modes}_w{width}.pt",
        "params": {"modes": 16, "width": 64, "num_layers":4}
    },
    "SPFNO": {
        "loader_fn": "load_spfno_model",
        "checkpoint_pattern": "checkpoints_{dataset_type_arg}_SPFNO/model_{dataset_type_arg}_SPFNO_trainT{T_train_str}_m{modes}_w{width}.pt", # Removed timestamp
        "params": {"modes": 16, "width": 64, "num_layers":4, "transform_type": "dirichlet", "use_projection_filter":True}
    },
    "BENO": {
        "loader_fn": "load_beno_model",
        "checkpoint_pattern": "checkpoints_{dataset_type_arg}_BENO_Stepper/model_{dataset_type_arg}_BENO_Stepper_trainT{T_train_str}_emb{embed_dim}.pt",
        "params": {"embed_dim": 64, "hidden_dim": 64, "gnn_layers": 4, "transformer_layers": 2, "nhead": 4}
    },
    "LNO": {
        "loader_fn": "load_lno_model",
        "checkpoint_pattern": "checkpoints_{dataset_type_arg}_LNO_AR/model_{dataset_type_arg}_LNO_AR_trainT{T_train_str}_M{M_latent_points}_D{D_embedding}.pt", # Removed timestamp
        "params": {"M_latent_points": 64, "D_embedding": 64, "L_transformer_blocks": 3, "transformer_nhead": 4, "transformer_dim_feedforward": 128, "coord_dim": 2}
    },
     "LNS_AE": {
        "loader_fn": "load_lns_ae_model",
        "ae_checkpoint_pattern": "checkpoints_{dataset_type_arg}_LatentDON_Stepper/model_ae_{dataset_type_arg}_LatentDON_Stepper_AE_w{ae_initial_width}_lc{ae_latent_channels}.pt",
        "stepper_checkpoint_pattern": "checkpoints_{dataset_type_arg}_LatentDON_Stepper/model_prop_{dataset_type_arg}_LatentDON_Stepper_Prop_p{combined_output_p}.pt",
        "params": { "ae_initial_width": 64, "ae_downsample_blocks": 3, "ae_latent_channels": 16,
                    "branch_hidden_dims": [128,128], "trunk_hidden_dims": [64,64], "combined_output_p": 128, }
    },
    "POD_DL_ROM": {
        "loader_fn": "load_pod_dl_rom_model",
        "checkpoint_pattern": "checkpoints_{dataset_type_arg}_POD_DL_ROM/model_{dataset_type_arg}_POD_DL_ROM_trainT{T_train_str}_N{N_pod}_n{n_latent}.pt", # Removed timestamp
        "params": {"N_pod": 64, "n_latent": 8, "omega_h":0.5},
        "basis_cache_pattern": "pod_bases_cache_{dataset_type_arg}/pod_basis_{state_key}_nx{nx}_N{N_pod}_trainNT225.npy" # Corrected to use NT_train
    },
}

# --- Utility Functions ---
def load_data(dataset_name_key, base_path=""):
    config = DATASET_CONFIGS[dataset_name_key]
    dataset_path = config['path']
    if not os.path.exists(dataset_path): print(f"Dataset file not found: {dataset_path}"); return None, None, None
    with open(dataset_path, 'rb') as f: data_list_all = pickle.load(f)
    random.Random(SEED).shuffle(data_list_all)
    n_total = len(data_list_all); n_train = int(0.8 * n_total)
    val_data_list = data_list_all[n_train:n_train+100]
    if not val_data_list: print(f"No validation data for {dataset_name_key}."); return None, None, None
    val_dataset = UniversalPDEDataset(val_data_list, dataset_type=config['dataset_type_arg'], train_nt_limit=None)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False) # batch_size=1 for per-sample processing
    return val_data_list, val_loader, config

def get_checkpoint_path(pattern, dataset_type_arg, T_train_val, params, base_checkpoint_path, timestamp_wildcard="*"):
    T_train_str = str(float(T_train_val))
    format_params = params.copy()
    format_params['dataset_type_arg'] = dataset_type_arg
    format_params['T_train_str'] = T_train_str
    
    # Only add timestamp to format_params if the pattern actually uses it
    if '{timestamp}' in pattern:
        format_params['timestamp'] = timestamp_wildcard
    elif timestamp_wildcard != "*" and "*" in pattern: # If pattern has literal '*' but we passed specific timestamp
        print(f"Warning: Pattern '{pattern}' contains '*' but specific timestamp was intended. Using wildcard.")
        # This case is tricky; assumes '*' in pattern is for timestamp if timestamp_wildcard is not the default '*'
    
    # Check for missing keys before formatting
    import re
    required_keys_in_pattern = set(re.findall(r'{(\w+)}', pattern))
    missing_keys = required_keys_in_pattern - set(format_params.keys())
    if missing_keys:
        # print(f"Debug: Pattern: {pattern}")
        # print(f"Debug: Format Params Keys: {list(format_params.keys())}")
        print(f"Warning: Missing keys {missing_keys} in params for pattern {pattern}. This might cause a formatting error.")
        # Attempt to fill with dummy values for formatting to proceed, though it likely won't match
        for key in missing_keys:
            format_params[key] = "MISSING_PARAM"


    try:
        formatted_path_segment = pattern.format(**format_params)
    except KeyError as e:
        print(f"  ERROR formatting checkpoint pattern '{pattern}' with params {format_params}. Missing key: {e}")
        return None

    full_glob_pattern = os.path.join(base_checkpoint_path, formatted_path_segment)
    print(f"  Searching for checkpoint with pattern: {full_glob_pattern}")
    matching_files = glob.glob(full_glob_pattern)
    if matching_files:
        found_path = max(matching_files, key=os.path.getmtime) if len(matching_files) > 1 and "*" in full_glob_pattern else matching_files[0]
        print(f"  Found checkpoint: {found_path}")
        return found_path
    else:
        print(f"  Warning: No checkpoint found for pattern {full_glob_pattern}")
        return None

# --- Model Loading Functions (Copied and adapted, ensure correctness) ---
# ... (load_rom_full_model, load_fno_model, etc. - assumed correct from previous version)
def load_rom_full_model(checkpoint_path, model_params, dataset_config, device):
    print(f"Attempting to load BAROM model from: {checkpoint_path}")
    if not checkpoint_path or not os.path.exists(checkpoint_path): # Check moved here
        print(f"BAROM checkpoint not found at resolved path: {checkpoint_path}")
        return None
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_keys = dataset_config['state_keys']
    nx = dataset_config['nx']
    dummy_data_list = [pickle.load(open(dataset_config['path'], 'rb'))[0]]
    dummy_dataset = UniversalPDEDataset(dummy_data_list, dataset_type=dataset_config['dataset_type_arg'])
    bc_state_dim_actual = dummy_dataset.bc_state_dim
    num_controls_actual = dummy_dataset.num_controls
    model = MultiVarAttentionROM(
        state_variable_keys=state_keys, nx=nx, basis_dim=model_params['basis_dim'],
        d_model=model_params['d_model'], bc_state_dim=bc_state_dim_actual,
        num_controls=num_controls_actual, num_heads=model_params['num_heads'],
        add_error_estimator=model_params.get('add_error_estimator', False),
        shared_attention=model_params.get('shared_attention', False)
    )
    model.load_state_dict(ckpt['model_state_dict']); model.to(device); model.eval()
    print("BAROM model loaded successfully."); return model

def load_fno_model(checkpoint_path, model_params, dataset_config, device):
    print(f"Attempting to load FNO model from: {checkpoint_path}")
    if not checkpoint_path or not os.path.exists(checkpoint_path): print(f"FNO checkpoint not found: {checkpoint_path}"); return None
    ckpt = torch.load(checkpoint_path, map_location=device)
    dummy_data_list = [pickle.load(open(dataset_config['path'], 'rb'))[0]]
    dummy_dataset = UniversalPDEDataset(dummy_data_list, dataset_type=dataset_config['dataset_type_arg'])
    input_channels = dataset_config['num_state_vars'] + dummy_dataset.bc_state_dim + dummy_dataset.num_controls
    output_channels = dataset_config['num_state_vars']
    model = FNO1d(
        modes=ckpt.get('modes', model_params['modes']), width=ckpt.get('width', model_params['width']),
        input_channels=input_channels, output_channels=output_channels,
        num_layers=ckpt.get('num_layers', model_params.get('num_layers',4))
    )
    model.load_state_dict(ckpt['model_state_dict']); model.to(device); model.eval()
    print("FNO model loaded successfully."); return model

def load_spfno_model(checkpoint_path, model_params, dataset_config, device):
    print(f"Attempting to load SPFNO model from: {checkpoint_path}")
    if not checkpoint_path or not os.path.exists(checkpoint_path): print(f"SPFNO checkpoint not found: {checkpoint_path}"); return None
    ckpt = torch.load(checkpoint_path, map_location=device)
    dummy_data_list = [pickle.load(open(dataset_config['path'], 'rb'))[0]]
    dummy_dataset = UniversalPDEDataset(dummy_data_list, dataset_type=dataset_config['dataset_type_arg'])
    input_channels = dataset_config['num_state_vars'] + dummy_dataset.bc_state_dim + dummy_dataset.num_controls
    output_channels = dataset_config['num_state_vars']
    transform_type = ckpt.get('transform_type', model_params.get('transform_type', 'dirichlet'))
    model = SPFNO_SPFNO1d(
        modes=ckpt.get('modes', model_params['modes']), width=ckpt.get('width', model_params['width']),
        input_channels=input_channels, output_channels=output_channels,
        num_layers=ckpt.get('num_layers', model_params.get('num_layers',4)),
        transform_type=transform_type,
        use_projection_filter=ckpt.get('use_projection_filter', model_params.get('use_projection_filter', True))
    )
    model.load_state_dict(ckpt['model_state_dict']); model.nx = dataset_config['nx']; model.to(device); model.eval()
    print("SPFNO model loaded successfully."); return model

def load_beno_model(checkpoint_path, model_params, dataset_config, device):
    print(f"Attempting to load BENO model from: {checkpoint_path}")
    if not checkpoint_path or not os.path.exists(checkpoint_path): print(f"BENO checkpoint not found: {checkpoint_path}"); return None
    ckpt = torch.load(checkpoint_path, map_location=device)
    dummy_data_list = [pickle.load(open(dataset_config['path'], 'rb'))[0]]
    dummy_dataset = UniversalPDEDataset(dummy_data_list, dataset_type=dataset_config['dataset_type_arg'])
    bc_ctrl_dim_input = dummy_dataset.bc_state_dim + dummy_dataset.num_controls
    model = BENOStepper(
        nx=dataset_config['nx'], num_state_vars=dataset_config['num_state_vars'],
        bc_ctrl_dim_input=bc_ctrl_dim_input, state_keys=dataset_config['state_keys'],
        embed_dim=ckpt.get('embed_dim', model_params['embed_dim']),
        hidden_dim=ckpt.get('hidden_dim', model_params['hidden_dim']),
        gnn_layers=ckpt.get('gnn_layers', model_params['gnn_layers']),
        transformer_layers=ckpt.get('transformer_layers', model_params['transformer_layers']),
        nhead=ckpt.get('nhead', model_params['nhead'])
    )
    model.load_state_dict(ckpt['model_state_dict']); model.dataset_type = dataset_config['dataset_type_arg']; model.to(device); model.eval()
    print("BENO model loaded successfully."); return model

def load_lno_model(checkpoint_path, model_params, dataset_config, device):
    print(f"Attempting to load LNO model from: {checkpoint_path}")
    if not checkpoint_path or not os.path.exists(checkpoint_path): print(f"LNO checkpoint not found: {checkpoint_path}"); return None
    ckpt = torch.load(checkpoint_path, map_location=device)
    dummy_data_list = [pickle.load(open(dataset_config['path'], 'rb'))[0]]
    dummy_dataset = UniversalPDEDataset(dummy_data_list, dataset_type=dataset_config['dataset_type_arg'])
    total_bc_ctrl_dim = dummy_dataset.bc_state_dim + dummy_dataset.num_controls
    num_state_vars_lno = ckpt.get('num_state_vars', dataset_config['num_state_vars'])
    coord_dim_lno = ckpt.get('coord_dim', model_params.get('coord_dim', 2))
    D_embedding_lno = ckpt.get('D_embedding', model_params.get('D_embedding', 64)) # Define D_embedding before use
    model = LNO_LatentNeuralOperator(
        num_state_vars=num_state_vars_lno, coord_dim=coord_dim_lno, total_bc_ctrl_dim=total_bc_ctrl_dim,
        M_latent_points=ckpt.get('M_latent_points', model_params.get('M_latent_points', 64)),
        D_embedding=D_embedding_lno,
        L_transformer_blocks=ckpt.get('L_transformer_blocks', model_params.get('L_transformer_blocks', 3)),
        transformer_nhead=ckpt.get('transformer_nhead', model_params.get('transformer_nhead', 4)),
        transformer_dim_feedforward=ckpt.get('transformer_dim_feedforward', model_params.get('transformer_dim_feedforward', D_embedding_lno * 2)),
        projector_hidden_dims=ckpt.get('projector_hidden_dims', model_params.get('projector_hidden_dims', [D_embedding_lno, D_embedding_lno*2])),
        final_mlp_hidden_dims=ckpt.get('final_mlp_hidden_dims', model_params.get('final_mlp_hidden_dims', [D_embedding_lno*2, D_embedding_lno]))
    )
    model.load_state_dict(ckpt['model_state_dict']); model.nx = dataset_config['nx']; model.domain_L = dataset_config['L']
    model.state_keys = dataset_config['state_keys']; model.to(device); model.eval()
    print("LNO model loaded successfully."); return model

def load_lns_ae_model(ae_checkpoint_path, stepper_checkpoint_path, model_params, dataset_config, device):
    print(f"Attempting to load LNS-AE model from AE: {ae_checkpoint_path} and Stepper: {stepper_checkpoint_path}")
    if not ae_checkpoint_path or not os.path.exists(ae_checkpoint_path): print(f"LNS AE checkpoint not found: {ae_checkpoint_path}"); return None
    if not stepper_checkpoint_path or not os.path.exists(stepper_checkpoint_path): print(f"LNS Stepper checkpoint not found: {stepper_checkpoint_path}"); return None
    ae_ckpt = torch.load(ae_checkpoint_path, map_location=device); stepper_ckpt = torch.load(stepper_checkpoint_path, map_location=device)
    num_state_vars = dataset_config['num_state_vars']; target_nx_full = dataset_config['nx']
    ae_initial_width = ae_ckpt.get('ae_initial_width', model_params['ae_initial_width'])
    ae_downsample_blocks = ae_ckpt.get('ae_downsample_blocks', model_params['ae_downsample_blocks'])
    ae_latent_channels = ae_ckpt.get('ae_latent_channels', model_params['ae_latent_channels'])
    ae_final_latent_nx = target_nx_full // (2**ae_downsample_blocks)
    autoencoder = LNS_Autoencoder(
        num_state_vars=num_state_vars, target_nx_full=target_nx_full, ae_initial_width=ae_initial_width,
        ae_downsample_blocks=ae_downsample_blocks, ae_latent_channels=ae_latent_channels, final_latent_nx=ae_final_latent_nx
    )
    autoencoder.load_state_dict(ae_ckpt['model_state_dict']); autoencoder.to(device); autoencoder.eval()
    dummy_data_list = [pickle.load(open(dataset_config['path'], 'rb'))[0]]
    dummy_dataset = UniversalPDEDataset(dummy_data_list, dataset_type=dataset_config['dataset_type_arg'])
    bc_ctrl_input_dim = dummy_dataset.bc_state_dim + dummy_dataset.num_controls
    latent_stepper = LNS_LatentStepperNet(
        latent_dim_c=stepper_ckpt.get('latent_dim_c', ae_latent_channels),
        latent_dim_x=stepper_ckpt.get('latent_dim_x', ae_final_latent_nx),
        bc_ctrl_input_dim=bc_ctrl_input_dim,
        branch_hidden_dims=stepper_ckpt.get('branch_hidden_dims', model_params['branch_hidden_dims']),
        trunk_hidden_dims=stepper_ckpt.get('trunk_hidden_dims', model_params['trunk_hidden_dims']),
        combined_output_p=stepper_ckpt.get('combined_output_p', model_params['combined_output_p'])
    )
    latent_stepper.load_state_dict(stepper_ckpt['model_state_dict']); latent_stepper.to(device); latent_stepper.eval()
    print("LNS-AE model (AE and Stepper) loaded successfully."); return autoencoder, latent_stepper

def load_pod_dl_rom_model(checkpoint_path, model_params, dataset_config, device, base_checkpoint_path_ignored):
    print(f"Attempting to load POD-DL-ROM model from: {checkpoint_path}")
    if not checkpoint_path or not os.path.exists(checkpoint_path): print(f"POD-DL-ROM checkpoint not found: {checkpoint_path}"); return None
    ckpt = torch.load(checkpoint_path, map_location=device)
    N_pod = ckpt.get('N_pod', model_params['N_pod']); n_latent = ckpt.get('n_latent', model_params['n_latent'])
    num_state_vars = dataset_config['num_state_vars']; state_keys = dataset_config['state_keys']
    nx = dataset_config['nx']; NT_train = dataset_config['NT_train']
    V_N_dict = {}; basis_cache_pattern = MODEL_CONFIGS["POD_DL_ROM"]["basis_cache_pattern"]
    for skey in state_keys:
        basis_fname_segment = basis_cache_pattern.format(
            dataset_type_arg=dataset_config['dataset_type_arg'], state_key=skey, nx=nx, N_pod=N_pod, NT_train=NT_train
        )
        full_basis_path = basis_fname_segment # Assumes pattern is relative to script or absolute
        print(f"  Looking for POD basis for {skey} at: {full_basis_path}")
        if os.path.exists(full_basis_path): V_N_dict[skey] = torch.tensor(np.load(full_basis_path)).float().to(device)
        else: print(f"  ERROR: POD basis not found for {skey} at {full_basis_path}."); return None
    dummy_data_list = [pickle.load(open(dataset_config['path'], 'rb'))[0]]
    dummy_dataset = UniversalPDEDataset(dummy_data_list, dataset_type=dataset_config['dataset_type_arg'])
    dfnn_input_dim = 1 + dummy_dataset.bc_state_dim + dummy_dataset.num_controls
    model = POD_DL_ROM(
        V_N_dict=V_N_dict, n_latent=n_latent, dfnn_input_dim=dfnn_input_dim, N_pod=N_pod, num_state_vars=num_state_vars
    )
    model.load_state_dict(ckpt['model_state_dict']); model.to(device); model.eval()
    print("POD-DL-ROM model loaded successfully."); return model

MODEL_LOADER_MAP = {
    "BAROM": load_rom_full_model, "FNO": load_fno_model, "SPFNO": load_spfno_model,
    "BENO": load_beno_model, "LNO": load_lno_model, "LNS_AE": load_lns_ae_model,
    "POD_DL_ROM": load_pod_dl_rom_model,
}
@torch.no_grad()
def predict_fno_spfno_beno(model, model_name, initial_state_norm, bc_ctrl_seq_norm, num_steps, dataset_config):
    nx = dataset_config['nx']; num_vars = dataset_config['num_state_vars']
    pred_seq_norm = torch.zeros(num_steps, nx, num_vars, device=DEVICE)
    current_u_norm = initial_state_norm.clone()
    if num_steps > 0 : pred_seq_norm[0, :, :] = current_u_norm.squeeze(0)

    # Timing and Memory
    start_time = time.time()
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize() # Ensure previous ops are done
        torch.cuda.reset_peak_memory_stats(DEVICE) # Reset for this specific inference call

    for t in range(num_steps - 1):
        bc_ctrl_t_norm = bc_ctrl_seq_norm[:, t, :].unsqueeze(1)
        bc_ctrl_t_spatial_norm = bc_ctrl_t_norm.repeat(1, nx, 1)
        model_input_norm = torch.cat((current_u_norm, bc_ctrl_t_spatial_norm), dim=-1)
        
        if model_name == "BENO":
            bc_ctrl_for_beno = bc_ctrl_seq_norm[:, t, :] 
            u_next_norm = model(current_u_norm, bc_ctrl_for_beno)
        else: # FNO, SPFNO
            u_next_norm = model(model_input_norm)
        
        pred_seq_norm[t+1, :, :] = u_next_norm.squeeze(0)
        current_u_norm = u_next_norm

    if DEVICE.type == 'cuda':
        torch.cuda.synchronize() # Ensure model execution is complete
    end_time = time.time()
    inference_time_sec = end_time - start_time
    
    peak_memory_mb = 0
    if DEVICE.type == 'cuda':
        peak_memory_bytes = torch.cuda.max_memory_allocated(DEVICE)
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)

    output_dict = {}
    for i, skey in enumerate(dataset_config['state_keys']): 
        output_dict[skey] = pred_seq_norm[:, :, i]
    
    return output_dict, inference_time_sec, peak_memory_mb

# Apply similar changes to:
# predict_rom_full(model, initial_state_dict_norm, bc_ctrl_seq_norm, num_steps, dataset_config)
# predict_lno(model, initial_state_norm, bc_ctrl_seq_norm, num_steps, dataset_config)
# predict_lns_ae(autoencoder, latent_stepper, initial_state_norm, bc_ctrl_seq_norm, num_steps, dataset_config, model_params_lns)
# predict_pod_dl_rom(model, bc_ctrl_seq_norm, num_steps, dataset_config, T_train_ref)

# Example for predict_rom_full:
@torch.no_grad()
def predict_rom_full(model, initial_state_dict_norm, bc_ctrl_seq_norm, num_steps, dataset_config):
    a0_dict = {}; batch_size = list(initial_state_dict_norm.values())[0].shape[0]
    BC_ctrl_t0_norm = bc_ctrl_seq_norm[:, 0, :]; U_B0_lifted_norm = model.lifting(BC_ctrl_t0_norm)
    for i, key in enumerate(model.state_keys):
        U0_norm_var = initial_state_dict_norm[key];
        if U0_norm_var.dim() == 2: U0_norm_var = U0_norm_var.unsqueeze(-1)
        U_B0_norm_var = U_B0_lifted_norm[:, i, :].unsqueeze(-1)
        U0_star_norm_var = U0_norm_var - U_B0_norm_var
        Phi_var = model.get_basis(key).to(DEVICE)
        Phi_T_var = Phi_var.transpose(0,1).unsqueeze(0).expand(batch_size, -1, -1)
        a0_norm_var = torch.bmm(Phi_T_var, U0_star_norm_var).squeeze(-1)
        a0_dict[key] = a0_norm_var

    start_time = time.time()
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(DEVICE)

    pred_seq_norm_dict_of_lists, _ = model(a0_dict, bc_ctrl_seq_norm[:, :num_steps, :], T=num_steps)
    
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    inference_time_sec = end_time - start_time
    
    peak_memory_mb = 0
    if DEVICE.type == 'cuda':
        peak_memory_bytes = torch.cuda.max_memory_allocated(DEVICE)
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)

    pred_seq_norm_dict_of_tensors = {}
    for key_model_state in model.state_keys: # Ensure you iterate over model.state_keys as in original
        # Ensure the key from pred_seq_norm_dict_of_lists is used correctly
        if key_model_state in pred_seq_norm_dict_of_lists:
             # Handle cases where a list might be empty or not correctly formed
            if pred_seq_norm_dict_of_lists[key_model_state] and isinstance(pred_seq_norm_dict_of_lists[key_model_state], list) and len(pred_seq_norm_dict_of_lists[key_model_state]) > 0:
                # Original logic: torch.cat(pred_seq_norm_dict_of_lists[key_model_state], dim=0).squeeze(-1)
                # This assumes list elements are tensors that can be concatenated along dim=0
                # and then squeezed. If pred_seq_norm_dict_of_lists[key_model_state] is already
                # the final tensor sequence (e.g., [T, Nx, 1] or [T, Nx]), adjust accordingly.
                # For BAROM, it seems it's a list of tensors, one per time step.
                concatenated_tensor = torch.cat(pred_seq_norm_dict_of_lists[key_model_state], dim=0) # dim=0 for time steps
                if concatenated_tensor.dim() > 2 and concatenated_tensor.shape[-1] == 1: # If shape is [T, Nx, 1]
                    pred_seq_norm_dict_of_tensors[key_model_state] = concatenated_tensor.squeeze(-1)
                else: # If shape is already [T, Nx] or other
                    pred_seq_norm_dict_of_tensors[key_model_state] = concatenated_tensor

            else: # Handle empty list or unexpected structure
                print(f"Warning: pred_seq_norm_dict_of_lists for key '{key_model_state}' is empty or not a list of tensors in predict_rom_full.")
                # Fallback: create an empty tensor or handle as error
                # Using an empty tensor of appropriate shape if possible, or None
                # This depends on how downstream code handles missing keys or empty tensors.
                # For now, let's assume it should be [num_steps, nx]
                nx_val = dataset_config.get('nx', 1) # Get nx from dataset_config
                pred_seq_norm_dict_of_tensors[key_model_state] = torch.empty((num_steps, nx_val), device=DEVICE)


        else:
            print(f"Warning: Key '{key_model_state}' not found in pred_seq_norm_dict_of_lists from BAROM model output.")
            nx_val = dataset_config.get('nx', 1)
            pred_seq_norm_dict_of_tensors[key_model_state] = torch.empty((num_steps, nx_val), device=DEVICE)


    return pred_seq_norm_dict_of_tensors, inference_time_sec, peak_memory_mb

# --- For LNS_AE ---
@torch.no_grad()
def predict_lns_ae(autoencoder, latent_stepper, initial_state_norm, bc_ctrl_seq_norm, num_steps, dataset_config, model_params_lns):
    nx = dataset_config['nx']; num_vars = dataset_config['num_state_vars']
    pred_seq_phys_norm = torch.zeros(num_steps, nx, num_vars, device=DEVICE)
    u_t0_phys_norm_enc_in = initial_state_norm.permute(0,2,1) # [B, C, Nx]

    start_time = time.time()
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(DEVICE)

    z_current_latent = autoencoder.encoder(u_t0_phys_norm_enc_in)
    if num_steps > 0: pred_seq_phys_norm[0, :, :] = initial_state_norm.squeeze(0) # [Nx, C]

    for t in range(num_steps - 1):
        bc_ctrl_input_latent_step = bc_ctrl_seq_norm[:, t+1, :] # Use t+1 for BC to predict state at t+1
        z_next_pred_latent = latent_stepper(z_current_latent, bc_ctrl_input_latent_step)
        u_next_pred_physical_norm_enc_out = autoencoder.decoder(z_next_pred_latent) # [B, C, Nx]
        u_next_pred_physical_norm = u_next_pred_physical_norm_enc_out.permute(0,2,1) # [B, Nx, C]
        pred_seq_phys_norm[t+1, :, :] = u_next_pred_physical_norm.squeeze(0) # [Nx, C]
        z_current_latent = z_next_pred_latent
    
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    inference_time_sec = end_time - start_time
    
    peak_memory_mb = 0
    if DEVICE.type == 'cuda':
        peak_memory_bytes = torch.cuda.max_memory_allocated(DEVICE)
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)

    output_dict = {}
    for i, skey in enumerate(dataset_config['state_keys']): 
        output_dict[skey] = pred_seq_phys_norm[:, :, i] # [T, Nx]
    
    return output_dict, inference_time_sec, peak_memory_mb

# --- For LNO ---
@torch.no_grad()
def predict_lno(model, initial_state_norm, bc_ctrl_seq_norm, num_steps, dataset_config):
    nx = dataset_config['nx']; L_domain = dataset_config['L']; num_vars = dataset_config['num_state_vars']
    pred_seq_norm = torch.zeros(num_steps, nx, num_vars, device=DEVICE) # [T, Nx, C]
    current_u_norm = initial_state_norm.clone() # [B, Nx, C]
    if num_steps > 0: pred_seq_norm[0, :, :] = current_u_norm.squeeze(0) # [Nx, C]

    x_coords_spatial = torch.linspace(0, L_domain, nx, device=DEVICE)
    t_pseudo_input = torch.zeros_like(x_coords_spatial) # For t_k
    t_pseudo_output = torch.ones_like(x_coords_spatial) # For t_{k+1} (relative step of 1)
    
    pos_in_step = torch.stack([x_coords_spatial, t_pseudo_input], dim=-1).unsqueeze(0) # [1, Nx, 2]
    pos_out_step = torch.stack([x_coords_spatial, t_pseudo_output], dim=-1).unsqueeze(0) # [1, Nx, 2]

    start_time = time.time()
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(DEVICE)

    for t in range(num_steps - 1):
        bc_ctrl_t_norm = bc_ctrl_seq_norm[:, t, :] # BCs at current time t
        u_next_norm = model(pos_in_step, current_u_norm, pos_out_step, bc_ctrl_t_norm) # Predicts state at t+1
        pred_seq_norm[t+1, :, :] = u_next_norm.squeeze(0) # [Nx, C]
        current_u_norm = u_next_norm
        
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    inference_time_sec = end_time - start_time
    
    peak_memory_mb = 0
    if DEVICE.type == 'cuda':
        peak_memory_bytes = torch.cuda.max_memory_allocated(DEVICE)
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)

    output_dict = {}
    for i, skey in enumerate(dataset_config['state_keys']): 
        output_dict[skey] = pred_seq_norm[:, :, i] # [T, Nx]
    
    return output_dict, inference_time_sec, peak_memory_mb

# --- For POD_DL_ROM ---
@torch.no_grad()
def predict_pod_dl_rom(model, bc_ctrl_seq_norm, num_steps, dataset_config, T_train_ref):
    nx = dataset_config['nx']
    pred_seq_norm_dict = {key: torch.zeros(num_steps, nx, device=DEVICE) for key in model.state_keys}

    start_time = time.time()
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(DEVICE)

    for t_eval_idx in range(num_steps):
        # time_norm_for_dfnn is relative to the number of steps being evaluated *now*
        # not relative to T_train_ref directly in this loop, that's more for training context.
        # Here, it's just a normalized time input for the DFNN over the prediction horizon.
        time_norm_val = t_eval_idx / (num_steps - 1.0 if num_steps > 1 else 1.0)
        time_norm_for_dfnn = torch.tensor([time_norm_val], device=DEVICE).float().unsqueeze(0) # [1, 1]
        
        BC_Ctrl_t_actual = bc_ctrl_seq_norm[:, t_eval_idx, :] # [B, bc_ctrl_dim]
        dfnn_input_val = torch.cat((time_norm_for_dfnn, BC_Ctrl_t_actual), dim=-1) # [B, 1 + bc_ctrl_dim]
        
        u_h_predicted_dict_t, _ = model(dfnn_input_val) # model.forward returns dict of [B, Nx]
        
        for key_model in model.state_keys: 
            pred_seq_norm_dict[key_model][t_eval_idx, :] = u_h_predicted_dict_t[key_model].squeeze() # Squeeze batch if B=1

    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    inference_time_sec = end_time - start_time
    
    peak_memory_mb = 0
    if DEVICE.type == 'cuda':
        peak_memory_bytes = torch.cuda.max_memory_allocated(DEVICE)
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)
        
    return pred_seq_norm_dict, inference_time_sec, peak_memory_mb

# --- Metric Calculation & Visualization ---
def calculate_metrics(pred_denorm, gt_denorm):
    if pred_denorm.shape != gt_denorm.shape:
        min_len = min(pred_denorm.shape[0], gt_denorm.shape[0])
        pred_denorm = pred_denorm[:min_len]; gt_denorm = gt_denorm[:min_len]
        if pred_denorm.shape != gt_denorm.shape: return {'mse': float('nan'), 'rmse': float('nan'), 'rel_err': float('nan'), 'max_err': float('nan')}
    mse = np.mean((pred_denorm - gt_denorm)**2); rmse = np.sqrt(mse)
    rel_err = np.linalg.norm(pred_denorm - gt_denorm, 'fro') / (np.linalg.norm(gt_denorm, 'fro') + 1e-10)
    max_err = np.max(np.abs(pred_denorm - gt_denorm)) if pred_denorm.size > 0 else 0.0
    return {'mse': mse, 'rmse': rmse, 'rel_err': rel_err, 'max_err': max_err}

def plot_comparison(gt_seq_denorm_dict, predictions_denorm_dict, dataset_name, sample_idx_str,
                    state_keys_to_plot, L_domain, T_horizon, save_path_base):
    num_models = len(predictions_denorm_dict) + 1; num_vars = len(state_keys_to_plot)
    fig, axs = plt.subplots(num_vars, num_models, figsize=(5 * num_models, 4 * num_vars), squeeze=False)
    fig.suptitle(f"Benchmark: {dataset_name} (Sample {sample_idx_str}) @ T={T_horizon:.2f}", fontsize=16)

    plot_names = ["Ground Truth"] + list(predictions_denorm_dict.keys())
    for i, skey in enumerate(state_keys_to_plot):
        # —— 1. 收集这一行所有有效数据 —— 
        all_series = []
        gt_data = gt_seq_denorm_dict[skey]                     # shape [T, nx]
        if gt_data.size>0: all_series.append(gt_data)
        for preds in predictions_denorm_dict.values():
            arr = preds[skey]
            if arr is not None and arr.size>0:
                all_series.append(arr)

        # —— 2. 计算全局 vmin/vmax（示例用 2%–98% 百分位剪裁） —— 
        flat = np.concatenate([a.ravel() for a in all_series])
        vmin, vmax = np.percentile(flat, [2, 98])
        # 如果想要严格 min/max，可以直接：
        # vmin, vmax = flat.min(), flat.max()

        # —— 3. 用同一个 SymLogNorm —— 
        norm = SymLogNorm(
            linthresh=1e-3,      # 根据你的数据调整阈值
            linscale=1.0,
            vmin=max(vmin, 1e-8),# 避免 ≤0
            vmax=vmax
        )

        # —— 4. 画这一行的每个子图，都传相同的 norm —— 
        for j, name in enumerate(plot_names):
            ax = axs[i, j]
            if name == "Ground Truth":
                data = gt_data
            else:
                data = predictions_denorm_dict[name][skey]

            if data is None or data.size == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
            else:
                im = ax.imshow(
                    data,
                    aspect='auto', origin='lower',
                    extent=[0, L_domain, 0, T_horizon],
                    cmap='RdBu_r',
                    norm=norm
                )
                # 只在最右侧一列画 colorbar
                if j == len(plot_names) - 1:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax.set_title(f"{name}\n({skey})")
            ax.set_xlabel("x")
            if j == 0:
                ax.set_ylabel(f"{skey}\nt (physical)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    final_save_path = os.path.join(save_path_base, f"comparison_{dataset_name}_sample{sample_idx_str}_T{T_horizon:.2f}.png")
    os.makedirs(save_path_base, exist_ok=True) # Ensure directory exists
    plt.savefig(final_save_path); print(f"Saved comparison plot to {final_save_path}"); plt.close(fig)

# --- Main Benchmarking Logic ---
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import time
import pickle
import argparse
from torch.utils.data import Dataset, DataLoader
import glob # For finding files with wildcards
import traceback
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm

# Assuming the rest of your imports and class/function definitions (UniversalPDEDataset, model definitions, 
# MODEL_CONFIGS, DATASET_CONFIGS, load_data, get_checkpoint_path, model loaders, 
# prediction functions, calculate_metrics, plot_comparison) are present above this main function.

# --- Utility Functions (ensure count_parameters is defined here or imported) ---
def count_parameters(model):
    """Counts the number of trainable parameters in a PyTorch model."""
    if model is None:
        return 0
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- Main Benchmarking Logic ---
def main(datasets_to_benchmark, models_to_benchmark):
    print(f"Device: {DEVICE}")
    print(f"Benchmarking datasets: {datasets_to_benchmark}")
    print(f"Benchmarking models: {models_to_benchmark}")
    
    # Updated structure for overall_aggregated_metrics
    overall_aggregated_metrics = {} 

    # Initialize overall_aggregated_metrics for all datasets and models first
    # This ensures the structure exists even if a model fails to load for a specific dataset
    for dataset_name_key_init in DATASET_CONFIGS.keys(): # Iterate over all possible datasets
        if dataset_name_key_init not in datasets_to_benchmark: # Only for those being benchmarked
            continue
        ds_config_init = DATASET_CONFIGS[dataset_name_key_init]
        overall_aggregated_metrics[dataset_name_key_init] = {
            model_name_init: {
                'param_count': 0,
                'inference_times_ms': [],
                'gpu_memory_mb': [],
                'state_metrics': {
                    skey: {'mse': [], 'rel_err': [], 'rmse': [], 'max_err': []} 
                    for skey in ds_config_init['state_keys']
                }
            }
            # Ensure all models, even those not in models_to_benchmark for this run, have an entry
            # This is useful if the script is run with different model subsets later for the same output analysis.
            # However, for clarity of this specific run, we will filter later when printing.
            for model_name_init in MODEL_CONFIGS.keys() 
        }


    for dataset_name_key in datasets_to_benchmark:
        print(f"\n===== Benchmarking on Dataset: {dataset_name_key.upper()} =====")
        val_data_list, val_loader, ds_config = load_data(dataset_name_key)
        if val_loader is None: 
            print(f"Could not load data for {dataset_name_key}. Skipping.")
            continue
        
        # Ensure this dataset's entry exists and is correctly structured
        if dataset_name_key not in overall_aggregated_metrics: # Should be pre-initialized
            overall_aggregated_metrics[dataset_name_key] = {
                model_name: {
                    'param_count': 0, 'inference_times_ms': [], 'gpu_memory_mb': [],
                    'state_metrics': {skey: {'mse': [], 'rel_err': [], 'rmse': [], 'max_err': []} for skey in ds_config['state_keys']}
                } for model_name in MODEL_CONFIGS.keys() # Initialize for all possible models
            }
        # Ensure all models to benchmark have their entries for this dataset
        for model_name in models_to_benchmark:
            if model_name not in overall_aggregated_metrics[dataset_name_key]:
                overall_aggregated_metrics[dataset_name_key][model_name] = {
                    'param_count': 0, 'inference_times_ms': [], 'gpu_memory_mb': [],
                    'state_metrics': {skey: {'mse': [], 'rel_err': [], 'rmse': [], 'max_err': []} for skey in ds_config['state_keys']}
                }
        
        num_val_samples = len(val_data_list)
        vis_sample_count = min(MAX_VISUALIZATION_SAMPLES, num_val_samples)
        visualization_indices = list(range(vis_sample_count)) # First N samples for visualization
        gt_data_for_visualization = {} # {sample_idx: gt_full_seq_denorm_dict_for_that_sample}

        loaded_models = {} # Will store {'model_name': {'model': instance_or_tuple, 'params': count}}
        for model_name in models_to_benchmark:
            print(f"  -- Pre-loading Model: {model_name} for dataset {dataset_name_key} --")
            if model_name not in MODEL_CONFIGS or model_name not in MODEL_LOADER_MAP:
                print(f"  Skipping model {model_name}: Not configured or loader not found.")
                loaded_models[model_name] = {'model': None, 'params': 0}
                if model_name in overall_aggregated_metrics[dataset_name_key]:
                    overall_aggregated_metrics[dataset_name_key][model_name]['param_count'] = 0
                continue
            
            model_config = MODEL_CONFIGS[model_name]
            loader_fn = MODEL_LOADER_MAP[model_name]
            model_instance_or_tuple = None
            param_count = 0

            try:
                if model_name == "LNS_AE":
                    ae_ckpt_path = get_checkpoint_path(model_config["ae_checkpoint_pattern"], ds_config['dataset_type_arg'], ds_config['T_train'], model_config["params"], BASE_CHECKPOINT_PATH)
                    stepper_ckpt_path = get_checkpoint_path(model_config["stepper_checkpoint_pattern"], ds_config['dataset_type_arg'], ds_config['T_train'], model_config["params"], BASE_CHECKPOINT_PATH)
                    if not ae_ckpt_path or not stepper_ckpt_path:
                        model_instance_or_tuple = None
                    else:
                        model_instance_or_tuple = loader_fn(ae_ckpt_path, stepper_ckpt_path, model_config["params"], ds_config, DEVICE)
                    if model_instance_or_tuple and model_instance_or_tuple[0] and model_instance_or_tuple[1]:
                        ae_instance, stepper_instance = model_instance_or_tuple
                        param_count = count_parameters(ae_instance) + count_parameters(stepper_instance)
                    else:
                        model_instance_or_tuple = None 
                elif model_name == "POD_DL_ROM":
                    checkpoint_path_resolved = get_checkpoint_path(model_config["checkpoint_pattern"], ds_config['dataset_type_arg'], ds_config['T_train'], model_config["params"], BASE_CHECKPOINT_PATH)
                    if not checkpoint_path_resolved:
                        model_instance_or_tuple = None
                    else:
                        model_instance_or_tuple = loader_fn(checkpoint_path_resolved, model_config["params"], ds_config, DEVICE, BASE_CHECKPOINT_PATH)
                    if model_instance_or_tuple:
                        param_count = count_parameters(model_instance_or_tuple)
                else:
                    checkpoint_path_resolved = get_checkpoint_path(model_config["checkpoint_pattern"], ds_config['dataset_type_arg'], ds_config['T_train'], model_config["params"], BASE_CHECKPOINT_PATH)
                    if not checkpoint_path_resolved:
                        model_instance_or_tuple = None
                    else:
                        model_instance_or_tuple = loader_fn(checkpoint_path_resolved, model_config["params"], ds_config, DEVICE)
                    if model_instance_or_tuple:
                        param_count = count_parameters(model_instance_or_tuple)
            except Exception as e:
                print(f"  ERROR during model loading for {model_name}: {e}")
                traceback.print_exc()
                model_instance_or_tuple = None
                param_count = 0
            
            loaded_models[model_name] = {'model': model_instance_or_tuple, 'params': param_count}
            if model_name in overall_aggregated_metrics[dataset_name_key]:
                 overall_aggregated_metrics[dataset_name_key][model_name]['param_count'] = param_count
            else: 
                 overall_aggregated_metrics[dataset_name_key][model_name] = { # Should not be hit if pre-init is correct
                    'param_count': param_count, 'inference_times_ms': [], 'gpu_memory_mb': [],
                    'state_metrics': {skey: {'mse': [], 'rel_err': [], 'rmse': [], 'max_err': []} for skey in ds_config['state_keys']}
                 }

            if model_instance_or_tuple is None:
                print(f"  Failed to pre-load model {model_name}. Will be skipped for this dataset.")
            elif model_name == "LNS_AE" and (model_instance_or_tuple is None or model_instance_or_tuple[0] is None or model_instance_or_tuple[1] is None):
                 print(f"  Failed to pre-load LNS_AE model components. Will be skipped for this dataset.")
                 loaded_models[model_name] = {'model': None, 'params': 0} 
                 if model_name in overall_aggregated_metrics[dataset_name_key]:
                    overall_aggregated_metrics[dataset_name_key][model_name]['param_count'] = 0

        predictions_for_visualization_all_models = {vis_idx: {} for vis_idx in visualization_indices}

        for val_idx, (sample_state_list_norm, sample_bc_ctrl_seq_norm, sample_norm_factors) in enumerate(val_loader):
            if (val_idx+1)%10 == 0:
                print(f"  Processing validation sample {val_idx+1}/{num_val_samples} for dataset {dataset_name_key}...")
            
            initial_state_norm_dict = {}
            gt_full_seq_denorm_dict_current_sample = {}

            for idx_skey, skey in enumerate(ds_config['state_keys']):
                initial_state_norm_dict[skey] = sample_state_list_norm[idx_skey][:, 0, :].to(DEVICE) # [B, Nx]
                gt_seq_norm_var = sample_state_list_norm[idx_skey].squeeze(0).to(DEVICE) # [T, Nx]
                mean_val = sample_norm_factors[f'{skey}_mean'].item()
                std_val = sample_norm_factors[f'{skey}_std'].item()
                gt_full_seq_denorm_dict_current_sample[skey] = gt_seq_norm_var.cpu().numpy() * std_val + mean_val
            
            if val_idx in visualization_indices:
                gt_data_for_visualization[val_idx] = gt_full_seq_denorm_dict_current_sample

            T_horizon_benchmark = 2.0 # ds_config['T_file'] # Benchmark over the full horizon available in the file
            num_benchmark_steps = int((T_horizon_benchmark / ds_config['T_file']) * (ds_config['NT_file'] - 1)) + 1
            num_benchmark_steps = min(num_benchmark_steps, ds_config['NT_file']) # Ensure it doesn't exceed file NT

            gt_benchmark_horizon_denorm_dict = {}
            for skey in ds_config['state_keys']:
                gt_benchmark_horizon_denorm_dict[skey] = gt_full_seq_denorm_dict_current_sample[skey][:num_benchmark_steps, :]
            
            sample_bc_ctrl_seq_norm_benchmark = sample_bc_ctrl_seq_norm[:, :num_benchmark_steps, :].to(DEVICE)
            
            initial_state_single_tensor_norm_list = []
            for skey in ds_config['state_keys']:
                # initial_state_norm_dict[skey] is [B, Nx], need [B, Nx, 1] for cat
                initial_state_single_tensor_norm_list.append(initial_state_norm_dict[skey].unsqueeze(-1)) 
            initial_state_single_tensor_norm = torch.cat(initial_state_single_tensor_norm_list, dim=-1).to(DEVICE) # [B, Nx, NumStateVars]

            for model_name in models_to_benchmark:
                model_data = loaded_models.get(model_name)
                if not model_data or model_data['model'] is None: 
                    # print(f"  Skipping {model_name} for sample {val_idx} as it was not loaded.") # Already printed during loading
                    continue

                current_model_instance_or_tuple = model_data['model']
                ae_instance, stepper_instance = None, None
                current_model_instance = current_model_instance_or_tuple
                if model_name == "LNS_AE":
                    if not isinstance(current_model_instance_or_tuple, tuple) or len(current_model_instance_or_tuple) != 2:
                        # print(f"  LNS_AE loaded incorrectly for sample {val_idx}. Skipping model for this sample.")
                        continue
                    ae_instance, stepper_instance = current_model_instance_or_tuple
                    if ae_instance is None or stepper_instance is None:
                        # print(f"  LNS_AE components not loaded for sample {val_idx}. Skipping model for this sample.")
                        continue
                
                model_config_params = MODEL_CONFIGS[model_name]["params"] 
                pred_seq_norm_dict_model = {}
                inference_time_sec = -1.0
                peak_memory_mb = -1.0

                try:
                    if model_name == "BAROM":
                        initial_state_rom = {k: v.to(DEVICE) for k,v in initial_state_norm_dict.items()}
                        pred_seq_norm_dict_model, inference_time_sec, peak_memory_mb = predict_rom_full(current_model_instance, initial_state_rom, sample_bc_ctrl_seq_norm_benchmark, num_benchmark_steps, ds_config)
                    elif model_name in ["FNO", "SPFNO", "BENO"]:
                        pred_seq_norm_dict_model, inference_time_sec, peak_memory_mb = predict_fno_spfno_beno(current_model_instance, model_name, initial_state_single_tensor_norm, sample_bc_ctrl_seq_norm_benchmark, num_benchmark_steps, ds_config)
                    elif model_name == "LNO":
                        pred_seq_norm_dict_model, inference_time_sec, peak_memory_mb = predict_lno(current_model_instance, initial_state_single_tensor_norm, sample_bc_ctrl_seq_norm_benchmark, num_benchmark_steps, ds_config)
                    elif model_name == "LNS_AE":
                        pred_seq_norm_dict_model, inference_time_sec, peak_memory_mb = predict_lns_ae(ae_instance, stepper_instance, initial_state_single_tensor_norm, sample_bc_ctrl_seq_norm_benchmark, num_benchmark_steps, ds_config, model_config_params)
                    elif model_name == "POD_DL_ROM":
                        T_train_ref_pod = ds_config['T_train'] # This T_train_ref might be for model's internal logic if needed
                        pred_seq_norm_dict_model, inference_time_sec, peak_memory_mb = predict_pod_dl_rom(current_model_instance, sample_bc_ctrl_seq_norm_benchmark, num_benchmark_steps, ds_config, T_train_ref_pod)
                except Exception as e:
                    print(f"  ERROR during prediction for {model_name} on sample {val_idx} of {dataset_name_key}: {e}")
                    traceback.print_exc()
                    continue 

                if model_name in overall_aggregated_metrics[dataset_name_key]:
                    overall_aggregated_metrics[dataset_name_key][model_name]['inference_times_ms'].append(inference_time_sec * 1000) 
                    overall_aggregated_metrics[dataset_name_key][model_name]['gpu_memory_mb'].append(peak_memory_mb)
                else: 
                    print(f"Warning: {model_name} not found in overall_aggregated_metrics for {dataset_name_key} during metric storage.")

                model_preds_denorm_this_sample = {}
                for skey in ds_config['state_keys']:
                    if skey not in pred_seq_norm_dict_model or pred_seq_norm_dict_model[skey] is None: 
                        print(f"Warning: Predictions for state key '{skey}' not found or is None for model {model_name} on sample {val_idx}.")
                        continue 
                    
                    pred_norm_var_np = pred_seq_norm_dict_model[skey].cpu().numpy()
                    gt_denorm_var = gt_benchmark_horizon_denorm_dict[skey]
                    
                    if pred_norm_var_np.shape[0] != gt_denorm_var.shape[0]:
                        min_t_len = min(pred_norm_var_np.shape[0], gt_denorm_var.shape[0])
                        print(f"Warning: Truncating prediction/GT for metric calculation due to length mismatch for {skey} in {model_name}. Pred: {pred_norm_var_np.shape[0]}, GT: {gt_denorm_var.shape[0]}. Using: {min_t_len}")
                        pred_norm_var_np_aligned = pred_norm_var_np[:min_t_len]
                        gt_denorm_var_aligned = gt_denorm_var[:min_t_len]
                    else:
                        pred_norm_var_np_aligned = pred_norm_var_np
                        gt_denorm_var_aligned = gt_denorm_var

                    if pred_norm_var_np_aligned.size == 0 or gt_denorm_var_aligned.size == 0:
                        print(f"Warning: Empty aligned prediction or GT for {skey} in {model_name}. Skipping metrics.")
                        continue

                    mean_val = sample_norm_factors[f'{skey}_mean'].item(); std_val = sample_norm_factors[f'{skey}_std'].item()
                    pred_denorm_var = pred_norm_var_np_aligned * std_val + mean_val
                    model_preds_denorm_this_sample[skey] = pred_denorm_var
                    
                    metrics = calculate_metrics(pred_denorm_var, gt_denorm_var_aligned) 
                    if model_name in overall_aggregated_metrics[dataset_name_key] and \
                       skey in overall_aggregated_metrics[dataset_name_key][model_name]['state_metrics']:
                        for metric_name_key, metric_val in metrics.items():
                            overall_aggregated_metrics[dataset_name_key][model_name]['state_metrics'][skey][metric_name_key].append(metric_val)
                
                if val_idx in visualization_indices:
                    predictions_for_visualization_all_models[val_idx][model_name] = model_preds_denorm_this_sample
            torch.cuda.empty_cache()

        # After processing all samples for the current dataset, generate visualizations
        print(f"\n  Generating visualizations for {dataset_name_key}...")
        for vis_idx in visualization_indices:
            if vis_idx not in gt_data_for_visualization:
                print(f"    Skipping visualization for sample index {vis_idx}, GT data not found.")
                continue
            
            current_gt_denorm = gt_data_for_visualization[vis_idx]
            # Slice GT to benchmark horizon for plotting consistency
            current_gt_denorm_sliced = {
                skey_plot: data_plot[:num_benchmark_steps, :] for skey_plot, data_plot in current_gt_denorm.items()
            }

            current_preds_for_plot = predictions_for_visualization_all_models.get(vis_idx, {})
            if not current_preds_for_plot:
                print(f"    Skipping visualization for sample index {vis_idx}, no predictions stored.")
                continue

            plot_comparison(current_gt_denorm_sliced, current_preds_for_plot,
                            dataset_name_key, f"{vis_idx}", ds_config['state_keys'],
                            ds_config['L'], T_horizon_benchmark, 
                            os.path.join(BASE_RESULTS_PATH, dataset_name_key)) 

    # --- Print Final Summary of All Aggregated Results ---
    print("\n\n===== BENCHMARKING COMPLETE - AGGREGATED METRICS =====")
    for dataset_name_key, model_data_agg_map in overall_aggregated_metrics.items():
        if not any(models_to_benchmark_for_ds in model_data_agg_map for models_to_benchmark_for_ds in models_to_benchmark):
            continue 

        print(f"\n--- Aggregated Results for Dataset: {dataset_name_key.upper()} ---")
        
        filtered_model_data_agg = {m_name: m_data for m_name, m_data in model_data_agg_map.items() if m_name in models_to_benchmark}

        for model_name, agg_data in filtered_model_data_agg.items():
            print(f"  Model: {model_name}")
            if not agg_data or not agg_data.get('inference_times_ms'): 
                print("    No performance metrics recorded (model might have failed loading or all samples failed).")
                param_c = agg_data.get('param_count', 0)
                print(f"    Parameters: {param_c:,}")
                has_state_metrics = False
                if 'state_metrics' in agg_data:
                    for skey_metrics in agg_data['state_metrics'].values():
                        if skey_metrics.get('mse'):
                            has_state_metrics = True
                            break
                if not has_state_metrics and not agg_data.get('inference_times_ms'):
                     print("    No state-based metrics recorded.")
            else: 
                param_count = agg_data.get('param_count', 0)
                avg_inf_time_ms = np.mean(agg_data['inference_times_ms']) if agg_data['inference_times_ms'] else float('nan')
                avg_gpu_mem_mb = np.mean(agg_data['gpu_memory_mb']) if agg_data['gpu_memory_mb'] else float('nan')
                max_gpu_mem_mb = np.max(agg_data['gpu_memory_mb']) if agg_data['gpu_memory_mb'] else float('nan')
                
                print(f"    Parameters: {param_count:,}")
                print(f"    Avg Inference Time: {avg_inf_time_ms:.2f} ms/sample")
                if DEVICE.type == 'cuda':
                    print(f"    Avg Peak GPU Memory: {avg_gpu_mem_mb:.2f} MB/sample")
                    print(f"    Max Peak GPU Memory: {max_gpu_mem_mb:.2f} MB (across samples)")
                else:
                    print(f"    GPU Memory: N/A (running on CPU)")
            
            if 'state_metrics' in agg_data:
                for skey, metrics_lists in agg_data['state_metrics'].items():
                    if not metrics_lists['mse']: 
                        if not any(metrics_lists.values()):
                             print(f"    {skey}: No state-based metrics recorded.")
                             continue
                    
                    num_samples_for_skey = len(metrics_lists['mse'])
                    if num_samples_for_skey == 0 and not agg_data.get('inference_times_ms'): 
                        print(f"    {skey}: No state-based metrics recorded (0 samples).")
                        continue
                    elif num_samples_for_skey == 0 and agg_data.get('inference_times_ms'): 
                         print(f"    {skey}: State-based metrics missing for all samples (though model ran).")
                         continue

                    avg_mse = np.mean([m for m in metrics_lists['mse'] if not np.isnan(m)]) if any(not np.isnan(m) for m in metrics_lists['mse']) else float('nan')
                    avg_rmse = np.mean([m for m in metrics_lists['rmse'] if not np.isnan(m)]) if any(not np.isnan(m) for m in metrics_lists['rmse']) else float('nan')
                    avg_rel_err = np.mean([m for m in metrics_lists['rel_err'] if not np.isnan(m)]) if any(not np.isnan(m) for m in metrics_lists['rel_err']) else float('nan')
                    avg_max_err = np.mean([m for m in metrics_lists['max_err'] if not np.isnan(m)]) if any(not np.isnan(m) for m in metrics_lists['max_err']) else float('nan')
                    
                    print(f"    {skey}: Avg MSE={avg_mse:.4e}, Avg RMSE={avg_rmse:.4e}, Avg RelErr={avg_rel_err:.4e}, Avg MaxErr={avg_max_err:.4e} (from {num_samples_for_skey} samples)")
            elif not agg_data.get('inference_times_ms'): 
                print("    No state-based metrics structure found.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark PDE Solution Models.")
    parser.add_argument('--datasets', nargs='+', default=['heat_nonlinear_feedback_gain', 'reaction_diffusion_neumann_feedback','convdiff'], 
                        choices=list(DATASET_CONFIGS.keys()), help='Which datasets to benchmark.')
    parser.add_argument('--models', nargs='+', default=['BAROM', 'SPFNO','BENO','LNS_AE','LNO','POD_DL_ROM'], 
                        choices=list(MODEL_CONFIGS.keys()), help='Which models to benchmark.')

    args = parser.parse_args()
    main(args.datasets, args.models)


def count_parameters(model):
    """Counts the number of trainable parameters in a PyTorch model."""
    if model is None:
        return 0
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark PDE Solution Models.")
    parser.add_argument('--datasets', nargs='+', default=['heat_nonlinear_feedback_gain', 'reaction_diffusion_neumann_feedback','convdiff'], # Example default
                        choices=list(DATASET_CONFIGS.keys()), help='Which datasets to benchmark.')
    parser.add_argument('--models', nargs='+', default=['BAROM',  'SPFNO','BENO','LNS_AE','LNO','POD_DL_ROM'], # Example default 'FNO',
                        choices=list(MODEL_CONFIGS.keys()), help='Which models to benchmark.')

    args = parser.parse_args()
    main(args.datasets, args.models)
