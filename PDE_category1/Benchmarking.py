# =============================================================================
# PDE Model Benchmarking Script
# =============================================================================
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import matplotlib.pyplot as plt
import random
import time
import pickle
import argparse
import importlib.util # To load models from user-provided files
from matplotlib.colors import LogNorm, SymLogNorm # For advanced color scaling

# --- Utility: Fixed Random Seed ---
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# =============================================================================
# 1. Load Model Definitions Dynamically
# =============================================================================
def load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

MODEL_SCRIPTS_PATH = "" # User's model scripts are in the same directory

try:
    rom_module = load_module_from_file("rom_full_module", os.path.join(MODEL_SCRIPTS_PATH, "ROM_FULL.py"))
    MultiVarAttentionROM = rom_module.MultiVarAttentionROM
    # UniversalLifting_ROM = rom_module.UniversalLifting 
    # ImprovedUpdateFFN_ROM = rom_module.ImprovedUpdateFFN
    # MultiHeadAttentionROM_ROM = rom_module.MultiHeadAttentionROM

    fno_module = load_module_from_file("fno_full_module", os.path.join(MODEL_SCRIPTS_PATH, "FNO_FULL.py"))
    FNO1d = fno_module.FNO1d

    spfno_module = load_module_from_file("spfno_full_module", os.path.join(MODEL_SCRIPTS_PATH, "SPFNO_FULL.py"))
    SPFNO1d = spfno_module.SPFNO1d
    # ProjectionFilter1d_SPFNO = spfno_module.ProjectionFilter1d 

    beno_module = load_module_from_file("beno_full_module", os.path.join(MODEL_SCRIPTS_PATH, "BENO_FULL.py"))
    BENOStepper = beno_module.BENOStepper

    lno_module = load_module_from_file("lno_full_module", os.path.join(MODEL_SCRIPTS_PATH, "LNO_FULL.py"))
    LatentNeuralOperator = lno_module.LatentNeuralOperator

    pod_dl_rom_module = load_module_from_file("pod_dl_rom_full_module", os.path.join(MODEL_SCRIPTS_PATH, "POD_DL_ROM_FULL.py"))
    POD_DL_ROM = pod_dl_rom_module.POD_DL_ROM
    # Encoder_PODDL = pod_dl_rom_module.Encoder 
    # Decoder_PODDL = pod_dl_rom_module.Decoder
    # DFNN_PODDL = pod_dl_rom_module.DFNN

    lns_ae_module = load_module_from_file("lns_ae_full_module", os.path.join(MODEL_SCRIPTS_PATH, "LNS_AE_FULL.py"))
    LNS_Autoencoder = lns_ae_module.LNS_Autoencoder
    LatentStepperNet = lns_ae_module.LatentStepperNet
    
    UniversalPDEDataset = fno_module.UniversalPDEDataset 
    print("Successfully loaded model definitions.")

except ImportError as e:
    print(f"Error importing model definitions: {e}")
    print("Please ensure all model scripts are in the specified MODEL_SCRIPTS_PATH.")
    exit()
except AttributeError as e:
    print(f"Attribute error while loading model classes: {e}")
    print("Check if class names in the scripts match those expected here.")
    exit()
except FileNotFoundError as e:
    print(f"Error: A model script file was not found: {e}")
    print(f"Please ensure MODEL_SCRIPTS_PATH ('{MODEL_SCRIPTS_PATH}') is correct and all .py files are present.")
    exit()

# =============================================================================
# 2. Global Configuration
# =============================================================================
FULL_T_IN_DATAFILE = 2.0
FULL_NT_IN_DATAFILE = 600 
TRAIN_T_TARGET = 1.0
TRAIN_NT_FOR_MODEL_CALC = int((TRAIN_T_TARGET / FULL_T_IN_DATAFILE) * (FULL_NT_IN_DATAFILE - 1)) + 1

BASE_DATASET_PATH = "./datasets_full/"
BASE_CHECKPOINT_PATH = "./New_ckpt_2/" 
BASE_RESULTS_PATH = "./benchmark_results_all_models_1/" 
os.makedirs(BASE_RESULTS_PATH, exist_ok=True)

NUM_VISUALIZATION_SAMPLES = 50 
VALIDATION_SPLIT_RATIO = 0.2 

# =============================================================================
# 3. Dataset Handling and Normalization
# =============================================================================
def get_global_normalization_stats(data_list, dataset_config, train_indices):
    print(f"Computing global normalization statistics for {dataset_config['name']}...")
    state_keys = dataset_config['state_keys']
    
    if not train_indices:
        raise ValueError("Training indices list is empty. Cannot compute normalization stats.")
    
    dummy_sample_for_dims = {sk: np.zeros((TRAIN_NT_FOR_MODEL_CALC, dataset_config['nx'])) for sk in state_keys}
    dummy_sample_for_dims['BC_State'] = np.zeros((TRAIN_NT_FOR_MODEL_CALC, dataset_config['expected_bc_state_dim']))
    expected_controls = dataset_config.get('expected_num_controls', 0)
    dummy_sample_for_dims['BC_Control'] = np.zeros((TRAIN_NT_FOR_MODEL_CALC, expected_controls if expected_controls > 0 else 0))


    temp_dataset_for_dims = UniversalPDEDataset(
        [dummy_sample_for_dims], 
        dataset_config['name'],
        train_nt_limit=TRAIN_NT_FOR_MODEL_CALC
    )
    bc_state_dim_stats = temp_dataset_for_dims.bc_state_dim
    num_controls_stats = temp_dataset_for_dims.num_controls
    bc_state_key_from_ds = temp_dataset_for_dims.bc_state_key
    bc_control_key_from_ds = temp_dataset_for_dims.bc_control_key

    all_state_data_train = {key: [] for key in state_keys}
    all_bc_state_train = []
    all_bc_control_train = []

    for idx in train_indices:
        sample = data_list[idx]
        current_nt_for_stats = TRAIN_NT_FOR_MODEL_CALC

        for key in state_keys:
            state_seq_full = sample[key]
            all_state_data_train[key].append(state_seq_full[:current_nt_for_stats, ...])

        if bc_state_key_from_ds in sample:
            bc_state_seq_full = sample[bc_state_key_from_ds]
            all_bc_state_train.append(bc_state_seq_full[:current_nt_for_stats, :])
        else:
            print(f"Warning: '{bc_state_key_from_ds}' not in sample {idx} during stat calculation. Appending zeros.")
            all_bc_state_train.append(np.zeros((current_nt_for_stats, bc_state_dim_stats)))

        if num_controls_stats > 0:
            if bc_control_key_from_ds in sample and sample[bc_control_key_from_ds] is not None:
                bc_control_seq_full = sample[bc_control_key_from_ds]
                if bc_control_seq_full.ndim == 2 and bc_control_seq_full.shape[1] == num_controls_stats:
                    all_bc_control_train.append(bc_control_seq_full[:current_nt_for_stats, :])
                else:
                    print(f"Warning: BC_Control for sample {idx} has unexpected shape {bc_control_seq_full.shape} or type. Expected ({current_nt_for_stats},{num_controls_stats}). Appending zeros.")
                    all_bc_control_train.append(np.zeros((current_nt_for_stats, num_controls_stats)))
            else: 
                all_bc_control_train.append(np.zeros((current_nt_for_stats, num_controls_stats)))
    
    global_stats = {}
    for key in state_keys:
        if not all_state_data_train[key]:
            print(f"Warning: No data collected for state key '{key}' for normalization. Using 0 mean, 1 std.")
            global_stats[f'state_{key}_mean'] = 0.0
            global_stats[f'state_{key}_std'] = 1.0
            continue
        concatenated_state = np.concatenate(all_state_data_train[key], axis=0)
        global_stats[f'state_{key}_mean'] = np.mean(concatenated_state)
        global_stats[f'state_{key}_std'] = np.std(concatenated_state) + 1e-8
        print(f"  Global norm for {key}: mean={global_stats[f'state_{key}_mean']:.4f}, std={global_stats[f'state_{key}_std']:.4f}")

    if all_bc_state_train:
        concatenated_bc_state = np.concatenate(all_bc_state_train, axis=0)
        global_stats['BC_State_means'] = np.mean(concatenated_bc_state, axis=0)
        global_stats['BC_State_stds'] = np.std(concatenated_bc_state, axis=0) + 1e-8
    else: 
        global_stats['BC_State_means'] = np.zeros(bc_state_dim_stats)
        global_stats['BC_State_stds'] = np.ones(bc_state_dim_stats)

    if num_controls_stats > 0 and all_bc_control_train:
        concatenated_bc_control = np.concatenate(all_bc_control_train, axis=0)
        global_stats['BC_Control_means'] = np.mean(concatenated_bc_control, axis=0)
        global_stats['BC_Control_stds'] = np.std(concatenated_bc_control, axis=0) + 1e-8
    elif num_controls_stats > 0: 
        global_stats['BC_Control_means'] = np.zeros(num_controls_stats)
        global_stats['BC_Control_stds'] = np.ones(num_controls_stats)

    print("Global normalization statistics computed.")
    return global_stats


class GloballyNormalizedPDEDataset(Dataset):
    def __init__(self, data_list, dataset_config, global_norm_stats, sequence_length_to_load=None):
        self.data_list = data_list
        self.dataset_config = dataset_config
        self.global_norm_stats = global_norm_stats
        self.state_keys = dataset_config['state_keys']
        self.num_state_vars = len(self.state_keys)
        
        first_sample_nt = data_list[0][self.state_keys[0]].shape[0]
        self.sequence_length_to_load = sequence_length_to_load if sequence_length_to_load is not None else first_sample_nt

        dummy_sample_for_dims = {sk: np.zeros((self.sequence_length_to_load, dataset_config['nx'])) for sk in self.state_keys}
        dummy_sample_for_dims['BC_State'] = np.zeros((self.sequence_length_to_load, dataset_config['expected_bc_state_dim']))
        expected_controls = dataset_config.get('expected_num_controls', 0)
        dummy_sample_for_dims['BC_Control'] = np.zeros((self.sequence_length_to_load, expected_controls if expected_controls > 0 else 0))
        
        temp_universal_dataset = UniversalPDEDataset([dummy_sample_for_dims], dataset_config['name'], train_nt_limit=self.sequence_length_to_load)
        self.bc_state_key = temp_universal_dataset.bc_state_key
        self.bc_control_key = temp_universal_dataset.bc_control_key
        self.bc_state_dim = temp_universal_dataset.bc_state_dim
        self.num_controls = temp_universal_dataset.num_controls

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        current_nt = self.sequence_length_to_load

        state_tensors_norm_list = []
        for key in self.state_keys:
            state_seq_full = sample[key]
            state_seq = state_seq_full[:current_nt, ...]
            mean = self.global_norm_stats[f'state_{key}_mean']
            std = self.global_norm_stats[f'state_{key}_std']
            state_norm = (state_seq - mean) / std
            state_tensors_norm_list.append(torch.tensor(state_norm).float())

        bc_state_seq_full = sample[self.bc_state_key]
        bc_state_seq = bc_state_seq_full[:current_nt, :]
        bc_state_norm_tensor = torch.zeros_like(torch.from_numpy(bc_state_seq).float(), dtype=torch.float32)
        if self.bc_state_dim > 0:
            means = self.global_norm_stats['BC_State_means']
            stds = self.global_norm_stats['BC_State_stds']
            for d_idx in range(self.bc_state_dim):
                bc_state_norm_tensor[:, d_idx] = (torch.from_numpy(bc_state_seq[:, d_idx]).float() - means[d_idx]) / (stds[d_idx] + 1e-8)
        
        bc_control_norm_tensor = torch.empty((current_nt, 0), dtype=torch.float32)
        if self.num_controls > 0:
            if self.bc_control_key in sample and sample[self.bc_control_key] is not None:
                bc_control_seq_full = sample[self.bc_control_key]
                bc_control_seq = bc_control_seq_full[:current_nt, :]
                if bc_control_seq.shape[1] == self.num_controls: 
                    temp_bc_control_norm = torch.zeros_like(torch.from_numpy(bc_control_seq).float(), dtype=torch.float32)
                    means_ctrl = self.global_norm_stats.get('BC_Control_means', np.zeros(self.num_controls))
                    stds_ctrl = self.global_norm_stats.get('BC_Control_stds', np.ones(self.num_controls))
                    for d_idx in range(self.num_controls):
                        temp_bc_control_norm[:, d_idx] = (torch.from_numpy(bc_control_seq[:, d_idx]).float() - means_ctrl[d_idx]) / (stds_ctrl[d_idx] + 1e-8)
                    bc_control_norm_tensor = temp_bc_control_norm
                else: 
                    bc_control_norm_tensor = torch.zeros((current_nt, self.num_controls), dtype=torch.float32)
            else: 
                 bc_control_norm_tensor = torch.zeros((current_nt, self.num_controls), dtype=torch.float32)

        bc_ctrl_tensor_norm = torch.cat((bc_state_norm_tensor, bc_control_norm_tensor), dim=-1)
        return state_tensors_norm_list, bc_ctrl_tensor_norm, self.global_norm_stats


def denormalize_data(data_norm, state_key_name, global_stats):
    mean = global_stats[f'state_{state_key_name}_mean']
    std = global_stats[f'state_{state_key_name}_std']
    return data_norm * std + mean

# =============================================================================
# 4. Model Loading and Prediction Wrappers
# =============================================================================

def load_model_checkpoint(model, checkpoint_path):
    if not checkpoint_path or not os.path.exists(checkpoint_path): 
        print(f"Warning: Checkpoint path is invalid or not found: {checkpoint_path}")
        return False
    try:
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False) 
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        if isinstance(state_dict, dict):
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict, strict=False)
            print(f"Successfully loaded weights for {model.__class__.__name__}.")
            return True
        else:
            print(f"Warning: Checkpoint for {model.__class__.__name__} not a dict or no 'model_state_dict'.")
            return False
    except Exception as e:
        print(f"Error loading checkpoint for {model.__class__.__name__} from {checkpoint_path}: {e}")
        return False

def _create_dummy_sample_for_dims(dataset_config):
    dummy_sample = {sk: np.zeros((TRAIN_NT_FOR_MODEL_CALC, dataset_config['nx'])) for sk in dataset_config['state_keys']}
    dummy_sample['BC_State'] = np.zeros((TRAIN_NT_FOR_MODEL_CALC, dataset_config['expected_bc_state_dim']))
    expected_controls = dataset_config.get('expected_num_controls', 0)
    dummy_sample['BC_Control'] = np.zeros((TRAIN_NT_FOR_MODEL_CALC, expected_controls if expected_controls > 0 else 0 ))
    return [dummy_sample]


def get_fno_model(dataset_config, model_config_params, checkpoint_path_override=None):
    num_state_vars = len(dataset_config['state_keys'])
    dummy_data_for_dims = _create_dummy_sample_for_dims(dataset_config)
    temp_ds = UniversalPDEDataset(dummy_data_for_dims, dataset_config['name'], train_nt_limit=TRAIN_NT_FOR_MODEL_CALC)
    actual_bc_state_dim = temp_ds.bc_state_dim
    actual_num_controls = temp_ds.num_controls

    fno_input_channels = num_state_vars + actual_bc_state_dim + actual_num_controls
    fno_output_channels = num_state_vars
    
    model = FNO1d(
        modes=model_config_params.get('modes', 16),
        width=model_config_params.get('width', 64),
        input_channels=fno_input_channels,
        output_channels=fno_output_channels,
        num_layers=model_config_params.get('layers', 4)
    ).to(DEVICE)
    
    ckpt_folder = os.path.join(BASE_CHECKPOINT_PATH, f"checkpoints_{dataset_config['name']}_FNO")
    base_ckpt_name_pattern = f"model_{dataset_config['name']}_FNO_trainT{TRAIN_T_TARGET}_m{model_config_params.get('modes', 16)}_w{model_config_params.get('width', 64)}"
    
    found_ckpt_path = None
    if os.path.exists(ckpt_folder):
        possible_files = [f for f in os.listdir(ckpt_folder) if f.startswith(base_ckpt_name_pattern) and f.endswith(".pt")]
        if possible_files:
            found_ckpt_path = os.path.join(ckpt_folder, possible_files[0]) 
            print(f"Found potential FNO checkpoint: {found_ckpt_path}")

    ckpt_path = checkpoint_path_override if checkpoint_path_override else found_ckpt_path
    
    if not load_model_checkpoint(model, ckpt_path): 
        print(f"Could not load FNO model for {dataset_config['name']}. It will use random weights.")
    return model

def get_spfno_model(dataset_config, model_config_params, checkpoint_path_override=None):
    num_state_vars = len(dataset_config['state_keys'])
    dummy_data_for_dims = _create_dummy_sample_for_dims(dataset_config)
    temp_ds = UniversalPDEDataset(dummy_data_for_dims, dataset_config['name'], train_nt_limit=TRAIN_NT_FOR_MODEL_CALC)
    actual_bc_state_dim = temp_ds.bc_state_dim
    actual_num_controls = temp_ds.num_controls
    spfno_input_channels = num_state_vars + actual_bc_state_dim + actual_num_controls
    spfno_output_channels = num_state_vars

    model = SPFNO1d(
        modes=model_config_params.get('modes', 16),
        width=model_config_params.get('width', 64),
        input_channels=spfno_input_channels,
        output_channels=spfno_output_channels,
        num_layers=model_config_params.get('layers', 4),
        transform_type=dataset_config.get('spfno_transform_type', 'dirichlet'),
        use_projection_filter=model_config_params.get('use_projection_filter', True)
    ).to(DEVICE)
    model.nx = dataset_config['nx']

    default_ckpt_path = os.path.join(BASE_CHECKPOINT_PATH, f"checkpoints_{dataset_config['name']}_SPFNO",
                         f"model_{dataset_config['name']}_SPFNO_trainT{TRAIN_T_TARGET}_m{model_config_params.get('modes', 16)}_w{model_config_params.get('width', 64)}.pt")
    ckpt_path = checkpoint_path_override if checkpoint_path_override else default_ckpt_path

    if not load_model_checkpoint(model, ckpt_path):
        print(f"Could not load SPFNO model for {dataset_config['name']}. It will use random weights.")
    return model


def get_beno_model(dataset_config, model_config_params, checkpoint_path_override=None):
    num_state_vars = len(dataset_config['state_keys'])
    dummy_data_for_dims = _create_dummy_sample_for_dims(dataset_config)
    temp_ds = UniversalPDEDataset(dummy_data_for_dims, dataset_config['name'], train_nt_limit=TRAIN_NT_FOR_MODEL_CALC)
    actual_bc_ctrl_dim_input = temp_ds.bc_state_dim + temp_ds.num_controls
    
    model = BENOStepper(
        nx=dataset_config['nx'],
        num_state_vars=num_state_vars,
        bc_ctrl_dim_input=actual_bc_ctrl_dim_input,
        state_keys=dataset_config['state_keys'],
        embed_dim=model_config_params.get('embed_dim', 64),
        hidden_dim=model_config_params.get('hidden_dim', 64),
        gnn_layers=model_config_params.get('gnn_layers', 4),
        transformer_layers=model_config_params.get('transformer_layers', 2),
        nhead=model_config_params.get('nhead', 4)
    ).to(DEVICE)
    
    default_ckpt_path = os.path.join(BASE_CHECKPOINT_PATH, f"checkpoints_{dataset_config['name']}_BENO_Stepper",
                         f"model_{dataset_config['name']}_BENO_Stepper_trainT{TRAIN_T_TARGET}_emb{model_config_params.get('embed_dim', 64)}.pt")
    ckpt_path = checkpoint_path_override if checkpoint_path_override else default_ckpt_path
    
    if not load_model_checkpoint(model, ckpt_path):
        print(f"Could not load BENO model for {dataset_config['name']}. It will use random weights.")
    return model

def get_lno_model(dataset_config, model_config_params, checkpoint_path_override=None):
    num_state_vars = len(dataset_config['state_keys'])
    model = LatentNeuralOperator(
        num_state_vars=num_state_vars,
        coord_dim=2, 
        M_latent_points=model_config_params.get('M_latent_points', 64),
        D_embedding=model_config_params.get('D_embedding', 64),
        L_transformer_blocks=model_config_params.get('L_transformer_blocks', 3),
        transformer_nhead=model_config_params.get('transformer_nhead', 4),
        transformer_dim_feedforward=model_config_params.get('transformer_dim_feedforward', model_config_params.get('D_embedding', 64) * 2),
        projector_hidden_dims=model_config_params.get('projector_hidden_dims', [64,128]), 
        final_mlp_hidden_dims=model_config_params.get('final_mlp_hidden_dims', [128,64])   
    ).to(DEVICE)
    model.state_keys = dataset_config['state_keys'] 
    model.num_state_vars = num_state_vars
    model.nx = dataset_config['nx']
    model.domain_L = dataset_config.get('L', 1.0)

    default_ckpt_path = os.path.join(BASE_CHECKPOINT_PATH, f"checkpoints_{dataset_config['name']}_LNO_AR",
                         f"model_{dataset_config['name']}_LNO_AR_trainT{TRAIN_T_TARGET}_M{model_config_params.get('M_latent_points', 64)}_D{model_config_params.get('D_embedding', 64)}.pt")
    ckpt_path = checkpoint_path_override if checkpoint_path_override else default_ckpt_path

    if not load_model_checkpoint(model, ckpt_path):
        print(f"Could not load LNO model for {dataset_config['name']}. It will use random weights.")
    return model

def get_pod_dl_rom_model(dataset_config, model_config_params, pod_bases_override=None, checkpoint_path_override=None):
    num_state_vars = len(dataset_config['state_keys'])
    current_nx = dataset_config['nx']
    N_pod = model_config_params.get('N_pod', 64)
    
    V_N_basis_dict = {}
    if pod_bases_override:
        V_N_basis_dict = {k: v.to(DEVICE) for k,v in pod_bases_override.items()}
    else: 
        basis_cache_dir_from_script = f"./pod_bases_cache_{dataset_config['name']}" 
        basis_cache_dir_alternative = os.path.join(BASE_CHECKPOINT_PATH, f"checkpoints_{dataset_config['name']}_POD_DL_ROM", "pod_bases_cache")
        
        basis_cache_dir_to_use = basis_cache_dir_from_script
        if not os.path.isdir(basis_cache_dir_from_script):
            if os.path.isdir(basis_cache_dir_alternative):
                basis_cache_dir_to_use = basis_cache_dir_alternative
            else:
                print(f"ERROR: POD basis cache directory not found at {basis_cache_dir_from_script} or {basis_cache_dir_alternative} for POD-DL-ROM.")
                return None
        
        for key in dataset_config['state_keys']:
            basis_path = os.path.join(basis_cache_dir_to_use, 
                                      f'pod_basis_{key}_nx{current_nx}_N{N_pod}_trainNT{TRAIN_NT_FOR_MODEL_CALC}.npy')
            if os.path.exists(basis_path):
                V_N_basis_dict[key] = torch.from_numpy(np.load(basis_path)).float().to(DEVICE)
            else:
                print(f"ERROR: POD basis not found for {key} at {basis_path} for POD-DL-ROM. Cannot initialize model.")
                return None

    if len(V_N_basis_dict) != num_state_vars:
        print(f"ERROR: POD bases missing for some state variables for POD-DL-ROM. Found {len(V_N_basis_dict)}, expected {num_state_vars}.")
        return None

    dummy_data_for_dims = _create_dummy_sample_for_dims(dataset_config)
    temp_ds = UniversalPDEDataset(dummy_data_for_dims, dataset_config['name'], train_nt_limit=TRAIN_NT_FOR_MODEL_CALC)
    actual_bc_ctrl_dim_input = temp_ds.bc_state_dim + temp_ds.num_controls
    dfnn_input_dim = 1 + actual_bc_ctrl_dim_input 

    model = POD_DL_ROM(
        V_N_dict=V_N_basis_dict,
        n_latent=model_config_params.get('n_latent', 8),
        dfnn_input_dim=dfnn_input_dim,
        N_pod=N_pod,
        num_state_vars=num_state_vars,
    ).to(DEVICE)

    default_ckpt_path = os.path.join(BASE_CHECKPOINT_PATH, f"checkpoints_{dataset_config['name']}_POD_DL_ROM",
                         f"model_{dataset_config['name']}_POD_DL_ROM_trainT{TRAIN_T_TARGET}_N{N_pod}_n{model_config_params.get('n_latent', 8)}.pt")
    ckpt_path = checkpoint_path_override if checkpoint_path_override else default_ckpt_path
    
    if not load_model_checkpoint(model, ckpt_path):
        print(f"Could not load POD-DL-ROM model for {dataset_config['name']}. It will use random weights.")
    return model


def get_lns_ae_model(dataset_config, model_config_params, checkpoint_path_prefix_override=None):
    num_state_vars = len(dataset_config['state_keys'])
    current_nx = dataset_config['nx']
    ae_initial_width = model_config_params.get('ae_initial_width', 64)
    ae_downsample_blocks = model_config_params.get('ae_downsample_blocks', 3)
    ae_latent_channels = model_config_params.get('ae_latent_channels', 16)
    final_latent_nx = current_nx // (2**ae_downsample_blocks)

    autoencoder = LNS_Autoencoder(
        num_state_vars=num_state_vars,
        target_nx_full=current_nx,
        ae_initial_width=ae_initial_width,
        ae_downsample_blocks=ae_downsample_blocks,
        ae_latent_channels=ae_latent_channels,
        final_latent_nx=final_latent_nx
    ).to(DEVICE)
    
    base_lns_ckpt_dir = os.path.join(BASE_CHECKPOINT_PATH, f"checkpoints_{dataset_config['name']}_LatentDON_Stepper")

    ae_ckpt_path_default = os.path.join(base_lns_ckpt_dir, "lns_ae_stage1.pt")
    ae_ckpt_path = checkpoint_path_prefix_override + "_ae_stage1.pt" if checkpoint_path_prefix_override else ae_ckpt_path_default

    if not load_model_checkpoint(autoencoder, ae_ckpt_path):
        print(f"Could not load LNS Autoencoder for {dataset_config['name']}. Latent Stepper may not work correctly.")
    
    dummy_data_for_dims = _create_dummy_sample_for_dims(dataset_config)
    temp_ds = UniversalPDEDataset(dummy_data_for_dims, dataset_config['name'], train_nt_limit=TRAIN_NT_FOR_MODEL_CALC)
    actual_bc_ctrl_dim_input = temp_ds.bc_state_dim + temp_ds.num_controls

    latent_stepper = LatentStepperNet(
        latent_dim_c=ae_latent_channels,
        latent_dim_x=final_latent_nx,
        bc_ctrl_input_dim=actual_bc_ctrl_dim_input,
        branch_hidden_dims=model_config_params.get('branch_hidden_dims', [128, 128]),
        trunk_hidden_dims=model_config_params.get('trunk_hidden_dims', [64, 64]),
        combined_output_p=model_config_params.get('combined_output_p', 128)
    ).to(DEVICE)

    stepper_ckpt_path_default = os.path.join(base_lns_ckpt_dir, "latent_don_stepper_stage2.pt")
    stepper_ckpt_path = checkpoint_path_prefix_override + "_stepper_stage2.pt" if checkpoint_path_prefix_override else stepper_ckpt_path_default
    
    if not load_model_checkpoint(latent_stepper, stepper_ckpt_path):
        print(f"Could not load LNS Latent Stepper for {dataset_config['name']}. It will use random weights.")
    
    return autoencoder, latent_stepper


def get_user_rom_model(dataset_config, model_config_params, pod_bases_override=None, checkpoint_path_override=None):
    num_state_vars = len(dataset_config['state_keys'])
    current_nx = dataset_config['nx']
    
    basis_dim = model_config_params.get('basis_dim', 32)
    d_model = model_config_params.get('d_model', 512)
    num_heads = model_config_params.get('num_heads', 8)
    shared_attention = model_config_params.get('shared_attention', False)

    dummy_data_for_dims = _create_dummy_sample_for_dims(dataset_config)
    temp_ds = UniversalPDEDataset(dummy_data_for_dims, dataset_config['name'], train_nt_limit=TRAIN_NT_FOR_MODEL_CALC)
    bc_state_dim_rom = temp_ds.bc_state_dim
    num_controls_rom = temp_ds.num_controls

    model = MultiVarAttentionROM(
        state_variable_keys=dataset_config['state_keys'],
        nx=current_nx,
        basis_dim=basis_dim,
        d_model=d_model,
        bc_state_dim=bc_state_dim_rom, 
        num_controls=num_controls_rom, 
        num_heads=num_heads,
        add_error_estimator=False, 
        shared_attention=shared_attention
    ).to(DEVICE)

    pod_bases_loaded = {}
    if pod_bases_override:
        pod_bases_loaded = {k: v.to(DEVICE) for k,v in pod_bases_override.items()}
    else:
        basis_dir_rom = os.path.join(BASE_CHECKPOINT_PATH, f"_checkpoints_{dataset_config['name']}", 'pod_bases') 
        for key in dataset_config['state_keys']:
            basis_path = os.path.join(basis_dir_rom, f'pod_basis_{key}_nx{current_nx}.npy') 
            if os.path.exists(basis_path):
                loaded_b = np.load(basis_path)
                if loaded_b.shape == (current_nx, basis_dim):
                     pod_bases_loaded[key] = torch.from_numpy(loaded_b).float().to(DEVICE)
                else:
                    print(f"Warning: Shape mismatch for ROM POD basis {key}. Expected ({current_nx}, {basis_dim}), got {loaded_b.shape}")
            else:
                print(f"ERROR: ROM POD basis not found for {key} at {basis_path}. Cannot initialize ROM model correctly.")
                return None
    
    if len(pod_bases_loaded) != num_state_vars:
        print(f"ERROR: ROM POD bases missing for some state variables. Found {len(pod_bases_loaded)}, expected {num_state_vars}.")
        return None

    with torch.no_grad():
        for key_phi in dataset_config['state_keys']:
            if key_phi in pod_bases_loaded and key_phi in model.Phi:
                model.Phi[key_phi].copy_(pod_bases_loaded[key_phi])
            else:
                 print(f"Warning: POD basis for {key_phi} not loaded into ROM model Phi. Using random init.")

    default_ckpt_path = os.path.join(BASE_CHECKPOINT_PATH, f"_checkpoints_{dataset_config['name']}", 
                         f"rom_{dataset_config['name']}_b{basis_dim}_d{d_model}_v0.pt") 
    ckpt_path = checkpoint_path_override if checkpoint_path_override else default_ckpt_path
    
    if not load_model_checkpoint(model, ckpt_path):
        print(f"Could not load User ROM model for {dataset_config['name']}. It will use random weights.")
    return model


# =============================================================================
# 5. Autoregressive Prediction and Evaluation
# =============================================================================

def perform_rollout(model, model_type_str, initial_state_norm_list, bc_ctrl_seq_norm,
                    num_steps_to_predict, dataset_config, global_stats,
                    model_specific_params=None):
    model.eval()
    state_keys = dataset_config['state_keys']
    num_state_vars = len(state_keys)
    nx = dataset_config['nx']
    
    if num_state_vars > 1:
        current_u_norm_stacked = torch.stack(initial_state_norm_list, dim=-1) 
    else:
        current_u_norm_stacked = initial_state_norm_list[0].unsqueeze(-1)

    predicted_states_norm_sequence = {key: [] for key in state_keys}
    for i_sk, sk in enumerate(state_keys):
        predicted_states_norm_sequence[sk].append(initial_state_norm_list[i_sk].squeeze(0)) 

    with torch.no_grad():
        if model_type_str in ['FNO', 'SPFNO', 'BENO']: 
            temp_current_u_norm_stacked = current_u_norm_stacked.clone() 
            for t_step in range(num_steps_to_predict - 1):
                bc_ctrl_n_step = bc_ctrl_seq_norm[t_step:t_step+1, :]     
                
                if model_type_str == 'BENO':
                    next_u_norm_stacked = model(temp_current_u_norm_stacked, bc_ctrl_n_step)
                else: 
                    bc_ctrl_n_expanded = bc_ctrl_n_step.unsqueeze(1).repeat(1, nx, 1) 
                    model_input = torch.cat((temp_current_u_norm_stacked, bc_ctrl_n_expanded), dim=-1)
                    next_u_norm_stacked = model(model_input) 
                
                for i_sk, sk_loop in enumerate(state_keys): 
                    predicted_states_norm_sequence[sk_loop].append(next_u_norm_stacked[0, :, i_sk])
                temp_current_u_norm_stacked = next_u_norm_stacked

        elif model_type_str == 'LNO': 
            domain_L = dataset_config.get('L', 1.0)
            x_coords_spatial = torch.linspace(0, domain_L, nx, device=DEVICE)
            t_pseudo_input = torch.zeros_like(x_coords_spatial)
            t_pseudo_output = torch.ones_like(x_coords_spatial)
            pos_in_step = torch.stack([x_coords_spatial, t_pseudo_input], dim=-1).unsqueeze(0)  
            pos_out_step = torch.stack([x_coords_spatial, t_pseudo_output], dim=-1).unsqueeze(0) 
            temp_current_u_norm_stacked_lno = current_u_norm_stacked.clone() 
            for _ in range(num_steps_to_predict - 1):
                next_u_norm_stacked_lno = model(pos_in_step, temp_current_u_norm_stacked_lno, pos_out_step)
                for i_sk, sk_loop in enumerate(state_keys):
                    predicted_states_norm_sequence[sk_loop].append(next_u_norm_stacked_lno[0, :, i_sk])
                temp_current_u_norm_stacked_lno = next_u_norm_stacked_lno

        elif model_type_str == 'LNS_AE': 
            autoencoder = model_specific_params['autoencoder']
            latent_stepper = model 
            z_current_latent = autoencoder.encoder(current_u_norm_stacked.permute(0,2,1)) 
            for t_step in range(num_steps_to_predict - 1):
                bc_ctrl_input_roll = bc_ctrl_seq_norm[t_step+1:t_step+2, :] 
                z_next_pred_latent = latent_stepper(z_current_latent, bc_ctrl_input_roll)
                u_next_pred_physical_norm_perm = autoencoder.decoder(z_next_pred_latent) 
                u_next_pred_physical_norm_stacked = u_next_pred_physical_norm_perm.permute(0,2,1) 
                for i_sk, sk_loop in enumerate(state_keys):
                    predicted_states_norm_sequence[sk_loop].append(u_next_pred_physical_norm_stacked[0, :, i_sk])
                z_current_latent = z_next_pred_latent

        elif model_type_str == 'POD_DL_ROM': 
            for sk_pop in state_keys: predicted_states_norm_sequence[sk_pop].pop(0)
            for t_idx in range(num_steps_to_predict):
                time_norm_for_dfnn = torch.tensor([t_idx / (TRAIN_NT_FOR_MODEL_CALC - 1.0 if TRAIN_NT_FOR_MODEL_CALC > 1 else 1.0)],
                                                  device=DEVICE).float().unsqueeze(0) 
                bc_ctrl_t_actual = bc_ctrl_seq_norm[t_idx:t_idx+1, :] 
                dfnn_input_val = torch.cat((time_norm_for_dfnn, bc_ctrl_t_actual), dim=-1)
                u_h_predicted_dict_t_norm, _ = model(dfnn_input_val) 
                for i_sk, sk_loop in enumerate(state_keys):
                    predicted_states_norm_sequence[sk_loop].append(u_h_predicted_dict_t_norm[sk_loop].squeeze(0).squeeze(-1)) 

        elif model_type_str == 'BAROM_ImpBC': 
            a0_dict_norm = {}
            U_B0_lifted_rom = model.lifting(bc_ctrl_seq_norm[0:1, :]) 

            for i_sk, sk_loop in enumerate(state_keys):
                U0_var_norm_unsqueeze = initial_state_norm_list[i_sk].unsqueeze(-1) 
                U_B0_var_rom_unsqueeze = U_B0_lifted_rom[:, i_sk, :].unsqueeze(-1) 
                U0_star_var_norm = U0_var_norm_unsqueeze - U_B0_var_rom_unsqueeze 
                
                Phi_var_rom = model.get_basis(sk_loop).to(DEVICE) 
                Phi_T_var_rom_bmm = Phi_var_rom.transpose(0, 1).unsqueeze(0) 
                
                a0_dict_norm[sk_loop] = torch.bmm(Phi_T_var_rom_bmm, U0_star_var_norm).squeeze(-1) 
            
            pred_U_hat_seq_dict_norm, _ = model(a0_dict_norm, bc_ctrl_seq_norm.unsqueeze(0), T=num_steps_to_predict)
            
            for sk_clear in state_keys: predicted_states_norm_sequence[sk_clear].pop(0) 
            
            for t_idx in range(num_steps_to_predict):
                for i_sk, sk_loop in enumerate(state_keys):
                    predicted_states_norm_sequence[sk_loop].append(pred_U_hat_seq_dict_norm[sk_loop][t_idx].squeeze(0).squeeze(-1))
        else:
            raise ValueError(f"Unknown model type for rollout: {model_type_str}")

    final_predictions_norm_list = []
    for sk_loop in state_keys: 
        if predicted_states_norm_sequence[sk_loop]: 
             final_predictions_norm_list.append(torch.stack(predicted_states_norm_sequence[sk_loop], dim=0)) 
        else: 
             final_predictions_norm_list.append(torch.empty((0,nx), device=DEVICE))
             
    return final_predictions_norm_list


def calculate_metrics_trajectory(predictions_list_denorm, ground_truth_list_denorm, state_keys):
    metrics = {key: {} for key in state_keys}
    overall_pred_flat = []
    overall_gt_flat = []

    for i, key in enumerate(state_keys):
        pred_traj = predictions_list_denorm[i]
        gt_traj = ground_truth_list_denorm[i]

        if pred_traj.shape != gt_traj.shape:
            min_t = min(pred_traj.shape[0], gt_traj.shape[0])
            pred_traj = pred_traj[:min_t]
            gt_traj = gt_traj[:min_t]
            if min_t == 0:
                metrics[key]['mse'] = np.nan; metrics[key]['rmse'] = np.nan; metrics[key]['relative_error'] = np.nan; metrics[key]['max_error'] = np.nan
                continue
        
        if gt_traj.size == 0: 
            metrics[key]['mse'] = np.nan; metrics[key]['rmse'] = np.nan; metrics[key]['relative_error'] = np.nan; metrics[key]['max_error'] = np.nan
            continue

        mse = np.mean((pred_traj - gt_traj)**2)
        rmse = np.sqrt(mse)
        max_error = np.max(np.abs(pred_traj - gt_traj)) if pred_traj.size > 0 else np.nan 

        gt_norm_fro = np.linalg.norm(gt_traj, 'fro')
        relative_error = np.linalg.norm(pred_traj - gt_traj, 'fro') / (gt_norm_fro + 1e-9) if gt_norm_fro > 1e-9 else np.nan
        
        metrics[key]['mse'] = mse
        metrics[key]['rmse'] = rmse
        metrics[key]['relative_error'] = relative_error
        metrics[key]['max_error'] = max_error
        
        overall_pred_flat.append(pred_traj.flatten())
        overall_gt_flat.append(gt_traj.flatten())

    if overall_pred_flat and overall_gt_flat:
        overall_pred_vec = np.concatenate(overall_pred_flat)
        overall_gt_vec = np.concatenate(overall_gt_flat)
        overall_gt_norm = np.linalg.norm(overall_gt_vec)
        metrics['overall_relative_error'] = np.linalg.norm(overall_pred_vec - overall_gt_vec) / (overall_gt_norm + 1e-9) if overall_gt_norm > 1e-9 else np.nan
        metrics['overall_max_error'] = np.max(np.abs(overall_pred_vec - overall_gt_vec)) if overall_pred_vec.size > 0 else np.nan
    else:
        metrics['overall_relative_error'] = np.nan
        metrics['overall_max_error'] = np.nan
        
    return metrics

# =============================================================================
# 6. Visualization (Using User's Advanced Version)
# =============================================================================
def visualize_predictions(gt_denorm_list,
                          predictions_denorm_dict,
                          dataset_config,
                          sample_idx,
                          T_horizon,
                          results_dir_dataset): # Removed global_stats as it's not used here
    state_keys = dataset_config['state_keys']
    L = dataset_config.get('L', 1.0)
    model_names = list(predictions_denorm_dict.keys())

    for i_sk, sk in enumerate(state_keys):
        gt = gt_denorm_list[i_sk]
        all_fields = [gt] + [preds[i_sk] for preds in predictions_denorm_dict.values() if i_sk < len(preds)] 
        all_fields = [a for a in all_fields if a.size > 0] 
        if not all_fields:
            print(f"  Skipping visualization for var {sk}, sample {sample_idx}, T={T_horizon} - no valid field data.")
            continue

        global_min = min(a.min() for a in all_fields)
        global_max = max(a.max() for a in all_fields)
        
        if global_min == global_max:
            global_max += 1e-6 

        abs_max_val = max(abs(global_min), abs(global_max)) 
        use_symlog = abs_max_val > 1e-6 and (global_min < -1e-9 or global_max > abs_max_val * 1e-3) 

        if use_symlog:
            linthresh_val = max(abs_max_val * 1e-4, 1e-9) 
            norm_field = SymLogNorm(linthresh=linthresh_val, linscale=0.5, vmin=global_min, vmax=global_max, base=10)
            cmap_field = 'coolwarm' 
        else:
            norm_field = None 
            cmap_field = 'viridis'

        all_errors = []
        for model_name_err in model_names:
            if i_sk < len(predictions_denorm_dict[model_name_err]):
                arr = predictions_denorm_dict[model_name_err][i_sk]
                if arr.shape == gt.shape and arr.size > 0 and gt.size > 0:
                    all_errors.append(np.abs(arr - gt))
        
        norm_err = None
        cmap_err = 'magma' 
        if all_errors:
            valid_errors = [e for e in all_errors if e.size > 0]
            if valid_errors:
                err_min_val = min(e[e > 1e-9].min() if np.any(e > 1e-9) else np.inf for e in valid_errors) 
                err_max_val = max(e.max() for e in valid_errors)
                if err_max_val > 1e-9 and err_min_val < err_max_val : 
                    norm_err = LogNorm(vmin=max(err_min_val, err_max_val * 1e-6), vmax=err_max_val) 
            
        nrows = 1 + len(model_names) + len(model_names) 
        fig, axs = plt.subplots(nrows, 1, figsize=(7, 2.5 * nrows), squeeze=False)
        fig.suptitle(f"{dataset_config['name']} – {sk} – Sample {sample_idx} – T={T_horizon:.2f}", fontsize=14)

        row = 0
        im_gt_plot = axs[row,0].imshow(gt, origin='lower', aspect='auto', extent=[0, L, 0, T_horizon], norm=norm_field, cmap=cmap_field, vmin=global_min if norm_field is None else None, vmax=global_max if norm_field is None else None)
        axs[row,0].set_ylabel("Time"); axs[row,0].set_title("Ground Truth")
        fig.colorbar(im_gt_plot, ax=axs[row,0], label=f"{sk} value")
        row += 1

        for model_name_plot in model_names:
            if i_sk < len(predictions_denorm_dict[model_name_plot]):
                arr_pred = predictions_denorm_dict[model_name_plot][i_sk]
                if arr_pred.size > 0:
                    im_p = axs[row,0].imshow(arr_pred, origin='lower', aspect='auto', extent=[0, L, 0, T_horizon], norm=norm_field, cmap=cmap_field, vmin=global_min if norm_field is None else None, vmax=global_max if norm_field is None else None)
                    fig.colorbar(im_p, ax=axs[row,0], label=f"{sk} value")
                else: axs[row,0].text(0.5,0.5, "Pred data empty", ha="center", va="center")
            else: axs[row,0].text(0.5,0.5, "Pred data missing", ha="center", va="center")
            axs[row,0].set_ylabel("Time"); axs[row,0].set_title(f"{model_name_plot} Prediction")
            row += 1
        
        last_error_im = None
        for model_name_plot in model_names:
            ax_err = axs[row,0]
            if i_sk < len(predictions_denorm_dict[model_name_plot]):
                arr_err_pred = predictions_denorm_dict[model_name_plot][i_sk]
                if arr_err_pred.shape == gt.shape and arr_err_pred.size > 0 and gt.size > 0:
                    err_data = np.abs(arr_err_pred - gt)
                    last_error_im = ax_err.imshow(err_data, origin='lower', aspect='auto', extent=[0, L, 0, T_horizon], norm=norm_err, cmap=cmap_err) 
                    ax_err.set_title(f"{model_name_plot} |Error| {'(log)' if norm_err else ''}")
                else: ax_err.text(0.5,0.5, "Error N/A", ha="center", va="center"); ax_err.set_title(f"{model_name_plot} |Error|")
            else: ax_err.text(0.5,0.5, "Error N/A", ha="center", va="center"); ax_err.set_title(f"{model_name_plot} |Error|")
            ax_err.set_ylabel("Time")
            row += 1
        
        if last_error_im: 
             fig.colorbar(last_error_im, ax=axs[row-1,0], label=f"{sk} |Error|")

        axs[-1,0].set_xlabel("Spatial Domain (x)")
        fig.tight_layout(rect=[0,0,0.95,0.96]) 

        os.makedirs(results_dir_dataset, exist_ok=True)
        fname = f"sample{sample_idx}_var_{sk}_T{T_horizon:.1f}.png".replace('.','p')
        fig.savefig(os.path.join(results_dir_dataset, fname))
        print(f"Saved visualization: {fname}")
        plt.close(fig)

# =============================================================================
# 7. Main Benchmarking Loop
# =============================================================================
def main_benchmark(args):
    set_seed(args.seed)
    
    dataset_configs = {
        'advection': {'name': 'advection', 'path_suffix': 'advection_data_10000s_128nx_600nt.pkl',
                      'state_keys': ['U'], 'nx': 128, 'L': 1.0, 
                      'expected_bc_state_dim': 2, 'expected_num_controls': 0,
                      'spfno_transform_type': 'dirichlet'},
        'burgers': {'name': 'burgers', 'path_suffix': 'burgers_data_10000s_128nx_600nt.pkl',
                    'state_keys': ['U'], 'nx': 128, 'L': 1.0,
                    'expected_bc_state_dim': 2, 'expected_num_controls': 0,
                    'spfno_transform_type': 'mixed'}, 
        'euler': {'name': 'euler', 'path_suffix': 'euler_data_10000s_128nx_600nt.pkl',
                  'state_keys': ['rho', 'u'], 'nx': 128, 'L': 1.0,
                  'expected_bc_state_dim': 4, 'expected_num_controls': 0,
                  'spfno_transform_type': 'dirichlet'}, 
        'darcy': {'name': 'darcy', 'path_suffix': 'darcy_data_10000s_128nx_600nt.pkl',
                  'state_keys': ['P'], 'nx': 128, 'L': 1.0, 
                  'expected_bc_state_dim': 2, 'expected_num_controls': 0,
                  'spfno_transform_type': 'dirichlet'},
    }

    model_configs = {
        'BAROM_ImpBC': {'loader': get_user_rom_model, 'params': {'basis_dim': 32, 'd_model': 512, 'num_heads': 8, 'shared_attention': False}},
        'FNO': {'loader': get_fno_model, 'params': {'modes': 16, 'width': 64, 'layers': 4}},
        'SPFNO': {'loader': get_spfno_model, 'params': {'modes': 16, 'width': 64, 'layers': 4, 'use_projection_filter': True}},
        'BENO': {'loader': get_beno_model, 'params': {'embed_dim': 64, 'hidden_dim': 64, 'gnn_layers': 4, 'transformer_layers': 2, 'nhead': 4}},
        'LNO': {'loader': get_lno_model, 'params': { 
            'M_latent_points': 64, 'D_embedding': 64, 
            'L_transformer_blocks': 3, 'transformer_nhead': 4,
            'projector_hidden_dims': [64, 128], 
            'final_mlp_hidden_dims': [128, 64]    
            }},
        'POD_DL_ROM': {'loader': get_pod_dl_rom_model, 'params': {'N_pod': 64, 'n_latent': 8}}, 
        'LNS_AE': {'loader': get_lns_ae_model, 'params': { 
            'ae_initial_width': 64, 'ae_downsample_blocks': 3, 'ae_latent_channels': 16,
            'branch_hidden_dims': [128, 128], 'trunk_hidden_dims': [64, 64], 'combined_output_p': 128
            }},
    }
    
    evaluation_T_horizons = [TRAIN_T_TARGET] 
    if FULL_T_IN_DATAFILE > TRAIN_T_TARGET + 1e-5: 
        evaluation_T_horizons.append(TRAIN_T_TARGET + 0.5 * (FULL_T_IN_DATAFILE - TRAIN_T_TARGET))
        evaluation_T_horizons.append(FULL_T_IN_DATAFILE)
    evaluation_T_horizons = sorted(list(set(h for h in evaluation_T_horizons if h <= FULL_T_IN_DATAFILE + 1e-6)))
    print(f"Benchmarking will evaluate rollouts up to T_horizons: {evaluation_T_horizons}")

    for dataset_name_arg in args.datasets:
        if dataset_name_arg not in dataset_configs:
            print(f"Warning: Dataset '{dataset_name_arg}' not configured. Skipping.")
            continue
        
        current_dataset_config = dataset_configs[dataset_name_arg]
        dataset_file_path = os.path.join(BASE_DATASET_PATH, current_dataset_config['path_suffix'])
        results_dir_dataset = os.path.join(BASE_RESULTS_PATH, current_dataset_config['name'])
        os.makedirs(results_dir_dataset, exist_ok=True)
        
        print(f"\n===== Benchmarking for Dataset: {current_dataset_config['name'].upper()} =====")
        
        if not os.path.exists(dataset_file_path):
            print(f"Data file not found: {dataset_file_path}. Skipping dataset.")
            continue
        with open(dataset_file_path, 'rb') as f: all_samples_raw = pickle.load(f)

        num_total_samples = len(all_samples_raw)
        indices = list(range(num_total_samples))
        # For consistent train/val split if script is re-run with same seed
        # If you want a truly random split each time, uncomment random.shuffle(indices)
        # random.shuffle(indices) 
        split_idx = int(num_total_samples * (1 - VALIDATION_SPLIT_RATIO))
        train_indices = indices[:split_idx]; val_indices = indices[split_idx:]

        if not val_indices:
            print(f"No validation samples for {current_dataset_config['name']} after split. Skipping.")
            continue
        
        global_stats = get_global_normalization_stats(all_samples_raw, current_dataset_config, train_indices)
        val_dataset_globally_normed = GloballyNormalizedPDEDataset(
            [all_samples_raw[i] for i in val_indices], 
            current_dataset_config, global_stats, sequence_length_to_load=None
        )
        val_loader = DataLoader(val_dataset_globally_normed, batch_size=1, num_workers=1, shuffle=False)

        all_models_metrics_for_dataset = {} 
        # Stores: {orig_sample_idx: {T_horizon: {'gt': gt_list_denorm, 'preds': {model_name: pred_list_denorm}}}}
        current_dataset_visualization_data = {}


        for model_name, model_info in model_configs.items():
            if args.models and model_name not in args.models: continue

            print(f"\n--- Evaluating Model: {model_name} on {current_dataset_config['name']} ---")
            
            model_specific_params_for_rollout = None
            if model_name == 'LNS_AE':
                autoencoder, latent_stepper = model_info['loader'](current_dataset_config, model_info['params'])
                if autoencoder is None or latent_stepper is None: 
                    print(f"Failed to load LNS_AE components for {current_dataset_config['name']}. Skipping model.")
                    continue
                current_model_instance = latent_stepper 
                model_specific_params_for_rollout = {'autoencoder': autoencoder}
            else:
                current_model_instance = model_info['loader'](current_dataset_config, model_info['params'])
            
            if current_model_instance is None:
                 print(f"Failed to load model {model_name} for {current_dataset_config['name']}. Skipping.")
                 continue

            current_model_instance.to(DEVICE); current_model_instance.eval()
            all_models_metrics_for_dataset[model_name] = {}
            
            # Temporary storage for the current model's predictions for visualization
            current_model_viz_preds_buffer = {} # {orig_idx: {T_horizon: pred_list_denorm}}


            aggregated_metrics_model_all_horizons = {
                T_horz: {
                    **{f"{sk}_mse": [] for sk in current_dataset_config['state_keys']},
                    **{f"{sk}_rmse": [] for sk in current_dataset_config['state_keys']},
                    **{f"{sk}_relative_error": [] for sk in current_dataset_config['state_keys']},
                    **{f"{sk}_max_error": [] for sk in current_dataset_config['state_keys']},
                    'overall_relative_error': [],
                    'overall_max_error': []
                } for T_horz in evaluation_T_horizons
            }

            for val_sample_loader_idx, (initial_state_norm_list_batch, bc_ctrl_seq_norm_batch, _) in enumerate(val_loader):
                initial_state_norm_list_val = [s_batch[0, 0, :].unsqueeze(0).to(DEVICE) for s_batch in initial_state_norm_list_batch]
                bc_ctrl_seq_norm_val = bc_ctrl_seq_norm_batch[0].to(DEVICE) 
                gt_norm_list_full_sample = [s_batch[0].to(DEVICE) for s_batch in initial_state_norm_list_batch] 

                original_sample_idx = val_indices[val_sample_loader_idx] 

                for T_horizon_eval in evaluation_T_horizons:
                    nt_for_horizon = int((T_horizon_eval / FULL_T_IN_DATAFILE) * (FULL_NT_IN_DATAFILE - 1)) + 1
                    nt_for_horizon = min(nt_for_horizon, bc_ctrl_seq_norm_val.shape[0]) 

                    if nt_for_horizon <=1: continue

                    predicted_states_norm_list_horizon = perform_rollout(
                        current_model_instance, model_name, initial_state_norm_list_val,
                        bc_ctrl_seq_norm_val[:nt_for_horizon,:], 
                        nt_for_horizon, current_dataset_config, global_stats,
                        model_specific_params=model_specific_params_for_rollout
                    ) 

                    pred_denorm_list_horizon = []
                    gt_denorm_list_horizon = [] 
                    for i_sk, sk_eval in enumerate(current_dataset_config['state_keys']):
                        pred_denorm_list_horizon.append(
                            denormalize_data(predicted_states_norm_list_horizon[i_sk].cpu().numpy(), sk_eval, global_stats)
                        )
                        gt_denorm_list_horizon.append(
                            denormalize_data(gt_norm_list_full_sample[i_sk][:nt_for_horizon,:].cpu().numpy(), sk_eval, global_stats)
                        )
                    
                    current_sample_metrics = calculate_metrics_trajectory(pred_denorm_list_horizon, gt_denorm_list_horizon, current_dataset_config['state_keys'])
                    
                    for sk_met in current_dataset_config['state_keys']:
                        aggregated_metrics_model_all_horizons[T_horizon_eval][f"{sk_met}_mse"].append(current_sample_metrics[sk_met]['mse'])
                        aggregated_metrics_model_all_horizons[T_horizon_eval][f"{sk_met}_rmse"].append(current_sample_metrics[sk_met]['rmse'])
                        aggregated_metrics_model_all_horizons[T_horizon_eval][f"{sk_met}_relative_error"].append(current_sample_metrics[sk_met]['relative_error']) 
                        aggregated_metrics_model_all_horizons[T_horizon_eval][f"{sk_met}_max_error"].append(current_sample_metrics[sk_met]['max_error'])

                    aggregated_metrics_model_all_horizons[T_horizon_eval]['overall_relative_error'].append(current_sample_metrics['overall_relative_error'])
                    aggregated_metrics_model_all_horizons[T_horizon_eval]['overall_max_error'].append(current_sample_metrics['overall_max_error'])

                    # Store data for visualization
                    if val_sample_loader_idx < NUM_VISUALIZATION_SAMPLES: 
                        if original_sample_idx not in current_model_viz_preds_buffer:
                            current_model_viz_preds_buffer[original_sample_idx] = {}
                        current_model_viz_preds_buffer[original_sample_idx][T_horizon_eval] = pred_denorm_list_horizon
                        
                        # Store GT only once per sample and horizon (e.g., when processing the first model or if not yet stored)
                        if original_sample_idx not in current_dataset_visualization_data:
                            current_dataset_visualization_data[original_sample_idx] = {}
                        if T_horizon_eval not in current_dataset_visualization_data[original_sample_idx]:
                             current_dataset_visualization_data[original_sample_idx][T_horizon_eval] = {'gt': gt_denorm_list_horizon, 'preds': {}}
                        elif 'gt' not in current_dataset_visualization_data[original_sample_idx][T_horizon_eval]: # Ensure GT is there
                             current_dataset_visualization_data[original_sample_idx][T_horizon_eval]['gt'] = gt_denorm_list_horizon
                
                if (val_sample_loader_idx + 1) % 10 == 0:
                    print(f"  Processed {val_sample_loader_idx+1}/{len(val_loader)} samples for {model_name} on {current_dataset_config['name']}.")

            print(f"\n  Aggregated Metrics for {model_name} on {current_dataset_config['name']}:")
            all_models_metrics_for_dataset[model_name] = {} 
            for T_horz_print, metrics_lists in aggregated_metrics_model_all_horizons.items():
                print(f"    T_horizon={T_horz_print:.2f}:")
                all_models_metrics_for_dataset[model_name][T_horz_print] = {} 
                for sk_print in current_dataset_config['state_keys']:
                    avg_mse  = np.nanmean(metrics_lists[f"{sk_print}_mse"])
                    avg_rmse = np.nanmean(metrics_lists[f"{sk_print}_rmse"])
                    avg_rel  = np.nanmean(metrics_lists[f"{sk_print}_relative_error"])
                    avg_max  = np.nanmean(metrics_lists[f"{sk_print}_max_error"])
                    print(f"      {sk_print}: MSE={avg_mse:.4e}, RMSE={avg_rmse:.4e}, RelL2={avg_rel:.4e}, MaxErr={avg_max:.4e}")
                    all_models_metrics_for_dataset[model_name][T_horz_print][f"{sk_print}_mse"] = avg_mse
                    all_models_metrics_for_dataset[model_name][T_horz_print][f"{sk_print}_rmse"] = avg_rmse
                    all_models_metrics_for_dataset[model_name][T_horz_print][f"{sk_print}_relative_error"] = avg_rel
                    all_models_metrics_for_dataset[model_name][T_horz_print][f"{sk_print}_max_error"] = avg_max

                overall_avg_rel = np.nanmean(metrics_lists['overall_relative_error'])
                overall_avg_max = np.nanmean(metrics_lists['overall_max_error'])
                print(f"      Overall: RelL2={overall_avg_rel:.4e}, MaxErr={overall_avg_max:.4e}")
                all_models_metrics_for_dataset[model_name][T_horz_print]['overall_relative_error'] = overall_avg_rel
                all_models_metrics_for_dataset[model_name][T_horz_print]['overall_max_error'] = overall_avg_max
            
            # Add current model's predictions to the central visualization collector
            for orig_idx, horizon_preds in current_model_viz_preds_buffer.items():
                for t_horz, pred_data in horizon_preds.items():
                    if orig_idx in current_dataset_visualization_data and \
                       t_horz in current_dataset_visualization_data[orig_idx]:
                        current_dataset_visualization_data[orig_idx][t_horz]['preds'][model_name] = pred_data
            
        print(f"\n--- Generating Visualizations for Dataset: {current_dataset_config['name'].upper()} ---")
        if current_dataset_visualization_data:
            num_samples_to_visualize = min(NUM_VISUALIZATION_SAMPLES, len(val_indices))
            
            for i_viz_sample_loop in range(num_samples_to_visualize):
                original_idx_viz_key = val_indices[i_viz_sample_loop]

                if original_idx_viz_key not in current_dataset_visualization_data:
                    print(f"  Skipping visualization for sample original_idx={original_idx_viz_key}, data not found in viz collector.")
                    continue
                
                sample_viz_data_all_horizons = current_dataset_visualization_data[original_idx_viz_key]
                print(f"  Visualizing sample original_idx={original_idx_viz_key} ({i_viz_sample_loop+1}/{num_samples_to_visualize})")

                for T_horizon_viz_plot, horizon_data in sample_viz_data_all_horizons.items():
                    gt_to_plot_list = horizon_data.get('gt')
                    preds_to_plot_dict = horizon_data.get('preds')

                    if gt_to_plot_list is None or not preds_to_plot_dict:
                        print(f"    Skipping T={T_horizon_viz_plot} for sample {original_idx_viz_key}, GT or Preds not found for this horizon.")
                        continue
                        
                    visualize_predictions(
                        gt_to_plot_list,
                        preds_to_plot_dict, 
                        current_dataset_config,
                        original_idx_viz_key, 
                        T_horizon_viz_plot,
                        results_dir_dataset 
                    )
        else:
            print("No data collected for visualization for this dataset.")

        metrics_file_path = os.path.join(results_dir_dataset, "all_models_metrics.pkl")
        with open(metrics_file_path, 'wb') as f_metric:
            pickle.dump(all_models_metrics_for_dataset, f_metric) # Save metrics for this dataset
        print(f"Saved aggregated metrics for {current_dataset_config['name']} to {metrics_file_path}")

    print("\n===== BENCHMARKING COMPLETE =====")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark PDE Surrogate Models.")
    parser.add_argument('--datasets', nargs='+', default=['advection', 'burgers', 'euler', 'darcy'],
                        choices=['advection', 'burgers', 'euler', 'darcy'],
                        help='List of datasets to benchmark.')
    parser.add_argument('--models', nargs='+', default=None, 
                        choices=['BAROM_ImpBC','FNO','SPFNO', 'BENO', 'LNO', 'POD_DL_ROM', 'LNS_AE'],
                        help='List of models to benchmark (default: all).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    
    args = parser.parse_args()
    main_benchmark(args)
