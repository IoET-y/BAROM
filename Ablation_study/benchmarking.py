# BENCHMARK_ABLATION_MODELS.PY
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

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
# --- Assume your model classes are importable ---
# You might need to adjust these imports based on your file structure
# For example, if they are all in ROM_FULL.py, ROM_LSTM.py etc.
# from ROM_FULL import MultiVarAttentionROM, UniversalLifting, ImprovedUpdateFFN, MultiHeadAttentionROM as ROMBaseAttention
# from ROM_LSTM import LSTMMultiVarROM # Assuming LSTMMultiVarROM is in a separate file or combined

# Placeholder: For this script to be runnable, these classes need to be defined or imported.
# Using the definitions from previous interactions for now.
# [PASTE UniversalPDEDataset, ImprovedUpdateFFN, UniversalLifting, 
#  MultiHeadAttentionROM (as ROMBaseAttention), MultiVarAttentionROM, LSTMMultiVarROM classes here 
#  from your training scripts or import them if they are in separate modules]

# Simplified example for imports (replace with your actual imports)
# Assuming your model definitions are in a file named `rom_models.py`
# Create a dummy rom_models.py if you don't have one, or adjust path
try:
    from ROM_LSTM import MultiVarAttentionROM, LSTMMultiVarROM, UniversalLifting, ImprovedUpdateFFN, MultiHeadAttentionROM as ROMBaseAttention
    print("Successfully imported model classes from rom_models.")
except ImportError:
    print("Failed to import model classes. Please ensure MultiVarAttentionROM, LSTMMultiVarROM, "
          "UniversalLifting, ImprovedUpdateFFN, and ROMBaseAttention (as MultiHeadAttentionROM) "
          "are defined or correctly imported in this script or from a 'rom_models.py' file.")
    # Fallback to defining dummy classes if import fails, for the script to be parsable
    # THIS IS A PLACEHOLDER - REPLACE WITH YOUR ACTUAL MODEL DEFINITIONS OR IMPORTS
    class DummyModel(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); self.fc = nn.Linear(1,1)
        def forward(self, *args, **kwargs): return self.fc(torch.randn(1,1))
        def get_basis(self, key): return torch.randn(64,32) # Dummy
        def _compute_U_B(self, bc_ctrl): return torch.randn(bc_ctrl.shape[0], 1, 64) # Dummy
    
    MultiVarAttentionROM = DummyModel
    LSTMMultiVarROM = DummyModel
    UniversalLifting = DummyModel
    ImprovedUpdateFFN = DummyModel
    ROMBaseAttention = DummyModel
    print("Using dummy model classes as fallback.")


# --- UniversalPDEDataset Definition (from your provided script) ---
class UniversalPDEDataset(Dataset):
    def __init__(self, data_list, dataset_type, train_nt_limit=None):
        if not data_list:
            raise ValueError("data_list cannot be empty")
        self.data_list = data_list
        self.dataset_type = dataset_type
        self.train_nt_limit = train_nt_limit
        first_sample = data_list[0]
        # Simplified initialization based on your provided benchmark script
        self.nt_from_sample = first_sample['U'].shape[0]
        self.nx_from_sample = first_sample['U'].shape[1]
        self.ny_from_sample = 1 # Assuming 1D for these datasets
        self.state_keys = ['U'] # Assuming 'U' for reaction_diffusion
        self.num_state_vars = 1
        self.expected_bc_state_dim = 2 # For reaction_diffusion

        self.effective_nt = self.train_nt_limit if self.train_nt_limit is not None else self.nt_from_sample
        self.nx = self.nx_from_sample
        self.ny = self.ny_from_sample # Should be 1
        self.spatial_dim = self.nx * self.ny

        self.bc_state_key = 'BC_State'
        if self.bc_state_key not in first_sample:
            raise KeyError(f"'{self.bc_state_key}' not found in the first sample for {dataset_type}!")
        actual_bc_state_dim = first_sample[self.bc_state_key].shape[1]
        
        # For reaction_diffusion, expect 2 (left/right U or flux)
        if actual_bc_state_dim != self.expected_bc_state_dim:
             print(f"Info: For {dataset_type}, expected BC_State dim {self.expected_bc_state_dim}, got {actual_bc_state_dim}. Using actual: {actual_bc_state_dim}")
        self.bc_state_dim = actual_bc_state_dim


        self.bc_control_key = 'BC_Control'
        if self.bc_control_key in first_sample and first_sample[self.bc_control_key] is not None and first_sample[self.bc_control_key].size > 0:
            self.num_controls = first_sample[self.bc_control_key].shape[1]
        else:
            self.num_controls = 0 # Default if not present or empty

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]; norm_factors = {}; current_nt = self.effective_nt
        state_tensors_norm_list = []
        for key in self.state_keys: # Should be just 'U'
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


# --- Configuration ---
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DATASET_PATH = "./datasets_new_feedback/"
# Assuming your ablation checkpoints are in New_ckpt_2 as per the image
BASE_CHECKPOINT_PATH = "./New_ckpt_2/" 
BASE_RESULTS_PATH = "./benchmark_ablation_results/" # New path for these results
MAX_VISUALIZATION_SAMPLES = 5 # Reduced for quicker testing, increase as needed
os.makedirs(BASE_RESULTS_PATH, exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# --- Dataset Configuration (Focus on reaction_diffusion_neumann_feedback) ---
TARGET_DATASET_KEY = 'reaction_diffusion_neumann_feedback'
DATASET_CONFIGS = {
    TARGET_DATASET_KEY: {
        'path': os.path.join(BASE_DATASET_PATH, "reaction_diffusion_neumann_feedback_v1_5000s_64nx_300nt.pkl"),
        'T_file': 2.0, 'NT_file': 300, 
        'T_train': 1.5, # Info about training, used if ckpt name depends on it
        'NT_train': 225, # Derived: int((1.5/2.0)*(300-1))+1
        'nx': 64, 'L': 1.0, 'state_keys': ['U'], 'num_state_vars': 1,
        'dataset_type_arg': 'reaction_diffusion_neumann_feedback' # Matches folder name in checkpoint path
    }
}

# --- Ablation Model Configurations ---
# d_model_attention and num_heads_attention are fixed for AttentionROMs
D_MODEL_ATTENTION = 512
NUM_HEADS_ATTENTION = 8
# LSTM default params from your command
LSTM_HIDDEN_DIM_DEFAULT = 360
NUM_LSTM_LAYERS_DEFAULT = 1
LSTM_CONTROL_EMB_DIM_DEFAULT = 32

ABLATION_MODEL_CONFIGS = {
    "ROM_Attention_Poddim8": {
        "model_class_name": "MultiVarAttentionROM",
        "checkpoint_filename": f"barom_{TARGET_DATASET_KEY}_b8_d{D_MODEL_ATTENTION}.pt",
        "params": {"basis_dim": 8, "d_model": D_MODEL_ATTENTION, "num_heads": NUM_HEADS_ATTENTION, 
                   "use_fixed_lifting": False, "add_error_estimator": False, "shared_attention": False},
        "prediction_fn_name": "predict_rom_model"
    },
    "ROM_Attention_Poddim16": {
        "model_class_name": "MultiVarAttentionROM",
        "checkpoint_filename": f"barom_{TARGET_DATASET_KEY}_b16_d{D_MODEL_ATTENTION}.pt",
        "params": {"basis_dim": 16, "d_model": D_MODEL_ATTENTION, "num_heads": NUM_HEADS_ATTENTION,
                   "use_fixed_lifting": False, "add_error_estimator": False, "shared_attention": False},
        "prediction_fn_name": "predict_rom_model"
    },
    "ROM_Attention_Poddim24": {
        "model_class_name": "MultiVarAttentionROM",
        "checkpoint_filename": f"barom_{TARGET_DATASET_KEY}_b24_d{D_MODEL_ATTENTION}.pt",
        "params": {"basis_dim": 24, "d_model": D_MODEL_ATTENTION, "num_heads": NUM_HEADS_ATTENTION,
                   "use_fixed_lifting": False, "add_error_estimator": False, "shared_attention": False},
        "prediction_fn_name": "predict_rom_model"
    },
    "ROM_Attention_FixedLift": {
        "model_class_name": "MultiVarAttentionROM",
        "checkpoint_filename": f"barom_{TARGET_DATASET_KEY}_b32_d{D_MODEL_ATTENTION}_fixedlift.pt",
        "params": {"basis_dim": 32, "d_model": D_MODEL_ATTENTION, "num_heads": NUM_HEADS_ATTENTION,
                   "use_fixed_lifting": True, "add_error_estimator": False, "shared_attention": False},
        "prediction_fn_name": "predict_rom_model"
    },
    "ROM_Attention_RandomPhi": { # Model trained with random phi, still uses POD-initialized Phi structure
        "model_class_name": "MultiVarAttentionROM",
        "checkpoint_filename": f"barom_{TARGET_DATASET_KEY}_b32_d{D_MODEL_ATTENTION}_randphi.pt",
        "params": {"basis_dim": 32, "d_model": D_MODEL_ATTENTION, "num_heads": NUM_HEADS_ATTENTION,
                   "use_fixed_lifting": False, "add_error_estimator": False, "shared_attention": False},
        "prediction_fn_name": "predict_rom_model"
        # Note: The 'random_phi_init' is a training condition. The loaded model's Phi will be whatever it learned.
    }
}

# --- Utility Functions (load_data, calculate_metrics, plot_comparison - mostly from your script) ---
def load_data(dataset_name_key): # Simplified for single dataset
    config = DATASET_CONFIGS[dataset_name_key]
    dataset_path = config['path']
    if not os.path.exists(dataset_path):
        print(f"Dataset file not found: {dataset_path}"); return None, None, None
    with open(dataset_path, 'rb') as f: data_list_all = pickle.load(f)
    random.shuffle(data_list_all)
    # Use a fixed portion for validation for consistency, e.g., last 20%
    n_total = len(data_list_all)
    n_val_start_idx = int(0.8 * n_total)
    val_data_list = data_list_all[n_val_start_idx:] #debug!!!
    
    if not val_data_list: print(f"No validation data for {dataset_name_key}."); return None, None, None
    
    print(f"Using {len(val_data_list)} samples for validation from {dataset_name_key}.")
    val_dataset = UniversalPDEDataset(val_data_list, dataset_type=config['dataset_type_arg'], train_nt_limit=None) # Use full sequence for validation
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False) # batch_size=1 for sample-by-sample processing
    return val_data_list, val_loader, config

# calculate_metrics and plot_comparison are the same as your provided script
# ... (Omitted for brevity, assume they are defined as in your script) ...
def calculate_metrics(pred_denorm, gt_denorm):
    if pred_denorm.shape != gt_denorm.shape:
        min_len_t = min(pred_denorm.shape[0], gt_denorm.shape[0])
        min_len_x = min(pred_denorm.shape[1], gt_denorm.shape[1])

        pred_denorm = pred_denorm[:min_len_t, :min_len_x]; 
        gt_denorm = gt_denorm[:min_len_t, :min_len_x]
        
        if pred_denorm.shape != gt_denorm.shape: # Still mismatch after trying to align
            print(f"Warning: Shape mismatch persists after slicing. Pred: {pred_denorm.shape}, GT: {gt_denorm.shape}")
            return {'mse': float('nan'), 'rmse': float('nan'), 'rel_err': float('nan'), 'max_err': float('nan')}

    mse = np.mean((pred_denorm - gt_denorm)**2); rmse = np.sqrt(mse)
    rel_err = np.linalg.norm(pred_denorm - gt_denorm, 'fro') / (np.linalg.norm(gt_denorm, 'fro') + 1e-10)
    max_err = np.max(np.abs(pred_denorm - gt_denorm)) if pred_denorm.size > 0 else 0.0
    return {'mse': mse, 'rmse': rmse, 'rel_err': rel_err, 'max_err': max_err}

def plot_comparison(gt_seq_denorm_dict, predictions_denorm_dict, dataset_name, sample_idx_str,
                    state_keys_to_plot, L_domain, T_horizon, save_path_base):
    num_models_to_plot = len(predictions_denorm_dict) + 1 # GT + Predictions
    num_vars = len(state_keys_to_plot)
    if num_vars == 0: print("No state keys to plot."); return

    fig, axs = plt.subplots(num_vars, num_models_to_plot, figsize=(5 * num_models_to_plot, 4 * num_vars), squeeze=False)
    # fig.suptitle(f"Benchmark: {dataset_name} (Sample {sample_idx_str}) @ T={T_horizon:.2f}", fontsize=16)
    
    plot_model_names = ["Ground Truth"] + list(predictions_denorm_dict.keys())

    for i_skey, skey in enumerate(state_keys_to_plot):
        gt_data_var = gt_seq_denorm_dict.get(skey)
        if gt_data_var is None: 
            axs[i_skey,0].set_ylabel(f"{skey}\n(GT Missing)"); continue

        all_series_for_var = [gt_data_var] + \
                             [predictions_denorm_dict.get(model_name, {}).get(skey) for model_name in predictions_denorm_dict.keys()]
        
        valid_series = [s for s in all_series_for_var if s is not None and s.size > 0 and s.ndim == 2] # Ensure 2D for imshow
        if not valid_series: 
            axs[i_skey,0].set_ylabel(f"{skey}\n(No valid data)"); continue
        
        vmin = min(s.min() for s in valid_series); vmax = max(s.max() for s in valid_series)
        if abs(vmax - vmin) < 1e-9: vmax = vmin + 1.0 # Avoid singular range for colorbar

        for j_model, model_name_plot in enumerate(plot_model_names):
            ax = axs[i_skey, j_model]
            data_to_plot = gt_data_var if model_name_plot == "Ground Truth" else \
                           predictions_denorm_dict.get(model_name_plot, {}).get(skey, None)
            
            if data_to_plot is None or data_to_plot.size == 0 or data_to_plot.ndim != 2:
                ax.text(0.5, 0.5, "No data" if data_to_plot is None else "Invalid data shape", ha="center", va="center")
            else:
                # Ensure data_to_plot has the correct number of timesteps for this T_horizon
                # This should already be handled by slicing before calling plot_comparison
                im = ax.imshow(data_to_plot, aspect='auto', origin='lower', 
                               extent=[0, L_domain, 0, T_horizon], 
                               cmap='viridis', vmin=vmin, vmax=vmax)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            ax.set_title(f"{model_name_plot}\n({skey})", fontsize=10)
            if i_skey == num_vars -1 : ax.set_xlabel("x") # X-label only on bottom row
            if j_model == 0: ax.set_ylabel(f"{skey}\nt (physical)")
            else: ax.set_yticklabels([])


    fig.suptitle(f"Benchmark: {dataset_name} (Sample {sample_idx_str}) @ T={T_horizon:.2f}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to make space for suptitle
    
    final_save_path = os.path.join(save_path_base, f"comparison_{dataset_name}_sample{sample_idx_str}_T{T_horizon:.2f}.png")
    os.makedirs(save_path_base, exist_ok=True)
    plt.savefig(final_save_path); 
    print(f"Saved comparison plot to {final_save_path}"); 
    plt.close(fig)


# --- Model Loading Function for Ablations ---
def load_ablation_model(model_name_key, model_config, dataset_config, device):
    checkpoint_filename = model_config["checkpoint_filename"]
    # Construct full checkpoint path
    # Checkpoint path from image: ./New_ckpt_2/_checkpoints_reaction_diffusion_neumann_feedback/FILE_NAME
    ckpt_dir = os.path.join(BASE_CHECKPOINT_PATH, f"_checkpoints_{dataset_config['dataset_type_arg']}")
    checkpoint_path = os.path.join(ckpt_dir, checkpoint_filename)

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found for {model_name_key}: {checkpoint_path}")
        return None

    print(f"Loading checkpoint for {model_name_key} from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Get necessary params from dataset_config and model_config['params']
    state_keys = dataset_config['state_keys']
    nx = dataset_config['nx']
    
    # Need bc_state_dim and num_controls from a dummy dataset instance, as they can vary
    # This ensures the model is instantiated with the same dims as during its training
    # (assuming the training script used UniversalPDEDataset to get these dims)
    if not dataset_config['path']: # Should not happen if config is correct
        print("Error: Dataset path missing in dataset_config for dummy dataset creation.")
        return None
    try:
        with open(dataset_config['path'], 'rb') as f_dummy:
            dummy_data_sample = pickle.load(f_dummy)[0] # Load just one sample
        dummy_dataset = UniversalPDEDataset([dummy_data_sample], dataset_type=dataset_config['dataset_type_arg'])
        bc_state_dim_actual = dummy_dataset.bc_state_dim
        num_controls_actual = dummy_dataset.num_controls
    except Exception as e:
        print(f"Error creating dummy dataset to infer bc_dims for {model_name_key}: {e}")
        # Fallback if dummy dataset fails, try to get from checkpoint if saved, else error
        bc_state_dim_actual = ckpt.get('bc_state_dim', 2) # Default or error
        num_controls_actual = ckpt.get('num_controls', 2) # Default or error
        print(f"Warning: Using fallback bc_state_dim={bc_state_dim_actual}, num_controls={num_controls_actual}")


    model_params_from_config = model_config["params"]
    model_class_name = model_config["model_class_name"]
    
    # Override with checkpoint params if they exist (more reliable)
    basis_dim_ckpt = ckpt.get('basis_dim', model_params_from_config.get('basis_dim'))
    use_fixed_lifting_ckpt = ckpt.get('use_fixed_lifting', model_params_from_config.get('use_fixed_lifting', False))

    model = None
    if model_class_name == "MultiVarAttentionROM":
        d_model_ckpt = ckpt.get('d_model', model_params_from_config.get('d_model'))
        num_heads_ckpt = ckpt.get('num_heads', model_params_from_config.get('num_heads'))
        shared_attention_ckpt = ckpt.get('shared_attention', model_params_from_config.get('shared_attention', False))
        # add_error_estimator from checkpoint or config
        add_error_estimator_ckpt = ckpt.get('add_error_estimator', model_params_from_config.get('add_error_estimator', False))


        model = MultiVarAttentionROM(
            state_variable_keys=state_keys, nx=nx, basis_dim=basis_dim_ckpt,
            d_model=d_model_ckpt, bc_state_dim=bc_state_dim_actual,
            num_controls=num_controls_actual, num_heads=num_heads_ckpt,
            add_error_estimator=add_error_estimator_ckpt, 
            shared_attention=shared_attention_ckpt,
            use_fixed_lifting=use_fixed_lifting_ckpt
        )
    elif model_class_name == "LSTMMultiVarROM":
        lstm_hidden_dim_ckpt = ckpt.get('lstm_hidden_dim', model_params_from_config.get('lstm_hidden_dim'))
        num_lstm_layers_ckpt = ckpt.get('num_lstm_layers', model_params_from_config.get('num_lstm_layers'))
        control_embedding_dim_ckpt = ckpt.get('control_embedding_dim', model_params_from_config.get('control_embedding_dim'))
        add_error_estimator_ckpt = ckpt.get('add_error_estimator', model_params_from_config.get('add_error_estimator', False))

        model = LSTMMultiVarROM(
            state_variable_keys=state_keys, nx=nx, basis_dim=basis_dim_ckpt,
            bc_state_dim=bc_state_dim_actual, num_controls=num_controls_actual,
            lstm_hidden_dim=lstm_hidden_dim_ckpt,
            num_lstm_layers=num_lstm_layers_ckpt,
            control_embedding_dim=control_embedding_dim_ckpt,
            add_error_estimator=add_error_estimator_ckpt,
            use_fixed_lifting=use_fixed_lifting_ckpt
        )
    else:
        print(f"Unknown model class name: {model_class_name} for {model_name_key}")
        return None

    try:
        model.load_state_dict(ckpt['model_state_dict'], strict=False) # strict=False if lifting layer might be missing
    except RuntimeError as e:
        print(f"RuntimeError loading state_dict for {model_name_key}: {e}")
        print("This might be due to architectural mismatch not handled by strict=False (e.g. missing keys beyond lifting).")
        return None
    except KeyError as e:
        print(f"KeyError loading state_dict for {model_name_key} (likely 'model_state_dict' missing in ckpt): {e}")
        return None

    model.to(device)
    model.eval()
    print(f"Successfully loaded {model_name_key}.")
    return model

# --- Prediction Function for Ablation ROMs ---
@torch.no_grad()
def predict_rom_model(model, initial_state_dict_norm, bc_ctrl_seq_norm, num_steps, dataset_config):
    """
    Generic prediction function for both MultiVarAttentionROM and LSTMMultiVarROM.
    Assumes model has _compute_U_B, get_basis, and a compatible forward(a0_dict, BC_Ctrl_seq, T)
    """
    a0_dict = {}
    batch_size = list(initial_state_dict_norm.values())[0].shape[0] # batch_size is 1 for benchmark
    device_model = next(model.parameters()).device # Get model's device

    # Ensure all inputs are on the same device as the model
    initial_state_dict_norm = {k: v.to(device_model) for k, v in initial_state_dict_norm.items()}
    bc_ctrl_seq_norm = bc_ctrl_seq_norm.to(device_model)


    # Initial state projection to a0
    BC_ctrl_t0_norm = bc_ctrl_seq_norm[:, 0, :]
    # Use _compute_U_B which should exist in both model types
    U_B0_lifted_norm = model._compute_U_B(BC_ctrl_t0_norm) 

    for i_key, key in enumerate(model.state_keys):
        U0_norm_var = initial_state_dict_norm[key]
        if U0_norm_var.dim() == 2: U0_norm_var = U0_norm_var.unsqueeze(-1) # Ensure [B, N, 1]

        # U_B0_lifted_norm is [B, num_vars, N]. Need to select the correct variable.
        U_B0_norm_var = U_B0_lifted_norm[:, i_key, :].unsqueeze(-1) # [B, N, 1]
        
        U0_star_norm_var = U0_norm_var - U_B0_norm_var
        
        Phi_var = model.get_basis(key).to(device_model) # [N, basis_dim]
        Phi_T_var = Phi_var.transpose(0,1).unsqueeze(0).expand(batch_size, -1, -1) # [B, basis_dim, N]
        
        a0_norm_var = torch.bmm(Phi_T_var, U0_star_norm_var).squeeze(-1) # [B, basis_dim]
        a0_dict[key] = a0_norm_var

    # Model forward pass
    # The model's forward method should handle the rollout for T steps
    pred_seq_norm_dict_of_lists, _ = model(a0_dict, bc_ctrl_seq_norm[:, :num_steps, :], T=num_steps)
    
    pred_seq_norm_dict_of_tensors = {}
    for key in model.state_keys:
        if pred_seq_norm_dict_of_lists[key]: # Check if list is not empty
             # Each item in list is [batch_size, nx, 1]. Concatenate along time (new dim 0).
             # Then reshape/squeeze.
            stacked_preds = torch.stack(pred_seq_norm_dict_of_lists[key], dim=1) # [B, T, N, 1]
            pred_seq_norm_dict_of_tensors[key] = stacked_preds.squeeze(0).squeeze(-1) # [T, N] for B=1
        else: # Handle case where prediction list might be empty for some reason
            pred_seq_norm_dict_of_tensors[key] = torch.empty(0, dataset_config['nx'], device=device_model)


    return pred_seq_norm_dict_of_tensors


# --- Main Benchmarking Logic ---
def main(models_to_benchmark_keys):
    print(f"Device: {DEVICE}")
    dataset_name_key = TARGET_DATASET_KEY # Focus on one dataset
    print(f"Benchmarking models: {models_to_benchmark_keys} on dataset {dataset_name_key}")
    
    BENCHMARK_T_HORIZONS = [1.5, 1.75, 2.0] 

    overall_aggregated_metrics = {}    
    overall_aggregated_metrics[dataset_name_key] = {
        model_name: {
            T_h: {skey: {'mse': [], 'rel_err': [], 'rmse': [], 'max_err': []} 
                  for skey in DATASET_CONFIGS[dataset_name_key]['state_keys']}
            for T_h in BENCHMARK_T_HORIZONS 
            if T_h <= DATASET_CONFIGS[dataset_name_key]['T_file']
        } for model_name in models_to_benchmark_keys
    }
            
    val_data_list, val_loader, ds_config = load_data(dataset_name_key)
    if val_loader is None: 
        print(f"Failed to load data for {dataset_name_key}. Exiting.")
        return

    num_val_samples = len(val_data_list)
    vis_sample_count = min(MAX_VISUALIZATION_SAMPLES, num_val_samples)
    visualization_indices = random.sample(range(num_val_samples), vis_sample_count) if num_val_samples > 0 else []
    print(f"Will visualize {len(visualization_indices)} random samples: {visualization_indices}")
    
    gt_data_for_visualization = {vis_idx: None for vis_idx in visualization_indices}
    predictions_for_visualization_all_models_all_horizons = {
        vis_idx: {model_name: {} for model_name in models_to_benchmark_keys}
        for vis_idx in visualization_indices
    }

    loaded_models_cache = {} # Cache loaded models
    for model_name_key in models_to_benchmark_keys:
        print(f"\n  -- Pre-loading Model: {model_name_key} for dataset {dataset_name_key} --")
        if model_name_key not in ABLATION_MODEL_CONFIGS:
            print(f"  Skipping model {model_name_key}: Not configured in ABLATION_MODEL_CONFIGS.")
            loaded_models_cache[model_name_key] = None; continue
        
        model_arch_config = ABLATION_MODEL_CONFIGS[model_name_key]
        model_instance = load_ablation_model(model_name_key, model_arch_config, ds_config, DEVICE)
        loaded_models_cache[model_name_key] = model_instance
        if model_instance is None:
            print(f"  Failed to pre-load model {model_name_key}.")

    for val_idx, (sample_state_list_norm, sample_bc_ctrl_seq_norm, sample_norm_factors) in enumerate(val_loader):
        print(f"  Processing validation sample {val_idx+1}/{num_val_samples} for dataset {dataset_name_key}...")
        
        initial_state_norm_dict = {}
        gt_full_seq_denorm_dict_current_sample = {}
        for idx_skey, skey in enumerate(ds_config['state_keys']):
            # initial_state_norm_dict expects [batch, nx] for each key
            initial_state_norm_dict[skey] = sample_state_list_norm[idx_skey][:, 0, :].to(DEVICE) # [1, nx]
            
            gt_seq_norm_var = sample_state_list_norm[idx_skey].squeeze(0).to(DEVICE) # [NT_file, nx]
            mean_val = sample_norm_factors[f'{skey}_mean'].item(); std_val = sample_norm_factors[f'{skey}_std'].item()
            gt_full_seq_denorm_dict_current_sample[skey] = gt_seq_norm_var.cpu().numpy() * std_val + mean_val
        
        if val_idx in visualization_indices: 
            gt_data_for_visualization[val_idx] = gt_full_seq_denorm_dict_current_sample
        
        for T_current_horizon in BENCHMARK_T_HORIZONS:
            if T_current_horizon > ds_config['T_file']: continue
            
            num_benchmark_steps = int((T_current_horizon / ds_config['T_file']) * (ds_config['NT_file'] - 1)) + 1
            num_benchmark_steps = min(num_benchmark_steps, ds_config['NT_file'])

            gt_benchmark_horizon_denorm_dict = {}
            for skey in ds_config['state_keys']:
                gt_benchmark_horizon_denorm_dict[skey] = gt_full_seq_denorm_dict_current_sample[skey][:num_benchmark_steps, :]
            
            sample_bc_ctrl_seq_norm_benchmark = sample_bc_ctrl_seq_norm[:, :num_benchmark_steps, :].to(DEVICE)

            for model_name_key in models_to_benchmark_keys:
                current_model_instance = loaded_models_cache.get(model_name_key)
                if current_model_instance is None: continue
                
                model_arch_config = ABLATION_MODEL_CONFIGS[model_name_key]
                prediction_fn_name = model_arch_config["prediction_fn_name"]
                
                pred_seq_norm_dict_model = {}
                try:
                    if prediction_fn_name == "predict_rom_model":
                        # predict_rom_model needs initial_state_norm_dict with [B, Nx]
                        pred_seq_norm_dict_model = predict_rom_model(
                            current_model_instance, 
                            initial_state_norm_dict, # Pass the dict {key: [B,Nx]}
                            sample_bc_ctrl_seq_norm_benchmark, 
                            num_benchmark_steps, 
                            ds_config
                        )
                    else:
                        print(f"Unknown prediction function: {prediction_fn_name} for model {model_name_key}")
                        continue
                except Exception as e:
                    print(f"    ERROR during prediction for {model_name_key} on sample {val_idx}, T={T_current_horizon}: {e}"); traceback.print_exc(); continue

                model_preds_denorm_this_sample_this_horizon = {}
                for skey in ds_config['state_keys']:
                    if skey not in pred_seq_norm_dict_model or pred_seq_norm_dict_model[skey] is None or pred_seq_norm_dict_model[skey].numel() == 0:
                        print(f"Warning: No prediction data for state key '{skey}' from model '{model_name_key}'.")
                        continue
                    
                    pred_norm_var = pred_seq_norm_dict_model[skey].cpu().numpy() # Expected [T, Nx]
                    gt_denorm_var = gt_benchmark_horizon_denorm_dict[skey] # Expected [T, Nx]
                    
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
        if vis_idx not in gt_data_for_visualization or gt_data_for_visualization[vis_idx] is None:
            print(f"    Skipping visualization for sample index {vis_idx}, GT data not found."); continue
        
        for T_current_horizon_plot in BENCHMARK_T_HORIZONS:
            if T_current_horizon_plot > ds_config['T_file']: continue

            num_plot_steps = int((T_current_horizon_plot / ds_config['T_file']) * (ds_config['NT_file'] - 1)) + 1
            num_plot_steps = min(num_plot_steps, ds_config['NT_file'])

            current_gt_denorm_sliced_for_plot = {
                skey: data[:num_plot_steps, :] for skey, data in gt_data_for_visualization[vis_idx].items()
            }
            current_preds_for_plot_this_horizon = {}
            for model_name_plot_key in models_to_benchmark_keys:
                if model_name_plot_key in predictions_for_visualization_all_models_all_horizons[vis_idx] and \
                   T_current_horizon_plot in predictions_for_visualization_all_models_all_horizons[vis_idx][model_name_plot_key]:
                    current_preds_for_plot_this_horizon[model_name_plot_key] = predictions_for_visualization_all_models_all_horizons[vis_idx][model_name_plot_key][T_current_horizon_plot]
            
            if not current_preds_for_plot_this_horizon: continue

            plot_comparison(current_gt_denorm_sliced_for_plot, current_preds_for_plot_this_horizon,
                            dataset_name_key, f"{val_idx}", ds_config['state_keys'],
                            ds_config['L'], T_current_horizon_plot,
                            os.path.join(BASE_RESULTS_PATH, dataset_name_key, "plots")) # Subfolder for plots

    print("\n\n===== BENCHMARKING COMPLETE - AGGREGATED METRICS =====")
    # ... (Aggregated metrics printing - same as your script) ...
    for ds_name_print, model_data_agg_per_horizon in overall_aggregated_metrics.items():
        print(f"\n--- Aggregated Results for Dataset: {ds_name_print.upper()} ---")
        for model_name_print, horizon_metrics_data in model_data_agg_per_horizon.items():
            print(f"  Model: {model_name_print}")
            if not horizon_metrics_data: print("    No metrics recorded for this model."); continue
            for T_h_print, var_metrics_lists in sorted(horizon_metrics_data.items()):
                print(f"    Horizon T={T_h_print:.2f}:")
                for skey_print, metrics_lists in var_metrics_lists.items():
                    if not metrics_lists['mse']: print(f"      {skey_print}: No metrics recorded."); continue
                    avg_mse = np.mean(metrics_lists['mse']) if metrics_lists['mse'] else float('nan')
                    avg_rmse = np.mean(metrics_lists['rmse']) if metrics_lists['rmse'] else float('nan')
                    avg_rel_err = np.mean(metrics_lists['rel_err']) if metrics_lists['rel_err'] else float('nan')
                    avg_max_err = np.mean(metrics_lists['max_err']) if metrics_lists['max_err'] else float('nan')
                    print(f"      {skey_print}: Avg MSE={avg_mse:.3e}, Avg RMSE={avg_rmse:.3e}, Avg RelErr={avg_rel_err:.3e}, Avg MaxErr={avg_max_err:.3e} (from {len(metrics_lists['mse'])} samples)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark Ablation Study Models.")
    # By default, benchmark all configured ablation models for the target dataset
    parser.add_argument('--models', nargs='+', default=list(ABLATION_MODEL_CONFIGS.keys()),
                        choices=list(ABLATION_MODEL_CONFIGS.keys()), help='Which ablation models to benchmark.')
    # Dataset is fixed for this specific benchmark setup
    # parser.add_argument('--datasets', nargs='+', default=[TARGET_DATASET_KEY],
    #                     choices=[TARGET_DATASET_KEY], help='Which datasets to benchmark.')

    args = parser.parse_args()
    main(args.models)
