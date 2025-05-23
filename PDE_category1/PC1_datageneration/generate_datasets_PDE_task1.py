# Generate_dataset.py
# =============================================================================
#        Standalone Dataset Generation Script
# Generates datasets for Advection, Euler, Burgers, Darcy problems
# with nonlinear/time-varying/controlled boundary conditions.
# Saves data as pickle files for later use by models.
# =============================================================================
import os
import numpy as np
import random
import time
import pickle
import argparse
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve # For Burgers' solver

# ---------------------
# 固定随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)


# Note: PyTorch seeding is not needed here as it's NumPy/SciPy based.
# ---------------------

print(f"Script started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# =============================================================================
# 1. DATA GENERATOR FUNCTIONS
# =============================================================================

# --- 1.1 Advection Generator ---
def generate_pde_data_challenging_with_control(
    n_samples=300, nx=64, nt=100, T=1.0, L=1.0, c=1.0, r=1.0, num_controls=1
):
    """ Generates Advection-Reaction data with control. """
    print(f"Generating Advection data: n={n_samples}, nx={nx}, nt={nt}, T={T}, L={L}, c={c}, r={r}, nc={num_controls}")
    data_list = []
    dt = T / (nt - 1)
    dx = L / (nx - 1)
    t_vals = np.linspace(0, T, nt)
    x_vals = np.linspace(0, L, nx)
    valid_samples_count = 0

    for s in range(n_samples):
        if (s + 1) % 100 == 0: print(f"  Advection sample {s+1}/{n_samples}...")
        # --- Sample parameters ---
        mu1 = np.random.uniform(0.8, 1.2)
        A = np.random.uniform(0.5, 1.5); B = np.random.uniform(0.5, 1.5)
        E = np.random.uniform(0.5, 1.5); F_param = np.random.uniform(0.5, 1.5)
        C = np.random.uniform(0.5, 1.5); D = np.random.uniform(0.5, 1.5)
        G = np.random.uniform(0.5, 1.5)
        control_signals = np.zeros((nt, num_controls))
        for k in range(num_controls):
            control_amp = np.random.uniform(0.1, 0.5); control_freq = np.random.uniform(1.0, 5.0)
            control_phase = np.random.uniform(0, 2 * np.pi)
            control_k = control_amp * np.sin(2 * np.pi * control_freq * t_vals / T + control_phase)
            step_time1 = T * np.random.uniform(0.2, 0.4); step_time2 = T * np.random.uniform(0.6, 0.8)
            step_val1 = np.random.uniform(-0.3, 0.3); step_val2 = np.random.uniform(-0.3, 0.3)
            control_k[t_vals > step_time1] += step_val1; control_k[t_vals > step_time2] += step_val2
            smooth_window = max(1, nt // 50); control_signals[:, k] = np.convolve(control_k, np.ones(smooth_window)/smooth_window, mode='same') # Added smoothing
        control_effect_left = control_signals[:, 0] if num_controls > 0 else 0.0
        noise_std = 0.05
        # --- Define BCs ---
        BC_left_state = (A * np.sin(2*np.pi*t_vals/T) + B * np.cos(2*np.pi*t_vals/T) +
                   A * np.power(np.sin(2*np.pi*t_vals/T), 3) + E * np.sin(6*np.pi*t_vals/T) +
                   F_param * np.cos(6*np.pi*t_vals/T) + np.random.randn(*t_vals.shape) * noise_std)
        BC_left_actual = BC_left_state + control_effect_left
        BC_right_actual = (C * np.sin(2*np.pi*t_vals/T) + D * np.cos(4*np.pi*t_vals/T) +
                    D * np.power(np.cos(2*np.pi*t_vals/T), 3) + np.tanh(np.sin(2*np.pi*t_vals/T)) +
                    G * np.exp(-((t_vals - T/2)**2)/(0.01*T**2)) + np.random.randn(*t_vals.shape) * noise_std)
        BC_state = np.stack([BC_left_state, BC_right_actual], axis=-1)
        BC_control = control_signals
        # --- Solve PDE ---
        U = np.zeros((nt, nx))
        U[0, :] = mu1 * np.exp(-((x_vals - L/2)**2)/(0.1*L**2))
        U[0, 0] = BC_left_actual[0]; U[0, -1] = BC_right_actual[0]
        valid_sim = True
        cfl_limit = dx / dt if dt > 1e-9 else float('inf')
        if abs(c) > cfl_limit: print(f"  Warning: Advection CFL unstable |c|={abs(c):.2f} > dx/dt={cfl_limit:.2f}"); valid_sim=False # Basic check

        for n in range(nt - 1):
            if not valid_sim: break
            U[n, 0] = BC_left_actual[n]; U[n, -1] = BC_right_actual[n]
            if c > 0:
                 U[n+1, 1:-1] = U[n, 1:-1] - c * dt/dx * (U[n, 1:-1] - U[n, 0:-2]) + dt * r * U[n, 1:-1]
            else: # Use Lax-Friedrichs if c <= 0 (more stable than downwind)
                 U[n+1, 1:-1] = 0.5 * (U[n, 2:] + U[n, :-2]) - c * dt/(2*dx) * (U[n, 2:] - U[n, :-2]) + dt * r * U[n, 1:-1]
            if np.isnan(U[n+1, :]).any() or np.isinf(U[n+1, :]).any() or np.max(np.abs(U[n+1,:])) > 1e6:
                 # print(f"    Unstable Advection step {n} samp {s}. Skip.")
                 valid_sim = False; break
            U[n+1, 0] = BC_left_actual[n+1]; U[n+1, -1] = BC_right_actual[n+1]

        if valid_sim:
            sample_params = {'mu1': mu1, 'A': A, 'B': B, 'E': E, 'F': F_param, 'C': C, 'D': D, 'G': G, 'c': c, 'r': r, 'L': L, 'T': T, 'nx': nx, 'nt': nt}
            sample = {'U': U.astype(np.float32), 'BC_State': BC_state.astype(np.float32), 'BC_Control': BC_control.astype(np.float32), 'params': sample_params}
            data_list.append(sample)
            valid_samples_count += 1

    print(f"Finished Advection: Generated {valid_samples_count} valid samples out of {n_samples}.")
    return data_list

# --- 1.2 Euler Generator ---
def generate_euler_data_nonlinear_bc(n_samples=300, nx=64, nt=200, T=1.0, L=1.0, num_controls=2):
    """ Generates Isothermal Euler data with nonlinear/controlled BCs. """
    print(f"Generating Euler data: n={n_samples}, nx={nx}, nt={nt}, T={T}, L={L}, nc={num_controls}")
    data_list = []
    dt = T / (nt - 1)
    dx = L / (nx - 1)
    t_vals = np.linspace(0, T, nt)
    x_grid = np.linspace(0, L, nx)
    cfl_limit = dx / (1.0 + 1.0) # Approx limit assuming max|u|~1, c=1
    if dt > cfl_limit * 0.95: print(f"Warning: Euler CFL potentially unstable dt={dt:.4f} vs limit={cfl_limit:.4f}")
    valid_samples_count = 0

    for s in range(n_samples):
        if (s + 1) % 100 == 0: print(f"  Euler sample {s+1}/{n_samples}...")
        # --- Sample parameters ---
        rho_init_mean = np.random.uniform(0.8, 1.2); u_init_mean = np.random.uniform(-0.2, 0.2)
        rho_init_pert_amp = np.random.uniform(0.0, 0.1); u_init_pert_amp = np.random.uniform(0.0, 0.05)
        k_init = np.random.randint(1, 3)
        rho0 = rho_init_mean + rho_init_pert_amp * np.sin(k_init * np.pi * x_grid / L); rho0 = np.maximum(rho0, 0.1) # Ensure positive density
        u0 = u_init_mean + u_init_pert_amp * np.cos(k_init * np.pi * x_grid / L)
        # --- BC parameters ---
        rho_l_base = np.random.uniform(0.9, 1.1); rho_l_amp = np.random.uniform(0.05, 0.15); rho_l_freq = np.random.uniform(1.0, 3.0) * 2 * np.pi / T
        u_l_base = np.random.uniform(0.1, 0.3)
        rho_r_base = np.random.uniform(0.9, 1.1); rho_r_amp = np.random.uniform(0.05, 0.15); rho_r_freq = np.random.uniform(1.0, 3.0) * 2 * np.pi / T
        u_r_nl_coeff = np.random.uniform(0.1, 0.4); u_r_base = np.random.uniform(-0.1, 0.1)
        # --- Control signals ---
        control_signals = np.zeros((nt, num_controls))
        for k in range(num_controls): # Smoothed random steps
            num_steps = np.random.randint(3, 6); step_times = np.sort(np.random.uniform(0, T*0.9, num_steps)); step_vals = np.random.uniform(-0.3, 0.3, num_steps)
            raw_signal = np.zeros(nt); current_val = np.random.uniform(-0.1, 0.1); t_idx = 0
            for step_t, step_v in zip(step_times, step_vals):
                while t_idx < nt and t_vals[t_idx] < step_t: raw_signal[t_idx] = current_val; t_idx += 1
                current_val += step_v
            while t_idx < nt: raw_signal[t_idx] = current_val; t_idx += 1
            smooth_window = max(1, nt // 20); control_signals[:, k] = np.convolve(raw_signal, np.ones(smooth_window)/smooth_window, mode='same')
        u_c1 = control_signals[:, 0] if num_controls >= 1 else np.zeros(nt)
        u_c2 = control_signals[:, 1] if num_controls >= 2 else np.zeros(nt)
        # --- Solve PDE ---
        rho = np.zeros((nt, nx)); u = np.zeros((nt, nx)); rho[0, :] = rho0; u[0, :] = u0
        BC_State = np.zeros((nt, 4)); BC_Control = np.zeros((nt, num_controls))
        rho_n = rho0.copy(); u_n = u0.copy(); m_n = rho_n * u_n
        valid_simulation = True
        max_speed_sound = 1.0 # Since dp/drho = 1
        
        for n in range(nt - 1):
            # Calculate actual BC values
            rho_l_state = rho_l_base + rho_l_amp * np.sin(rho_l_freq * t_vals[n]); u_l_state = u_l_base
            rho_l_actual = max(0.1, rho_l_state); u_l_actual = u_l_state + u_c1[n] # Ensure positive rho
            rho_r_state = rho_r_base + rho_r_amp * np.cos(rho_r_freq * t_vals[n])
            u_r_state = u_r_base - u_r_nl_coeff * np.tanh(5.0 * (rho_r_state - rho_r_base))
            rho_r_actual = max(0.1, rho_r_state); u_r_actual = u_r_state + u_c2[n] # Ensure positive rho
            BC_State[n, 0]=rho_l_state; BC_State[n, 1]=u_l_state; BC_State[n, 2]=rho_r_state; BC_State[n, 3]=u_r_state
            BC_Control[n, :] = control_signals[n, :]
            # Lax-Friedrichs Step
            rho_np1 = np.zeros(nx); u_np1 = np.zeros(nx); m_np1 = np.zeros(nx)
            f1 = m_n; f2 = m_n**2 / (rho_n + 1e-9) + rho_n
            rho_np1[1:-1] = 0.5*(rho_n[2:]+rho_n[:-2]) - 0.5*dt/dx*(f1[2:]-f1[:-2])
            m_np1[1:-1] = 0.5*(m_n[2:]+m_n[:-2]) - 0.5*dt/dx*(f2[2:]-f2[:-2])
            rho_np1[0] = rho_l_actual; u_np1[0] = u_l_actual; m_np1[0] = rho_np1[0]*u_np1[0]
            rho_np1[-1] = rho_r_actual; u_np1[-1] = u_r_actual; m_np1[-1] = rho_np1[-1]*u_np1[-1]
            rho_next_safe = np.maximum(rho_np1, 1e-6); u_np1 = m_np1 / rho_next_safe # Ensure positive rho before division
            # Stability Check
            max_wave_speed = np.max(np.abs(u_np1)) + max_speed_sound
            if max_wave_speed > dx / dt: print(f"    Euler CFL violation step {n} samp {s}. Speed={max_wave_speed:.2f} > dx/dt={dx/dt:.2f}. Skip."); valid_simulation=False; break
            if np.any(rho_np1 < 1e-6): print(f"    Neg density step {n} samp {s}. Skip."); valid_simulation=False; break
            if np.isnan(rho_np1).any() or np.isnan(u_np1).any() or np.max(np.abs(rho_np1))>1e3 or np.max(np.abs(u_np1))>1e3: print(f"    NaN/Inf/Large step {n} samp {s}. Skip."); valid_simulation=False; break
            rho_n=rho_np1; u_n=u_np1; m_n=rho_n*u_n; rho[n+1,:]=rho_n; u[n+1,:]=u_n

        if valid_simulation:
            BC_State[nt-1,:]=BC_State[nt-2,:]; BC_Control[nt-1,:]=BC_Control[nt-2,:]
            sample_params = {'rho_init_mean': rho_init_mean, 'u_init_mean': u_init_mean,'rho_l_base': rho_l_base, 'u_l_base': u_l_base,'rho_r_base': rho_r_base, 'u_r_nl_coeff': u_r_nl_coeff, 'nx': nx, 'nt': nt, 'T': T, 'L': L}
            sample = {'rho': rho.astype(np.float32), 'u': u.astype(np.float32), 'BC_State': BC_State.astype(np.float32), 'BC_Control': BC_Control.astype(np.float32), 'params': sample_params}
            data_list.append(sample)
            valid_samples_count += 1

    print(f"Finished Euler: Generated {valid_samples_count} valid samples out of {n_samples}.")
    return data_list

# --- 1.3 Burgers Generator ---
def generate_burgers_data_nonlinear_bc(n_samples=100, nx=128, nt=100, T=1.0, L=1.0, nu=0.01, num_controls=2):
    """ Generates Burgers' data with nonlinear/controlled BCs using CN/LF. """
    print(f"Generating Burgers data: n={n_samples}, nx={nx}, nt={nt}, T={T}, L={L}, nu={nu:.4f}, nc={num_controls}")
    data_list = []
    dt = T / (nt - 1)
    dx = L / (nx - 1)
    x_grid = np.linspace(0, L, nx)
    t_grid = np.linspace(0, T, nt)
    valid_samples_count = 0

    for s in range(n_samples):
        if (s + 1) % 50 == 0: print(f"  Burgers sample {s+1}/{n_samples}...")
        # --- Sample Parameters ---
        nu_sample = nu * np.random.uniform(0.5, 1.5)
        alpha_cn = nu_sample * dt / (2 * dx**2) # For CN diffusion
        A_diff = sp.diags([-alpha_cn, 1 + 2 * alpha_cn, -alpha_cn], [-1, 0, 1], shape=(nx, nx), format='csc') # CN LHS matrix part
        k_init = np.random.randint(1, 4); u0 = np.sin(k_init * np.pi * x_grid / L) * np.random.uniform(0.5, 1.5)
        A_l = np.random.uniform(0.1, 0.5); omega_l = np.random.uniform(1.0, 4.0) * 2 * np.pi / T
        B_l = np.random.uniform(0.05, 0.2) * np.sign(np.random.randn()) # u(dx)^2 coeff
        beta_r = np.random.uniform(0.1, 0.5); u_ref_r = np.random.uniform(-0.5, 0.5) # Robin params nu*ux+beta*(u^3-uref^3)=uc2
        # --- Control signals ---
        control_signals = np.zeros((nt, num_controls))
        for k in range(num_controls): # Smoothed random steps
            num_steps = np.random.randint(2, 5); step_times = np.sort(np.random.uniform(0, T*0.9, num_steps)); step_vals = np.random.uniform(-0.3, 0.3, num_steps)
            raw_signal = np.zeros(nt); current_val = np.random.uniform(-0.1, 0.1); t_idx = 0
            for step_t, step_v in zip(step_times, step_vals):
                while t_idx < nt and t_grid[t_idx] < step_t: raw_signal[t_idx] = current_val; t_idx += 1
                current_val += step_v
            while t_idx < nt: raw_signal[t_idx] = current_val; t_idx += 1
            smooth_window = max(1, nt // 20); control_signals[:, k] = np.convolve(raw_signal, np.ones(smooth_window)/smooth_window, mode='same')
        u_c1 = control_signals[:, 0] if num_controls >= 1 else np.zeros(nt)
        u_c2 = control_signals[:, 1] if num_controls >= 2 else np.zeros(nt)
        # --- Solve PDE ---
        U = np.zeros((nt, nx)); U[0, :] = u0; BC_State = np.zeros((nt, 2)); BC_Control = np.zeros((nt, num_controls))
        u_n = u0.copy(); valid_simulation = True
        for n in range(nt - 1):
            # Calculate BCs
            bc_l_state = A_l * (1 + 0.5*np.sin(1.5*omega_l*t_grid[n])) * np.sin(omega_l * t_grid[n]) + B_l * u_n[1]**2
            bc_l_actual = bc_l_state + u_c1[n]
            un_r = u_n[-1]; un_r_m1 = u_n[-2]
            # State approx for Robin BC (ignoring uc2 here for state storage)
            # nu*(u_N - u_{N-1})/dx approx -beta*(u_{N-1}^3 - u_ref^3) => u_N approx ...
            bc_r_state_approx = un_r_m1 + (dx / (nu_sample+1e-9)) * (-beta_r * (un_r_m1**3 - u_ref_r**3))
            # Actual BC for solver (using simple Dirichlet for stability in this generator)
            bc_r_actual_solver = bc_r_state_approx + u_c2[n] # Simple Dirichlet for solver
            BC_State[n, 0] = bc_l_state; BC_State[n, 1] = bc_r_state_approx
            BC_Control[n, :] = control_signals[n, :]

            # --- Finite Difference Step (Operator Splitting: Advection then Diffusion) ---
            # Advection Step (Lax-Friedrichs for u*u_x)
            u_adv_inter = u_n.copy()
            nonlin_flux = 0.5 * u_n**2
            u_adv_inter[1:-1] = 0.5*(u_n[2:] + u_n[:-2]) - dt/(2*dx)*(nonlin_flux[2:] - nonlin_flux[:-2])
            # Apply BCs after advection step (simple extrapolation for internal calculation)
            u_adv_inter[0] = bc_l_actual
            u_adv_inter[-1] = bc_r_actual_solver # Use solver BC

            # Diffusion Step (Crank-Nicolson)
            # RHS = (I + alpha*D2)u_adv_inter (where D2 is discrete Laplacian)
            rhs = u_adv_inter + alpha_cn * (np.roll(u_adv_inter, -1) - 2 * u_adv_inter + np.roll(u_adv_inter, 1))
            # Apply BCs to RHS and LHS matrix for CN
            rhs[0] = bc_l_actual # Next step's value
            rhs[-1] = bc_r_actual_solver # Next step's value
            A_solve = A_diff.copy() # LHS = (I - alpha*D2)
            A_solve[0, :] = 0; A_solve[0, 0] = 1 # Apply Dirichlet to LHS
            A_solve[-1, :] = 0; A_solve[-1, -1] = 1

            try: u_np1 = spsolve(A_solve, rhs)
            except Exception as e: print(f"    Burgers solver fail step {n} samp {s}: {e}. Skip."); valid_simulation=False; break
            if np.isnan(u_np1).any() or np.isinf(u_np1).any() or np.max(np.abs(u_np1)) > 1e4: # Divergence check
                 print(f"    Burgers unstable step {n} samp {s}. MaxVal={np.max(np.abs(u_np1)):.2e}. Skip."); valid_simulation=False; break
            u_n = u_np1; U[n+1, :] = u_n

        if valid_simulation:
            BC_State[nt-1,:]=BC_State[nt-2,:]; BC_Control[nt-1,:]=BC_Control[nt-2,:]
            sample_params = {'nu': nu_sample, 'k_init': k_init, 'A_l': A_l, 'omega_l': omega_l, 'B_l': B_l, 'beta_r': beta_r, 'u_ref_r': u_ref_r, 'nx': nx, 'nt': nt, 'T': T, 'L': L}
            sample = {'U': U.astype(np.float32), 'BC_State': BC_State.astype(np.float32), 'BC_Control': BC_Control.astype(np.float32), 'params': sample_params}
            data_list.append(sample)
            valid_samples_count += 1

    print(f"Finished Burgers: Generated {valid_samples_count} valid samples out of {n_samples}.")
    return data_list

# --- 1.4 Darcy Generator (Conceptual) ---
def generate_darcy_data_nonlinear_bc(
    n_samples=100, nx=64, ny=64, num_controls=2
,nt=300):
    """
    真实的一维 Darcy 数据生成器，并存储全部参数：
    - PDE: -d/dx( k(x) dp/dx ) = 0
    - 左右 Dirichlet 边界：p(0)=p_left_state+uc_left, p(1)=p_right_state+uc_right
    - k(x) 为随机对数正态场
    - 存储参数：k_params, f, PD_base, omega_D, beta_D, QN_base, omega_N, gamma_N, p_ref_N, nx, ny, nt, T
    """
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve

    print(f"Generating Darcy data (1D PDE solve): n={n_samples}, nx={nx}, ny={ny}, nt={nt}, nc={num_controls}")
    nt = nt
    T = 1.0
    L = 1.0
    t_vals = np.linspace(0, T, nt)
    dx = L / (nx - 1)

    data_list = []
    for s in range(n_samples):
        if (s + 1) % 20 == 0:
            print(f"  Darcy sample {s+1}/{n_samples}...")

        # 1) 随机参数
        f_val      = 1.0
        P_D_base   = np.random.uniform(1.0, 2.0)
        omega_D    = np.random.uniform(0.5, 2.0) * 2 * np.pi / T
        beta_D     = np.random.uniform(0.01, 0.05)
        QN_base    = np.random.uniform(-0.5, 0.5)
        omega_N    = np.random.uniform(0.5, 2.0) * 2 * np.pi / T
        gamma_N    = np.random.uniform(1.0, 5.0)
        p_ref_N    = np.random.uniform(0.0, 1.0)

        # 2) 渗透率 k(x)
        k_x    = np.exp(np.random.randn(nx) * 0.5)
        k_half = 0.5 * (k_x[:-1] + k_x[1:])

        # 3) 构造稀疏矩阵 A
        main_diag  = np.zeros(nx)
        lower_diag = np.zeros(nx - 1)
        upper_diag = np.zeros(nx - 1)
        main_diag[1:-1]   = - (k_half[:-1] + k_half[1:]) / dx**2
        lower_diag[:-1]   = k_half[:-1] / dx**2
        upper_diag[1:]    = k_half[1:]  / dx**2
        main_diag[0]      = 1.0
        main_diag[-1]     = 1.0
        A = diags([main_diag, lower_diag, upper_diag], [0, -1, +1], format="csc")

        # 4) 边界状态（不含控制）
        p_left_state  = P_D_base * np.cos(omega_D * t_vals) * (1 + 0.5 * np.sin(omega_D * t_vals))
        p_right_state = QN_base    * np.sin(omega_N * t_vals) * (1 + 0.3 * np.cos(omega_N * t_vals))

        # 5) 控制信号
        control_signals = np.zeros((nt, num_controls), dtype=np.float32)
        for k in range(num_controls):
            steps = np.random.randint(2, 5)
            times = np.sort(np.random.rand(steps) * 0.9 * T)
            vals  = np.random.uniform(-0.5, 0.5, size=steps)
            raw   = np.zeros(nt)
            idx = 0
            for t_step, v in zip(times, vals):
                while idx < nt and t_vals[idx] < t_step:
                    raw[idx] = raw[idx-1] if idx>0 else v
                    idx += 1
                raw[idx-1] = v
            raw[idx:] = raw[idx-1]
            window = max(1, nt//20)
            control_signals[:, k] = np.convolve(raw, np.ones(window)/window, mode="same")

        # 6) 处理当 num_controls < 1/2 的情况，确保 uc_left/right 始终为数组
        if num_controls >= 1:
            uc_left = control_signals[:, 0]
        else:
            uc_left = np.zeros(nt, dtype=np.float32)
        if num_controls >= 2:
            uc_right = control_signals[:, 1]
        else:
            uc_right = np.zeros(nt, dtype=np.float32)

        # 7) 时间步求解
        P_seq = np.zeros((nt, nx), dtype=np.float32)
        for n in range(nt):
            b = np.zeros(nx, dtype=np.float32)
            b[0]    = p_left_state[n]  + uc_left[n]
            b[-1]   = p_right_state[n] + uc_right[n]
            P_seq[n, :] = spsolve(A, b)

        # 8) 存样本，包含所有参数
        sample = {
            "P":          P_seq,
            "BC_State":   np.stack([p_left_state, p_right_state], axis=-1).astype(np.float32),
            "BC_Control": control_signals,
            "params": {
                "k_params": 'random',
                "f":        f_val,
                "PD_base":  P_D_base,
                "omega_D":  omega_D,
                "beta_D":   beta_D,
                "QN_base":  QN_base,
                "omega_N":  omega_N,
                "gamma_N":  gamma_N,
                "p_ref_N":  p_ref_N,
                "nx":       nx,
                "ny":       ny,
                "nt":       nt,
                "T":        T,
            }
        }
        data_list.append(sample)

    print(f"Finished Darcy: Generated {len(data_list)} valid samples.")
    return data_list


# =============================================================================
# 2. Main Execution Block
# =============================================================================
def save_data(data_list, filename):
    """ Saves the data list to a pickle file. """
    if not data_list:
        print(f"Warning: No data generated for {filename}. Skipping save.")
        return
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Successfully saved {len(data_list)} samples to {filename}")
    except Exception as e:
        print(f"Error saving data to {filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PDE datasets with complex BCs.")
    parser.add_argument('--dataset', type=str, required=True, choices=['advection', 'euler', 'burgers', 'darcy'], help='Type of dataset to generate.')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to generate.')
    parser.add_argument('--nx', type=int, default=64, help='Spatial resolution (nx for 1D, assumes ny=nx for 2D Darcy).')
    parser.add_argument('--nt', type=int, default=300, help='Temporal resolution.')
    parser.add_argument('--T', type=float, default=1.0, help='Final time.')
    parser.add_argument('--L', type=float, default=1.0, help='Domain length.')
    parser.add_argument('--num_controls', type=int, default=0, help='Number of control inputs.')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save the dataset file.')
    parser.add_argument('--filename', type=str, default=None, help='Output filename (e.g., advection_data.pkl). Default uses dataset type and sample count.')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine output filename
    if args.filename is None:
        output_filename = os.path.join(args.output_dir, f"{args.dataset}_data_{args.num_samples}s_{args.nx}nx_{args.nt}nt.pkl")
    else:
        output_filename = os.path.join(args.output_dir, args.filename)

    start_time = time.time()
    generated_data = None

    # Call the appropriate generator function
    if args.dataset == 'advection':
        generated_data = generate_pde_data_challenging_with_control(
            n_samples=args.num_samples, nx=args.nx, nt=args.nt, T=args.T, L=args.L,
            num_controls=args.num_controls, c=1.0, r=0.5 # Example c, r
        )
    elif args.dataset == 'euler':
         # Adjust nt, T for Euler if needed, default generator uses 200, 1.0
        euler_nt =  args.nt
        euler_T =  args.T
        print(f"Note: Using nt={euler_nt}, T={euler_T} for Euler generation.")
        generated_data = generate_euler_data_nonlinear_bc(
            n_samples=args.num_samples, nx=args.nx, nt=euler_nt, T=euler_T, L=args.L,
            num_controls=args.num_controls
        )
    elif args.dataset == 'burgers':
        burgers_nu = 0.01 / np.pi
        burgers_T = args.T # Burgers often run longer
        burgers_nt =  args.nt
        print(f"Note: Using nu={burgers_nu:.4f}, nt={burgers_nt}, T={burgers_T} for Burgers generation.")
        generated_data = generate_burgers_data_nonlinear_bc(
            n_samples=args.num_samples, nx=args.nx, nt=burgers_nt, T=burgers_T, L=args.L,
            nu=burgers_nu, num_controls=args.num_controls
        )
    elif args.dataset == 'darcy':
        # Assumes ny = nx for Darcy
        darcy_ny = args.nx
        darcy_nt = args.nt # Fewer 'time' steps for steady-state like problems
        print(f"Note: Using ny={darcy_ny}, nt={darcy_nt} for Darcy generation (Conceptual Solver).")
        generated_data = generate_darcy_data_nonlinear_bc(
            n_samples=args.num_samples, nx=args.nx, ny=darcy_ny,nt = darcy_nt,num_controls=args.num_controls
        )

    # Save the generated data
    save_data(generated_data, output_filename)

    end_time = time.time()
    print(f"\nTotal generation time: {end_time - start_time:.2f} seconds.")
    print(f"Dataset saved to: {output_filename}")



# I generate data set using following command line:
# python generate_datasets.py --dataset advection --num_samples 10000 --nx 128 --nt 600 --T 2 --num_controls 0 --output_dir ../datasets_full
# python generate_datasets.py --dataset euler --num_samples 10000  --nx 128 --nt 600 --T 2 --num_controls 0 --output_dir ../datasets_full
# python generate_datasets.py --dataset burgers --num_samples 10000  --nx 128 --nt 600 --T 2 --num_controls 0 --output_dir ../datasets_full
# python generate_datasets.py --dataset darcy --num_samples 10000  --nx 128 --nt 600 --T 2 --num_controls 0 --output_dir ../datasets_full
