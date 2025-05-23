import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import pickle
import time
import random
import os
import argparse
# fix seed
seed = 42
random.seed(seed)
np.random.seed(seed)
# --- Helper function to generate complex time-varying signals (adapted from your generate_datasets.py) ---
def generate_complex_signal(t_vals, T, num_components=3, max_freq=5, amp_range=(1, 2.0), noise_std=0.05):
    signal = np.zeros_like(t_vals)
    for _ in range(num_components):
        amp = np.random.uniform(*amp_range)
        freq = np.random.uniform(1.0, max_freq) * 2 * np.pi / T
        phase = np.random.uniform(0, 2 * np.pi)
        if np.random.rand() < 0.7: # Sinusoidal
            signal += amp * np.sin(freq * t_vals + phase)
        else: # Polynomial-like or step-like feature
            poly_order = np.random.randint(1,4)
            coeffs = np.random.randn(poly_order+1) * amp
            norm_t = (t_vals / T - 0.5) * 2 # Normalize t to [-1, 1] for polynomial stability
            for i_p, c_p in enumerate(coeffs):
                signal += c_p * (norm_t ** i_p)

    # Add some steps or pulses
    num_pulses = np.random.randint(0, 2)
    for _ in range(num_pulses):
        pulse_center = np.random.uniform(0.1*T, 0.9*T)
        pulse_width = np.random.uniform(0.05*T, 0.15*T)
        pulse_amp = np.random.uniform(-amp_range[1], amp_range[1])
        signal += pulse_amp * np.exp(-((t_vals - pulse_center)**2) / (2 * (pulse_width/3)**2))

    if noise_std > 0:
        signal += np.random.randn(*t_vals.shape) * noise_std
        
    # Smooth the signal slightly
    smooth_window = max(1, len(t_vals) // 50)
    if smooth_window > 1:
        signal = np.convolve(signal, np.ones(smooth_window)/smooth_window, mode='same')
    return signal


# --- Scenario 1: Heat Equation with Time-Delayed Integral Feedback Control ---
def generate_heat_delayed_feedback_data(
    n_samples=100, nx=64, nt=200, T=1.0, L=1.0, alpha_range=(0.001, 0.005),
    K_fb_range=(-5.0, 5.0), tau_delay_range=(0.05, 0.2), num_controls=2
):
    print(f"Generating Heat Delayed Feedback data: n={n_samples}, nx={nx}, nt={nt}, T={T}, L={L}, num_controls={num_controls}")
    data_list = []
    dt = T / (nt - 1)
    dx = L / (nx - 1)
    t_vals = np.linspace(0, T, nt)
    x_grid = np.linspace(0, L, nx)
    
    history_max_steps = int(max(tau_delay_range) / dt) + 2 # Max steps to store for delay

    for s in range(n_samples):
        if (s + 1) % (n_samples // 10 if n_samples >=10 else 1) == 0:
            print(f"  Heat Delayed Feedback sample {s+1}/{n_samples}...")

        alpha = np.random.uniform(*alpha_range)
        K_fb = np.random.uniform(*K_fb_range)
        tau_delay = np.random.uniform(*tau_delay_range)
        delay_steps = max(1, int(tau_delay / dt)) # Number of discrete time steps for delay

        # Initial condition
        u0_freq = np.random.uniform(1, 3)
        u0_amp = np.random.uniform(0.5, 1.5)
        u0 = u0_amp * np.sin(np.pi * u0_freq * x_grid / L) * np.exp(-((x_grid-L/2)**2)/(0.1*L**2))

        # Base boundary functions (non-linear, time-varying)
        g0_t_base = generate_complex_signal(t_vals, T, amp_range=(0.2, 0.8))
        gL_t_base = generate_complex_signal(t_vals, T, amp_range=(0.1, 0.5)) # Base for the controlled boundary

        # External control signals
        c0_t_control = generate_complex_signal(t_vals, T, amp_range=(0.1, 0.3), noise_std=0.02)
        cL_t_control = generate_complex_signal(t_vals, T, amp_range=(0.1, 0.3), noise_std=0.02)
        
        # Store all applied controls if num_controls matches, otherwise adjust
        all_controls_for_storage = np.zeros((nt, num_controls))
        if num_controls >= 1: all_controls_for_storage[:,0] = c0_t_control
        if num_controls >= 2: all_controls_for_storage[:,1] = cL_t_control
        # Fill remaining with zeros or other sampled signals if num_controls > 2

        U = np.zeros((nt, nx))
        U[0, :] = u0
        U_history = np.zeros((history_max_steps, nx)) # Store recent history for delayed feedback
        U_history[0,:] = u0 # Initialize history with u0

        BC_State = np.zeros((nt, 2)) # For [g0_t_actual_no_control, gL_t_base + Feedback_term_no_cL_control]
        
        # Crank-Nicolson setup
        r = alpha * dt / (2 * dx**2)
        main_diag = np.ones(nx) * (1 + 2 * r)
        off_diag = np.ones(nx - 1) * (-r)
        A_cn = sp.diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csc')
        
        valid_sim = True
        for n in range(nt - 1):
            # Determine history index for delayed feedback
            # current_history_idx is (n % history_max_steps)
            # delayed_history_idx is ((n - delay_steps) % history_max_steps)
            # Ensure n - delay_steps is non-negative, otherwise use earliest available history (or u0)
            
            idx_for_delay = max(0, n - delay_steps) # Time step index from which to get u for feedback
            # Retrieve u(x, t-tau) from stored history. If n < delay_steps, use U[0]
            u_delayed = U[idx_for_delay, :]

            integral_feedback_val = K_fb * np.trapz(u_delayed, dx=dx)
            
            # Actual boundary values for solver
            u0_actual_solver = g0_t_base[n] + (c0_t_control[n] if num_controls >= 1 else 0.0)
            uL_actual_solver = gL_t_base[n] + integral_feedback_val + (cL_t_control[n] if num_controls >= 2 else 0.0)

            # Store BC_State terms (base + feedback, before external control)
            BC_State[n, 0] = g0_t_base[n]
            BC_State[n, 1] = gL_t_base[n] + integral_feedback_val
            
            # RHS for Crank-Nicolson
            b_rhs = U[n,:].copy()
            b_rhs[1:-1] = U[n,1:-1] + r * (U[n,2:] - 2*U[n,1:-1] + U[n,:-2])
            
            # Apply actual BCs for the (n+1) step to A_cn and b_rhs
            A_mod = A_cn.copy().tolil() # Modifiable copy
            
            # For u(0,t) = u0_actual_solver
            A_mod[0,0] = 1; A_mod[0,1] = 0
            b_rhs[0] = u0_actual_solver # This is for U[n+1,0]
            
            # For u(L,t) = uL_actual_solver
            A_mod[nx-1,nx-1] = 1; A_mod[nx-1,nx-2] = 0
            b_rhs[nx-1] = uL_actual_solver # This is for U[n+1, L]

            try:
                U[n+1, :] = spsolve(A_mod.tocsc(), b_rhs)
            except Exception as e:
                print(f"    Solver failed at step {n} for sample {s} (Heat Delayed): {e}. Skipping sample."); valid_sim=False; break
            
            if np.any(np.isnan(U[n+1,:])) or np.any(np.isinf(U[n+1,:])) or np.max(np.abs(U[n+1,:])) > 1e6:
                print(f"    Unstable simulation at step {n} for sample {s} (Heat Delayed). Max val: {np.max(np.abs(U[n+1,:])):.2e}. Skipping."); valid_sim=False; break
            
            # Update history (simple circular buffer idea, or just use U itself if history_max_steps is large enough for U)
            # For simplicity here, we directly index U, assuming nt is large enough relative to delay_steps for most of the simulation.
            # A more robust history buffer would be U_history[(n+1) % history_max_steps, :] = U[n+1, :]

        if valid_sim:
            # Fill last BC_State (often done by repeating previous)
            BC_State[nt-1, :] = BC_State[nt-2, :]
            
            sample_params = {'alpha': alpha, 'K_fb': K_fb, 'tau_delay': tau_delay, 'L':L, 'T':T, 'nx':nx, 'nt':nt}
            sample_data = {'U': U.astype(np.float32),
                           'BC_State': BC_State.astype(np.float32),
                           'BC_Control': all_controls_for_storage.astype(np.float32),
                           'params': sample_params}
            data_list.append(sample_data)

    print(f"Finished Heat Delayed Feedback: Generated {len(data_list)} valid samples out of {n_samples}.")
    return data_list


# --- Scenario 2: Reaction-Diffusion Equation with Integral Feedback on Neumann BC ---
def generate_reaction_diffusion_neumann_feedback_data(
    n_samples=100, nx=64, nt=200, T=1.0, L=1.0, alpha_range=(0.005, 0.02),
    mu_reaction_range=(0.5, 2.0), K_fb_range=(-1.0, 1.0), num_controls=2
):
    print(f"Generating Reaction-Diffusion Neumann Feedback data: n={n_samples}, nx={nx}, nt={nt}, T={T}, L={L}, num_controls={num_controls}")
    data_list = []
    dt = T / (nt - 1)
    dx = L / (nx - 1)
    t_vals = np.linspace(0, T, nt)
    x_grid = np.linspace(0, L, nx)

    for s in range(n_samples):
        if (s + 1) % (n_samples // 10 if n_samples >=10 else 1) == 0:
            print(f"  Reaction-Diffusion sample {s+1}/{n_samples}...")

        alpha = np.random.uniform(*alpha_range)
        mu_reaction = np.random.uniform(*mu_reaction_range)
        K_fb = np.random.uniform(*K_fb_range)

        u0_freq = np.random.uniform(1, 2)
        u0_amp = np.random.uniform(0.3, 0.8)
        u0 = u0_amp * np.sin(np.pi * u0_freq * x_grid / L)**2 # Ensure positive for typical reaction terms

        g0_t_base = generate_complex_signal(t_vals, T, amp_range=(0.2, 0.7)) # Base for Dirichlet at x=0
        gL_flux_base = generate_complex_signal(t_vals, T, amp_range=(-0.5, 0.5)) # Base for Neumann flux at x=L

        c0_t_control = generate_complex_signal(t_vals, T, amp_range=(0.1, 0.2), noise_std=0.01)
        cL_flux_control = generate_complex_signal(t_vals, T, amp_range=(0.1, 0.3), noise_std=0.01)
        
        all_controls_for_storage = np.zeros((nt, num_controls))
        if num_controls >= 1: all_controls_for_storage[:,0] = c0_t_control
        if num_controls >= 2: all_controls_for_storage[:,1] = cL_flux_control

        U = np.zeros((nt, nx))
        U[0, :] = u0
        
        BC_State = np.zeros((nt, 2)) # For [g0_t_base, gL_flux_base + Feedback_term_flux]

        # Crank-Nicolson for diffusion part
        r_diff = alpha * dt / (2 * dx**2)
        main_diag_diff = np.ones(nx) * (1 + 2 * r_diff)
        off_diag_diff = np.ones(nx - 1) * (-r_diff)
        A_cn_diff = sp.diags([off_diag_diff, main_diag_diff, off_diag_diff], [-1, 0, 1], format='csc')

        valid_sim = True
        for n in range(nt - 1):
            # --- Step 1: Explicit Reaction Step (Euler) ---
            u_reacted = U[n,:] + dt * (mu_reaction * U[n,:] * (1 - U[n,:]))
            # Apply physical constraints if necessary (e.g., positivity)
            u_reacted = np.clip(u_reacted, 0, 1.5) # Example clip for u(1-u) type reaction

            # --- Step 2: Implicit Diffusion Step (Crank-Nicolson) on u_reacted ---
            integral_feedback_val = K_fb * np.trapz(U[n,:], dx=dx) # Feedback based on U[n] for flux at n+1

            # Actual boundary values/fluxes for solver
            u0_actual_solver = g0_t_base[n] + (c0_t_control[n] if num_controls >= 1 else 0.0)
            # Neumann: alpha * du/dx(L,t) = flux_val --> alpha * (U[N-1] - U[N-2])/dx = flux_val
            # U[N-1] = U[N-2] + (dx/alpha) * flux_val
            neumann_flux_actual_solver = gL_flux_base[n] + integral_feedback_val + (cL_flux_control[n] if num_controls >= 2 else 0.0)

            BC_State[n, 0] = g0_t_base[n]
            BC_State[n, 1] = gL_flux_base[n] + integral_feedback_val # Store base flux + feedback flux

            # RHS for Crank-Nicolson on u_reacted
            b_rhs_diff = u_reacted.copy()
            b_rhs_diff[1:-1] = u_reacted[1:-1] + r_diff * (u_reacted[2:] - 2*u_reacted[1:-1] + u_reacted[:-2])
            
            A_mod_diff = A_cn_diff.copy().tolil()
            
            # Dirichlet at x=0
            A_mod_diff[0,0] = 1; A_mod_diff[0,1] = 0
            b_rhs_diff[0] = u0_actual_solver

            # Neumann at x=L for U[n+1]
            # (U[n+1,nx-1] - U[n+1,nx-2])/dx = neumann_flux_actual_solver / alpha
            # A_cn matrix equation for last row: -r U[n+1,nx-2] + (1+2r)U[n+1,nx-1] = b_rhs_diff[nx-1] (approx)
            # Modify A_mod_diff and b_rhs_diff for Neumann
            A_mod_diff[nx-1, nx-1] = 1 + r_diff # Corresponds to U_N
            A_mod_diff[nx-1, nx-2] = - (1 + r_diff) # Corresponds to -U_{N-1}
            # Effective RHS part for Neumann: (dx/alpha)*neumann_flux_actual_solver * (appropriate_coeff_from_CN)
            # Simplified approximation for solver: (implicit scheme makes exact specification complex for this form)
            # We directly enforce U[nx-1] using the flux: U[nx-1] = U_prev_at_L + dt * du/dt_from_flux_approx
            # For CN, it's more involved. Let's use finite difference for Neumann term in the CN system.
            # -(r)u_{N-2}^{n+1} + (1+r)u_{N-1}^{n+1} = b_rhs_neumann_part + (dx/alpha)*neumann_flux_actual_solver * r
            # Original last row: -r U_{N-2} + (1+2r)U_{N-1} = current_b_rhs[N-1]
            # New: A[N-1, N-2] = -1/dx; A[N-1,N-1] = 1/dx; b[N-1] = neumann_flux_actual_solver / alpha (for forward Euler like implementation of BC)
            # For CN, this needs careful discretization of the flux term.
            # For simplicity in this generator, approximating with a type of Robin:
            A_mod_diff[nx-1, nx-1] = 1 + r_diff + r_diff * dx # (1+2r)u_L - r u_{L-1}  ; Adjusting the diagonal
            A_mod_diff[nx-1, nx-2] = -r_diff
            b_rhs_diff[nx-1] += r_diff * dx * (neumann_flux_actual_solver / alpha)


            try:
                U[n+1, :] = spsolve(A_mod_diff.tocsc(), b_rhs_diff)
            except Exception as e:
                print(f"    Solver failed at step {n} for sample {s} (Reaction-Diff): {e}. Skipping sample."); valid_sim=False; break
            
            if np.any(np.isnan(U[n+1,:])) or np.any(np.isinf(U[n+1,:])) or np.max(np.abs(U[n+1,:])) > 1e3:
                print(f"    Unstable simulation at step {n} for sample {s} (Reaction-Diff). Max val: {np.max(np.abs(U[n+1,:])):.2e}. Skipping."); valid_sim=False; break

        if valid_sim:
            BC_State[nt-1, :] = BC_State[nt-2, :]
            sample_params = {'alpha': alpha, 'mu_reaction': mu_reaction, 'K_fb_neumann': K_fb, 'L':L, 'T':T, 'nx':nx, 'nt':nt}
            sample_data = {'U': U.astype(np.float32),
                           'BC_State': BC_State.astype(np.float32),
                           'BC_Control': all_controls_for_storage.astype(np.float32),
                           'params': sample_params}
            data_list.append(sample_data)

    print(f"Finished Reaction-Diffusion Neumann Feedback: Generated {len(data_list)} valid samples out of {n_samples}.")
    return data_list


# --- Scenario 3: Heat Equation with Non-linear Feedback Gain on Boundary ---
def generate_heat_nonlinear_feedback_gain_data(
    n_samples=100, nx=64, nt=200, T=1.0, L=1.0, alpha_range=(0.01, 0.05),
    K1_fb_range=(0.5, 1.5), K2_fb_range=(-0.5, 0.5), num_controls=2 # K2 for s^2 term
):
    print(f"Generating Heat Non-linear Gain Feedback data: n={n_samples}, nx={nx}, nt={nt}, T={T}, L={L}, num_controls={num_controls}")
    data_list = []
    dt = T / (nt - 1)
    dx = L / (nx - 1)
    t_vals = np.linspace(0, T, nt)
    x_grid = np.linspace(0, L, nx)

    for s in range(n_samples):
        if (s + 1) % (n_samples // 10 if n_samples >=10 else 1) == 0:
            print(f"  Heat Non-linear Gain sample {s+1}/{n_samples}...")

        alpha = np.random.uniform(*alpha_range)
        K1_fb = np.random.uniform(*K1_fb_range)
        K2_fb = np.random.uniform(*K2_fb_range) # Coefficient for the s^2 term in feedback

        u0_freq = np.random.uniform(1, 3)
        u0_amp = np.random.uniform(0.5, 1.5)
        u0 = u0_amp * np.sin(np.pi * u0_freq * x_grid / L)

        g0_t_base = generate_complex_signal(t_vals, T, amp_range=(0.2, 0.8))
        gL_t_base = generate_complex_signal(t_vals, T, amp_range=(0.1, 0.4)) # Base for the controlled boundary

        c0_t_control = generate_complex_signal(t_vals, T, amp_range=(0.1, 0.3), noise_std=0.02)
        cL_t_control = generate_complex_signal(t_vals, T, amp_range=(0.1, 0.3), noise_std=0.02)

        all_controls_for_storage = np.zeros((nt, num_controls))
        if num_controls >= 1: all_controls_for_storage[:,0] = c0_t_control
        if num_controls >= 2: all_controls_for_storage[:,1] = cL_t_control
        
        U = np.zeros((nt, nx))
        U[0, :] = u0
        BC_State = np.zeros((nt, 2))
        
        r = alpha * dt / (2 * dx**2)
        main_diag = np.ones(nx) * (1 + 2 * r)
        off_diag = np.ones(nx - 1) * (-r)
        A_cn = sp.diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csc')
        
        valid_sim = True
        for n in range(nt - 1):
            s_integral = np.trapz(U[n,:], dx=dx)
            # Non-linear feedback function F(s) = K1*s + K2*s^2
            nonlinear_feedback_val = K1_fb * s_integral + K2_fb * (s_integral**2)
            
            u0_actual_solver = g0_t_base[n] + (c0_t_control[n] if num_controls >= 1 else 0.0)
            uL_actual_solver = gL_t_base[n] + nonlinear_feedback_val + (cL_t_control[n] if num_controls >= 2 else 0.0)

            BC_State[n, 0] = g0_t_base[n]
            BC_State[n, 1] = gL_t_base[n] + nonlinear_feedback_val
            
            b_rhs = U[n,:].copy()
            b_rhs[1:-1] = U[n,1:-1] + r * (U[n,2:] - 2*U[n,1:-1] + U[n,:-2])
            
            A_mod = A_cn.copy().tolil()
            A_mod[0,0] = 1; A_mod[0,1] = 0; b_rhs[0] = u0_actual_solver
            A_mod[nx-1,nx-1] = 1; A_mod[nx-1,nx-2] = 0; b_rhs[nx-1] = uL_actual_solver
            
            try:
                U[n+1, :] = spsolve(A_mod.tocsc(), b_rhs)
            except Exception as e:
                print(f"    Solver failed at step {n} for sample {s} (Heat NonlinGain): {e}. Skipping sample."); valid_sim=False; break

            if np.any(np.isnan(U[n+1,:])) or np.any(np.isinf(U[n+1,:])) or np.max(np.abs(U[n+1,:])) > 1e6:
                print(f"    Unstable simulation at step {n} for sample {s} (Heat NonlinGain). Max val: {np.max(np.abs(U[n+1,:])):.2e}. Skipping."); valid_sim=False; break
        
        if valid_sim:
            BC_State[nt-1, :] = BC_State[nt-2, :]
            sample_params = {'alpha': alpha, 'K1_fb': K1_fb, 'K2_fb': K2_fb, 'L':L, 'T':T, 'nx':nx, 'nt':nt}
            sample_data = {'U': U.astype(np.float32),
                           'BC_State': BC_State.astype(np.float32),
                           'BC_Control': all_controls_for_storage.astype(np.float32),
                           'params': sample_params}
            data_list.append(sample_data)

    print(f"Finished Heat Non-linear Gain Feedback: Generated {len(data_list)} valid samples out of {n_samples}.")
    return data_list


def generate_convdiff_feedback_dataset(
    n_samples: int,
    nx: int,
    nt: int,
    L: float,
    T: float,
    a_min: float,
    a_max: float,
    D_min: float,
    D_max: float,
    K_I_min: float,
    K_I_max: float,
) -> list:
    """
    Generate a dataset of 1D convection–diffusion solutions with integral feedback control at the boundaries.

    PDE: u_t + a * u_x = D * u_xx,
    BCs: u(0,t) = r_left(t) + c_left(t),  u(L,t) = r_right(t) + c_right(t)
    Control: c_{\{left,right\}}(t) = K_I * \int_0^t [r(tau) - u(boundary,tau)] d\tau

    Returns:
        data_list: list of dicts, each with keys:
            'U'         : ndarray(nt, nx) solution matrix
            'BC_State'  : ndarray(nt, 2) [u(0,t), u(L,t)]
            'BC_Control': ndarray(nt, 2) control signals [c_left, c_right]
            'params'    : dict of sample parameters {L, T, a, D, K_I}
    """
    # Physical & numerical setup
    dx = L / (nx - 1)
    dt = T / (nt - 1)

    data_list = []
    rng = np.random.default_rng()

    for _ in range(n_samples):
        # Sample physics & control parameters
        a = rng.uniform(a_min, a_max)
        D = rng.uniform(D_min, D_max)
        K_I = rng.uniform(K_I_min, K_I_max)

        # Spatial & temporal grids
        x = np.linspace(0, L, nx)
        t = np.linspace(0, T, nt)

        # Nonlinear time‐varying reference trajectories at boundaries
        r_left = (
            1.0 + 0.5 * np.sin(2*np.pi * t / T + rng.uniform(0, 2*np.pi))
            + 0.2 * (t/T)**2
        )
        r_right = (
            0.5 * np.cos(3*2*np.pi * t / T + rng.uniform(0, 2*np.pi))
            + 0.3 * np.sin((t/T)**3)
        )

        # Pre‐allocate arrays
        U = np.zeros((nt, nx))
        BC_State = np.zeros((nt, 2))
        BC_Control = np.zeros((nt, 2))

        # Initial condition: smooth sinusoid + small noise
        u0 = (
            1.0 * np.sin(np.pi * x / L)
            + 0.1 * rng.standard_normal(nx)
        )
        U[0, :] = u0
        BC_State[0, :] = [u0[0], u0[-1]]

        # Build implicit diffusion matrix for interior nodes
        r_diff = D * dt / dx**2
        main_diag = (1 + 2*r_diff) * np.ones(nx-2)
        off_diag = -r_diff * np.ones(nx-3)
        A = sp.diags(
            diagonals=[off_diag, main_diag, off_diag],
            offsets=[-1, 0, 1],
            format='csc'
        )

        # Integral errors
        e_left = 0.0
        e_right = 0.0

        # Time‐stepping loop
        for n in range(1, nt):
            # accumulate integral of error
            e_left += (r_left[n-1] - U[n-1, 0]) * dt
            e_right += (r_right[n-1] - U[n-1, -1]) * dt
            # compute control actions
            c_left = K_I * e_left
            c_right = K_I * e_right
            BC_Control[n, :] = [c_left, c_right]

            # copy previous state and apply Dirichlet BCs
            u_prev = U[n-1, :].copy()
            u_prev[0] = r_left[n-1] + c_left
            u_prev[-1] = r_right[n-1] + c_right

            # 1) Advection step (upwind scheme)
            u_adv = u_prev.copy()
            if a >= 0:
                u_adv[1:] = (
                    u_prev[1:] - a * dt/dx * (u_prev[1:] - u_prev[:-1])
                )
            else:
                u_adv[:-1] = (
                    u_prev[:-1] - a * dt/dx * (u_prev[1:] - u_prev[:-1])
                )
            # maintain BCs
            u_adv[0], u_adv[-1] = u_prev[0], u_prev[-1]

            # 2) Diffusion step (implicit)
            b = u_adv[1:-1].copy()
            # incorporate Dirichlet BC into RHS
            b[0]   += r_diff * u_prev[0]
            b[-1]  += r_diff * u_prev[-1]
            u_inner = spsolve(A, b)

            # assemble new state
            u_new = np.zeros_like(u_prev)
            u_new[1:-1] = u_inner
            u_new[0], u_new[-1] = u_prev[0], u_prev[-1]

            # store
            U[n, :] = u_new
            BC_State[n, :] = [u_new[0], u_new[-1]]

        # package sample
        params = {'L': L, 'T': T, 'a': a, 'D': D, 'K_I': K_I}
        sample = {
            'U': U,
            'BC_State': BC_State,
            'BC_Control': BC_Control,
            'params': params
        }
        data_list.append(sample)

    return data_list



# --- Main execution block for dataset generation (similar to your generate_datasets.py) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PDE datasets with integral feedback control.")
    parser.add_argument('--dataset_type', type=str, required=True,
                        choices=['heat_delayed_feedback',
                                 'reaction_diffusion_neumann_feedback',
                                 'heat_nonlinear_feedback_gain', 'convdiff'],
                        help='Type of dataset to generate.')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples.') # Reduced default for quick test
    parser.add_argument('--nx', type=int, default=64, help='Spatial resolution.')
    parser.add_argument('--nt', type=int, default=200, help='Temporal resolution.')
    parser.add_argument('--T', type=float, default=1.0, help='Final time.')
    parser.add_argument('--L', type=float, default=1.0, help='Domain length.')
    parser.add_argument('--a_min',     type=float, default=0.5)
    parser.add_argument('--a_max',     type=float, default=2.0)
    parser.add_argument('--D_min',     type=float, default=0.01)
    parser.add_argument('--D_max',     type=float, default=0.1)
    parser.add_argument('--K_I_min',   type=float, default=0.0)
    parser.add_argument('--K_I_max',   type=float, default=1.0)
    parser.add_argument('--num_controls', type=int, default=2, help='Number of external control signals.')
    parser.add_argument('--output_dir', type=str, default='./datasets_new_feedback', help='Directory to save datasets.')
    parser.add_argument('--filename_suffix', type=str, default='', help='Suffix for the output filename.')


    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_filename = os.path.join(
        args.output_dir,
        f"{args.dataset_type}{args.filename_suffix}_{args.num_samples}s_{args.nx}nx_{args.nt}nt.pkl"
    )

    start_time_gen = time.time()
    generated_data_list = []

    if args.dataset_type == 'heat_delayed_feedback':
        generated_data_list = generate_heat_delayed_feedback_data(
            n_samples=args.num_samples, nx=args.nx, nt=args.nt, T=args.T, L=args.L,
            num_controls=args.num_controls
        )
    elif args.dataset_type == 'reaction_diffusion_neumann_feedback':
        generated_data_list = generate_reaction_diffusion_neumann_feedback_data(
            n_samples=args.num_samples, nx=args.nx, nt=args.nt, T=args.T, L=args.L,
            num_controls=args.num_controls
        )
    elif args.dataset_type == 'heat_nonlinear_feedback_gain':
        generated_data_list = generate_heat_nonlinear_feedback_gain_data(
            n_samples=args.num_samples, nx=args.nx, nt=args.nt, T=args.T, L=args.L,
            num_controls=args.num_controls
        )
    elif args.dataset_type == 'convdiff':
        generated_data_list = generate_convdiff_feedback_dataset(
                args.num_samples, args.nx, args.nt,
                args.L, args.T,
                args.a_min, args.a_max,
                args.D_min, args.D_max,
                args.K_I_min, args.K_I_max
            )    
    else:
        print(f"Unknown dataset type: {args.dataset_type}")
        exit()

    if generated_data_list:
        with open(output_filename, 'wb') as f:
            pickle.dump(generated_data_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Successfully saved {len(generated_data_list)} samples to {output_filename}")
    else:
        print(f"No data generated for {args.dataset_type}. File not saved.")

    end_time_gen = time.time()
    print(f"Total generation time for {args.dataset_type}: {end_time_gen - start_time_gen:.2f} seconds.")


# I generate dataset using following command lines:

# # convdiff
# python generate_integral_feedback_datasets.py --dataset_type convdiff --num_samples 5000 --nx 64 --nt 300 --T 2.0 --output_dir ./datasets_new_feedback --filename_suffix _v1

# # Generate Reaction-Diffusion with Neumann Integral Feedback
# python generate_feedbackdata.py --dataset_type reaction_diffusion_neumann_feedback --num_samples 5000 --nx 64 --nt 300 --T 2.0 --output_dir ./datasets_integral_feedback --filename_suffix _v1

# # Generate Heat Equation with Non-linear Feedback Gain
# python generate_feedbackdata.py --dataset_type heat_nonlinear_feedback_gain --num_samples 5000 --nx 64 --nt 300 --num_controls 1 --T 2.0 --output_dir ./datasets_integral_feedback --filename_suffix _v1
