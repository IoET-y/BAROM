nohup python BAROM_poddim.py --datatype reaction_diffusion_neumann_feedback --poddim 8 > out_BAROM_reaction_diffusion_neumann_feedback_8.log 2>&1 & 

nohup python BAROM_poddim.py --datatype reaction_diffusion_neumann_feedback --poddim 16 > out_BAROM_reaction_diffusion_neumann_feedback_16.log 2>&1 & 

nohup python BAROM_poddim.py --datatype reaction_diffusion_neumann_feedback --poddim 24 > out_BAROM_reaction_diffusion_neumann_feedback_24.log 2>&1 & 

python BAROM_fixedlifting.py --datatype reaction_diffusion_neumann_feedback --use_fixed_lifting > out_BAROM_reaction_diffusion_neumann_feedback_32_fixedlifting.log 2>&1 & 

python BAROM_Random_pod.py --datatype reaction_diffusion_neumann_feedback --random_phi_init > out_BAROM_reaction_diffusion_neumann_feedback_32_random_pod.log 2>&1 & 


python BAROM_Non_attention.py --datatype reaction_diffusion_neumann_feedback --model_variant explicit_bc_no_attn \
                         --basis_dim 32 \
                         --d_model 512 \
                         --bc_processed_dim 32 \
                         --hidden_bc_processor_dim 128  > out_EBC_Noattn_reaction_diffusion_neumann_feedback.log 2>&1 &

python BAROM_Non_attention.py --datatype reaction_diffusion_neumann_feedback --model_variant implicit_bc_no_attn \
                         --basis_dim 32 \
                         --d_model 512 \
                         --bc_processed_dim 32 \
                         --hidden_bc_processor_dim 128  > out_IBC_Noattn_reaction_diffusion_neumann_feedback.log 2>&1 &






--- Aggregated Results for Dataset: REACTION_DIFFUSION_NEUMANN_FEEDBACK ---    ALbation


ablation 1:    different pod_dim 8 16 24 32
  Model: ROM_Attention_Poddim8
    Horizon T=1.50:
      U: Avg MSE=2.507e-02, Avg RMSE=1.512e-01, Avg RelErr=2.833e-01, Avg MaxErr=9.239e-01 (from 1000 samples)
    Horizon T=1.75:
      U: Avg MSE=2.572e-02, Avg RMSE=1.535e-01, Avg RelErr=2.737e-01, Avg MaxErr=9.367e-01 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=2.645e-02, Avg RMSE=1.560e-01, Avg RelErr=2.658e-01, Avg MaxErr=9.604e-01 (from 1000 samples)
  Model: ROM_Attention_Poddim16
    Horizon T=1.50:
      U: Avg MSE=2.482e-02, Avg RMSE=1.506e-01, Avg RelErr=2.831e-01, Avg MaxErr=9.364e-01 (from 1000 samples)
    Horizon T=1.75:
      U: Avg MSE=2.549e-02, Avg RMSE=1.531e-01, Avg RelErr=2.740e-01, Avg MaxErr=9.507e-01 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=2.617e-02, Avg RMSE=1.556e-01, Avg RelErr=2.663e-01, Avg MaxErr=9.714e-01 (from 1000 samples)
  Model: ROM_Attention_Poddim24
    Horizon T=1.50:
      U: Avg MSE=2.537e-02, Avg RMSE=1.526e-01, Avg RelErr=2.865e-01, Avg MaxErr=9.203e-01 (from 1000 samples)
    Horizon T=1.75:
      U: Avg MSE=2.616e-02, Avg RMSE=1.554e-01, Avg RelErr=2.777e-01, Avg MaxErr=9.333e-01 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=2.685e-02, Avg RMSE=1.579e-01, Avg RelErr=2.696e-01, Avg MaxErr=9.533e-01 (from 1000 samples)
  Model: ROM_Attention_Poddim32
    Horizon T=1.5:
    U: Avg MSE=2.4586e-02, Avg RMSE=1.5018e-01, Avg RelErr=2.8193e-01, Avg MaxErr=9.3765e-01 (from 1000 samples)
    Horizon T=1.75:
      U: Avg MSE=2.5283e-02, Avg RMSE=1.5256e-01, Avg RelErr=2.7226e-01, Avg MaxErr=9.5210e-01 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=2.6157e-02, Avg RMSE=1.5545e-01, Avg RelErr=2.6475e-01, Avg MaxErr=9.7699e-01 (from 1000 samples)

ablation 2:   with Lifting Network or Not
  Model: ROM_Attention_FixedLift
    Horizon T=1.50:
      U: Avg MSE=7.348e-02, Avg RMSE=2.645e-01, Avg RelErr=4.948e-01, Avg MaxErr=1.108e+00 (from 1000 samples)
    Horizon T=1.75:
      U: Avg MSE=7.461e-02, Avg RMSE=2.669e-01, Avg RelErr=4.759e-01, Avg MaxErr=1.143e+00 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=7.930e-02, Avg RMSE=2.752e-01, Avg RelErr=4.693e-01, Avg MaxErr=1.200e+00 (from 1000 samples)
  Model: ROM_Attention_LiftNet
    Horizon T=1.5:
    U: Avg MSE=2.4586e-02, Avg RMSE=1.5018e-01, Avg RelErr=2.8193e-01, Avg MaxErr=9.3765e-01 (from 1000 samples)
    Horizon T=1.75:
      U: Avg MSE=2.5283e-02, Avg RMSE=1.5256e-01, Avg RelErr=2.7226e-01, Avg MaxErr=9.5210e-01 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=2.6157e-02, Avg RMSE=1.5545e-01, Avg RelErr=2.6475e-01, Avg MaxErr=9.7699e-01 (from 1000 samples)




ablation 3: physicals initialize Phi vs Random initialize Phi
  Model: ROM_Attention_RandomPhi
    Horizon T=1.50:
      U: Avg MSE=2.374e-02, Avg RMSE=1.465e-01, Avg RelErr=2.750e-01, Avg MaxErr=9.160e-01 (from 1000 samples)
    Horizon T=1.75:
      U: Avg MSE=2.421e-02, Avg RMSE=1.483e-01, Avg RelErr=2.649e-01, Avg MaxErr=9.313e-01 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=2.477e-02, Avg RMSE=1.503e-01, Avg RelErr=2.566e-01, Avg MaxErr=9.554e-01 (from 1000 samples)
  Model: ROM_Attention_PhysPhi
    Horizon T=1.5:
    U: Avg MSE=2.4586e-02, Avg RMSE=1.5018e-01, Avg RelErr=2.8193e-01, Avg MaxErr=9.3765e-01 (from 1000 samples)
    Horizon T=1.75:
      U: Avg MSE=2.5283e-02, Avg RMSE=1.5256e-01, Avg RelErr=2.7226e-01, Avg MaxErr=9.5210e-01 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=2.6157e-02, Avg RMSE=1.5545e-01, Avg RelErr=2.6475e-01, Avg MaxErr=9.7699e-01 (from 1000 samples)



ablation 4:  Boundary awared attention transformer vs LSTM with implicit BC_control input or explicit BC_control input
  Model: Attn_ImpBC_b32
    Horizon T=1.5:
    U: Avg MSE=2.4586e-02, Avg RMSE=1.5018e-01, Avg RelErr=2.8193e-01, Avg MaxErr=9.3765e-01 (from 1000 samples)
    Horizon T=1.75:
      U: Avg MSE=2.5283e-02, Avg RMSE=1.5256e-01, Avg RelErr=2.7226e-01, Avg MaxErr=9.5210e-01 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=2.6157e-02, Avg RMSE=1.5545e-01, Avg RelErr=2.6475e-01, Avg MaxErr=9.7699e-01 (from 1000 samples)
  Model: LSTM_ImpBC_b32
    Horizon T=1.50:
      U: Avg MSE=2.587e-02, Avg RMSE=1.534e-01, Avg RelErr=2.881e-01, Avg MaxErr=9.496e-01 (from 1000 samples)
    Horizon T=1.75:
      U: Avg MSE=2.637e-02, Avg RMSE=1.553e-01, Avg RelErr=2.776e-01, Avg MaxErr=9.610e-01 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=2.679e-02, Avg RMSE=1.569e-01, Avg RelErr=2.680e-01, Avg MaxErr=9.895e-01 (from 1000 samples)
  Model: LSTM_ExpBC_b32
    Horizon T=1.50:
      U: Avg MSE=1.197e-02, Avg RMSE=1.055e-01, Avg RelErr=1.973e-01, Avg MaxErr=7.353e-01 (from 1000 samples)
    Horizon T=1.75:
      U: Avg MSE=1.250e-02, Avg RMSE=1.080e-01, Avg RelErr=1.926e-01, Avg MaxErr=7.569e-01 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=1.282e-02, Avg RMSE=1.093e-01, Avg RelErr=1.867e-01, Avg MaxErr=7.846e-01 (from 1000 samples)      
  Model: Attn_ExpBC_b32
    Horizon T=1.50:
      U: Avg MSE=1.517e-02, Avg RMSE=1.185e-01, Avg RelErr=2.225e-01, Avg MaxErr=7.759e-01 (from 1000 samples)
    Horizon T=1.75:
      U: Avg MSE=1.575e-02, Avg RMSE=1.210e-01, Avg RelErr=2.164e-01, Avg MaxErr=7.923e-01 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=1.611e-02, Avg RMSE=1.225e-01, Avg RelErr=2.094e-01, Avg MaxErr=8.192e-01 (from 1000 samples)



===== BENCHMARKING COMPLETE - AGGREGATED METRICS =====

--- Aggregated Results for Dataset: REACTION_DIFFUSION_NEUMANN_FEEDBACK ---
  Model: NoAttn_ImpBC_b32_pervar
    Horizon T=1.00:
      U: Avg MSE=2.275e-02, Avg RMSE=1.435e-01, Avg RelErr=3.026e-01, Avg MaxErr=8.944e-01 (from 1000 samples)
    Horizon T=1.50:
      U: Avg MSE=2.460e-02, Avg RMSE=1.497e-01, Avg RelErr=2.810e-01, Avg MaxErr=9.276e-01 (from 1000 samples)
    Horizon T=1.75:
      U: Avg MSE=2.419e-02, Avg RMSE=1.480e-01, Avg RelErr=2.648e-01, Avg MaxErr=9.450e-01 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=2.678e-02, Avg RMSE=1.569e-01, Avg RelErr=2.675e-01, Avg MaxErr=9.643e-01 (from 1000 samples)

  Model: NoAttn_ExpBC_b32_pervar
    Horizon T=1.00:
      U: Avg MSE=1.241e-02, Avg RMSE=1.063e-01, Avg RelErr=2.235e-01, Avg MaxErr=7.531e-01 (from 1000 samples)
    Horizon T=1.50:
      U: Avg MSE=1.428e-02, Avg RMSE=1.145e-01, Avg RelErr=2.147e-01, Avg MaxErr=8.019e-01 (from 1000 samples)
    Horizon T=1.75:
      U: Avg MSE=1.298e-02, Avg RMSE=1.094e-01, Avg RelErr=1.960e-01, Avg MaxErr=7.425e-01 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=1.672e-02, Avg RMSE=1.246e-01, Avg RelErr=2.122e-01, Avg MaxErr=8.623e-01 (from 1000 samples)
      


      
===== BENCHMARKING COMPLETE - AGGREGATED METRICS EBC_ROM_V2=====


--- Aggregated Results for Dataset: HEAT_NONLINEAR_FEEDBACK_GAIN ---
  Model: Attn_ExpBC_b32_d512_h8_bcp64_v2
    Horizon T=1.50:
      U: Avg MSE=1.181e-02, Avg RMSE=1.004e-01, Avg RelErr=2.280e-01, Avg MaxErr=8.115e-01 (from 1000 samples)
    Horizon T=1.75:
      U: Avg MSE=1.199e-02, Avg RMSE=1.010e-01, Avg RelErr=2.348e-01, Avg MaxErr=8.198e-01 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=1.264e-02, Avg RMSE=1.036e-01, Avg RelErr=2.440e-01, Avg MaxErr=8.484e-01 (from 1000 samples)
      
--- Aggregated Results for Dataset: REACTION_DIFFUSION_NEUMANN_FEEDBACK ---
  Model: Attn_ExpBC_b32_d512_h8_bcp64_v2
    Horizon T=1.00:
      U: Avg MSE=9.287e-03, Avg RMSE=9.083e-02, Avg RelErr=1.924e-01, Avg MaxErr=6.842e-01 (from 1000 samples)
    Horizon T=1.50:
      U: Avg MSE=1.095e-02, Avg RMSE=9.971e-02, Avg RelErr=1.883e-01, Avg MaxErr=7.392e-01 (from 1000 samples)
    Horizon T=1.75:
      U: Avg MSE=1.168e-02, Avg RMSE=1.035e-01, Avg RelErr=1.862e-01, Avg MaxErr=7.630e-01 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=1.232e-02, Avg RMSE=1.066e-01, Avg RelErr=1.833e-01, Avg MaxErr=7.956e-01 (from 1000 samples)
      
--- Aggregated Results for Dataset: CONVDIFF ---
  Model: Attn_ExpBC_b32_d512_h8_bcp64_v2
    Horizon T=1.50:
      U: Avg MSE=2.200e-02, Avg RMSE=1.391e-01, Avg RelErr=1.535e-01, Avg MaxErr=4.591e-01 (from 1000 samples)
    Horizon T=1.75:
      U: Avg MSE=2.293e-02, Avg RMSE=1.415e-01, Avg RelErr=1.507e-01, Avg MaxErr=4.761e-01 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=2.379e-02, Avg RMSE=1.431e-01, Avg RelErr=1.476e-01, Avg MaxErr=4.891e-01 (from 1000 samples)



===== BENCHMARKING COMPLETE - AGGREGATED METRICS =====

--- Aggregated Results for Dataset: CONVDIFF ---
  Model: Attn_ExpBC_b32_d512_h8_bcp64_v2
    Horizon T=1.50:
      U: Avg MSE=2.081e-02, Avg RMSE=1.349e-01, Avg RelErr=1.496e-01, Avg MaxErr=4.703e-01 (from 1000 samples)
    Horizon T=1.75:
      U: Avg MSE=2.234e-02, Avg RMSE=1.391e-01, Avg RelErr=1.487e-01, Avg MaxErr=4.930e-01 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=2.379e-02, Avg RMSE=1.424e-01, Avg RelErr=1.471e-01, Avg MaxErr=5.099e-01 (from 1000 samples)




--- Aggregated Results for Dataset: HEAT_NONLINEAR_FEEDBACK_GAIN ---
  Model: BAROM
    Parameters: 983,843
    Avg Inference Time: 306.87 ms/sample
    Avg Peak GPU Memory: 33.06 MB/sample
    Max Peak GPU Memory: 33.06 MB (across samples)
    U: Avg MSE=2.6378e-02, Avg RMSE=1.5579e-01, Avg RelErr=3.8605e-01, Avg MaxErr=9.4523e-01 (from 100 samples)
  Model: SPFNO
    Parameters: 287,553
    Avg Inference Time: 484.61 ms/sample
    Avg Peak GPU Memory: 32.35 MB/sample
    Max Peak GPU Memory: 32.35 MB (across samples)
    U: Avg MSE=6.3300e-02, Avg RMSE=2.3866e-01, Avg RelErr=5.8034e-01, Avg MaxErr=1.6657e+00 (from 100 samples)
  Model: BENO
    Parameters: 2,674,305
    Avg Inference Time: 1069.63 ms/sample
    Avg Peak GPU Memory: 32.46 MB/sample
    Max Peak GPU Memory: 32.46 MB (across samples)
    U: Avg MSE=2.4522e+00, Avg RMSE=1.5020e+00, Avg RelErr=3.6215e+00, Avg MaxErr=3.6496e+00 (from 100 samples)
  Model: LNO
    Parameters: 179,841
    Avg Inference Time: 478.20 ms/sample
    Avg Peak GPU Memory: 32.48 MB/sample
    Max Peak GPU Memory: 32.48 MB (across samples)
    U: Avg MSE=2.5344e-01, Avg RMSE=4.4544e-01, Avg RelErr=1.0897e+00, Avg MaxErr=3.1883e+00 (from 100 samples)
  Model: LNS_AE
    Parameters: 1,310,353
    Avg Inference Time: 337.94 ms/sample
    Avg Peak GPU Memory: 33.42 MB/sample
    Max Peak GPU Memory: 33.42 MB (across samples)
    U: Avg MSE=1.2396e-01, Avg RMSE=3.2410e-01, Avg RelErr=7.7641e-01, Avg MaxErr=1.9012e+00 (from 100 samples)
  Model: POD_DL_ROM
    Parameters: 331,537
    Avg Inference Time: 154.95 ms/sample
    Avg Peak GPU Memory: 32.34 MB/sample
    Max Peak GPU Memory: 32.34 MB (across samples)
    U: Avg MSE=8.3670e-02, Avg RMSE=2.7659e-01, Avg RelErr=6.6657e-01, Avg MaxErr=1.6951e+00 (from 100 samples)

--- Aggregated Results for Dataset: CONVDIFF ---
  Model: BAROM
    Parameters: 983,907
    Avg Inference Time: 302.48 ms/sample
    Avg Peak GPU Memory: 33.07 MB/sample
    Max Peak GPU Memory: 33.07 MB (across samples)
    U: Avg MSE=5.0320e-02, Avg RMSE=2.1189e-01, Avg RelErr=2.1424e-01, Avg MaxErr=5.9345e-01 (from 100 samples)
  Model: SPFNO
    Parameters: 287,617
    Avg Inference Time: 483.59 ms/sample
    Avg Peak GPU Memory: 32.36 MB/sample
    Max Peak GPU Memory: 32.36 MB (across samples)
    U: Avg MSE=9.2561e-02, Avg RMSE=3.0365e-01, Avg RelErr=3.0862e-01, Avg MaxErr=1.4240e+00 (from 100 samples)
  Model: BENO
    Parameters: 2,676,417
    Avg Inference Time: 1069.67 ms/sample
    Avg Peak GPU Memory: 32.48 MB/sample
    Max Peak GPU Memory: 32.48 MB (across samples)
    U: Avg MSE=3.6653e+32, Avg RMSE=4.1715e+15, Avg RelErr=4.2662e+15, Avg MaxErr=1.0399e+17 (from 100 samples)
  Model: LNO
    Parameters: 179,905
    Avg Inference Time: 478.07 ms/sample
    Avg Peak GPU Memory: 32.49 MB/sample
    Max Peak GPU Memory: 32.49 MB (across samples)
    U: Avg MSE=6.8221e-01, Avg RMSE=6.7000e-01, Avg RelErr=6.8230e-01, Avg MaxErr=3.1629e+00 (from 100 samples)
  Model: LNS_AE
    Parameters: 1,310,417
    Avg Inference Time: 335.27 ms/sample
    Avg Peak GPU Memory: 33.43 MB/sample
    Max Peak GPU Memory: 33.43 MB (across samples)
    U: Avg MSE=1.4230e-01, Avg RMSE=3.5688e-01, Avg RelErr=3.5796e-01, Avg MaxErr=1.2056e+00 (from 100 samples)
  Model: POD_DL_ROM
    Parameters: 331,793
    Avg Inference Time: 154.80 ms/sample
    Avg Peak GPU Memory: 32.35 MB/sample
    Max Peak GPU Memory: 32.35 MB (across samples)
    U: Avg MSE=4.9908e-02, Avg RMSE=2.1918e-01, Avg RelErr=2.2078e-01, Avg MaxErr=1.4240e+00 (from 100 samples)


EQ9
===== BENCHMARKING COMPLETE - AGGREGATED METRICS =====
--- Aggregated Results for Dataset: HEAT_NONLINEAR_FEEDBACK_GAIN ---
  Model: MultiVarROMEq9_Default
    Horizon Physical T=1.50:
      U: Avg MSE=1.192e-02, Avg RMSE=1.013e-01, Avg RelErr=2.304e-01, Avg MaxErr=6.367e-01 (from 1000 valid samples)
    Horizon Physical T=2.00:
      U: Avg MSE=1.342e-02, Avg RMSE=1.069e-01, Avg RelErr=2.518e-01, Avg MaxErr=6.915e-01 (from 1000 valid samples)

--- Aggregated Results for Dataset: REACTION_DIFFUSION_NEUMANN_FEEDBACK ---
  Model: MultiVarROMEq9_Default
    Horizon Physical T=1.50:
      U: Avg MSE=1.184e-02, Avg RMSE=1.046e-01, Avg RelErr=1.961e-01, Avg MaxErr=7.344e-01 (from 1000 valid samples)
    Horizon Physical T=1.75:
      U: Avg MSE=1.275e-02, Avg RMSE=1.088e-01, Avg RelErr=1.941e-01, Avg MaxErr=7.651e-01 (from 1000 valid samples)
    Horizon Physical T=2.00:
      U: Avg MSE=1.342e-02, Avg RMSE=1.118e-01, Avg RelErr=1.908e-01, Avg MaxErr=8.058e-01 (from 1000 valid samples)

--- Aggregated Results for Dataset: CONVDIFF ---
  Model: MultiVarROMEq9_Default
    Horizon Physical T=1.50:
      U: Avg MSE=2.354e-02, Avg RMSE=1.447e-01, Avg RelErr=1.583e-01, Avg MaxErr=5.347e-01 (from 1000 valid samples)
    Horizon Physical T=2.00:
      U: Avg MSE=2.529e-02, Avg RMSE=1.500e-01, Avg RelErr=1.544e-01, Avg MaxErr=5.576e-01 (from 1000 valid samples)
(pogema) 268031@pde3:/workspace/PDE_Ablation$ 

