nohup python BAROM_ImpBC.py --datatype convdiff_feedback > out_BAROM_ImpBC_convdiff_feedback.log 2>&1 &
nohup python BAROM_ImpBC.py --datatype reaction_diffusion_neumann_feedback > out_BAROM_ImpBC_reaction_diffusion_neumann_feedback.log 2>&1 &
nohup python BAROM_ImpBC.py --datatype heat_nonlinear_feedback_gain > out_BAROM_ImpBC_heat_nonlinear_feedback_gain.log 2>&1 &

nohup python FNO.py --datatype convdiff  > out_FNO_convdiff_feedback.log 2>&1 &
nohup python FNO.py --datatype reaction_diffusion_neumann_feedback  > out_FNO_reaction_diffusion_neumann_feedback.log 2>&1 &
nohup python FNO.py --datatype heat_nonlinear_feedback_gain > out_FNO_heat_nonlinear_feedback_gain.log 2>&1 &

nohup python SPFNO.py --datatype convdiff > out_SPFNO_convdiff.log 2>&1 &
nohup python SPFNO.py --datatype reaction_diffusion_neumann_feedback > out_SPFNO_reaction_diffusion_neumann_feedback.log 2>&1 &
nohup python SPFNO.py --datatype heat_nonlinear_feedback_gain > out_SPFNO_heat_nonlinear_feedback_gain.log 2>&1 &

nohup python BENO.py --datatype convdiff > out_BENO_convdiff.log 2>&1 &
nohup python BENO.py --datatype reaction_diffusion_neumann_feedback > out_BENO_reaction_diffusion_neumann_feedback.log 2>&1 &
nohup python BENO.py --datatype heat_nonlinear_feedback_gain > out_BENO_heat_nonlinear_feedback_gain.log 2>&1 &


nohup python POD_DL_ROM.py --datatype convdiff > out_POD_DL_ROM_convdiff.log 2>&1 &
nohup python POD_DL_ROM.py --datatype reaction_diffusion_neumann_feedback > out_POD_DL_ROM_reaction_diffusion_neumann_feedback.log 2>&1 &
nohup python POD_DL_ROM.py --datatype heat_nonlinear_feedback_gain > out_POD_DL_ROM_heat_nonlinear_feedback_gain.log 2>&1 &

nohup python LNO.py --datatype convdiff > out_LNO_convdiff.log 2>&1 &
nohup python LNO.py --datatype reaction_diffusion_neumann_feedback > out_LNO_reaction_diffusion_neumann_feedback.log 2>&1 &
nohup python LNO.py --datatype heat_nonlinear_feedback_gain > out_LNO_heat_nonlinear_feedback_gain.log 2>&1 &

nohup python LNS_AE.py --datatype convdiff > out_LNS_AE_convdiff.log 2>&1 &
nohup python LNS_AE.py --datatype reaction_diffusion_neumann_feedback > out_LNS_AE_reaction_diffusion_neumann_feedback.log 2>&1 &
nohup python LNS_AE.py --datatype heat_nonlinear_feedback_gain > out_LNS_AE_heat_nonlinear_feedback_gain.log 2>&1 &



===== BENCHMARKING COMPLETE - AGGREGATED METRICS =====
    Horizon T=1.5:

--- Aggregated Results for Dataset: HEAT_NONLINEAR_FEEDBACK_GAIN ---
  Model: ROM_FULL
    U: Avg MSE=2.6030e-02, Avg RMSE=1.5246e-01, Avg RelErr=3.4556e-01, Avg MaxErr=8.6775e-01 (from 1000 samples)
  Model: FNO
    U: Avg MSE=8.4762e+10, Avg RMSE=1.0732e+04, Avg RelErr=1.7688e+04, Avg MaxErr=2.6292e+05 (from 1000 samples)
  Model: SPFNO
    U: Avg MSE=5.8058e-02, Avg RMSE=2.2582e-01, Avg RelErr=5.0793e-01, Avg MaxErr=1.5259e+00 (from 1000 samples)
  Model: BENO
    U: Avg MSE=2.9890e+00, Avg RMSE=1.6599e+00, Avg RelErr=3.7019e+00, Avg MaxErr=3.9250e+00 (from 1000 samples)
  Model: LNS_AE
    U: Avg MSE=1.1152e-01, Avg RMSE=3.0581e-01, Avg RelErr=6.7213e-01, Avg MaxErr=1.8522e+00 (from 1000 samples)
  Model: LNO
    U: Avg MSE=1.6616e-01, Avg RMSE=3.5398e-01, Avg RelErr=8.0227e-01, Avg MaxErr=2.4433e+00 (from 1000 samples)
  Model: POD_DL_ROM
    U: Avg MSE=9.9577e-02, Avg RMSE=2.9997e-01, Avg RelErr=6.6488e-01, Avg MaxErr=1.5952e+00 (from 1000 samples)

--- Aggregated Results for Dataset: REACTION_DIFFUSION_NEUMANN_FEEDBACK ---
  Model: ROM_FULL
    U: Avg MSE=2.4586e-02, Avg RMSE=1.5018e-01, Avg RelErr=2.8193e-01, Avg MaxErr=9.3765e-01 (from 1000 samples)
  Model: FNO
    U: Avg MSE=5.2020e+18, Avg RMSE=4.2635e+08, Avg RelErr=8.1845e+08, Avg MaxErr=1.4844e+10 (from 1000 samples)
  Model: SPFNO
    U: Avg MSE=5.1065e-02, Avg RMSE=2.1946e-01, Avg RelErr=4.1187e-01, Avg MaxErr=1.6040e+00 (from 1000 samples)
  Model: BENO
    U: Avg MSE=2.0436e+00, Avg RMSE=1.4084e+00, Avg RelErr=2.6328e+00, Avg MaxErr=4.1725e+00 (from 1000 samples)
  Model: LNS_AE
    U: Avg MSE=1.6777e-01, Avg RMSE=3.8723e-01, Avg RelErr=7.1425e-01, Avg MaxErr=1.5986e+00 (from 1000 samples)
  Model: LNO
    U: Avg MSE=5.5217e-02, Avg RMSE=2.2717e-01, Avg RelErr=4.2455e-01, Avg MaxErr=1.5043e+00 (from 1000 samples)
  Model: POD_DL_ROM
    U: Avg MSE=5.1935e-02, Avg RMSE=2.2375e-01, Avg RelErr=4.1968e-01, Avg MaxErr=1.6042e+00 (from 1000 samples)

--- Aggregated Results for Dataset: CONVDIFF ---
  Model: ROM_FULL
    U: Avg MSE=3.6175e-02, Avg RMSE=1.7919e-01, Avg RelErr=1.9399e-01, Avg MaxErr=4.9820e-01 (from 1000 samples)
  Model: FNO
    U: Avg MSE=5.7028e+09, Avg RMSE=2.7192e+03, Avg RelErr=2.6770e+03, Avg MaxErr=1.7441e+04 (from 1000 samples)
  Model: SPFNO
    U: Avg MSE=8.6842e-02, Avg RMSE=2.9366e-01, Avg RelErr=3.1447e-01, Avg MaxErr=1.4194e+00 (from 1000 samples)
  Model: BENO
    U: Avg MSE=2.8006e+23, Avg RMSE=6.7389e+10, Avg RelErr=7.6956e+10, Avg MaxErr=1.1265e+12 (from 1000 samples)
  Model: LNS_AE
    U: Avg MSE=1.0484e-01, Avg RMSE=3.0274e-01, Avg RelErr=3.2058e-01, Avg MaxErr=1.0500e+00 (from 1000 samples)
  Model: LNO
    U: Avg MSE=3.4275e-01, Avg RMSE=4.7958e-01, Avg RelErr=5.3144e-01, Avg MaxErr=2.2429e+00 (from 1000 samples)
  Model: POD_DL_ROM
    U: Avg MSE=4.8711e-02, Avg RMSE=2.1710e-01, Avg RelErr=2.3133e-01, Avg MaxErr=1.4194e+00 (from 1000 samples)
(pogema) 268031@pde5:/workspace/PDE_C$



===== BENCHMARKING COMPLETE - AGGREGATED METRICS =====

--- Aggregated Results for Dataset: HEAT_NONLINEAR_FEEDBACK_GAIN ---
  Model: ROM_FULL
    Horizon T=1.75:
      U: Avg MSE=2.6735e-02, Avg RMSE=1.5453e-01, Avg RelErr=3.5952e-01, Avg MaxErr=8.8091e-01 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=2.8267e-02, Avg RMSE=1.5877e-01, Avg RelErr=3.7507e-01, Avg MaxErr=9.2644e-01 (from 1000 samples)
  Model: FNO
    Horizon T=1.75:
      U: Avg MSE=3.6597e+14, Avg RMSE=6.1479e+05, Avg RelErr=1.0071e+06, Avg MaxErr=1.8855e+07 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=1.6208e+18, Avg RMSE=4.0335e+07, Avg RelErr=6.8639e+07, Avg MaxErr=1.4008e+09 (from 1000 samples)
  Model: SPFNO
    Horizon T=1.75:
      U: Avg MSE=6.0377e-02, Avg RMSE=2.3079e-01, Avg RelErr=5.3136e-01, Avg MaxErr=1.5746e+00 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=6.4578e-02, Avg RMSE=2.3892e-01, Avg RelErr=5.5711e-01, Avg MaxErr=1.6745e+00 (from 1000 samples)
  Model: BENO
    Horizon T=1.75:
      U: Avg MSE=2.6307e+00, Avg RMSE=1.5574e+00, Avg RelErr=3.5650e+00, Avg MaxErr=3.7106e+00 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=2.6812e+00, Avg RMSE=1.5721e+00, Avg RelErr=3.6583e+00, Avg MaxErr=3.7779e+00 (from 1000 samples)
  Model: LNS_AE
    Horizon T=1.75:
      U: Avg MSE=1.2382e-01, Avg RMSE=3.2417e-01, Avg RelErr=7.3181e-01, Avg MaxErr=1.9456e+00 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=1.3274e-01, Avg RMSE=3.3711e-01, Avg RelErr=7.7442e-01, Avg MaxErr=2.0124e+00 (from 1000 samples)
  Model: LNO
    Horizon T=1.75:
      U: Avg MSE=2.3159e-01, Avg RMSE=4.2002e-01, Avg RelErr=9.8342e-01, Avg MaxErr=2.9508e+00 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=3.1026e-01, Avg RMSE=4.7750e-01, Avg RelErr=1.1312e+00, Avg MaxErr=3.3873e+00 (from 1000 samples)
  Model: POD_DL_ROM
    Horizon T=1.75:
      U: Avg MSE=9.3372e-02, Avg RMSE=2.9010e-01, Avg RelErr=6.6052e-01, Avg MaxErr=1.6350e+00 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=9.0662e-02, Avg RMSE=2.8540e-01, Avg RelErr=6.6009e-01, Avg MaxErr=1.7210e+00 (from 1000 samples)

--- Aggregated Results for Dataset: REACTION_DIFFUSION_NEUMANN_FEEDBACK ---
  Model: ROM_FULL
    Horizon T=1.75:
      U: Avg MSE=2.5283e-02, Avg RMSE=1.5256e-01, Avg RelErr=2.7226e-01, Avg MaxErr=9.5210e-01 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=2.6157e-02, Avg RMSE=1.5545e-01, Avg RelErr=2.6475e-01, Avg MaxErr=9.7699e-01 (from 1000 samples)
  Model: FNO
    Horizon T=1.75:
      U: Avg MSE=1.4346e+23, Avg RMSE=6.2961e+10, Avg RelErr=1.1646e+11, Avg MaxErr=2.3827e+12 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=5.1955e+27, Avg RMSE=1.1114e+13, Avg RelErr=2.0034e+13, Avg MaxErr=4.5014e+14 (from 1000 samples)
  Model: SPFNO
    Horizon T=1.75:
      U: Avg MSE=5.4322e-02, Avg RMSE=2.2608e-01, Avg RelErr=4.0456e-01, Avg MaxErr=1.6578e+00 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=5.8135e-02, Avg RMSE=2.3347e-01, Avg RelErr=3.9998e-01, Avg MaxErr=1.7125e+00 (from 1000 samples)
  Model: BENO
    Horizon T=1.75:
      U: Avg MSE=2.3470e+00, Avg RMSE=1.5092e+00, Avg RelErr=2.6846e+00, Avg MaxErr=4.2550e+00 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=2.3909e+00, Avg RMSE=1.5231e+00, Avg RelErr=2.5878e+00, Avg MaxErr=4.3054e+00 (from 1000 samples)
  Model: LNS_AE
    Horizon T=1.75:
      U: Avg MSE=1.8871e-01, Avg RMSE=4.1033e-01, Avg RelErr=7.2101e-01, Avg MaxErr=1.6343e+00 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=2.0904e-01, Avg RMSE=4.3109e-01, Avg RelErr=7.2409e-01, Avg MaxErr=1.6646e+00 (from 1000 samples)
  Model: LNO
    Horizon T=1.75:
      U: Avg MSE=6.2067e-02, Avg RMSE=2.4085e-01, Avg RelErr=4.3044e-01, Avg MaxErr=1.5844e+00 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=6.8213e-02, Avg RMSE=2.5252e-01, Avg RelErr=4.3275e-01, Avg MaxErr=1.6633e+00 (from 1000 samples)
  Model: POD_DL_ROM
    Horizon T=1.75:
      U: Avg MSE=5.0334e-02, Avg RMSE=2.2017e-01, Avg RelErr=3.9372e-01, Avg MaxErr=1.6578e+00 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=5.1012e-02, Avg RMSE=2.2174e-01, Avg RelErr=3.7942e-01, Avg MaxErr=1.7125e+00 (from 1000 samples)

--- Aggregated Results for Dataset: CONVDIFF ---
  Model: ROM_FULL
    Horizon T=1.75:
      U: Avg MSE=3.8770e-02, Avg RMSE=1.8550e-01, Avg RelErr=1.9379e-01, Avg MaxErr=5.1824e-01 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=4.4318e-02, Avg RMSE=1.9712e-01, Avg RelErr=2.0047e-01, Avg MaxErr=5.6458e-01 (from 1000 samples)
  Model: FNO
    Horizon T=1.75:
      U: Avg MSE=6.0589e+12, Avg RMSE=8.8625e+04, Avg RelErr=8.9583e+04, Avg MaxErr=6.0744e+05 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=7.9250e+15, Avg RMSE=3.2054e+06, Avg RelErr=3.2031e+06, Avg MaxErr=2.3380e+07 (from 1000 samples)
  Model: SPFNO
    Horizon T=1.75:
      U: Avg MSE=8.9208e-02, Avg RMSE=2.9786e-01, Avg RelErr=3.1116e-01, Avg MaxErr=1.4194e+00 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=9.2524e-02, Avg RMSE=3.0358e-01, Avg RelErr=3.1034e-01, Avg MaxErr=1.4194e+00 (from 1000 samples)
  Model: BENO
    Horizon T=1.75:
      U: Avg MSE=nan, Avg RMSE=nan, Avg RelErr=nan, Avg MaxErr=nan (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=nan, Avg RMSE=nan, Avg RelErr=nan, Avg MaxErr=nan (from 1000 samples)
  Model: LNS_AE
    Horizon T=1.75:
      U: Avg MSE=1.2317e-01, Avg RMSE=3.2748e-01, Avg RelErr=3.3766e-01, Avg MaxErr=1.1128e+00 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=1.3864e-01, Avg RMSE=3.4950e-01, Avg RelErr=3.5362e-01, Avg MaxErr=1.1566e+00 (from 1000 samples)
  Model: LNO
    Horizon T=1.75:
      U: Avg MSE=4.5807e-01, Avg RMSE=5.5638e-01, Avg RelErr=5.9180e-01, Avg MaxErr=2.6334e+00 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=5.9680e-01, Avg RMSE=6.4258e-01, Avg RelErr=6.6114e-01, Avg MaxErr=3.1310e+00 (from 1000 samples)
  Model: POD_DL_ROM
    Horizon T=1.75:
      U: Avg MSE=4.5649e-02, Avg RMSE=2.0972e-01, Avg RelErr=2.1782e-01, Avg MaxErr=1.4194e+00 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=4.9380e-02, Avg RMSE=2.1786e-01, Avg RelErr=2.2099e-01, Avg MaxErr=1.4194e+00 (from 1000 samples)



--- Aggregated Results for Dataset: REACTION_DIFFUSION_NEUMANN_FEEDBACK ---    ALbation
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
  Model: ROM_Attention_FixedLift
    Horizon T=1.50:
      U: Avg MSE=7.348e-02, Avg RMSE=2.645e-01, Avg RelErr=4.948e-01, Avg MaxErr=1.108e+00 (from 1000 samples)
    Horizon T=1.75:
      U: Avg MSE=7.461e-02, Avg RMSE=2.669e-01, Avg RelErr=4.759e-01, Avg MaxErr=1.143e+00 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=7.930e-02, Avg RMSE=2.752e-01, Avg RelErr=4.693e-01, Avg MaxErr=1.200e+00 (from 1000 samples)
  Model: ROM_Attention_RandomPhi
    Horizon T=1.50:
      U: Avg MSE=2.374e-02, Avg RMSE=1.465e-01, Avg RelErr=2.750e-01, Avg MaxErr=9.160e-01 (from 1000 samples)
    Horizon T=1.75:
      U: Avg MSE=2.421e-02, Avg RMSE=1.483e-01, Avg RelErr=2.649e-01, Avg MaxErr=9.313e-01 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=2.477e-02, Avg RMSE=1.503e-01, Avg RelErr=2.566e-01, Avg MaxErr=9.554e-01 (from 1000 samples)
  Model: ROM_LSTM
    Horizon T=1.50:
      U: Avg MSE=1.197e-02, Avg RMSE=1.055e-01, Avg RelErr=1.973e-01, Avg MaxErr=7.353e-01 (from 1000 samples)
    Horizon T=1.75:
      U: Avg MSE=1.250e-02, Avg RMSE=1.080e-01, Avg RelErr=1.926e-01, Avg MaxErr=7.569e-01 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=1.282e-02, Avg RMSE=1.093e-01, Avg RelErr=1.867e-01, Avg MaxErr=7.846e-01 (from 1000 samples)





--- Aggregated Results for Dataset: REACTION_DIFFUSION_NEUMANN_FEEDBACK ---
  Model: ROM_Attention_bd32
    Horizon T=1.5:
    U: Avg MSE=2.4586e-02, Avg RMSE=1.5018e-01, Avg RelErr=2.8193e-01, Avg MaxErr=9.3765e-01 (from 1000 samples)
    Horizon T=1.75:
      U: Avg MSE=2.5283e-02, Avg RMSE=1.5256e-01, Avg RelErr=2.7226e-01, Avg MaxErr=9.5210e-01 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=2.6157e-02, Avg RMSE=1.5545e-01, Avg RelErr=2.6475e-01, Avg MaxErr=9.7699e-01 (from 1000 samples)
  Model: FNO
    Horizon T=1.5:
    U: Avg MSE=5.2020e+18, Avg RMSE=4.2635e+08, Avg RelErr=8.1845e+08, Avg MaxErr=1.4844e+10 (from 1000 samples)
    Horizon T=1.75:
      U: Avg MSE=1.4346e+23, Avg RMSE=6.2961e+10, Avg RelErr=1.1646e+11, Avg MaxErr=2.3827e+12 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=5.1955e+27, Avg RMSE=1.1114e+13, Avg RelErr=2.0034e+13, Avg MaxErr=4.5014e+14 (from 1000 samples)
  Model: SPFNO
    Horizon T=1.5:
    U: Avg MSE=5.1065e-02, Avg RMSE=2.1946e-01, Avg RelErr=4.1187e-01, Avg MaxErr=1.6040e+00 (from 1000 samples)
    Horizon T=1.75:
      U: Avg MSE=5.4322e-02, Avg RMSE=2.2608e-01, Avg RelErr=4.0456e-01, Avg MaxErr=1.6578e+00 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=5.8135e-02, Avg RMSE=2.3347e-01, Avg RelErr=3.9998e-01, Avg MaxErr=1.7125e+00 (from 1000 samples)
  Model: BENO
    Horizon T=1.5:
    U: Avg MSE=2.0436e+00, Avg RMSE=1.4084e+00, Avg RelErr=2.6328e+00, Avg MaxErr=4.1725e+00 (from 1000 samples)
    Horizon T=1.75:
      U: Avg MSE=2.3470e+00, Avg RMSE=1.5092e+00, Avg RelErr=2.6846e+00, Avg MaxErr=4.2550e+00 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=2.3909e+00, Avg RMSE=1.5231e+00, Avg RelErr=2.5878e+00, Avg MaxErr=4.3054e+00 (from 1000 samples)
  Model: LNS_AE
    Horizon T=1.5:
    U: Avg MSE=1.6777e-01, Avg RMSE=3.8723e-01, Avg RelErr=7.1425e-01, Avg MaxErr=1.5986e+00 (from 1000 samples)
    Horizon T=1.75:
      U: Avg MSE=1.8871e-01, Avg RMSE=4.1033e-01, Avg RelErr=7.2101e-01, Avg MaxErr=1.6343e+00 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=2.0904e-01, Avg RMSE=4.3109e-01, Avg RelErr=7.2409e-01, Avg MaxErr=1.6646e+00 (from 1000 samples)
  Model: LNO
    Horizon T=1.5:
    U: Avg MSE=5.5217e-02, Avg RMSE=2.2717e-01, Avg RelErr=4.2455e-01, Avg MaxErr=1.5043e+00 (from 1000 samples)
    Horizon T=1.75:
      U: Avg MSE=6.2067e-02, Avg RMSE=2.4085e-01, Avg RelErr=4.3044e-01, Avg MaxErr=1.5844e+00 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=6.8213e-02, Avg RMSE=2.5252e-01, Avg RelErr=4.3275e-01, Avg MaxErr=1.6633e+00 (from 1000 samples)
  Model: POD_DL_ROM
    Horizon T=1.5:
    U: Avg MSE=5.1935e-02, Avg RMSE=2.2375e-01, Avg RelErr=4.1968e-01, Avg MaxErr=1.6042e+00 (from 1000 samples)
    Horizon T=1.75:
      U: Avg MSE=5.0334e-02, Avg RMSE=2.2017e-01, Avg RelErr=3.9372e-01, Avg MaxErr=1.6578e+00 (from 1000 samples)
    Horizon T=2.00:
      U: Avg MSE=5.1012e-02, Avg RMSE=2.2174e-01, Avg RelErr=3.7942e-01, Avg MaxErr=1.7125e+00 (from 1000 samples)
