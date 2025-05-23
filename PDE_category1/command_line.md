python preprocess_data_for_ROM.py \
    --raw_data_path datasets_full/burgers_data_10000s_128nx_600nt.pkl \
    --dataset_type burgers \
    --train_nt 300 \
    --output_path tensor_dataset/data_ROM_burgers.pt

nohup python BAROM_ImpBC.py --datatype advection > out_ROM_adv.log 2>&1 &
nohup python BAROM_ImpBC.py --datatype euler > out_ROM_euler.log 2>&1 &
nohup python BAROM_ImpBC.py --datatype burgers > out_ROM_burgers.log 2>&1 &
nohup python BAROM_ImpBC.py --datatype darcy > out_ROM_darcy.log 2>&1 &

nohup python FNO.py  > out_FNO_adv.log 2>&1 &
nohup python FNO.py  > out_FNO_euler.log 2>&1 &
nohup python FNO.py  > out_FNO_burgers.log 2>&1 &
nohup python FNO.py  > out_FNO_darcy.log 2>&1 &

nohup python DON.py --datatype advection > out_DON_adv.log 2>&1 &
nohup python DON.py --datatype euler > out_DON_euler.log 2>&1 &
nohup python DON.py --datatype burgers > out_DON_burgers.log 2>&1 &
nohup python DON.py --datatype darcy  > out_DON_darcy.log 2>&1 &

nohup python OPNO.py --datatype advection > out_OPNO_adv.log 2>&1 &
nohup python OPNO.py --datatype euler > out_OPNO_euler.log 2>&1 &
nohup python OPNO.py --datatype burgers > out_OPNO_burgers.log 2>&1 &
nohup python OPNO.py --datatype darcy  > out_OPNO_darcy.log 2>&1 &

nohup python SPFNO.py --datatype advection > out_SPFNO_adv.log 2>&1 &
nohup python SPFNO.py --datatype euler > out_SPFNO_euler.log 2>&1 &
nohup python SPFNO.py --datatype burgers > out_SPFNO_burgers.log 2>&1 &
nohup python SPFNO.py --datatype darcy  > out_SPFNO_darcy.log 2>&1 &

python preprocess_data_for_BENO.py \
    --raw_data_path datasets_full/euler_data_10000s_128nx_600nt.pkl \
    --dataset_type euler \
    --train_nt 300 \
    --output_path tensor_dataset/data_BENO_euler.pt
    
nohup python BENO.py --datatype advection > out_BENO_adv.log 2>&1 &
nohup python BENO-Copy1.py --datatype euler > out_BENO_euler.log 2>&1 &
nohup python BENO.py --datatype burgers > out_BENO_burgers.log 2>&1 &
nohup python BENO.py --datatype darcy  > out_BENO_darcy.log 2>&1 &

nohup python POD_DL_ROM.py --datatype advection > out_POD_DL_ROM_adv.log 2>&1 &
nohup python POD_DL_ROM.py --datatype euler > out_POD_DL_ROM_euler.log 2>&1 &
nohup python POD_DL_ROM.py --datatype burgers > out_POD_DL_ROM_burgers.log 2>&1 &
nohup python POD_DL_ROM.py --datatype darcy  > out_POD_DL_ROM_darcy.log 2>&1 &

nohup python LNO.py --datatype advection > out_LNO_adv.log 2>&1 &
nohup python LNO.py --datatype euler > out_LNO_euler.log 2>&1 &
nohup python LNO.py --datatype burgers > out_LNO_burgers.log 2>&1 &
nohup python LNO.py --datatype darcy  > out_LNO_darcy.log 2>&1 &

datasets_full/advection_data_10000s_128nx_600nt.pkl
datasets_full/euler_data_10000s_128nx_600nt.pkl
datasets_full/burgers_data_10000s_128nx_600nt.pkl
datasets_full/darcy_data_10000s_128nx_600nt.pkl

nohup python LNS_AE.py --datatype advection > out_LNS_AE_adv.log 2>&1 &
nohup python LNS_AE.py --datatype euler > out_LNS_AE_euler.log 2>&1 &
nohup python LNS_AE.py --datatype burgers > out_LNS_AE_burgers.log 2>&1 &
nohup python LNS_AE.py --datatype darcy  > out_LNS_AE_darcy.log 2>&1 &
