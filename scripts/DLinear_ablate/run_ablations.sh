#!/bin/sh
# DLinear_ablate decomposition ablation: no_decomp, trend_residual, db1 and sym4,
# on the same datasets as the Vesa_ablate decomposition ablation (ETTm1, WTH, SOLTX).

data_config="scripts/defaults/"
gpu_config="scripts/defaults/gpu.toml"

# Decomposition Ablations
model_config="scripts/DLinear_ablate/toml/test_decomp.toml"
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --model_config $model_config
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --model_config $model_config
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --model_config $model_config
