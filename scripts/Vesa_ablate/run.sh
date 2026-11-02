#!/bin/sh
# Vesa decomposition ablation on ETTm1, WTH and SOLTX, seeds 2021-2023.

model_config="scripts/Vesa_ablate/toml/"
data_config="scripts/defaults/"
gpu_config="scripts/defaults/gpu.toml"

python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --model_config "$model_config/wavelet.toml"
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --model_config "$model_config/wavelet.toml"
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --model_config "$model_config/wavelet.toml"
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --model_config "$model_config/trend_residual.toml"
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --model_config "$model_config/trend_residual.toml"
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --model_config "$model_config/trend_residual.toml"
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --model_config "$model_config/no_decomp.toml"
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --model_config "$model_config/no_decomp.toml"
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --model_config "$model_config/no_decomp.toml"
