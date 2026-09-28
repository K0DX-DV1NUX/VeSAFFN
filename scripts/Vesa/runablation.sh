#!/bin/sh

model_config="scripts/Vesa/toml/standalone/ablation2.toml"
data_config="scripts/defaults/"
gpu_config="scripts/defaults/gpu.toml"


python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --model_config $model_config
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --model_config $model_config
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --model_config $model_config

