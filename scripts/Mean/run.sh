#!/bin/sh

model_config="scripts/Mean/toml/model.toml"
data_config="scripts/defaults/"
gpu_config="scripts/defaults/gpu.toml"

python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/ETTh1.toml" --model_config $model_config
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/ETTh2.toml" --model_config $model_config
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --model_config $model_config
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/ETTm2.toml" --model_config $model_config
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --model_config $model_config
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/ELT.toml" --model_config $model_config
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/TRF.toml" --model_config $model_config