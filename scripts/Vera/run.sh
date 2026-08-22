#!/bin/sh

model_config1="scripts/Vera/toml/model1.toml"
model_config2="scripts/Vera/toml/model2.toml"
data_config="scripts/defaults/"
gpu_config="scripts/defaults/gpu.toml"

python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/ETTh1.toml" --model_config $model_config1
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/ETTh2.toml" --model_config $model_config1
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --model_config $model_config1
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/ETTm2.toml" --model_config $model_config1
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --model_config $model_config2
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --model_config $model_config2
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/SOLAL.toml" --model_config $model_config2
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/ELT.toml" --model_config $model_config2
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/METRLA.toml" --model_config $model_config2
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/PEMSBAY.toml" --model_config $model_config2
