#!/bin/sh

model_config="scripts/FITS/toml/"
data_config="scripts/defaults/"
gpu_config="scripts/defaults/gpu.toml"

# base_T (samples per dominant period): hourly=24, ETTm=96, 10-min=144, 5-min=288
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/ETTh1.toml" --model_config "$model_config/hourly.toml"
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/ETTh2.toml" --model_config "$model_config/hourly.toml"
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --model_config "$model_config/ett_15min.toml"
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/ETTm2.toml" --model_config "$model_config/ett_15min.toml"
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --model_config "$model_config/tenmin.toml"
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --model_config "$model_config/tenmin.toml"
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/SOLAL.toml" --model_config "$model_config/tenmin.toml"
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/ELT.toml" --model_config "$model_config/hourly.toml"
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/METRLA.toml" --model_config "$model_config/fivemin.toml"
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/PEMSBAY.toml" --model_config "$model_config/fivemin.toml"
