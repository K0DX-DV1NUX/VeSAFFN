#!/bin/sh
# Trains one seed per dataset and horizon, for use as an offline KD teacher.
# Then run: python scripts/promote_kd_checkpoints.py --model iTransformer

model_config="scripts/iTransformer/toml/model.toml"
data_config="scripts/defaults/"
gpu_config="scripts/defaults/gpu.toml"

python -u runExp.py --seed 2021 --gpu_config $gpu_config --data_config "$data_config/ETTh1.toml" --model_config $model_config
python -u runExp.py --seed 2021 --gpu_config $gpu_config --data_config "$data_config/ETTh2.toml" --model_config $model_config
python -u runExp.py --seed 2021 --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --model_config $model_config
python -u runExp.py --seed 2021 --gpu_config $gpu_config --data_config "$data_config/ETTm2.toml" --model_config $model_config
python -u runExp.py --seed 2021 --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --model_config $model_config
python -u runExp.py --seed 2021 --gpu_config $gpu_config --data_config "$data_config/SOLAL.toml" --model_config $model_config
python -u runExp.py --seed 2021 --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --model_config $model_config
python -u runExp.py --seed 2021 --gpu_config $gpu_config --data_config "$data_config/ELT.toml" --model_config $model_config
# python -u runExp.py --seed 2021 --gpu_config $gpu_config --data_config "$data_config/TRF.toml" --model_config $model_config
python -u runExp.py --seed 2021 --gpu_config $gpu_config --data_config "$data_config/METRLA.toml" --model_config $model_config
python -u runExp.py --seed 2021 --gpu_config $gpu_config --data_config "$data_config/PEMSBAY.toml" --model_config $model_config
