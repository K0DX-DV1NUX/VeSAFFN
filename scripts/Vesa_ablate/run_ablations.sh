#!/bin/sh


data_config="scripts/defaults/"
gpu_config="scripts/defaults/gpu.toml"

#General vs Channel Ablations
# model_config="scripts/Vesa_ablate/toml/test_gen_vs_channel.toml"
# python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --model_config $model_config
# python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --model_config $model_config
# python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --model_config $model_config
# python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/ELT.toml" --model_config $model_config


# Decompostion Ablations for VeSA-FFN
model_config="scripts/Vesa_ablate/toml/test_decomp.toml"
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --model_config $model_config
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --model_config $model_config
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --model_config $model_config


# Rank Ablations
# model_config="scripts/Vesa_ablate/toml/test_rank.toml"
# python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --model_config $model_config
# python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --model_config $model_config
# python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --model_config $model_config