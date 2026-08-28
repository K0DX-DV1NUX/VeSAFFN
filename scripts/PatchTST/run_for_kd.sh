#!/bin/sh

model_config="scripts/PatchTST/toml/"
data_config="scripts/defaults/"
gpu_config="scripts/defaults/gpu.toml"

# -------------------------
# Trains PatchTST normally, but for a single fixed seed only -- this is not
# for benchmarking (see run.sh for that), it's to produce one checkpoint per
# (dataset, pred_len) to promote into a fixed offline-KD teacher. After this
# finishes, run:
#   python scripts/promote_kd_checkpoints.py --model PatchTST
# -------------------------
python -u runExp.py --seed 2021 --gpu_config $gpu_config --data_config "$data_config/ETTh1.toml" --model_config "$model_config/model1.toml"
python -u runExp.py --seed 2021 --gpu_config $gpu_config --data_config "$data_config/ETTh2.toml" --model_config "$model_config/model1.toml"
python -u runExp.py --seed 2021 --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --model_config "$model_config/model2.toml"
python -u runExp.py --seed 2021 --gpu_config $gpu_config --data_config "$data_config/ETTm2.toml" --model_config "$model_config/model2.toml"
python -u runExp.py --seed 2021 --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --model_config "$model_config/model2.toml"
python -u runExp.py --seed 2021 --gpu_config $gpu_config --data_config "$data_config/SOLAL.toml" --model_config "$model_config/model2.toml"
python -u runExp.py --seed 2021 --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --model_config "$model_config/model2.toml"
python -u runExp.py --seed 2021 --gpu_config $gpu_config --data_config "$data_config/ELT.toml" --model_config "$model_config/model2.toml"
# python -u runExp.py --seed 2021 --gpu_config $gpu_config --data_config "$data_config/TRF.toml" --model_config "$model_config/model2.toml"
python -u runExp.py --seed 2021 --gpu_config $gpu_config --data_config "$data_config/METRLA.toml" --model_config "$model_config/model2.toml"
python -u runExp.py --seed 2021 --gpu_config $gpu_config --data_config "$data_config/PEMSBAY.toml" --model_config "$model_config/model2.toml"
