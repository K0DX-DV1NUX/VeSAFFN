#!/bin/sh

student_config="scripts/Vera/toml/model.toml"
teacher_config="scripts/PatchTST/toml/"
data_config="scripts/defaults/"
gpu_config="scripts/defaults/gpu.toml"

# -------------------------
# Offline KD: PatchTST teacher is fixed, only Vera trains.
# Needs scripts/PatchTST/pre_trained/<Dataset>/pl<PredLen>/checkpoint.pth for
# every (dataset, pred_len) below -- see scripts/PatchTST/run_for_kd.sh (a
# single-seed training run) and scripts/promote_kd_checkpoints.py.
# PatchTST uses model1.toml for ETTh1/ETTh2 and model2.toml everywhere else,
# matching scripts/PatchTST/run.sh.
# -------------------------
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh1.toml" --kd_mode offline --teacher_config "$teacher_config/model1.toml" --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh2.toml" --kd_mode offline --teacher_config "$teacher_config/model1.toml" --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --kd_mode offline --teacher_config "$teacher_config/model2.toml" --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm2.toml" --kd_mode offline --teacher_config "$teacher_config/model2.toml" --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --kd_mode offline --teacher_config "$teacher_config/model2.toml" --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLAL.toml" --kd_mode offline --teacher_config "$teacher_config/model2.toml" --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --kd_mode offline --teacher_config "$teacher_config/model2.toml" --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ELT.toml" --kd_mode offline --teacher_config "$teacher_config/model2.toml" --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/TRF.toml" --kd_mode offline --teacher_config "$teacher_config/model2.toml" --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/METRLA.toml" --kd_mode offline --teacher_config "$teacher_config/model2.toml" --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/PEMSBAY.toml" --kd_mode offline --teacher_config "$teacher_config/model2.toml" --student_config $student_config
