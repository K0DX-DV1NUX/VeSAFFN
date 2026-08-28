#!/bin/sh

student_config1="scripts/Vera/toml/model1.toml"
student_config2="scripts/Vera/toml/model2.toml"
teacher_config="scripts/FrNet/toml/"
data_config="scripts/defaults/"
gpu_config="scripts/defaults/gpu.toml"

# -------------------------
# Offline KD: FrNet teacher is fixed, only Vera trains.
# Needs scripts/FrNet/pre_trained/<Dataset>/pl<PredLen>/checkpoint.pth for
# every (dataset, pred_len) below -- see scripts/FrNet/run_for_kd.sh (a
# single-seed training run) and scripts/promote_kd_checkpoints.py.
# SOLTX/SOLAL/ELT/TRF/METRLA/PEMSBAY share ELT_TRF.toml, matching
# scripts/FrNet/run.sh.
# -------------------------
python scripts/promote_kd_checkpoints.py --model FrNet
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh1.toml" --kd_mode offline --teacher_config "$teacher_config/ETTh1.toml" --student_config $student_config1
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh2.toml" --kd_mode offline --teacher_config "$teacher_config/ETTh2.toml" --student_config $student_config1
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --kd_mode offline --teacher_config "$teacher_config/ETTm1.toml" --student_config $student_config1
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm2.toml" --kd_mode offline --teacher_config "$teacher_config/ETTm2.toml" --student_config $student_config1
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --kd_mode offline --teacher_config "$teacher_config/ELT_TRF.toml" --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLAL.toml" --kd_mode offline --teacher_config "$teacher_config/ELT_TRF.toml" --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --kd_mode offline --teacher_config "$teacher_config/WTH.toml" --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ELT.toml" --kd_mode offline --teacher_config "$teacher_config/ELT_TRF.toml" --student_config $student_config2
# python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/TRF.toml" --kd_mode offline --teacher_config "$teacher_config/ELT_TRF.toml" --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/METRLA.toml" --kd_mode offline --teacher_config "$teacher_config/ELT_TRF.toml" --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/PEMSBAY.toml" --kd_mode offline --teacher_config "$teacher_config/ELT_TRF.toml" --student_config $student_config2
