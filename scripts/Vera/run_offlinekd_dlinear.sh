#!/bin/sh

student_config1="scripts/Vera/toml/model1.toml"
student_config2="scripts/Vera/toml/model2.toml"
teacher_config="scripts/DLinear/toml/model.toml"
data_config="scripts/defaults/"
gpu_config="scripts/defaults/gpu.toml"

# -------------------------
# Offline KD: DLinear teacher is fixed, only Vera trains.
# Needs scripts/DLinear/pre_trained/<Dataset>/pl<PredLen>/checkpoint.pth for
# every (dataset, pred_len) below -- see scripts/DLinear/run_for_kd.sh (a
# single-seed training run) and scripts/promote_kd_checkpoints.py. DLinear
# uses a single model.toml across every dataset.
# -------------------------
python scripts/promote_kd_checkpoints.py --model DLinear
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh1.toml" --kd_mode offline --teacher_config $teacher_config --student_config $student_config1
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh2.toml" --kd_mode offline --teacher_config $teacher_config --student_config $student_config1
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --kd_mode offline --teacher_config $teacher_config --student_config $student_config1
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm2.toml" --kd_mode offline --teacher_config $teacher_config --student_config $student_config1
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --kd_mode offline --teacher_config $teacher_config --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLAL.toml" --kd_mode offline --teacher_config $teacher_config --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --kd_mode offline --teacher_config $teacher_config --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ELT.toml" --kd_mode offline --teacher_config $teacher_config --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/METRLA.toml" --kd_mode offline --teacher_config $teacher_config --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/PEMSBAY.toml" --kd_mode offline --teacher_config $teacher_config --student_config $student_config2
