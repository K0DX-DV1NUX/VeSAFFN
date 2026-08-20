#!/bin/sh

student_config="scripts/Vera/toml/model.toml"
teacher_config="scripts/DLinear/toml/model.toml"
data_config="scripts/defaults/"
gpu_config="scripts/defaults/gpu.toml"

# -------------------------
# Online KD: DLinear teacher and Vera student train together.
# DLinear uses a single model.toml across every dataset.
# -------------------------
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh1.toml" --kd_mode online --teacher_config $teacher_config --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh2.toml" --kd_mode online --teacher_config $teacher_config --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --kd_mode online --teacher_config $teacher_config --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm2.toml" --kd_mode online --teacher_config $teacher_config --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --kd_mode online --teacher_config $teacher_config --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLAL.toml" --kd_mode online --teacher_config $teacher_config --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --kd_mode online --teacher_config $teacher_config --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ELT.toml" --kd_mode online --teacher_config $teacher_config --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/TRF.toml" --kd_mode online --teacher_config $teacher_config --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/METRLA.toml" --kd_mode online --teacher_config $teacher_config --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/PEMSBAY.toml" --kd_mode online --teacher_config $teacher_config --student_config $student_config
