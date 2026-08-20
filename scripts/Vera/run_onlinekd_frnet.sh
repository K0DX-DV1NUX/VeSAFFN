#!/bin/sh

student_config="scripts/Vera/toml/model.toml"
teacher_config="scripts/FrNet/toml/"
data_config="scripts/defaults/"
gpu_config="scripts/defaults/gpu.toml"

# -------------------------
# Online KD: FrNet teacher and Vera student train together.
# FrNet has a per-dataset model.toml; SOLTX/SOLAL/ELT/TRF/METRLA/PEMSBAY
# share ELT_TRF.toml, matching scripts/FrNet/run.sh.
# -------------------------
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh1.toml" --kd_mode online --teacher_config "$teacher_config/ETTh1.toml" --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh2.toml" --kd_mode online --teacher_config "$teacher_config/ETTh2.toml" --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --kd_mode online --teacher_config "$teacher_config/ETTm1.toml" --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm2.toml" --kd_mode online --teacher_config "$teacher_config/ETTm2.toml" --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --kd_mode online --teacher_config "$teacher_config/ELT_TRF.toml" --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLAL.toml" --kd_mode online --teacher_config "$teacher_config/ELT_TRF.toml" --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --kd_mode online --teacher_config "$teacher_config/WTH.toml" --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ELT.toml" --kd_mode online --teacher_config "$teacher_config/ELT_TRF.toml" --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/TRF.toml" --kd_mode online --teacher_config "$teacher_config/ELT_TRF.toml" --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/METRLA.toml" --kd_mode online --teacher_config "$teacher_config/ELT_TRF.toml" --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/PEMSBAY.toml" --kd_mode online --teacher_config "$teacher_config/ELT_TRF.toml" --student_config $student_config
