#!/bin/sh

student_config="scripts/FITS/toml/"
teacher_config="scripts/PatchTST/toml/"
data_config="scripts/defaults/"
gpu_config="scripts/defaults/gpu.toml"

# Student toml follows the dataset's sampling frequency (see run.sh).
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh1.toml" --kd_mode online --teacher_config "$teacher_config/model1.toml" --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh2.toml" --kd_mode online --teacher_config "$teacher_config/model1.toml" --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --kd_mode online --teacher_config "$teacher_config/model2.toml" --student_config "$student_config/ett_15min.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm2.toml" --kd_mode online --teacher_config "$teacher_config/model2.toml" --student_config "$student_config/ett_15min.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --kd_mode online --teacher_config "$teacher_config/model2.toml" --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLAL.toml" --kd_mode online --teacher_config "$teacher_config/model2.toml" --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --kd_mode online --teacher_config "$teacher_config/model2.toml" --student_config "$student_config/tenmin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ELT.toml" --kd_mode online --teacher_config "$teacher_config/model2.toml" --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/METRLA.toml" --kd_mode online --teacher_config "$teacher_config/model2.toml" --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/PEMSBAY.toml" --kd_mode online --teacher_config "$teacher_config/model2.toml" --student_config "$student_config/fivemin.toml"
