#!/bin/sh

student_config1="scripts/Vera/toml/model1.toml"
student_config2="scripts/Vera/toml/model2.toml"
teacher_config="scripts/iTransformer/toml/model.toml"
data_config="scripts/defaults/"
gpu_config="scripts/defaults/gpu.toml"

# -------------------------
# Online KD: iTransformer teacher and Vera student train together.
# iTransformer uses a single model.toml across every dataset.
# -------------------------
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh1.toml" --kd_mode online --teacher_config $teacher_config --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh2.toml" --kd_mode online --teacher_config $teacher_config --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --kd_mode online --teacher_config $teacher_config --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm2.toml" --kd_mode online --teacher_config $teacher_config --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --kd_mode online --teacher_config $teacher_config --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLAL.toml" --kd_mode online --teacher_config $teacher_config --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --kd_mode online --teacher_config $teacher_config --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ELT.toml" --kd_mode online --teacher_config $teacher_config --student_config $student_config
#python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/TRF.toml" --kd_mode online --teacher_config $teacher_config --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/METRLA.toml" --kd_mode online --teacher_config $teacher_config --student_config $student_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/PEMSBAY.toml" --kd_mode online --teacher_config $teacher_config --student_config $student_config
