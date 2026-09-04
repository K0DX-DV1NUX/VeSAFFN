#!/bin/sh

student_config1="scripts/Vera/toml/model1.toml"
student_config2="scripts/Vera/toml/model2.toml"
teacher_config="scripts/Chronos2/toml/model.toml"
data_config="scripts/defaults/"
gpu_config="scripts/defaults/gpu.toml"

# -------------------------
# Offline KD: Chronos2 teacher is fixed, only Vera trains. Unlike the other
# offline-KD teachers, Chronos2 needs no promote_kd_checkpoints.py step --
# it never trains locally (d_pretrained=true in its model.toml), so it
# always loads its own fixed pretrained weights (see models/Chronos2.py).
# Chronos2 uses a single model.toml across every dataset.
# -------------------------
# python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh1.toml" --kd_mode offline --teacher_config $teacher_config --student_config $student_config1
# python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh2.toml" --kd_mode offline --teacher_config $teacher_config --student_config $student_config1
# python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --kd_mode offline --teacher_config $teacher_config --student_config $student_config1
# python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm2.toml" --kd_mode offline --teacher_config $teacher_config --student_config $student_config1
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --kd_mode offline --teacher_config $teacher_config --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLAL.toml" --kd_mode offline --teacher_config $teacher_config --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --kd_mode offline --teacher_config $teacher_config --student_config $student_config2
# python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ELT.toml" --kd_mode offline --teacher_config $teacher_config --student_config $student_config2
# python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/METRLA.toml" --kd_mode offline --teacher_config $teacher_config --student_config $student_config2
# python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/PEMSBAY.toml" --kd_mode offline --teacher_config $teacher_config --student_config $student_config2
