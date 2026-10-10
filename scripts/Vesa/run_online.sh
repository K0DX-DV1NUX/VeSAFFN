#!/bin/sh
# Online KD with Vesa as student and every teacher.

data_config="scripts/defaults/"
gpu_config="scripts/defaults/gpu.toml"

# Teacher: DLinear
student_config1="scripts/Vesa/toml/kd/online/model1.toml"
student_config2="scripts/Vesa/toml/kd/online/model2.toml"
teacher_config="scripts/DLinear/toml/model.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh1.toml" --teacher_config $teacher_config --student_config $student_config1
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh2.toml" --teacher_config $teacher_config --student_config $student_config1
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --teacher_config $teacher_config --student_config $student_config1
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm2.toml" --teacher_config $teacher_config --student_config $student_config1
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --teacher_config $teacher_config --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLAL.toml" --teacher_config $teacher_config --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --teacher_config $teacher_config --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ELT.toml" --teacher_config $teacher_config --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/METRLA.toml" --teacher_config $teacher_config --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/PEMSBAY.toml" --teacher_config $teacher_config --student_config $student_config2

# Teacher: FreqMoE
student_config1="scripts/Vesa/toml/kd/online/model1.toml"
student_config2="scripts/Vesa/toml/kd/online/model2.toml"
teacher_config="scripts/FreqMoE/toml/model.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh1.toml" --teacher_config $teacher_config --student_config $student_config1
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh2.toml" --teacher_config $teacher_config --student_config $student_config1
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --teacher_config $teacher_config --student_config $student_config1
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm2.toml" --teacher_config $teacher_config --student_config $student_config1
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --teacher_config $teacher_config --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLAL.toml" --teacher_config $teacher_config --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --teacher_config $teacher_config --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ELT.toml" --teacher_config $teacher_config --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/METRLA.toml" --teacher_config $teacher_config --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/PEMSBAY.toml" --teacher_config $teacher_config --student_config $student_config2

# Teacher: FrNet
student_config1="scripts/Vesa/toml/kd/online/model1.toml"
student_config2="scripts/Vesa/toml/kd/online/model2.toml"
teacher_config="scripts/FrNet/toml/"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh1.toml" --teacher_config "$teacher_config/ETTh1.toml" --student_config $student_config1
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh2.toml" --teacher_config "$teacher_config/ETTh2.toml" --student_config $student_config1
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --teacher_config "$teacher_config/ETTm1.toml" --student_config $student_config1
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm2.toml" --teacher_config "$teacher_config/ETTm2.toml" --student_config $student_config1
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --teacher_config "$teacher_config/ELT_TRF.toml" --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLAL.toml" --teacher_config "$teacher_config/ELT_TRF.toml" --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --teacher_config "$teacher_config/WTH.toml" --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ELT.toml" --teacher_config "$teacher_config/ELT_TRF.toml" --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/METRLA.toml" --teacher_config "$teacher_config/ELT_TRF.toml" --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/PEMSBAY.toml" --teacher_config "$teacher_config/ELT_TRF.toml" --student_config $student_config2

# Teacher: iTransformer
student_config1="scripts/Vesa/toml/kd/online/model1.toml"
student_config2="scripts/Vesa/toml/kd/online/model2.toml"
teacher_config="scripts/iTransformer/toml/model.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh1.toml" --teacher_config $teacher_config --student_config $student_config1
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh2.toml" --teacher_config $teacher_config --student_config $student_config1
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --teacher_config $teacher_config --student_config $student_config1
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm2.toml" --teacher_config $teacher_config --student_config $student_config1
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --teacher_config $teacher_config --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLAL.toml" --teacher_config $teacher_config --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --teacher_config $teacher_config --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ELT.toml" --teacher_config $teacher_config --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/METRLA.toml" --teacher_config $teacher_config --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/PEMSBAY.toml" --teacher_config $teacher_config --student_config $student_config2

# Teacher: PatchTST
student_config1="scripts/Vesa/toml/kd/online/model1.toml"
student_config2="scripts/Vesa/toml/kd/online/model2.toml"
teacher_config="scripts/PatchTST/toml/"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh1.toml" --teacher_config "$teacher_config/model1.toml" --student_config $student_config1
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh2.toml" --teacher_config "$teacher_config/model1.toml" --student_config $student_config1
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --teacher_config "$teacher_config/model2.toml" --student_config $student_config1
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm2.toml" --teacher_config "$teacher_config/model2.toml" --student_config $student_config1
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --teacher_config "$teacher_config/model2.toml" --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLAL.toml" --teacher_config "$teacher_config/model2.toml" --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --teacher_config "$teacher_config/model2.toml" --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ELT.toml" --teacher_config "$teacher_config/model2.toml" --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/METRLA.toml" --teacher_config "$teacher_config/model2.toml" --student_config $student_config2
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/PEMSBAY.toml" --teacher_config "$teacher_config/model2.toml" --student_config $student_config2
