#!/bin/sh
# Offline KD with FITS as student and every teacher: basic, then TimeDistill.

data_config="scripts/defaults/"
gpu_config="scripts/defaults/gpu.toml"

# Copy trained teacher checkpoints into scripts/<Model>/pre_trained/.
python scripts/promote_kd_checkpoints.py --model DLinear
python scripts/promote_kd_checkpoints.py --model FreqMoE
python scripts/promote_kd_checkpoints.py --model FrNet
python scripts/promote_kd_checkpoints.py --model iTransformer
python scripts/promote_kd_checkpoints.py --model PatchTST

# ===== Basic offline KD =====

# Teacher: Chronos2
student_config="scripts/FITS/toml/kd/offline/basic/"
teacher_config="scripts/Chronos2/toml/model.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh1.toml" --teacher_config $teacher_config --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh2.toml" --teacher_config $teacher_config --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --teacher_config $teacher_config --student_config "$student_config/ett_15min.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm2.toml" --teacher_config $teacher_config --student_config "$student_config/ett_15min.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --teacher_config $teacher_config --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLAL.toml" --teacher_config $teacher_config --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --teacher_config $teacher_config --student_config "$student_config/tenmin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ELT.toml" --teacher_config $teacher_config --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/METRLA.toml" --teacher_config $teacher_config --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/PEMSBAY.toml" --teacher_config $teacher_config --student_config "$student_config/fivemin.toml"

# Teacher: DLinear
student_config="scripts/FITS/toml/kd/offline/basic/"
teacher_config="scripts/DLinear/toml/model.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh1.toml" --teacher_config $teacher_config --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh2.toml" --teacher_config $teacher_config --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --teacher_config $teacher_config --student_config "$student_config/ett_15min.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm2.toml" --teacher_config $teacher_config --student_config "$student_config/ett_15min.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --teacher_config $teacher_config --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLAL.toml" --teacher_config $teacher_config --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --teacher_config $teacher_config --student_config "$student_config/tenmin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ELT.toml" --teacher_config $teacher_config --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/METRLA.toml" --teacher_config $teacher_config --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/PEMSBAY.toml" --teacher_config $teacher_config --student_config "$student_config/fivemin.toml"

# Teacher: FreqMoE
student_config="scripts/FITS/toml/kd/offline/basic/"
teacher_config="scripts/FreqMoE/toml/model.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh1.toml" --teacher_config $teacher_config --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh2.toml" --teacher_config $teacher_config --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --teacher_config $teacher_config --student_config "$student_config/ett_15min.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm2.toml" --teacher_config $teacher_config --student_config "$student_config/ett_15min.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --teacher_config $teacher_config --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLAL.toml" --teacher_config $teacher_config --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --teacher_config $teacher_config --student_config "$student_config/tenmin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ELT.toml" --teacher_config $teacher_config --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/METRLA.toml" --teacher_config $teacher_config --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/PEMSBAY.toml" --teacher_config $teacher_config --student_config "$student_config/fivemin.toml"

# Teacher: FrNet
student_config="scripts/FITS/toml/kd/offline/basic/"
teacher_config="scripts/FrNet/toml/"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh1.toml" --teacher_config "$teacher_config/ETTh1.toml" --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh2.toml" --teacher_config "$teacher_config/ETTh2.toml" --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --teacher_config "$teacher_config/ETTm1.toml" --student_config "$student_config/ett_15min.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm2.toml" --teacher_config "$teacher_config/ETTm2.toml" --student_config "$student_config/ett_15min.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --teacher_config "$teacher_config/ELT_TRF.toml" --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLAL.toml" --teacher_config "$teacher_config/ELT_TRF.toml" --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --teacher_config "$teacher_config/WTH.toml" --student_config "$student_config/tenmin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ELT.toml" --teacher_config "$teacher_config/ELT_TRF.toml" --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/METRLA.toml" --teacher_config "$teacher_config/ELT_TRF.toml" --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/PEMSBAY.toml" --teacher_config "$teacher_config/ELT_TRF.toml" --student_config "$student_config/fivemin.toml"

# Teacher: iTransformer
student_config="scripts/FITS/toml/kd/offline/basic/"
teacher_config="scripts/iTransformer/toml/model.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh1.toml" --teacher_config $teacher_config --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh2.toml" --teacher_config $teacher_config --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --teacher_config $teacher_config --student_config "$student_config/ett_15min.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm2.toml" --teacher_config $teacher_config --student_config "$student_config/ett_15min.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --teacher_config $teacher_config --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLAL.toml" --teacher_config $teacher_config --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --teacher_config $teacher_config --student_config "$student_config/tenmin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ELT.toml" --teacher_config $teacher_config --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/METRLA.toml" --teacher_config $teacher_config --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/PEMSBAY.toml" --teacher_config $teacher_config --student_config "$student_config/fivemin.toml"

# Teacher: PatchTST
student_config="scripts/FITS/toml/kd/offline/basic/"
teacher_config="scripts/PatchTST/toml/"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh1.toml" --teacher_config "$teacher_config/model1.toml" --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh2.toml" --teacher_config "$teacher_config/model1.toml" --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --teacher_config "$teacher_config/model2.toml" --student_config "$student_config/ett_15min.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm2.toml" --teacher_config "$teacher_config/model2.toml" --student_config "$student_config/ett_15min.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --teacher_config "$teacher_config/model2.toml" --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLAL.toml" --teacher_config "$teacher_config/model2.toml" --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --teacher_config "$teacher_config/model2.toml" --student_config "$student_config/tenmin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ELT.toml" --teacher_config "$teacher_config/model2.toml" --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/METRLA.toml" --teacher_config "$teacher_config/model2.toml" --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/PEMSBAY.toml" --teacher_config "$teacher_config/model2.toml" --student_config "$student_config/fivemin.toml"

# ===== TimeDistill =====

# Teacher: Chronos2
student_config="scripts/FITS/toml/kd/offline/timedistill/"
teacher_config="scripts/Chronos2/toml/model.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh1.toml" --teacher_config $teacher_config --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh2.toml" --teacher_config $teacher_config --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --teacher_config $teacher_config --student_config "$student_config/ett_15min.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm2.toml" --teacher_config $teacher_config --student_config "$student_config/ett_15min.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --teacher_config $teacher_config --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLAL.toml" --teacher_config $teacher_config --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --teacher_config $teacher_config --student_config "$student_config/tenmin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ELT.toml" --teacher_config $teacher_config --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/METRLA.toml" --teacher_config $teacher_config --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/PEMSBAY.toml" --teacher_config $teacher_config --student_config "$student_config/fivemin.toml"

# Teacher: DLinear
student_config="scripts/FITS/toml/kd/offline/timedistill/"
teacher_config="scripts/DLinear/toml/model.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh1.toml" --teacher_config $teacher_config --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh2.toml" --teacher_config $teacher_config --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --teacher_config $teacher_config --student_config "$student_config/ett_15min.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm2.toml" --teacher_config $teacher_config --student_config "$student_config/ett_15min.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --teacher_config $teacher_config --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLAL.toml" --teacher_config $teacher_config --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --teacher_config $teacher_config --student_config "$student_config/tenmin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ELT.toml" --teacher_config $teacher_config --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/METRLA.toml" --teacher_config $teacher_config --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/PEMSBAY.toml" --teacher_config $teacher_config --student_config "$student_config/fivemin.toml"

# Teacher: FreqMoE
student_config="scripts/FITS/toml/kd/offline/timedistill/"
teacher_config="scripts/FreqMoE/toml/model.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh1.toml" --teacher_config $teacher_config --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh2.toml" --teacher_config $teacher_config --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --teacher_config $teacher_config --student_config "$student_config/ett_15min.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm2.toml" --teacher_config $teacher_config --student_config "$student_config/ett_15min.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --teacher_config $teacher_config --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLAL.toml" --teacher_config $teacher_config --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --teacher_config $teacher_config --student_config "$student_config/tenmin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ELT.toml" --teacher_config $teacher_config --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/METRLA.toml" --teacher_config $teacher_config --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/PEMSBAY.toml" --teacher_config $teacher_config --student_config "$student_config/fivemin.toml"

# Teacher: FrNet
student_config="scripts/FITS/toml/kd/offline/timedistill/"
teacher_config="scripts/FrNet/toml/"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh1.toml" --teacher_config "$teacher_config/ETTh1.toml" --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh2.toml" --teacher_config "$teacher_config/ETTh2.toml" --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --teacher_config "$teacher_config/ETTm1.toml" --student_config "$student_config/ett_15min.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm2.toml" --teacher_config "$teacher_config/ETTm2.toml" --student_config "$student_config/ett_15min.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --teacher_config "$teacher_config/ELT_TRF.toml" --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLAL.toml" --teacher_config "$teacher_config/ELT_TRF.toml" --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --teacher_config "$teacher_config/WTH.toml" --student_config "$student_config/tenmin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ELT.toml" --teacher_config "$teacher_config/ELT_TRF.toml" --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/METRLA.toml" --teacher_config "$teacher_config/ELT_TRF.toml" --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/PEMSBAY.toml" --teacher_config "$teacher_config/ELT_TRF.toml" --student_config "$student_config/fivemin.toml"

# Teacher: iTransformer
student_config="scripts/FITS/toml/kd/offline/timedistill/"
teacher_config="scripts/iTransformer/toml/model.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh1.toml" --teacher_config $teacher_config --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh2.toml" --teacher_config $teacher_config --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --teacher_config $teacher_config --student_config "$student_config/ett_15min.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm2.toml" --teacher_config $teacher_config --student_config "$student_config/ett_15min.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --teacher_config $teacher_config --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLAL.toml" --teacher_config $teacher_config --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --teacher_config $teacher_config --student_config "$student_config/tenmin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ELT.toml" --teacher_config $teacher_config --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/METRLA.toml" --teacher_config $teacher_config --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/PEMSBAY.toml" --teacher_config $teacher_config --student_config "$student_config/fivemin.toml"

# Teacher: PatchTST
student_config="scripts/FITS/toml/kd/offline/timedistill/"
teacher_config="scripts/PatchTST/toml/"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh1.toml" --teacher_config "$teacher_config/model1.toml" --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh2.toml" --teacher_config "$teacher_config/model1.toml" --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --teacher_config "$teacher_config/model2.toml" --student_config "$student_config/ett_15min.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTm2.toml" --teacher_config "$teacher_config/model2.toml" --student_config "$student_config/ett_15min.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLTX.toml" --teacher_config "$teacher_config/model2.toml" --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/SOLAL.toml" --teacher_config "$teacher_config/model2.toml" --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --teacher_config "$teacher_config/model2.toml" --student_config "$student_config/tenmin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ELT.toml" --teacher_config "$teacher_config/model2.toml" --student_config "$student_config/hourly.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/METRLA.toml" --teacher_config "$teacher_config/model2.toml" --student_config "$student_config/fivemin.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/PEMSBAY.toml" --teacher_config "$teacher_config/model2.toml" --student_config "$student_config/fivemin.toml"
