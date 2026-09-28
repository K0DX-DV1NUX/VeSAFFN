#!/bin/sh
# Offline KD for Vesa (d_kd_mode = offline): basic, then TimeDistill, every teacher.
# Comment out any teacher section (or a whole block) to skip it.

data_config="scripts/defaults/"
gpu_config="scripts/defaults/gpu.toml"

# Offline teachers read fixed checkpoints from scripts/<Model>/pre_trained/ (Chronos2 needs none).
python scripts/promote_kd_checkpoints.py --model DLinear
python scripts/promote_kd_checkpoints.py --model FreqMoE
python scripts/promote_kd_checkpoints.py --model FrNet
python scripts/promote_kd_checkpoints.py --model iTransformer
python scripts/promote_kd_checkpoints.py --model PatchTST

# =========================
# Basic offline KD (toml/kd/offline/basic)
# =========================

# -------------------------
# Teacher: Chronos2
# -------------------------
student_config1="scripts/Vesa/toml/kd/offline/basic/model1.toml"
student_config2="scripts/Vesa/toml/kd/offline/basic/model2.toml"
teacher_config="scripts/Chronos2/toml/model.toml"
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

# -------------------------
# Teacher: DLinear
# -------------------------
student_config1="scripts/Vesa/toml/kd/offline/basic/model1.toml"
student_config2="scripts/Vesa/toml/kd/offline/basic/model2.toml"
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

# -------------------------
# Teacher: FreqMoE
# -------------------------
student_config1="scripts/Vesa/toml/kd/offline/basic/model1.toml"
student_config2="scripts/Vesa/toml/kd/offline/basic/model2.toml"
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

# -------------------------
# Teacher: FrNet
# -------------------------
student_config1="scripts/Vesa/toml/kd/offline/basic/model1.toml"
student_config2="scripts/Vesa/toml/kd/offline/basic/model2.toml"
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

# -------------------------
# Teacher: iTransformer
# -------------------------
student_config1="scripts/Vesa/toml/kd/offline/basic/model1.toml"
student_config2="scripts/Vesa/toml/kd/offline/basic/model2.toml"
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

# -------------------------
# Teacher: PatchTST
# -------------------------
student_config1="scripts/Vesa/toml/kd/offline/basic/model1.toml"
student_config2="scripts/Vesa/toml/kd/offline/basic/model2.toml"
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

# =========================
# TimeDistill (toml/kd/offline/timedistill)
# =========================

# -------------------------
# Teacher: Chronos2
# -------------------------
student_config1="scripts/Vesa/toml/kd/offline/timedistill/model1.toml"
student_config2="scripts/Vesa/toml/kd/offline/timedistill/model2.toml"
teacher_config="scripts/Chronos2/toml/model.toml"
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

# -------------------------
# Teacher: DLinear
# -------------------------
student_config1="scripts/Vesa/toml/kd/offline/timedistill/model1.toml"
student_config2="scripts/Vesa/toml/kd/offline/timedistill/model2.toml"
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

# -------------------------
# Teacher: FreqMoE
# -------------------------
student_config1="scripts/Vesa/toml/kd/offline/timedistill/model1.toml"
student_config2="scripts/Vesa/toml/kd/offline/timedistill/model2.toml"
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

# -------------------------
# Teacher: FrNet
# -------------------------
student_config1="scripts/Vesa/toml/kd/offline/timedistill/model1.toml"
student_config2="scripts/Vesa/toml/kd/offline/timedistill/model2.toml"
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

# -------------------------
# Teacher: iTransformer
# -------------------------
student_config1="scripts/Vesa/toml/kd/offline/timedistill/model1.toml"
student_config2="scripts/Vesa/toml/kd/offline/timedistill/model2.toml"
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

# -------------------------
# Teacher: PatchTST
# -------------------------
student_config1="scripts/Vesa/toml/kd/offline/timedistill/model1.toml"
student_config2="scripts/Vesa/toml/kd/offline/timedistill/model2.toml"
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
