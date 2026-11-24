#!/bin/sh

data_config="scripts/defaults/"
gpu_config="scripts/defaults/gpu.toml"



# Basic KD ablation with Vesa_ablate as student and FrNet as teacher on ETTh2 and PEMSBAY, seeds 2021-2023.
# Offline: alpha in {0.2, 0.7, 1.0} (frozen pretrained FrNet, no warmup).
# Online: warmup in {1, 5, 10, 30} at alpha 1.0.
# alpha = 0 (no KD) is the channelwise run in test_gen_vs_channel.toml.
python scripts/promote_kd_checkpoints.py --model FrNet
teacher_config="scripts/FrNet/toml/"
offline_config="scripts/Vesa_ablate/toml/test_offline_alpha.toml"
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh2.toml" --teacher_config "$teacher_config/ETTh2.toml" --student_config $offline_config
python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/PEMSBAY.toml" --teacher_config "$teacher_config/ELT_TRF.toml" --student_config $offline_config

# online_config="scripts/Vesa_ablate/toml/test_online_warmup.toml"
# python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/ETTh2.toml" --teacher_config "$teacher_config/ETTh2.toml" --student_config $online_config
# python -u runKD.py --gpu_config $gpu_config --data_config "$data_config/PEMSBAY.toml" --teacher_config "$teacher_config/ELT_TRF.toml" --student_config $online_config
