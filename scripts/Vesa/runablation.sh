#!/bin/sh

gpu_config="scripts/defaults/gpu.toml"
params_config="scripts/Vesa/toml/params_ablation.toml"

python -u runParams.py --gpu_config $gpu_config --params_config $params_config --models Vesa --output VesaAblation.csv
