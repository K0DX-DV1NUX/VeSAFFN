#!/bin/sh

gpu_config="scripts/defaults/gpu.toml"
params_config="scripts/defaults/params.toml"

python -u runParams.py --gpu_config $gpu_config --params_config $params_config
