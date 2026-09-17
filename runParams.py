import argparse
import tomllib
import random

import numpy as np
import pandas as pd
import torch

from base.config import ExpConfig
from exp.exp_params import Exp_Params

# -------------------------------------------------
# Model profiled by runParams.py, mapped to the model.toml variant each one
# trains with on ETTh1 (see the corresponding scripts/<Model>/run.sh).
# -------------------------------------------------
MODEL_CONFIGS = {
    "DLinear": "scripts/DLinear/toml/model.toml",
    "FrNet": "scripts/FrNet/toml/ETTh1.toml",
    "iTransformer": "scripts/iTransformer/toml/model.toml",
    "PatchTST": "scripts/PatchTST/toml/model1.toml",
    "Vesa": "scripts/Vesa/toml/model1.toml",
    # Chronos-2 is a fixed, pretrained zero-shot forecaster (no per-horizon
    # architecture tuning), so its model.toml carries no "m_*" overrides --
    # see scripts/Chronos2/toml/model.toml.
    "Chronos2": "scripts/Chronos2/toml/model.toml",
}

# Forward-pass-only profiling has no training dynamics, so a single fixed
# seed is enough for reproducibility.
PARAMS_SEED = 2021


def load_toml(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def model_run_overrides(model_values, pred_len):
    '''
    Some models tune their architecture per-horizon in model.toml (e.g.
    iTransformer widens m_d_model for longer pred_len, FrNet changes
    m_period_list). Find the model.toml run matching this pred_len and
    return only its model-config ("m_*") keys, so the params.toml run --
    which owns d_seq_len/d_pred_len/channels for the profiling grid --
    is never overwritten.
    '''
    for run in model_values.get("runs", []):
        if run.get("d_pred_len") == pred_len:
            return {k: v for k, v in run.items() if k.startswith("m_")}
    return {}


def main():
    parser = argparse.ArgumentParser(description="Model size / MACs / FLOPs profiler")
    parser.add_argument("--gpu_config", type=str, required=True, help="Path to experiment TOML config for gpu settings.")
    parser.add_argument("--params_config", type=str, required=True, help="Path to experiment TOML config for the channel/seq_len/pred_len grid.")
    parser.add_argument("--output", type=str, default="Params.csv", help="Path to the output CSV file.")
    args = parser.parse_args()

    gpu_values = load_toml(args.gpu_config)
    params_values = load_toml(args.params_config)

    random.seed(PARAMS_SEED)
    torch.manual_seed(PARAMS_SEED)
    np.random.seed(PARAMS_SEED)

    rows = []

    for label, model_config_path in MODEL_CONFIGS.items():
        model_values = load_toml(model_config_path)

        for run in params_values["runs"]:
            overrides = model_run_overrides(model_values, run["d_pred_len"])

            # -------------------------
            # Order of Priority follows:
            # params_config[default] < gpu_config < model_config[default] < model_config[runs] < params_config[runs]
            # -------------------------
            merged = {
                **params_values['default'],
                **gpu_values.get('gpu', {}),
                **model_values['default'],
                **overrides,
                **run,
            }
            cfgs = ExpConfig(**merged)

            cfgs.d_use_gpu = True if torch.cuda.is_available() and cfgs.d_use_gpu else False
            if cfgs.d_use_gpu and cfgs.d_use_multi_gpu:
                cfgs.d_devices = [int(device_id) for device_id in cfgs.d_devices]
                cfgs.d_gpu = cfgs.d_devices[0]

            print('>>>>>>>profiling: {} (ch={}, sl={}, pl={})<<<<<<<'.format(
                label, cfgs.d_in_features, cfgs.d_seq_len, cfgs.d_pred_len))

            try:
                exp = Exp_Params(cfgs)
                stats = exp.profile()
            except Exception as e:
                print('!!!!!!!failed: {} (ch={}, sl={}, pl={}): {}'.format(
                    label, cfgs.d_in_features, cfgs.d_seq_len, cfgs.d_pred_len, e))
                continue

            rows.append({
                "label": label,
                "model": cfgs.d_model,
                "channels": cfgs.d_in_features,
                "seq_len": cfgs.d_seq_len,
                "pred_len": cfgs.d_pred_len,
                "batch_size": cfgs.d_batch_size,
                "total_params": stats["total_params"],
                "trainable_params": stats["trainable_params"],
                "active_params": stats["active_params"],
                "macs": stats["macs"],
                "gmacs": stats["macs"] / 1e9,
                "flops": stats["flops"],
                "gflops": stats["flops"] / 1e9,
                "size_mb": stats["size_mb"],
                "latency_ms": stats["latency_ms"],
            })

            torch.cuda.empty_cache()

    # -------------------------
    # A single shared CSV, overwritten each run -- unlike Exp_Main, which
    # appends to a Result_{model}.csv per model.
    # -------------------------
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print('Saved {} rows to {}'.format(len(df), args.output))


if __name__ == "__main__":
    main()
