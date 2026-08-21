import argparse
import tomllib
import torch
import random
import numpy as np

from base.config import ExpConfig
from exp.exp_main import Exp_Main
from utils.tools import check_and_prepare_dirs, save_results, inverse_transform
from utils.plotting import plot_results


def main():
    parser = argparse.ArgumentParser(description="LTSF Runner")
    parser.add_argument("--data_config",type=str,required=True, help="Path to experiment TOML config for dataset.")
    parser.add_argument("--gpu_config", type=str, required=True, help="Path to experiment TOML config for gpu settings.")
    parser.add_argument("--model_config", type=str, required=True, help="Path to experiment TOML config for model settings.")
    parser.add_argument("--seed", type=int, default=None,
                         help="Run only this single seed instead of the default 2021-2023 sweep "
                              "(e.g. for training a one-off fixed teacher checkpoint for offline KD).")
    args = parser.parse_args()

    seeds = [args.seed] if args.seed is not None else range(2021, 2024)

    # -------------------------
    # Load Running Configurations.
    # -------------------------
    def load_toml(path):
        with open(path, "rb") as f:
            configs = tomllib.load(f)
        return configs

    data_values = load_toml(args.data_config)
    gpu_values = load_toml(args.gpu_config)
    model_values = load_toml(args.model_config)


    # -------------------------
    # Run Experiments.
    # Seed Values = 2021, 2022, 2023 (or a single seed via --seed).
    #
    # -------------------------
    for run in model_values["runs"]:
        for d_seed in seeds:

            # -------------------------
            # Merged will allow to overide "default" settings, by settings in "runs".
            # Ex: if data_values.default.d_learning_rate=0.001, model.default.d_learning_rate=0.1, then,
            # merged.d_learning_rate=0.1.
            #
            # Order of Priority follows:
            # data_config[default] < model_config[default] < run
            # -------------------------
            merged = {
                **data_values['default'], 
                **gpu_values.get('gpu',{}),
                **model_values['default'],
                **run,
            }
            cfgs = ExpConfig(**merged)

            random.seed(d_seed)
            torch.manual_seed(d_seed)
            np.random.seed(d_seed)

            # --------------------------------
            # Set the GPU settings.
            # --------------------------------
            cfgs.d_use_gpu = True if torch.cuda.is_available() and cfgs.d_use_gpu else False

            # If using multiple GPUs, set up the device ids and the primary gpu.
            if cfgs.d_use_gpu and cfgs.d_use_multi_gpu:
                cfgs.d_devices = [int(device_id) for device_id in cfgs.d_devices]
                cfgs.d_gpu = cfgs.d_devices[0]

            # --------------------------------
            # Print the arguments for the experiment. 
            # This is useful for logging and reproducibility.
            # --------------------------------
            print('Configuration in experiment:')
            print(cfgs)

            setting = '{}-{}-ft{}-sl{}-pl{}-seed{}'.format(
                cfgs.d_model,
                cfgs.d_data,
                cfgs.d_forecast_type,
                cfgs.d_seq_len,
                cfgs.d_pred_len,
                d_seed)

            cfgs.d_setting = setting

            # --------------------------------
            # Prepare directories.
            # --------------------------------
            check_and_prepare_dirs(cfgs)

            # --------------------------------
            # RUN EXPERIMENT.
            # --------------------------------
            exp = Exp_Main(cfgs)

            # If training is enabled.
            if cfgs.d_is_training:
                    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(cfgs.d_setting))
                    exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(cfgs.d_setting))
            preds, trues = exp.test(cfgs.d_setting)

            # If output needs to be inverse transformed.
            if cfgs.d_inverse_transform:
                preds = inverse_transform(cfgs, preds)
                trues = inverse_transform(cfgs, trues)

            # --------------------------------
            # Save and plot results.
            # --------------------------------
            save_results(cfgs, preds, trues)
            if cfgs.d_plot_results:
                plot_results(cfgs, preds, trues, zoom_to=400)

            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
