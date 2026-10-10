import argparse
import tomllib
import torch
import random
import numpy as np

from base.config import ExpConfig, KDConfig
from exp.exp_kd import Exp_KD, Exp_KD_Offline
from exp.exp_timedistill import Exp_TimeDistill_Offline
from utils.tools import check_and_prepare_dirs, save_results, inverse_transform
from utils.plotting import plot_results


def main():
    parser = argparse.ArgumentParser(description="Knowledge Distillation Runner (online or offline)")
    parser.add_argument("--data_config", type=str, required=True, help="Path to experiment TOML config for dataset.")
    parser.add_argument("--gpu_config", type=str, required=True, help="Path to experiment TOML config for gpu settings.")
    parser.add_argument("--teacher_config", type=str, required=True, help="Path to experiment TOML config for the KD teacher model.")
    parser.add_argument("--student_config", type=str, required=True, help="Path to experiment TOML config for the KD student model.")
    args = parser.parse_args()

    def load_toml(path):
        with open(path, "rb") as f:
            configs = tomllib.load(f)
        return configs

    data_values = load_toml(args.data_config)
    gpu_values = load_toml(args.gpu_config)
    teacher_values = load_toml(args.teacher_config)
    student_values = load_toml(args.student_config)

    # The teacher's own per-horizon m_* settings.
    def teacher_run_overrides(pred_len):
        for t_run in teacher_values.get("runs", []):
            if t_run.get("d_pred_len") == pred_len:
                return {k: v for k, v in t_run.items() if k.startswith("m_")}
        return {}

    # The student's runs drive the grid; seeds 2021-2023.
    for run in student_values["runs"]:
        for d_seed in range(2021, 2024):

            # Teacher shares the student's d_* keys, keeps its own m_* keys.
            shared_window = {k: v for k, v in run.items() if k.startswith("d_")}

            teacher_merged = {
                **data_values['default'],
                **gpu_values.get('gpu', {}),
                **teacher_values['default'],
                **teacher_run_overrides(run.get("d_pred_len")),
                **shared_window,
            }
            student_merged = {
                **data_values['default'],
                **gpu_values.get('gpu', {}),
                **student_values['default'],
                **run,
            }

            teacher_cfgs = ExpConfig(**teacher_merged)
            student_cfgs = KDConfig(**student_merged)

            # Pick the KD experiment from d_kd_mode / d_kd_type.
            if student_cfgs.d_kd_type == "timedistill":
                Exp_KD_Class = Exp_TimeDistill_Offline
            elif student_cfgs.d_kd_mode == "online":
                Exp_KD_Class = Exp_KD
            else:
                Exp_KD_Class = Exp_KD_Offline

            random.seed(d_seed)
            torch.manual_seed(d_seed)
            np.random.seed(d_seed)

            # Fall back to CPU if CUDA is unavailable.
            for cfgs in (teacher_cfgs, student_cfgs):
                cfgs.d_use_gpu = True if torch.cuda.is_available() and cfgs.d_use_gpu else False
                if cfgs.d_use_gpu and cfgs.d_use_multi_gpu:
                    cfgs.d_devices = [int(device_id) for device_id in cfgs.d_devices]
                    cfgs.d_gpu = cfgs.d_devices[0]

            print('Teacher configuration in experiment:')
            print(teacher_cfgs)
            print('Student configuration in experiment:')
            print(student_cfgs)

            # Run name, used for checkpoints and results.
            setting = '{}-{}-{}-{}-{}-ft{}-sl{}-pl{}-seed{}'.format(
                "KD" if student_cfgs.d_kd_type == "basic" else "TimeDistill",
                student_cfgs.d_kd_mode,
                teacher_cfgs.d_model,
                student_cfgs.d_model,
                student_cfgs.d_data,
                student_cfgs.d_forecast_type,
                student_cfgs.d_seq_len,
                student_cfgs.d_pred_len,
                d_seed)

            teacher_cfgs.d_setting = setting
            student_cfgs.d_setting = setting

            check_and_prepare_dirs(student_cfgs)

            exp = Exp_KD_Class(teacher_cfgs, student_cfgs)

            if student_cfgs.d_is_training:
                    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                    exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            preds, trues = exp.test(setting)

            if student_cfgs.d_inverse_transform:
                preds = inverse_transform(student_cfgs, preds)
                trues = inverse_transform(student_cfgs, trues)

            save_results(student_cfgs, preds, trues)
            if student_cfgs.d_plot_results:
                plot_results(student_cfgs, preds, trues, zoom_to=400)

            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
