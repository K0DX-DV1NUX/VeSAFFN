import argparse
import tomllib
import torch
import random
import numpy as np

from base.config import ExpConfig
from exp.exp_kd import Exp_KD, Exp_KD_Offline
from utils.tools import check_and_prepare_dirs, save_results, inverse_transform
from utils.plotting import plot_results


def main():
    parser = argparse.ArgumentParser(description="Knowledge Distillation Runner (online or offline)")
    parser.add_argument("--data_config", type=str, required=True, help="Path to experiment TOML config for dataset.")
    parser.add_argument("--gpu_config", type=str, required=True, help="Path to experiment TOML config for gpu settings.")
    parser.add_argument("--teacher_config", type=str, required=True, help="Path to experiment TOML config for the KD teacher model.")
    parser.add_argument("--student_config", type=str, required=True, help="Path to experiment TOML config for the KD student model.")
    parser.add_argument("--kd_mode", type=str, default="online", choices=["online", "offline"],
                         help="'online' trains teacher and student jointly. 'offline' freezes the teacher "
                              "(a self-contained pretrained model like Chronos2, or a fixed checkpoint at "
                              "scripts/<Model>/pre_trained/checkpoint.pth) and only trains the student.")
    args = parser.parse_args()

    Exp_KD_Class = Exp_KD if args.kd_mode == "online" else Exp_KD_Offline

    # -------------------------
    # Load Running Configurations.
    # -------------------------
    def load_toml(path):
        with open(path, "rb") as f:
            configs = tomllib.load(f)
        return configs

    data_values = load_toml(args.data_config)
    gpu_values = load_toml(args.gpu_config)
    teacher_values = load_toml(args.teacher_config)
    student_values = load_toml(args.student_config)

    def teacher_run_overrides(pred_len):
        '''
        The teacher may tune its own architecture per-horizon in its own
        model.toml (e.g. iTransformer widening m_d_model at longer
        pred_len). Find the teacher run matching this pred_len and return
        only its model-config ("m_*") keys, so the student's run -- which
        owns d_seq_len/d_pred_len/d_test_stride for the grid being iterated
        -- is never overwritten.
        '''
        for t_run in teacher_values.get("runs", []):
            if t_run.get("d_pred_len") == pred_len:
                return {k: v for k, v in t_run.items() if k.startswith("m_")}
        return {}

    # -------------------------
    # Run Experiments.
    # Seed Values = 2021, 2022, 2023
    #
    # The channel/window grid to iterate is driven by the student's runs;
    # the teacher shares the same seq_len/pred_len/test_stride for each run
    # so both models train on identical batches, while keeping its own
    # architecture (m_*) settings from teacher_config[default] (plus any
    # per-horizon override from its own runs -- see teacher_run_overrides).
    #
    # Order of Priority follows:
    # data_config[default] < gpu_config < {teacher,student}_config[default] < run
    # -------------------------
    for run in student_values["runs"]:
        for d_seed in range(2021, 2024):

            # Only the dataset/window keys (d_seq_len, d_pred_len, d_test_stride)
            # are shared with the teacher -- the student's own m_* overrides for
            # this run must not leak into the teacher's architecture.
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
            student_cfgs = ExpConfig(**student_merged)

            random.seed(d_seed)
            torch.manual_seed(d_seed)
            np.random.seed(d_seed)

            # --------------------------------
            # Set the GPU settings. Identical for teacher and student, since
            # both merges pull from the same gpu_config.
            # --------------------------------
            for cfgs in (teacher_cfgs, student_cfgs):
                cfgs.d_use_gpu = True if torch.cuda.is_available() and cfgs.d_use_gpu else False
                if cfgs.d_use_gpu and cfgs.d_use_multi_gpu:
                    cfgs.d_devices = [int(device_id) for device_id in cfgs.d_devices]
                    cfgs.d_gpu = cfgs.d_devices[0]

            # --------------------------------
            # Print the arguments for the experiment.
            # This is useful for logging and reproducibility.
            # --------------------------------
            print('Teacher configuration in experiment:')
            print(teacher_cfgs)
            print('Student configuration in experiment:')
            print(student_cfgs)

            setting = 'KD-{}-{}-{}-{}-ft{}-sl{}-pl{}-seed{}'.format(
                args.kd_mode,
                teacher_cfgs.d_model,
                student_cfgs.d_model,
                student_cfgs.d_data,
                student_cfgs.d_forecast_type,
                student_cfgs.d_seq_len,
                student_cfgs.d_pred_len,
                d_seed)

            teacher_cfgs.d_setting = setting
            student_cfgs.d_setting = setting

            # --------------------------------
            # Prepare directories. Teacher and student share the same
            # setting, so a single call covers both.
            # --------------------------------
            check_and_prepare_dirs(student_cfgs)

            # --------------------------------
            # RUN EXPERIMENT.
            # --------------------------------
            exp = Exp_KD_Class(teacher_cfgs, student_cfgs)

            # If training is enabled.
            if student_cfgs.d_is_training:
                    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                    exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            preds, trues = exp.test(setting)

            # If output needs to be inverse transformed.
            if student_cfgs.d_inverse_transform:
                preds = inverse_transform(student_cfgs, preds)
                trues = inverse_transform(student_cfgs, trues)

            # --------------------------------
            # Save and plot results (student's, the deliverable of a KD run).
            # --------------------------------
            save_results(student_cfgs, preds, trues)
            if student_cfgs.d_plot_results:
                plot_results(student_cfgs, preds, trues, zoom_to=400)

            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
