from contextlib import nullcontext
import os
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset

from data_provider.data_factory import TimeSeriesDataset
from models import Chronos2, DLinear, FITS, FreqMoE, PatchTST, iTransformer, FrNet, Vesa
from utils.learning_rates import adjust_learning_rate
from utils.metrics import metric


# Allowed online KD teachers: models that can be trained jointly with the
# student. Chronos-2 can never appear here -- it has no trainable weights of
# its own (see models/Chronos2.py, d_pretrained=true), so "training" it
# jointly is meaningless.
TEACHER_MODELS = {
    'PatchTST': PatchTST,
    'iTransformer': iTransformer,
    'FrNet': FrNet,
    'DLinear': DLinear,
    'FreqMoE': FreqMoE,
}

# Allowed offline KD teachers: everything online allows, plus Chronos-2,
# since offline mode never trains the teacher either way.
OFFLINE_TEACHER_MODELS = {
    **TEACHER_MODELS,
    'Chronos2': Chronos2,
}

STUDENT_MODELS = {
    'Vesa': Vesa,
    # FITS emits look-back + horizon; _forward keeps only the last pred_len steps.
    'FITS': FITS,
}


# -------------------------------------------------
# Free functions shared by Exp_KD (online) and Exp_KD_Offline. None of these
# need any per-experiment state beyond the args/device passed in, so they're
# plain functions rather than methods duplicated across both classes.
# -------------------------------------------------

def _acquire_device(args):
    if args.d_use_gpu:
        visible_devices = (
            str(args.d_gpu)
            if not args.d_use_multi_gpu
            else ",".join(str(device_id) for device_id in args.d_devices)
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
        device = torch.device('cuda:{}'.format(args.d_gpu))
        print('Use GPU: cuda:{}'.format(args.d_gpu))
    else:
        device = torch.device('cpu')
        print('Use CPU')
    return device


def _get_data(args, flag):
    timeenc = 0 if args.d_embed != 'timeF' else 1

    flag_list = {
        "train": {"shuffle": True,
                "drop_last": True,
                "stride": args.d_train_stride,
                "batch_size": args.d_batch_size},

        "val": {"shuffle": False,
                "drop_last": False,
                "stride": args.d_test_stride,
                "batch_size": args.d_batch_size},

        "test": {"shuffle": False,
                 "drop_last": False,
                 "stride": args.d_test_stride,
                 "batch_size": 1},
            }

    dataset = TimeSeriesDataset(
        args=args,
        timeenc=timeenc,
        flag=flag,
        stride=flag_list[flag]["stride"]
    )

    print(f"{flag} Dataset: ({len(dataset)}, {args.d_seq_len}, {args.d_in_features})")

    loader = DataLoader(
        dataset,
        batch_size=flag_list[flag]["batch_size"],
        shuffle=flag_list[flag]["shuffle"],
        num_workers=args.d_num_workers,
        drop_last=flag_list[flag]["drop_last"],
        pin_memory=True
    )

    return loader


def _select_criterion(args):
    losses_dict = {
        'mae': nn.L1Loss(),
        'mse': nn.MSELoss(),
        'smooth': nn.SmoothL1Loss(),
    }
    return losses_dict.get(args.d_loss, nn.MSELoss())


def _get_inputs(device, args, x, y, x_mark, y_mark):
    batch_x = x.float().to(device)
    batch_y = y.float().to(device)

    if 'PEMS' in args.d_data or 'Solar' in args.d_data:
        batch_x_mark = None
        batch_y_mark = None
    else:
        batch_x_mark = x_mark.float().to(device)
        batch_y_mark = y_mark.float().to(device)

    dec_inp = torch.zeros_like(y[:, -args.d_pred_len:, :]).float()
    dec_inp = torch.cat([y[:, :args.d_label_len, :], dec_inp], dim=1).float().to(device)

    return batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp


def _model_args(input_type, batch_x, batch_x_mark, dec_inp, batch_y_mark):
    input_type = input_type.lower()
    if input_type == "x_only":
        return (batch_x,)
    if input_type == "x_mark_incl":
        return (batch_x, batch_x_mark)
    return (batch_x, batch_x_mark, dec_inp, batch_y_mark)


def _forward(model, args, batch_x, batch_x_mark, dec_inp, batch_y_mark, amp_context):
    '''
    Runs one model's forward pass and slices it to the forecast horizon.
    Teacher and student each use their own d_model_input_type, since a
    transformer-style teacher (e.g. iTransformer) may need the mark/dec
    inputs a linear-style student (Vesa, FITS) does not.
    '''
    model_input_args = _model_args(args.d_model_input_type, batch_x, batch_x_mark, dec_inp, batch_y_mark)

    with amp_context:
        outputs = model(*model_input_args)

    if args.d_model_input_type.lower() not in ["x_only", "x_mark_incl"] and args.d_output_attention:
        outputs = outputs[0]

    f_dim = -1 if args.d_forecast_type == 'MS' else 0
    return outputs[:, -args.d_pred_len:, f_dim:]


class _IndexedDataset(Dataset):
    '''
    Wraps a dataset to also return each sample's own index. Used only for
    Exp_KD_Offline's train_loader, so that a frozen teacher's outputs --
    precomputed once, in dataset order -- can be looked up correctly for
    each sample no matter how the DataLoader shuffles batch order.
    '''
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (*self.dataset[idx], idx)


class _EarlyStoppingKD:
    '''
    Early stopping keyed off the student's ground-truth validation loss --
    the metric that ultimately matters for the distilled model. In online
    mode the teacher trains too, so pass it in to have its weights saved
    alongside the student's; in offline mode the teacher is fixed and lives
    outside the run (see Exp_KD_Offline), so it's omitted and only the
    student gets checkpointed.
    '''
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, student, path, teacher=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, student, path, teacher)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, student, path, teacher)
            self.counter = 0

    def save_checkpoint(self, val_loss, student, path, teacher=None):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        state = {"student": student.state_dict()}
        if teacher is not None:
            state["teacher"] = teacher.state_dict()
        torch.save(state, path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class Exp_KD:
    '''
    Online knowledge-distillation experiment: a teacher and a student model
    train together, batch by batch, on the same data, each through its own
    optimizer stepped from one shared backward pass. The loss is an
    unweighted sum of three terms:
        L(gt, student) + L(gt, teacher) + L(teacher, student)
    Ground truth supervises both models directly. The teacher-student term
    is NOT detached on either side -- gradients from it flow into both
    models at once, so this is genuine mutual learning: the teacher gets
    pulled toward the student's predictions just as much as the student
    gets pulled toward the teacher's, on top of each independently learning
    from ground truth.

    Unlike Exp_Main, this class is driven by two configs -- teacher_args and
    student_args -- built from each model's own model.toml. Training-loop
    level settings (batch size, epochs, dataset, learning rate, ...) are
    guaranteed identical between the two by construction (see runKD.py) and
    are read from student_args throughout.

    Optional teacher warm-up: if student_args.d_kd_teacher_warmup_epochs is
    set (default 0, i.e. today's "cold start" -- joint training from epoch
    0), the first N epochs train the teacher alone on L(gt, teacher); the
    student and the KD term sit out entirely (no forward pass, no
    validation, no early stopping) until the teacher has had a chance to
    reach its own accuracy. See train() for the epoch-level branch.

    The teacher and student each get their own OneCycleLR schedule rather
    than sharing one. The teacher's spans the full d_train_epochs (it
    trains every epoch, warm-up included). The student's spans only
    d_train_epochs - warmup_epochs and starts counting from the first
    joint-training epoch, so when the student joins the loss it begins at
    the same base_lr the teacher began at, instead of inheriting whatever
    LR a single shared schedule would already have decayed to after the
    warm-up epochs.
    '''

    def __init__(self, teacher_args, student_args):
        if teacher_args.d_model not in TEACHER_MODELS:
            raise ValueError(
                f"'{teacher_args.d_model}' is not an allowed online KD teacher. "
                f"Allowed teachers: {sorted(TEACHER_MODELS)}. "
                f"(Fixed/pretrained models like Chronos2 can only be used for offline KD.)"
            )
        if student_args.d_model not in STUDENT_MODELS:
            raise ValueError(
                f"'{student_args.d_model}' is not an allowed KD student. "
                f"Allowed students: {sorted(STUDENT_MODELS)}."
            )

        self.teacher_args = teacher_args
        self.student_args = student_args
        self.device = _acquire_device(self.student_args)

        if self.student_args.d_is_training:
            self.train_loader = _get_data(self.student_args, flag='train')
            self.vali_loader = _get_data(self.student_args, flag='val')
            self.test_loader = _get_data(self.student_args, flag='test')
        else:
            self.test_loader = _get_data(self.student_args, flag='test')

        self.teacher = TEACHER_MODELS[teacher_args.d_model].Model(teacher_args).float().to(self.device)
        self.student = STUDENT_MODELS[student_args.d_model].Model(student_args).float().to(self.device)

    def _select_optimizer(self):
        '''
        Separate optimizers per model, each stepped from the same combined
        loss -- this is what makes the distillation "online": the teacher
        keeps learning from ground truth in the same step that the student
        learns from it. Kept separate (rather than one optimizer over both
        models' params) so each model can run its own learning-rate
        schedule: the teacher trains for the whole run and its schedule
        should span all of d_train_epochs, while the student only starts
        training once the teacher's warm-up ends and needs its own
        schedule to start from base_lr at that point instead of picking up
        wherever the teacher's schedule happens to be (see train()).
        '''
        teacher_optim = optim.Adam(self.teacher.parameters(), lr=self.student_args.d_learning_rate)
        student_optim = optim.Adam(self.student.parameters(), lr=self.student_args.d_learning_rate)
        return teacher_optim, student_optim

    def _kd_loss(self, teacher_out, student_out, batch_y, criterion):
        '''
        All three terms are summed with equal weight. Neither side of the
        teacher-student term is detached -- gradients from it flow into
        both models, not just the student.
        '''
        student_gt_loss = criterion(student_out, batch_y)
        teacher_gt_loss = criterion(teacher_out, batch_y)
        student_kd_loss = criterion(teacher_out, student_out)

        train_loss = student_gt_loss + teacher_gt_loss + student_kd_loss
        return train_loss, teacher_gt_loss, student_gt_loss, student_kd_loss

    def vali(self, vali_loader, criterion):
        '''
        Validates on the student's ground-truth loss only -- the metric that
        determines early stopping and which checkpoint is kept.
        '''
        total_loss = []

        self.student.eval()

        with torch.no_grad():
            for _, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = _get_inputs(self.device, self.student_args, batch_x, batch_y, batch_x_mark, batch_y_mark)

                amp_context = torch.cuda.amp.autocast() if self.student_args.d_use_amp else nullcontext()

                student_out = _forward(self.student, self.student_args, batch_x, batch_x_mark, dec_inp, batch_y_mark, amp_context)

                f_dim = -1 if self.student_args.d_forecast_type == 'MS' else 0
                target = batch_y[:, -self.student_args.d_pred_len:, f_dim:].to(self.device)

                loss = criterion(student_out.detach().cpu(), target.detach().cpu())
                total_loss.append(loss)

        total_loss = np.average(total_loss)

        self.student.train()

        return total_loss

    def train(self, setting):
        path = os.path.join(self.student_args.d_checkpoint_path, self.student_args.d_setting)

        time_now = time.time()
        train_steps = len(self.train_loader)
        early_stopping = _EarlyStoppingKD(patience=self.student_args.d_patience, verbose=True)

        teacher_optim, student_optim = self._select_optimizer()
        criterion = _select_criterion(self.student_args)

        if self.student_args.d_use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # Teacher warm-up: for the first `warmup_epochs` epochs, only the
        # teacher trains (on L(gt, teacher)); the student and the KD term
        # sit out entirely. Defaults to 0 -- unset in a toml, this is
        # exactly today's cold-start joint training.
        warmup_epochs = int(getattr(self.student_args, "d_kd_teacher_warmup_epochs", 0))
        if warmup_epochs >= self.student_args.d_train_epochs:
            print(f"Warning: d_kd_teacher_warmup_epochs ({warmup_epochs}) >= d_train_epochs "
                  f"({self.student_args.d_train_epochs}); clamping so at least 1 epoch of "
                  f"joint KD training runs.")
            warmup_epochs = max(0, self.student_args.d_train_epochs - 1)

        # The teacher's schedule spans the whole run (it trains every
        # epoch, warm-up included). The student's schedule spans only the
        # epochs it actually trains for -- it is stepped for the first
        # time on the first joint-training epoch, so it starts at the same
        # base_lr the teacher started at, instead of inheriting whatever
        # LR the teacher's schedule has decayed to by then.
        teacher_scheduler = lr_scheduler.OneCycleLR(optimizer=teacher_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.student_args.d_pct_start,
                                            epochs=self.student_args.d_train_epochs,
                                            max_lr=self.student_args.d_learning_rate)
        student_scheduler = lr_scheduler.OneCycleLR(optimizer=student_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.student_args.d_pct_start,
                                            epochs=self.student_args.d_train_epochs - warmup_epochs,
                                            max_lr=self.student_args.d_learning_rate)

        self.teacher.train()
        self.student.train()

        for epoch in range(self.student_args.d_train_epochs):
            in_warmup = epoch < warmup_epochs
            iter_count = 0
            train_loss = []
            self.teacher.train()
            self.student.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.train_loader):
                iter_count += 1

                teacher_optim.zero_grad(set_to_none=True)
                student_optim.zero_grad(set_to_none=True)

                batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = _get_inputs(self.device, self.student_args, batch_x, batch_y, batch_x_mark, batch_y_mark)

                amp_context = torch.cuda.amp.autocast() if self.student_args.d_use_amp else nullcontext()

                teacher_out = _forward(self.teacher, self.teacher_args, batch_x, batch_x_mark, dec_inp, batch_y_mark, amp_context)

                f_dim = -1 if self.student_args.d_forecast_type == 'MS' else 0
                target = batch_y[:, -self.student_args.d_pred_len:, f_dim:].to(self.device)

                if in_warmup:
                    # No student forward, no KD term -- the student is
                    # untouched until the teacher's warm-up finishes.
                    loss = teacher_gt_loss = criterion(teacher_out, target)
                    student_gt_loss = teacher_student_loss = None
                else:
                    student_out = _forward(self.student, self.student_args, batch_x, batch_x_mark, dec_inp, batch_y_mark, amp_context)
                    loss, teacher_gt_loss, student_gt_loss, teacher_student_loss = self._kd_loss(teacher_out, student_out, target, criterion)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.student_args.d_train_epochs - epoch) * train_steps - i)
                    if in_warmup:
                        print("\t[teacher warm-up {0}/{1}] iters: {2}, epoch: {3} | teacher_gt loss: {4:.7f}".format(
                            epoch + 1, warmup_epochs, i + 1, epoch + 1, teacher_gt_loss.item()))
                    else:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f} (student_gt: {3:.7f}, teacher_gt: {4:.7f}, teacher_student: {5:.7f})".format(
                            i + 1, epoch + 1, loss.item(), student_gt_loss.item(), teacher_gt_loss.item(), teacher_student_loss.item()))
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.student_args.d_use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(teacher_optim)
                    if not in_warmup:
                        scaler.step(student_optim)
                    scaler.update()
                else:
                    loss.backward()
                    teacher_optim.step()
                    if not in_warmup:
                        student_optim.step()

                if self.student_args.d_lradj == 'TST':
                    adjust_learning_rate(teacher_optim, teacher_scheduler, epoch + 1, self.student_args, printout=False)
                    teacher_scheduler.step()
                    if not in_warmup:
                        student_epoch = epoch - warmup_epochs + 1
                        adjust_learning_rate(student_optim, student_scheduler, student_epoch, self.student_args, printout=False)
                        student_scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            if in_warmup:
                # Student hasn't trained yet -- skip validation/early
                # stopping so the warm-up window doesn't eat into the
                # patience budget meant for the joint phase.
                print("\nEpoch: {0}, Steps: {1} | Teacher warm-up Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))
            else:
                vali_loss = self.vali(self.vali_loader, criterion)
                test_loss = self.vali(self.test_loader, criterion)

                print("\nEpoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))

                early_stopping(vali_loss, self.student, path, teacher=self.teacher)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            if self.student_args.d_lradj != 'TST':
                adjust_learning_rate(teacher_optim, teacher_scheduler, epoch + 1, self.student_args, printout=False)
                print('Updating teacher learning rate to {}'.format(teacher_optim.param_groups[0]['lr']))
                if not in_warmup:
                    student_epoch = epoch - warmup_epochs + 1
                    adjust_learning_rate(student_optim, student_scheduler, student_epoch, self.student_args, printout=False)
                    print('Updating student learning rate to {}'.format(student_optim.param_groups[0]['lr']))
            else:
                print('Updating teacher learning rate to {}'.format(teacher_scheduler.get_last_lr()[0]))
                if not in_warmup:
                    print('Updating student learning rate to {}'.format(student_scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        checkpoint = torch.load(best_model_path, map_location="cpu")
        self.teacher.load_state_dict(checkpoint["teacher"])
        self.student.load_state_dict(checkpoint["student"])

        return self.teacher, self.student

    @torch.no_grad()
    def test(self, setting):
        '''
        Assesses the student only. The teacher exists purely to help train
        the student (see _kd_loss); it plays no part in validation or test
        -- the student alone produces the final predictions.
        '''
        if not self.student_args.d_is_training:
            print('loading model')
            checkpoint = torch.load(
                os.path.join(self.student_args.d_checkpoint_path, setting, 'checkpoint.pth'), map_location="cpu")
            self.teacher.load_state_dict(checkpoint["teacher"])
            self.student.load_state_dict(checkpoint["student"])

        preds, trues = [], []

        self.student.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = _get_inputs(self.device, self.student_args, batch_x, batch_y, batch_x_mark, batch_y_mark)

                student_out = _forward(self.student, self.student_args, batch_x, batch_x_mark, dec_inp, batch_y_mark, nullcontext())

                f_dim = -1 if self.student_args.d_forecast_type == 'MS' else 0
                batch_y = batch_y[:, -self.student_args.d_pred_len:, f_dim:].to(self.device)

                preds.append(student_out.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        total_test_len = self.test_loader.dataset._data_len()
        valid_pred_len = total_test_len - self.student_args.d_seq_len

        def _flatten(arr):
            # flatten all windows except the last
            flat = arr[:-1].reshape(-1, arr.shape[-1])
            remaining = valid_pred_len - flat.shape[0]
            if remaining > 0:
                # need extra values from last window
                flat = np.concatenate([flat, arr[-1][-remaining:]], axis=0)
            elif remaining < 0:
                # too many values, need to trim tail
                flat = flat[:valid_pred_len]
            return flat

        flat_preds = _flatten(preds)
        flat_trues = _flatten(trues)

        mae, mse, rmse, mape, mspe, rse, corr, r2 = metric(flat_preds, flat_trues)

        print('[student] mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))

        with open(f"Result_{self.teacher_args.d_model}_{self.student_args.d_model}.csv", 'a') as f:
            f.write(f"{setting}-mse:{mse}-mae:{mae}\n")

        return flat_preds, flat_trues


def _frozen_teacher_checkpoint_path(model_name, dataset, pred_len):
    '''
    A fixed teacher's checkpoint is keyed by (model, dataset, pred_len), not
    just model: most architectures' output layer is shaped by pred_len and
    their input width by the dataset's channel count, so a single flat
    checkpoint per model can't cover the full run grid. See
    scripts/promote_kd_checkpoints.py for populating this layout from a
    normal (non-KD) training run.
    '''
    return os.path.join("scripts", model_name, "pre_trained", dataset, f"pl{pred_len}", "checkpoint.pth")


class Exp_KD_Offline:
    '''
    Offline knowledge-distillation experiment: the teacher is fixed and
    never trains here. It is either a genuinely pretrained, self-contained
    model (Chronos-2 -- see models/Chronos2.py, d_pretrained=true, which
    loads its own frozen weights internally) or a previously-trained
    checkpoint for PatchTST/iTransformer/FrNet/DLinear placed at
    scripts/<Model>/pre_trained/<Dataset>/pl<PredLen>/checkpoint.pth. Only
    the student's parameters are ever optimized.

    The optimized objective is an unweighted sum of two terms:
        L(gt, student) + L(student, teacher.detach())
    teacher_gt_loss (L(gt, teacher)) is also computed and reported every
    logging step, purely as a diagnostic of the fixed teacher's own
    accuracy -- it is never added to the backpropagated loss. Since the
    teacher is frozen, that term has zero gradient with respect to every
    trainable parameter; including it would only add a constant to the
    reported loss value while paying the memory cost of building an
    autograd graph through the teacher for nothing.
    '''

    # Subclasses with a different loss (e.g. Exp_TimeDistill_Offline) override
    # this so their results don't land in the same CSV as basic offline KD.
    RESULT_PREFIX = "Result"

    def __init__(self, teacher_args, student_args):
        if teacher_args.d_model not in OFFLINE_TEACHER_MODELS:
            raise ValueError(
                f"'{teacher_args.d_model}' is not an allowed offline KD teacher. "
                f"Allowed teachers: {sorted(OFFLINE_TEACHER_MODELS)}."
            )
        if student_args.d_model not in STUDENT_MODELS:
            raise ValueError(
                f"'{student_args.d_model}' is not an allowed KD student. "
                f"Allowed students: {sorted(STUDENT_MODELS)}."
            )

        self.teacher_args = teacher_args
        self.student_args = student_args
        self.device = _acquire_device(self.student_args)

        if self.student_args.d_is_training:
            self.train_loader = _get_data(self.student_args, flag='train')
            self.vali_loader = _get_data(self.student_args, flag='val')
            self.test_loader = _get_data(self.student_args, flag='test')
        else:
            self.test_loader = _get_data(self.student_args, flag='test')

        self.teacher = self._build_frozen_teacher().to(self.device)
        self.student = STUDENT_MODELS[student_args.d_model].Model(student_args).float().to(self.device)

        if self.student_args.d_is_training:
            # The teacher is frozen and never trains in offline mode, so its
            # output for a given window is the same in every epoch. Compute
            # it once here (in dataset order) instead of re-running the
            # teacher on every training iteration of every epoch -- for a
            # cheap teacher (DLinear, etc.) that redundant work was easy to
            # miss, but for a full pretrained model like Chronos-2 it's the
            # difference between one pass over the data and one pass per
            # epoch. train_loader is then rebuilt over an index-wrapped
            # dataset so shuffled batches can still look up the right
            # cached row per sample.
            indexed_train_dataset = _IndexedDataset(self.train_loader.dataset)
            self.teacher_cache = self._build_teacher_cache(indexed_train_dataset)
            self.train_loader = DataLoader(
                indexed_train_dataset,
                batch_size=self.student_args.d_batch_size,
                shuffle=True,
                num_workers=self.student_args.d_num_workers,
                drop_last=True,
                pin_memory=True,
            )

    def _build_frozen_teacher(self):
        '''
        Chronos-2 (d_pretrained=true) already loads its own fixed weights
        internally -- nothing more to do. Every other allowed teacher needs
        a locally-trained checkpoint placed at
        scripts/<Model>/pre_trained/<Dataset>/pl<PredLen>/checkpoint.pth
        (see the module-level docstring and scripts/promote_kd_checkpoints.py);
        this is where "fixing" a normally-trainable model like PatchTST into
        a static offline teacher happens, for this exact (dataset, pred_len).
        '''
        args = self.teacher_args
        model = OFFLINE_TEACHER_MODELS[args.d_model].Model(args).float()

        if not args.d_pretrained:
            checkpoint_path = _frozen_teacher_checkpoint_path(args.d_model, args.d_data, args.d_pred_len)
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"Offline KD needs a fixed teacher checkpoint for '{args.d_model}' on "
                    f"'{args.d_data}' at pred_len={args.d_pred_len}, at '{checkpoint_path}' "
                    f"(or set d_pretrained=true in its model.toml for a self-contained model "
                    f"like Chronos2). Train it normally first for a single seed (e.g. "
                    f"`python runExp.py ... --seed 2021`), then run "
                    f"scripts/promote_kd_checkpoints.py to place its checkpoint there."
                )
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()

        return model

    @torch.no_grad()
    def _build_teacher_cache(self, indexed_dataset):
        '''
        Runs the frozen teacher exactly once over every window in
        `indexed_dataset`, in a single unshuffled pass, and returns a
        tensor of its outputs ordered by dataset index. train() then
        indexes into this instead of calling the teacher at all.
        '''
        cache_loader = DataLoader(
            indexed_dataset,
            batch_size=self.student_args.d_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.student_args.d_num_workers,
            pin_memory=True,
        )

        outputs = [None] * len(indexed_dataset)
        for batch_x, batch_y, batch_x_mark, batch_y_mark, idx in cache_loader:
            batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = _get_inputs(
                self.device, self.student_args, batch_x, batch_y, batch_x_mark, batch_y_mark
            )
            teacher_out = _forward(
                self.teacher, self.teacher_args, batch_x, batch_x_mark, dec_inp, batch_y_mark, nullcontext()
            )
            for pos, i in enumerate(idx.tolist()):
                outputs[i] = teacher_out[pos].detach().cpu()

        return torch.stack(outputs, dim=0)

    def _select_optimizer(self):
        # The teacher is fixed -- only the student's parameters are trained.
        return optim.Adam(self.student.parameters(), lr=self.student_args.d_learning_rate)

    def _kd_loss(self, teacher_out, student_out, batch_y, criterion):
        student_gt_loss = criterion(student_out, batch_y)
        student_kd_loss = criterion(student_out, teacher_out)
        train_loss = student_gt_loss + student_kd_loss

        teacher_gt_loss = criterion(teacher_out.detach(), batch_y)  # diagnostic only, not backpropagated

        return train_loss, teacher_gt_loss, student_gt_loss, student_kd_loss


    def vali(self, vali_loader, criterion):
        '''
        Validates on the student's ground-truth loss only -- the metric that
        determines early stopping and which checkpoint is kept. Mirrors
        Exp_KD.vali (the teacher is fixed here anyway, so there's nothing
        for it to do during validation).
        '''
        total_loss = []

        self.student.eval()

        with torch.no_grad():
            for _, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = _get_inputs(self.device, self.student_args, batch_x, batch_y, batch_x_mark, batch_y_mark)

                amp_context = torch.cuda.amp.autocast() if self.student_args.d_use_amp else nullcontext()

                student_out = _forward(self.student, self.student_args, batch_x, batch_x_mark, dec_inp, batch_y_mark, amp_context)

                f_dim = -1 if self.student_args.d_forecast_type == 'MS' else 0
                target = batch_y[:, -self.student_args.d_pred_len:, f_dim:].to(self.device)

                loss = criterion(student_out.detach().cpu(), target.detach().cpu())
                total_loss.append(loss)

        total_loss = np.average(total_loss)

        self.student.train()

        return total_loss

    def train(self, setting):
        path = os.path.join(self.student_args.d_checkpoint_path, self.student_args.d_setting)

        time_now = time.time()
        train_steps = len(self.train_loader)
        early_stopping = _EarlyStoppingKD(patience=self.student_args.d_patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = _select_criterion(self.student_args)

        if self.student_args.d_use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.student_args.d_pct_start,
                                            epochs=self.student_args.d_train_epochs,
                                            max_lr=self.student_args.d_learning_rate)

        # The teacher is fixed -- .eval() once, up front, and never touched again.
        self.teacher.eval()
        self.student.train()

        for epoch in range(self.student_args.d_train_epochs):
            iter_count = 0
            train_loss = []
            self.student.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, idx) in enumerate(self.train_loader):
                iter_count += 1

                model_optim.zero_grad(set_to_none=True)

                batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = _get_inputs(self.device, self.student_args, batch_x, batch_y, batch_x_mark, batch_y_mark)

                amp_context = torch.cuda.amp.autocast() if self.student_args.d_use_amp else nullcontext()

                # Precomputed once in __init__ (see _build_teacher_cache) --
                # the teacher is frozen, so its output for this window was
                # already known before training started.
                teacher_out = self.teacher_cache[idx].to(self.device)
                student_out = _forward(self.student, self.student_args, batch_x, batch_x_mark, dec_inp, batch_y_mark, amp_context)

                f_dim = -1 if self.student_args.d_forecast_type == 'MS' else 0
                target = batch_y[:, -self.student_args.d_pred_len:, f_dim:].to(self.device)

                loss, teacher_gt_loss, student_gt_loss, student_kd_loss = self._kd_loss(teacher_out, student_out, target, criterion)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.student_args.d_train_epochs - epoch) * train_steps - i)
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f} (student_gt: {3:.7f}, student_kd: {4:.7f}, teacher_gt[diagnostic]: {5:.7f})".format(
                        i + 1, epoch + 1, loss.item(), student_gt_loss.item(), student_kd_loss.item(), teacher_gt_loss.item()))
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.student_args.d_use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.student_args.d_lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.student_args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(self.vali_loader, criterion)
            test_loss = self.vali(self.test_loader, criterion)

            print("\nEpoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            # No teacher= arg: the teacher never changes in offline mode, so
            # there's no point re-saving its (possibly hundreds of MB of)
            # weights on every improved checkpoint -- it's already pinned
            # externally at scripts/<Model>/pre_trained/.
            early_stopping(vali_loss, self.student, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.student_args.d_lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.student_args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        checkpoint = torch.load(best_model_path, map_location="cpu")
        self.student.load_state_dict(checkpoint["student"])

        return self.teacher, self.student

    @torch.no_grad()
    def test(self, setting):
        '''
        Assesses the student only, exactly like Exp_KD.test. The teacher is
        fixed and was already loaded once in __init__, so there's nothing
        to reload for it here.
        '''
        if not self.student_args.d_is_training:
            print('loading model')
            checkpoint = torch.load(
                os.path.join(self.student_args.d_checkpoint_path, setting, 'checkpoint.pth'), map_location="cpu")
            self.student.load_state_dict(checkpoint["student"])

        preds, trues = [], []

        self.student.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = _get_inputs(self.device, self.student_args, batch_x, batch_y, batch_x_mark, batch_y_mark)

                student_out = _forward(self.student, self.student_args, batch_x, batch_x_mark, dec_inp, batch_y_mark, nullcontext())

                f_dim = -1 if self.student_args.d_forecast_type == 'MS' else 0
                batch_y = batch_y[:, -self.student_args.d_pred_len:, f_dim:].to(self.device)

                preds.append(student_out.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        total_test_len = self.test_loader.dataset._data_len()
        valid_pred_len = total_test_len - self.student_args.d_seq_len

        def _flatten(arr):
            flat = arr[:-1].reshape(-1, arr.shape[-1])
            remaining = valid_pred_len - flat.shape[0]
            if remaining > 0:
                flat = np.concatenate([flat, arr[-1][-remaining:]], axis=0)
            elif remaining < 0:
                flat = flat[:valid_pred_len]
            return flat

        flat_preds = _flatten(preds)
        flat_trues = _flatten(trues)

        mae, mse, rmse, mape, mspe, rse, corr, r2 = metric(flat_preds, flat_trues)

        print('[student] mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))

        with open(f"{self.RESULT_PREFIX}_{self.teacher_args.d_model}_{self.student_args.d_model}.csv", 'a') as f:
            f.write(f"{setting}-mse:{mse}-mae:{mae}\n")

        return flat_preds, flat_trues
