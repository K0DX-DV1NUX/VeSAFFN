import torch
import torch.nn.functional as F

from exp.exp_kd import Exp_KD_Offline


def _multi_scale(y, num_scales, window):
    '''
    [B, S, C] -> list of M + 1 views [B, S / window^m, C], m = 0..M.
    Paper Eq. 3 writes the downsampler as a stride-2 conv; the official code
    uses a parameter-free AvgPool1d, which is what's reproduced here so the
    student gains no extra trainable parameters from distillation.
    '''
    x = y.permute(0, 2, 1)  # AvgPool1d pools over the last (time) dim
    views = [x]
    for _ in range(num_scales):
        x = F.avg_pool1d(x, kernel_size=window)
        views.append(x)
    return [v.permute(0, 2, 1) for v in views]


def _period_distribution(y, tau):
    '''
    Paper Eq. 7-8: amplitude of the rFFT along time, DC bin dropped, turned
    into a per-channel distribution over the S/2 remaining frequencies (i.e.
    over periods ceil(S/i)) by a temperature-tau softmax. [B, S, C] -> [B, S/2, C].
    '''
    amp = torch.fft.rfft(y, dim=1).abs()[:, 1:, :]
    return F.softmax(amp / tau, dim=1)


def timedistill_prediction_loss(student_out, teacher_out, criterion, num_scales, window, tau):
    '''
    TimeDistill's prediction-level distillation (paper Eq. 4 + Eq. 9):
        L_scale  = 1/(M+1) * sum_m  L(Y_t^m, Y_s^m)
        L_period = KL(Q_t || Q_s)
    Returns (L_scale, L_period). The teacher side is always treated as a
    constant target.
    '''
    teacher_out = teacher_out.detach()

    student_views = _multi_scale(student_out, num_scales, window)
    teacher_views = _multi_scale(teacher_out, num_scales, window)
    scale_loss = sum(criterion(s, t) for s, t in zip(student_views, teacher_views)) / len(student_views)

    # Matches the official code exactly rather than a pure log_softmax KL:
    # amplitudes grow with pred_len, so at tau=0.5 the softmax is very peaked
    # and the +1e-8 floor caps the student's log-prob at ~-18.4 per bin --
    # without it the KL can blow up and swamp L(gt, student).
    # reduction='mean' averages over every (batch, frequency, channel) entry,
    # also as in the official code, rather than a per-sample 'batchmean'.
    period_loss = F.kl_div(
        torch.log(_period_distribution(student_out, tau) + 1e-8),
        _period_distribution(teacher_out, tau),
        reduction='mean',
    )

    return scale_loss, period_loss


class Exp_TimeDistill_Offline(Exp_KD_Offline):
    '''
    Offline TimeDistill (Ni et al., 2025, arXiv:2502.15016): the teacher is
    frozen and pretrained, exactly as in Exp_KD_Offline, and only the
    student trains. Everything except the training loss -- allowed
    teachers/students, frozen-teacher loading, the one-off teacher output
    cache, optimizer, OneCycleLR schedule, validation on the student's
    ground-truth loss, early stopping and test -- is inherited unchanged
    from Exp_KD_Offline, so the two are directly comparable.

    The optimized objective replaces Exp_KD_Offline's
        L(gt, student) + L(student, teacher)
    with the paper's prediction-level objective (Eq. 12, alpha term):
        L(gt, student) + alpha * (L_scale + L_period)
    Hyperparameters are the student toml's d_td_* keys (see base/config.py,
    which also validates them).

    The paper's feature-level term (beta * (L_scale^H + L_period^H)) is not
    used: it needs intermediate hidden features from both models plus a
    trainable regressor aligning them, and neither student (Vesa, FITS) nor
    most teachers here (e.g. Chronos-2) expose a learned hidden
    representation to align -- so this is TimeDistill with beta = 0.
    '''

    RESULT_PREFIX = "Result_TimeDistill"

    def _kd_loss(self, teacher_out, student_out, batch_y, criterion):
        args = self.student_args
        student_gt_loss = criterion(student_out, batch_y)

        # FFT/softmax/KL in fp32 even under AMP -- half-precision rFFT is
        # both restricted (power-of-two lengths only) and numerically poor.
        scale_loss, period_loss = timedistill_prediction_loss(
            student_out.float(), teacher_out.float(), criterion,
            args.d_td_scales, args.d_td_window, args.d_td_tau,
        )
        student_kd_loss = args.d_td_alpha * (scale_loss + period_loss)

        train_loss = student_gt_loss + student_kd_loss

        teacher_gt_loss = criterion(teacher_out.detach(), batch_y)  # diagnostic only, not backpropagated

        return train_loss, teacher_gt_loss, student_gt_loss, student_kd_loss
