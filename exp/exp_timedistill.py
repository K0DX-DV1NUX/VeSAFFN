import torch
import torch.nn.functional as F

from exp.exp_kd import Exp_KD_Offline


def _multi_scale(y, num_scales, window):
    # y: [B, H, C] -> M+1 views [B, H / window^m, C], m = 0..M.
    x = y.permute(0, 2, 1)                              # [B, C, H]
    views = [x]
    for _ in range(num_scales):
        x = F.avg_pool1d(x, kernel_size=window)        # [B, C, H / window^m]
        views.append(x)
    return [v.permute(0, 2, 1) for v in views]


def _period_distribution(y, tau):
    # y: [B, H, C] -> softmax over amplitude spectrum without DC, [B, H/2, C].
    amp = torch.fft.rfft(y, dim=1).abs()[:, 1:, :]
    return F.softmax(amp / tau, dim=1)


def timedistill_prediction_loss(student_out, teacher_out, criterion, num_scales, window, tau):
    # Returns (multi-scale loss, period KL loss); the teacher is a fixed target.
    teacher_out = teacher_out.detach()

    student_views = _multi_scale(student_out, num_scales, window)
    teacher_views = _multi_scale(teacher_out, num_scales, window)
    scale_loss = sum(criterion(s, t) for s, t in zip(student_views, teacher_views)) / len(student_views)

    # +1e-8 keeps the log finite.
    period_loss = F.kl_div(
        torch.log(_period_distribution(student_out, tau) + 1e-8),
        _period_distribution(teacher_out, tau),
        reduction='mean',
    )

    return scale_loss, period_loss


class Exp_TimeDistill_Offline(Exp_KD_Offline):
    # Offline TimeDistill (arXiv:2502.15016), prediction-level terms only.
    # loss = L(y, student) + alpha * (L_scale + L_period)

    RESULT_PREFIX = "Result_TimeDistill"

    def _kd_loss(self, teacher_out, student_out, batch_y, criterion):
        args = self.student_args
        student_gt_loss = criterion(student_out, batch_y)

        # Computed in fp32.
        scale_loss, period_loss = timedistill_prediction_loss(
            student_out.float(), teacher_out.float(), criterion,
            args.d_td_scales, args.d_td_window, args.d_td_tau,
        )
        student_kd_loss = args.d_td_alpha * (scale_loss + period_loss)

        train_loss = student_gt_loss + student_kd_loss

        # Logged only, not backpropagated.
        teacher_gt_loss = criterion(teacher_out.detach(), batch_y)

        return train_loss, teacher_gt_loss, student_gt_loss, student_kd_loss
