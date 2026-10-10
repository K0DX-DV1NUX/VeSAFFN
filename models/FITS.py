# FITS: Frequency Interpolation Time Series forecasting.
# Source: https://github.com/VEWOXIC/FITS

import torch
import torch.nn as nn


class Model(nn.Module):
    # Returns look-back + forecast, [B, L+H, C].
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.d_seq_len
        self.pred_len = configs.d_pred_len
        self.individual = configs.m_individual
        self.channels = configs.d_in_features

        # Cut-off frequency: (L // base_T + 1) * h_order + 10, unless m_cut_freq is set.
        cut_freq = getattr(configs, "m_cut_freq", 0)
        if not cut_freq:
            cut_freq = (self.seq_len // configs.m_base_T + 1) * configs.m_h_order + 10
        self.dominance_freq = min(int(cut_freq), self.seq_len // 2 + 1)
        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len

        up_len = int(self.dominance_freq * self.length_ratio)
        if self.individual:
            self.freq_upsampler = nn.ModuleList()
            for i in range(self.channels):
                self.freq_upsampler.append(nn.Linear(self.dominance_freq, up_len).to(torch.cfloat))
        else:
            self.freq_upsampler = nn.Linear(self.dominance_freq, up_len).to(torch.cfloat)

    def forward(self, x):
        # x: [B, L, C]
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        x = x / torch.sqrt(x_var)                                  # [B, L, C]

        # Low-pass filter in frequency domain.
        low_specx = torch.fft.rfft(x, dim=1)                       # [B, L/2+1, C]
        low_specx = low_specx[:, 0:self.dominance_freq, :]         # [B, F, C]

        # Complex upsampling F -> F * (L+H)/L.
        if self.individual:
            low_specxy_ = torch.zeros([low_specx.size(0), int(self.dominance_freq * self.length_ratio), low_specx.size(2)], dtype=low_specx.dtype).to(low_specx.device)
            for i in range(self.channels):
                low_specxy_[:, :, i] = self.freq_upsampler[i](low_specx[:, :, i])
        else:
            low_specxy_ = self.freq_upsampler(low_specx.permute(0, 2, 1)).permute(0, 2, 1)  # [B, F', C]

        # Zero-pad to full spectrum and return to time domain.
        low_specxy = torch.zeros([low_specxy_.size(0), int((self.seq_len + self.pred_len) / 2 + 1), low_specxy_.size(2)], dtype=low_specxy_.dtype).to(low_specxy_.device)  # [B, (L+H)/2+1, C]
        low_specxy[:, 0:low_specxy_.size(1), :] = low_specxy_
        low_xy = torch.fft.irfft(low_specxy, dim=1)                # [B, L+H, C]
        low_xy = low_xy * self.length_ratio

        xy = low_xy * torch.sqrt(x_var) + x_mean                   # [B, L+H, C]
        return xy
