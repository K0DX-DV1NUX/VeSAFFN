# TimeEmb: static-dynamic disentanglement with global time embeddings and frequency filtering.
# Paper: Xia et al., "TimeEmb: A Lightweight Static-Dynamic Disentanglement Framework for Time Series
# Forecasting", arXiv:2510.00461. Source: https://github.com/showmeon/TimeEmb
#
# Shapes: B = batch, L = seq_len, H = pred_len, C = channels, F = L // 2 + 1 rFFT bins.
# Input type "x_index": forward(x, time_index), time_index [B, 3] = (position, hour, weekday) at the first
# forecast step (data_provider TIME_INDEX_COLS), as upstream's hour_index / day_index.
#
# Static part: embeddings E_hour[hour % m_hour_length], E_day[weekday % m_day_length], each [C, F], subtracted
# from the real part of the window's spectrum. Dynamic part: the remainder is filtered by rfft(w), then the
# embeddings are added back, the spectrum is inverted and an MLP maps L -> H.
#
# Training loss (upstream exp_main), via d_auxillary_loss = true and aux_loss below:
#   m_rec_lambda * MSE(y_hat, y) + m_auxi_lambda * mean |FFT_t(y_hat) - FFT_t(y)|
# Validation and test use d_loss as for every other model.

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.d_seq_len
        self.pred_len = configs.d_pred_len
        self.enc_in = configs.d_in_features
        self.d_model = getattr(configs, "m_d_model", 512)
        self.use_revin = bool(getattr(configs, "m_use_revin", True))
        self.use_hour_index = bool(getattr(configs, "m_use_hour_index", True))
        self.use_day_index = bool(getattr(configs, "m_use_day_index", False))
        self.hour_length = getattr(configs, "m_hour_length", 24)
        self.day_length = getattr(configs, "m_day_length", 7)

        # Loss weights for aux_loss (upstream rec_lambda / auxi_lambda).
        self.rec_lambda = getattr(configs, "m_rec_lambda", 0.0)
        self.auxi_lambda = getattr(configs, "m_auxi_lambda", 1.0)
        assert self.rec_lambda > 0 or self.auxi_lambda > 0, "m_rec_lambda and m_auxi_lambda cannot both be 0."

        self.scale = 0.02
        n_freq = self.seq_len // 2 + 1

        # MLP over time, L -> H, shared across channels.
        self.model = nn.Sequential(
            nn.Linear(self.seq_len, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.pred_len),
        )

        # Static embeddings in the frequency domain, zeros at init. Only the enabled ones are created,
        # so the parameter count matches what the forward pass uses.
        if self.use_hour_index:
            self.emb_hour = nn.Parameter(torch.zeros(self.hour_length, self.enc_in, n_freq))  # [hour_length, C, F]
        else:
            self.register_parameter("emb_hour", None)
        if self.use_day_index:
            self.emb_day = nn.Parameter(torch.zeros(self.day_length, self.enc_in, n_freq))    # [day_length, C, F]
        else:
            self.register_parameter("emb_day", None)

        # Learnable time-domain filter, used through its rFFT.
        self.w = nn.Parameter(self.scale * torch.randn(1, self.seq_len))                     # [1, L]

    def forward(self, x, time_index):
        # x: [B, L, C], time_index: [B, 3]
        # Instance norm.
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)                    # [B, 1, C]
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5               # [B, 1, C]
            x = (x - seq_mean) / torch.sqrt(seq_var)

        x = x.permute(0, 2, 1)                                               # [B, C, L]
        x_freq = torch.fft.rfft(x, dim=2, norm="ortho")                      # [B, C, F] complex
        w = torch.fft.rfft(self.w, dim=1, norm="ortho")                      # [1, F] complex

        # Remove the static part (real part only).
        x_real, x_imag = x_freq.real, x_freq.imag                            # [B, C, F] each
        if self.use_hour_index:
            emb_hour = self.emb_hour[time_index[:, 1] % self.hour_length]    # [B, C, F]
            x_real = x_real - emb_hour
        if self.use_day_index:
            emb_day = self.emb_day[time_index[:, 2] % self.day_length]       # [B, C, F]
            x_real = x_real - emb_day

        # Filter the dynamic part.
        y = torch.complex(x_real, x_imag) * w                                # [B, C, F]

        # Add the static part back (real part only).
        y_real, y_imag = y.real, y.imag
        if self.use_day_index:
            y_real = y_real + emb_day
        if self.use_hour_index:
            y_real = y_real + emb_hour

        y = torch.fft.irfft(torch.complex(y_real, y_imag), n=self.seq_len, dim=2, norm="ortho")  # [B, C, L]
        y = self.model(y).permute(0, 2, 1)                                   # [B, H, C]

        # Instance denorm.
        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        return y

    def aux_loss(self, outputs, batch_y, **_):
        # Upstream training loss: rec_lambda * MSE + auxi_lambda * MAE of the complex FFT difference over time
        # (auxi_mode = fft, auxi_type = complex, auxi_loss = MAE, module_first = 1).
        # outputs, batch_y: [B, H, C]
        loss = 0.0
        if self.rec_lambda:
            loss = loss + self.rec_lambda * F.mse_loss(outputs, batch_y)
        if self.auxi_lambda:
            diff = torch.fft.fft(outputs, dim=1) - torch.fft.fft(batch_y, dim=1)  # [B, H, C] complex
            loss = loss + self.auxi_lambda * diff.abs().mean()
        return loss
