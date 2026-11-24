import math

import torch
import torch.nn as nn
from layers.RevIN import RevIN
from layers.Decomposition import NoDecomp, TrendResidualDecomp, WaveletDecomp

# DLinear_ablate: DLinear with a swappable decomposition, for wavelet vs trend-residual.
# One linear head per band, like DLinear's seasonal and trend heads.
#   m_decomp = "trend_residual": moving-average split, heads L -> H, outputs summed (= DLinear).
#   m_decomp = "wavelet":        one-level DWT, heads L/2 -> H/2 on the coefficients, then inverse DWT.
#   m_decomp = "none":           a single L -> H head.

# Shapes: B = batch, L = seq_len, H = pred_len, C = channels, K = n_bands,
# Lb = L / length_div, Hb = H / length_div.


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.d_seq_len
        self.pred_len = configs.d_pred_len
        self.channels = configs.d_in_features

        self.individual = bool(getattr(configs, "m_individual", True))
        self.use_norm = bool(getattr(configs, "m_use_norm", False))
        self.decomp_type = getattr(configs, "m_decomp", "trend_residual")

        if self.decomp_type == "wavelet":
            if self.seq_len % 2 or self.pred_len % 2:
                raise ValueError(f"Wavelet decomposition needs even lengths, got {self.seq_len} and {self.pred_len}.")
            self.decomp = WaveletDecomp(getattr(configs, "m_wavelet", "db1"))
        elif self.decomp_type == "trend_residual":
            self.decomp = TrendResidualDecomp(getattr(configs, "m_moving_avg", 25))
        elif self.decomp_type == "none":
            self.decomp = NoDecomp()
        else:
            raise ValueError(f"Unknown m_decomp: {self.decomp_type}")

        self.n_bands = self.decomp.n_bands
        self.band_in = self.seq_len // self.decomp.length_div
        self.band_out = self.pred_len // self.decomp.length_div

        # Per-band heads, with DLinear's init: weights 1 / in, bias as nn.Linear.
        # individual: [K, C, Hb, Lb] (one head per band and channel), else [K, Hb, Lb].
        head_shape = (self.n_bands, self.channels) if self.individual else (self.n_bands,)
        self.weight = nn.Parameter(torch.full((*head_shape, self.band_out, self.band_in), 1.0 / self.band_in))
        bound = 1.0 / math.sqrt(self.band_in)
        self.bias = nn.Parameter(torch.empty(*head_shape, self.band_out).uniform_(-bound, bound))

        self.revin = RevIN(self.channels, eps=1e-5, affine=True) if self.use_norm else None

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x_enc: [B, L, C]
        if self.channels != x_enc.shape[-1]:
            raise ValueError(f"Model Configuration does not match the dataset. Number of channels in the dataset: {x_enc.shape[-1]}, number of channels expected by the model: {self.channels}")

        x = self.revin(x_enc, mode="norm") if self.use_norm else x_enc   # [B, L, C]
        x = x.permute(0, 2, 1)                                            # [B, C, L]

        bands = self.decomp(x)                                            # [B, K*C, Lb], band-major
        b = bands.shape[0]
        bands = bands.reshape(b, self.n_bands, self.channels, self.band_in)  # [B, K, C, Lb]

        if self.individual:
            out = torch.einsum("bkcl,kchl->bkch", bands, self.weight) + self.bias          # [B, K, C, Hb]
        else:
            out = torch.einsum("bkcl,khl->bkch", bands, self.weight) + self.bias[:, None]  # [B, K, C, Hb]

        out = out.reshape(b, self.n_bands * self.channels, self.band_out)  # [B, K*C, Hb]
        out = self.decomp.inverse(out)                                     # [B, C, H]
        out = out.permute(0, 2, 1)                                         # [B, H, C]

        if self.use_norm:
            out = self.revin(out, mode="denorm")                           # [B, H, C]
        return out
