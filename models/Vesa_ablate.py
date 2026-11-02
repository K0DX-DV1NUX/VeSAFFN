import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN
from models.DLinear import series_decomp


# Shapes: B = batch, L = seq_len, H = pred_len, C = channels,
# D = hidden dim (m_seq_dim), r = rank (m_vesa_rank), N = wavelet filter length.


class VesaLinear(nn.Module):
    # W = P_A[:in, :r] diag(lambda_a) P_B[:r, :out] diag(lambda_b), one set of scales for all channels.
    def __init__(self, in_features, out_features, shared_pa, shared_pb, bias=True, rank=None):
        super(VesaLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.shared_pa = shared_pa
        self.shared_pb = shared_pb
        self.rank = rank or shared_pa.shape[0]

        # Layer-specific scaling vectors.
        self.lambda_a = nn.Parameter(nn.init.ones_(torch.empty(self.rank)))
        self.lambda_b = nn.Parameter(nn.init.ones_(torch.empty(self.out_features)))
        if bias:
            self.bias = nn.Parameter(nn.init.zeros_(torch.empty(self.out_features)))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        # x: [B, 2C, in]
        pa = self.shared_pa[:self.in_features, :self.rank]     # [in, r]
        pb = self.shared_pb[:self.rank, :self.out_features]    # [r, out]

        Pa = pa * self.lambda_a                                # [in, r]
        Pb = pb * self.lambda_b                                # [r, out]
        x = x @ Pa @ Pb                                        # [B, 2C, in] -> [B, 2C, r] -> [B, 2C, out]
        if self.bias is not None:
            x = x + self.bias
        return x


class VesaLinearChannelwise(nn.Module):
    # Same as VesaLinear, but every channel has its own lambda_a, lambda_b and bias.
    def __init__(self, in_features, out_features, num_channels, shared_pa, shared_pb, bias=True, rank=None):
        super(VesaLinearChannelwise, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_channels = num_channels
        self.shared_pa = shared_pa
        self.shared_pb = shared_pb
        self.rank = rank or shared_pa.shape[1]

        # Per-channel scaling vectors.
        self.lambda_a = nn.Parameter(nn.init.ones_(torch.empty(num_channels, self.rank)))
        self.lambda_b = nn.Parameter(nn.init.ones_(torch.empty(num_channels, out_features)))
        if bias:
            self.bias = nn.Parameter(nn.init.zeros_(torch.empty(num_channels, out_features)))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        # x: [B, 2C, in]
        pa = self.shared_pa[:self.in_features, :self.rank]     # [in, r]
        pb = self.shared_pb[:self.rank, :self.out_features]    # [r, out]

        xp = x @ pa                                            # [B, 2C, r]
        xp = xp * self.lambda_a.unsqueeze(0)                   # [B, 2C, r]

        out = xp @ pb                                          # [B, 2C, out]
        out = out * self.lambda_b.unsqueeze(0)                 # [B, 2C, out]

        if self.bias is not None:
            out = out + self.bias.unsqueeze(0)
        return out


class WaveletDecomp(nn.Module):
    # One-level orthogonal DWT with periodic boundaries, as fixed conv1d filters from PyWavelets.
    n_bands = 2
    length_div = 2

    def __init__(self, wavelet):
        super(WaveletDecomp, self).__init__()
        wave = pywt.Wavelet(wavelet)
        if not wave.orthogonal:
            raise ValueError(f"Wavelet {wavelet} is not orthogonal, the transposed filters would not invert it.")

        # conv1d does not flip its kernel, so the filters go in time-reversed.
        filters = torch.tensor([wave.dec_lo[::-1], wave.dec_hi[::-1]], dtype=torch.float32)
        self.register_buffer("filters", filters.unsqueeze(1), persistent=False)  # [2, 1, N]
        self.pad = wave.dec_len // 2 - 1

    def forward(self, x):
        # x: [B, C, L]
        b, c, l = x.shape
        x = x.reshape(b * c, 1, l)                             # [BC, 1, L]
        x = F.pad(x, (self.pad, self.pad), mode="circular")    # [BC, 1, L + N - 2]
        coeffs = F.conv1d(x, self.filters, stride=2)           # [BC, 2, L/2]
        return coeffs.reshape(b, c, 2, -1).transpose(1, 2).reshape(b, 2 * c, -1)  # [B, 2C, L/2], approx then detail

    def inverse(self, coeffs):
        # coeffs: [B, 2C, H/2], approx then detail
        b, c2, h = coeffs.shape
        c = c2 // 2
        coeffs = coeffs.reshape(b, 2, c, h).transpose(1, 2).reshape(b * c, 2, h)  # [BC, 2, H/2]
        coeffs = F.pad(coeffs, (self.pad, self.pad), mode="circular")             # [BC, 2, H/2 + N - 2]
        out = F.conv_transpose1d(coeffs, self.filters, stride=2)                  # [BC, 1, H + 3N - 6]
        start = 3 * self.pad
        return out[..., start:start + 2 * h].reshape(b, c, 2 * h)                 # [B, C, H]


class TrendResidualDecomp(nn.Module):
    # DLinear moving-average split. Length is unchanged, so the inverse is a sum.
    n_bands = 2
    length_div = 1

    def __init__(self, kernel_size):
        super(TrendResidualDecomp, self).__init__()
        self.decomp = series_decomp(kernel_size)

    def forward(self, x):
        # x: [B, C, L]
        res, trend = self.decomp(x.permute(0, 2, 1))           # [B, L, C] each
        return torch.cat([res, trend], dim=2).permute(0, 2, 1)  # [B, 2C, L]

    def inverse(self, bands):
        # bands: [B, 2C, H]
        res, trend = bands.chunk(2, dim=1)                     # [B, C, H] each
        return res + trend                                     # [B, C, H]


class NoDecomp(nn.Module):
    n_bands = 1
    length_div = 1

    def forward(self, x):
        return x

    def inverse(self, bands):
        return bands


class Model(nn.Module):
    # VeSA-FFN ablation over the decomposition: wavelet, trend_residual or none.
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.d_seq_len
        self.pred_len = configs.d_pred_len
        self.in_features = configs.d_in_features
        self.out_features = configs.d_out_features

        self.m_individual = getattr(configs, "m_individual", False)
        self.seq_dim = getattr(configs, "m_seq_dim", 128)
        self.use_norm = getattr(configs, "m_use_norm", True)
        self.vesa_rank = getattr(configs, "m_vesa_rank", 16)
        self.decomp_type = getattr(configs, "m_decomp", "wavelet")

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

        self.ffn_in = self.seq_len // self.decomp.length_div
        self.ffn_out = self.pred_len // self.decomp.length_div
        ffn_channels = self.decomp.n_bands * self.in_features

        # Shared low-rank matrices, reused by both FFN layers.
        Pa = torch.empty(max(self.ffn_in, self.seq_dim), self.vesa_rank)
        Pb = torch.empty(self.vesa_rank, max(self.seq_dim, self.ffn_out))

        nn.init.xavier_uniform_(Pa)
        nn.init.xavier_uniform_(Pb)

        self.shared_pa = nn.Parameter(Pa)
        self.shared_pb = nn.Parameter(Pb)

        if self.use_norm:
            self.revin = RevIN(configs.d_in_features, eps=1e-5, affine=True)
        else:
            self.revin = None

        # FFN: ffn_in -> D -> ffn_out.
        if self.m_individual:
            self.layer1 = VesaLinearChannelwise(
                in_features=self.ffn_in,
                out_features=self.seq_dim,
                num_channels=ffn_channels,
                shared_pa=self.shared_pa,
                shared_pb=self.shared_pb,
                rank=self.vesa_rank,
            )
            self.act = nn.GELU()
            self.layer2 = VesaLinearChannelwise(
                in_features=self.seq_dim,
                out_features=self.ffn_out,
                num_channels=ffn_channels,
                shared_pa=self.shared_pa,
                shared_pb=self.shared_pb,
                rank=self.vesa_rank,
            )
        else:
            self.layer1 = VesaLinear(
                in_features=self.ffn_in,
                out_features=self.seq_dim,
                shared_pa=self.shared_pa,
                shared_pb=self.shared_pb,
                rank=self.vesa_rank
            )
            self.act = nn.GELU()
            self.layer2 = VesaLinear(
                in_features=self.seq_dim,
                out_features=self.ffn_out,
                shared_pa=self.shared_pa,
                shared_pb=self.shared_pb,
                rank=self.vesa_rank
            )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x_enc: [B, L, C]
        if self.use_norm:
            x = self.revin(x_enc, mode="norm")                 # [B, L, C]
        else:
            x = x_enc

        x = x.permute(0, 2, 1)                                 # [B, C, L]

        bands = self.decomp(x)                                 # [B, n_bands * C, ffn_in]

        # Input kept for feature-level distillation.
        self._kd_feature = x                                   # [B, C, L]

        bands_out = self.layer2(self.act(self.layer1(bands)))  # [B, n_bands * C, ffn_in] -> [..., D] -> [..., ffn_out]
        out = self.decomp.inverse(bands_out)                   # [B, C, H]

        out = out.permute(0, 2, 1)                             # [B, H, C]
        if self.use_norm:
            out = self.revin(out, mode="denorm")               # [B, H, C]

        return out
