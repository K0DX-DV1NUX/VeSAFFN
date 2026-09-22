import math
import torch
import torch.nn as nn
from layers.RevIN import RevIN


# Orthonormal low-pass analysis filters. The high-pass filter is derived as the
# quadrature mirror: g[n] = (-1)^n * h[len(h) - 1 - n].
_WAVELET_FILTERS = {
    "haar": [1 / math.sqrt(2), 1 / math.sqrt(2)],
    "db2": [0.4829629131445341, 0.8365163037378079, 0.2241438680420134, -0.1294095225512604],
    "db4": [
        0.23037781330885523, 0.7148465705525415, 0.6308807679295904, -0.02798376941698385,
        -0.18703481171888114, 0.030841381835986965, 0.032883011666982945, -0.010597401784997278,
    ],
    "db6": [
        0.11154074335008017, 0.4946238903983854, 0.7511339080215775, 0.3152503517092432,
        -0.22626469396516913, -0.12976686756709563, 0.09750160558707936, 0.02752286553001629,
        -0.031582039318031156, 0.0005538422009938016, 0.004777257510945510, -0.001077301085308479,
    ],
}
_WAVELET_ALIASES = {"none": "none", "haar": "haar", "db1": "haar", "db2": "db2", "db4": "db4", "db6": "db6"}


def _dwt_matrix(length, wavelet):
    """Single-level periodic DWT as an orthogonal [length, length] matrix.

    Rows [0, length/2) produce the approximation, rows [length/2, length) the detail.
    Forward: coeffs = x @ W.T; inverse (perfect reconstruction): x = coeffs @ W.
    """
    h = _WAVELET_FILTERS[wavelet]
    n_taps = len(h)
    g = [((-1) ** n) * h[n_taps - 1 - n] for n in range(n_taps)]
    half = length // 2
    W = torch.zeros(length, length)
    for k in range(half):
        for n in range(n_taps):
            W[k, (2 * k + n) % length] += h[n]
            W[half + k, (2 * k + n) % length] += g[n]
    return W


class VesaLinear(nn.Module):
    def __init__(self, in_features, out_features, shared_pa, shared_pb, bias=True, rank=None):
        super(VesaLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.shared_pa = shared_pa
        self.shared_pb = shared_pb
        self.rank = rank or shared_pa.shape[0]

        self.lambda_a = nn.Parameter(nn.init.ones_(torch.empty(self.rank)))
        self.lambda_b = nn.Parameter(nn.init.ones_(torch.empty(self.out_features)))
        if bias:
            self.bias = nn.Parameter(nn.init.zeros_(torch.empty(self.out_features)))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        pa = self.shared_pa[:self.in_features, :self.rank]
        pb = self.shared_pb[:self.rank, :self.out_features]

        Pa = pa * self.lambda_a
        Pb = pb * self.lambda_b
        x = x @ Pa @ Pb
        if self.bias is not None:
            x = x + self.bias
        return x


class VesaLinearChannelwise(nn.Module):
    def __init__(self, in_features, out_features, num_channels, shared_pa, shared_pb, bias=True, rank=None):
        super(VesaLinearChannelwise, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_channels = num_channels
        self.shared_pa = shared_pa
        self.shared_pb = shared_pb
        self.rank = rank or shared_pa.shape[1]

        self.lambda_a = nn.Parameter(nn.init.ones_(torch.empty(num_channels, self.rank)))
        self.lambda_b = nn.Parameter(nn.init.ones_(torch.empty(num_channels, out_features)))
        if bias:
            self.bias = nn.Parameter(nn.init.zeros_(torch.empty(num_channels, out_features)))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        pa = self.shared_pa[:self.in_features, :self.rank] 
        pb = self.shared_pb[:self.rank, :self.out_features]

        xp = x @ pa                            
        xp = xp * self.lambda_a.unsqueeze(0)    

        out = xp @ pb                           
        out = out * self.lambda_b.unsqueeze(0)

        if self.bias is not None:
            out = out + self.bias.unsqueeze(0)
        return out


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.d_seq_len
        self.pred_len = configs.d_pred_len
        self.in_features = configs.d_in_features
        self.out_features = configs.d_out_features

        self.m_individual = getattr(configs, "m_individual", True)
        self.seq_dim = getattr(configs, "m_seq_dim", 128)
        self.use_norm = getattr(configs, "m_use_norm", True)
        self.vesa_rank = getattr(configs, "m_vesa_rank", 16)

        wavelet = str(getattr(configs, "m_wavelet_type", "haar")).lower()
        if wavelet not in _WAVELET_ALIASES:
            raise ValueError(
                f"Unknown m_wavelet_type '{wavelet}'. Choose from: none, haar, db2, db4, db6."
            )
        self.wavelet_type = _WAVELET_ALIASES[wavelet]

        if self.wavelet_type == "none":
            # No decomposition: a single band covering the full sequence.
            self.n_bands = 1
            self.seq_len_half = self.seq_len
            self.pred_len_half = self.pred_len
        else:
            if self.seq_len % 2 or self.pred_len % 2:
                raise ValueError("Wavelet decomposition requires even d_seq_len and d_pred_len.")
            self.n_bands = 2
            self.seq_len_half = self.seq_len // 2
            self.pred_len_half = self.pred_len // 2
            self.register_buffer("dwt_in", _dwt_matrix(self.seq_len, self.wavelet_type), persistent=False)
            self.register_buffer("dwt_out", _dwt_matrix(self.pred_len, self.wavelet_type), persistent=False)

        Pa = torch.empty(max(self.seq_len_half, self.seq_dim), self.vesa_rank)
        Pb = torch.empty(self.vesa_rank, max(self.seq_dim, self.pred_len_half))

        nn.init.xavier_uniform_(Pa)
        nn.init.xavier_uniform_(Pb)

        self.shared_pa = nn.Parameter(Pa)
        self.shared_pb = nn.Parameter(Pb)

        if self.use_norm:
            self.revin = RevIN(configs.d_in_features, eps=1e-5, affine=True)
        else:
            self.revin = None

        if self.m_individual:
            self.layer1 = VesaLinearChannelwise(
                in_features=self.seq_len_half,
                out_features=self.seq_dim,
                num_channels=self.n_bands * self.in_features,
                shared_pa=self.shared_pa,
                shared_pb=self.shared_pb,
                rank=self.vesa_rank,
            )
            self.act = nn.GELU()
            self.layer2 = VesaLinearChannelwise(
                in_features=self.seq_dim,
                out_features=self.pred_len_half,
                num_channels=self.n_bands * self.in_features,
                shared_pa=self.shared_pa,
                shared_pb=self.shared_pb,
                rank=self.vesa_rank,
            )
        else:
            self.layer1 = VesaLinear(
                in_features=self.seq_len_half,
                out_features=self.seq_dim,
                shared_pa=self.shared_pa,
                shared_pb=self.shared_pb,
                rank=self.vesa_rank
            )
            self.act = nn.GELU()
            self.layer2 = VesaLinear(
                in_features=self.seq_dim,
                out_features=self.pred_len_half,
                shared_pa=self.shared_pa,
                shared_pb=self.shared_pb,
                rank=self.vesa_rank
            )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        b, l, c = x_enc.shape
        if self.use_norm:
            x = self.revin(x_enc, mode="norm")
        else:
            x = x_enc

        x = x.permute(0, 2, 1)

        if self.wavelet_type == "none":
            stacked = x
        else:
            coeffs = x @ self.dwt_in.T
            stacked = torch.cat([coeffs[..., :self.seq_len_half], coeffs[..., self.seq_len_half:]], dim=1)

        # Stashed as a side effect (not part of the return value) so KD orchestration
        # code can read it for feature-level distillation without changing this
        # forward() signature. Shape: [B, in_features, seq_len].
        self._kd_feature = x

        stacked_out = self.layer2(self.act(self.layer1(stacked)))

        if self.wavelet_type == "none":
            out = stacked_out
        else:
            approx_out, detail_out = stacked_out.split(c, dim=1)
            out = torch.cat([approx_out, detail_out], dim=-1) @ self.dwt_out

        out = out.permute(0, 2, 1)
        if self.use_norm:
            out = self.revin(out, mode="denorm")

        return out