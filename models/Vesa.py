import math
import torch
import torch.nn as nn
from layers.RevIN import RevIN


# Shapes: B = batch, L = seq_len, H = pred_len, C = channels,
# D = hidden dim (m_seq_dim), r = rank (m_vesa_rank).


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


class Model(nn.Module):
    # VeSA-FFN with one-level Haar wavelet decomposition.
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

        self.seq_len_half = self.seq_len // 2
        self.pred_len_half = self.pred_len // 2

        # Shared low-rank matrices, reused by both FFN layers.
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

        # FFN: L/2 -> D -> H/2.
        if self.m_individual:
            self.layer1 = VesaLinearChannelwise(
                in_features=self.seq_len_half,
                out_features=self.seq_dim,
                num_channels=2 * self.in_features,
                shared_pa=self.shared_pa,
                shared_pb=self.shared_pb,
                rank=self.vesa_rank,
            )
            self.act = nn.GELU()
            self.layer2 = VesaLinearChannelwise(
                in_features=self.seq_dim,
                out_features=self.pred_len_half,
                num_channels=2 * self.in_features,
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
        # x_enc: [B, L, C]
        b, l, c = x_enc.shape
        if self.use_norm:
            x = self.revin(x_enc, mode="norm")                 # [B, L, C]
        else:
            x = x_enc

        x = x.permute(0, 2, 1)                                 # [B, C, L]

        # Haar DWT.
        x_even, x_odd = x[..., 0::2], x[..., 1::2]             # [B, C, L/2] each
        approx, detail = (x_even + x_odd) / math.sqrt(2), (x_even - x_odd) / math.sqrt(2)  # [B, C, L/2] each
        stacked = torch.cat([approx, detail], dim=1)           # [B, 2C, L/2]

        # Input kept for feature-level distillation.
        self._kd_feature = x                                   # [B, C, L]


        stacked_out = self.layer2(self.act(self.layer1(stacked)))  # [B, 2C, L/2] -> [B, 2C, D] -> [B, 2C, H/2]
        approx_out, detail_out = stacked_out.split(c, dim=1)   # [B, C, H/2] each

        # Inverse Haar DWT.
        out_even, out_odd = (approx_out + detail_out) / math.sqrt(2), (approx_out - detail_out) / math.sqrt(2)  # [B, C, H/2] each
        out = torch.zeros(b, c, self.pred_len, device=approx_out.device, dtype=approx_out.dtype)  # [B, C, H]
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd

        out = out.permute(0, 2, 1)                             # [B, H, C]
        if self.use_norm:
            out = self.revin(out, mode="denorm")               # [B, H, C]

        return out
