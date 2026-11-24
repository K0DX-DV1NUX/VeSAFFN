import torch
import torch.nn as nn
from layers.RevIN import RevIN
from layers.Decomposition import NoDecomp, TrendResidualDecomp, WaveletDecomp


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
