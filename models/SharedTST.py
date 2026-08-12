import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN


class SharedLinear(nn.Module):
    """
    """
    def __init__(self, in_features, out_features, bias=True):
        super(SharedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        rank = 10
        self.P = nn.Parameter(nn.init.xavier_uniform_(torch.empty(in_features, rank)))
        self.d = nn.Parameter(nn.init.xavier_uniform_(torch.empty(rank, out_features)))
        if bias:
            self.bias = nn.Parameter(nn.init.zeros_(torch.empty(self.out_features)))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        W = F.gelu(self.P @ self.d)
        x = x @ W
        if self.bias is not None:
            x = x + self.bias
        return x


class WaveletDecomp:
    """
    Splits a sequence into low-frequency (approximation) and high-frequency
    (detail) Haar wavelet components, then reconstructs the sequence from them.
    Ported from HaDAM.py.
    """

    def decompose(self, x):
        """
        input x: [b, c, l]
        output approx: [b, c, l / 2]
        output detail: [b, c, l / 2]
        """
        assert x.shape[-1] % 2 == 0, "Input length must be even for Haar decomposition."

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # Low-pass and high-pass Haar filters.
        approx = (x_even + x_odd) / math.sqrt(2)
        detail = (x_even - x_odd) / math.sqrt(2)

        return approx, detail

    def reconstruct(self, approx, detail):
        """
        input approx: [b, c, l / 2]
        input detail: [b, c, l / 2]
        output: [b, c, l]
        """
        assert approx.shape == detail.shape, "Approximation and detail must have the same shape for reconstruction."

        x_even = (approx + detail) / math.sqrt(2)
        x_odd = (approx - detail) / math.sqrt(2)

        # Interleave even and odd positions back into the original time order.
        B, C, L_half = approx.shape
        L = L_half * 2

        x = torch.zeros(B, C, L, device=approx.device, dtype=approx.dtype)
        x[..., 0::2] = x_even
        x[..., 1::2] = x_odd

        return x


class ConvolutionalAttention(nn.Module):
    """
    Builds Q, K, and V with depthwise 1D convolutions, then applies scaled
    dot-product attention across channels. Ported from HaDAM.py: input/output are
    [b, c, d] -- Q/K/V keep that shape, so SDPA attends over the channel axis (c)
    using each channel's length-d vector as its token embedding, with the conv
    kernel giving Q/K local temporal context instead of a single opaque vector.
    """
    def __init__(self, channel_dim, kernel_size, dropout):
        super(ConvolutionalAttention, self).__init__()

        self.DWConv_qk = nn.Conv1d(
            in_channels=channel_dim,
            out_channels=2 * channel_dim,
            kernel_size=kernel_size,
            stride=2,
            groups=channel_dim,
            bias=True
        )

        self.DWConv_v = nn.Conv1d(
            in_channels=channel_dim,
            out_channels=channel_dim,
            kernel_size=1,
            stride=1,
            padding="same",
            groups=channel_dim,
            bias=True
        )

        self.dropout = float(dropout)

    def forward(self, x):
        # x: (B, C, D)

        x_conv = self.DWConv_qk(x)  # (B, C, D)
        Q, K = x_conv[:, 0::2, :], x_conv[:, 1::2, :]  # each: (B, C, D)

        V = self.DWConv_v(x)  # (B, C, D)

        out = F.scaled_dot_product_attention(
            Q, K, V, dropout_p=self.dropout if self.training else 0.0
        )  # (B, C, D)

        return out


class ConvAttentionBlock(nn.Module):
    """
    Pre-norm residual block: a gated convolutional-attention sub-layer (HaDAM-style,
    attends across channels) followed by a VeraLinear FFN sub-layer.
    """
    def __init__(self, channel_dim, seq_dim,
                 attn_kernel_size, dropout=0.0):
        super(ConvAttentionBlock, self).__init__()

        self.seq_dim = seq_dim
        self.alpha = nn.Parameter(torch.zeros(1, channel_dim, 1))
        self.attention = ConvolutionalAttention(
            channel_dim=channel_dim,
            kernel_size=attn_kernel_size,
            dropout=dropout,
        )

        self.norm1 = nn.LayerNorm(self.seq_dim)
        self.norm2 = nn.LayerNorm(self.seq_dim)

        self.ffn = SharedLinear(
            in_features=self.seq_dim,
            out_features=self.seq_dim,
            bias=True
        )

        self.activation = nn.GELU()
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B', C, D) -- D == seq_dim exactly (no flattened patch axis here).
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn_dropout(F.gelu(self.ffn(self.norm2(x))))
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.d_seq_len
        self.pred_len = configs.d_pred_len
        self.in_features = configs.d_in_features
        self.out_features = configs.d_out_features
        self.use_norm = getattr(configs, "m_use_norm", True)
        self.dropout = getattr(configs, "m_dropout", 0.0)
        self.num_blocks = getattr(configs, "m_num_blocks", 1)
        self.attn_kernel_size = getattr(configs, "m_attn_head_kernel_size", 32)
        self.seq_dim = getattr(configs, "m_seq_dim", 512)

        assert self.seq_len % 2 == 0, "d_seq_len must be even for Haar decomposition."
        assert self.pred_len % 2 == 0, "d_pred_len must be even for Haar decomposition."
        self.decomp = WaveletDecomp()


        # max_dim = max(self.seq_dim, self.seq_len //2, self.pred_len //2)
        # self.shared_U = nn.Parameter(nn.init.xavier_uniform_(torch.empty(max_dim, max_dim)))

        

        if self.use_norm:
            self.revin = RevIN(
                configs.d_in_features,
                eps=1e-5,
                affine=getattr(configs, "m_revin_affine", True),
                subtract_last=getattr(configs, "m_revin_subtract_last", False),
            )


        self.embed = SharedLinear(
            in_features=self.seq_len//2,
            out_features=self.seq_dim,
            bias=True,
        )

        self.blocks = nn.ModuleList([
            ConvAttentionBlock(
                channel_dim=2 * self.in_features,
                seq_dim=self.seq_dim,
                attn_kernel_size=self.attn_kernel_size,
                dropout=self.dropout,
            )
            for _ in range(self.num_blocks)
        ])

        self.pred = SharedLinear(
            in_features=self.seq_dim,
            out_features=self.pred_len//2,
            bias=True,
        )

    def forward(self, x):
        # x: (B, L, C)
        b, l, c = x.shape

        if self.use_norm:
            x = self.revin(x, mode="norm")  # (B, L, C)

        x = x.permute(0, 2, 1)  # (B, C, L)
        # x = x + self.pos_embed  # broadcast (1, 1, L) over batch and channels

        approx, detail = self.decomp.decompose(x)  # each: (B, C, L/2)
        x = torch.cat([approx, detail], dim=1)  # (B, 2C, L/2)
       
        x = self.embed(x)  # (B, 2C, D)

        for block in self.blocks:
            x = block(x)  # (B, 2C, D)

        x = self.pred(x)  # (B, 2C, H/2)

        approx_out, detail_out = torch.split(x, c, dim=1)  # each: (B, C, H/2)
        out = self.decomp.reconstruct(approx_out, detail_out)  # (B, C, H)

        out = out.permute(0, 2, 1)  # (B, H, C)
        if self.use_norm:
            out = self.revin(out, mode="denorm")

        return out
