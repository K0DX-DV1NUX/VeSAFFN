import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN


class VeraLinear(nn.Module):
    """
    VeRA linear layer using delta_W = diag(lambda_b) @ P_B @ diag(lambda_d) @ P_A.

    P_B: [max_out_features, rank], shared across all VeraLinear modules.
    P_A: [rank, max_in_features], shared across all VeraLinear modules.
    lambda_b: [out_features], local to this module.
    lambda_d: [rank], local to this module.
    """
    def __init__(self, in_features, out_features, shared_pa, shared_pb, bias=True, rank=None):
        super(VeraLinear, self).__init__()
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


class HaarWaveletDecomp:
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
        self.channel_dim = channel_dim

        self.DWConv_qk = nn.Conv1d(
            in_channels=channel_dim,
            out_channels=2 * channel_dim,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
            groups=channel_dim,
            bias=False
        )

        self.DWConv_v = nn.Conv1d(
            in_channels=channel_dim,
            out_channels=channel_dim,
            kernel_size=1,
            stride=1,
            padding="same",
            groups=channel_dim,
            bias=False
        )

        self.dropout = float(dropout)

    def forward(self, x):
        # x: (B, C, D)
        x_conv = self.DWConv_qk(x)  # (B, 2C, D)
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
    def __init__(self, channel_dim, seq_dim, shared_pa, shared_pb, vera_rank,
                 attn_kernel_size, dropout=0.0):
        super(ConvAttentionBlock, self).__init__()

        self.alpha = nn.Parameter(torch.zeros(1, channel_dim, 1))
        self.attention = ConvolutionalAttention(
            channel_dim=channel_dim,
            kernel_size=attn_kernel_size,
            dropout=dropout,
        )

        self.norm1 = nn.LayerNorm(seq_dim)
        self.norm2 = nn.LayerNorm(seq_dim)

        self.ffn = nn.Linear(seq_dim, seq_dim)
        self.activation = nn.GELU()
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, D)

        # --- Gated conv-attention sub-layer, across channels ---
        gate = torch.sigmoid(self.alpha)  # (1, C, 1)
        x = x + gate * self.attention(self.norm1(x))

        # --- Pre-norm FFN sub-layer (single VeraLinear) ---
        residual = x
        h = self.activation(self.ffn(self.norm2(x)))
        h = self.ffn_dropout(h)
        x = residual + h

        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.d_seq_len
        self.pred_len = configs.d_pred_len
        self.in_features = configs.d_in_features
        self.out_features = configs.d_out_features
        self.use_norm = configs.m_use_norm
        self.seq_dim = configs.m_seq_dim
        self.dropout = getattr(configs, "m_dropout", 0.0)
        self.num_blocks = configs.m_num_blocks
        self.attn_kernel_size = configs.m_attn_head_kernel_size

        self.vera_rank = getattr(configs, "m_vera_rank", 16)

        assert self.seq_len % 2 == 0, "d_seq_len must be even for Haar decomposition."
        assert self.pred_len % 2 == 0, "d_pred_len must be even for Haar decomposition."
        self.decomp = HaarWaveletDecomp()

        # Shared low-rank basis. embed/final are the only VeraLinear users now (ffn
        # is a plain nn.Linear). in_features seen: {seq_len/2, seq_dim}; out_features
        # seen: {seq_dim, pred_len/2} -- embed and final operate on the Haar-halved
        # lengths, since approx/detail are each seq_len/2 (resp. pred_len/2) long.
        Pa = torch.empty(max(self.seq_len // 2, self.seq_dim), self.vera_rank)
        Pb = torch.empty(self.vera_rank, max(self.seq_dim, self.pred_len // 2))

        nn.init.xavier_uniform_(Pa)
        nn.init.xavier_uniform_(Pb)

        self.shared_pa = nn.Parameter(Pa)
        self.shared_pb = nn.Parameter(Pb)

        if self.use_norm:
            self.revin = RevIN(
                configs.d_in_features,
                eps=1e-5,
                affine=getattr(configs, "m_revin_affine", True),
                subtract_last=getattr(configs, "m_revin_subtract_last", False),
            )

        # Positional embedding indexed over raw sequence length L, added before the
        # L -> seq_dim projection collapses each channel's series into one vector.
        # self.pos_embed = nn.Parameter(torch.empty(1, 1, self.seq_len))
        # nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

        self.embed = VeraLinear(
            self.seq_len // 2, self.seq_dim,
            self.shared_pa, self.shared_pb, rank=self.vera_rank
        )
        self.embed_activation = nn.GELU()
        self.embed_dropout = nn.Dropout(self.dropout)

        # Approx and detail components are concatenated along the channel axis
        # before this point, so every block sees 2 * in_features channels.
        self.blocks = nn.ModuleList([
            ConvAttentionBlock(
                channel_dim=2 * self.in_features,
                seq_dim=self.seq_dim,
                shared_pa=self.shared_pa,
                shared_pb=self.shared_pb,
                vera_rank=self.vera_rank,
                attn_kernel_size=self.attn_kernel_size,
                dropout=self.dropout,
            )
            for _ in range(self.num_blocks)
        ])

        self.final = VeraLinear(
            self.seq_dim, self.pred_len // 2,
            self.shared_pa, self.shared_pb, rank=self.vera_rank
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

        x = self.embed_activation(self.embed(x))  # (B, 2C, D)
        x = self.embed_dropout(x)

        for block in self.blocks:
            x = block(x)  # (B, 2C, D)

        x = self.final(x)  # (B, 2C, H/2)

        approx_out, detail_out = torch.split(x, c, dim=1)  # each: (B, C, H/2)
        out = self.decomp.reconstruct(approx_out, detail_out)  # (B, C, H)

        out = out.permute(0, 2, 1)  # (B, H, C)
        if self.use_norm:
            out = self.revin(out, mode="denorm")

        return out
