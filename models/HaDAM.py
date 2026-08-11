import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.RevIN import RevIN

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class HaarWaveletDecomp:
    """
    Splits a sequence into low-frequency (approximation) and high-frequency
    (detail) Haar wavelet components, then reconstructs the sequence from them.
    """

    def __init__(self):
       pass

    def decompose(self, x):
        """
        input x: [b, c, l]
        output approx: [b, c, l / 2]
        output detail: [b, c, l / 2]
        """
        assert x.shape[-1] % 2 == 0, "Input length must be even for Haar decomposition."

        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]

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
        x_odd  = (approx - detail) / math.sqrt(2)

        # Interleave even and odd positions back into the original time order.
        B, C, L_half = approx.shape
        L = L_half * 2

        x = torch.zeros(B, C, L, device=approx.device, dtype=approx.dtype)
        x[..., 0::2] = x_even
        x[..., 1::2] = x_odd

        return x


class DualAxisEmbedding(nn.Module):
    """
    Converts each non-overlapping time patch into a learned patch embedding while
    keeping the channel dimension independent.
    """

    def __init__(self, patch_size, patch_dim, in_N, out_N):
        super(DualAxisEmbedding, self).__init__()

        self.patch_size = patch_size
        self.patch_dim = patch_dim

        self.da_embed = DualAxisLinear(
            in_patch_size=patch_size,
            out_patch_size=patch_dim,
            in_N=in_N,
            out_N=out_N
        )


    def forward(self, x):
        '''
        input x: [b, c, l]
        output: [b, c, n * d_p]
        '''
        out = self.da_embed(x)
        return out  # [b, c, n * d_p]


class ConvolutionalAttention(nn.Module):
    """
    Builds Q, K, and V with depthwise 1D convolutions, then applies scaled
    dot-product attention over the embedded sequence for each channel stream.
    """

    def __init__(self, channel_dim, kernel_size,
                dropout):
        super(ConvolutionalAttention, self).__init__()

        self.channel_dim = channel_dim

        # Depthwise convolutions preserve per-channel structure while producing Q, K, and V.
        self.DWConv_qk = nn.Conv1d(
            in_channels=channel_dim,
            out_channels=2 * channel_dim,
            kernel_size=kernel_size,
            stride=2,
            #padding="same",
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
        '''
        input x: [b, c, l]
        output: [b, c, l]
        '''
        x_conv = self.DWConv_qk(x) # [b, 2 * c, l]
        Q, K = x_conv[:, 0::2, :], x_conv[:, 1::2, :] # each: [b, c, l]

        V = self.DWConv_v(x) # [b, c, l]

        out = F.scaled_dot_product_attention(
            Q, K, V, dropout_p=self.dropout if self.training else 0.0
        ) # [b, c, l]

        return out # [b, c, l]

    

class ConvAttentionMixer(nn.Module):
    """
    Residual block that refines patch embeddings with convolutional attention
    followed by a patch-wise MLP projection.
    """

    def __init__(self, channel_dim,N,patch_dim, attn_kernel_size, dropout=0.1):
        super(ConvAttentionMixer, self).__init__()

        self.alpha = nn.Parameter(torch.zeros(1, channel_dim, 1))

        self.attention = ConvolutionalAttention(
                channel_dim=channel_dim,
                kernel_size=attn_kernel_size,
                dropout=dropout
            )
        
        self.pre_norm1 = nn.LayerNorm(N * patch_dim)
        self.pre_norm2 = nn.LayerNorm(N * patch_dim)
        
        self.mlp = DualAxisLinear(
            in_patch_size=patch_dim,
            out_patch_size=patch_dim,
            in_N=N,
            out_N=N)
         

    def forward(self, x):
        '''
        input x: [b, c, n * d_p]
        output: [b, c, n * d_p]
        '''
        # Attention sub-layer with a residual connection.
        gate = torch.sigmoid(self.alpha)  # [1, c, 1]
        out = x + gate * self.attention(self.pre_norm1(x))

        # Patch projection sub-layer with a residual connection.
        out = out + self.mlp(self.pre_norm2(out))

        return out


class Model(nn.Module):
    """
    Forecasting model that normalizes the input, decomposes it into Haar
    approximation/detail components, embeds those components as patches, refines
    them with ConvolutionAttentionMixer blocks, predicts future wavelet components, and reconstructs
    the final forecast.
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.d_seq_len
        self.pred_len = configs.d_pred_len
        self.in_features = configs.d_in_features
        self.out_features = configs.d_out_features

        self.in_patch_size = configs.m_in_patch_size
        self.out_patch_size = configs.m_out_patch_size
        self.patch_dim = configs.m_patch_dim

        self.decomp_type = getattr(configs, "m_decomp_type", None)

        if self.decomp_type == "haar":
            assert self.seq_len % self.in_patch_size == 0, "Sequence length should be divisible by patch size to create non-overlapping patches."
            assert self.pred_len % self.out_patch_size == 0, "Prediction length should be divisible by patch size for the patch-based prediction."
            self.N_L = self.seq_len // (2 * self.in_patch_size)
            self.N_H = self.pred_len // (2 * self.out_patch_size)

             # Haar wavelet transform separates trend-like and detail-like signals.
            self.decomp = HaarWaveletDecomp()

            self.output_pred2 = DualAxisLinear(
                in_patch_size=self.patch_dim,
                out_patch_size=self.out_patch_size,
                in_N=self.N_L,
                out_N=self.N_H
            )

        elif self.decomp_type == "trend-residual":
            self.decomp = series_decomp(kernel_size=25)
            self.N_L = self.seq_len // self.in_patch_size
            self.N_H = self.pred_len // self.out_patch_size

            self.output_pred2 = DualAxisLinear(
                in_patch_size=self.patch_dim,
                out_patch_size=self.out_patch_size,
                in_N=self.N_L,
                out_N=self.N_H
            )
        
        else:
            self.N_L = self.seq_len // self.in_patch_size
            self.N_H = self.pred_len // self.out_patch_size



        # Reversible instance normalization for non-stationary time-series inputs.
        self.revin = RevIN(
            self.in_features,
            affine=configs.m_revin_affine,
            subtract_last=configs.m_revin_subtract_last
        )

        self.da_embed = DualAxisEmbedding(
            patch_size=self.in_patch_size,
            patch_dim=self.patch_dim,
            in_N=self.N_L,
            out_N=self.N_L
        )

        self.conv_attn_mixers = nn.Sequential(*[
            ConvAttentionMixer(
                channel_dim=2* self.in_features if self.decomp_type is not None else self.in_features,
                N = self.N_L,
                patch_dim=self.patch_dim,
                attn_kernel_size=configs.m_attn_head_kernel_size,
                dropout=configs.m_attn_head_dropout,
            )
            for _ in range(configs.m_n_layers)
        ])
    
        self.output_pred1 = DualAxisLinear(
            in_patch_size=self.patch_dim,
            out_patch_size=self.out_patch_size,
            in_N=self.N_L,
            out_N=self.N_H
        )

    
    def forward(self, x):
        '''
        input x: [b, l, c]
        output: [b, pred_len, c]
        '''
        b, l, c = x.shape

        # Normalize before modeling, then move channels before length.
        x = self.revin(x, mode='norm') # [b, l, c]
        x = x.permute(0, 2, 1) # [b, c, l]

        if self.decomp_type == "haar":
            # Decompose into low-frequency approximation and high-frequency detail.
            val1, val2 = self.decomp.decompose(x)  # each: [b, c, l / 2]
            x = torch.cat([val1, val2], dim=1)  # [b, 2c, l / 2]
        
        elif self.decomp_type == "trend-residual":
            val1, val2 = self.decomp(x)  # each: [b, c, l]
            x = torch.cat([val1, val2], dim=1)  # [b, 2c, l]

        

        # Embed wavelet components into patch tokens and refine them with Convolution Attention Mixer blocks.
        embed= self.da_embed(x)  # [b, 2c, n_l * d_p]
        embed = self.conv_attn_mixers(embed)  # [b, 2c, n_l * d_p]

        if self.decomp_type is not None:
            out1, out2 = torch.split(embed, c, dim=1)
            approx_out = self.output_pred1(out1)  # [b, c, n_h * patch_size]
            detail_out = self.output_pred2(out2)  # [b, c, n_h * patch_size]
        else:
            out = self.output_pred1(embed)  # [b, c, n_h * patch_size]

        if self.decomp_type == "haar":
            out = self.decomp.reconstruct(approx_out, detail_out)  # [b, c, pred_len]
        elif self.decomp_type == "trend-residual":
            out = approx_out + detail_out  # [b, c, pred_len]

        # Return to [batch, length, channel] and undo RevIN normalization.
        out = self.revin(out.permute(0, 2, 1), mode='denorm')

        return out
    
    # def _ensure_aux_loss_params(self, channels=None, device=None, dtype=None):
    #     # Register auxiliary-loss weights only when auxiliary loss is used.
    #     if hasattr(self, "w_loss"):
    #         return

    #     channels = channels if channels is not None else self.out_features
    #     ref_param = next(self.parameters())
    #     device = device if device is not None else ref_param.device
    #     dtype = dtype if dtype is not None else ref_param.dtype

    #     self.register_parameter(
    #         "w_loss",
    #         torch.nn.Parameter(torch.zeros(channels, device=device, dtype=dtype))
    #     )

    # def aux_loss(self, pred, true):
    #     # Compute per-channel prediction error before reducing across variables.
    #     criterion = torch.nn.MSELoss(reduction="none")
    #     error = criterion(pred, true)  # [b, h, c]
    #     loss_c = error.mean(dim=(0, 1))  # [c]

    #     # Learn positive channel weights while keeping their total scale stable.
    #     w = torch.softmax(self.w_loss, dim=0) * len(self.w_loss)  # [c]
    #     w = w / w.sum() * len(w)  # [c]
        
    #     loss = (w * loss_c).sum()

    #     # Track a smoothed baseline loss per channel for relative training speed.
    #     if not hasattr(self, "L0"):
    #         self.L0 = loss_c.detach()
    #     else:
    #         self.L0 = 0.99 * self.L0 + 0.01 * loss_c.detach()

    #     # Build target weights that emphasize channels learning more slowly.
    #     r = loss_c / (self.L0 + 1e-8)
    #     r = r / r.mean()  # [c]

    #     target = r ** 0.5
    #     target = target / target.sum() * len(target)  # [c]
    #     balance_loss = F.l1_loss(w, target.detach())

    #     # Combine weighted forecasting loss with the channel-balancing penalty.
    #     total_loss = loss + 0.2 * balance_loss

    #     return total_loss


class DualAxisLinear(nn.Module):
    """
    Projects flattened patch sequences in two steps: first across values inside
    each patch, then across the number of patches.
    """

    def __init__(self, in_patch_size, out_patch_size, in_N, out_N):
        super(DualAxisLinear, self).__init__()
        self.in_patch_size = in_patch_size
        self.out_patch_size = out_patch_size
        self.in_N = in_N
        self.out_N = out_N
        self.patch_proj = nn.Linear(in_patch_size, out_patch_size)
        self.len_proj = nn.Linear(in_N, out_N)

    def forward(self, x):
        '''
        input x: [b, c, in_n * in_patch_size]
        output: [b, c, out_n * out_patch_size]
        '''
        batch, channels, length = x.shape
        
        x = x.contiguous().view(batch, channels, self.in_N, self.in_patch_size) # [b, c, in_n, in_patch_size]
        
        x = F.gelu(self.patch_proj(x))  # [b, c, in_n, out_patch_size]
        
        x = self.len_proj(x.permute(0, 1, 3, 2))  # [b, c, out_patch_size, out_n]

        out = x.contiguous().view(batch, channels, self.out_N * self.out_patch_size)  # [b, c, out_n * out_patch_size]
        
        return out
