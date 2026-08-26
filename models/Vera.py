import math
import torch
import torch.nn as nn
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


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.d_seq_len
        self.pred_len = configs.d_pred_len
        self.in_features = configs.d_in_features
        self.out_features = configs.d_out_features
        

        self.seq_dim = configs.m_seq_dim
        self.use_norm = configs.m_use_norm

        self.vera_rank = getattr(configs, "m_vera_rank", 16)

        # [1,1]/[1,-1] (Haar-style) approx/detail coefficients each cover
        # one pair of input/target steps, so both streams run at half length.
        self.seq_len_half = self.seq_len // 2
        self.pred_len_half = self.pred_len // 2

        Pa = torch.empty(max(self.seq_len_half, self.seq_dim), self.vera_rank)
        Pb = torch.empty(self.vera_rank, max(self.seq_dim, self.pred_len_half))

        nn.init.xavier_uniform_(Pa)
        nn.init.xavier_uniform_(Pb)

        self.shared_pa = nn.Parameter(Pa)
        self.shared_pb = nn.Parameter(Pb)


        if self.use_norm:
            self.revin = RevIN(configs.d_in_features, eps=1e-5, affine=True)
        else:
            self.revin = None

        self.start_layer = nn.Sequential(
            VeraLinear(
                in_features=self.seq_len_half,
                out_features=self.seq_dim,
                shared_pa=self.shared_pa,
                shared_pb=self.shared_pb,
                rank=self.vera_rank
            ),
        )

        self.mid_layer = nn.Sequential(
            nn.Conv1d(
                2 * self.in_features,
                2 * self.in_features,
                kernel_size=configs.m_dwconv_kernel_size,
                stride=1,
                padding="same",
                groups=2 * self.in_features
            ),
            VeraLinear(
                in_features=self.seq_dim,
                out_features=self.seq_dim,
                shared_pa=self.shared_pa,
                shared_pb=self.shared_pb,
                rank=self.vera_rank
            ),
            nn.GELU(),
        )

        self.final_layer = nn.Sequential(
            nn.LayerNorm(self.seq_dim),
            VeraLinear(
                in_features=self.seq_dim,
                out_features=self.pred_len_half,
                shared_pa=self.shared_pa,
                shared_pb=self.shared_pb,
                rank=self.vera_rank
            ),
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        b, l, c = x_enc.shape
        if self.use_norm:
            x = self.revin(x_enc, mode="norm")
        else:
            x = x_enc

        x = x.permute(0, 2, 1)
        # [1,1]/[1,-1] filters over consecutive pairs give the Haar-style
        # approx (sum) and detail (difference) coefficients, halving seq_len.
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        approx, detail = (x_even + x_odd) /math.sqrt(2), (x_even - x_odd) /math.sqrt(2)
        x1 = torch.concat([approx, detail], dim=1)
        x1 = self.start_layer(x1)
        x1 = self.mid_layer(x1)
        # Stashed as a side effect (not part of the return value) so KD orchestration
        # code can read it for feature-level distillation without changing this
        # forward() signature. Shape: [B, in_features, seq_len].
        self._kd_feature = x
        x1 = self.final_layer(x1)

        approx_out, detail_out = torch.split(x1, self.in_features, dim=1)
        out_even, out_odd = (approx_out + detail_out) / math.sqrt(2), (approx_out - detail_out) / math.sqrt(2)
        out = torch.zeros(b, c, self.pred_len, device=approx_out.device, dtype=approx_out.dtype)
        out[...,0::2] = out_even
        out[...,1::2] = out_odd


        out = out.permute(0, 2, 1)
        if self.use_norm:
            out = self.revin(out, mode="denorm")

        return out
