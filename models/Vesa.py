import math
import torch
import torch.nn as nn
from layers.RevIN import RevIN


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

        self.m_individual = getattr(configs, "m_individual", False)
        self.seq_dim = getattr(configs, "m_seq_dim", 128)
        self.use_norm = getattr(configs, "m_use_norm", True)
        self.vesa_rank = getattr(configs, "m_vesa_rank", 16)

        self.seq_len_half = self.seq_len // 2
        self.pred_len_half = self.pred_len // 2

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
        b, l, c = x_enc.shape
        if self.use_norm:
            x = self.revin(x_enc, mode="norm")
        else:
            x = x_enc

        x = x.permute(0, 2, 1)

        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        approx, detail = (x_even + x_odd) / math.sqrt(2), (x_even - x_odd) / math.sqrt(2)
        stacked = torch.cat([approx, detail], dim=1)

        # Stashed as a side effect (not part of the return value) so KD orchestration
        # code can read it for feature-level distillation without changing this
        # forward() signature. Shape: [B, in_features, seq_len].
        self._kd_feature = x


        stacked_out = self.layer2(self.act(self.layer1(stacked)))
        approx_out, detail_out = stacked_out.split(c, dim=1)

        out_even, out_odd = (approx_out + detail_out) / math.sqrt(2), (approx_out - detail_out) / math.sqrt(2)
        out = torch.zeros(b, c, self.pred_len, device=approx_out.device, dtype=approx_out.dtype)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd

        out = out.permute(0, 2, 1)
        if self.use_norm:
            out = self.revin(out, mode="denorm")

        return out