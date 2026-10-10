import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer, ProbAttention, DSAttention
from layers.Embed import DataEmbedding
import numpy as np
from layers.RevIN import RevIN


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class Model(nn.Module):
    # Client: Transformer encoder over variates plus a linear branch.
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.d_pred_len
        self.output_attention = configs.m_output_attention
        configs.m_d_model = configs.d_seq_len
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.m_output_attention), configs.m_d_model, configs.m_n_heads),
                    configs.m_d_model,
                    configs.m_d_ff,
                    dropout=configs.m_dropout,
                    activation=configs.m_activation
                ) for l in range(configs.m_e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.m_d_model)
        )
        self.proj = nn.Linear(configs.m_d_model, self.pred_len, bias=True)
        self.Linear = nn.Sequential()
        self.Linear.add_module('Linear',nn.Linear(configs.m_d_seq_len, self.pred_len))
        self.w_dec = torch.nn.Parameter(torch.FloatTensor([configs.w_lin]*configs.d_enc_in),requires_grad=True)
        self.revin_layer = RevIN(configs.d_enc_in)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # x_enc: [B, L, C]
        x_enc = self.revin_layer(x_enc, 'norm')                             # [B, L, C]
        enc_out = x_enc.permute(0, 2, 1)                                    # [B, C, L]
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)     # [B, C, L]
        dec_out = self.proj(enc_out)                                        # [B, C, H]
        dec_out = dec_out.permute(0, 2, 1)                                  # [B, H, C]
        linear_out = self.Linear(x_enc.permute(0,2,1)).permute(0,2,1)       # [B, H, C]
        dec_out = self.revin_layer(dec_out[:, -self.pred_len:, :]+self.w_dec*linear_out, 'denorm')  # [B, H, C]

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out