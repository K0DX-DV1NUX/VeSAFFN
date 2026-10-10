import math
from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import numpy as np
from layers.RevIN import RevIN
from layers.PatchTST_layers import series_decomp

# FrNet: period-based patching with frequency-rotation layers.
# Shapes: B = batch, C = channels, L = seq_len, H = pred_len, N = patches, P = patch length, E = emb, D = E/2+1.


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.d_seq_len
        self.label_len = configs.d_label_len
        self.pred_len = configs.d_pred_len
        self.c_in = configs.d_in_features
        self.decomposition = configs.m_decomposition
        kernel_size = configs.m_kernel_size
        self.decomp_module = series_decomp(kernel_size)
        self.revin_layer = RevIN(self.c_in, affine=configs.m_affine, subtract_last=configs.m_subtract_last)
        self.trend_revin_layer = RevIN(self.c_in, affine=configs.m_affine, subtract_last=configs.m_subtract_last)
        self.period_list = configs.m_period_list
        self.emb = configs.m_emb
        
        self.model = FRNet_backone(
            self.c_in, self.seq_len, self.pred_len, self.emb, configs.m_use_norm,
            configs.m_dropout, configs.m_e_layers,
            configs.m_pred_head_type,
            configs.m_aggregation_type,
            configs.m_channel_attention,
            configs.m_global_freq_pred,
            self.period_list
        )
        if self.decomposition:
            self.model_trend = FRNet_trend_backone(
                self.c_in, self.seq_len, self.pred_len, self.emb, configs.m_use_norm,
                configs.m_dropout, configs.m_e_layers,
                configs.m_pred_head_type,
                configs.m_aggregation_type,
                configs.m_channel_attention,
                configs.m_global_freq_pred,
                configs.m_patch_len, configs.m_stride
            )
        self.linear_projection =  nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.pred_len, self.pred_len)
        )

    def forward(self, x):
        # x: [B, L, C]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)                                # [B, L, C] each
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # [B, C, L] each
            res = self.model(res_init)                                                  # [B, C, H]
            trend = self.model_trend(trend_init)                                        # [B, C, H]
            x = res + trend
            x = x.permute(0,2,1)                                                        # [B, H, C]
        else:
            x = x.permute(0,2,1)                                                        # [B, C, L]
            x = self.model(x)                                                           # [B, C, H]
            x = x.permute(0,2,1)                                                        # [B, H, C]
    
        return x
  
class FRNet_backone(nn.Module):
    # Seasonal branch: one FRBackbone per period in period_list.
    def __init__(self, c_in, seq_len, pred_len, emb, revin, dropout, e_layers, pred_head_type, aggregation_type, channel_attention, global_freq_pred, period_list):
        super().__init__()
        self.period_list = period_list
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = c_in
        self.revin = revin
        self.channel_attention = channel_attention
        self.FRBackboneList = nn.ModuleList() 
        self.global_freq_pred = global_freq_pred
        if self.global_freq_pred:
            self.dominance_freq=int(self.seq_len // self.period_list[0] + 1) * 2 + 10 
            self.length_ratio = (self.pred_len)/self.seq_len 
            self.freq_Linear = nn.Linear(self.dominance_freq, int(self.dominance_freq*self.length_ratio)).to(torch.cfloat)
            self.fit_channelfuse = ChannelAttentionLayer(FullAttention(attention_dropout=0.1, output_attention=False), seq_len, 8)
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=True, subtract_last=False)
        for k in range(len(self.period_list)):
            self.FRBackboneList.append(FRBackbone(self.period_list[k], c_in, seq_len, pred_len, emb, dropout, e_layers, pred_head_type, channel_attention))
        self.aggregation_type = aggregation_type
        if aggregation_type == 'linear': 
            if global_freq_pred:
                self.fuse_linear = nn.Linear(pred_len*(1 + len(self.period_list)), pred_len)
            else:
                self.fuse_linear = nn.Linear(pred_len*len(self.period_list), pred_len)
        
    def forward(self, x):
        # x: [B, C, L]
        if self.revin: 
            x = x.permute(0,2,1)
            x = self.revin_layer(x, 'norm')
            x = x.permute(0,2,1)
        res = []  # per-branch forecasts, each [B, C, H]
        if self.global_freq_pred:
            low_specx = torch.fft.rfft(x, dim=-1)  # [B, C, L/2+1]
            low_specx[:,:,self.dominance_freq:]=0
            low_specx = low_specx[:,:,0:self.dominance_freq]  # [B, C, F]
            low_specxy_ = self.freq_Linear(low_specx)  # [B, C, F*H/L]
            low_specxy = torch.zeros([low_specxy_.size(0), low_specxy_.size(1), int((self.pred_len)/2+1)],dtype=low_specxy_.dtype).to(low_specxy_.device)  # [B, C, H/2+1]

            low_specxy[:,:,0:low_specxy_.size(-1)]=low_specxy_ 
            low_xy=torch.fft.irfft(low_specxy, dim=-1)  # [B, C, H]
            res.append(low_xy)
        
        for k in range(len(self.period_list)):
            length = self.seq_len
            period_i = self.period_list[k]
            if length % period_i != 0:
                new_length = (length // period_i) * period_i
                x_k = x[:, :, -new_length:]
            else:
                new_length = length
                x_k = x
            x_k = x_k.unfold(dimension=-1, size=period_i, step=period_i)  # [B, C, N, P]
            pred_k = self.FRBackboneList[k](x_k)  # [B, C, H]
            res.append(pred_k)
        if self.aggregation_type == 'linear':
            concatenated_res = torch.cat(res, dim=-1)  # [B, C, K*H]
            res = self.fuse_linear(concatenated_res)  # [B, C, H]
        elif self.aggregation_type == 'avg':
            concatenated_res = torch.stack(res, dim=-1)  # [B, C, H, K]
            num = len(self.period_list) + 1 if self.global_freq_pred else len(self.period_list)
            res = torch.sum(concatenated_res * (1 / num), -1)  # [B, C, H]
        if self.revin: 
            res = res.permute(0,2,1)
            res = self.revin_layer(res, 'denorm')
            res = res.permute(0,2,1)
        return res

class FRNet_trend_backone(nn.Module):
    # Trend branch: one FRTrendBackbone per overlapping patch size.
    def __init__(self, c_in, seq_len, pred_len, emb, revin, dropout, e_layers, pred_head_type, aggregation_type, channel_attention, global_freq_pred, patch_len, stride):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = c_in
        self.revin = revin
        self.channel_attention = channel_attention
        self.patch_list = [32, 64]
        self.stride_list = [16, 32]
        self.FRBackboneList = nn.ModuleList() 
        self.global_freq_pred = global_freq_pred
        if self.global_freq_pred:
            self.dominance_freq=int(self.seq_len // self.patch_list[0] + 1) * 2 + 10 
            self.length_ratio = (self.pred_len)/self.seq_len 
            self.freq_Linear = nn.Linear(self.dominance_freq, int(self.dominance_freq*self.length_ratio)).to(torch.cfloat)
            self.fit_channelfuse = ChannelAttentionLayer(FullAttention(attention_dropout=0.1, output_attention=False), seq_len, 8)
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=True, subtract_last=False)
        for k in range(len(self.patch_list)):
            self.FRBackboneList.append(FRTrendBackbone(self.patch_list[k], self.stride_list[k], c_in, seq_len, pred_len, emb, dropout, e_layers, channel_attention))
        self.aggregation_type = aggregation_type
        if aggregation_type == 'linear': 
            if global_freq_pred:
                self.fuse_linear = nn.Linear(pred_len*(1 + len(self.patch_list)), pred_len)
            else:
                self.fuse_linear = nn.Linear(pred_len*len(self.patch_list), pred_len)
        
    def forward(self, x):
        # x: [B, C, L]
        if self.revin: 
            x = x.permute(0,2,1)
            x = self.revin_layer(x, 'norm')
            x = x.permute(0,2,1)
        res = []  # per-branch forecasts, each [B, C, H]
        if self.global_freq_pred:
            low_specx = torch.fft.rfft(x, dim=-1)  # [B, C, L/2+1]
            low_specx[:,:,self.dominance_freq:]=0
            low_specx = low_specx[:,:,0:self.dominance_freq]  # [B, C, F]
            low_specxy_ = self.freq_Linear(low_specx)  # [B, C, F*H/L]
            low_specxy = torch.zeros([low_specxy_.size(0), low_specxy_.size(1), int((self.pred_len)/2+1)],dtype=low_specxy_.dtype).to(low_specxy_.device)  # [B, C, H/2+1]

            low_specxy[:,:,0:low_specxy_.size(-1)]=low_specxy_ 
            low_xy=torch.fft.irfft(low_specxy, dim=-1)  # [B, C, H]
            res.append(low_xy)
        
        for k in range(len(self.patch_list)):
            length = self.seq_len
            stride_i = self.stride_list[k]
            patch_i = self.patch_list[k]
            x_k = x
            x_k = x_k.unfold(dimension=-1, size=patch_i, step=stride_i)  # [B, C, N, P]
            pred_k = self.FRBackboneList[k](x_k)  # [B, C, H]
            res.append(pred_k)
        if self.aggregation_type == 'linear':
            concatenated_res = torch.cat(res, dim=-1)  # [B, C, K*H]
            res = self.fuse_linear(concatenated_res)  # [B, C, H]
        elif self.aggregation_type == 'avg':
            concatenated_res = torch.stack(res, dim=-1)  # [B, C, H, K]
            num = len(self.patch_list) + 1 if self.global_freq_pred else len(self.patch_list)
            res = torch.sum(concatenated_res * (1 / num), -1)  # [B, C, H]
        if self.revin: 
            res = res.permute(0,2,1)
            res = self.revin_layer(res, 'denorm')
            res = res.permute(0,2,1)
        return res

class FRTrendBackbone(nn.Module):
    # Same as FRBackbone, but on overlapping patches and without a truncation head.
    def __init__(self, patch_len, stride_len, c_in, seq_len, pred_len, emb, dropout, e_layers, channel_attention):
        super(FRTrendBackbone, self).__init__()
        self.patch_len = patch_len
        self.strde_len = stride_len
        self.W_P = nn.Linear(patch_len, emb)
        self.dropout = nn.Dropout(dropout)
        self.e_layers = e_layers
        self.experts_num = 8
        patch_input_len = int((seq_len - patch_len) / stride_len + 1)
        frequency_num = emb // 2 + 1
        channel_frequency_num = patch_input_len*emb
        self.period_pred = nn.ModuleList([Period_Frequency_Mixer_Predictor(patch_input_len, patch_input_len, frequency_num) for i in range(e_layers)])
        self.channel_attention = channel_attention
        if channel_attention:
            self.Channel_attention_layers = ChannelAttentionLayer(FullAttention(attention_dropout=0.1, output_attention=False), channel_frequency_num, 8)

        self.W_pos = nn.Parameter(torch.empty(1,1, patch_input_len, emb)) 
        nn.init.uniform_(self.W_pos, -0.02, 0.02)
        self.pred_linear = nn.Linear(patch_input_len * emb, pred_len)
        
    def RIN(self, x):
        x_mean = torch.mean(x, dim=2, keepdim=True)
        x = x - x_mean
        x_var=torch.var(x, dim=2, keepdim=True)+ 1e-5
        x = x / torch.sqrt(x_var)
        return x, x_var, x_mean 
        
    def forward(self, x):
        x = self.dropout(self.W_P(x)) + self.W_pos  # [B, C, N, P] -> [B, C, N, E]
        B, C, N, E = x.size() 
        if self.channel_attention:
            x = torch.reshape(x, (B, C, N*E))   # [B, C, N*E]
            x = self.Channel_attention_layers(x,x,x)[0]
            x = torch.reshape(x, (B, C, N, E))  # [B, C, N, E]
        period_fft = torch.fft.rfft(x, dim=-1)  # [B, C, N, D]
        _, _, _, D = period_fft.size()  
        # Frequency rotation layers, normalised by magnitude, with residual.
        for i in range(self.e_layers):  
            period_fft_resNet = period_fft 
            magnitude = torch.abs(period_fft).mean(dim=-2, keepdim=True)
            normalized_magnitude = magnitude / torch.sum(magnitude, dim=-1, keepdim=True)
            period_fft = period_fft / normalized_magnitude
            period_fft = torch.reshape(period_fft, (B*C, N, D))  # [B*C, N, D]
            period_fft = self.period_pred[i](period_fft.permute(0,2,1)).permute(0,2,1)  # [B*C, N, D]
            period_fft = torch.reshape(period_fft, (B, C, N, D))   # [B, C, N, D]
            period_fft = period_fft * normalized_magnitude
            period_fft = period_fft + period_fft_resNet
        x = torch.fft.irfft(period_fft, dim=-1)  # [B, C, N, E]

        x = torch.flatten(x, start_dim=-2, end_dim=-1)  # [B, C, N*E]
        x = self.pred_linear(x)  # [B, C, H]
        return x

class FRBackbone(nn.Module):
    # Embeds patches, rotates them in frequency, then projects to H.
    def __init__(self, period_len, c_in, seq_len, pred_len, emb, dropout, e_layers, pred_head_type, channel_attention):
        super(FRBackbone, self).__init__()
        self.period_len = period_len
        self.W_P = nn.Linear(period_len, emb)
        self.dropout = nn.Dropout(dropout)
        self.e_layers = e_layers
        self.experts_num = 8
        period_input_len = math.floor(seq_len/period_len) 
        period_output_len =  math.ceil(pred_len/period_len) 
        self.period_output_len = period_output_len
        frequency_num = emb // 2 + 1
        channel_frequency_num = period_input_len*emb
        self.period_pred = nn.ModuleList([Period_Frequency_Mixer_Predictor(period_input_len, period_input_len, frequency_num) for i in range(e_layers)])
        self.channel_attention = channel_attention
        if channel_attention:
            self.Channel_attention_layers = ChannelAttentionLayer(FullAttention(attention_dropout=0.1, output_attention=False), channel_frequency_num, 8)

        if pred_head_type == 'truncation':
            self.pred_head = Period_Frequency_Mixer_PredictorHead(period_input_len, period_output_len, frequency_num)
        self.W_pos = nn.Parameter(torch.empty(1,1, period_input_len, emb)) 
        nn.init.uniform_(self.W_pos, -0.02, 0.02)
        self.pred_head_type = pred_head_type
        if pred_head_type == 'truncation':
            self.pred_linear = nn.Linear(period_output_len * emb, pred_len)
        else:
            self.pred_linear = nn.Linear(period_input_len * emb, pred_len)
        
    def RIN(self, x):
        x_mean = torch.mean(x, dim=2, keepdim=True)
        x = x - x_mean
        x_var=torch.var(x, dim=2, keepdim=True)+ 1e-5
        x = x / torch.sqrt(x_var)
        return x, x_var, x_mean 
        
    def forward(self, x):
        x = self.dropout(self.W_P(x)) + self.W_pos  # [B, C, N, P] -> [B, C, N, E]
        B, C, N, E = x.size() 
        if self.channel_attention:
            x = torch.reshape(x, (B, C, N*E))   # [B, C, N*E]
            x = self.Channel_attention_layers(x,x,x)[0]
            x = torch.reshape(x, (B, C, N, E))  # [B, C, N, E]
        period_fft = torch.fft.rfft(x, dim=-1)  # [B, C, N, D]
        _, _, _, D = period_fft.size()  
        # Frequency rotation layers, normalised by magnitude, with residual.
        for i in range(self.e_layers):  
            period_fft_resNet = period_fft 
            magnitude = torch.abs(period_fft).mean(dim=-2, keepdim=True)
            normalized_magnitude = magnitude / torch.sum(magnitude, dim=-1, keepdim=True)
            period_fft = period_fft / normalized_magnitude
            period_fft = torch.reshape(period_fft, (B*C, N, D))  # [B*C, N, D]
            period_fft = self.period_pred[i](period_fft.permute(0,2,1)).permute(0,2,1)  # [B*C, N, D]
            period_fft = torch.reshape(period_fft, (B, C, N, D))   # [B, C, N, D]
            period_fft = period_fft * normalized_magnitude
            period_fft = period_fft + period_fft_resNet
        x = torch.fft.irfft(period_fft, dim=-1)  # [B, C, N, E]
        
        if self.pred_head_type == 'truncation':
            period_fft = torch.fft.rfft(x, dim=-1)  # [B, C, N, D]
            _, _, _, D = period_fft.size()  
            magnitude = torch.abs(period_fft).mean(dim=-2, keepdim=True)
            normalized_magnitude = magnitude / torch.sum(magnitude, dim=-1, keepdim=True)
            period_fft = period_fft / normalized_magnitude
            period_fft = torch.reshape(period_fft, (B*C, N, D))  # [B*C, N, D]
            period_fft = self.pred_head(period_fft.permute(0,2,1)).permute(0,2,1)  # [B*C, N_out, D]
            period_fft = torch.reshape(period_fft, (B, C, self.period_output_len, D))   # [B, C, N_out, D]
            period_fft = period_fft * normalized_magnitude
            x = torch.fft.irfft(period_fft, dim=-1)  # [B, C, N_out, E]
        x = torch.flatten(x, start_dim=-2, end_dim=-1)  # [B, C, N*E]
        x = self.pred_linear(x)  # [B, C, H]
        return x
        
class Period_Frequency_Mixer_Predictor(nn.Module):
    # Complex mixing across patches, then across frequencies, with residuals.
    def __init__(self, in_features, out_features, frequency_num):
        super(Period_Frequency_Mixer_Predictor, self).__init__()
        self.linear_1 =  nn.Linear(in_features, out_features).to(torch.cfloat) 
        self.frequency_mix = nn.Linear(frequency_num, frequency_num).to(torch.cfloat) 
    def forward(self, x):
        output = self.linear_1(x) + x
        output = self.frequency_mix(output.permute(0,2,1)).permute(0,2,1) + output
        return  output
    
class Period_Frequency_Mixer_PredictorHead(nn.Module):
    # Same mixer, mapping N input patches to N_out patches.
    def __init__(self, in_features, out_features, frequency_num):
        super(Period_Frequency_Mixer_PredictorHead, self).__init__()
        self.linear_1 =  nn.Linear(in_features, out_features).to(torch.cfloat) 
        self.frequency_mix = nn.Linear(frequency_num, frequency_num).to(torch.cfloat) 
    def forward(self, x):
        output = self.linear_1(x)
        output = self.frequency_mix(output.permute(0,2,1)).permute(0,2,1) + output
        return  output

class ChannelAttentionLayer(nn.Module):
    # Multi-head attention across channels.
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(ChannelAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
    
class FullAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)
