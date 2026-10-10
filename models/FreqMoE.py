# FreqMoE: frequency-band mixture of experts with residual frequency blocks.
# Source: https://github.com/sunbus100/FreqMoE-main

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexReLU(nn.Module):
    def forward(self, x):
        real = torch.relu(x.real)
        imag = torch.relu(x.imag)
        return torch.complex(real, imag)


class ComplexDropout(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(ComplexDropout, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        real = self.dropout(x.real)
        imag = self.dropout(x.imag)
        return torch.complex(real, imag)


class FreqDecompMoE(nn.Module):
    # Splits the spectrum into learnable bands and recombines them with gating weights.

    def __init__(self, expert_num, seq_len):
        super(FreqDecompMoE, self).__init__()
        self.expert_num = expert_num
        self.seq_len = seq_len
        self.freq_len = seq_len // 2 + 1

        self.band_boundaries = nn.Parameter(torch.rand(self.expert_num - 1))

        self.gating_network = nn.Sequential(nn.Linear(self.freq_len, self.freq_len), nn.ReLU(), nn.Linear(self.freq_len, self.expert_num))


    def forward(self, x):
        # x: [B, C, L]
        x_mean = torch.mean(x, dim=2, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=2, keepdim=True) + 1e-5
        x = x / torch.sqrt(x_var)
        freq_x = torch.fft.rfft(x, dim=-1)                      # [B, C, F], F = L/2+1
        total_freq_size = freq_x.size(-1)

        boundaries = torch.sigmoid(self.band_boundaries)
        boundaries, _ = torch.sort(boundaries)


        boundaries = torch.cat([
            torch.tensor([0.0], device=boundaries.device),
            boundaries,
            torch.tensor([1.0], device=boundaries.device)
        ])


        # Band edges as frequency indices, [E + 1].
        indices = (boundaries * total_freq_size).long()

        indices[-1] = total_freq_size

        components = []
        for i in range(self.expert_num):
            start_idx = indices[i].item()
            end_idx = indices[i + 1].item()

            freq_mask = torch.zeros_like(freq_x)
            if end_idx > start_idx:
                freq_mask[:, :, start_idx:end_idx] = 1

            expert_component = freq_x * freq_mask

            components.append(expert_component.unsqueeze(-1))  # [B, C, F, 1]

        freq_magnitude = torch.abs(freq_x)

        gating_input = freq_magnitude.mean(dim=1)               # [B, F]

        gating_scores = nn.Softmax(dim=-1)(self.gating_network(gating_input))  # [B, E]

        components = torch.cat(components, dim=-1)              # [B, C, F, E]

        gating_scores = gating_scores.unsqueeze(1).unsqueeze(2) # [B, 1, 1, E]

        combined_freq_output = torch.sum(components * gating_scores, dim=-1)  # [B, C, F]
        combined_output = torch.fft.irfft(combined_freq_output, n=self.seq_len)  # [B, C, L]
        residual = x - combined_output

        combined_output = combined_output * torch.sqrt(x_var) + x_mean


        return combined_output, boundaries, gating_scores.squeeze(1).squeeze(1)


class Block(nn.Module):
    # Complex two-layer upsampler in frequency: L -> L + H.
    def __init__(self, seq_len, pred_len, enc_in, dropout):
        super(Block, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len
        self.dominance_freq = int(self.seq_len / 2 + 1)
        self.pre_freq = int((self.seq_len + self.pred_len) / 2 + 1)
        self.freq_upsampler = nn.Linear(self.dominance_freq, self.pre_freq).to(torch.cfloat)
        self.freq_upsampler1 = nn.Linear(self.pre_freq, self.pre_freq).to(torch.cfloat)
        self.dropout = dropout
        self.complex_relu = ComplexReLU()
        self.complex_dropout = ComplexDropout(dropout_rate=self.dropout)

    def forward(self, x):
        # x: [B, L, C]
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var=torch.var(x, dim=1, keepdim=True)+ 1e-5
        x = x / torch.sqrt(x_var)

        spec_x = torch.fft.rfft(x, dim=1)                                           # [B, L/2+1, C]
        spec_x_up = self.freq_upsampler(spec_x.permute(0, 2, 1)).permute(0, 2, 1)   # [B, (L+H)/2+1, C]
        spec_x_up = self.complex_relu(spec_x_up)
        spec_x_up = self.complex_dropout(spec_x_up)
        spec_x_up = self.freq_upsampler1(spec_x_up.permute(0,2,1)).permute(0, 2, 1)
        x_up = torch.fft.irfft(spec_x_up, dim=1)                                    # [B, L+H, C]
        x_up = x_up * self.length_ratio
        x_up = (x_up) * torch.sqrt(x_var) + x_mean
        return x_up


class Model(nn.Module):
    def __init__(self, conifgs):
        super(Model, self).__init__()
        self.seq_len = conifgs.d_seq_len
        self.pred_len = conifgs.d_pred_len
        self.channels = conifgs.d_in_features
        self.num_blocks = conifgs.m_freq_num_blocks
        self.exeprt_num = conifgs.m_expert_num
        self.dropout = conifgs.m_dropout_freq

        self.blocks = nn.ModuleList([Block(self.seq_len, self.pred_len, conifgs.d_in_features, self.dropout) for _ in range(self.num_blocks)])

        self.moe = FreqDecompMoE(self.exeprt_num, self.seq_len)

    def forward(self, x):
        # x: [B, L, C]
        x, bound, weight = self.moe(x.permute(0, 2, 1))    # [B, C, L]
        x = x.permute(0, 2, 1)                             # [B, L, C]
        residual = x
        total_prediction = torch.zeros(
            x.shape[0], self.pred_len, x.shape[2], device=x.device, dtype=x.dtype
        )                                                  # [B, H, C]

        # Each block fits the residual left by the previous ones.
        for i, block in enumerate(self.blocks):
            prediction = block(residual)                   # [B, L+H, C]

            if i == 0:
                residual = x[:, :self.seq_len] - prediction[:, :self.seq_len]
            else:
                residual = residual - prediction[:, :self.seq_len]

            total_prediction += prediction[:, self.seq_len:self.pred_len + self.seq_len]

        return total_prediction                            # [B, H, C]
