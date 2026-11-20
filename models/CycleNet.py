# CycleNet: Residual Cycle Forecasting (RCF) with a linear / MLP backbone.
# Paper: Lin et al., "CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns",
# NeurIPS 2024, arXiv:2409.18479. Source: https://github.com/ACAT-SCUT/CycleNet
#
# Shapes: B = batch, L = seq_len, H = pred_len, C = channels, W = cycle length (m_cycle).
# Input type "x_index": forward(x, time_index), time_index [B, 3] = (position, hour, weekday) at the first
# forecast step (data_provider TIME_INDEX_COLS). As upstream, the cycle index is that row position mod W.

import torch
import torch.nn as nn


class RecurrentCycle(nn.Module):
    # Learnable cycle Q [W, C], read with modulo index arithmetic (upstream's gather version).
    def __init__(self, cycle_len, channel_size):
        super().__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = nn.Parameter(torch.zeros(cycle_len, channel_size))   # [W, C]

    def forward(self, index, length):
        # index: [B] start positions in the cycle -> Q rows for `length` steps, [B, length, C]
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len
        return self.data[gather_index]


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.d_seq_len
        self.pred_len = configs.d_pred_len
        self.enc_in = configs.d_in_features
        self.cycle_len = configs.m_cycle
        self.model_type = getattr(configs, "m_model_type", "linear")
        self.d_model = getattr(configs, "m_d_model", 512)
        self.use_revin = bool(getattr(configs, "m_use_revin", True))
        assert self.model_type in ["linear", "mlp"], f"m_model_type must be linear or mlp, got {self.model_type}"

        self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)

        if self.model_type == "linear":
            self.model = nn.Linear(self.seq_len, self.pred_len)
        else:
            self.model = nn.Sequential(
                nn.Linear(self.seq_len, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.pred_len),
            )

    def forward(self, x, time_index):
        # x: [B, L, C], time_index: [B, 3]
        cycle_index = time_index[:, 0] % self.cycle_len                     # [B]

        # Instance norm.
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)                    # [B, 1, C]
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5               # [B, 1, C]
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # Remove the cycle from the look-back window.
        x = x - self.cycleQueue(cycle_index, self.seq_len)                   # [B, L, C]

        # Channel-independent forecast of the residual, shared weights.
        y = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)                  # [B, H, C]

        # Add the cycle back for the horizon.
        y = y + self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)  # [B, H, C]

        # Instance denorm.
        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        return y
