# SparseTSF: cross-period sparse forecasting.
# Source: https://github.com/lss-1138/SparseTSF

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # get parameters
        self.seq_len = configs.d_seq_len
        self.pred_len = configs.d_pred_len
        self.enc_in = configs.d_in_features
        self.period_len = configs.m_period_len
        self.d_model = configs.m_d_model
        self.model_type = configs.m_model_type
        assert self.model_type in ['linear', 'mlp']

        assert self.seq_len % self.period_len == 0 and self.pred_len % self.period_len == 0, \
            'SparseTSF needs d_seq_len and d_pred_len to be multiples of m_period_len.'

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.period_len // 2),
                                stride=1, padding=self.period_len // 2, padding_mode="zeros", bias=False)

        if self.model_type == 'linear':
            self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)
        elif self.model_type == 'mlp':
            self.mlp = nn.Sequential(
                nn.Linear(self.seg_num_x, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.seg_num_y)
            )


    def forward(self, x):
        batch_size = x.shape[0]
        # normalization and permute     b,s,c -> b,c,s
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)

        # 1D convolution aggregation
        x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x

        # downsampling: b,c,s -> bc,n,w -> bc,w,n
        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)

        # sparse forecasting
        if self.model_type == 'linear':
            y = self.linear(x)  # bc,w,m
        elif self.model_type == 'mlp':
            y = self.mlp(x)

        # upsampling: bc,w,m -> bc,m,w -> b,c,s
        y = y.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)

        # permute and denorm
        y = y.permute(0, 2, 1) + seq_mean

        return y