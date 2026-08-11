import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Naive forecasting baselines.

    m_naive_type = "simple":
        Repeat the final observed value for every future time step.

    m_naive_type = "ltsf":
        Copy the final d_pred_len values from the input sequence.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.d_seq_len
        self.pred_len = configs.d_pred_len
        self.channels = configs.d_in_features
        self.naive_type = getattr(configs, "m_naive_type", "simple").lower()
        self.dummy_param = nn.Parameter(torch.zeros(1))

        if self.naive_type not in {"simple", "ltsf"}:
            raise ValueError(
                f"Unknown m_naive_type '{self.naive_type}'. "
                "Expected one of: simple, ltsf."
            )

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        if self.channels != x.shape[-1]:
            raise ValueError(
                "Model configuration does not match the dataset. "
                f"Number of channels in the dataset: {x.shape[-1]}, "
                f"number of channels expected by the model: {self.channels}"
            )

        if self.naive_type == "simple":
            forecast = x[:, -1:, :].repeat(1, self.pred_len, 1)
            return forecast + (0.0 * self.dummy_param)

        if self.pred_len > x.shape[1]:
            raise ValueError(
                "LTSF naive requires d_pred_len to be less than or equal to "
                f"the input sequence length. Got d_pred_len={self.pred_len}, "
                f"input length={x.shape[1]}."
            )

        forecast = x[:, -self.pred_len:, :]
        return forecast + (0.0 * self.dummy_param)
