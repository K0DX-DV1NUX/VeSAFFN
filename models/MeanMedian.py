import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Mean/median forecasting baselines.

    m_stat_type = "mean":
        Repeat the per-channel mean of the input sequence for every future step.

    m_stat_type = "median":
        Repeat the per-channel median of the input sequence for every future step.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.d_seq_len
        self.pred_len = configs.d_pred_len
        self.channels = configs.d_in_features
        self.stat_type = getattr(configs, "m_stat_type", "mean").lower()

        self.dummy_param = nn.Parameter(torch.zeros(1))

        if self.stat_type not in {"mean", "median"}:
            raise ValueError(
                f"Unknown m_stat_type '{self.stat_type}'. "
                "Expected one of: mean, median."
            )

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        if self.channels != x.shape[-1]:
            raise ValueError(
                "Model configuration does not match the dataset. "
                f"Number of channels in the dataset: {x.shape[-1]}, "
                f"number of channels expected by the model: {self.channels}"
            )

        if self.stat_type == "mean":
            forecast_value = x.mean(dim=1, keepdim=True)
        else:
            forecast_value = x.median(dim=1, keepdim=True).values

        return forecast_value.repeat(1, self.pred_len, 1) + (0.0 * self.dummy_param)
