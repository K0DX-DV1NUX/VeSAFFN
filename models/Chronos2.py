import os

import torch
import torch.nn as nn

from chronos import Chronos2Pipeline

# Zero-shot, frozen Chronos-2 forecaster.
PRETRAINED_MODEL_ID = "amazon/chronos-2"

# Weights are downloaded once and cached here.
PRETRAINED_LOCAL_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "scripts", "Chronos2", "pre_trained")
)


def _load_pipeline():
    if os.path.exists(os.path.join(PRETRAINED_LOCAL_DIR, "config.json")):
        return Chronos2Pipeline.from_pretrained(PRETRAINED_LOCAL_DIR)

    pipeline = Chronos2Pipeline.from_pretrained(PRETRAINED_MODEL_ID, device_map="cuda")
    os.makedirs(PRETRAINED_LOCAL_DIR, exist_ok=True)
    pipeline.save_pretrained(PRETRAINED_LOCAL_DIR)
    return pipeline


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.d_pred_len

        pipeline = _load_pipeline()
        for param in pipeline.model.parameters():
            param.requires_grad_(False)
        pipeline.model.eval()

        self.pipeline = pipeline
        # Registered as a submodule so .to(device) moves it.
        self.chronos_model = pipeline.model

    def forward(self, x):
        # x: [B, L, C]
        # The pipeline expects a CPU tensor.
        context = x.detach().cpu().permute(0, 2, 1)        # [B, C, L]

        _, mean = self.pipeline.predict_quantiles(
            context,
            prediction_length=self.pred_len,
        )                                                   # list of B x [C, H]

        mean = torch.stack(mean, dim=0)                     # [B, C, H]
        return mean.permute(0, 2, 1).to(x.device)           # [B, H, C]
