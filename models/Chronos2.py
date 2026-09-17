import os

import torch
import torch.nn as nn

from chronos import Chronos2Pipeline

# Chronos-2 is used as a fixed, zero-shot forecaster: the same published
# checkpoint is loaded for every run, frozen, and never fine-tuned locally
# (see exp_main.py / base/config.py -- d_pretrained=true for this model
# skips training and local checkpoint loading entirely).
PRETRAINED_MODEL_ID = "amazon/chronos-2"

# Weights are cached here (not ~/.cache/huggingface) so the exact checkpoint
# used is pinned inside the project rather than whatever happens to be on a
# given machine. Downloaded once, on first use; every run after that loads
# from disk with no network access.
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
        # Registers the underlying HF model as a submodule so Exp_Main's
        # .float()/.to(device) calls on this wrapper actually move it --
        # `pipeline` itself isn't an nn.Module.
        self.chronos_model = pipeline.model

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        # Chronos2Pipeline.predict() builds its own pinned-memory DataLoader
        # internally and moves each batch to the model's device itself, so
        # it expects a CPU input -- Exp_Main already moved x to self.device,
        # and handing it a CUDA tensor trips torch's pin_memory step
        # ("cannot pin ... only dense CPU tensors can be pinned").
        context = x.detach().cpu().permute(0, 2, 1)  # [Batch, Channel, Input length]

        _, mean = self.pipeline.predict_quantiles(
            context,
            prediction_length=self.pred_len,
        )

        mean = torch.stack(mean, dim=0)  # [Batch, Channel, pred_len]
        return mean.permute(0, 2, 1).to(x.device)  # [Batch, pred_len, Channel]
