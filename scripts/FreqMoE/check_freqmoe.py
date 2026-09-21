"""Sanity check for models/FreqMoE.py: run from the repo root with
    .venv/bin/python scripts/FreqMoE/check_freqmoe.py
"""
import sys
import tomllib
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base.config import ExpConfig
from models import FreqMoE


def build_cfg(pred_len: int, channels: int = 7) -> ExpConfig:
    with open("scripts/FreqMoE/toml/model.toml", "rb") as f:
        m = tomllib.load(f)["default"]
    return ExpConfig(**{**m, "d_data": "ETTh1", "d_root_path": "../dataset", "d_data_path": "ETTh1.csv",
                        "d_seq_len": 336, "d_pred_len": pred_len,
                        "d_in_features": channels, "d_out_features": channels})


if __name__ == "__main__":
    for pred_len in (96, 720):
        cfg = build_cfg(pred_len)
        model = FreqMoE.Model(cfg).float()
        x = torch.randn(4, cfg.d_seq_len, cfg.d_in_features)

        y = model(x)
        assert torch.is_tensor(y), "forward must return a single tensor (exp_main expects that)"
        assert y.shape == (4, pred_len, cfg.d_in_features), y.shape
        assert torch.isfinite(y).all()

        y.pow(2).mean().backward()
        # band_boundaries gets no gradient: upstream turns it into indices via .long(), which cuts
        # the graph (kept as-is to stay faithful to the reference code).
        dead = [n for n, p in model.named_parameters() if p.grad is None]
        assert dead == ["moe.band_boundaries"], dead

        model.eval()
        with torch.no_grad():
            assert model(x).shape == (4, pred_len, cfg.d_in_features)

        n_obj = sum(p.numel() for p in model.parameters())
        n_real = sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())
        print(f"pred_len={pred_len}: out={tuple(y.shape)} params(objects)={n_obj} params(real scalars)={n_real}")
    print("OK")
