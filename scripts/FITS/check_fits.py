import sys
import tomllib
from pathlib import Path

import torch

# Shape and gradient check for FITS with every standalone toml.
# Run from the repo root: python scripts/FITS/check_fits.py

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base.config import ExpConfig
from models import FITS


def build_cfg(toml: str, pred_len: int, channels: int = 7) -> ExpConfig:
    with open(f"scripts/FITS/toml/standalone/{toml}.toml", "rb") as f:
        m = tomllib.load(f)["default"]
    return ExpConfig(**{**m, "d_data": "ETTh1", "d_root_path": "../dataset", "d_data_path": "ETTh1.csv",
                        "d_seq_len": 336, "d_pred_len": pred_len,
                        "d_in_features": channels, "d_out_features": channels})


if __name__ == "__main__":
    for toml in ("hourly", "ett_15min", "tenmin", "fivemin"):
        for individual in (0, 1):
            for pred_len in (96, 720):
                cfg = build_cfg(toml, pred_len)
                cfg.m_individual = individual
                model = FITS.Model(cfg).float()
                x = torch.randn(4, cfg.d_seq_len, cfg.d_in_features)

                y = model(x)
                assert torch.is_tensor(y), "forward must return a single tensor (exp_main expects that)"
                assert y.shape == (4, cfg.d_seq_len + pred_len, cfg.d_in_features), y.shape
                assert y[:, -pred_len:].shape == (4, pred_len, cfg.d_in_features)
                assert torch.isfinite(y).all()

                y.pow(2).mean().backward()
                assert all(p.grad is not None for p in model.parameters()), "unconnected parameter"

                n_obj = sum(p.numel() for p in model.parameters())
                print(f"{toml:9s} individual={individual} pred_len={pred_len}: cut_freq={model.dominance_freq} params(objects)={n_obj}")
    print("OK")
