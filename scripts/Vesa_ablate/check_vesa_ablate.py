# Sanity checks for Vesa_ablate. Run from the repo root: python scripts/Vesa_ablate/check_vesa_ablate.py
import os
import sys
import tomllib
from types import SimpleNamespace

import numpy as np
import pywt
import torch

sys.path.append(os.getcwd())

from models import Vesa, Vesa_ablate
from models.Vesa_ablate import WaveletDecomp

WAVELETS = ["db1", "db3", "db6", "coif1", "coif2", "coif3", "sym4", "sym6", "sym8"]
TOL = 1e-5


def make_configs(**overrides):
    values = dict(
        d_seq_len=336, d_pred_len=96, d_in_features=7, d_out_features=7,
        m_individual=False, m_seq_dim=256, m_vesa_rank=16, m_use_norm=True,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def check_wavelets():
    torch.manual_seed(0)
    for name in WAVELETS:
        decomp = WaveletDecomp(name)
        for length in (96, 336, 720):
            x = torch.randn(4, 7, length)
            coeffs = decomp(x)
            assert coeffs.shape == (4, 14, length // 2)

            # Same coefficients as PyWavelets.
            ca, cd = pywt.dwt(x.numpy(), name, mode="periodization", axis=-1)
            assert np.abs(coeffs[:, :7].numpy() - ca).max() < TOL, name
            assert np.abs(coeffs[:, 7:].numpy() - cd).max() < TOL, name

            # Perfect reconstruction.
            assert (decomp.inverse(coeffs) - x).abs().max() < TOL, name

            # Same inverse as PyWavelets.
            rec = pywt.idwt(coeffs[:, :7].numpy(), coeffs[:, 7:].numpy(), name, mode="periodization", axis=-1)
            assert np.abs(decomp.inverse(coeffs).numpy() - rec).max() < TOL, name


def check_trend_residual():
    x = torch.randn(4, 7, 336)
    decomp = Vesa_ablate.TrendResidualDecomp(25)
    bands = decomp(x)
    assert bands.shape == (4, 14, 336)
    assert (decomp.inverse(bands) - x).abs().max() < TOL


def check_haar_matches_vesa():
    # db1 must reproduce the original Haar model exactly.
    for individual in (False, True):
        configs = make_configs(m_individual=individual)
        torch.manual_seed(0)
        base = Vesa.Model(configs)
        torch.manual_seed(0)
        ablate = Vesa_ablate.Model(configs)
        x = torch.randn(4, 336, 7)
        assert (base(x) - ablate(x)).abs().max() < TOL


def check_tomls_end_to_end():
    toml_dir = "scripts/Vesa_ablate/toml"
    tags = set()
    for file_name in ("wavelet.toml", "trend_residual.toml", "no_decomp.toml"):
        with open(os.path.join(toml_dir, file_name), "rb") as f:
            values = tomllib.load(f)
        for run in values["runs"]:
            merged = {**values["default"], **run}
            tags.add((merged["d_run_tag"], merged["d_pred_len"]))
            for channels in (7, 21, 56):
                configs = SimpleNamespace(**{**merged, "d_in_features": channels, "d_out_features": channels})
                model = Vesa_ablate.Model(configs).float()
                out = model(torch.randn(2, configs.d_seq_len, channels))
                assert out.shape == (2, configs.d_pred_len, channels)
                out.mean().backward()
    assert len(tags) == 11 * 4


if __name__ == "__main__":
    check_wavelets()
    check_trend_residual()
    check_haar_matches_vesa()
    check_tomls_end_to_end()
    print("All Vesa_ablate checks passed.")
