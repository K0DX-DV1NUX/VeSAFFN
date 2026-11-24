# Decompositions for Vesa_ablate: one-level wavelet, DLinear trend-residual, or none.
# Each maps [B, C, L] -> [B, n_bands * C, L / length_div] and inverts it back.

import torch
import torch.nn as nn
import torch.nn.functional as F


# Shapes: B = batch, C = channels, L = input length, H = output length, N = filter length.


# Orthogonal wavelet filters in conv1d (time-reversed) order, values from PyWavelets.
WAVELET_FILTERS = {
    "db1": {
        "lo": [
            0.7071067811865476,
            0.7071067811865476,
        ],
        "hi": [
            0.7071067811865476,
            -0.7071067811865476,
        ],
    },
    "db3": {
        "lo": [
            0.33267055295008263,
            0.8068915093110925,
            0.45987750211849154,
            -0.13501102001025458,
            -0.08544127388202666,
            0.03522629188570953,
        ],
        "hi": [
            0.03522629188570953,
            0.08544127388202666,
            -0.13501102001025458,
            -0.45987750211849154,
            0.8068915093110925,
            -0.33267055295008263,
        ],
    },
    "db6": {
        "lo": [
            0.11154074335010947,
            0.49462389039845306,
            0.7511339080210954,
            0.31525035170919763,
            -0.22626469396543983,
            -0.12976686756726194,
            0.09750160558732304,
            0.027522865530305727,
            -0.03158203931748603,
            0.0005538422011614961,
            0.004777257510945511,
            -0.0010773010853084796,
        ],
        "hi": [
            -0.0010773010853084796,
            -0.004777257510945511,
            0.0005538422011614961,
            0.03158203931748603,
            0.027522865530305727,
            -0.09750160558732304,
            -0.12976686756726194,
            0.22626469396543983,
            0.31525035170919763,
            -0.7511339080210954,
            0.49462389039845306,
            -0.11154074335010947,
        ],
    },
    "coif1": {
        "lo": [
            -0.07273261951252645,
            0.3378976624574818,
            0.8525720202116004,
            0.3848648468648578,
            -0.07273261951252645,
            -0.015655728135791993,
        ],
        "hi": [
            -0.015655728135791993,
            0.07273261951252645,
            0.3848648468648578,
            -0.8525720202116004,
            0.3378976624574818,
            0.07273261951252645,
        ],
    },
    "coif2": {
        "lo": [
            0.01638733646320364,
            -0.04146493678687178,
            -0.0673725547237256,
            0.3861100668227629,
            0.8127236354494135,
            0.4170051844232391,
            -0.07648859907828076,
            -0.05943441864643109,
            0.02368017194684777,
            0.005611434819368834,
            -0.0018232088709110323,
            -0.000720549445520347,
        ],
        "hi": [
            -0.000720549445520347,
            0.0018232088709110323,
            0.005611434819368834,
            -0.02368017194684777,
            -0.05943441864643109,
            0.07648859907828076,
            0.4170051844232391,
            -0.8127236354494135,
            0.3861100668227629,
            0.0673725547237256,
            -0.04146493678687178,
            -0.01638733646320364,
        ],
    },
    "coif3": {
        "lo": [
            -0.003793512864380802,
            0.007782596425672746,
            0.023452696142077168,
            -0.06577191128146936,
            -0.06112339000297255,
            0.40517690240911824,
            0.7937772226260872,
            0.42848347637737,
            -0.07179982161915484,
            -0.08230192710629983,
            0.03455502757329774,
            0.015880544863669452,
            -0.009007976136730624,
            -0.0025745176881367972,
            0.0011175187708306303,
            0.0004662169598204029,
            -7.0983302506379e-05,
            -3.459977319727278e-05,
        ],
        "hi": [
            -3.459977319727278e-05,
            7.0983302506379e-05,
            0.0004662169598204029,
            -0.0011175187708306303,
            -0.0025745176881367972,
            0.009007976136730624,
            0.015880544863669452,
            -0.03455502757329774,
            -0.08230192710629983,
            0.07179982161915484,
            0.42848347637737,
            -0.7937772226260872,
            0.40517690240911824,
            0.06112339000297255,
            -0.06577191128146936,
            -0.023452696142077168,
            0.007782596425672746,
            0.003793512864380802,
        ],
    },
    "sym4": {
        "lo": [
            0.0322231006040427,
            -0.012603967262037833,
            -0.09921954357684722,
            0.29785779560527736,
            0.8037387518059161,
            0.49761866763201545,
            -0.02963552764599851,
            -0.07576571478927333,
        ],
        "hi": [
            -0.07576571478927333,
            0.02963552764599851,
            0.49761866763201545,
            -0.8037387518059161,
            0.29785779560527736,
            0.09921954357684722,
            -0.012603967262037833,
            -0.0322231006040427,
        ],
    },
    "sym6": {
        "lo": [
            -0.007800708325034148,
            0.0017677118642428036,
            0.04472490177066578,
            -0.021060292512300564,
            -0.07263752278646252,
            0.3379294217276218,
            0.787641141030194,
            0.4910559419267466,
            -0.048311742585633,
            -0.11799011114819057,
            0.0034907120842174702,
            0.015404109327027373,
        ],
        "hi": [
            0.015404109327027373,
            -0.0034907120842174702,
            -0.11799011114819057,
            0.048311742585633,
            0.4910559419267466,
            -0.787641141030194,
            0.3379294217276218,
            0.07263752278646252,
            -0.021060292512300564,
            -0.04472490177066578,
            0.0017677118642428036,
            0.007800708325034148,
        ],
    },
    "sym8": {
        "lo": [
            0.0018899503327594609,
            -0.0003029205147213668,
            -0.01495225833704823,
            0.003808752013890615,
            0.049137179673607506,
            -0.027219029917056003,
            -0.05194583810770904,
            0.3644418948353314,
            0.7771857517005235,
            0.4813596512583722,
            -0.061273359067658524,
            -0.1432942383508097,
            0.007607487324917605,
            0.03169508781149298,
            -0.0005421323317911481,
            -0.0033824159510061256,
        ],
        "hi": [
            -0.0033824159510061256,
            0.0005421323317911481,
            0.03169508781149298,
            -0.007607487324917605,
            -0.1432942383508097,
            0.061273359067658524,
            0.4813596512583722,
            -0.7771857517005235,
            0.3644418948353314,
            0.05194583810770904,
            -0.027219029917056003,
            -0.049137179673607506,
            0.003808752013890615,
            0.01495225833704823,
            -0.0003029205147213668,
            -0.0018899503327594609,
        ],
    },
}


class WaveletDecomp(nn.Module):
    # One-level orthogonal DWT with periodic boundaries, as fixed conv1d filters.
    n_bands = 2
    length_div = 2

    def __init__(self, wavelet):
        super(WaveletDecomp, self).__init__()
        if wavelet not in WAVELET_FILTERS:
            raise ValueError(f"Unknown m_wavelet: {wavelet}. Available: {list(WAVELET_FILTERS)}")

        lo, hi = WAVELET_FILTERS[wavelet]["lo"], WAVELET_FILTERS[wavelet]["hi"]
        filters = torch.tensor([lo, hi], dtype=torch.float32)
        self.register_buffer("filters", filters.unsqueeze(1), persistent=False)  # [2, 1, N]
        self.pad = len(lo) // 2 - 1

    def forward(self, x):
        # x: [B, C, L]
        b, c, l = x.shape
        x = x.reshape(b * c, 1, l)                             # [BC, 1, L]
        x = F.pad(x, (self.pad, self.pad), mode="circular")    # [BC, 1, L + N - 2]
        coeffs = F.conv1d(x, self.filters, stride=2)           # [BC, 2, L/2]
        return coeffs.reshape(b, c, 2, -1).transpose(1, 2).reshape(b, 2 * c, -1)  # [B, 2C, L/2], approx then detail

    def inverse(self, coeffs):
        # coeffs: [B, 2C, H/2], approx then detail
        b, c2, h = coeffs.shape
        c = c2 // 2
        coeffs = coeffs.reshape(b, 2, c, h).transpose(1, 2).reshape(b * c, 2, h)  # [BC, 2, H/2]
        coeffs = F.pad(coeffs, (self.pad, self.pad), mode="circular")             # [BC, 2, H/2 + N - 2]
        out = F.conv_transpose1d(coeffs, self.filters, stride=2)                  # [BC, 1, H + 3N - 6]
        start = 3 * self.pad
        return out[..., start:start + 2 * h].reshape(b, c, 2 * h)                 # [B, C, H]


# Same moving-average decomposition as models/DLinear.py.


class moving_avg(nn.Module):
    # Moving average over time, padded to keep length L.
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [B, L, C]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)          # [B, L + k - 1, C]
        x = self.avg(x.permute(0, 2, 1))               # [B, C, L]
        x = x.permute(0, 2, 1)                         # [B, L, C]
        return x


class series_decomp(nn.Module):
    # Splits x into residual and trend, both [B, L, C].
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class TrendResidualDecomp(nn.Module):
    # DLinear moving-average split. Length is unchanged, so the inverse is a sum.
    n_bands = 2
    length_div = 1

    def __init__(self, kernel_size):
        super(TrendResidualDecomp, self).__init__()
        self.decomp = series_decomp(kernel_size)

    def forward(self, x):
        # x: [B, C, L]
        res, trend = self.decomp(x.permute(0, 2, 1))           # [B, L, C] each
        return torch.cat([res, trend], dim=2).permute(0, 2, 1)  # [B, 2C, L]

    def inverse(self, bands):
        # bands: [B, 2C, H]
        res, trend = bands.chunk(2, dim=1)                     # [B, C, H] each
        return res + trend                                     # [B, C, H]


class NoDecomp(nn.Module):
    n_bands = 1
    length_div = 1

    def forward(self, x):
        return x

    def inverse(self, bands):
        return bands
