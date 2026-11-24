import os
import time
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import torch

from fvcore.nn import FlopCountAnalysis
from torch.overrides import TorchFunctionMode

from models import DLinear, DLinear_ablate, PatchTST, iTransformer, FrNet, FITS, FreqMoE, SparseTSF, CycleNet, TimeEmb, Vesa, Vesa_ablate, Chronos2
from utils.timefeatures import time_features


class _ParamUseTracker(TorchFunctionMode):
    # Records every nn.Parameter passed to a torch op during forward, so a
    # parameter read by a module other than its owner (e.g. a shared basis
    # sliced inside another layer) still counts as active.
    def __init__(self):
        super().__init__()
        self.used = set()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        for a in list(args) + list(kwargs.values()):
            if isinstance(a, torch.nn.Parameter):
                self.used.add(id(a))
            elif isinstance(a, (list, tuple)):
                self.used.update(id(t) for t in a if isinstance(t, torch.nn.Parameter))
        return func(*args, **kwargs)


class Exp_Params:
    # Profiles params, FLOPs, size and latency on one random batch.

    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        if self.args.d_use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.d_gpu)
            return torch.device('cuda:{}'.format(self.args.d_gpu))
        return torch.device('cpu')

    def _build_model(self):
        model_dict = {
            'DLinear': DLinear,
            'DLinear_ablate': DLinear_ablate,
            'PatchTST': PatchTST,
            'iTransformer': iTransformer,
            'FrNet': FrNet,
            'FITS': FITS,
            'FreqMoE': FreqMoE,
            'SparseTSF': SparseTSF,
            'CycleNet': CycleNet,
            'TimeEmb': TimeEmb,
            'Vesa': Vesa,
            'Vesa_ablate': Vesa_ablate,
            'Chronos2': Chronos2,
        }

        return model_dict[self.args.d_model].Model(self.args).float()

    def _dummy_mark(self, length):
        # Time features shaped like the dataset's, [1, length, M].
        dates = pd.date_range("2020-01-01", periods=length, freq=self.args.d_freq)

        if self.args.d_embed != 'timeF':
            df_stamp = pd.DataFrame({"date": dates})
            df_stamp['month'] = df_stamp.date.dt.month
            df_stamp['day'] = df_stamp.date.dt.day
            df_stamp['weekday'] = df_stamp.date.dt.weekday
            df_stamp['hour'] = df_stamp.date.dt.hour
            if self.args.d_freq in ['t', 'min', 'minute', 's']:
                df_stamp['minute'] = df_stamp.date.dt.minute
            if self.args.d_freq == 's':
                df_stamp['second'] = df_stamp.date.dt.second
            stamp = df_stamp.drop(columns=['date']).values
        else:
            stamp = time_features(dates, freq=self.args.d_freq).transpose(1, 0)

        return torch.from_numpy(stamp).float().unsqueeze(0).to(self.device)

    def _build_inputs(self):
        args = self.args
        batch_x = torch.randn(args.d_batch_size, args.d_seq_len, args.d_in_features, device=self.device)  # [B, L, C]

        input_type = args.d_model_input_type.lower()

        if input_type == "x_only":
            return (batch_x,)

        if input_type == "x_mark_incl":
            batch_x_mark = self._dummy_mark(args.d_seq_len).repeat(args.d_batch_size, 1, 1)
            return (batch_x, batch_x_mark)

        if input_type == "x_index":
            # Integer time indices at the first forecast step, [B, 3] = (position, hour, weekday).
            time_index = torch.stack([
                torch.randint(0, 100000, (args.d_batch_size,)),
                torch.randint(0, 24, (args.d_batch_size,)),
                torch.randint(0, 7, (args.d_batch_size,)),
            ], dim=1).to(self.device)
            return (batch_x, time_index)

        # Encoder-decoder inputs.
        batch_x_mark = self._dummy_mark(args.d_seq_len).repeat(args.d_batch_size, 1, 1)
        dec_len = args.d_label_len + args.d_pred_len
        batch_y_mark = self._dummy_mark(dec_len).repeat(args.d_batch_size, 1, 1)
        dec_inp = torch.zeros(args.d_batch_size, dec_len, args.d_in_features, device=self.device)

        return (batch_x, batch_x_mark, dec_inp, batch_y_mark)

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def profile(self, warmup=10, reps=50):
        '''
        Returns size/compute/latency stats for the model under the current
        configuration:
          - total / trainable parameters (shared tensors counted once);
          - active parameters: those read by a forward pass (differs from
            total when a model owns branches this config never calls);
          - MACs / FLOPs from fvcore's default handles only (FLOPs = 2 * MACs).
            Matmul/conv/norm ops are counted; FFTs and element-wise ops are
            not, and a complex MAC is counted as one MAC;
          - size in MB of parameters + buffers (complex tensors take 8 bytes,
            so their real and imaginary parts are both counted);
          - mean latency of one forward pass at the configured batch size.
        '''
        self.model.eval()
        inputs = self._build_inputs()

        # Active parameters: every nn.Parameter passed to a torch op during
        # one forward pass. This pass also serves as the first warm-up.
        tracker = _ParamUseTracker()
        with tracker:
            self.model(*inputs)

        params = list(self.model.parameters())   # shared tensors appear once
        total_params = sum(p.numel() for p in params)
        trainable_params = sum(p.numel() for p in params if p.requires_grad)
        active_params = sum(p.numel() for p in params if id(p) in tracker.used)
        size_bytes = sum(t.numel() * t.element_size() for t in params + list(self.model.buffers()))

        flops_analysis = FlopCountAnalysis(self.model, inputs)
        flops_analysis.unsupported_ops_warnings(False)
        flops_analysis.uncalled_modules_warnings(False)
        macs = flops_analysis.total()

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "active_params": active_params,
            "macs": macs,
            "flops": macs * 2,
            "size_mb": size_bytes / (1024 ** 2),
            "latency_ms": self._latency_ms(inputs, warmup, reps),
        }

    def _latency_ms(self, inputs, warmup, reps):
        # Mean wall-clock time of one uninstrumented forward pass, after
        # warm-up passes (CUDA kernel selection, allocator, caches).
        for _ in range(warmup):
            self.model(*inputs)
        self._sync()
        start = time.perf_counter()
        for _ in range(reps):
            self.model(*inputs)
        self._sync()
        return (time.perf_counter() - start) * 1000 / reps
