import os
import time
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import torch

from fvcore.nn import FlopCountAnalysis

from models import DLinear, PatchTST, iTransformer, FrNet, FITS, FreqMoE, Vesa, Chronos2
from utils.timefeatures import time_features


class Exp_Params:
    '''
    Forward-pass-only experiment used to profile a model's size and compute
    cost for a given configuration. Unlike Exp_Main, it never loads a real
    dataset -- a single dummy batch is built directly from the config and
    pushed through the model once to measure it.
    '''

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
            'PatchTST': PatchTST,
            'iTransformer': iTransformer,
            'FrNet': FrNet,
            'FITS': FITS,
            'FreqMoE': FreqMoE,
            'Vesa': Vesa,
            'Chronos2': Chronos2,
        }

        return model_dict[self.args.d_model].Model(self.args).float()

    def _dummy_mark(self, length):
        '''
        Builds a dummy time-feature tensor with the same last dimension
        TimeSeriesDataset would produce for the configured d_freq/d_embed,
        so models expecting x_mark inputs see correctly shaped tensors.
        '''
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
        batch_x = torch.randn(args.d_batch_size, args.d_seq_len, args.d_in_features, device=self.device)

        input_type = args.d_model_input_type.lower()

        if input_type == "x_only":
            return (batch_x,)

        if input_type == "x_mark_incl":
            batch_x_mark = self._dummy_mark(args.d_seq_len).repeat(args.d_batch_size, 1, 1)
            return (batch_x, batch_x_mark)

        # Encoder-decoder style models (x_mark_dec_incld), mirroring Exp_Main._get_inputs.
        batch_x_mark = self._dummy_mark(args.d_seq_len).repeat(args.d_batch_size, 1, 1)
        dec_len = args.d_label_len + args.d_pred_len
        batch_y_mark = self._dummy_mark(dec_len).repeat(args.d_batch_size, 1, 1)
        dec_inp = torch.zeros(args.d_batch_size, dec_len, args.d_in_features, device=self.device)

        return (batch_x, batch_x_mark, dec_inp, batch_y_mark)

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def profile(self):
        '''
        Runs a single forward pass and returns size/compute/latency stats
        for the model under the current configuration: total/trainable
        parameter counts, the subset of parameters actually exercised by
        this forward pass ("active" -- differs from total when a model owns
        branches a given config never calls, e.g. an unused decomposition
        head), MACs / FLOPs, the model's footprint in MB, and wall-clock
        inference latency for one forward pass at the configured batch size.
        '''
        self.model.eval()
        inputs = self._build_inputs()

        # One untimed forward pass to warm up (CUDA kernel/cuDNN autotuning,
        # lazy weight materialization, etc.) so the timed pass below isn't
        # skewed by one-time setup cost.
        self.model(*inputs)
        self._sync()

        # Forward hooks tell us which submodules actually ran for this
        # config, so params owned by dead/unused branches can be excluded.
        # This pass also doubles as the timed pass, so an expensive model
        # (e.g. Chronos2's full pipeline) isn't run more times than needed.
        fired = set()
        handles = [
            m.register_forward_hook(lambda mod, inp, out, _m=m: fired.add(id(_m)))
            for m in self.model.modules()
        ]
        try:
            start = time.perf_counter()
            self.model(*inputs)
            self._sync()
            latency_s = time.perf_counter() - start
        finally:
            for handle in handles:
                handle.remove()

        # A Parameter/buffer shared across submodules (e.g. Vesa's shared_pa/
        # shared_pb, re-registered on every VesaLinear that holds a
        # reference) must only be counted once -- dedupe by tensor identity
        # rather than by owning module.
        total_params = 0
        trainable_params = 0
        size_bytes = 0
        seen_param_ids = set()
        seen_buffer_ids = set()
        active_param_ids = set()
        params_by_id = {}
        for m in self.model.modules():
            for p in m.parameters(recurse=False):
                pid = id(p)
                if pid not in seen_param_ids:
                    seen_param_ids.add(pid)
                    params_by_id[pid] = p
                    total_params += p.numel()
                    size_bytes += p.numel() * p.element_size()
                    if p.requires_grad:
                        trainable_params += p.numel()
                if id(m) in fired:
                    active_param_ids.add(pid)
            # Buffers (e.g. fixed rotation matrices, running stats) hold no
            # gradient and aren't parameters, but they're live tensors on
            # device and belong in the model's memory footprint.
            for b in m.buffers(recurse=False):
                bid = id(b)
                if bid not in seen_buffer_ids:
                    seen_buffer_ids.add(bid)
                    size_bytes += b.numel() * b.element_size()

        active_params = sum(params_by_id[pid].numel() for pid in active_param_ids)

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
            "latency_ms": latency_s * 1000,
        }
