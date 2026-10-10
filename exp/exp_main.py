from contextlib import nullcontext
import shutil
import os
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from data_provider.data_factory import TimeSeriesDataset
from models import Chronos2, DLinear, FITS, FreqMoE, MeanMedian, Naive, PatchTST, iTransformer, FrNet, Vesa
from utils.learning_rates import adjust_learning_rate
from utils.tools import EarlyStopping
from utils.metrics import metric


class Exp_Main:
    # Standalone training and testing of one model.
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()

        if self.args.d_is_training and not self.args.d_pretrained:
            self.train_loader = self._get_data(flag='train')
            self.vali_loader = self._get_data(flag='val')
            self.test_loader = self._get_data(flag='test')
        else:
            self.test_loader = self._get_data(flag='test')

        self.model = self._build_model().to(self.device)
        


    def _acquire_device(self):
        if self.args.d_use_gpu:
            visible_devices = (
                str(self.args.d_gpu)
                if not self.args.d_use_multi_gpu
                else ",".join(str(device_id) for device_id in self.args.d_devices)
            )
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
            device = torch.device('cuda:{}'.format(self.args.d_gpu))
            print('Use GPU: cuda:{}'.format(self.args.d_gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        # Add new models here.
        model_dict = {
            'Chronos2': Chronos2,
            'DLinear': DLinear,
            'MeanMedian': MeanMedian,
            'Naive': Naive,
            'PatchTST': PatchTST,
            'iTransformer': iTransformer,
            'FrNet': FrNet,
            'FITS': FITS,
            'FreqMoE': FreqMoE,
            'Vesa': Vesa,
        }
        
        model = model_dict[self.args.d_model].Model(self.args).float()

        if self.args.d_use_multi_gpu and self.args.d_use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.d_devices)
        
        return model

    def _get_data(self, flag):
        timeenc = 0 if self.args.d_embed != 'timeF' else 1

        # Test uses batch size 1.
        flag_list = {
            "train": {"shuffle": True,
                    "drop_last": True, 
                    "stride": self.args.d_train_stride, 
                    "batch_size": self.args.d_batch_size},

            "val": {"shuffle": False, 
                    "drop_last": False, 
                    "stride": self.args.d_test_stride, 
                    "batch_size": self.args.d_batch_size},

            "test": {"shuffle": False, 
                     "drop_last": False, 
                     "stride": self.args.d_test_stride,
                     "batch_size": 1},
                }

        dataset = TimeSeriesDataset(
            args = self.args,
            timeenc=timeenc,
            flag=flag,
            stride=flag_list[flag]["stride"]
        )

        print(f"{flag} Dataset: ({len(dataset)}, {self.args.d_seq_len}, {self.args.d_in_features})")

        loader = DataLoader(
            dataset,
            batch_size=flag_list[flag]["batch_size"],
            shuffle=flag_list[flag]["shuffle"],
            num_workers=self.args.d_num_workers,
            drop_last=flag_list[flag]["drop_last"],
            pin_memory=True
        )

        return loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.d_learning_rate)
        
        return model_optim

    def _select_criterion(self):
        losses_dict = {
            'mae': nn.L1Loss(),
            'mse': nn.MSELoss(),
            'smooth': nn.SmoothL1Loss(),
        }
        criterion = losses_dict.get(self.args.d_loss, nn.MSELoss())
        
        return criterion


    def _get_inputs(self, x, y, x_mark, y_mark):
        batch_x = x.float().to(self.device)
        batch_y = y.float().to(self.device)
        
        # No time features for PEMS and Solar.
        if 'PEMS' in self.args.d_data or 'Solar' in self.args.d_data:
            batch_x_mark = None
            batch_y_mark = None
        else:
            batch_x_mark = x_mark.float().to(self.device)
            batch_y_mark = y_mark.float().to(self.device)

        # Decoder input: label_len known steps followed by pred_len zeros.
        dec_inp = torch.zeros_like(y[:, -self.args.d_pred_len:, :]).float()
        dec_inp = torch.cat([y[:, :self.args.d_label_len, :], dec_inp], dim=1).float().to(self.device)

        return batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp

            
    
    def vali(self, vali_loader, criterion):
        total_loss = []
        
        self.model.eval()
        
        with torch.no_grad():
            for _, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                
                batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = self._get_inputs(batch_x, batch_y, batch_x_mark, batch_y_mark)

                amp_context = torch.cuda.amp.autocast() if self.args.d_use_amp else nullcontext()

                input_type = self.args.d_model_input_type.lower()

                if input_type == "x_only":
                    model_args = (batch_x,)
                elif input_type == "x_mark_incl":
                    model_args = (batch_x, batch_x_mark)
                else:
                    model_args = (batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                with amp_context:
                    outputs = self.model(*model_args)
                
                if input_type not in ["x_only", "x_mark_incl"] and self.args.d_output_attention:
                    outputs = outputs[0]

                f_dim = -1 if self.args.d_forecast_type == 'MS' else 0
                outputs = outputs[:, -self.args.d_pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.d_pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)

        total_loss = np.average(total_loss)

        self.model.train()

        return total_loss

    def train(self, setting):
        path = os.path.join(self.args.d_checkpoint_path, self.args.d_setting)

        time_now = time.time()

        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args.d_patience, verbose=True)

        # Create auxiliary-loss parameters before building the optimizer.
        if self.args.d_auxillary_loss and hasattr(self.model, "_ensure_aux_loss_params"):
            self.model._ensure_aux_loss_params()

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.d_use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.d_pct_start,
                                            epochs=self.args.d_train_epochs,
                                            max_lr=self.args.d_learning_rate)

        self.model.train()
        for epoch in range(self.args.d_train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.train_loader):
                iter_count += 1

                model_optim.zero_grad(set_to_none=True)

                batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = self._get_inputs(batch_x, batch_y, batch_x_mark, batch_y_mark)

                amp_context = torch.cuda.amp.autocast() if self.args.d_use_amp else nullcontext()

                input_type = self.args.d_model_input_type.lower()

                if input_type == "x_only":
                    model_args = (batch_x,)
                elif input_type == "x_mark_incl":
                    model_args = (batch_x, batch_x_mark)
                else:
                    model_args = (batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                with amp_context:
                    outputs = self.model(*model_args)
                
                if input_type not in ["x_only", "x_mark_incl"] and self.args.d_output_attention:
                    outputs = outputs[0]

                f_dim = -1 if self.args.d_forecast_type == 'MS' else 0
                outputs = outputs[:, -self.args.d_pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.d_pred_len:, f_dim:].to(self.device)

                if self.args.d_auxillary_loss:
                    if not hasattr(self.model, 'aux_loss'):
                        raise AttributeError(
                            f"d_auxillary_loss is enabled but model '{self.args.d_model}' has no aux_loss method."
                        )
                    loss = self.model.aux_loss(outputs, batch_y)
                else:
                    loss = criterion(outputs, batch_y)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.d_train_epochs - epoch) * train_steps - i)
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.d_use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                # 'TST' steps the scheduler every batch.
                if self.args.d_lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(self.vali_loader, criterion)
            test_loss = self.vali(self.test_loader, criterion)
            self.model.train()

            print("\nEpoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            # Saves the best checkpoint on validation loss.
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.d_lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location="cpu"))


        return self.model

    @torch.no_grad()
    def test(self, setting):
        if not self.args.d_is_training and not self.args.d_pretrained:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.d_checkpoint_path, setting, 'checkpoint.pth'), map_location="cpu"))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = self._get_inputs(batch_x, batch_y, batch_x_mark, batch_y_mark)

                input_type = self.args.d_model_input_type.lower()

                if input_type == "x_only":
                    model_args = (batch_x,)
                elif input_type == "x_mark_incl":
                    model_args = (batch_x, batch_x_mark)
                else:
                    model_args = (batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                outputs = self.model(*model_args)

                if input_type not in ["x_only", "x_mark_incl"] and self.args.d_output_attention:
                    outputs = outputs[0]


                f_dim = -1 if self.args.d_forecast_type == 'MS' else 0
                outputs = outputs[:, -self.args.d_pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.d_pred_len:, f_dim:].to(self.device)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
            
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)


        total_test_len = self.test_loader.dataset._data_len()
        valid_pred_len = total_test_len - self.args.d_seq_len

        num_windows = preds.shape[0]
        pred_len = preds.shape[1]
        
        # Stitch windows into one series of length len(test) - seq_len.
        flat_preds = preds[:-1].reshape(-1, preds.shape[-1])
        flat_trues = trues[:-1].reshape(-1, trues.shape[-1])
        
        current_len = flat_preds.shape[0]
        remaining = valid_pred_len - current_len

        if remaining > 0:
            flat_preds = np.concatenate([flat_preds, preds[-1][-remaining:]], axis=0)
            flat_trues = np.concatenate([flat_trues, trues[-1][-remaining:]], axis=0)
        elif remaining < 0:
            flat_preds = flat_preds[:valid_pred_len]
            flat_trues = flat_trues[:valid_pred_len]


        mae, mse, rmse, mape, mspe, rse, corr, r2 = metric(flat_preds, flat_trues)


        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        

        with open(f"Result_{self.args.d_model}.csv", 'a') as f:
            f.write(f"{setting}-mse:{mse}-mae:{mae}\n")


        return flat_preds, flat_trues
