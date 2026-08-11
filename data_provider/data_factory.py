import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import joblib
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesDataset(Dataset):

    def __init__(
        self,
        args,
        flag,
        stride,
        timeenc=0,
        split_ratio=(0.7, 0.1, 0.2),
    ):

        
        self.seq_len = args.d_seq_len
        self.label_len = args.d_label_len
        self.pred_len = args.d_pred_len
        self.stride = stride
        self.is_training = args.d_is_training

        self.forecast_type = args.d_forecast_type
        self.target = args.d_target
        self.timeenc = timeenc
        self.freq = args.d_freq

        self.root_path = args.d_root_path
        self.data_path = args.d_data_path
        self.checkpoints_path = os.path.join(args.d_checkpoint_path, args.d_setting)

        assert flag in ['train', 'val', 'test']
        self.set_type = {'train':0,'val':1,'test':2}[flag]

        self.train_ratio, self.val_ratio, self.test_ratio = split_ratio

        self.__read_data__()



    def __read_data__(self):

        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        cols = list(df_raw.columns)

        if isinstance(self.target, int):
            target_col = cols[self.target]
        else:
            assert self.target in cols
            target_col = self.target

        cols.remove(target_col)
        cols.remove('date')

        df_raw = df_raw[['date'] + cols + [target_col]]

        total_len = len(df_raw)

        num_train = int(total_len * self.train_ratio)
        num_val = int(total_len * self.val_ratio)

        border1s = [
            0,
            num_train - self.seq_len,
            num_train + num_val - self.seq_len
        ]

        border2s = [
            num_train,
            num_train + num_val,
            total_len
        ]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.forecast_type in ['M','MS']:
            df_data = df_raw[df_raw.columns[1:]]
        else:
            df_data = df_raw[[target_col]]


        if self.is_training:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            joblib.dump(self.scaler, os.path.join(self.checkpoints_path, "scaler.pkl"))

        else:                    
            self.scaler = joblib.load(os.path.join(self.checkpoints_path, "scaler.pkl"))

        data = self.scaler.transform(df_data.values)


        df_stamp = df_raw[['date']][border1:border2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])

        if self.timeenc == 0:

            df_stamp['month'] = df_stamp.date.dt.month
            df_stamp['day'] = df_stamp.date.dt.day
            df_stamp['weekday'] = df_stamp.date.dt.weekday
            df_stamp['hour'] = df_stamp.date.dt.hour

            if self.freq in ['t','min','minute','s']:
                df_stamp['minute'] = df_stamp.date.dt.minute

            if self.freq == 's':
                df_stamp['second'] = df_stamp.date.dt.second

            data_stamp = df_stamp.drop(columns=['date']).values

        else:

            data_stamp = time_features(
                pd.to_datetime(df_stamp['date'].values),
                freq=self.freq
            )
            data_stamp = data_stamp.transpose(1,0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        last_start = max(0, len(self.data_x) - self.seq_len - self.pred_len)

        if last_start < 0:
            starts = np.array([0])
        else:
            starts = np.arange(0, last_start + 1, self.stride)

        # ensure final window reaches dataset end
        if starts[-1] != last_start:
            starts = np.append(starts, last_start)       

        s_begin = starts
        s_end = s_begin + self.seq_len

        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        self.samples = np.stack([s_begin, s_end, r_begin, r_end], axis=1)

    def __getitem__(self, idx):

        s_begin, s_end, r_begin, r_end = self.samples[idx]

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.samples)
    
    def _data_len(self):
        return len(self.data_y)