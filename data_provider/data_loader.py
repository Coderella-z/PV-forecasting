import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

# 根据原始进行修改
class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='MS', data_path='station011-s.csv', data_path_NWP='station011_nwp_1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.data_path_NWP = data_path_NWP
        self.__read_data__()
        
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.scaler_NWP = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        df_raw_NWP = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path_NWP))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        cols_NWP = list(df_raw_NWP.columns)
        cols_NWP.remove(self.target)
        cols_NWP.remove('date')
        df_raw_NWP = df_raw_NWP[['date'] + cols_NWP + [self.target]]

        num_train = int(len(df_raw) * 0.7)
        # print("num_train",num_train)
        num_test = int(len(df_raw) * 0.2)
        # print("num_test",num_test)
        num_vali = len(df_raw) - num_train - num_test
        # print("num_vali",num_vali)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        # print("border1s",border1s)
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        # print("border2s",border2s)
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            df_data_NWP = df_raw_NWP[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values[:, :])
            self.scaler_NWP.fit(df_data_NWP.values[:, :])
            data1 = self.scaler.transform(df_data.values[:, :])
            data_NWP1 = self.scaler_NWP.transform(df_data_NWP.values[:, :])
            # print("Transformed data1 shape:", data1.shape)
            data = data1
            # print("Concatenated data shape:", data.shape)
            data_NWP = data_NWP1
        else:
            data = df_data.values
            data_NWP = df_data_NWP.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        df_stamp_NWP = df_raw_NWP[['date']][border1:border2]
        df_stamp_NWP['date'] = pd.to_datetime(df_stamp_NWP.date)
        if self.timeenc == 0:
            df_stamp_NWP['month'] = df_stamp_NWP.date.apply(lambda row: row.month, 1)
            df_stamp_NWP['day'] = df_stamp_NWP.date.apply(lambda row: row.day, 1)
            df_stamp_NWP['weekday'] = df_stamp_NWP.date.apply(lambda row: row.weekday(), 1)
            df_stamp_NWP['hour'] = df_stamp_NWP.date.apply(lambda row: row.hour, 1)
            data_stamp_NWP = df_stamp_NWP.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp_NWP = time_features(pd.to_datetime(df_stamp_NWP['date'].values), freq=self.freq)
            data_stamp_NWP = data_stamp_NWP.transpose(1, 0)

        self.data_x_NWP = data_NWP[border1:border2]
        self.data_y_NWP = data_NWP[border1:border2]
        self.data_stamp_NWP = data_stamp

        # print("data_x shape:", self.data_x.shape)
        # print("data_y shape:", self.data_y.shape)
        # print("data_stamp shape:", self.data_stamp.shape)

    def __getitem__(self, index):
        s_begin=index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        s_begin2 = s_begin + self.seq_len
        s_end2 = s_begin2 +  self.pred_len
        seq_x_NWP = self.data_x_NWP[s_begin2:s_end2]
        seq_y_NWP = self.data_y_NWP[s_begin2:s_end2]
        seq_x_NWP_mark = self.data_stamp_NWP[s_begin2:s_end2]
        seq_y_NWP_mark = self.data_stamp_NWP[s_begin2:s_end2]
        return seq_x, seq_y, seq_x_mark, seq_y_mark,seq_y_NWP

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def inverse_transform_NWP(self, data_NWP):
        return self.scaler_NWP.inverse_transform(data_NWP)
