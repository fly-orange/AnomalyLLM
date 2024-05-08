import os
import numpy as np
import torch
import torch.nn.functional as f
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from .augmentations import DataTransform


class TimeSeriesWithAnomalies2(Dataset):
    def __init__(self, data_dir, split, configs, **kwargs): 
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self._load_data(data_dir, split, configs)

    def _load_data(self, data_dir, split, configs):
        if split == 'test':
            datafile = os.path.join(data_dir, 'test_data.npy')
            labelfile = os.path.join(data_dir, 'test_label.npy')
        else:
            datafile = os.path.join(data_dir, 'train_data.npy')
            labelfile = os.path.join(data_dir, 'train_label.npy')

        data = np.load(datafile)  # (B, T, D)
        wlabel = np.load(labelfile)   # (B)
        wlabel[wlabel==2]=1

        if split=='train':
            self.data = data[: int(0.8*len(data))]
            self.wlabel = wlabel[: int(0.8*len(data))]
        elif split=='valid':
            self.data = data[int(0.8*len(data)):]
            self.wlabel = wlabel[int(0.8*len(data)):]
        else:
            self.data = data
            self.wlabel = wlabel


        # if hasattr(configs, 'augmentation'):
        self.aug1, self.aug2 = DataTransform(self.data, configs)
        self.aug1 = torch.Tensor(self.aug1)
        self.aug2 = torch.Tensor(self.aug2)
        # print(self.aug1.dtype, self.data.dtype)

        self.data = self.data.transpose(0,2,1)
        self.aug1 = self.aug1.permute(0,2,1)
        self.aug2 = self.aug2.permute(0,2,1)
        print(self.data.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
    
        return self.data[idx], self.wlabel[idx], self.aug1[idx], self.aug2[idx]
         


class NormalTimeSeries(Dataset):
    def __init__(self, data_dir, split_size, split, **kwargs): 
        super().__init__()
        self.data_dir = data_dir
        self.split_size = split_size
        self._load_data(data_dir, split)

    def _load_data(self, data_dir, split):
        
        data, dlabel, wlabel = [], [], []
        filenames = os.listdir(os.path.join(data_dir, split))
        for filename in filenames:
            filepath = os.path.join(data_dir, split, filename)
            data_, label_, (length, input_size) = np.load(filepath, allow_pickle=True)
            data_, dlabel_, wlabel_ = self._preprocess(data_, label_, self.split_size)
            
            data.append(data_)  # (B, T, D)
            dlabel.append(dlabel_)
            wlabel.append(wlabel_)

        self.input_size = input_size
        self.data = torch.cat(data, dim=0)
        self.dlabel = torch.cat(dlabel, dim=0)
        self.wlabel = torch.cat(wlabel, dim=0)

    def _preprocess(self, data, label, split_size):

        # normalize
        scaler = StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)

        # split
        data = torch.Tensor(data) # (T,D)
        data = f.pad(data, (0, 0, split_size - data.shape[0] % split_size, 0), 'constant', 0)  #扩充成split size的倍数
        data = torch.unsqueeze(data, dim=0) # (1, T, D)
        data = torch.cat(torch.split(data, split_size, dim=1), dim=0) # (B, T, D)

        label = torch.Tensor(label) # (T, )
        label = f.pad(label, (split_size - label.shape[0] % split_size, 0), 'constant', 0)
        label = torch.unsqueeze(label, dim=0)
        label = torch.cat(torch.split(label, split_size, dim=1), dim=0) # (B, T)
        
        dlabel = label
        wlabel = torch.max(label, dim=1)[0]  # 一个序列存在异常则整体为异常

        data = data[wlabel==0]
        dlabel = dlabel[wlabel==0]
        wlabel = wlabel[wlabel==0]
        
        return data, dlabel, wlabel

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'data': self.data[idx],
            'dlabel': self.dlabel[idx],
            'wlabel': self.wlabel[idx]
        } 


class AbnormalTimeSeries(Dataset):
    def __init__(self, data_dir, split_size, split, **kwargs): 
        super().__init__()
        self.data_dir = data_dir
        self.split_size = split_size
        self._load_data(data_dir, split)

    def _load_data(self, data_dir, split):
        
        data, dlabel, wlabel = [], [], []
        filenames = os.listdir(os.path.join(data_dir, split))
        for filename in filenames:
            filepath = os.path.join(data_dir, split, filename)
            data_, label_, (length, input_size) = np.load(filepath, allow_pickle=True)
            data_, dlabel_, wlabel_ = self._preprocess(data_, label_, self.split_size)
            
            data.append(data_)  # (B, T, D)
            dlabel.append(dlabel_)
            wlabel.append(wlabel_)

        self.input_size = input_size
        self.data = torch.cat(data, dim=0)
        self.dlabel = torch.cat(dlabel, dim=0)
        self.wlabel = torch.cat(wlabel, dim=0)

    def _preprocess(self, data, label, split_size):

        # normalize
        scaler = StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)

        # split
        data = torch.Tensor(data) # (T,D)
        data = f.pad(data, (0, 0, split_size - data.shape[0] % split_size, 0), 'constant', 0)  #扩充成split size的倍数
        data = torch.unsqueeze(data, dim=0) # (1, T, D)
        data = torch.cat(torch.split(data, split_size, dim=1), dim=0) # (B, T, D)

        label = torch.Tensor(label) # (T, )
        label = f.pad(label, (split_size - label.shape[0] % split_size, 0), 'constant', 0)
        label = torch.unsqueeze(label, dim=0)
        label = torch.cat(torch.split(label, split_size, dim=1), dim=0) # (B, T)
        
        dlabel = label
        wlabel = torch.max(label, dim=1)[0]  # 一个序列存在异常则整体为异常

        data = data[wlabel==1]
        dlabel = dlabel[wlabel==1]
        wlabel = wlabel[wlabel==1]
        
        return data, dlabel, wlabel

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'data': self.data[idx],
            'dlabel': self.dlabel[idx],
            'wlabel': self.wlabel[idx]
        } 