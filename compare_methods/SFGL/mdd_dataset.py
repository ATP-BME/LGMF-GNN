from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
import torch
from torch import tensor,float32
import pandas as pd
from random import shuffle
import h5py
import numpy as np
from sklearn import preprocessing

class Data_MDD(Dataset):
    def __init__(self, data_path, csv_path, k_fold=None):
        self.data_path = data_path
        self.csv_path = csv_path
        self.timeseries_dict = {}
        self.people_dict = {}
        self.f = pd.read_csv(self.csv_path)
        data = h5py.File(self.data_path,'r')
        bold_timeseries = torch.from_numpy(data['timeseries'][:].transpose(2,0,1))
        min_max = preprocessing.StandardScaler()
        tt = []
        for x in bold_timeseries:
            x = x.permute(1, 0)
            x = min_max.fit_transform(x)
            x = x.T
            tt.append(torch.from_numpy(x))
        bold_timeseries = torch.stack(tt, dim=0)
        people = self.f.to_numpy()[:, 2:].astype(np.float32)
        people = torch.from_numpy(preprocessing.StandardScaler().fit_transform(people))
        for id in range(bold_timeseries.shape[0]):
            self.timeseries_dict[id] = bold_timeseries[id,:,:]
            self.people_dict[id] = people[id,:]

        self.num_timepoints, self.num_nodes = list(self.timeseries_dict.values())[0].shape
        self.full_subject_list = list(self.timeseries_dict.keys())
        if k_fold is None:
            self.subject_list = self.full_subject_list
        else:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0) if k_fold is not None else None
        self.k = None

        Label = self.f['LABEL']
        self.num_classes = len(Label.unique())
        self.behavioral_dict = Label.to_dict()
        self.full_label_list = [self.behavioral_dict[int(subject)] for subject in self.full_subject_list]

    def __len__(self):
        return len(self.subject_list) if self.k is not None else len(self.full_subject_list)

    def set_fold(self, fold, train=True):
        assert self.k_fold is not None
        self.k = fold
        train_idx, test_idx = list(self.k_fold.split(self.full_subject_list, self.full_label_list))[fold]
        if train: shuffle(train_idx)
        self.subject_list = [self.full_subject_list[idx] for idx in train_idx] if train else [self.full_subject_list[idx] for idx in test_idx]

    def __getitem__(self, idx):
        subject = self.subject_list[idx]
        timeseries = self.timeseries_dict[subject]
        people = self.people_dict[subject]
        label = self.behavioral_dict[int(subject)]

        if label==0:
            label = tensor(0)
        elif label==1:
            label = tensor(1)
        else:
            raise

        return {'id': subject, 'timeseries': tensor(timeseries, dtype=float32), 'label': label, 'people':people}
