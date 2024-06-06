import torch
from torch_geometric.data import InMemoryDataset,Data
from os.path import join, isfile
from os import listdir
import numpy as np
import os.path as osp
from imports.read_abide_stats_parall import read_data


class ABIDEDataset(InMemoryDataset):
    def __init__(self, root, name,raw_folder='raw_depression_HC_T1', transform=None, pre_transform=None):
        self.root = root
        self.name = name
        self.raw_folder = raw_folder
        # self.raw_dir = raw_folder
        super(ABIDEDataset, self).__init__(root,transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        print('using data:',self.processed_paths[0])

        self.task_type = 'binary classification'
        # self.num_classes = 2
        print('num_classes:',self.num_classes)
        self.eval_metric = 'rocauc'
        self.binary = False

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.raw_folder)
    @property
    def processed_dir(self) -> str:
        
        if 'train' in self.raw_folder:
            print('train dataset')
            # return osp.join(self.root, 'processed_train_corr')
            # return osp.join(self.root, 'processed_corr_same_protocol_train_corr')
            # return osp.join(self.root, 'processed_ALFF_same_protocol_train_corr')
            return osp.join(self.root, 'processed_corr_T1_same_protocol_train_corr')


        if 'test' in self.raw_folder:
            print('test dataset')
            # return osp.join(self.root, 'processed_test_corr')
            # return osp.join(self.root, 'processed_corr_same_protocol_test_corr')
            # return osp.join(self.root, 'processed_ALFF_same_protocol_test_corr')
            return osp.join(self.root, self.raw_folder,'processed_corr_T1_same_protocol_test_corr')

        else:
            return osp.join(self.root, self.raw_folder, 'processed')

        # return osp.join(self.root, 'processed_same_protocol_dp_HC_corr')
        # return osp.join(self.root, 'processed_same_protocol_ASD_HC_corr')
        # return osp.join(self.root,'all_processed_T1_gdc')




    @property
    def raw_file_names(self):
        # data_dir = osp.join(self.root,'raw')
        data_dir = osp.join(self.root,self.raw_folder)
        if 'SRPBS' in self.root:
            sub_list = np.genfromtxt("/data0/lsy/sub_list/same_protocol_balanced_selected_by_error.txt",dtype=str)
            onlyfiles = [osp.join(data_dir,'{}.h5'.format(sub_name)) for sub_name in sub_list]
        elif 'openneuro' in self.root:
            sub_list = np.genfromtxt("/data0/lsy/openneuro/subject_IDs_selected.txt",dtype=str)
            onlyfiles = [osp.join(data_dir,'{}.h5'.format(sub_name.replace('opn','sub'))) for sub_name in sub_list]
        elif 'Anding' in self.root:
            sub_list = np.genfromtxt("/data0/lsy/Anding1/subject_IDs_selected.txt",dtype=str)
            onlyfiles = [osp.join(data_dir,'{}.h5'.format(sub_name)) for sub_name in sub_list]
        elif 'REST' in self.root:
            sub_list = np.genfromtxt("/data0/lsy/REST_meta_MDD/group/subject_IDs_selected.txt",dtype=str)
            onlyfiles = [osp.join(data_dir,'{}.h5'.format(sub_name)) for sub_name in sub_list]
        else:
            onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
        onlyfiles.sort()
        return onlyfiles
    @property
    def processed_file_names(self):
        return  'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        return

    def process(self):
        # Read data into huge `Data` list.
        self.data, self.slices = read_data(self.raw_dir,self.raw_file_names)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))


class MergedDataset(InMemoryDataset):
    def __init__(self, datasets):
        super(MergedDataset, self).__init__()
        self.data, self.slices = self.collate(datasets)
        self.task_type = 'binary classification'

    @staticmethod
    def collate(datasets):
        data_list = []
        for dataset in datasets:
            data_list.extend(dataset)
        return InMemoryDataset.collate(data_list)