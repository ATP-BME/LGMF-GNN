import data.ABIDEParser as Reader
import numpy as np
import torch
from utils.gcn_utils import preprocess_features
from sklearn.model_selection import StratifiedKFold


class dataloader():
    def __init__(self):
        self.pd_dict = {}
        self.node_ftr_dim = 2000
        self.num_classes = 2

    def load_data(self, connectivity='correlation', atlas1='ho',atlas2='aal'):
        ''' load multimodal data from ABIDE
        return: imaging features (raw), labels, non-image data
        '''
        subject_IDs = Reader.get_ids()
        labels = Reader.get_subject_score(subject_IDs, score='DX_GROUP')
        num_nodes = len(subject_IDs)

        sites = Reader.get_subject_score(subject_IDs, score='SITE_ID')
        years = Reader.get_subject_score(subject_IDs, score='Education (years)')
        unique = np.unique(list(sites.values())).tolist()  # 该函数是去除数组中的重复数字，并进行排序之后输出
        ages = Reader.get_subject_score(subject_IDs, score='Age')
        genders = Reader.get_subject_score(subject_IDs, score='Sex')

        y_onehot = np.zeros([num_nodes, self.num_classes])
        y = np.zeros([num_nodes])
        site = np.zeros([num_nodes], dtype=np.int)
        year = np.zeros([num_nodes], dtype=np.float32)
        age = np.zeros([num_nodes], dtype=np.float32)
        gender = np.zeros([num_nodes], dtype=np.int)
        for i in range(num_nodes):
            y_onehot[i, int(labels[subject_IDs[i]]) - 1] = 1
            y[i] = int(labels[subject_IDs[i]])
            site[i] = unique.index(sites[subject_IDs[i]])
            age[i] = float(ages[subject_IDs[i]])
            gender[i] = genders[subject_IDs[i]]
            year[i] = float(years[subject_IDs[i]])
        self.y = y - 1

        self.raw_features1 = Reader.get_networks(subject_IDs, kind=connectivity, atlas_name=atlas1)
        self.raw_features2 = Reader.get_networks2(subject_IDs, kind=connectivity, atlas_name=atlas2)

        phonetic_data = np.zeros([num_nodes, 3], dtype=np.float32)
        phonetic_data[:, 0] = year
        # phonetic_data[:, 0] = site
        phonetic_data[:, 1] = gender
        phonetic_data[:, 2] = age
        # phonetic_data[:, 3] = site

        # self.pd_dict['SITE_ID'] = np.copy(phonetic_data[:,0])
        self.pd_dict['YEAR'] = np.copy(phonetic_data[:, 0])
        self.pd_dict['Sex'] = np.copy(phonetic_data[:, 1])
        self.pd_dict['Age'] = np.copy(phonetic_data[:, 2])

        # feature_matrix, label: (0 or -1), phonetic_data.shape = (num_nodes, num_phonetic_dim)
        return self.raw_features1,self.raw_features2, self.y, phonetic_data

    def data_split(self, n_folds):
        # split data by k-fold CV
        skf = StratifiedKFold(n_splits=n_folds,shuffle=True)
        cv_splits = list(skf.split(self.raw_features1, self.y))
        # cv_splits2 = list(skf.split(self.raw_features2, self.y))
        return cv_splits

    def get_node_features(self, train_ind):
        '''preprocess node features for ev-gcn
        '''
        # self.node_ftr_dim: 要选择多少个特征
        node_ftr1 = Reader.feature_selection(self.raw_features1, self.y, train_ind, self.node_ftr_dim)
        node_ftr2 = Reader.feature_selection(self.raw_features2, self.y, train_ind, self.node_ftr_dim)
        self.node_ftr1 = preprocess_features(node_ftr1)  # D^-1 dot node_ftr
        self.node_ftr2 = preprocess_features(node_ftr2)  # D^-1 dot node_ftr
        return self.node_ftr1,self.node_ftr2

    def get_PAE_inputs(self, nonimg):
        # nonimg = num_node x phonetic_dim
        '''get PAE inputs for ev-gcn
        '''
        # construct edge network inputs
        n = self.node_ftr1.shape[0]
        num_edge = n * (1 + n) // 2 - n  # 上三角阵的元素个数（减去对角线的）
        pd_ftr_dim = nonimg.shape[1]
        edge_index = np.zeros([2, num_edge], dtype=np.int64)
        edgenet_input = np.zeros([num_edge, 2 * pd_ftr_dim], dtype=np.float32)
        aff_score = np.zeros(num_edge, dtype=np.float32)
        # static affinity score used to pre-prune edges
        aff_adj = Reader.get_static_affinity_adj(self.node_ftr1, self.pd_dict)
        flatten_ind = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_index[:, flatten_ind] = [i, j]
                edgenet_input[flatten_ind] = np.concatenate((nonimg[i], nonimg[j]))
                aff_score[flatten_ind] = aff_adj[i][j]
                flatten_ind += 1

        assert flatten_ind == num_edge, "Error in computing edge input"

        keep_ind = np.where(aff_score > 1.1)[0]
        edge_index = edge_index[:, keep_ind]
        edgenet_input = edgenet_input[keep_ind]

        return edge_index, edgenet_input


if __name__ == "__main__":
    site = np.zeros([4], dtype=np.int)
    print(site)
    print(site.shape)
