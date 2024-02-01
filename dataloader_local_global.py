import data.ABIDEParser as Reader
import numpy as np
import torch
import torch.utils.data as utils
from utils.gcn_utils import preprocess_features
from sklearn.model_selection import StratifiedKFold,KFold
import yaml
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings("ignore")
from opt import *
opt = OptInit().initialize()
with open(opt.config_filename) as f:
    config = yaml.load(f, Loader=yaml.Loader)

class dataloader_lg():
    def __init__(self):
        self.pd_dict = {}
        self.node_ftr_dim = opt.node_feature_dim #2000
        self.num_classes = 2
        self.n_sub = 0

    def load_data(self, subject_IDs=None,connectivity='correlation', atlas1='aal',atlas2='cc200'):
        ''' load multimodal data from ABIDE
        return: imaging features (raw), labels, non-image data
        '''
        if subject_IDs is None:
            subject_IDs = Reader.get_ids()
        # 标签
        labels = Reader.get_subject_score(subject_IDs, score='diag')  # 返回的是字典 诊断结果
        num_nodes = len(subject_IDs) # fot global gragh
        self.n_sub = num_nodes

        sites = Reader.get_subject_score(subject_IDs, score='site') # 扫描地
        unique = np.unique(list(sites.values())).tolist()
        site_num = len(unique)
        ages = Reader.get_subject_score(subject_IDs, score='age')
        genders = Reader.get_subject_score(subject_IDs, score='sex')
        # dsms=Reader.get_subject_score(subject_IDs,score='DSM_IV_TR') # 受教育程度/量表得分
        # protocols=Reader.get_subject_score(subject_IDs,score='protocol') # protocol磁共振采集协议
        # hands=Reader.get_subject_score(subject_IDs,score='hand') # 惯用手
        unique_labels = np.unique(list(labels.values())).tolist()
        unique_labels.sort()
        print('unique labels:',unique_labels)
        print('unique sites:',unique)

        y_onehot = np.zeros([num_nodes, self.num_classes])
        y = np.zeros([num_nodes])
        site = np.zeros([num_nodes], dtype=np.int32)
        age = np.zeros([num_nodes], dtype=np.float32)
        gender = np.zeros([num_nodes], dtype=np.int32)
        # dsm=np.zeros([num_nodes],dtype=np.int)
        protocol = np.zeros([num_nodes],dtype=np.int32)
        hand = np.zeros([num_nodes],dtype=np.int32)

        for i in range(num_nodes):
            y_onehot[i, unique_labels.index(labels[subject_IDs[i]])] = 1  # 0: [1,0]   1: [0,1]
            # y[i] = int(labels[subject_IDs[i]])
            y[i] = unique_labels.index(labels[subject_IDs[i]])

            site[i] = unique.index(sites[subject_IDs[i]])  # 用下标代替扫描地这个特征
            age[i] = float(ages[subject_IDs[i]])
            gender[i] = genders[subject_IDs[i]]
            # dsm[i] = dsms[subject_IDs[i]]
            # protocol[i] = protocols[subject_IDs[i]]
            # hand[i] = hands[subject_IDs[i]]

        # self.y = y - 1
        self.y = y
        self.site = site

        # 看了下源码，get_networks和get_networks2是一样的
        # 得到展平后的邻接矩阵的上三角

        phonetic_data = np.zeros([num_nodes, 3], dtype=np.float32)
        phonetic_data[:, 0] = site
        phonetic_data[:, 1] = gender
        phonetic_data[:, 2] = age
        # phonetic_data[:,3] = dsm
        # phonetic_data[:,3] = protocol
        # phonetic_data[:,4] = hand
        self.pheno = phonetic_data


        self.pd_dict['site'] = np.copy(phonetic_data[:, 0])
        self.pd_dict['sex'] = np.copy(phonetic_data[:, 1])
        self.pd_dict['age'] = np.copy(phonetic_data[:, 2])
        # self.pd_dict['protocol'] = np.copy(phonetic_data[:,3])
        # self.pd_dict['hand'] = np.copy(phonetic_data[:,4])


        # feature_matrix, label: (0 or -1), phonetic_data.shape = (num_nodes, num_phonetic_dim)
        return subject_IDs,self.y, phonetic_data,site_num,site

    def data_split(self, n_folds,train_val_num):
        # split data by k-fold CV
        n_sub = train_val_num # train HC:MDD=416:186 new signal
        id = list(range(n_sub))
        import random
        # random.seed(321)
        # random.shuffle(id)

        kf = KFold(n_splits=n_folds, random_state=321,shuffle = True)
        # kf2 = KFold(n_splits=kfold-1, shuffle=True, random_state = 666)
        train_index = list()
        val_index = list()

        for tr,va in kf.split(np.array(id)):
            val_index.append(va)
            train_index.append(tr)
            

        train_id = train_index
        val_id = val_index

        return train_id,val_id
    
    def data_split_site(self):
        # split data by site
        train_index = list()
        val_index = list()
        train_inds,test_inds = [],[]
        # selected by site
        # site_test = [16, 18, 23, 24] # SRPBS MDD HBN
        # site_test = [0,2,7,8] # SRPBS-HBN
        site_test = [21] # SRPBS-MDD-OPN
        # site_test = [5] # SRPBS-OPN
        for site in site_test:
            test_ind = np.array(np.where(self.pheno[:,0]==site)).squeeze()
            train_ind = [ind for ind in range(self.pheno.shape[0]) if ind not in test_ind]
            train_index.append(train_ind)
            val_index.append(test_ind)

        return [train_index,val_index]

    def get_node_features(self, train_ind):
        '''preprocess node features for ev-gcn
        '''
        # self.node_ftr_dim: 要选择多少个特征
        node_ftr1 = Reader.feature_selection(self.raw_features1, self.y, train_ind, self.node_ftr_dim) # AAL
        node_ftr2 = Reader.feature_selection(self.raw_features2, self.y, train_ind, self.node_ftr_dim) # CC200
        node_ftr3 = Reader.feature_selection(self.raw_features3, self.y, train_ind, self.node_ftr_dim) # T1 + ALFF
        self.node_ftr1 = preprocess_features(node_ftr1)  # D^-1 dot node_ftr
        self.node_ftr2 = preprocess_features(node_ftr2)  # D^-1 dot node_ftr 按行除以均值
        self.node_ftr3 = preprocess_features(node_ftr3)  # D^-1 dot node_ftr 按行除以均值
        return self.node_ftr1,self.node_ftr2,self.node_ftr3

    def get_PAE_inputs(self, nonimg,embeddings):
        # nonimg = num_node x phonetic_dim
        '''
        get PAE inputs for ev-gcn
        PAE用于基于非影像数据计算节点之间边的权重
        nonimg: N sub x 一维非影像特征向量

        return:
            基于表型数据相似程度打分的边连接和边权重矩阵，边剪枝的策略是保留分数>1.1的边
        '''
        # construct edge network inputs
        n = embeddings.shape[0]
        node_ftr = np.array(embeddings.detach().cpu().numpy())
        num_edge = n * (1 + n) // 2 - n  # 上三角阵的元素个数（减去对角线的）计算一共有多少条可能的边
        pd_ftr_dim = nonimg.shape[1] # phenotypic feature dim
        edge_index = np.zeros([2, num_edge], dtype=np.int64)
        edgenet_input = np.zeros([num_edge, 2 * pd_ftr_dim], dtype=np.float32) # 所有可能的边以及边所连接的两个节点的影像特征
        aff_score = np.zeros(num_edge, dtype=np.float32)
        # static affinity score used to pre-prune edges
        aff_adj = Reader.get_static_affinity_adj(node_ftr, self.pd_dict) # 输入是影像特征和表型特征，返回根据表征计算的边权值
        flatten_ind = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_index[:, flatten_ind] = [i, j] # i指向j的边
                edgenet_input[flatten_ind] = np.concatenate((nonimg[i], nonimg[j]))
                aff_score[flatten_ind] = aff_adj[i][j] #i指向j的边的权重
                # print(aff_score[flatten_ind])
                flatten_ind += 1

        assert flatten_ind == num_edge, "Error in computing edge input"

        keep_ind = np.where(aff_score > opt.pheno_edge_threshold)[0] # 保留权重>1.1的边
        # print('pheno edge kept:', len(keep_ind))
        edge_index = edge_index[:, keep_ind]
        edgenet_input = edgenet_input[keep_ind]

        return edge_index, edgenet_input


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def prepare_local_data(timeseries_path,t1_root):
    data = np.load(timeseries_path, allow_pickle=True).item()
    final_fc = data["timeseires"]
    final_pearson = data["corr"]
    labels = data["label"]
    labels[labels == 2] = 1
    sub_names = data['sub_name']
    t1_feature = []
    # load T1 feature
    for sub_name in sub_names:
        white_path = os.path.join(t1_root,'white','{}.npy'.format(sub_name))
        gray_path = os.path.join(t1_root,'gray','{}.npy'.format(sub_name))
        at1_feature = np.concatenate((np.load(white_path),np.load(gray_path)))
        t1_feature.append(np.expand_dims(at1_feature,axis=0))
    t1_feature = np.concatenate(t1_feature,axis=0)

    # select out balanced ID
    # if 'SRPBS' in timeseries_path: sub_txt_path = "/data0/lsy/sub_list/same_protocol_balanced.txt" # SRPBS
    if 'SRPBS' in timeseries_path: sub_txt_path = "D:\data\depression\sub_list\same_protocol_balanced_selected_by_error.txt" # SRPBS
    if 'MDD' in timeseries_path: sub_txt_path = "D:\data\depression\REST_meta_MDD\group\subject_IDs_selected.txt" # REST_meta_MDD
    if 'HBN' in timeseries_path: sub_txt_path = "/data0/lsy/HBN/subject_IDs.txt" # HBN
    
    # if 'MDD' in timeseries_path: sub_txt_path = "/data0/lsy/REST_meta_MDD/group/subject_IDs_selected_s20.txt" # REST_meta_MDD

    balanced_names = np.genfromtxt(sub_txt_path, dtype=str) # REST_meta_MDD
    print('subject loaded:', sub_txt_path)

    balanced_ind = [index  for index,name in enumerate(sub_names) if name in balanced_names]
    random.seed(123)
    random.shuffle(balanced_ind)
    final_fc = final_fc[balanced_ind]
    labels = labels[balanced_ind]
    sub_names = sub_names[balanced_ind]
    # no combat
    # final_pearson = final_pearson[balanced_ind] # no combat
    # combat
    # if 'SRPBS' in timeseries_path: combat_dict=np.load('/data0/lsy/combat/SRPBS_FC_{}_combat.npy'.format('AAL'),allow_pickle=True).item()
    if 'SRPBS' in timeseries_path: combat_dict=np.load("D:\data\depression\combat_label_ml\SRPBS_FC_{}_combat.npy".format('AAL'),allow_pickle=True).item()
    if 'MDD' in timeseries_path: combat_dict = np.load("D:\data\depression\combat_label_ml\REST_meta_MDD_FC_{}}_combat.npy".format('AAL'),allow_pickle=True).item()
    if 'HBN' in timeseries_path: combat_dict = np.load("/data0/lsy/combat_label_ml/HBN_FC_{}_combat.npy".format('AAL'),allow_pickle=True).item()
    
    final_pearson = [combat_dict[sub] for sub in sub_names]
    final_pearson = [squareform(pearson) for pearson in final_pearson]
    final_pearson = np.array(final_pearson)

    t1_feature = t1_feature[balanced_ind]
    # random.seed(321)
    # random.shuffle(final_fc)
    # random.seed(321)
    # random.shuffle(final_pearson)
    # random.seed(321)
    # random.shuffle(labels)
    # random.seed(321)
    # random.shuffle(sub_names)
    # random.seed(321)
    # random.shuffle(t1_feature)
    str_names = sub_names
    if ('MDD' in timeseries_path) or ('HBN' in timeseries_path):
        if 'MDD' in timeseries_path: meta_file = os.path.join("D:\data\depression\REST_meta_MDD\REST_meta_MDD_pheno.csv")
        if 'HBN' in timeseries_path: meta_file = os.path.join("/data0/lsy/HBN/HBN_pheno.csv")
        
        meta_file = pd.read_csv(meta_file, header=0)
        id2uuid = meta_file[["participant_id", "uuid"]]
        # pandas to map
        id2uuid = id2uuid.set_index("participant_id")
        id2uuid = id2uuid.to_dict()['uuid']
        sub_names = [id2uuid[sub_id] for sub_id in sub_names]

    sub_names = [int(name[4:]) for name in sub_names]
    sub_names = np.array(sub_names,dtype=np.int64)
    # scaler = StandardScaler(mean=np.mean(final_fc), std=np.std(final_fc))
    # final_fc = scaler.transform(final_fc)
    final_fc = [(data_t - np.mean(data_t,axis=1).reshape(-1,1))/np.std(data_t,axis=1).reshape(-1,1) for data_t in final_fc]
    final_fc = np.array(final_fc)
    print('nan in final_fc:',np.isnan(final_fc).any())
    # final_pearson = [np.arctanh(mat) for mat in final_pearson]
    # final_pearson = np.array(final_pearson)
    
    final_fc, final_pearson, labels,t1_feature = [torch.from_numpy(data).float() for data in (final_fc, final_pearson, labels,t1_feature)]
    

    return final_fc, final_pearson, labels,torch.from_numpy(sub_names),t1_feature,str_names
    # dataset = utils.TensorDataset(
    #     final_fc,
    #     final_pearson,
    #     labels,
    #     torch.from_numpy(sub_names),
    #     t1_feature
    # )
    

    # local_dataloader = utils.DataLoader(dataset, batch_size=config["data"]["batch_size"], shuffle=False, drop_last=False)

    # return local_dataloader,str_names

def prepare_local_dataloader(timeseries_path,t1_root):
    all_final_fc = []
    all_final_pearson = []
    all_labels = []
    all_idx_names = []
    all_t1_feature = []
    all_str_names = []

    for i in range(len(timeseries_path)):
        final_fc, final_pearson, labels,idx_names,t1_feature,str_names = prepare_local_data(timeseries_path[i],t1_root[i])
        if len(all_idx_names) != 0: idx_names = idx_names + len(all_idx_names[-1])
        all_final_fc.append(final_fc[:,:,:config['data']['window_width']])
        all_final_pearson.append(final_pearson)
        all_labels.append(labels)
        all_idx_names.append(idx_names)
        all_t1_feature.append(t1_feature)
        all_str_names.append(str_names)
    
    final_fc = torch.cat(all_final_fc,dim=0)
    final_pearson = torch.cat(all_final_pearson)
    labels = torch.cat(all_labels)
    idx_names = torch.cat(all_idx_names)
    t1_feature = torch.cat(all_t1_feature)
    str_names = np.concatenate(all_str_names)
    
    dataset = utils.TensorDataset(
        final_fc,
        final_pearson,
        labels,
        idx_names,
        t1_feature
    )
    
    local_dataloader = utils.DataLoader(dataset, batch_size=config["data"]["batch_size"], shuffle=False, drop_last=False)

    return local_dataloader,str_names


if __name__ == "__main__":
    site = np.zeros([4], dtype=np.int)
    print(site)
    print(site.shape)
