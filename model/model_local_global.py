import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear as Lin
import os
import numpy as np
import yaml
import scipy.sparse as sp
from sklearn.metrics import accuracy_score

from utils.gcn_utils import normalize,normalize_torch
from utils.graph_mixup import g_mixup
from model.model_local import FBNETGEN
from model.model_global import MAMFGCN
from dataloader_local_global import dataloader_lg,prepare_local_dataloader

from opt import *
opt = OptInit().initialize()
device = opt.device

with open(opt.config_filename) as f:
        config = yaml.load(f, Loader=yaml.Loader)

class PAE(torch.nn.Module):
    def __init__(self, input_dim, dropout=0.2):
        super(PAE, self).__init__()
        hidden=128
        self.parser =nn.Sequential(
                nn.Linear(input_dim, hidden, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden, bias=True),
                )
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.input_dim = input_dim
        self.model_init()
        self.relu = nn.ReLU(inplace=True)
        self.elu = nn.ReLU()

    def forward(self, x):
        x1 = x[:,0:self.input_dim]
        x2 = x[:,self.input_dim:]
        h1 = self.parser(x1) 
        h2 = self.parser(x2) 
        p = (self.cos(h1,h2) + 1)*0.5
        # p = abs(self.cos(h1, h2))
        # print(p)
        # print("p=",p)
        # print(p.shape)
        return p

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):  #isinstance() 函数来判断一个对象是否是一个已知的类型，类似 type()。
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

class LGMF_GNN(torch.nn.Module):
    def __init__(self,nonimg,site_num,roi_num,local_fea_dim,global_fea_dim,timeseries_len,local_dataloader,train_HC_ind,train_MDD_ind) -> None:
        super(LGMF_GNN,self).__init__()

        self.nonimg = nonimg
        self.site_num = site_num
        self.roi_num = roi_num
        self.local_fea_dim = local_fea_dim
        self.global_fea_dim = global_fea_dim
        self.time_series_len = timeseries_len
        # self.local_dataloader = prepare_local_dataloader(config['data']['time_seires'],config['data']['t1_root'])
        self.local_dataloader = local_dataloader
        self.edge_dropout = opt.edropout
        self.train_HC_ind = train_HC_ind
        self.train_MDD_ind = train_MDD_ind
        self._setup()
    
    def _setup(self):
        self.local_gnn = FBNETGEN(config['model'], 
                                  site_num = self.site_num,
                                  roi_num=self.roi_num, 
                                  node_feature_dim=self.local_fea_dim, 
                                  time_series=self.time_series_len,
                                  embed_dim=config['model']['embedding_size'])
        self.global_gnn = MAMFGCN(nfeat=self.global_fea_dim, #2000,
                                  nhid=32,
                                  out=16,
                                  nclass=2,
                                  nhidlayer=1,
                                  dropout=opt.dropout,
                                  baseblock="inceptiongcn",
                                  inputlayer="gcn",
                                  outputlayer="gcn",
                                  nbaselayer=6,
                                  activation=F.relu,
                                  withbn=False,
                                  withloop=False,
                                  aggrmethod="concat",
                                  mixmode=False,)
        self.edge_net = PAE(2*self.nonimg.shape[1]//2,0.6)

        # self.local_gnn.load_state_dict(torch.load('/home/sjtu/liushuyu/project/LGMF-GCN/result_local/02-15-16-26-08_SRPBS_fbnetgen_normal_gru_loss_group_loss_sparsity_loss_8_4/best_model.pt'),strict=False)
        # print('local model loaded!')
        self.local_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.local_site_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self,dl,train_ind=None,n=None):
        # dl = dataloader_lg()
        embeddings = []
        t1_features = []
        labels = []
        local_preds = []
        local_site_preds = []
        if not opt.train:
            local_m = []
            local_attention = []
        local_loss = 0
        local_site_loss = 0
        local_acc = 0
        local_site_acc = 0
        for data_in,pearson,label,sub_names,t1_feature in self.local_dataloader:
            label = label.long()
            data_in, pearson, label,t1_feature = data_in.to(device), pearson.to(device), label.to(device),t1_feature.to(device)
            # print(label)
        # for data_in,pearson,label,sub_names,t1_feature in self.local_dataloader:
        #     data_in, pearson, label = data_in.to(device), pearson.to(device), label.to(device)
            # embedding,attention_score,t1_reduced, m, edge_variance = self.local_gnn(data_in,pearson,sub_names,t1_feature)
            local_output,local_site_output,embedding,attention_score,t1_reduced,m,_ = self.local_gnn(data_in,pearson,t1_feature,sub_names)
            # local_loss += self.local_loss_fn(local_output,label)
            embeddings.append(embedding)
            t1_features.append(t1_reduced)
            labels.append(label)
            local_preds.append(local_output)
            local_site_preds.append(local_site_output)
            if not opt.train:
                local_m.append(m)
                local_attention.append(attention_score)
        embeddings = torch.cat(tuple(embeddings))
        t1_features = torch.cat(tuple(t1_features))
        labels = torch.cat(tuple(labels))
        local_preds = torch.cat(tuple(local_preds))
        local_site_preds = torch.cat(tuple(local_site_preds))
        if not opt.train:
            local_m = torch.cat(tuple(local_m))
            local_attention = torch.cat(tuple(local_attention))
            np.save('HC_local_m.npy',local_m[self.train_MDD_ind].detach().cpu().numpy())
            np.save('MDD_local_m.npy',local_m[self.train_MDD_ind].detach().cpu().numpy())
            np.save('HC_local_attention.npy',local_attention[self.train_MDD_ind].detach().cpu().numpy())
            np.save('MDD_local_attention.npy',local_attention[self.train_MDD_ind].detach().cpu().numpy())


        if train_ind is not None:
            # disease cls
            local_loss = self.local_loss_fn(local_preds[train_ind],labels[train_ind])
            local_acc = accuracy_score(labels[train_ind].detach().cpu(),local_preds[train_ind].max(1)[1].detach().cpu())
            # site cls
            y_site = torch.from_numpy(dl.site).long().to(device)
            local_site_loss = self.local_site_loss_fn(local_site_preds[train_ind],y_site[train_ind].to(device))
            local_site_acc = accuracy_score(y_site[train_ind].detach().cpu(),local_site_preds[train_ind].max(1)[1].detach().cpu())
            
            # local_loss = local_loss - local_site_loss

        np.save('label.npy',{'labels':labels.detach().cpu().numpy(),'site_labels':dl.site})
        # normalize with sum
        # sum_e = torch.pow(torch.sum(embeddings,dim=-1,keepdim=True).reshape(-1,1),-1)
        # sum_e[torch.isinf(sum_e)] = 0
        # embeddings = embeddings * sum_e
        # sum_e = torch.pow(torch.sum(t1_features,dim=-1,keepdim=True).reshape(-1,1),-1)
        # sum_e[torch.isinf(sum_e)] = 0
        # t1_features = t1_features * sum_e

        # normalize with norm
        embeddings = embeddings / torch.norm(embeddings, dim=-1, keepdim=True)
        t1_features = t1_features / torch.norm(t1_features, dim=-1, keepdim=True)
        # print(np.isnan(t1_features.cpu().detach().numpy()).any())


        edge_index, edge_input = dl.get_PAE_inputs(self.nonimg, embeddings) 
        # edge input: concat of noimg node feature of the nodes linked by the edge
        edge_input = (edge_input- edge_input.mean(axis=0)) / (edge_input.std(axis=0)+1e-6)
        # print(np.isnan(edge_input).any())
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(opt.device)
        edge_input = torch.from_numpy(edge_input).to(opt.device)

        if self.edge_dropout > 0:
            if self.training:
                one_mask = torch.ones([edge_input.shape[0],1]).to(device)
                self.drop_mask = F.dropout(one_mask, self.edge_dropout, True)
                self.bool_mask = torch.squeeze(self.drop_mask.type(torch.bool))
                edge_index = edge_index[:, self.bool_mask]
                edge_input = edge_input[self.bool_mask]

        
        edge_weight = torch.squeeze(self.edge_net(edge_input)) # W_i,j
        # sadj = np.zeros([embeddings.shape[0], embeddings.shape[0]])
        sadj = torch.zeros([embeddings.shape[0], embeddings.shape[0]]).to(opt.device)

        for i in range(edge_index.shape[1]):
            sadj[edge_index[0][i]][edge_index[1][i]] = edge_weight[i] # 根据变性数据得到的边以及权值填入NxN的权值矩阵
            sadj[edge_index[1][i]][edge_index[0][i]] = edge_weight[i]

        # fadj1
        # fadj1 = (torch.cosine_similarity(embeddings.unsqueeze(1),embeddings.unsqueeze(0),dim=-1) + 1) * 0.5
        # fadj2 = (torch.cosine_similarity(t1_features.unsqueeze(1),t1_features.unsqueeze(0),dim=-1) + 1) * 0.5

        ## 或者
        # nembeddings = embeddings / torch.norm(embeddings, dim=-1, keepdim=True)
        # nt1_features = t1_features / torch.norm(t1_features, dim=-1, keepdim=True)
        fadj1 = (torch.mm(embeddings,embeddings.T) +1 ) * 0.5
        fadj2 = (torch.mm(t1_features,t1_features.T) +1 ) * 0.5
        
        if n is None:
            # generating random samples
            k_num = range(8,12)
            # k_num = range(12,17)
            k_num = np.random.choice(k_num, size=1)[0]
            
            # k_num = opt.n
        else:
            k_num = n
        
        knn_fadj1 = torch.zeros(fadj1.shape)
        knn_fadj2 = torch.zeros(fadj2.shape)
        knn_fadj1[torch.arange(len(fadj1)).unsqueeze(1),torch.topk(fadj1,k_num).indices]=1
        knn_fadj2[torch.arange(len(fadj2)).unsqueeze(1),torch.topk(fadj2,k_num).indices]=1
        fadj1 = knn_fadj1
        fadj2 = knn_fadj2


        # print(np.isnan(sadj.detach().cpu().numpy()).any())
        # print(np.isnan(fadj1.detach().cpu().numpy()).any())
        # print(np.isnan(fadj1.detach().cpu().numpy()).any())
        # sadj = prepare_adj(sadj)
        fadj1 = prepare_adj(fadj1).to(opt.device)
        fadj2 = prepare_adj(fadj2).to(opt.device)
        sadj = prepare_adj(sadj).to(opt.device)

        # sadj = torch.as_tensor(sadj, dtype=torch.float32).to(opt.device) # 表型数据计算的邻接矩阵
        # fadj1 = torch.as_tensor(fadj1, dtype=torch.float32).to(opt.device) # 影像数据计算的邻接矩阵
        # fadj2 = torch.as_tensor(fadj2, dtype=torch.float32).to(opt.device)
        if opt.mixup and self.training:
            embeddings,fadj1 = g_mixup(embeddings,fadj1,self.train_HC_ind,mixup_rate = opt.mixup_rate)
            embeddings,fadj1 = g_mixup(embeddings,fadj1,self.train_MDD_ind,mixup_rate = opt.mixup_rate)

        
        node_logits, att, emb1, com1, com2,com3, emb2,emb3 = self.global_gnn([embeddings,t1_features], sadj,fadj1,fadj2)
        return node_logits, att, emb1, com1, com2,com3, emb2,emb3,k_num,local_loss,local_site_loss,local_acc,local_site_acc

# def prepare_adj(adj):
#     adj = adj.detach().cpu().numpy()
#     # nfadj = adj.to_sparse().requires_grad_(True)
#     nfadj = sp.coo_matrix(adj)
#     nfadj = nfadj + nfadj.T.multiply(nfadj.T > nfadj) - nfadj.multiply(nfadj.T > nfadj)
#     nfadj = normalize(nfadj + sp.eye(nfadj.shape[0]))
#     nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
#     return nfadj

def prepare_adj(adj):
    # nfadj = adj.to_sparse_coo()
    # nfadj = adj.to_sparse().requires_grad_(True)
    nfadj = adj
    nfadj = nfadj + nfadj.T.multiply(nfadj.T > nfadj) - nfadj.multiply(nfadj.T > nfadj)
    nfadj = normalize_torch(torch.eye(nfadj.shape[0]).to(torch.device(nfadj.device)) + nfadj)
    # nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    # nfadj = adj.to_sparse_coo()
    return nfadj


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)