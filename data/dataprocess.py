import sys
import pickle as pkl
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn.metrics import pairwise_distances as pair
from utils.gcn_utils import normalize
from dataloader_global import dataloader_g as dataloader
from opt import *

opt = OptInit().initialize()
device = opt.device

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def process_data(dataset):
    names = ['y', 'ty', 'ally','x', 'tx', 'allx','graph']
    objects = []
    for i in range(len(names)):
        with open("../data/cache/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    y, ty, ally, x, tx, allx, graph = tuple(objects)
    print(graph)
    test_idx_reorder = parse_index_file("../data/cache/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = features.toarray()
    print(features)
    f = open('../data/{}/{}.adj'.format(dataset, dataset), 'w+')
    for i in range(len(graph)):
        adj_list = graph[i]
        for adj in adj_list:
            f.write(str(i) + '\t' + str(adj) + '\n')
    f.close()

    label_list = []
    for i in labels:
        label = np.where(i == np.max(i))[0][0]
        label_list.append(label)
    np.savetxt('../data/{}/{}.label'.format(dataset, dataset), np.array(label_list), fmt='%d')
    np.savetxt('../data/{}/{}.test'.format(dataset, dataset), np.array(test_idx_range), fmt='%d')
    np.savetxt('../data/{}/{}.feature'.format(dataset, dataset), features, fmt='%f')


def construct_graph( features, topk,train_ind):
    fname = './data/' + 'SRPBS' + '/knn_aal/tmp.txt'
    # print(fname)
    f = open(fname, 'w')
    ##### Kernel
    # dist = -0.5 * pair(features) ** 2
    # dist = np.exp(dist)

    #### Cosine
    dist = cos(features) # 计算余弦相似度，输入降维和归一化之后的邻接矩阵特征，每个节点是一个一维向量 (1035, 1035)
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        # 找到最大的K个值的索引
        inds.append(ind)

    for i, v in enumerate(inds): # i是node的索引，v是选出来的topk个相连的节点的索引
        for vv in v:
            if vv == i:
                pass # 去除自环的边
            else:
                f.write('{} {}\n'.format(i, vv)) # 源节点 目标节点，表示这两个节点之间有边相连
                # f.write('{} {}\n'.format(train_ind[i], train_ind[vv])) # 源节点 目标节点，表示这两个节点之间有边相连

    f.close()

def construct_graph2( features, topk,train_ind):
    fname = './data/' + 'SRPBS' + '/knn_cc200/tmp.txt'
    # print(fname)
    f = open(fname, 'w')
    ##### Kernel
    # dist = -0.5 * pair(features) ** 2
    # dist = np.exp(dist)

    #### Cosine
    dist = cos(features)
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)

    for i, v in enumerate(inds):
        for vv in v:
            if vv == i:
                pass
            else:
                f.write('{} {}\n'.format(i, vv))
                # f.write('{} {}\n'.format(train_ind[i], train_ind[vv]))

    f.close()

def construct_graph3( features, topk,train_ind):
    fname = './data/' + 'SRPBS' + '/knn_T1/tmp.txt'
    # print(fname)
    f = open(fname, 'w')
    ##### Kernel
    # dist = -0.5 * pair(features) ** 2
    # dist = np.exp(dist)

    #### Cosine
    dist = cos(features)
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)

    for i, v in enumerate(inds):
        for vv in v:
            if vv == i:
                pass
            else:
                f.write('{} {}\n'.format(i, vv))
                # f.write('{} {}\n'.format(train_ind[i], train_ind[vv]))

    f.close()

def generate_knn(data):
    for topk in range(2, 10):

        print(data)
        construct_graph( data, topk)
        f1 = open('../data/' + 'ABIDE' + '/knn/tmp.txt','r')
        f2 = open('../data/' + 'ABIDE' + '/knn/c' + str(topk) + '.txt', 'w')
        lines = f1.readlines()
        for line in lines:
            start, end = line.strip('\n').split(' ')
            if int(start) < int(end):
                f2.write('{} {}\n'.format(start, end))
        f2.close()

''' process cora/citeseer/pubmed data '''
#process_data('citeseer')

'''generate KNN graph'''
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


#generate_knn('uai')
def load_graph(f_type,config):
    dl = dataloader()
    raw_features, y, nonimg = dl.load_data()
    featuregraph_path = "./data/ABIDE/knn_{}/c".format(f_type) + str(config) + '.txt'

    feature_edges = np.genfromtxt(featuregraph_path, dtype=np.int32)

    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    # print(fedges.shape)

    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(raw_features.shape[0],raw_features.shape[0]), dtype=np.float32)
    # print("----", fadj)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj) 
    # 邻接矩阵选择有向边中的最大权值 https://blog.csdn.net/panbaoran913/article/details/124042046
    # 即只要存在i->j或者j->i就认为两个节点之间有边

    nfadj = normalize(fadj + sp.eye(fadj.shape[0]))


    # struct_edges = np.genfromtxt(config.structgraph_path, dtype=np.int32)
    #
    # sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    # sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(config.n, config.n), dtype=np.float32)
    # sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    # nsadj = normalize(sadj+sp.eye(sadj.shape[0]))
    #
    # nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    # print(nfadj)
    # print(nfadj.shape)
    # print(nsadj)
    # print(nsadj.shape)
    return  nfadj

def load_graph2( config):
    '''
    same with load graph,
    but use two types of atlas
    '''
    dl = dataloader()
    raw_features1,raw_features2, y, nonimg = dl.load_data()
    # print(raw_features1.shape)
    # print(raw_features2.shape)
    featuregraph_path = './data/' + 'SRPBS' + '/knn_aal/c' + str(config) + '.txt'
    featuregraph_path2 = './data/' + 'SRPBS' + '/knn_cc200/c' + str(config) + '.txt'
#featuregrapg很重要需要修改
    feature_edges = np.genfromtxt(featuregraph_path, dtype=np.int32)
    feature_edges2 = np.genfromtxt(featuregraph_path2, dtype=np.int32)

    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    fedges2 = np.array(list(feature_edges2), dtype=np.int32).reshape(feature_edges2.shape)
    # print(fedges.shape[0])

    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(raw_features1.shape[0],raw_features1.shape[0]), dtype=np.float32)
    fadj2 = sp.coo_matrix((np.ones(fedges2.shape[0]), (fedges2[:, 0], fedges2[:, 1])),shape=(raw_features2.shape[0], raw_features2.shape[0]), dtype=np.float32)
    # print("----", fadj.shape)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    fadj2 = fadj2 + fadj2.T.multiply(fadj2.T > fadj2) - fadj2.multiply(fadj2.T > fadj2)

    nfadj = normalize(fadj + sp.eye(fadj.shape[0]))
    nfadj2 = normalize(fadj2 + sp.eye(fadj2.shape[0]))



    # struct_edges = np.genfromtxt(config.structgraph_path, dtype=np.int32)
    #
    # sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    # sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(config.n, config.n), dtype=np.float32)
    # sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    # nsadj = normalize(sadj+sp.eye(sadj.shape[0]))
    #
    # nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    nfadj2 = sparse_mx_to_torch_sparse_tensor(nfadj2)

    fadj = sparse_mx_to_torch_sparse_tensor(fadj)
    fadj2 = sparse_mx_to_torch_sparse_tensor(fadj2)
    # print(nfadj)
    # print(nfadj.shape)
    # print(nsadj)
    # print(nsadj.shape)
    return  nfadj,nfadj2

def load_graph3( config):
    '''
    same with load graph,
    but use two types of atlas and T1
    '''
    dl = dataloader()
    raw_features1,raw_features2,raw_features3, y, nonimg = dl.load_data()
    # print(raw_features1.shape)
    # print(raw_features2.shape)
    featuregraph_path = './data/' + 'SRPBS' + '/knn_aal/c' + str(config) + '.txt'
    featuregraph_path2 = './data/' + 'SRPBS' + '/knn_cc200/c' + str(config) + '.txt'
    featuregraph_path3 = './data/' + 'SRPBS' + '/knn_T1/c' + str(config) + '.txt'


    feature_edges = np.genfromtxt(featuregraph_path, dtype=np.int32)
    feature_edges2 = np.genfromtxt(featuregraph_path2, dtype=np.int32)
    feature_edges3 = np.genfromtxt(featuregraph_path3, dtype=np.int32)

    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    fedges2 = np.array(list(feature_edges2), dtype=np.int32).reshape(feature_edges2.shape)
    fedges3 = np.array(list(feature_edges3), dtype=np.int32).reshape(feature_edges3.shape)

    # print(fedges.shape[0])

    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(raw_features1.shape[0],raw_features1.shape[0]), dtype=np.float32)
    fadj2 = sp.coo_matrix((np.ones(fedges2.shape[0]), (fedges2[:, 0], fedges2[:, 1])),shape=(raw_features2.shape[0], raw_features2.shape[0]), dtype=np.float32)
    fadj3 = sp.coo_matrix((np.ones(fedges3.shape[0]), (fedges3[:, 0], fedges3[:, 1])),shape=(raw_features3.shape[0], raw_features3.shape[0]), dtype=np.float32)
    
    # print("----", fadj.shape)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    fadj2 = fadj2 + fadj2.T.multiply(fadj2.T > fadj2) - fadj2.multiply(fadj2.T > fadj2)
    fadj3 = fadj3 + fadj3.T.multiply(fadj3.T > fadj3) - fadj3.multiply(fadj3.T > fadj3)


    nfadj = normalize(fadj + sp.eye(fadj.shape[0]))
    nfadj2 = normalize(fadj2 + sp.eye(fadj2.shape[0]))
    nfadj3 = normalize(fadj3 + sp.eye(fadj3.shape[0]))



    # struct_edges = np.genfromtxt(config.structgraph_path, dtype=np.int32)
    #
    # sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    # sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(config.n, config.n), dtype=np.float32)
    # sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    # nsadj = normalize(sadj+sp.eye(sadj.shape[0]))
    #
    # nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    nfadj2 = sparse_mx_to_torch_sparse_tensor(nfadj2)
    nfadj3 = sparse_mx_to_torch_sparse_tensor(nfadj3)

    fadj = sparse_mx_to_torch_sparse_tensor(fadj)
    fadj2 = sparse_mx_to_torch_sparse_tensor(fadj2)
    fadj3 = sparse_mx_to_torch_sparse_tensor(fadj3)
    # print(nfadj)
    # print(nfadj.shape)
    # print(nsadj)
    # print(nsadj.shape)
    return  nfadj,nfadj2,nfadj3

def loss_dependence(emb1, emb2, dim):
    # R = torch.eye(dim).cuda() - (1/dim) * torch.ones(dim, dim).cuda()
    R = torch.eye(dim).to(device) - (1/dim) * torch.ones(dim, dim).to(device)

    K1 = torch.mm(emb1, emb1.t())
    K2 = torch.mm(emb2, emb2.t())
    RK1 = torch.mm(R, K1)
    RK2 = torch.mm(R, K2)
    HSIC = torch.trace(torch.mm(RK1, RK2))
    return HSIC
def common_loss2(emb1, emb2):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    cost = torch.mean((cov1 - cov2)**2)
    return cost

def common_loss3(emb1, emb2,emb3):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb3 = emb3 - torch.mean(emb3, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    emb3 = torch.nn.functional.normalize(emb3, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    cov3 = torch.matmul(emb3, emb3.t())
    cost1 = torch.mean((cov1 - cov2)**2)
    cost2 = torch.mean((cov3 - cov2) ** 2)
    cost3 = torch.mean((cov1 - cov3) ** 2)
    cost=(cost1+cost2+cost3)/3

    return cost

def common_loss4(emb1, emb2,emb3,emb4):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb3 = emb3 - torch.mean(emb3, dim=0, keepdim=True)
    emb4 = emb4 - torch.mean(emb4, dim=0, keepdim=True)

    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    emb3 = torch.nn.functional.normalize(emb3, p=2, dim=1)
    emb4 = torch.nn.functional.normalize(emb4, p=2, dim=1)

    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    cov3 = torch.matmul(emb3, emb3.t())
    cov4 = torch.matmul(emb4, emb4.t())
    cost1 = torch.mean((cov1 - cov2) ** 2)
    cost2 = torch.mean((cov1 - cov3) ** 2)
    cost3 = torch.mean((cov1 - cov4) ** 2)
    cost4 = torch.mean((cov2 - cov3) ** 2) # + torch.mean((cov2 - cov4) ** 2) + torch.mean((cov3 - cov4) ** 2)
    cost=(cost1+cost2+cost3+cost4)/4

    return cost