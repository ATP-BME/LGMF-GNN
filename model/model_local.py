from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, MaxPool1d, Linear, GRU
import math
import os
from torch.autograd import Function

EXTRACT = False # save graph feature vector

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))


def gumbel_softmax_sample(logits, temperature, eps=1e-10):
    sample = sample_gumbel(logits.size(), eps=eps)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

def gumbel_softmax(logits, temperature, hard=False, eps=1e-10):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y_soft = gumbel_softmax_sample(logits, temperature=temperature, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        y_hard = torch.zeros(*shape).cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


class GruKRegion(nn.Module):

    def __init__(self, kernel_size=8, layers=4, out_size=8, dropout=0.5):
        super().__init__()
        self.gru = GRU(kernel_size, kernel_size, layers,
                       bidirectional=True, batch_first=True)

        self.kernel_size = kernel_size

        self.linear = nn.Sequential(
            nn.Dropout(dropout),
            Linear(kernel_size*2, kernel_size),
            nn.LeakyReLU(negative_slope=0.2),
            # nn.Tanh(),
            Linear(kernel_size, out_size)
        )

    def forward(self, raw):

        b, k, d = raw.shape # batch,nROI,timeseries

        x = raw.view((b*k, -1, self.kernel_size))

        x, h = self.gru(x)

        x = x[:, -1, :]

        x = x.view((b, k, -1))

        x = self.linear(x)
        return x


class ConvKRegion(nn.Module):

    def __init__(self, k=1, out_size=8, kernel_size=8, pool_size=16, time_series=512):
        super().__init__()
        self.conv1 = Conv1d(in_channels=k, out_channels=32,
                            kernel_size=kernel_size, stride=2)

        output_dim_1 = (time_series-kernel_size)//2+1

        self.conv2 = Conv1d(in_channels=32, out_channels=32,
                            kernel_size=8)
        output_dim_2 = output_dim_1 - 8 + 1
        self.conv3 = Conv1d(in_channels=32, out_channels=16,
                            kernel_size=8)
        output_dim_3 = output_dim_2 - 8 + 1
        self.max_pool1 = MaxPool1d(pool_size)
        output_dim_4 = output_dim_3 // pool_size * 16
        self.in0 = nn.InstanceNorm1d(time_series)
        self.in1 = nn.BatchNorm1d(32)
        self.in2 = nn.BatchNorm1d(32)
        self.in3 = nn.BatchNorm1d(16)

        self.linear = nn.Sequential(
            Linear(output_dim_4, 32),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(32, out_size)
        )

    def forward(self, x):

        b, k, d = x.shape

        x = torch.transpose(x, 1, 2)

        x = self.in0(x)

        x = torch.transpose(x, 1, 2)
        x = x.contiguous()

        x = x.view((b*k, 1, d))

        x = self.conv1(x)

        x = self.in1(x)
        x = self.conv2(x)

        x = self.in2(x)
        x = self.conv3(x)

        x = self.in3(x)
        x = self.max_pool1(x)

        x = x.view((b, k, -1))

        x = self.linear(x)

        return x


class SeqenceModel(nn.Module):

    def __init__(self, model_config, roi_num=360, time_series=512):
        super().__init__()

        if model_config['extractor_type'] == 'cnn':
            self.extract = ConvKRegion(
                out_size=model_config['embedding_size'], kernel_size=model_config['window_size'],
                time_series=time_series, pool_size=4, )
        elif model_config['extractor_type'] == 'gru':
            self.extract = GruKRegion(
                out_size=model_config['embedding_size'], kernel_size=model_config['window_size'],
                layers=model_config['num_gru_layers'], dropout=model_config['dropout'])

        self.linear = nn.Sequential(
            Linear(model_config['embedding_size']*roi_num, 256),
            nn.Dropout(model_config['dropout']),
            nn.ReLU(),
            Linear(256, 32),
            nn.Dropout(model_config['dropout']),
            nn.ReLU(),
            Linear(32, 2)
        )

    def forward(self, x):
        x = self.extract(x)
        x = x.flatten(start_dim=1)
        x = self.linear(x)
        return x


class Embed2GraphByProduct(nn.Module):

    def __init__(self, input_dim, roi_num=264):
        super().__init__()

    def forward(self, x):

        m = torch.einsum('ijk,ipk->ijp', x, x)
        # i,j,k分别表示x的三个维度， i,p,k分别表示x的三个维度， 最后矩阵相乘得到的维度为i,j,p,
        # 其实就是batch size维度不变，后两个维度装置相乘 h*h^T
        m = torch.unsqueeze(m, -1) # 在最后插入一个维度

        return m


class Embed2GraphByLinear(nn.Module):

    def __init__(self, input_dim, roi_num=360):
        super().__init__()

        self.fc_out = nn.Linear(input_dim * 2, input_dim)
        self.fc_cat = nn.Linear(input_dim, 1)

        def encode_onehot(labels):
            classes = set(labels)
            classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                            enumerate(classes)}
            labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                     dtype=np.int32)
            return labels_onehot

        off_diag = np.ones([roi_num, roi_num])
        rel_rec = np.array(encode_onehot(
            np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(
            np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec).cuda()
        self.rel_send = torch.FloatTensor(rel_send).cuda()

    def forward(self, x):

        batch_sz, region_num, _ = x.shape
        receivers = torch.matmul(self.rel_rec, x)

        senders = torch.matmul(self.rel_send, x)
        x = torch.cat([senders, receivers], dim=2)
        x = torch.relu(self.fc_out(x))
        x = self.fc_cat(x)

        x = torch.relu(x)

        m = torch.reshape(
            x, (batch_sz, region_num, region_num, -1))
        return m



class GNNPredictor(nn.Module):

    def __init__(self, node_input_dim, site_num,roi_num=360,embed_dim=32):
        super().__init__()
        inner_dim = roi_num
        self.roi_num = roi_num
        self.gcn = nn.Sequential(
            nn.Linear(node_input_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
            # nn.Tanh(),
            Linear(inner_dim, inner_dim)
        )
        self.bn1 = torch.nn.BatchNorm1d(inner_dim)

        self.gcn1 = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
            # nn.Tanh(),
        )
        self.bn2 = torch.nn.BatchNorm1d(inner_dim)
        self.gcn2 = nn.Sequential(
            nn.Linear(inner_dim, 64),
            nn.LeakyReLU(negative_slope=0.2),
            # nn.Tanh(),
            nn.Linear(64, embed_dim), 
            nn.LeakyReLU(negative_slope=0.2),
            # nn.Tanh(),
        )
        self.bn3 = torch.nn.BatchNorm1d(inner_dim)
        self.attention = SelfAttention(roi_num,dim_q=32,dim_v=1)

        self.fcn = nn.Sequential(
            nn.Linear(8*roi_num, 256), # cc200
            # nn.Linear(14*roi_num, 256), # aal
            nn.LeakyReLU(negative_slope=0.2),
            # nn.Tanh(),
            nn.Linear(256, 32),
            nn.LeakyReLU(negative_slope=0.2),
            # nn.Tanh(),
            nn.Linear(32, 2)
        )
        self.t1_map = nn.Sequential(
            nn.Linear(2416,928), 
            nn.ReLU()
        )
        self.site_d = nn.Sequential(
            nn.Linear(8*roi_num, 128), # cc200
            # nn.Linear(14*roi_num, 256), # aal
            nn.LeakyReLU(negative_slope=0.2),
            # nn.Tanh(),
            nn.Linear(128, 32),
            nn.LeakyReLU(negative_slope=0.2),
            # nn.Tanh(),
            nn.Linear(32, site_num)
        )


    def forward(self, m, node_feature,t1_feature=None,sub_names=None): # A, pearson correlaton matrix
        bz = m.shape[0]

        x = torch.einsum('ijk,ijp->ijp', m, node_feature) # j=k

        x = self.gcn(x)

        x = x.reshape((bz*self.roi_num, -1))
        x = self.bn1(x)
        x = x.reshape((bz, self.roi_num, -1))

        x = torch.einsum('ijk,ijp->ijp', m, x)

        x = self.gcn1(x)

        x = x.reshape((bz*self.roi_num, -1))
        x = self.bn2(x)
        x = x.reshape((bz, self.roi_num, -1))

        x = torch.einsum('ijk,ijp->ijp', m, x)

        x = self.gcn2(x)

        x = self.bn3(x) # [bs,nroi,embed dim]

        # attention_score = F.softmax(torch.sum(m,dim=1),dim=1) * m.shape[-1]
        
        attention_score = self.attention(m)
        attention_score = F.softmax(attention_score,dim=1) * m.shape[-1]
        # attention_score = F.softmax(torch.sum(m,dim=1),dim=1) * m.shape[-1]
        x = attention_score.reshape(bz,-1,1)*x
        # x = torch.sum(x,dim=1)
        x = x.view(bz,-1)
        # 加入梯度翻转层用于对抗训练(最小化域分类loss，但是分类器部分正常回传，梯度部分翻转后回传)
        alpha = 0.5
        reverse_x = ReverseLayerF.apply(x, alpha)
        t1_reduced = self.t1_map(t1_feature)

        # x = x.view(bz,-1)
        if EXTRACT:
            if not os.path.exists("/data0/lsy/SRPBS_new/FBNETGEN_graph_feature/aal_1624"):
                os.makedirs("/data0/lsy/SRPBS_new/FBNETGEN_graph_feature/aal_1624")
            sub_names = 'sub_'+str(sub_names.item()).zfill(4)
            np.save("/data0/lsy/SRPBS_new/FBNETGEN_graph_feature/aal_1624/{}.npy".format(sub_names),x.detach().cpu().squeeze().numpy())
            print('extracted:',sub_names)

        # return self.fcn(x),attention_score,t1_reduced
        return self.fcn(x),self.site_d(reverse_x),x,attention_score,t1_reduced


class SelfAttention(nn.Module):
    def __init__(self, input_dim, dim_q, dim_v):
        super(SelfAttention, self).__init__()
        # dim_q = dim_k
        self.dim_q, self.dim_k, self.dim_v = dim_q, dim_q, dim_v

        self.Q = nn.Linear(input_dim, dim_q)
        self.K = nn.Linear(input_dim, dim_q)
        self.V = nn.Linear(input_dim, dim_v)
        self._norm_fact = 1 / math.sqrt(self.dim_k)

    def forward(self, x):
        # Q: [batch_size,seq_len,dim_q]
        # K: [batch_size,seq_len,dim_k]
        # V: [batch_size,seq_len,dim_v]
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        # print(f'x.shape:{x.shape} , Q.shape:{Q.shape} , K.shape: {K.shape} , V.shape:{V.shape}')

        attention = nn.Softmax(dim=-1)(
            torch.matmul(Q, K.permute(0, 2, 1)))  # Q * K.T() # batch_size * seq_len * seq_len

        attention = torch.matmul(attention, V).reshape(x.shape[0], x.shape[1],
                                                       -1)  # Q * K.T() * V # batch_size * seq_len * dim_v

        return attention

class FBNETGEN(nn.Module):

    def __init__(self, model_config, site_num,roi_num=360, node_feature_dim=360, time_series=512,embed_dim=32):
        super().__init__()
        self.graph_generation = model_config['graph_generation']
        if model_config['extractor_type'] == 'cnn':
            self.extract = ConvKRegion(
                out_size=model_config['embedding_size'], kernel_size=model_config['window_size'],
                time_series=time_series)
        elif model_config['extractor_type'] == 'gru':
            self.extract = GruKRegion(
                out_size=model_config['embedding_size'], kernel_size=model_config['window_size'],
                layers=model_config['num_gru_layers'],dropout=model_config['dropout'])
        if self.graph_generation == "linear":
            self.emb2graph = Embed2GraphByLinear(
                model_config['embedding_size'], roi_num=roi_num)
        elif self.graph_generation == "product":
            self.emb2graph = Embed2GraphByProduct(
                model_config['embedding_size'], roi_num=roi_num)

        self.predictor = GNNPredictor(node_feature_dim,site_num=site_num, roi_num=roi_num,embed_dim=embed_dim)

    def forward(self, t, nodes,t1_feature,sub_names=None):
        x = self.extract(t) # x -> h
        # x = F.softmax(x, dim=-1)
        m = self.emb2graph(x) # A = h * h^T

        m = m[:, :, :, 0] # 前面插入了一个维度，这里取出来还是三个维度 [batch size,node num,node num]

        bz, _, _ = m.shape

        edge_variance = torch.mean(torch.var(m.reshape((bz, -1)), dim=1))

        local_output,local_site_output,local_embedding,attention_score,t1_reduced = self.predictor(m, nodes,t1_feature,sub_names)

        return local_output,local_site_output,local_embedding,attention_score,t1_reduced, m, edge_variance


class E2EBlock(torch.nn.Module):
    '''E2Eblock.'''

    def __init__(self, in_planes, planes, roi_num, bias=True):
        super().__init__()
        self.d = roi_num
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a]*self.d, 3)+torch.cat([b]*self.d, 2)


class BrainNetCNN(torch.nn.Module):
    def __init__(self, roi_num):
        super().__init__()
        self.in_planes = 1
        self.d = roi_num

        self.e2econv1 = E2EBlock(1, 32, roi_num, bias=True)
        self.e2econv2 = E2EBlock(32, 64, roi_num, bias=True)
        self.E2N = torch.nn.Conv2d(64, 1, (1, self.d))
        self.N2G = torch.nn.Conv2d(1, 256, (self.d, 1))
        self.dense1 = torch.nn.Linear(256, 128)
        self.dense2 = torch.nn.Linear(128, 30)
        self.dense3 = torch.nn.Linear(30, 2)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        out = F.leaky_relu(self.e2econv1(x), negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out), negative_slope=0.33)
        out = F.leaky_relu(self.E2N(out), negative_slope=0.33)
        out = F.dropout(F.leaky_relu(
            self.N2G(out), negative_slope=0.33), p=0.5)
        out = out.view(out.size(0), -1)
        out = F.dropout(F.leaky_relu(
            self.dense1(out), negative_slope=0.33), p=0.5)
        out = F.dropout(F.leaky_relu(
            self.dense2(out), negative_slope=0.33), p=0.5)
        out = F.leaky_relu(self.dense3(out), negative_slope=0.33)

        return out


class FCNet(nn.Module):

    def __init__(self, node_size, seq_len, kernel_size=3):
        super().__init__()

        self.ind1, self.ind2 = torch.triu_indices(node_size, node_size, offset=1)

        seq_len -= kernel_size//2*2
        channel1 = 32
        self.block1 = nn.Sequential(
            Conv1d(in_channels=1, out_channels=channel1,
                            kernel_size=kernel_size),
            nn.BatchNorm1d(channel1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        seq_len //= 2  

        seq_len -= kernel_size//2*2
        channel2 = 64
        self.block2 = nn.Sequential(
            Conv1d(in_channels=channel1, out_channels=channel2,
                            kernel_size=kernel_size),
            nn.BatchNorm1d(channel2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        seq_len //= 2 

        seq_len -= kernel_size//2*2
        channel3 = 96
        self.block3 = nn.Sequential(
            Conv1d(in_channels=channel2, out_channels=channel3,
                            kernel_size=kernel_size),
            nn.BatchNorm1d(channel3),
            nn.LeakyReLU()
        )

        channel4 = 64
        self.block4 = nn.Sequential(
            Conv1d(in_channels=channel3, out_channels=channel4,
                            kernel_size=kernel_size),
            Conv1d(in_channels=channel4, out_channels=channel4,
                            kernel_size=kernel_size),
            nn.MaxPool1d(kernel_size=2, stride=2)  
        )
        seq_len -= kernel_size//2*2
        seq_len -= kernel_size//2*2
        seq_len //= 2  
        
               
        self.fc = nn.Linear(in_features=seq_len*channel4, out_features=32)

        self.diff_mode = nn.Sequential(
            nn.Linear(in_features=32*2, out_features=32),
            nn.Linear(in_features=32, out_features=32),
            nn.Linear(in_features=32, out_features=2)
        )

    def forward(self, x):
        bz, _, time_series = x.shape

        x = x.reshape((bz*2, 1, time_series))

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        
        x = x.reshape((bz, 2, -1))

        x = self.fc(x)

        x = x.reshape((bz, -1))

        diff = self.diff_mode(x)

        return diff

        
