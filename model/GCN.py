import numpy as np
import torch
from torch.nn import Linear as Lin, Sequential as Seq
import torch_geometric as tg
import torch.nn.functional as F
from torch import nn
from PAE import PAE
from model.layers import general_GCN_layer, snowball_layer,truncated_krylov_layer

class graph_convolutional_network(nn.Module):
    def __init__(self, nfeat, nlayers, nhid, nclass, dropout_rate):
        super(graph_convolutional_network, self).__init__()
        self.nfeat, self.nlayers, self.nhid, self.nclass = nfeat, nlayers, nhid, nclass
        self.dropout_rate = dropout_rate
        self.hidden = nn.ModuleList()

    def reset_parameters(self):
        for layer in self.hidden:
            layer.reset_parameters()
        self.out.reset_parameters()


class snowball(graph_convolutional_network):
    def __init__(self, nfeat, nlayers, nhid, nclass, dropout, activation):
        super(snowball, self).__init__(nfeat, nlayers, nhid, nclass, dropout)
        self.activation = activation
        for k in range(nlayers):
            self.hidden.append(snowball_layer(k * nhid + nfeat, nhid))
        self.out = snowball_layer(nlayers * nhid + nfeat, nclass)

    def forward(self, x, adj):
        list_output_blocks = []
        for layer, layer_num in zip(self.hidden, np.arange(self.nlayers)):
            if layer_num == 0:
                list_output_blocks.append(
                    F.dropout(self.activation(layer(x, adj)), self.dropout, training=self.training))
            else:
                list_output_blocks.append(
                    F.dropout(self.activation(layer(torch.cat([x] + list_output_blocks[0: layer_num], 1), adj)),
                              self.dropout, training=self.training))
        output = self.out(torch.cat([x] + list_output_blocks, 1), adj, eye=False)

        # output = (output - output.mean(axis=0)) / output.std(axis=0)
        F.normalize(output)
        return output


class truncated_krylov(graph_convolutional_network):
    def __init__(self, nfeat, nlayers, nhid, nclass, dropout, activation, n_blocks, adj, features):
        super(truncated_krylov, self).__init__(nfeat, nlayers, nhid, nclass, dropout)
        self.activation = activation
        LIST_A_EXP, LIST_A_EXP_X, A_EXP = [], [], torch.eye(adj.size()[0], dtype=adj.dtype).cuda()
        if str(adj.layout) == 'torch.sparse_coo':
            dense_adj = adj.to_dense()
        else:
            dense_adj = adj
        for _ in range(n_blocks):
            if nlayers > 1:
                indices = torch.nonzero(A_EXP).t()
                values = A_EXP[indices[0], indices[1]]
                LIST_A_EXP.append(torch.sparse.FloatTensor(indices, values, A_EXP.size()))

            LIST_A_EXP_X.append(torch.mm(A_EXP, features))
            torch.cuda.empty_cache()
            A_EXP = torch.mm(A_EXP, dense_adj)
        self.hidden.append(truncated_krylov_layer(nfeat, n_blocks, nhid, LIST_A_EXP_X_CAT=torch.cat(LIST_A_EXP_X, 1)))
        for _ in range(nlayers - 1):
            self.hidden.append(truncated_krylov_layer(nhid, n_blocks, nhid, LIST_A_EXP=LIST_A_EXP))
        self.out = truncated_krylov_layer(nhid, 1, nclass)

    def forward(self, x, adj):
        list_output_blocks = []
        for layer, layer_num in zip(self.hidden, np.arange(self.nlayers)):
            if layer_num == 0:
                list_output_blocks.append(
                    F.dropout(self.activation(layer(x, adj)), self.dropout, training=self.training))
            else:
                list_output_blocks.append(
                    F.dropout(self.activation(layer(list_output_blocks[layer_num - 1], adj)), self.dropout,
                              training=self.training))
        output = self.out(list_output_blocks[self.nlayers - 1], adj, eye=True)
        # output = (output - output.mean(axis=0)) / output.std(axis=0)
        output=F.normalize(output)
        return output



class EV_GCN(torch.nn.Module):
    def __init__(self, input_dim, num_classes, dropout, edgenet_input_dim, edge_dropout, hgc, lg):
        super(EV_GCN, self).__init__()
        K = 3
        hidden = [hgc for i in range(lg)]
        self.dropout = dropout
        self.edge_dropout = edge_dropout
        bias = False
        self.relu = torch.nn.ReLU(inplace=True)
        self.lg = lg
        self.gconv = nn.ModuleList()
        for i in range(lg):
            in_channels = input_dim if i == 0 else hidden[i - 1]
            self.gconv.append(tg.nn.ChebConv(in_channels, hidden[i], K, normalization='sym', bias=bias))
            # self.gconv.append(snowball(2000,8,16,2,0.2,nn.Tanh()))

        cls_input_dim = sum(hidden)


        self.cls = nn.Sequential(
            torch.nn.Linear(cls_input_dim, 256),
            torch.nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            torch.nn.Linear(256, num_classes))

        self.edge_net = PAE(input_dim=edgenet_input_dim // 2, dropout=dropout)
        # a=PAE(input_dim=edgenet_input_dim // 2, dropout=dropout)
        # print("a===",a)
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, features, edge_index, edgenet_input, enforce_edropout=False):
        # if self.edge_dropout > 0:
        #     if enforce_edropout or self.training:
        #         one_mask = torch.ones([edgenet_input.shape[0], 1]).cuda()
        #         print(one_mask)
        #         self.drop_mask = F.dropout(one_mask, self.edge_dropout, True)
        #         self.bool_mask = torch.squeeze(self.drop_mask.type(torch.bool))
        #         edge_index = edge_index[:, self.bool_mask]
        #         print("edge_index",edge_index)
        #         print(edge_index.shape)
        #         edgenet_input = edgenet_input[self.bool_mask]

        edge_weight = torch.squeeze(self.edge_net(edgenet_input))
        # print("edge_weight11111", edge_weight)
        # print(edge_weight.shape)
        features = F.dropout(features, self.dropout, self.training)
        # print(features.shape)

        h = self.relu(self.gconv[0](features, edge_index, edge_weight))
        h0 = h

        for i in range(1, self.lg):
            h = F.dropout(h, self.dropout, self.training)
            h = self.relu(self.gconv[i](h, edge_index, edge_weight))
            jk = torch.cat((h0, h), axis=1)
            h0 = jk
        logit = self.cls(jk)

        return logit, edge_weight

