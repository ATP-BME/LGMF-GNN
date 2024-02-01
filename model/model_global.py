
import torch.nn as nn
import torch.nn.functional as F
from model.layers import PairNorm,GraphConvolution, GraphConvolution2, InecptionGCNBlock, GraphConvolutionBS,GraphConvolutionBS2, snowball_layer,GCN
from model.GCN import snowball, truncated_krylov, graph_convolutional_network
import numpy as np
from torch.nn.parameter import Parameter
import torch
import networkx as nx
from torch.nn.modules.module import Module
import math
from opt import *

opt = OptInit().initialize()
device = opt.device
# cudaid = "cuda:0"
# device = torch.device(cudaid)

def normalize( A, symmetric=True):
    # A = torch.from_numpy(A)
    A=A.cpu()
    # A = A.numpy()
    # A = A + torch.eye(A.size(0))
    # A=torch.from_numpy(A)
    # A=A.numpy()
    d = A.sum(1)
    # d=torch.from_numpy(d)
    if symmetric:
        D = torch.diag(torch.pow(d, -0.5))
        return D.mm(A).mm(D)
    else:
        D = torch.diag(torch.pow(d, -1))
        return D.mm(A)

class GCNModel(nn.Module):
    """
       The model for the single kind of deepgcn blocks.

       The model architecture likes:
       inputlayer(nfeat)--block(nbaselayer, nhid)--...--outputlayer(nclass)--softmax(nclass)
                           |------  nhidlayer  ----|
       The total layer is nhidlayer*nbaselayer + 2.
       All options are configurable.
    """

    def __init__(self,
                 nfeat=2000,
                 nhid=32,
                 out=16,
                 nclass=2,
                 nhidlayer=1,
                 dropout=0.2,
                 baseblock="inceptiongcn",
                 inputlayer="gcn",
                 outputlayer="gcn",
                 nbaselayer=6,
                 activation=lambda x: x,
                 withbn=True,
                 withloop=True,
                 aggrmethod="concat",
                 mixmode=False):
        """
        Initial function.
        :param nfeat: the input feature dimension.
        :param nhid:  the hidden feature dimension.
        :param nclass: the output feature dimension.
        :param nhidlayer: the number of hidden blocks.
        :param dropout:  the dropout ratio.
        :param baseblock: the baseblock type, can be "mutigcn", "resgcn", "densegcn" and "inceptiongcn".
        :param inputlayer: the input layer type, can be "gcn", "dense", "none".
        :param outputlayer: the input layer type, can be "gcn", "dense".
        :param nbaselayer: the number of layers in one hidden block.
        :param activation: the activation function, default is ReLu.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param aggrmethod: the aggregation function for baseblock, can be "concat" and "add". For "resgcn", the default
                           is "add", for others the default is "concat".
        :param mixmode: enable cpu-gpu mix mode. If true, put the inputlayer to cpu.
        """
        super(GCNModel, self).__init__()
        self.mixmode = mixmode
        self.dropout = dropout

        if baseblock == "inceptiongcn":
            self.BASEBLOCK = InecptionGCNBlock
        else:
            raise NotImplementedError("Current baseblock %s is not supported." % (baseblock))
        if inputlayer == "gcn":
            # input gc
            self.ingc = GraphConvolutionBS(nfeat, nhid,out,0.2)
            baseblockinput = out
        elif inputlayer == "none":
            self.ingc = lambda x: x
            baseblockinput = nfeat


        outactivation = lambda x: x
        if outputlayer == "gcn":
            self.outgc = GraphConvolutionBS(baseblockinput,out, nclass,0.2)
        # elif outputlayer ==  "none": #here can not be none
        #    self.outgc = lambda x: x


        # hidden layer
        self.midlayer = nn.ModuleList()
        # Dense is not supported now.
        # for i in xrange(nhidlayer):
        for i in range(nhidlayer):
            gcb = self.BASEBLOCK(in_features=baseblockinput,
                                 nhid=nhid,
                                 out_features=out,
                                 nbaselayer=nbaselayer,
                                 withbn=withbn,
                                 withloop=withloop,
                                 activation=activation,
                                 dropout=dropout,
                                 dense=False,
                                 aggrmethod=aggrmethod)
            self.midlayer.append(gcb)
            baseblockinput = gcb.get_outdim()
        # output gc
        outactivation = lambda x: x  # we donot need nonlinear activation here.
        self.outgc = GraphConvolutionBS(baseblockinput, nhid,out, 0.2)#nhid原来为nclass

        self.reset_parameters()
        if mixmode:
            self.midlayer = self.midlayer.to(device)
            self.outgc = self.outgc.to(device)

    def reset_parameters(self):
        pass

    def forward(self, fea, adj):
        # input
        if self.mixmode:
            x = self.ingc(fea, adj.cpu())
        else:
            x = self.ingc(fea, adj)

        x = F.dropout(x, self.dropout, training=self.training)
        if self.mixmode:
            x = x.to(device)

        # mid block connections
        # for i in xrange(len(self.midlayer)):
        for i in range(len(self.midlayer)):
            midgc = self.midlayer[i]
            x = midgc(x, adj)
        # output, no relu and dropput here.
        x = self.outgc(x, adj)
        x = F.normalize(x)
        # x = F.log_softmax(x, dim=1)
        return x


class snowball(graph_convolutional_network):
    def __init__(self, nfeat, nlayers, nhid, nclass, dropout_rate, activation):
        super(snowball, self).__init__(nfeat, nlayers, nhid, nclass, dropout_rate)
        self.activation = activation
        for k in range(nlayers):
            self.hidden.append(snowball_layer(k * nhid + nfeat, nhid)) # basic GCN layer
            # self.hidden.append(GraphConvolutionBS2(k * nhid + nfeat, nhid,bias=True,res=False)) # basic GCN layer
        self.out = snowball_layer(nlayers * nhid + nfeat, nclass) # basic GCN layer
        # self.out = GraphConvolutionBS2(nlayers * nhid + nfeat, nclass,bias=True,res=False) # basic GCN layer

        # self.dropout = nn.Dropout(p=dropout_rate)
        self.norm = PairNorm(opt.norm_mode, opt.norm_scale)

    def forward(self, x, adj):
        list_output_blocks = [] # list to save snowball embedding at different stage H1...Hn
        for layer, layer_num in zip(self.hidden, np.arange(self.nlayers)):
            if layer_num == 0:
                list_output_blocks.append(
                    # F.dropout(self.activation(layer(x, adj)), self.dropout, training=self.training))
                    # F.dropout(self.activation(layer(x, adj)), self.dropout_rate, training=self.training))
                    F.dropout(self.activation(self.norm(layer(x, adj))), self.dropout_rate, training=self.training)) # pairnorm
            else:
                list_output_blocks.append(
                    # F.dropout(self.activation(layer(torch.cat([x] + list_output_blocks[0: layer_num], 1), adj)),
                    F.dropout(self.activation(self.norm(layer(torch.cat([x] + list_output_blocks[0: layer_num], 1), adj))),
                              self.dropout_rate, training=self.training))
        output = self.out(torch.cat([x] + list_output_blocks, 1), adj, eye=False) # basic GCN
        # output = self.out(torch.cat([x] + list_output_blocks, 1), adj) # GCN with bn

        # output = (output - output.mean(axis=0)) / output.std(axis=0)

        output=F.normalize(output)
        return output,list_output_blocks

class snowball_final_concat(graph_convolutional_network):
    def __init__(self, nfeat, nlayers, nhid, nclass, dropout_rate, activation):
        super(snowball_final_concat, self).__init__(nfeat, nlayers, nhid, nclass, dropout_rate)
        self.activation = activation
        for k in range(nlayers):
            if k == 0:
                self.hidden.append(GraphConvolutionBS2(nfeat, nhid,bias=True,res=False)) # basic GCN layer
            else:
            # self.hidden.append(snowball_layer(k * nhid + nfeat, nhid)) # basic GCN layer
                self.hidden.append(GraphConvolutionBS2(nhid, nhid,bias=True,res=True)) # basic GCN layer
        # self.out = snowball_layer(nlayers * nhid + nfeat, nclass) # basic GCN layer
        self.out = GraphConvolutionBS2(nlayers * nhid + nfeat, nclass,bias=True,res=False) # basic GCN layer

        # self.dropout = nn.Dropout(p=dropout_rate)
        self.norm = PairNorm(opt.norm_mode, opt.norm_scale)

    def forward(self, x, adj):
        list_output_blocks = [] # list to save snowball embedding at different stage H1...Hn
        for layer, layer_num in zip(self.hidden, np.arange(self.nlayers)):
            if layer_num == 0:
                list_output_blocks.append(
                    # F.dropout(self.activation(layer(x, adj)), self.dropout, training=self.training))
                    # F.dropout(self.activation(layer(x, adj)), self.dropout_rate, training=self.training))
                    F.dropout(self.activation(self.norm(layer(x, adj))), self.dropout_rate, training=self.training)) # pairnorm
            else:
                list_output_blocks.append(
                    # F.dropout(self.activation(layer(torch.cat([x] + list_output_blocks[0: layer_num], 1), adj)),
                    F.dropout(self.activation(self.norm(layer(list_output_blocks[layer_num-1], adj))),
                              self.dropout_rate, training=self.training))
        # output = self.out(torch.cat([x] + list_output_blocks, 1), adj, eye=False) # basic GCN
        output = self.out(torch.cat([x] + list_output_blocks, 1), adj) # GCN with bn

        # output = (output - output.mean(axis=0)) / output.std(axis=0)

        output=F.normalize(output)
        return output

class Attention(nn.Module):
    '''
    input:
        concacted final graph embedding Haal || Hho || Hc || Hp
    '''
    def __init__(self, in_size, hidden_size=2):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        # torch.backends.cudnn.enabled = False
        w = self.project(z)
        beta = torch.softmax(w, dim=1) # attention value
        # beta=1
        return (beta * z).sum(1), beta # a1Haal+a2Hho+a3Hc+a4Hp



class MAMFGCN(nn.Module):
    # def __init__(self, nfeat, nclass, nhid1, nhid2, n, dropout):
    def __init__(self, nfeat, nhid,out, nclass,nhidlayer,dropout,baseblock,inputlayer,outputlayer,nbaselayer,activation,withbn,withloop,aggrmethod,mixmode):
        super(MAMFGCN, self).__init__()

        # snowball_layer_num = 7 #default=9
        snowball_layer_num = opt.snowball_layer_num
        print('snowball layer num:',snowball_layer_num)
        self.SGCN1 = snowball(nfeat, snowball_layer_num, nhid, out, dropout, nn.Tanh()) # timeseries signal
        self.SGCN2 = snowball(nfeat, snowball_layer_num, nhid, out, dropout, nn.Tanh()) # T1
        self.SGCN3 = snowball(nfeat, snowball_layer_num, nhid, out, dropout, nn.Tanh()) # phynotype
        self.CGCN = snowball(nfeat, snowball_layer_num, nhid, out, dropout, nn.Tanh())


        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(out, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attention = Attention(out).to(device)
        self.tanh = nn.Tanh()

        self.MLP = nn.Sequential(
            nn.Linear(out, nclass),
            # nn.LogSoftmax(dim=1)
            nn.Softmax(dim=1)
        )

    # def forward(self, x, sadj, fadj,fadj2):
    def forward(self, x, sadj, fadj, fadj2):
        emb1,emb_list = self.SGCN1(x[1], sadj) # Special_GCN out1 -- sadj structure graph
        np.save('emb_list_demo',torch.stack(emb_list).detach().cpu().numpy())
        emb2,emb_list = self.SGCN2(x[0], fadj) # Special_GCN out2 -- fadj feature graph
        np.save('emb_list_func',torch.stack(emb_list).detach().cpu().numpy())
        emb3,emb_list = self.SGCN3(x[0], fadj2)
        np.save('emb_list_stru',torch.stack(emb_list).detach().cpu().numpy())
        

        com1,_ = self.CGCN(x[1], sadj)  # Common_GCN out1 -- sadj structure graph
        com2,_ = self.CGCN(x[0], fadj)  # Common_GCN out2 -- fadj feature graph
        com3,_ = self.CGCN(x[0], fadj2)
        
        Xcom = (com1 + com2 + com3) / 3
        ##attention
        # emb = torch.stack([emb1, emb2,emb3], dim=1)
        emb = torch.stack([emb1, emb2, emb3, Xcom], dim=1)
        # emb = torch.stack([emb1,emb2], dim=1)

        emb, att = self.attention(emb)
        output = self.MLP(emb)
        return output,att, emb1, com1, com2,com3, emb2, emb3
