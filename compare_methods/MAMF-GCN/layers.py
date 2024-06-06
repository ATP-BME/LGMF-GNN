import math, torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules.module import Module
import torch.nn as nn


class general_GCN_layer(Module):
    def __init__(self):
        super(general_GCN_layer, self).__init__()

    @staticmethod
    def multiplication(A, B):
        if str(A.layout) == 'torch.sparse_coo':
            return torch.spmm(A, B)
        else:
            return torch.mm(A, B)

class InecptionGCNBlock(Module):
    """
    The multiple layer GCN with inception connection block.
    """

    def __init__(self, in_features,nhid,out_features, nbaselayer,
                 withbn=True, withloop=True, activation=F.relu, dropout=True,
                 aggrmethod="concat", dense=False):
        """
        The multiple layer GCN with inception connection block.
        :param in_features: the input feature dimension.
        :param out_features: the hidden feature dimension.
        :param nbaselayer: the number of layers in the base block.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param activation: the activation function, default is ReLu.
        :param dropout: the dropout ratio.
        :param aggrmethod: the aggregation function for baseblock, can be "concat" and "add". For "resgcn", the default
                           is "add", for others the default is "concat".
        :param dense: not applied. The default is False, cannot be changed.
        """
        super(InecptionGCNBlock, self).__init__()
        self.in_features = in_features
        self.nhid = nhid
        self.out_features = out_features
        self.hiddendim = out_features
        self.nbaselayer = nbaselayer
        self.activation = activation
        self.aggrmethod = aggrmethod
        self.dropout = dropout
        self.withbn = withbn
        self.withloop = withloop
        self.midlayers = nn.ModuleList()
        self.__makehidden()

        if self.aggrmethod == "concat":
            self.out_features = in_features + out_features * nbaselayer
        elif self.aggrmethod == "add":
            if in_features != self.hiddendim:
                raise RuntimeError("The dimension of in_features and hiddendim should be matched in 'add' model.")
            self.out_features = out_features
        else:
            raise NotImplementedError("The aggregation method only support 'concat', 'add'.")

    def __makehidden(self):
        # for j in xrange(self.nhiddenlayer):
        for j in range(self.nbaselayer):
            reslayer = nn.ModuleList()
            # for i in xrange(j + 1):
            for i in range(j + 1):
                if i == 0:
                    layer = GraphConvolutionBS(self.in_features,self.nhid, self.hiddendim, self.activation, self.withbn,
                                               self.withloop)
                else:
                    layer = GraphConvolutionBS(self.hiddendim,self.nhid, self.hiddendim, self.activation, self.withbn,
                                               self.withloop)
                reslayer.append(layer)
            self.midlayers.append(reslayer)

    def forward(self, input, adj):
        x = input
        for reslayer in self.midlayers:
            subx = input
            for gc in reslayer:
                subx = gc(subx, adj)
                subx = F.dropout(subx, self.dropout, training=self.training)
            x = self._doconcat(x, subx)
            x = F.normalize(x)
        return x

    def get_outdim(self):
        return self.out_features

    def _doconcat(self, x, subx):
        if self.aggrmethod == "concat":
            return torch.cat((x, subx), 1)
        elif self.aggrmethod == "add":
            return x + subx

    def __repr__(self):
        return "%s %s (%d - [%d:%d] > %d)" % (self.__class__.__name__,
                                              self.aggrmethod,
                                              self.in_features,
                                              self.hiddendim,
                                              self.nbaselayer,
                                              self.out_features)
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc3 = GraphConvolution(nhid, 32)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training = self.training)
        # x = F.relu(self.gc3(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # x = (x - x.mean(axis=0)) / x.std(axis=0)
        x=F.normalize(x)
        # torch.nn.functional.normalize(x)
        # print(x.shape)
        return x
class GraphConvolutionBS(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GraphConvolutionBS, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc3 = GraphConvolution(nhid, 32)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training = self.training)
        # x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # x = (x - x.mean(axis=0)) / x.std(axis=0)
        x=F.normalize(x)
        # torch.nn.functional.normalize(x)
        # print(x.shape)
        return x
class GraphConvolutionBS2(Module):
    """
    GCN Layer with BN, Self-loop and Res connection.
    """

    def __init__(self, in_features, out_features, activation=lambda x: x, withbn=True, withloop=True, bias=True,
                 res=False):
        """
        Initial function.
        :param in_features: the input feature dimension.
        :param out_features: the output feature dimension.
        :param activation: the activation function.
        :param withbn: using batch normalization.
        :param withloop: using self feature modeling.
        :param bias: enable bias.
        :param res: enable res connections.
        """
        super(GraphConvolutionBS2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = activation
        self.res = res

        # Parameter setting.
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # Is this the best practice or not?
        if withloop:
            self.self_weight = Parameter(torch.FloatTensor(in_features, out_features))
        else:
            self.register_parameter("self_weight", None)

        if withbn:
            self.bn = torch.nn.BatchNorm1d(out_features)
        else:
            self.register_parameter("bn", None)

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.self_weight is not None:
            stdv = 1. / math.sqrt(self.self_weight.size(1))
            self.self_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)

        # Self-loop
        if self.self_weight is not None:
            output = output + torch.mm(input, self.self_weight)

        if self.bias is not None:
            output = output + self.bias
        # BN
        if self.bn is not None:
            output = self.bn(output)
        # Res
        if self.res:
            return self.sigma(output) + input
        else:
            return self.sigma(output)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class snowball_layer(general_GCN_layer):
    def __init__(self, in_features, out_features):
        super(snowball_layer, self).__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight, self.bias = Parameter(torch.FloatTensor(self.in_features, self.out_features).cuda()), Parameter(
            torch.FloatTensor(self.out_features).cuda())
        self.reset_parameters()

    def reset_parameters(self):
        stdv_weight, stdv_bias = 1. / math.sqrt(self.weight.size(1)), 1. / math.sqrt(self.bias.size(0))
        torch.nn.init.uniform_(self.weight, -stdv_weight, stdv_weight)
        torch.nn.init.uniform_(self.bias, -stdv_bias, stdv_bias)

    def forward(self, input, adj, eye=False):
        XW = torch.mm(input, self.weight)
        if eye:
            return XW + self.bias
        else:
            return self.multiplication(adj, XW) + self.bias

class truncated_krylov_layer(general_GCN_layer):
    def __init__(self, in_features, n_blocks, out_features, LIST_A_EXP=None, LIST_A_EXP_X_CAT=None):
        super(truncated_krylov_layer, self).__init__()
        self.LIST_A_EXP = LIST_A_EXP
        self.LIST_A_EXP_X_CAT = LIST_A_EXP_X_CAT
        self.in_features, self.out_features, self.n_blocks = in_features, out_features, n_blocks
        self.shared_weight, self.output_bias = Parameter(
            torch.FloatTensor(self.in_features * self.n_blocks, self.out_features).cuda()), Parameter(
            torch.FloatTensor(self.out_features).cuda())
        self.reset_parameters()

    def reset_parameters(self):
        stdv_shared_weight, stdv_output_bias = 1. / math.sqrt(self.shared_weight.size(1)), 1. / math.sqrt(
            self.output_bias.size(0))
        torch.nn.init.uniform_(self.shared_weight, -stdv_shared_weight, stdv_shared_weight)
        torch.nn.init.uniform_(self.output_bias, -stdv_output_bias, stdv_output_bias)

    def forward(self, input, adj, eye=True):

        if self.n_blocks == 1:
            output = torch.mm(input, self.shared_weight)
            output = (output - output.mean(axis=0)) / output.std(axis=0)
        elif self.LIST_A_EXP_X_CAT is not None:
            output = torch.mm(self.LIST_A_EXP_X_CAT, self.shared_weight)
            output = (output - output.mean(axis=0)) / output.std(axis=0)
        elif self.LIST_A_EXP is not None:
            feature_output = []
            for i in range(self.n_blocks):
                AX = self.multiplication(self.LIST_A_EXP[i], input)
                feature_output.append(AX)
            output = torch.mm(torch.cat(feature_output, 1), self.shared_weight)
            output = (output - output.mean(axis=0)) / output.std(axis=0)
        if eye:
            return output + self.output_bias
        else:
            return self.multiplication(adj, output) + self.output_bias

class GraphConvolution2(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution2, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        # support.cpu()
        output = torch.spmm(adj, support)
        # output = F.normalize(output)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


