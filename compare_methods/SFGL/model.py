import torch
import numpy as np
import torch.nn as nn
from einops import rearrange

class LayerGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, epsilon=True):
        super().__init__()
        if epsilon: self.epsilon = nn.Parameter(torch.Tensor([[0.0]]))
        else: self.epsilon = 0.0
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim), nn.BatchNorm1d(output_dim), nn.ReLU())

    def forward(self, v, a):
        v_aggregate = torch.sparse.mm(a, v)
        v_aggregate += self.epsilon * v
        v_combine = self.mlp(v_aggregate)
        return v_combine

class ModuleMeanReadout(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, node_axis=1):
        return x.mean(node_axis), torch.zeros(size=[1,1,1], dtype=torch.float32)

class ModuleSERO(nn.Module):
    def __init__(self, hidden_dim, input_dim, dropout=0.1, upscale=1.0):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(hidden_dim, round(upscale*hidden_dim)), nn.BatchNorm1d(round(upscale*hidden_dim)), nn.GELU())
        self.attend = nn.Linear(round(upscale*hidden_dim), input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, node_axis=1):
        x_readout = x.mean(node_axis)
        x_shape = x_readout.shape
        x_embed = self.embed(x_readout.reshape(-1,x_shape[-1]))
        x_graphattention = torch.sigmoid(self.attend(x_embed)).view(*x_shape[:-1],-1)
        permute_idx = list(range(node_axis))+[len(x_graphattention.shape)-1]+list(range(node_axis,len(x_graphattention.shape)-1))
        x_graphattention = x_graphattention.permute(permute_idx)
        return (x * self.dropout(x_graphattention.unsqueeze(-1))).mean(node_axis), x_graphattention.permute(1,0,2)

class ModuleGARO(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, upscale=1.0, **kwargs):
        super().__init__()
        self.embed_query = nn.Linear(hidden_dim, round(upscale*hidden_dim))
        self.embed_key = nn.Linear(hidden_dim, round(upscale*hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, node_axis=1):
        x_q = self.embed_query(x.mean(node_axis, keepdims=True))
        x_k = self.embed_key(x)
        x_graphattention = torch.sigmoid(torch.matmul(x_q, rearrange(x_k, 't b n c -> t b c n'))/np.sqrt(x_q.shape[-1])).squeeze(2)
        return (x * self.dropout(x_graphattention.unsqueeze(-1))).mean(node_axis), x_graphattention.permute(1,0,2)

class ModuleTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, input_dim))

    def forward(self, x):
        x_attend, attn_matrix = self.multihead_attn(x, x, x)
        x_attend = self.dropout1(x_attend)
        x_attend = self.layer_norm1(x_attend)
        x_attend2 = self.mlp(x_attend)
        x_attend = x_attend + self.dropout2(x_attend2)
        x_attend = self.layer_norm2(x_attend)
        return x_attend, attn_matrix

def batch_adj(X):
    # Size of input: batch x node x time
    # Size of output: batch x node x node
    result = []
    for x in X:
        mean_x = torch.mean(x, 1, keepdim=True)
        xm = x.sub(mean_x.expand_as(x))
        c = xm.mm(xm.t())
        c = c / (x.size(1) - 1)
        d = torch.diag(c)
        stddev = torch.pow(d, 0.5)
        c = c.div(stddev.expand_as(c))
        c = c.div(stddev.expand_as(c).t())
        c = torch.clamp(c, -1.0, 1.0)
        result.append(c)
    return  torch.stack(result, dim=0)

def up_triu(X, k=1):
    # Size of input: batch x node x node
    # Size of output: batch x
    r = []
    for x in X:
        x = x.numpy()
        uptri = np.triu(x, k)
        uptri = torch.from_numpy(uptri)
        r.append(uptri)
    return torch.stack(r, dim=0)

def flatten(X, non_zero=True):
    r = []
    for x in X:
        x = x.view(-1)
        if non_zero:
            idx = torch.nonzero(x)
            x = x[idx].squeeze()
        r.append(x)
    return torch.stack(r, dim=0)

# STAGIN
class ModelSTAGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_heads, num_layers, sparsity, dropout=0.5, cls_token='sum', readout='sero', garo_upscale=1.0):
        # input_dim: node, hidden_dim: dimension of hidden layer, num_classes: class number
        # num_heads: head number of Transformer
        # num_layers: number of GIN layers
        # sparsity: threshold of adjacency matrix sparsity
        super().__init__()
        assert cls_token in ['sum', 'mean', 'param']
        if cls_token=='sum': self.cls_token = lambda x: x.sum(0)
        elif cls_token=='mean': self.cls_token = lambda x: x.mean(0)
        elif cls_token=='param': self.cls_token = lambda x: x[-1]
        else: raise
        if readout=='garo': readout_module = ModuleGARO
        elif readout=='sero': readout_module = ModuleSERO
        elif readout=='mean': readout_module = ModuleMeanReadout
        else: raise

        self.token_parameter = nn.Parameter(torch.randn([num_layers, 1, 1, hidden_dim])) if cls_token=='param' else None

        self.num_classes = num_classes
        self.sparsity = sparsity

        self.percentile = Percentile()
        self.initial_linear = nn.Linear(116, hidden_dim)
        self.gnn_layers = nn.ModuleList()
        self.readout_modules = nn.ModuleList()
        self.transformer_modules = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for i in range(num_layers):
            self.gnn_layers.append(LayerGIN(hidden_dim, hidden_dim, hidden_dim))
            self.readout_modules.append(readout_module(hidden_dim=hidden_dim, input_dim=input_dim, dropout=0.1))
            self.transformer_modules.append(ModuleTransformer(hidden_dim, 2*hidden_dim, num_heads=num_heads, dropout=0.1))
            self.linear_layers.append(nn.Linear(hidden_dim, num_classes))

    def _collate_adjacency(self, a, sparsity, sparse=True):
        i_list = []
        v_list = []

        for sample, _dyn_a in enumerate(a):
            for timepoint, _a in enumerate(_dyn_a):
                thresholded_a = (_a > self.percentile(_a, 100-sparsity))
                _i = thresholded_a.nonzero(as_tuple=False)
                _v = torch.ones(len(_i))
                _i += sample * a.shape[1] * a.shape[2] + timepoint * a.shape[2]
                i_list.append(_i)
                v_list.append(_v)
        _i = torch.cat(i_list).T.to(a.device)
        _v = torch.cat(v_list).to(a.device)

        return torch.sparse.FloatTensor(_i, _v, (a.shape[0]*a.shape[1]*a.shape[2], a.shape[0]*a.shape[1]*a.shape[3]))

    def forward(self, a):
        # a: dynamic FCN
        # Size of a: batch x window x node x node
        logit = 0.0
        logit2 = 0.0
        reg_ortho = 0.0
        attention = {'node-attention': [], 'time-attention': []}
        latent_list = []
        minibatch_size, num_timepoints, num_nodes = a.shape[:3]
        h = a
        h = rearrange(h, 'b t n c -> (b t n) c')
        h = self.initial_linear(h)
        adj = self._collate_adjacency(a, self.sparsity)
        for layer, (G, R, T, L) in enumerate(zip(self.gnn_layers, self.readout_modules, self.transformer_modules, self.linear_layers)):
            h = G(h, adj)
            h_bridge = rearrange(h, '(b t n) c -> t b n c', t=num_timepoints, b=minibatch_size, n=num_nodes)
            h_readout, node_attn = R(h_bridge, node_axis=2)
            if self.token_parameter is not None: h_readout = torch.cat([h_readout, self.token_parameter[layer].expand(-1,h_readout.shape[1],-1)])
            h_attend, time_attn = T(h_readout)
            ortho_latent = rearrange(h_bridge, 't b n c -> (t b) n c')
            matrix_inner = torch.bmm(ortho_latent, ortho_latent.permute(0,2,1))
            reg_ortho += (matrix_inner/matrix_inner.max(-1)[0].unsqueeze(-1) - torch.eye(num_nodes, device=matrix_inner.device)).triu().norm(dim=(1,2)).mean()

            latent = self.cls_token(h_attend)
            logit += latent
            logit2 += self.dropout(L(latent))

            attention['node-attention'].append(node_attn)
            attention['time-attention'].append(time_attn)
            latent_list.append(latent)

        attention['node-attention'] = torch.stack(attention['node-attention'], dim=1).detach().cpu()
        attention['time-attention'] = torch.stack(attention['time-attention'], dim=1).detach().cpu()
        latent = torch.stack(latent_list, dim=1)

        return logit, logit2, attention, latent, reg_ortho

# personalized branch
class Population(nn.Module):
    def __init__(self, p_dim, bold_dim, hidden_dim, latent_dim, output_dim, Type, out_dim):
        # p_dim: dimension of demographic information
        # bold_dim: dimension of flattened bold upper triangular matrix
        # hidden_dim: dimension of hidden layers
        # latent_dim: dimension of logit from STAGIN
        # output_dim: dimension of the layer before score
        # Type: feature aggregation method
        # out_dim: class number
        super(Population, self).__init__()
        self.Type = Type
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(bold_dim, hidden_dim)
        self.fc_p = nn.Linear(p_dim, output_dim)
        if self.Type == 'sum': self.fc_latent = nn.Linear(latent_dim, output_dim)
        else: self.fc_latent = nn.Linear(latent_dim, output_dim+hidden_dim)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(hidden_dim)
        if self.Type == 'sum': self.fc_final = nn.Linear(output_dim, out_dim)
        else: self.fc_final = nn.Linear(output_dim+hidden_dim, out_dim)

    def forward(self, p, bold, latent, g):
        # p: demographic information, bold: flattened bold upper triangular matrix, latent: logit from STAGIN, g: gamma
        bold = self.dropout(self.relu(self.fc(self.dropout(bold))))
        p = self.fc_p(p)
        if self.Type == 'sum':
            f_o = g*(bold+p)+(1-g)*self.fc_latent(latent)
            out = self.fc_final(f_o)
        else:
            f_o = g*(torch.cat((bold,p),dim=1))+(1-g)*self.fc_latent(latent)
            out = self.fc_final(f_o)
        return out, f_o

# Percentile class based on
# https://github.com/aliutkus/torchpercentile
class Percentile(torch.autograd.Function):
    def __init__(self):
        super().__init__()

    def __call__(self, input, percentiles):
        return self.forward(input, percentiles)

    def forward(self, input, percentiles):
        input = torch.flatten(input)
        input_dtype = input.dtype
        input_shape = input.shape
        if isinstance(percentiles, int):
            percentiles = (percentiles,)
        if not isinstance(percentiles, torch.Tensor):
            percentiles = torch.tensor(percentiles, dtype=torch.double)
        if not isinstance(percentiles, torch.Tensor):
            percentiles = torch.tensor(percentiles)
        input = input.double()
        percentiles = percentiles.to(input.device).double()
        input = input.view(input.shape[0], -1)
        in_sorted, in_argsort = torch.sort(input, dim=0)
        positions = percentiles * (input.shape[0]-1) / 100
        floored = torch.floor(positions)
        ceiled = floored + 1
        ceiled[ceiled > input.shape[0] - 1] = input.shape[0] - 1
        weight_ceiled = positions-floored
        weight_floored = 1.0 - weight_ceiled
        d0 = in_sorted[floored.long(), :] * weight_floored[:, None]
        d1 = in_sorted[ceiled.long(), :] * weight_ceiled[:, None]
        self.save_for_backward(input_shape, in_argsort, floored.long(),
                               ceiled.long(), weight_floored, weight_ceiled)
        result = (d0+d1).view(-1, *input_shape[1:])
        return result.type(input_dtype)

    def backward(self, grad_output):
        """
        backward the gradient is basically a lookup table, but with weights
        depending on the distance between each point and the closest
        percentiles
        """
        (input_shape, in_argsort, floored, ceiled,
         weight_floored, weight_ceiled) = self.saved_tensors

        # the argsort in the flattened in vector

        cols_offsets = (
            torch.arange(
                    0, input_shape[1], device=in_argsort.device)
            )[None, :].long()
        in_argsort = (in_argsort*input_shape[1] + cols_offsets).view(-1).long()
        floored = (
            floored[:, None]*input_shape[1] + cols_offsets).view(-1).long()
        ceiled = (
            ceiled[:, None]*input_shape[1] + cols_offsets).view(-1).long()

        grad_input = torch.zeros((in_argsort.size()), device=self.device)
        grad_input[in_argsort[floored]] += (grad_output
                                            * weight_floored[:, None]).view(-1)
        grad_input[in_argsort[ceiled]] += (grad_output
                                           * weight_ceiled[:, None]).view(-1)

        grad_input = grad_input.view(*input_shape)
        return grad_input

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

