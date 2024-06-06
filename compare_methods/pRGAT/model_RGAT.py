"""实现GAT 类。"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from RGATConv import RGATConv



class RGAT(nn.Module):
        def __init__(self, in_c, hid_c, out_c, log_attention_weights=False):
            super(RGAT, self).__init__()
            self.conv1 = RGATConv(in_channels=in_c, out_channels=hid_c)
            self.conv2 = RGATConv(in_channels=hid_c, out_channels=out_c)

        def forward(self, data, edge):  # forward calculation function
            # data.x data.edge_index
            x = data.x  # [N, C]
            edge_index = data.edge_index  # [2 ,E]
            hid = self.conv1(x=x, edge_index=edge_index, edge_value=edge)  # [N, D]
            hid = F.relu(hid)

            out_c = self.conv2(hid, edge_index, edge)

            soft = F.log_softmax(out_c, dim=1)

            return soft


