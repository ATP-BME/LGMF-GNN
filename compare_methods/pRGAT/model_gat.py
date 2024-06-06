"""实现GAT 类。"""
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as pyg_nn



class GAT(nn.Module):
        def __init__(self, in_c, hid_c, out_c, log_attention_weights=False):
            super(GAT, self).__init__()
            self.conv1 = pyg_nn.GATConv(in_channels=in_c, out_channels=hid_c)
            self.conv2 = pyg_nn.GATConv(in_channels=hid_c, out_channels=out_c)

        def forward(self, data):
            hid = self.conv1(data.x, data.edge_index)  # [N, D]
            hid = F.relu(hid)

            out_c = self.conv2(hid, data.edge_index)

            soft = F.log_softmax(out_c, dim=1)  # [N, out_c]

            return soft


