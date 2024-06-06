from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import torch.nn as nn
import torch.nn.functional as F


def train_val_test_split(kfold = 5, fold = 0):
    #n_sub = 1035
    # n_sub = 494 # HC:DP=1:1
    n_sub = 1012 # HC:DP=3:1
    # n_sub = 745 # same protocol HC:DP=501:244
    # n_sub = 226 # same protocolASD HC
    id = list(range(n_sub))


    import random
    random.seed(123)
    random.shuffle(id)

    kf = KFold(n_splits=kfold, random_state=123,shuffle = True)
    kf2 = KFold(n_splits=kfold-1, shuffle=True, random_state = 666)


    test_index = list()
    train_index = list()
    val_index = list()

    for tr,te in kf.split(np.array(id)):
        test_index.append(te)
        tr_id, val_id = list(kf2.split(tr))[0]
        train_index.append(tr[tr_id])
        val_index.append(tr[val_id])

    train_id = train_index[fold]
    test_id = test_index[fold]
    val_id = val_index[fold]

    return train_id,val_id,test_id

def train_val_split(kfold = 10, fold = 0):
    # n_sub = 1194 # train HC + MDD*3
    # n_sub = 652 # train HC:MDD= new ALFF and corr
    # n_sub = 628 # train HC:MDD= new corr T1
    n_sub = 457 # compare SRPBS
    # n_sub = 1570 # REST_meta_MDD

    id = list(range(n_sub))


    import random
    random.seed(123)
    random.shuffle(id)

    kf = KFold(n_splits=kfold, random_state=123,shuffle = True)
    # kf2 = KFold(n_splits=kfold-1, shuffle=True, random_state = 666)


    train_index = list()
    val_index = list()

    for tr,va in kf.split(np.array(id)):
        val_index.append(va)
        train_index.append(tr)
        

    train_id = train_index[fold]
    val_id = val_index[fold]

    return train_id,val_id

def _expand_binary_labels(labels,label_weights,label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels),0) #[bs,nclass] shaped 0 tensor 
    inds = torch.nonzero(labels>=1).squeeze()
    if inds.numel() >0:
        bin_labels[inds,labels[inds]] = 1 # change label to one-hot
    bin_label_weights = label_weights.view(-1,1).expand(label_weights.size(0),label_channels)
    return bin_labels, bin_label_weights

# 原文链接：https://blog.csdn.net/weixin_44736161/article/details/122403920

# GHM-C loss, loss for biased class training samples
class GHMC(nn.Module):
    """GHM Classification Loss.

    Details of the theorem can be viewed in the paper
    "Gradient Harmonized Single-stage Detector".
    https://arxiv.org/abs/1811.05181

    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        use_sigmoid (bool): Can only be true for BCE based loss now.
        loss_weight (float): The weight of the total GHM-C loss.
    """

    def __init__(self, bins=10, momentum=0, use_sigmoid=True, loss_weight=1.0):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        edges = torch.arange(bins + 1).float() / bins
        self.register_buffer('edges', edges)
        self.edges[-1] += 1e-6
        if momentum > 0:
            acc_sum = torch.zeros(bins)
            self.register_buffer('acc_sum', acc_sum)
        self.use_sigmoid = use_sigmoid
        if not self.use_sigmoid:
            raise NotImplementedError
        self.loss_weight = loss_weight

    def forward(self, pred, target, label_weight, *args, **kwargs):
        """Calculate the GHM-C loss.

        Args:
            pred (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
            label_weight (float tensor of size [batch_num, class_num]):
                the value is 1 if the sample is valid and 0 if ignored.
        Returns:
            The gradient harmonized loss.
        """
        # the target should be binary class label
        if pred.dim() != target.dim():
            target, label_weight = _expand_binary_labels(target, label_weight, pred.size(-1)) # pred [64,2]
        target, label_weight = target.float(), label_weight.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)

        # gradient length
        # sigmoid梯度计算
        g = torch.abs(pred.sigmoid().detach() - target) # [bs,nclass]
        # 有效的label的位置
        valid = label_weight > 0 #[bs,nclass]
        # 有效的label的数量
        tot = max(valid.float().sum().item(), 1.0) #128
        n = 0  # n valid bins
        for i in range(self.bins):
            # 将对应的梯度值划分到对应的bin中， 0-1
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            # 该bin中存在多少个样本
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    # moment计算num bin
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    # 权重等于总数/num bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            # scale系数
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(
            pred, target, weights, reduction='sum') / tot
        return loss * self.loss_weight