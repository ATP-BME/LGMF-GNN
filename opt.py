# 20230503
import os
import datetime
import argparse
import random
import numpy as np
import torch
import logging
import logging.config

class OptInit():
    def __init__(self):
        # dataset
        parser = argparse.ArgumentParser(description='PyTorch implementation of EV-GCN')

        # dataset
        parser.add_argument('--config_filename', default='setting/SRPBS_fbnetgen.yaml', type=str, help='Configuration filename for training the model.')
        # parser.add_argument('--config_filename', default='setting/REST_meta_MDD_fbnetgen.yaml', type=str, help='Configuration filename for training the model.')
        # parser.add_argument('--config_filename', default='setting/All_fbnetgen.yaml', type=str, help='Configuration filename for training the model.')

        parser.add_argument('--test dataset', default='test balanced new t1 feature', type=str, help='mode test')
        parser.add_argument('--mode', default='mode6', type=str, help='mode of node feature')
        # mode 
        # 1. AAL correlation + CC200 correlation + phenotype
        # 2. AAL correlation + CC200 correlation + phenotype + T1
        # 3. AAL embedding + CC200 embedding + phenotype
        # 4. AAL correlation + CC200 graph embedding + phenotype
        # 4. AAL graph embedding + CC200 graph embedding + phenotype + T1
        # 5. AAL correlation|graph embedding + CC200 correlation|graph embedding + phenotype
        # 6. AAL correlation|graph embedding + CC200 correlation|graph embedding + phenotype + T1
        parser.add_argument('--exp_info', default='use one feature | snowball| combat label ml | adversarial domain loss | (8,12) ', type=str, help='mode of node feature')
        parser.add_argument('--pheno_edge_threshold', type=int, default=1.1, help='mode of test dataset')

        parser.add_argument('--train', default=0, type=int, help='train(default) or evaluate')
        parser.add_argument('--construct_graph', default=1, type=int, help='train(default) or evaluate')
        parser.add_argument('--use_cpu', action='store_true', help='use cpu?')

        parser.add_argument('--mixup', default=True, type = bool,help='use graph mixup')
        parser.add_argument('--mixup_rate', default=0.3, type = float,help='the percentage mixup nodes')

        parser.add_argument('--shift_robust', default=False, type=bool, help='use shift robust loss')
        parser.add_argument('--shift_loss_weight', default=4, type=float, help='shift loss weight default: 1 ')

        parser.add_argument('--hgc', type=int, default=16, help='hidden units of gconv layer')
        parser.add_argument('--lg', type=int, default=4, help='number of gconv layers')
        parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
        parser.add_argument('--wd', default=5e-5, type=float, help='weight decay')#5e-5
        parser.add_argument('--num_iter', default=500, type=int, help='number of epochs for training')
        parser.add_argument('--node_feature_dim', default=512, type=int, help='dimension of node feature default 2000|3600|900')
        parser.add_argument('--edropout', type=float, default=0.3, help='edge dropout rate')
        parser.add_argument('--dropout', default=0.5, type=float, help='ratio of dropout')
        parser.add_argument('--snowball_layer_num', default=9, type=int, help='num of snowball layer')
        parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
        parser.add_argument('--ckpt_path', type=str, default='./save_models/SRPBS/exp1', help='checkpoint path to save trained models')
        parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
        parser.add_argument('--dataset', type=str, default='cora', help='Dataset (cora, citeseer, pubmed)')
        parser.add_argument('--public', type=int, default=0, help='Use the Public Setting of the Dataset of not')
        parser.add_argument('--percent', type=float, default=0.0003, help='Percentage of training set.')
        parser.add_argument('--network', type=str, default='linear_snowball',
                            help='Network type (snowball, linear_snowball, truncated_krylov)')
        parser.add_argument('--validation', type=int, default=1, help='1 for tuning on validation set, 0 for not')
        # MODEL HYPERPARAMETERS
        parser.add_argument('--weight_decay', type=float, default=5e-3, help='Weight decay.')
        parser.add_argument('--hidden', type=int, default=128, help='Width of hidden layers')
        parser.add_argument('--layers', type=int, default=8, help='Number of hidden layers, i.e. network depth')
        parser.add_argument('--activation', type=str, default="tanh", help='Activation Function')
        parser.add_argument('--n', type=int, default=12, help='knn default 8')
        parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer')
        parser.add_argument('--stepsize', type=int, default=100, help='scheduler step size')
        parser.add_argument('--gamma', type=float, default=0.5, help='scheduler shrinking rate')
        parser.add_argument('--n_blocks', type=int, default=5,
                            help='Number of Krylov blocks for truncated_krylov network')
        # for PairNorm
        # - PairNorm mode, use PN-SI or PN-SCS for GCN and GAT. With more than 5 layers get lots improvement.
        parser.add_argument('--norm_mode', type=str, default='PN', help='Mode for PairNorm, {None, PN, PN-SI, PN-SCS}')
        parser.add_argument('--norm_scale', type=float, default=1.0, help='Row-normalization scale')
        # STOPPING CRITERIA
        parser.add_argument('--epochs', type=int, default=200, help='Number of max epochs to train.')
        parser.add_argument('--consecutive', type=int, default=200, help='Consecutive 100% training accuracy to stop')
        parser.add_argument('--early_stopping', type=int, default=100, help='Early Stopping')
        parser.add_argument('--epochs_after_peak', type=int, default=200,
                            help='Number of More Epochs Needed after 100% Training Accuracy Happens')
        # MISC
        parser.add_argument('--seed', type=int, default=42, help='Random seed.')
        parser.add_argument('--walltime', type=float, default=10800, help='Random seed.')
        parser.add_argument('--runtimes', type=int, default=10, help='Runtimes.')
        parser.add_argument('--debug', type=int, default=1, help='1 for prompts during running, 0 for none')
        parser.add_argument('--identifier', type=int, default=1234567, help='Identifier for the job')
        # FOR TORCH IMPLEMENTATION
        parser.add_argument('--amp', type=int, default=2,
                            help='1, 2 and 3 for NVIDIA apex amp optimization O1, O2 and O3, 0 for off')
        args = parser.parse_args()

        args.time = datetime.datetime.now().strftime("%y%m%d")

        if args.use_cpu:
            args.device = torch.device('cpu')
        else:
            args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            # print(" Using GPU in torch")

        if args.activation == 'identity' or args.network == 'linear_snowball':
            activation = lambda X: X
        elif args.activation == 'tanh':
            activation = torch.tanh
        else:
            activation = eval("F.%s" % args.activation)
        self.args = args

    def print_args(self):
        # self.args.printer args
        print("==========       CONFIG      =============")
        for arg, content in self.args.__dict__.items():
            print("{}:{}".format(arg, content))
        print("==========     CONFIG END    =============")
        print("\n")
        phase = 'train' if self.args.train==1 else 'eval'
        print('===> Phase is {}.'.format(phase))

    def initialize(self):
        self.set_seed(123)
        #self.logging_init()
        # self.print_args()
        return self.args

    def set_seed(self, seed=123):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


