import torch
from torch_geometric.loader import DataLoader
from imports.ABIDEDataset import ABIDEDataset,MergedDataset
from imports.utils import train_val_test_split,train_val_split,GHMC
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from gnn import GNN
from torch.utils.tensorboard import SummaryWriter
import os

from tqdm import tqdm
import argparse
import time
import numpy as np

### importing OGB
from myogb.graphproppred.dataset_pyg import PygGraphPropPredDataset
from myogb.graphproppred.evaluate import Evaluator

torch.manual_seed(123)

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

def train(model, device, loader, optimizer, task_type,epoch,writer=None):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y

            if "classification" in task_type: 
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.view(-1,1).to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()
            writer.add_scalars('Loss', {'train_loss': loss.item()},epoch*len(loader)+step)


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            if evaluator.eval_metric == 'acc' or evaluator.eval_metric == 'F1':
                pred = torch.tensor(torch.sigmoid(pred)>0.5,dtype=batch.y.dtype).view(y_true[-1].shape).detach().cpu()
                y_pred.append(pred)
            else:
                y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=2,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--eval_metric', type=str, default='report',
                        help='acc, rocauc, F1,ap,rmse,report')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=2,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--node_dim', type=int, default=200,
                        help='dimensionality of node features (default: 300)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate of optimizer')
    parser.add_argument('--stepsize', type=int, default=50, 
                        help='scheduler step size')
    parser.add_argument('--gamma', type=float, default=0.6, 
                        help='scheduler shrinking rate')
    
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    # parser.add_argument('--dataroot', type=str, default='/data0/lsy/SRPBS_arrange/ALFF_ROI/', 
    parser.add_argument('--dataroot', type=str, default='/data0/lsy/REST_meta_MDD/ROISignals_FunImgARCWF/',
    # parser.add_argument('--dataroot', type=str, default='/data0/lsy/SRPBS_new/ROISignal_AAL/',  # SRPBS AAL
    # parser.add_argument('--dataroot', type=str, default='/data0/lsy/SRPBS_new/BrainGNN/', 
    # parser.add_argument('--dataroot', type=str, default='/data0/lsy/openneuro/processed/',
    # parser.add_argument('--dataroot', type=str, default='/data0/lsy/Anding1/ROI_signal/',
                        help='root directory of the dataset')
    parser.add_argument('--nclass', type=int, default=1, 
                        help='num of classes (the dim of output logits)')
    parser.add_argument('--fold', type=int, default=0, 
                        help='training which fold')
    parser.add_argument('--model_path', type=str, default='./compare_REST/CC200', 
                        help='path to save checkpoint')
    parser.add_argument('--log_path', type=str, default='./compare_REST/CC200', 
                        help='path to save tensorboard log')

    


    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### init tensorboard
    writer = SummaryWriter(os.path.join(args.log_path,str(args.fold)))

    # args.model_path = os.path.join(args.model_path,'gdc_'+str(args.fold)) # gcn
    # args.model_path = os.path.join(args.model_path,'gin2_gdc_'+str(args.fold)) # gin
    args.model_path = os.path.join(args.model_path,'{}3_gdc_{}'.format(args.gnn,str(args.fold)))

    # if not os.path.exists(args.model_path):
    #     os.makedirs(args.model_path)

    ### automatic dataloading and splitting
    # dataset = PygGraphPropPredDataset(name = args.dataset)

    # custom dataset
    name = 'tmp'
    if 'SRPBS' in args.dataroot:
        # dataset = ABIDEDataset(args.dataroot,name,raw_folder='SRPBS') # SRPBS CC200
        dataset = ABIDEDataset(args.dataroot,name,raw_folder='raw') # SRPBS AAL
    elif 'REST' in args.dataroot:
        # dataset = ABIDEDataset(args.dataroot,name,raw_folder='raw_aal')
        dataset = ABIDEDataset(args.dataroot,name,raw_folder='raw_cc200')
    else:
        assert 'openneuro' in args.dataroot or 'Anding' in args.dataroot, 'wrong dataroot'
        dataset_SRPBS = ABIDEDataset('/data0/lsy/SRPBS_new/ROISignal_AAL/',name,raw_folder='raw') # SRPBS AAL
        dataset_REST = ABIDEDataset('/data0/lsy/REST_meta_MDD/ROISignals_FunImgARCWF/',name,raw_folder='raw_aal') # REST raw_cc200 raw_aal
        dataset_HBN = ABIDEDataset(args.dataroot,name,raw_folder='raw') # openneuro raw_aal anding raw
        # dataset = ConcatDataset([dataset_SRPBS, dataset_REST,dataset_HBN])
        dataset = MergedDataset([dataset_SRPBS, dataset_REST, dataset_HBN])

    # dataset = ABIDEDataset(path,name,raw_folder='same_protocol_dp_HC_T1')
    dataset.data.y = dataset.data.y.squeeze()
    dataset.data.x[dataset.data.x == float('inf')] = 0
    dataset.data.y[dataset.data.y == 2] = 1

    # dataset_test = ABIDEDataset(args.dataroot,name,raw_folder='raw_corr_same_protocol_test')
    # dataset = ABIDEDataset(path,name,raw_folder='same_protocol_dp_HC_T1')
    # dataset_test.data.y = dataset_test.data.y.squeeze()
    # dataset_test.data.x[dataset_test.data.x == float('inf')] = 0


    if args.feature == 'full':
        pass 
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:,:2]
        dataset.data.edge_attr = dataset.data.edge_attr[:,:2]

    # split_idx = dataset.get_idx_split()
    # leave one out cv
    if 'openneuro' not in args.dataroot and 'Anding' not in args.dataroot:
        tr_index,val_index = train_val_split(fold=args.fold)
        print('fold:',args.fold)
    else:
        print('out test spliting...')
        import random
        random.seed(123)
        tr_index = list([i for i in range(len(dataset_SRPBS) + len(dataset_REST))])
        val_index = list([i for i in range(len(dataset_SRPBS) + len(dataset_REST),len(dataset))])
        random.shuffle(tr_index)
        random.shuffle(val_index)
        print('done.')
    # cv
    # tr_index,val_index,test_index = train_val_test_split(fold=args.fold)
    print(len(dataset))

    train_dataset = dataset[tr_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[val_index]
    # test_dataset = dataset_test
    print('train:',len(train_dataset))
    print('valid:',len(val_dataset))
    print('test:',len(test_dataset))

    ### automatic evaluator. takes dataset name as input
    # evaluator = Evaluator(num_tasks=args.nclass,eval_metric=args.eval_metric)
    evaluator_acc = Evaluator(num_tasks=args.nclass,eval_metric='acc',fold=args.fold,model_name=args.gnn)
    evaluator_auc = Evaluator(num_tasks=args.nclass,eval_metric='rocauc',fold=args.fold,model_name=args.gnn)
    evaluator_prf = Evaluator(num_tasks=args.nclass,eval_metric='F1',fold=args.fold,model_name=args.gnn)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    if args.gnn == 'gin':
        print('Using GIN')
        model = GNN(gnn_type = 'gin', num_tasks = args.nclass, num_layer = args.num_layer, node_dim=args.node_dim, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_tasks = args.nclass, num_layer = args.num_layer, node_dim=args.node_dim, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type = 'gcn', num_tasks = args.nclass, num_layer = args.num_layer, node_dim=args.node_dim, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_tasks = args.nclass, num_layer = args.num_layer, node_dim=args.node_dim, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    else:
        raise ValueError('Invalid GNN type')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    # model.load_state_dict(torch.load('/home/sjtu/liushuyu/project/GIN/model_corr/0/0_700.pth'))
    # model.load_state_dict(torch.load('/home/sjtu/liushuyu/project/GIN/model_T1/0/0_1650.pth'))
    model.load_state_dict(torch.load(os.path.join(args.model_path,str(args.fold)+'_best.pth')))
    # model.load_state_dict(torch.load('/home/sjtu/liushuyu/project/GIN/model_T1/gdc_0/0_1000.pth'))


    print('model loaded!')
    
    print('Evaluating...')
    # perf = eval(model, device, train_loader, evaluator)
    # perf = eval(model, device, valid_loader, evaluator)
    # perf = eval(model, device, test_loader, evaluator)
    perf_acc = eval(model, device, test_loader, evaluator_acc)
    perf_auc = eval(model, device, test_loader, evaluator_auc)
    perf_prf = eval(model, device, test_loader, evaluator_prf)

    print(perf_acc['acc'])
    print('Test Acc:{:.5f},\tAuc:{:.5f},\tPrecision:{:.5f},\tRecall:{:.5f},\tF1-score:{:.5f}'
          .format(perf_acc['acc'],perf_auc['rocauc'],perf_prf['precision'],perf_prf['recall'],perf_prf['F1']))

    # print('cm:')
    # print(perf['cm'])
    # print('calssification report:')
    # print(perf['report'])


if __name__ == "__main__":
    main()
