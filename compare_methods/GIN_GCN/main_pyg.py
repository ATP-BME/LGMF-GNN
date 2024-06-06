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
# from torch.utils.data import ConcatDataset

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
            if evaluator.eval_metric == 'acc':
                pred = torch.tensor(torch.sigmoid(pred)>0.5,dtype=batch.y.dtype).view(y_true[-1].shape).detach().cpu()
                y_pred.append(pred)
            else:
                y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)

##### count bad data########
def dict_to_csv(error_dict,label_dict,csv_path='./error.csv'):
    with open(csv_path, 'w') as f:
        for key in error_dict.keys():
            f.write("%s,%s,%s\n"%('sub-'+str(key).zfill(4),error_dict[key],label_dict[key]))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=2,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--eval_metric', type=str, default='acc',
                        help='acc, rocauc, F1,ap,rmse')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=3,
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
    
    parser.add_argument('--epochs', type=int, default=120,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    # parser.add_argument('--dataroot', type=str, default='/data0/lsy/SRPBS_arrange/ALFF_ROI/', 
    # parser.add_argument('--dataroot', type=str, default='/data0/lsy/SRPBS_new/BrainGNN/', # SRPBS CC200
    # parser.add_argument('--dataroot', type=str, default='/data0/lsy/SRPBS_new/ROISignal_AAL/',  # SRPBS AAL
    # parser.add_argument('--dataroot', type=str, default='/data0/lsy/REST_meta_MDD/ROISignals_FunImgARCWF/',
    parser.add_argument('--dataroot', type=str, default='/data0/lsy/openneuro/processed/', # opn AAL C200
    # parser.add_argument('--dataroot', type=str, default='/data0/lsy/Anding1/ROI_signal/', # anding AAL
    # parser.add_argument('--dataroot', type=str, default='/data0/lsy/Anding1/ROI_signal_CC200/', # anding CC200
                        help='root directory of the dataset')
    parser.add_argument('--nclass', type=int, default=1, 
                        help='num of classes (the dim of output logits)')
    parser.add_argument('--fold', type=int, default=0, 
                        help='training which fold')
    parser.add_argument('--model_path', type=str, default='./compare_openneuro/CC200', 
                        help='path to save checkpoint')
    parser.add_argument('--log_path', type=str, default='./compare_openneuro/CC200', 
                        help='path to save tensorboard log')

    


    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    args = parser.parse_args()
    

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### init tensorboard
    writer = SummaryWriter(os.path.join(args.log_path,'gdc_'+str(args.fold)))

    args.model_path = os.path.join(args.model_path,'{}3_gdc_{}'.format(args.gnn,str(args.fold)))
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    print(args)
    ### automatic dataloading and splitting
    # dataset = PygGraphPropPredDataset(name = args.dataset)

    # custom dataset
    name = 'tmp'
    # dataset = ABIDEDataset(args.dataroot,name,raw_folder='train_T1')
    # dataset = ABIDEDataset(args.dataroot,name,raw_folder='raw_corr_same_protocol_train')
    # dataset = ABIDEDataset(args.dataroot,name,raw_folder='raw_ALFF_same_protocol_train')

    if 'SRPBS' in args.dataroot:
        # dataset = ABIDEDataset(args.dataroot,name,raw_folder='SRPBS') # SRPBS CC200
        dataset = ABIDEDataset(args.dataroot,name,raw_folder='raw') # SRPBS AAL
    elif 'REST' in args.dataroot:
        dataset = ABIDEDataset(args.dataroot,name,raw_folder='raw_aal') # REST raw_cc200 raw_aal
    else:
        assert 'openneuro' in args.dataroot or 'Anding' in args.dataroot, 'wrong dataroot'
        ### AAL
        # dataset_SRPBS = ABIDEDataset('/data0/lsy/SRPBS_new/ROISignal_AAL/',name,raw_folder='raw') # SRPBS AAL
        # dataset_REST = ABIDEDataset('/data0/lsy/REST_meta_MDD/ROISignals_FunImgARCWF/',name,raw_folder='raw_aal') # REST raw_cc200 raw_aal
        # dataset_HBN = ABIDEDataset(args.dataroot,name,raw_folder='raw') # openneuro raw_aal anding raw
        # # dataset = ConcatDataset([dataset_SRPBS, dataset_REST,dataset_HBN])
        # dataset = MergedDataset([dataset_SRPBS, dataset_REST, dataset_HBN])

        ### CC200
        dataset_SRPBS = ABIDEDataset('/data0/lsy/SRPBS_new/ROI_signal_extract_cc200/',name,raw_folder='raw') # SRPBS CC200
        dataset_REST = ABIDEDataset('/data0/lsy/REST_meta_MDD/ROISignals_FunImgARCWF/',name,raw_folder='raw_cc200') # REST raw_cc200
        dataset_HBN = ABIDEDataset(args.dataroot,name,raw_folder='raw_cc200') # openneuro raw_cc200 anding raw
        # dataset = ConcatDataset([dataset_SRPBS, dataset_REST,dataset_HBN])
        dataset = MergedDataset([dataset_SRPBS, dataset_REST, dataset_HBN])



    dataset.data.y = dataset.data.y.squeeze()
    dataset.data.x[dataset.data.x == float('inf')] = 0
    dataset.data.y[dataset.data.y == 2] = 1

    # # dataset_test = ABIDEDataset(args.dataroot,name,raw_folder='test_T1')
    # dataset_test = ABIDEDataset(args.dataroot,name,raw_folder='raw_corr_same_protocol_test')
    # dataset_test = ABIDEDataset(args.dataroot,name,raw_folder='raw_ALFF_same_protocol_test')
    # dataset_test = ABIDEDataset(args.dataroot,name,raw_folder='raw_corr_T1_same_protocol_test')
    # dataset_test.data.y = dataset_test.data.y.squeeze()
    # dataset_test.data.x[dataset_test.data.x == float('inf')] = 0
    print('dataset length:',len(dataset.data.y))

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
    evaluator = Evaluator(num_tasks=args.nclass,eval_metric='acc')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)


    # error_dict = {}
    # label_dict = {}
    # for sub,label in zip(train_dataset.data.sub_id.numpy(),train_dataset.data.y.numpy()):
    #     error_dict[str(sub)] = 0
    #     label_dict[str(sub)] = label
    # # record test subjects
    # txt_file = open('./data/all_test.txt','w',encoding='utf-8')
    # # for sub in test_dataset.data.sub_id.numpy()[te_index]:
    # for sub in test_dataset.data.sub_id.numpy():
    #     sub = 'sub_'+str(sub).zfill(4)
    #     txt_file.write(sub+'\n')
    # txt_file.close()

    if args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', num_tasks = args.nclass, num_layer = args.num_layer, node_dim=args.node_dim, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
        print('model selected: GIN')
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_tasks = args.nclass, num_layer = args.num_layer, node_dim=args.node_dim, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type = 'gcn', num_tasks = args.nclass, num_layer = args.num_layer, node_dim=args.node_dim, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
        print('model selected: GCN')
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_tasks = args.nclass, num_layer = args.num_layer, node_dim=args.node_dim, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    else:
        raise ValueError('Invalid GNN type')

    print(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)


    valid_curve = []
    test_curve = []
    train_curve = []
    best_val_acc = 0

    for epoch in range(1, args.epochs + 1):
        scheduler.step()
        for param_group in optimizer.param_groups:
            current_lr=param_group['lr']
        print("=====Epoch {}".format(epoch))
        print('Training...')
        print("LR:", current_lr)
        writer.add_scalars('Lr',{'learning_rate':current_lr}, epoch)

        train(model, device, train_loader, optimizer, dataset.task_type,epoch,writer)

        print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        train_curve.append(train_perf[args.eval_metric])
        valid_curve.append(valid_perf[args.eval_metric])
        test_curve.append(test_perf[args.eval_metric])
        writer.add_scalars(format(args.eval_metric),{'train':train_perf[args.eval_metric],'val':valid_perf[args.eval_metric],'test':test_perf[args.eval_metric]}, epoch)
        # writer.add_scalars('train/{}'.format(args.eval_metric),train_perf, epoch)

        ### save checkpoint
        # if epoch % 50 == 0 and epoch > 0:
        #     torch.save(model.state_dict(), os.path.join(args.model_path,str(args.fold)+'_{}.pth'.format(epoch)))

        if valid_perf[args.eval_metric] > best_val_acc:
            best_val_acc = valid_perf[args.eval_metric]
            torch.save(model.state_dict(),os.path.join(args.model_path,str(args.fold)+'_best.pth'))



    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))
    print('Best train:{}'.format(best_train))
    print('Best val epoch:{}'.format(best_val_epoch+1))

    if not args.filename == '':
        torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch], 'Train': train_curve[best_val_epoch], 'BestTrain': best_train}, args.filename)


if __name__ == "__main__":
    main()
