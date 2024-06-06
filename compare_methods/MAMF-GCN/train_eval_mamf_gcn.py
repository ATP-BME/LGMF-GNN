import sklearn.preprocessing
from sklearn.metrics import roc_auc_score,roc_curve,auc
import matplotlib.pyplot as plt
from torch import nn, softmax
from sklearn.metrics import roc_auc_score,roc_curve
from models import MAMFGCN
import torch.nn.functional as F
from PAE import PAE
from opt import *
from utils.metrics import accuracy, auc, prf
from scipy.special import softmax
from data.dataprocess import *

if __name__ == '__main__':
    opt = OptInit().initialize()

    print('  Loading dataset ...')
    dl = dataloader()
    raw_features1,raw_features2, y, nonimg = dl.load_data()
    n_folds = 10
    cv_splits = dl.data_split(n_folds)
    # print(cv_splits)
    global mean_tpr
    global mean_fpr
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    global cnt
    cnt = 0
    corrects = np.zeros(n_folds, dtype=np.int32)
    accs = np.zeros(n_folds, dtype=np.float32)
    aucs = np.zeros(n_folds, dtype=np.float32)
    prfs = np.zeros([n_folds, 3], dtype=np.float32)


    for fold in range(n_folds):
        print("\r\n========================== Fold {} ==========================".format(fold))
        train_ind = cv_splits[fold][0]
        test_ind = cv_splits[fold][1]

        if torch.cuda.is_available():
            torch.cuda.manual_seed(n_folds)

        np.random.seed(n_folds)  # Numpy module.
        random.seed(n_folds)

        print('  Constructing graph data...')
        # extract node features
        node_ftr1,node_ftr2 = dl.get_node_features(train_ind)

        for topk in range(2, 16):
            construct_graph(node_ftr1, topk)
            f1 = open('/media/pjc/expriment/mdd_exam/pjc/EV_GCN-MDD/data/MDD/knn/tmp.txt', 'r')
            f2 = open('/media/pjc/expriment/mdd_exam/pjc/EV_GCN-MDD/data/MDD/knn/c' + str(topk) + '.txt', 'w')
            lines = f1.readlines()
            for line in lines:
                start, end = line.strip('\n').split(' ')
                if int(start) < int(end):
                    f2.write('{} {}\n'.format(start, end))
            f2.close()
        for topk in range(2, 16):
            # 复制后记得改construct_graph里的路径
            construct_graph2(node_ftr2, topk)
            f1 = open('/media/pjc/expriment/mdd_exam/pjc/EV_GCN-MDD/data/MDD/knn_ho2/tmp.txt', 'r')
            f2 = open('/media/pjc/expriment/mdd_exam/pjc/EV_GCN-MDD/data/MDD/knn_ho2/c' + str(topk) + '.txt', 'w')
            lines = f1.readlines()
            for line in lines:
                start, end = line.strip('\n').split(' ')
                if int(start) < int(end):
                    f2.write('{} {}\n'.format(start, end))
            f2.close()

        # get PAE inputs
        edge_index, edgenet_input = dl.get_PAE_inputs(nonimg)

        # normalization for PAE
        edgenet_input = (edgenet_input - edgenet_input.mean(axis=0)) / edgenet_input.std(axis=0)
        edge_net = PAE(2 * nonimg.shape[1] // 2, 0.2)
        edgenet_input = torch.from_numpy(edgenet_input)
        edge_weight = torch.squeeze(edge_net(edgenet_input))

        fadj,fadj2 = load_graph(config=opt.n)
        aij = np.zeros([raw_features1.shape[0], raw_features1.shape[0]])
        for i in range(raw_features1.shape[0]):
            n = edge_index.shape[1]
            for k in range(n):
                if i == edge_index[0][k]:
                    aij[i][edge_index[1][k]] = edge_weight[k]

        sadj =aij

        model = MAMFGCN(nfeat=2000,
                      nhid=32,
                      out=16,
                      nclass=2,
                      nhidlayer=1,
                      dropout=0.4,
                      baseblock="inceptiongcn",
                      inputlayer="gcn",
                      outputlayer="gcn",
                      nbaselayer=6,
                      activation=F.relu,
                      withbn=False,
                      withloop=False,
                      aggrmethod="concat",
                      mixmode=False,
                     )
        model = model.to(opt.device)

        # build loss, optimizer, metric
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)

        features_cuda = torch.as_tensor(node_ftr1, dtype=torch.float32).to(opt.device)
        features_cuda2 = torch.as_tensor(node_ftr2, dtype=torch.float32).to(opt.device)
        edge_index = torch.as_tensor(edge_index, dtype=torch.long).to(opt.device)
        edgenet_input = torch.as_tensor(edgenet_input, dtype=torch.float32).to(opt.device)
        aij = torch.as_tensor(aij, dtype=torch.float32).to(opt.device)
        sadj = torch.as_tensor(sadj, dtype=torch.float32).to(opt.device)
        fadj = torch.as_tensor(fadj, dtype=torch.float32).to(opt.device)
        fadj2 = torch.as_tensor(fadj2, dtype=torch.float32).to(opt.device)
        labels = torch.as_tensor(y, dtype=torch.long).to(opt.device)
        fold_model_path = opt.ckpt_path + "/fold{}.pth".format(fold)


        def plot_embedding(data, label, title):
            plt.figure()
            x_min, x_max = np.min(data, 0), np.max(data, 0)
            data = (data - x_min) / (x_max - x_min)
            p = [[0] for _ in range(10)]
            p2 = [[0] for _ in range(10)]
            for i in range(len(label)):
                if label[i] == 0:
                    p = plt.scatter(data[i, 0], data[i, 1], lw=0.1, c='#FFD700')#, alpha=0.8
                elif label[i] == 1:
                    p2 = plt.scatter(data[i, 0], data[i, 1], lw=0.1, c='#800080')
            plt.legend((p, p2), ('HC', 'MDD'))
            plt.savefig('./draw_figure/MDD/Result{:d}.png'.format(fold), dpi=600)

        def train():
            print("  Number of training samples %d" % len(train_ind))
            print("  Start training...\r\n")
            acc = 0
            for epoch in range(opt.num_iter):
                model.train()
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    node_logits, att, emb1, com1, com2,com3, emb2,emb3 = model(features_cuda, sadj, fadj,fadj2)
                    loss_class = loss_fn(node_logits[train_ind], labels[train_ind])
                    loss_dep = (loss_dependence(emb1, com1, raw_features1.shape[0])
                                + loss_dependence(emb2, com2, raw_features1.shape[0])
                                +loss_dependence(emb3, com3, raw_features1.shape[0])) / 3

                    loss_com = common_loss(com1, com2,com3)
                    loss = loss_class + 1e-12 * loss_dep + 0.00005 * loss_com
                    loss.backward()
                    optimizer.step()
                correct_train, acc_train = accuracy(node_logits[train_ind].detach().cpu().numpy(), y[train_ind])

                model.eval()
                with torch.set_grad_enabled(False):

                    node_logits, att,emb1, com1, com2,com3, emb2,emb3 = model(features_cuda, sadj, fadj,fadj2)

                logits_test = node_logits[test_ind].detach().cpu().numpy()
                correct_test, acc_test = accuracy(logits_test, y[test_ind])
                # pos_probs = softmax(logits_test, axis=1)[:, 1]
                # pos_probs = logits_test[:, 1]
                # fpr,tpr,thresholds =roc_curve(pos_probs, y[test_ind])
                # auc_plot =roc_auc_score(pos_probs, y[test_ind])
                auc_test = auc(logits_test, y[test_ind])
                prf_test = prf(logits_test, y[test_ind])

                # plt.plot(fpr,tpr)
                # plt.title("auc=%.4f"%(auc_plot))
                # plt.xlabel("False Positive Rate")
                # plt.ylabel("True Positive Rate")
                # plt.fill_between(fpr,tpr,where=(tpr>0),color='green',alpha=0.5)
                # plt.show()

                # print("Epoch: {},\tce loss: {:.5f},\ttrain acc: {:.5f}".format(epoch, loss.item(), acc_train.item()))
                if acc_test > acc and epoch > 9:
                    acc = acc_test
                    correct = correct_test
                    aucs[fold] = auc_test
                    prfs[fold] = prf_test
                    if opt.ckpt_path != '':
                        if not os.path.exists(opt.ckpt_path):
                            # print("Checkpoint Directory does not exist! Making directory {}".format(opt.ckpt_path))
                            os.makedirs(opt.ckpt_path)
                        torch.save(model.state_dict(), fold_model_path)

            accs[fold] = acc
            corrects[fold] = correct
            print("\r\n => Fold {} test accuacry {:.5f}".format(fold, acc))



        def evaluate():
            print("  Number of testing samples %d" % len(test_ind))
            print('  Start testing...')
            global cnt
            global mean_tpr
            model.load_state_dict(torch.load(fold_model_path))
            model.eval()
            # node_logits = model(features_cuda, edge_index, edgenet_input)
            node_logits, att, emb1, com1, com2,com3, emb2, emb,emb3 = model(features_cuda, sadj, fadj,fadj2)
            logits_test = node_logits[test_ind].detach().cpu().numpy()
            corrects[fold], accs[fold] = accuracy(logits_test, y[test_ind])
            aucs[fold] = auc(logits_test, y[test_ind])
            prfs[fold] = prf(logits_test, y[test_ind])
            cnt += 1
            pos_probs = softmax(logits_test, axis=1)[:, 1]
            fpr, tpr, thresholds = roc_curve(y[test_ind], pos_probs)
            mean_tpr += np.interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = sklearn.metrics.auc(fpr, tpr)
            lw = 2
            plt.plot(fpr, tpr, lw=lw, label='ROC fold {0:d} curve (area= {1:.2f})'.format(cnt, roc_auc))

            print("  Fold {} test accuracy {:.5f}, AUC {:.5f}".format(fold, accs[fold], aucs[fold]))




        if opt.train == 1:
            train()
        elif opt.train == 0:
            evaluate()

    print("\r\n========================== Finish ==========================")
    n_samples = raw_features1.shape[0]
    acc_nfold = np.sum(corrects) / n_samples
    print("=> Average test accuracy in {}-fold CV: {:.5f}".format(n_folds, acc_nfold))
    print("=> Average test AUC in {}-fold CV: {:.5f}".format(n_folds, np.mean(aucs)))
    se, sp, f1 = np.mean(prfs, axis=0)
    print("=> Average test sensitivity {:.5f}, specificity {:.5f}, F1-score {:.5f}".format(se, sp, f1))

