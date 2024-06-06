import random
import os
import scipy.io as io
from tqdm import tqdm
from weight_avg import *
import option, bold
from model import *
from abide_dataset import *
from mdd_dataset import *

# init
argv = option.parse()

# seed and device
torch.manual_seed(argv.seed)
np.random.seed(argv.seed)
random.seed(argv.seed)
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(argv.seed)
else:
    device = torch.device("cpu")

# calculate accuracy
def acc(Label, Pred):
    true_list = np.array([Label[i] - Pred[i] for i in range(len(Label))])
    true_list[true_list != 0] = 1
    err = true_list.sum()
    acc = (len(Label)-err)/len(Label)
    return acc

# dataset
Data = argv.dataset
if Data == 'MDD':
    # MDD
    train_bold_dir = r'D:\zjhexp\data\MDD\MDD_FL\bold'
    train_info_dir = r'D:\zjhexp\data\MDD\MDD_FL\info'
    bold_file = os.listdir(train_bold_dir)
    print(bold_file)
    info_file = os.listdir(train_info_dir)
    print(info_file)
    site_num = len(bold_file)
    sample_num_list = []
    for num in range(site_num):
        data_path = train_bold_dir+'/'+bold_file[num]
        csv_path = train_info_dir+'/'+info_file[num]
        exec ("dataset%s=Data_MDD(data_path, csv_path, k_fold=argv.k_fold)"%num)
        exec("dataset_test%s=Data_MDD(data_path, csv_path, k_fold=argv.k_fold)" % num)
        exec ("sample_num_list.append(len(dataset%s))"%num)
        exec ("dataloader%s=torch.utils.data.DataLoader(dataset%d, batch_size=argv.minibatch_size, shuffle=False)"%(num,num))
        exec ("dataloader_test%s=torch.utils.data.DataLoader(dataset_test%d, batch_size=1, shuffle=False)" % (num, num))
else:
    # ABIDE
    train_bold_dir = r'D:\zjhexp\data\ABIDE\ABIDE_FL\bold'
    train_info_dir = r'D:\zjhexp\data\ABIDE\ABIDE_FL\info'
    bold_file = os.listdir(train_bold_dir)
    print(bold_file)
    info_file = os.listdir(train_info_dir)
    print(info_file)
    site_num = len(bold_file)
    sample_num_list = []
    for num in range(site_num):
        data_path = train_bold_dir+'/'+bold_file[num]
        csv_path = train_info_dir+'/'+info_file[num]
        exec ("dataset%s=Data_ABIDE(data_path, csv_path, k_fold=argv.k_fold)"%num)
        exec("dataset_test%s=Data_ABIDE(data_path, csv_path, k_fold=argv.k_fold)" % num)
        exec ("sample_num_list.append(len(dataset%s))"%num)
        exec ("dataloader%s=torch.utils.data.DataLoader(dataset%d, batch_size=argv.minibatch_size, shuffle=False)"%(num,num))
        exec ("dataloader_test%s=torch.utils.data.DataLoader(dataset_test%d, batch_size=1, shuffle=False)" % (num, num))

# train
def step(model, p_model, criterion, dyn_a, t, p, label, reg_lambda, gamma, clip_grad=0.0, device='cpu', optimizer=None, p_optimizer=None, scheduler=None, train_people=argv.train_people):
    # model: shared branch, p_model: personalize branch
    # criterion: loss
    # dyn_a: dynamic FCN, t: BOLD timeseries, p: demographics information, label: label
    # reg_lambda: value of lambda, gamma: value of gamma

    if optimizer is None: model.eval(), p_model.eval()
    else: model.train(), p_model.train()

    # run model
    logit,_, attention, latent, reg_ortho = model(dyn_a.to(device)) # batch x window x node x node
    adj = batch_adj(t.permute(1,2,0)) # batch x node x time
    feat = flatten(up_triu(adj))
    logit,_ = p_model(p.to(device), feat.to(device), logit, gamma)
    loss = criterion(logit, label.to(device))
    reg_ortho *= reg_lambda
    loss += reg_ortho

    # optimize model
    if optimizer is not None:
       optimizer.zero_grad()
       if p_optimizer is not None and train_people == True:
           p_optimizer.zero_grad()
       loss.backward()
       if clip_grad > 0.0: torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad)
       optimizer.step()
       if p_optimizer is not None and train_people == True:
           p_optimizer.step()
       if scheduler is not None:
           scheduler.step()

    return logit, loss, attention, latent, reg_ortho

# model_init
# shared branch
model_init = ModelSTAGIN(
    input_dim=116,
    hidden_dim=argv.hidden_dim,
    num_classes=2,
    num_heads=argv.num_heads,
    num_layers=argv.num_layers,
    sparsity=argv.sparsity,
    dropout=argv.dropout,
    cls_token=argv.cls_token,
    readout=argv.readout)
parameter_init = model_init.state_dict()
parameter_init_fix = model_init.state_dict()
# personalize branch
p_model_init = Population(3, 6670, argv.ph, argv.hidden_dim, argv.ph, argv.Type, 2)
p_parameter_init = [p_model_init.state_dict()]*site_num
p_parameter_init_fix = [p_model_init.state_dict()]*site_num

# fl train
R = 0.0
for k in range(argv.k_fold):
    R_k = []
    lr = argv.lr
    for i in range(site_num):
        exec ('L%s=[]'%i)

    # train
    for iter_num in range(argv.num_iters):
        para_list = []
        if argv.train_people: p_para_list=[]
        for site in range(site_num):
            exec('dataset%s.set_fold(k,train=True)'%site)
            exec ("dataloader=dataloader%s"%site)
            model = ModelSTAGIN(
                input_dim=116,
                hidden_dim=argv.hidden_dim,
                num_classes=2,
                num_heads=argv.num_heads,
                num_layers=argv.num_layers,
                sparsity=argv.sparsity,
                dropout=argv.dropout,
                cls_token=argv.cls_token,
                readout=argv.readout) # shared branch
            model.to(device)
            model.load_state_dict(parameter_init) # read server-side parameters
            p_model = Population(3, 6670, argv.ph, argv.hidden_dim, argv.ph, argv.Type, 2) # personalized branch
            p_model.to(device)
            p_model.load_state_dict(p_parameter_init[site]) # read local parameters

            criterion = torch.nn.CrossEntropyLoss() # CrossEntropyLoss
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            if argv.train_people: optimizer_p = torch.optim.Adam(p_model.parameters(), lr=lr)
            else: optimizer_p=None

            for epoch in range(argv.num_epochs):
                loss_accumulate = 0.0
                reg_ortho_accumulate = 0.0
                Label = []
                Pred = []
                for i, x in enumerate(tqdm(dataloader, ncols=80, desc=f'k:{k},i:{iter_num},s:{site},e:{epoch}')):
                    dyn_a, sampling_points = bold.process_dynamic_fc(x['timeseries'], argv.window_size,
                                                                     argv.window_stride,
                                                                     argv.dynamic_length) # generate dynamic FCN
                    t = x['timeseries'].permute(1, 0, 2) # time x batch x node
                    p = x['people']
                    label = x['label']
                    Label.extend(label.tolist())
                    logit, loss, attention, latent, reg_ortho = step(
                        model=model,
                        p_model=p_model,
                        criterion=criterion,
                        dyn_a=dyn_a,
                        t=t,
                        p=p,
                        label=label,
                        reg_lambda=argv.reg_lambda,
                        gamma=argv.gamma,
                        clip_grad=argv.clip_grad,
                        device=device,
                        optimizer=optimizer,
                        p_optimizer=optimizer_p,
                        train_people=argv.train_people)
                    pred = logit.argmax(1)
                    prob = logit.softmax(1)
                    Pred.extend(pred.tolist())
                    loss_accumulate += loss.detach().cpu().numpy() # cross-entropy loss
                    reg_ortho_accumulate += reg_ortho.detach().cpu().numpy() # orthogonality loss
                exec ('L%s.append(loss_accumulate)'%site)
                Acc = acc(Label,Pred)
                print('acc:',Acc)

            # save model
            if iter_num == argv.num_iters-1:
                torch.save(model.state_dict(), argv.save_root_path + '/' + 'site' + str(site) + '_fold' + str(k) + '_model.pth')
                torch.save(p_model.state_dict(), argv.save_root_path + '/' + 'site' + str(site) + '_fold' + str(k) + '_p_model.pth')

            para_list.append(model.state_dict())
            if argv.train_people: p_para_list.append(p_model.state_dict())

        # federated aggregation
        parameter_init = W_A(para_list, sample_num_list) # only shared branch participate in parameter aggregation
        if argv.train_people: p_parameter_init = p_para_list # personalized branch does not participate in parameter aggregation

        # test
        for site in range(site_num):
            exec('dataset_test%s.set_fold(k,train=False)' % site)
            exec("dataloader_test=dataloader_test%s"%site)
            model = ModelSTAGIN(
                input_dim=116,
                hidden_dim=argv.hidden_dim,
                num_classes=2,
                num_heads=argv.num_heads,
                num_layers=argv.num_layers,
                sparsity=argv.sparsity,
                dropout=argv.dropout,
                cls_token=argv.cls_token,
                readout=argv.readout)
            model.to(device)
            model.load_state_dict(parameter_init)
            p_model = Population(3, 6670, argv.ph, argv.hidden_dim, argv.ph, argv.Type, 2)
            p_model.to(device)
            p_model.load_state_dict(p_parameter_init[site])
            model.eval()
            p_model.eval()
            criterion = torch.nn.CrossEntropyLoss()
            Label = []
            Pred = []
            if iter_num==argv.num_iters-1: Prob = []
            for i, x in enumerate(tqdm(dataloader_test, ncols=80, desc=f'k:{k},i:{iter_num},s:{site}')):
                dyn_a, sampling_points = bold.process_dynamic_fc(x['timeseries'], argv.window_size,
                                                                 argv.window_stride,
                                                                 argv.dynamic_length)
                t = x['timeseries'].permute(1, 0, 2)
                p = x['people']
                label = x['label']
                Label.append(label.item())
                logit, loss, attention, latent, reg_ortho = step(
                    model=model,
                    p_model=p_model,
                    criterion=criterion,
                    dyn_a=dyn_a,
                    t=t,
                    p=p,
                    label=label,
                    reg_lambda=argv.reg_lambda,
                    gamma=argv.gamma,
                    clip_grad=argv.clip_grad,
                    device=device,
                    optimizer=None)
                pred = logit.argmax(1)
                prob = logit.softmax(1)
                Pred.append(pred.item())
                if iter_num==argv.num_iters-1: Prob.append(prob.tolist())
            Acc = acc(Label, Pred)
            print('k:', k, 'i:', iter_num, 's:', site, 'result:')
            print(Acc)
            R_k.append(Acc)

            # save results
            if iter_num==argv.num_iters-1:
                Label = np.array(Label)
                Pred = np.array(Pred)
                Prob = torch.tensor(Prob).squeeze(1).numpy()
                np.save(argv.save_root_path + '/' + 'site' + str(site) + '_fold' + str(k) + '_Label', Label)
                np.save(argv.save_root_path + '/' + 'site' + str(site) + '_fold' + str(k) + '_Pred', Pred)
                np.save(argv.save_root_path + '/' + 'site' + str(site) + '_fold' + str(k) + '_Prob', Prob)

    # save fold results
    R_k = np.array(R_k).reshape(argv.num_iters, site_num)
    io.savemat(argv.save_root_path+'/'+'fold'+str(k)+'_result.mat', {'acc': R_k})

    # save loss
    for i in range(site_num):
        exec ('L%s=np.array(L%d)'%(i,i))
        exec ("np.save(argv.save_root_path + '/' + 'site' + str(i) + '_fold' + str(k) + '_Loss', L%s)"%i)

    print('fold ',str(k),'result:')
    print(R_k)
    R += R_k
    parameter_init = parameter_init_fix
    p_parameter_init = p_parameter_init_fix

print('final result:')
print(R/argv.k_fold)

# save final result
io.savemat(argv.save_root_path+'/'+'final_result.mat', {'acc': R/argv.k_fold})
