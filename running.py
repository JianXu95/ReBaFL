
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import tools
import numpy as np
import copy
import time
import math
import json
import tools
from tools import average_weights, average_weights_weighted, agg_classifier_weighted_p, average_weights_server_rate
from tools import get_head_agg_weight_softmax, get_head_agg_weight, get_head_agg_weight_softmax_truncated, get_global_head_agg_weight
from tools import Nearest_Neighbor_Mixing

def one_round_training(rule):
    # gradient aggregation rule
    Train_Round = {
                   'FedAvg':train_round_fedavg,
                   'BSMFedAvg':train_round_fedavg,
                   'FedRS':train_round_fedavg,
                   'FedLC':train_round_fedavg,
                   'FedProx':train_round_fedprox,
                   'Scaffold':train_round_scaffold,
                   'FedDyn':train_round_feddyn,
                   'MOON':train_round_fedavg,
                   'Local':train_round_standalone,
                   'ReBaFL':train_round_pfedproto,
                   'FedBABU':train_round_fedavg,
    }
    return Train_Round[rule]

def one_round_tuning(rule):
    # gradient aggregation rule
    Tune_Round = { 'Local':tune_round_fedavg,
                   'FedAvg':tune_round_fedavg,
                   'BSMFedAvg':tune_round_fedavg,
    }
    return Tune_Round[rule]

## training methods -------------------------------------------------------------------
# local training only
def train_round_standalone(args, global_model, local_clients, rnd, **kwargs):
    print(f'\n---- Global Communication Round : {rnd+1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    if (rnd >= args.epochs):
        m = num_users
    idx_users = np.random.choice(range(num_users), m, replace=False)
    idx_users = sorted(idx_users)
    local_weights, local_losses1, local_losses2 = [], [], []
    local_acc1 = []
    local_acc2 = []
    protos = []
    for idx in idx_users:
        local_client = local_clients[idx]
        local_epoch = args.local_epoch
        w, gc, loss1, loss2, acc1, acc2 = local_client.local_training(local_epoch=local_epoch)
        local_weights.append(copy.deepcopy(w))
        local_losses1.append(copy.deepcopy(loss1))
        local_losses2.append(copy.deepcopy(loss2))
        local_acc1.append(acc1)
        local_acc2.append(acc2)

    # get global weights
    global_weight = average_weights(local_weights)
    # update global model
    global_model.load_state_dict(global_weight)

    loss_avg1 = sum(local_losses1) / len(local_losses1)
    loss_avg2 = sum(local_losses2) / len(local_losses2)
    acc_avg1 = sum(local_acc1) / len(local_acc1)
    acc_avg2 = sum(local_acc2) / len(local_acc2)

    return loss_avg1, loss_avg2, acc_avg1, acc_avg2, protos


# vanila FedAvg
def train_round_fedavg(args, global_model, local_clients, rnd, metadata=None, **kwargs):
    print(f'\n---- Global Communication Round : {rnd+1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    if (rnd >= args.epochs):
        m = num_users
    # idx_users = np.random.choice(range(num_users), m, replace=False)
    # rates = [0.5, 0.5, 0.3, 0.3, 0.2]
    # rates = [0.5, 0.5, 0.5, 0.5, 0.5]
    # frac = np.array([rates[i%5] for i in range(num_users)])
    # idx_users = np.where((np.random.rand(num_users) <= frac))[0]
    if rnd % args.straggling == 0:
        idx_users = np.where(np.random.rand(num_users) < args.frac)[0]
        while len(idx_users)<1:
            idx_users = np.where(np.random.rand(num_users) < args.frac)[0]
        args.user_list = idx_users
    else:
        idx_users = args.user_list
    # idx_users = np.where(np.random.binomial(size=num_users, n=1, p=args.frac)>0)[0]
    idx_users = sorted(idx_users)
    local_weights, local_losses1, local_losses2 = [], [], []
    local_acc1 = []
    local_acc2 = []
    agg_weight = []
    protos = []

    if rnd < args.epochs:
        global_weight = global_model.state_dict()
        for idx in idx_users:
            local_client = local_clients[idx]
            agg_weight.append(local_client.agg_weight)
            local_epoch = args.local_epoch
            local_client.update_local_model(global_weight=copy.deepcopy(global_weight))
            w, gc, loss1, loss2, acc1, acc2 = local_client.local_training(local_epoch=local_epoch, round=rnd)
            local_weights.append(copy.deepcopy(w))
            local_losses1.append(copy.deepcopy(loss1))
            local_losses2.append(copy.deepcopy(loss2))
            local_acc1.append(acc1)
            local_acc2.append(acc2)

        # var = tools.variance_weights(local_weights)
        # print("Weight Variance: {}".format(var))

        # get global weights
        global_weight = average_weights_weighted(local_weights, agg_weight)
        # update global model
        global_model.load_state_dict(global_weight)
        fine_tune = args.ft
        if metadata is not None and fine_tune:
            global_weight = model_finetune(args.device, model=copy.deepcopy(global_model), dataset=metadata)
            global_model.load_state_dict(copy.deepcopy(global_weight))
        # distribute new model param
        for idx in range(num_users):
            local_client = local_clients[idx]
            local_client.update_local_model(global_weight=copy.deepcopy(global_weight))
    else:
        if rnd == args.epochs:
            global_weight = global_model.state_dict()
            for idx in idx_users:
                local_client = local_clients[idx]
                local_client.update_local_model(global_weight=copy.deepcopy(global_weight))

        for idx in idx_users:
            local_client = local_clients[idx]
            local_epoch = args.local_epoch
            w, loss1, loss2, acc1, acc2 = local_client.local_fine_tuning(local_epoch=local_epoch, round=rnd)
            local_losses1.append(copy.deepcopy(loss1))
            local_losses2.append(copy.deepcopy(loss2))
            local_acc1.append(acc1)
            local_acc2.append(acc2)

    loss_avg1 = sum(local_losses1) / len(local_losses1)
    loss_avg2 = sum(local_losses2) / len(local_losses2)
    acc_avg1 = sum(local_acc1) / len(local_acc1)
    acc_avg2 = sum(local_acc2) / len(local_acc2)

    return loss_avg1, loss_avg2, acc_avg1, acc_avg2, protos

# FedProx, regularization
def train_round_fedprox(args, global_model, local_clients, rnd, **kwargs):
    print(f'\n---- Global Communication Round : {rnd+1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    if (rnd >= args.epochs):
        m = num_users
    # idx_users = np.random.choice(range(num_users), m, replace=False)
    # rates = [0.5, 0.5, 0.3, 0.3, 0.2]
    # # rates = [0.5, 0.5, 0.5, 0.5, 0.5]
    # frac = np.array([rates[i%5] for i in range(num_users)])
    # idx_users = np.where((np.random.rand(num_users) <= frac))[0]
    if rnd % args.straggling == 0:
        idx_users = np.where(np.random.rand(num_users) < args.frac)[0]
        while len(idx_users)<1:
            idx_users = np.where(np.random.rand(num_users) < args.frac)[0]
        args.user_list = idx_users
    else:
        idx_users = args.user_list
    idx_users = sorted(idx_users)
    local_weights, local_losses1, local_losses2 = [], [], []
    local_acc1 = []
    local_acc2 = []
    protos = []

    global_weight = global_model.state_dict()
    
    for idx in idx_users:
        local_client = local_clients[idx]
        local_epoch = args.local_epoch
        local_client.update_local_model(global_weight=copy.deepcopy(global_weight))
        w, gc, loss1, loss2, acc1, acc2 = local_client.local_training(local_epoch=local_epoch)
        local_weights.append(copy.deepcopy(w))
        local_losses1.append(copy.deepcopy(loss1))
        local_losses2.append(copy.deepcopy(loss2))
        local_acc1.append(acc1)
        local_acc2.append(acc2)

    # get global weights
    global_weight = average_weights(local_weights)
    # update global model
    global_model.load_state_dict(global_weight)

    loss_avg1 = sum(local_losses1) / len(local_losses1)
    loss_avg2 = sum(local_losses2) / len(local_losses2)
    acc_avg1 = sum(local_acc1) / len(local_acc1)
    acc_avg2 = sum(local_acc2) / len(local_acc2)

    return loss_avg1, loss_avg2, acc_avg1, acc_avg2, protos


# FedDyn, regularization
def train_round_feddyn(args, global_model, local_clients, rnd, **kwargs):
    print(f'\n---- Global Communication Round : {rnd+1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    if (rnd >= args.epochs):
        m = num_users
    # idx_users = np.random.choice(range(num_users), m, replace=False)
    # rates = [0.5, 0.5, 0.3, 0.3, 0.2]
    # # rates = [0.5, 0.5, 0.5, 0.5, 0.5]
    # frac = np.array([rates[i%5] for i in range(num_users)])
    # idx_users = np.where((np.random.rand(num_users) <= frac))[0]
    if rnd % args.straggling == 0:
        idx_users = np.where(np.random.rand(num_users) < args.frac)[0]
        while len(idx_users)<1:
            idx_users = np.where(np.random.rand(num_users) < args.frac)[0]
        args.user_list = idx_users
    else:
        idx_users = args.user_list
    m = len(idx_users)
    idx_users = sorted(idx_users)
    local_weights, local_losses1, local_losses2 = [], [], []
    local_acc1 = []
    local_acc2 = []
    protos = []

    global_weight = copy.deepcopy(global_model.state_dict())
    lr = args.lr#*0.998**(rnd)
    
    for idx in idx_users:
        local_client = local_clients[idx]
        local_epoch = args.local_epoch
        local_client.update_local_model(global_weight=copy.deepcopy(global_weight))
        w, gc, loss1, loss2, acc1, acc2 = local_client.local_training(local_epoch=local_epoch, local_lr=lr)
        local_weights.append(copy.deepcopy(w))
        local_losses1.append(copy.deepcopy(loss1))
        local_losses2.append(copy.deepcopy(loss2))
        local_acc1.append(acc1)
        local_acc2.append(acc2)
    alpha = local_clients[idx_users[0]].alpha
    ht = copy.deepcopy(local_clients[idx_users[0]].ht)

    # get global weights
    global_weight_new = average_weights(local_weights)
    for key in global_weight.keys():
        ht[key] = ht[key] - m/num_users*(global_weight_new[key]-global_weight[key]).clone().detach()
        global_weight_new[key] = global_weight_new[key] - ht[key].clone().detach()
    # update global model
    global_model.load_state_dict(global_weight_new)
    for idx in range(num_users):
        local_client = local_clients[idx]
        local_client.ht = copy.deepcopy(ht)

    loss_avg1 = sum(local_losses1) / len(local_losses1)
    loss_avg2 = sum(local_losses2) / len(local_losses2)
    acc_avg1 = sum(local_acc1) / len(local_acc1)
    acc_avg2 = sum(local_acc2) / len(local_acc2)

    return loss_avg1, loss_avg2, acc_avg1, acc_avg2, protos


# Scaffold, drift control / debias
def train_round_scaffold(args, global_model, local_clients, rnd, **kwargs):
    print(f'\n---- Global Communication Round : {rnd+1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    if (rnd >= args.epochs):
        m = num_users
    # idx_users = np.random.choice(range(num_users), m, replace=False)
    # rates = [0.5, 0.5, 0.3, 0.3, 0.2]
    # # rates = [0.5, 0.5, 0.5, 0.5, 0.5]
    # frac = np.array([rates[i%5] for i in range(num_users)])
    # idx_users = np.where((np.random.rand(num_users) <= frac))[0]
    if rnd % args.straggling == 0:
        idx_users = np.where(np.random.rand(num_users) < args.frac)[0]
        while len(idx_users)<1:
            idx_users = np.where(np.random.rand(num_users) < args.frac)[0]
        args.user_list = idx_users
    else:
        idx_users = args.user_list
    idx_users = sorted(idx_users)
    delta_weights, local_losses1, local_losses2 = [], [], []
    delta_grads = []
    local_acc1 = []
    local_acc2 = []
    protos = []
    
    for idx in idx_users:
        local_client = local_clients[idx]
        local_epoch = args.local_epoch
        delta_w, delta_ci, loss1, loss2, acc1, acc2 = local_client.local_training(local_epoch=local_epoch)
        delta_weights.append(copy.deepcopy(delta_w))
        delta_grads.append(copy.deepcopy(delta_ci))
        local_losses1.append(loss1)
        local_losses2.append(loss2)
        local_acc1.append(acc1)
        local_acc2.append(acc2)

    # get global weights
    delta_w_global = torch.vstack(delta_weights).mean(dim=0)
    global_weights_new = tools.get_parameter_values(global_model) + delta_w_global
    # update global model
    tools.set_parameter_values(global_model, global_weights_new)
    # update global control variate
    delta_c_global = torch.vstack(delta_grads).mean(dim=0)
    # delta_global_c = (m/num_users)*delta_c_global
    delta_global_c = delta_c_global

    global_weight = global_model.state_dict()
    for idx in range(num_users):
        local_client = local_clients[idx]
        local_client.update_local_model(global_weight=copy.deepcopy(global_weight), delta_global_c=copy.deepcopy(delta_global_c))

    loss_avg1 = sum(local_losses1) / len(local_losses1)
    loss_avg2 = sum(local_losses2) / len(local_losses2)
    acc_avg1 = sum(local_acc1) / len(local_acc1)
    acc_avg2 = sum(local_acc2) / len(local_acc2)

    return loss_avg1, loss_avg2, acc_avg1, acc_avg2, protos


def train_round_pfedproto(args, global_model, local_clients, rnd, metadata=None,**kwargs):
    print(f'\n---- Global Communication Round : {rnd+1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    if (rnd >= args.epochs):
        m = num_users
    # idx_users = np.random.choice(range(num_users), m, replace=False)
    # rates = [0.5, 0.5, 0.3, 0.3, 0.2]
    # rates = [0.5, 0.5, 0.5, 0.5, 0.5]
    # frac = np.array([rates[i%5] for i in range(num_users)])
    if rnd % args.straggling == 0:
        idx_users = np.where(np.random.rand(num_users) < args.frac)[0]
        # idx_users = np.where((np.random.rand(num_users) <= frac))[0]
        while len(idx_users)<1:
            idx_users = np.where(np.random.rand(num_users) < args.frac)[0]
        args.user_list = idx_users
    else:
        idx_users = args.user_list
    m = len(idx_users)
    idx_users = sorted(idx_users)
    local_weights, local_losses1, local_losses2 = [], [], []
    local_acc1 = []
    local_acc2 = []
    local_cls = []
    local_protos = []
    local_sizes = []
    local_py = []
    agg_weight = []
    local_models = []
    protos = []
    local_finetune_params = []

    global_weight = global_model.state_dict()
    if rnd < (args.epochs):
        for idx in idx_users:
            local_client = local_clients[idx]
            local_client.update_local_model(global_weight=copy.deepcopy(global_weight))
            local_sizes.append(local_client.size_class)
            local_py.append(local_client.Py)
            local_epoch = args.local_epoch
            w, gc, loss1, loss2, acc1, acc2, g_param, protos = local_client.local_training(local_epoch=local_epoch, round=rnd)
            local_models.append(copy.deepcopy(w))
            local_weights.append(copy.deepcopy(w.state_dict()))
            local_losses1.append(loss1)
            local_losses2.append(loss2)
            local_acc1.append(acc1)
            local_acc2.append(acc2)
            local_cls.append(copy.deepcopy(g_param))
            local_protos.append(copy.deepcopy(protos))
            agg_weight.append(local_client.agg_weight)

        # update global prototype
        if args.agg_g:
            agg_weight = tools.adaptive_reweighting(agg_weight, local_py, lam=args.mu)
        # global_protos = local_clients[0].global_protos
        global_protos = tools.protos_aggregation(local_protos, local_sizes)
        
        # var = tools.variance_weights(local_weights)
        # get global weights
        # local_weights = Nearest_Neighbor_Mixing(local_weights, 5)
        global_weight_new = average_weights_weighted(local_weights, agg_weight)

        # update global model
        global_model.load_state_dict(global_weight_new)

        fine_tune = args.ft
        if metadata is not None and fine_tune:
            global_weight = model_finetune(args.device, model=copy.deepcopy(global_model), dataset=metadata)
            global_model.load_state_dict(copy.deepcopy(global_weight))
        
        for idx in range(num_users):
            local_client = local_clients[idx]
            local_client.update_local_model(global_weight=global_weight_new, global_protos=global_protos)

    
    else:
        for idx in range(num_users):
            local_client = local_clients[idx]
            local_client.update_local_model(global_weight=global_weight)
            local_epoch = args.local_epoch
            w, gc, loss1, loss2, acc1, acc2, params = local_client.local_fine_tuning(local_epoch=local_epoch, round=rnd)
            local_losses1.append(copy.deepcopy(loss1))
            local_losses2.append(copy.deepcopy(loss2))
            local_acc1.append(acc1)
            local_acc2.append(acc2)
            local_finetune_params.append(params)


    loss_avg1 = sum(local_losses1) / len(local_losses1)
    loss_avg2 = sum(local_losses2) / len(local_losses2)
    acc_avg1 = sum(local_acc1) / len(local_acc1)
    acc_avg2 = sum(local_acc2) / len(local_acc2)

    return loss_avg1, loss_avg2, acc_avg1, acc_avg2, protos


## local tuning for methods ----------------------------------------------------------
def tune_round_fedavg(args, global_model, local_clients, rnd, **kwargs):
        print(f'\n---- Global Communication Round : {rnd+1} ----')
        num_users = args.num_users
        m = num_users
        idx_users = np.random.choice(range(num_users), m, replace=False)
        idx_users = sorted(idx_users)
        local_losses1, local_losses2 = [], []
        local_acc1 = []
        local_acc2 = []
        local_weights = []
        feature_embed_list = []
        label_list = []
        
        for idx in idx_users:
            local_client = local_clients[idx]
            local_epoch = args.local_epoch
            w, loss1, loss2, acc1, acc2 = local_client.local_fine_tuning(local_epoch=local_epoch, round=rnd)
            local_weights.append(copy.deepcopy(w))
            local_losses1.append(copy.deepcopy(loss1))
            local_losses2.append(copy.deepcopy(loss2))
            local_acc1.append(acc1)
            local_acc2.append(acc2)
        
        print("Client Accuracy:", local_acc1)

        if args.agg_g:
            for idx in idx_users:
                local_client = local_clients[idx]
                avg_weight = [1.0 for i in range(num_users)]
                new_cls = agg_classifier_weighted_p(local_weights, avg_weight, local_client.w_local_keys, idx)
                local_client.update_local_model(new_cls)

        loss_avg1 = sum(local_losses1) / len(local_losses1)
        loss_avg2 = sum(local_losses2) / len(local_losses2)
        acc_avg1 = sum(local_acc1) / len(local_acc1)
        acc_avg2 = sum(local_acc2) / len(local_acc2)

        return loss_avg1, loss_avg2, acc_avg1, acc_avg2, feature_embed_list, label_list

# define model testing function
def test_classification(device, model, test_loader):
    model.eval()
    correct = 0
    total = len(test_loader.dataset)
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)  # pred = output.max(1, keepdim=True)[1]
            # total += labels.size(0)                    # pred.eq(target.view_as(pred)).sum().item()
            correct += (predicted == labels).sum().item()
    acc = 100.0*correct/total
    return acc

def test_classification_ensemble(device, clients, test_loader):
    correct = 0
    total = len(test_loader.dataset)
    labels = test_loader.dataset.targets
    labels = torch.tensor(labels)
    preds = []
    for client in clients:
        pred_i = client.local_test_prob(test_loader)
        preds.append(torch.from_numpy(pred_i))
    preds_esn = sum(preds)
    predicted = torch.max(preds_esn.data, dim=1)[1]
    correct = (predicted == labels).sum().item()
    acc = 100.0*correct/total
    return acc


def model_finetune(device, model, dataset):
    epoch_classifier = 5
    model.train()
    global_params = [p for name, p in model.named_parameters() if name not in model.classifier_weight_keys]
    local_params = [p for name, p in model.named_parameters() if name in model.classifier_weight_keys]
    optimizer = torch.optim.SGD([{'params': global_params, 'lr':0, 'name': "global"},
                                        {'params': local_params, 'lr': 0.01, "name": "local"}],
                                        momentum=0.5, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss()
    for ep in range(epoch_classifier):
        # local training for 1 epoch
        data_loader = iter(dataset)
        iter_num = len(data_loader)
        # start = time.time()
        iter_loss = []
        for it in range(iter_num):
            images, labels = next(data_loader)
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            _, output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            iter_loss.append(loss.item())

    return model.state_dict()


def test_weighted_accuracy(args, weight, model, test_loader):
    device = args.device
    cls_num = args.num_classes
    model.eval()
    correct = [0 for i in range(cls_num)]
    total = [0 for i in range(cls_num)]
    acc = [0 for i in range(cls_num)]
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)  # pred = output.max(1, keepdim=True)[1]
            # total += labels.size(0)                    # pred.eq(target.view_as(pred)).sum().item()
            for c in range(cls_num):
                correct[c] += (predicted == labels)*(labels == c).sum().item()
                total[c] += (labels == c).sum().item()
    for c in range(cls_num):
        acc[c] = 100.0*correct[c]/(total[c]+0.0001)
    acc_avg = ((torch.tensor(acc).to(device))*weight).sum(dim=0)
    return acc_avg


def prototype_inference(device, model, protos, test_loader):
    model.eval()
    correct = 0
    total = len(test_loader.dataset)
    proto_class = []
    for i in range(10):
        proto_class.append(protos[i])
    g_protos = torch.stack(proto_class)
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            features, outputs = model(inputs)
            dists = torch.stack([torch.norm(g_protos-point,dim=1) for point in features])
            _, predicted = torch.min(dists, 1)  # pred = output.max(1, keepdim=True)[1]
            correct += (predicted == labels).sum().item()
    acc = 100.0*correct/total
    return acc

