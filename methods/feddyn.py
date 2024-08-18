# coding: utf-8

import tools
import math
import torch
import copy
from copy import deepcopy
from torch import nn
# ---------------------------------------------------------------------------- #

class LocalUpdate_FedDyn(object):
    def __init__(self, idx, args, train_set, test_set, g_test_set, model):
        self.idx = idx
        self.args = args
        self.num_classes = args.num_classes
        self.train_data = train_set
        self.test_data = test_set
        self.g_test_data = g_test_set
        self.device = args.device
        self.criterion = nn.CrossEntropyLoss()
        self.nllloss = nn.NLLLoss()
        self.local_model = model
        self.w_local_keys = self.local_model.classifier_weight_keys
        self.agg_weight = self.aggregate_weight()
        self.Py = self.prior_y(self.train_data).to(self.device)
        self.alpha = 0.0001
        self.local_grad = deepcopy(model.state_dict())
        for key in self.local_grad.keys():
            self.local_grad[key] = 0.0*self.local_grad[key]
        self.ht = deepcopy(self.local_grad)

    def aggregate_weight(self):
        data_size = len(self.train_data.dataset)
        w = torch.tensor(data_size).to(self.device)
        return w

    def prior_y(self, dataset):
        py = torch.zeros(self.args.num_classes)
        total = len(dataset.dataset)
        data_loader = iter(dataset)
        iter_num = len(data_loader)
        for it in range(iter_num):
            images, labels = next(data_loader)
            for i in range(self.args.num_classes):
                py[i] = py[i] + (i == labels).sum()
        py = py/(total)
        # print(total,py)
        return py

    def balanced_softmax(self, logit):
        py = self.Py
        exp = torch.exp(logit)
        eps = 1e-2
        py1 = (1-eps)*py + eps/self.num_classes  # 0.01 for 2 class and 0.001 for 3 class, 0.0001 for more class
        py_smooth = py1/(py1.sum())
        pc_exp = exp*(py_smooth)
        pc_sftmx = pc_exp/(pc_exp.sum(dim=1).reshape((-1, 1))+1e-8)
        return pc_sftmx
    
    def local_test(self, test_loader):
        model = self.local_model
        model.eval()
        device = self.device
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
    
    def update_local_model(self, global_weight):
        self.local_model.load_state_dict(global_weight)
    
    def local_training(self, local_epoch, local_lr=None):
        # Set mode to train model
        model = self.local_model
        model.train()
        round_loss = 0
        iter_loss = []
        model.zero_grad()
        grad_accum = []
        alpha = self.alpha
        global_param = deepcopy(model.state_dict()) 
        local_grad_prev = deepcopy(self.local_grad)

        acc1 = self.local_test(self.test_data)

        # Set optimizer for the local updates, default sgd
        lr_local = local_lr if local_lr is not None else self.args.lr
        optimizer = torch.optim.SGD(model.parameters(), lr=lr_local, momentum=0.5, weight_decay=0.0005)
        if local_epoch>0:
            for ep in range(local_epoch):
                data_loader = iter(self.train_data)
                iter_num = len(data_loader)
                for it in range(iter_num):
                    images, labels = next(data_loader)
                    images, labels = images.to(self.device), labels.to(self.device)
                    model.zero_grad()
                    _, output = model(images)
                    loss = self.criterion(output, labels)
                    fed_grad_reg = 0.0
                    for index, param in model.named_parameters():
                        fed_grad_reg += alpha*torch.sum(param * (0.5*param+local_grad_prev[index]-global_param[index]))
                    loss = loss + fed_grad_reg
                    loss.backward()
                    # nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    iter_loss.append(loss.item())
        else:
            data_loader = iter(self.train_data)
            iter_num = self.args.local_iter
            for it in range(iter_num):
                images, labels = next(data_loader)
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                _, output = model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                optimizer.step()
                iter_loss.append(loss.item())
        round_loss1 = iter_loss[0] #sum(iter_loss)/len(iter_loss)
        round_loss2 = iter_loss[-1]
        acc2 = self.local_test(self.test_data)

        for index, param in model.named_parameters():
                self.local_grad[index] = self.local_grad[index] + (param-global_param[index]).clone().detach()
        
        return model.state_dict(), grad_accum, round_loss1, round_loss2, acc1, acc2


