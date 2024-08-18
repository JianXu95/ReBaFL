# coding: utf-8

import tools
import math
import torch
from torch import nn
import torch.nn.functional as F
# ---------------------------------------------------------------------------- #

class LocalUpdate_FedRS(object):
    def __init__(self, idx, args, train_set, test_set, g_test_set, model):
        self.idx = idx
        self.args = args
        self.train_data = train_set
        self.test_data = test_set
        self.g_test_data = g_test_set
        self.device = args.device
        self.logsoftmax = nn.LogSoftmax()
        self.nllloss = nn.NLLLoss()
        self.criterion = nn.CrossEntropyLoss()
        self.local_model = model
        self.w_local_keys = self.local_model.classifier_weight_keys
        self.Py = self.prior_y(self.train_data).to(self.device)
        self.Py1 = self.prior_y(self.test_data).to(self.device)
        self.agg_weight = self.aggregate_weight()

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

    def balanced_softmax(self, logit, py):
        # py = self.Py
        exp = torch.exp(logit)
        py1 = py + 1e-9
        py_smooth = py1/(py1.sum())
        pc_exp = exp*(py_smooth)
        pc_sftmx = pc_exp/(pc_exp.sum(dim=1).reshape((-1, 1))+1e-8)
        return pc_sftmx
    
    def restricted_softmax(self, logit, py):
        py1 = 1.0*(py.gt(0.0).float()) + 0.5*(py.eq(0.0).float())
        # py1 = 10.0*(py)
        pc_exp = torch.exp(py1*logit)
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
                # _, predicted = torch.max((self.Py*torch.exp(outputs)).data, 1)
                # total += labels.size(0)                    # pred.eq(target.view_as(pred)).sum().item()
                correct += (predicted == labels).sum().item()
        acc = 100.0*correct/total
        return acc
    
    def update_local_model(self, global_weight):
        # global_weights = global_model.state_dict()
        self.local_model.load_state_dict(global_weight)
    
    def local_training(self, local_epoch, round=0):
        # Set mode to train model
        model = self.local_model
        model.train()
        round_loss = []
        iter_loss = []
        model.zero_grad()
        grad_accum = []

        acc1 = self.local_test(self.test_data)

        # Set optimizer for the local updates, default sgd
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5, 
                                                                   weight_decay=0.0005)
        # multiple local epochs
        if local_epoch>0:
            for ep in range(local_epoch):
                data_loader = iter(self.train_data)
                iter_num = len(data_loader)
                for it in range(iter_num):
                    images, labels = next(data_loader)
                    images, labels = images.to(self.device), labels.to(self.device)
                    model.zero_grad()
                    _, output = model(images)
                    loglikelihood = torch.log(self.restricted_softmax(output, self.Py))
                    loss = self.nllloss(loglikelihood, labels)
                    loss.backward()
                    optimizer.step()
                    iter_loss.append(loss.item())
                round_loss.append(sum(iter_loss)/len(iter_loss))
        # multiple local iterations, but less than 1 epoch
        else:
            data_loader = iter(self.train_data)
            iter_num = self.args.local_iter
            for it in range(iter_num):
                images, labels = next(data_loader)
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                _, output = model(images)
                loglikelihood = torch.log(self.restricted_softmax(output, self.Py))
                loss = self.nllloss(loglikelihood, labels)
                loss.backward()
                optimizer.step()
                iter_loss.append(loss.item())
        # loss value
        round_loss1 = round_loss[0] #sum(iter_loss)/len(iter_loss)
        round_loss2 = round_loss[-1]
        acc2 = self.local_test(self.test_data)
        
        return model.state_dict(), grad_accum, round_loss1, round_loss2, acc1, acc2

    def local_fine_tuning(self, local_epoch, round=0):
        # Set mode to train model
        model = self.local_model
        model.train()
        round_loss = 0
        iter_loss = []
        model.zero_grad()
        grad_accum = []

        acc1 = self.local_test(self.test_data)

        # Set optimizer for the local updates, default sgd
        for name, param in model.named_parameters():
            if name in self.w_local_keys:
                param.requires_grad = True
            else:
                param.requires_grad = False
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr,
                                                   momentum=0.5, weight_decay=0.0005)
        # multiple local epochs
        if local_epoch>0:
            for ep in range(local_epoch):
                data_loader = iter(self.train_data)
                iter_num = len(data_loader)
                for it in range(iter_num):
                    images, labels = next(data_loader)
                    images, labels = images.to(self.device), labels.to(self.device)
                    model.zero_grad()
                    _, output = model(images)
                    output = torch.log(self.restricted_softmax(output, self.Py))
                    loss = self.nllloss(output,labels)
                    loss.backward()
                    optimizer.step()
                    iter_loss.append(loss.item())
        # multiple local iterations, but less than 1 epoch
        else:
            data_loader = iter(self.train_data)
            iter_num = self.args.local_iter
            for it in range(iter_num):
                images, labels = next(data_loader)
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                _, output = model(images)
                output = torch.log(self.balanced_softmax(output, self.Py))
                loss = self.nllloss(output,labels)
                loss.backward()
                optimizer.step()
                iter_loss.append(loss.item())
        # loss value
        round_loss1 = iter_loss[0] #sum(iter_loss)/len(iter_loss)
        round_loss2 = iter_loss[-1]
        acc2 = self.local_test(self.test_data)
        
        return model.state_dict(), round_loss1, round_loss2, acc1, acc2


