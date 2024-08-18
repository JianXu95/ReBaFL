# coding: utf-8

import tools
import math
import torch
from torch import nn
import torch.nn.functional as F
# ---------------------------------------------------------------------------- #

class LocalUpdate_StandAlone(object):
    def __init__(self, idx, args, train_set, test_set, g_test_set, model):
        self.idx = idx
        self.args = args
        self.train_data = train_set
        self.test_data = test_set
        self.g_test_data = g_test_set
        self.device = args.device
        self.criterion = nn.CrossEntropyLoss()
        self.local_model = model
        self.w_local_keys = self.local_model.classifier_weight_keys
    
    def local_test(self, test_loader):
        model = self.local_model
        model.eval()
        device = self.device
        correct = 0
        total = len(test_loader.dataset)
        with torch.no_grad():
            for inputs, labels in test_loader:
                if self.args.concept_shift:
                    labels = (labels+2*(self.idx%5))%10
                inputs, labels = inputs.to(device), labels.to(device)
                _, outputs = model(inputs)
                conf, predicted = torch.max(outputs.data, 1)  # pred = output.max(1, keepdim=True)[1]
                # total += labels.size(0)                    # pred.eq(target.view_as(pred)).sum().item()
                correct += (predicted == labels).sum().item()
                # mask = conf > 0.5
                # correct += (predicted == labels)[mask].sum().item()
                # total += mask.sum()
        if total == 0:
            acc = 0
        else:
            acc = 100.0*correct/total
        return acc

    def local_test_prob(self, test_loader):
        model = self.local_model
        model.eval()
        device = self.device
        confs = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                _, outputs = model(inputs)
                conf = F.softmax(outputs, dim=1)
                confs.append(conf)
        prob = torch.cat(confs).cpu().numpy()
        # print(prob[0])
        return prob
    
    def update_local_model(self, global_weight):
        local_weight = self.local_model.state_dict()
        w_local_keys = self.w_local_keys
        for k in local_weight.keys():
            if k not in w_local_keys:
                local_weight[k] = global_weight[k]
            else:
                local_weight[k] = 0*global_weight[k]
        self.local_model.load_state_dict(local_weight)
    
    def local_training(self, local_epoch):
        # Set mode to train model
        model = self.local_model
        round_loss = 0
        iter_loss = []
        model.zero_grad()
        grad_accum = []
        acc1 = self.local_test(self.test_data)
        model.train()
        # for name, param in model.named_parameters():
        #     if name in self.w_local_keys:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False
        # Set optimizer for the local updates, default sgd
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                                           lr=self.args.lr, momentum=0.5, weight_decay=0.0005)
        # multiple local epochs
        if local_epoch>0:
            for ep in range(local_epoch):
                data_loader = iter(self.train_data)
                iter_num = len(data_loader)
                for it in range(iter_num):
                    images, labels = next(data_loader)
                    if self.args.concept_shift:
                        labels = (labels+2*(self.idx%5))%10
                    images, labels = images.to(self.device), labels.to(self.device)
                    model.zero_grad()
                    _, output = model(images)
                    loss = self.criterion(output, labels)
                    loss.backward()
                    # grad_accum.append(tools.get_gradient_values(model))
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
                loss = self.criterion(output, labels)
                loss.backward()
                # grad_accum.append(tools.get_gradient_values(model))
                optimizer.step()
                iter_loss.append(loss.item())
        # loss value
        round_loss1 = iter_loss[0] #sum(iter_loss)/len(iter_loss)
        round_loss2 = iter_loss[-1]
        acc2 = self.local_test(self.test_data)
        
        return model.state_dict(), grad_accum, round_loss1, round_loss2, acc1, acc2


    def local_fine_tuning(self, local_epoch, round=0):
        # Set mode to train model
        model = self.local_model
        round_loss = 0
        iter_loss = []
        model.zero_grad()
        grad_accum = []

        acc1 = self.local_test(self.test_data)

        # Set optimizer for the local updates, default sgd
        model.train()
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
                    loss = self.criterion(output, labels)
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
                loss = self.criterion(output, labels)
                loss.backward()
                optimizer.step()
                iter_loss.append(loss.item())
        # loss value
        round_loss1 = iter_loss[0] #sum(iter_loss)/len(iter_loss)
        round_loss2 = iter_loss[-1]
        acc2 = self.local_test(self.test_data)
        
        return model.state_dict(), round_loss1, round_loss2, acc1, acc2