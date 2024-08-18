# coding: utf-8

import tools
import math
import copy
import torch
from torch import nn
import time
# ---------------------------------------------------------------------------- #

class LocalUpdate_MOON(object):
    def __init__(self, idx, args, train_set, test_set, g_test_set, model):
        self.idx = idx
        self.args = args
        self.train_data = train_set
        self.test_data = test_set
        self.g_test_data = g_test_set
        self.device = args.device
        self.criterion = nn.CrossEntropyLoss()
        self.cos = torch.nn.CosineSimilarity(dim=-1)
        self.local_model = model
        self.local_model_finetune = copy.deepcopy(model)
        self.local_model_old = copy.deepcopy(model)
        self.w_local_keys = self.local_model.classifier_weight_keys
        self.agg_weight = self.aggregate_weight()
        self.mu = 1.0
        self.temperature = 0.5

    def aggregate_weight(self):
        data_size = len(self.train_data.dataset)
        w = torch.tensor(data_size).to(self.device)
        return w
    
    def local_test(self, test_loader, test_model=None):
        model = self.local_model if test_model is None else test_model
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
                _, predicted = torch.max(outputs.data, 1)  # pred = output.max(1, keepdim=True)[1]
                # total += labels.size(0)                    # pred.eq(target.view_as(pred)).sum().item()
                correct += (predicted == labels).sum().item()
        acc = 100.0*correct/total
        return acc
    
    def update_local_model(self, global_weight):
        # local_weight = self.local_model.state_dict()
        # self.local_model_finetune.load_state_dict(local_weight)
        self.local_model.load_state_dict(global_weight)
    
    def local_training(self, local_epoch, round=0):
        # Set mode to train model
        model = self.local_model
        l_model = self.local_model_old
        l_model.eval()
        g_model = copy.deepcopy(model)
        g_model.eval()
        round_loss = 0
        iter_loss = []
        model.zero_grad()
        grad_accum = []

        w0 = tools.get_parameter_values(model)

        acc1 = self.local_test(self.test_data)

        # Set optimizer for the local updates, default sgd
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5, 
                                                           weight_decay=0.0005)
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
                    new_feature, output = model(images)
                    g_feature, _ = g_model(images)
                    l_feature, _ = l_model(images)
                    posi = self.cos(new_feature, g_feature)
                    nega = self.cos(new_feature, l_feature)
                    logits = torch.cat((posi.reshape(-1,1), nega.reshape(-1,1)), dim=1)
                    loss1 = self.criterion(output, labels)
                    targets = torch.LongTensor([0] * labels.shape[0]).to(self.device)
                    loss2 = self.criterion(logits/self.temperature, targets)
                    loss = loss1 + self.mu*loss2
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
                # grad_accum.append(tools.get_gradient_values(model))
                optimizer.step()
                iter_loss.append(loss.item())
        # loss value
        round_loss1 = iter_loss[0] #sum(iter_loss)/len(iter_loss)
        round_loss2 = iter_loss[-1]
        acc2 = self.local_test(self.test_data)
        # grad_mean = torch.stack(grad_accum).mean(dim=0)
        # w1 = w0 - 0.1*self.args.lr*grad_mean
        # tools.set_parameter_values(model,w1)
        self.local_model_old = copy.deepcopy(model)
        
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
                    if self.args.concept_shift:
                        labels = (labels+2*(self.idx%5))%10
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


