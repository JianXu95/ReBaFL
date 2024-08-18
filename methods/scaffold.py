# coding: utf-8

import tools
import math
import torch
from torch import nn
# ---------------------------------------------------------------------------- #

class LocalUpdate_Scaffold(object):
    def __init__(self, idx, args, train_set, test_set, g_test_set, model):
        self.idx = idx
        self.args = args
        self.train_data = train_set
        self.test_data = test_set
        self.g_test_data = g_test_set
        self.device = args.device
        self.criterion = nn.CrossEntropyLoss()
        self.local_model = model
        self.agg_weight = self.aggregate_weight()
        self.local_ci = torch.zeros(tools.get_parameter_values(model).size()).to(args.device)
        self.global_c = torch.zeros(tools.get_parameter_values(model).size()).to(args.device)

    def aggregate_weight(self):
        data_size = len(self.train_data.dataset)
        w = torch.tensor(data_size).to(self.device)
        return w
    
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
    
    def update_local_model(self, global_weight, delta_global_c):
        self.local_model.load_state_dict(global_weight)
        self.global_c += delta_global_c
    
    def local_training(self, local_epoch):
        # Set mode to train model
        model = self.local_model
        model.train()
        round_loss = 0
        iter_loss = []
        model.zero_grad()
        # grad_accum = []
        k_steps = 0

        # get initial local model weights
        local_w_0 = tools.get_parameter_values(model)

        acc1 = self.local_test(self.test_data)

        # Set optimizer for the local updates, default sgd
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5, 
                                                                weight_decay=0.0005)
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
                    nn.utils.clip_grad_norm_( model.parameters(), self.args.max_grad_norm)
                    # gradient modification (debias/drift control)
                    optimizer.step()
                    iter_loss.append(loss.item())
                    k_steps += 1
                    local_w_t = tools.get_parameter_values(model)
                    local_w_t = local_w_t - self.args.lr * (self.global_c - self.local_ci)
                    tools.set_parameter_values(model, local_w_t)
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
                # gradient modification (debias/drift control)
                grad_raw = tools.get_gradient_values(model)
                grad_new = grad_raw - self.local_ci + self.global_c
                # set new gradient
                tools.set_gradient_values(model, grad_new)
                optimizer.step()
                iter_loss.append(loss.item())
                k_steps += 1
        # get updated local model weights
        local_w_1 = tools.get_parameter_values(model)
        # update local_ci and return (delta_w, delta_ci)
        local_ci_plus = self.local_ci - self.global_c + torch.div((local_w_0-local_w_1), k_steps*self.args.lr)
        delta_w = local_w_1-local_w_0
        delta_ci = local_ci_plus - self.local_ci
        # update local_ci
        self.local_ci = local_ci_plus

        round_loss1 = iter_loss[0] #sum(iter_loss)/len(iter_loss)
        round_loss2 = iter_loss[-1]
        acc2 = self.local_test(self.test_data)
        
        return delta_w, delta_ci, round_loss1, round_loss2, acc1, acc2


