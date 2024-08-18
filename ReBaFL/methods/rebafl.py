# coding: utf-8

import tools
import math
import copy
import torch
from torch import nn
from copy import deepcopy
import torch.nn.functional as F
from tools import LabelSmoothingLoss
# ---------------------------------------------------------------------------- 


class LocalUpdate_ReBaFL(object):
    def __init__(self, idx, args, train_set, test_set, g_test_set, model):
        self.idx = idx
        self.args = args
        self.num_classes = args.num_classes
        self.train_data = train_set
        self.test_data = test_set
        self.g_test_data = g_test_set
        self.device = args.device
        self.criterion = nn.CrossEntropyLoss()
        self.logsoftmax = nn.LogSoftmax()
        self.nllloss = nn.NLLLoss()
        self.l1_loss = nn.L1Loss()
        self.softmax = nn.Softmax()
        self.kl_loss = nn.KLDivLoss()
        self.local_model = model
        self.local_model_finetune = copy.deepcopy(model)
        self.last_model = deepcopy(model)
        self.w_local_keys = self.local_model.classifier_weight_keys
        self.size_class = self.size_label(self.train_data)
        self.Py = self.prior_y(self.train_data).to(self.device)
        self.avail_class = (self.Py .gt(0.0).float()).sum()
        self.Py_smooth = (self.Py + 0.01)/((self.Py + 0.01).sum())
        self.agg_weight = self.aggregate_weight()
        self.global_protos = None
        self.mse_loss = nn.MSELoss()
        self.cossim = torch.nn.CosineSimilarity(dim=-1)
        self.pdist = torch.nn.PairwiseDistance(p=2)
        self.temperature = 0.1
        self.lam = args.lam if self.avail_class>1 else 0.0 # 1.0
        self.global_model_copy = deepcopy(model)
        self.local_grad = deepcopy(model.state_dict())
        for key in self.local_grad.keys():
            self.local_grad[key] = 0.0*self.local_grad[key]
        self.local_weight_t1 = tools.get_parameter_values(model)
        self.local_weight_t2 = tools.get_parameter_values(model)

    def aggregate_weight(self):
        data_size = len(self.train_data.dataset)
        ent = 1.0*tools.entropy(self.Py)**1.0
        w = torch.tensor(data_size).to(self.device)
        # w = (torch.exp(ent)*data_size).to(self.device)
        return w
    
    def size_label(self, dataset):
        size_per_label = {}
        size_class = torch.zeros(self.num_classes)
        total = len(self.train_data.dataset)
        for inputs, labels in self.train_data:
            for i in range(self.num_classes):
                size_class[i] = size_class[i] + (i == labels).sum()
        for i in range(self.num_classes):
            # if size_class[i]>0:
            size_per_label[i] = size_class[i].to(self.device)
        return size_per_label

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

    def prior_y_batch(self, labels):
        py = torch.zeros(self.args.num_classes)
        total = len(labels)
        for i in range(self.args.num_classes):
            py[i] = py[i] + (i == labels).sum()
        py = py/(total)
        # print(total,py)
        py = py.to(self.device)
        return py
    
    def balanced_softmax1(self, logit):
        py = self.Py
        exp = torch.exp(logit)
        eps = self.args.eps # 1e-3 if (self.avail_class > 3) else 1e-2
        py1 = (1-eps)*py + eps/self.num_classes  # 0.01 for 2 class and 0.001 for 3 class, 0.0001 for more class
        py_smooth = py1/(py1.sum())
        pc_exp = exp*(py_smooth)
        pc_sftmx = pc_exp/(pc_exp.sum(dim=1).reshape((-1, 1))+1e-8)
        return pc_sftmx

    def balanced_softmax2(self, logit, py):
        exp = torch.exp(logit)
        eps = self.args.eps # 1e-3 if (self.avail_class > 3) else 1e-2
        py1 = (1-eps)*py + eps/self.num_classes  # 0.01 for 2 class and 0.001 for 3 class, 0.0001 for more class
        py_smooth = py1/(py1.sum())
        pc_exp = exp*(py_smooth)
        pc_sftmx = pc_exp/(pc_exp.sum(dim=1).reshape((-1, 1))+1e-8)
        return pc_sftmx

    def local_test(self, testloader=None):
        model = self.local_model
        model.eval()
        device = self.device
        correct = 0
        test_loader = self.test_data if testloader is None else testloader
        total = len(test_loader.dataset)
        loss_test = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                _, outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                loss_test.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)  # pred = output.max(1, keepdim=True)[1]
                # total += labels.size(0)                    # pred.eq(target.view_as(pred)).sum().item()
                correct += (predicted == labels).sum().item()
        acc = 100.0*correct/total
        return acc, sum(loss_test)/len(loss_test)
    
    def update_local_model(self, global_weight, global_protos=None):
        local_weight = self.local_model.state_dict()
        w_local_keys = self.w_local_keys
        alpha = 0.0
        for k in local_weight.keys():
            if k not in w_local_keys:
                local_weight[k] = global_weight[k] #global_weight[k]
            else:
                local_weight[k] = alpha*local_weight[k]+(1-alpha)*global_weight[k]
        self.local_model.load_state_dict(local_weight)
        if global_protos is not None:
            self.global_protos = global_protos
            # self.update_global_protos()

    def update_global_protos(self):
        global_protos = self.global_protos
        g_classes, g_protos = [], []
        for i in range(self.num_classes):
            g_classes.append(torch.tensor(i))
            # proto = global_protos[i] if i in global_protos else 0
            g_protos.append(global_protos[i])
        self.g_classes = torch.stack(g_classes).to(self.device)
        self.g_protos = torch.stack(g_protos)
    
    def get_local_protos(self):
        model = self.local_model
        local_protos_list = {}
        for inputs, labels in self.train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            features, outputs = model(inputs)
            protos = features.clone().detach()
            for i in range(len(labels)):
                if labels[i].item() in local_protos_list.keys():
                    local_protos_list[labels[i].item()].append(protos[i,:])
                else:
                    local_protos_list[labels[i].item()] = [protos[i,:]]
        local_protos = tools.get_protos(local_protos_list)
        return local_protos

    def get_local_covariance(self, inputs, labels):
        model = self.local_model
        local_feature_list = {}
        features, outputs = model(inputs)
        protos = features.clone().detach()
        for i in range(len(labels)):
            if labels[i].item() in local_feature_list.keys():
                local_feature_list[labels[i].item()].append(protos[i,:])
            else:
                local_feature_list[labels[i].item()] = [protos[i,:]]
        
        local_covariance = tools.get_covariance(local_feature_list)

        return local_covariance

    def get_local_covariance_full(self):
        model = self.local_model
        local_feature_list = {}
        for inputs, labels in self.train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            features, outputs = model(inputs)
            protos = features.clone().detach()
            for i in range(len(labels)):
                if labels[i].item() in local_feature_list.keys():
                    local_feature_list[labels[i].item()].append(protos[i,:])
                else:
                    local_feature_list[labels[i].item()] = [protos[i,:]]

        local_covariance = tools.get_covariance(local_feature_list)

        return local_covariance

    def local_training(self, local_epoch, round=0):
        # Set mode to train model
        model = self.local_model
        model.train()
        round_loss = []
        iter_loss = []
        model.zero_grad()
        global_protos = self.global_protos
        local_w_1 = tools.get_parameter_values(model)

        acc1, loss_inital = self.local_test(self.test_data)
        # acc1, loss_inital = self.local_test_py(self.test_data)
        self.last_loss = loss_inital
        self.last_model = deepcopy(model)
        model_last = deepcopy(model)
        model_last.eval()

        # get local prototypes before training, dict:={label: list of sample features}
        if self.lam>0:
            local_protos1 = self.get_local_protos()
        else:
            local_protos1 = None
        # global_protos = local_protos1

        # Set optimizer for the local updates, default sgd
        local_lr = self.args.lr #*0.998**(round)
        optimizer = torch.optim.SGD(model.parameters(), lr=local_lr, momentum=0.5, 
                                                                  weight_decay=0.0005)

        for ep in range(local_epoch):
            # local training for 1 epoch
            data_loader = iter(self.train_data)
            iter_num = len(data_loader)
            for it in range(iter_num):
                images, labels = next(data_loader)
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                feature, output = model(images)
                loglikelihood = torch.log(self.balanced_softmax1(output))
                loss0 = self.nllloss(loglikelihood, labels)
                
                loss1 = 0
                if self.lam > 0 and global_protos and round > 0:
                    protos_new = feature.clone().detach()
                    features = feature.clone().detach()
                    protos_aug = feature.clone().detach()
                    labels_agu = labels.clone().detach()
                    y = labels.clone().detach()

                    for i in range(len(labels)):
                        yi = labels[i].item()
                        y = i%self.num_classes
                        if (y in global_protos) and (yi in global_protos):
                            protos_aug[i] = 1.0*(global_protos[y]-global_protos[yi]) + protos_new[i]
                            labels_agu[i] = y
                    g_logits = model.feature2logit(protos_aug)
                    py = self.prior_y_batch(labels_agu)
                    loglikelihood = torch.log(self.balanced_softmax2(g_logits, py))
                    loss1 = self.nllloss(loglikelihood, labels_agu)
                loss = loss0 + self.lam * loss1
                loss.backward()
                optimizer.step()
                iter_loss.append(loss.item())
            round_loss.append(sum(iter_loss)/len(iter_loss))
            iter_loss = []
            # ---------------------------------------------------------------------------
        
        # get new local prototypes after training
        local_protos2 = self.get_local_protos()

        round_loss1 = round_loss[0]
        round_loss2 = round_loss[-1]
        acc2, _ = self.local_test(self.test_data)
        local_w_2 = tools.get_parameter_values(model)

        w_update = local_w_2 - local_w_1

        
        return deepcopy(model), w_update, round_loss1, round_loss2, acc1, acc2, 0, local_protos2


    def local_fine_tuning(self, local_epoch, round=0):
        # Set mode to train model
        model = self.local_model
        model.train()
        iter_loss = []
        model.zero_grad()
        grad_accum = []
        is_last_round = (round > self.args.epochs-1)

        acc0, _ = self.local_test(self.test_data)
        _, loss_inital = self.local_test(self.test_data)
        self.last_loss = loss_inital
        self.last_model = deepcopy(model)

        # Set optimizer for the local updates, default sgd
        for name, param in model.named_parameters():
            if name in self.w_local_keys:
                param.requires_grad = True
            else:
                param.requires_grad = False
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr,
                                                   momentum=0.5, weight_decay=0.0005)
        if local_epoch>0:
            for ep in range(local_epoch):
                data_loader = iter(self.train_data)
                iter_num = len(data_loader)
                # start = time.time()
                for it in range(iter_num):
                    images, labels = next(data_loader)
                    if self.args.concept_shift:
                        labels = (labels+2*(self.idx//10))%10
                    images, labels = images.to(self.device), labels.to(self.device)
                    model.zero_grad()
                    _, output = model(images)
                    loss = self.criterion(output, labels)
                    loss.backward()
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
        acc1, _ = self.local_test(self.test_data)
        param_new = tools.get_parameter_values(model)
        
        return model.state_dict(), grad_accum, round_loss1, round_loss2, acc0, acc1,  param_new