import numpy as np
import torch
from data_loader import get_dataset
from running import test_classification, one_round_training
from methods import local_update
from models import MLP, LR, CifarCNN, CNN_FMNIST
from options import args_parser
import copy
import time


if __name__ == '__main__':
    args = args_parser()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    # load dataset and user groups
    train_loader, test_loader, global_test_loader, val_loader = get_dataset(args)
    seed = 520
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # construct model
    if args.dataset in ['cifar', 'cifar10']:
        global_model = CifarCNN(num_classes=args.num_classes).to(device)
    elif args.dataset in ['mnist', 'fmnist']:
        global_model = CNN_FMNIST().to(device)
    elif args.dataset == 'emnist':
        args.num_classes = 62
        global_model = CNN_FMNIST(num_classes=args.num_classes).to(device)
    else:
        global_model = LR().to(device)

    # Training Rule
    LocalUpdate = local_update(args.train_rule)
    # One Round Training Function
    train_round_parallel = one_round_training(args.train_rule)

    # Training
    iteration = len(train_loader[0].dataset) // (args.local_bs)
    max_local_iter_1epoch = len(train_loader[0].dataset) // (args.local_bs)
    print("max local_iter per epoch: {}".format(max_local_iter_1epoch))
    train_loss, train_acc = [], []
    test_acc = []
    local_accs1, local_accs2 = [], []
    local_stds1, local_stds2 = [], []
#======================================================================================================#
###### training for 1-local-iteration
    local_clients = []
    for idx in range(args.num_users):
        local_clients.append(LocalUpdate(idx=idx, args=args, train_set=train_loader[idx], test_set=test_loader[idx], 
                                         g_test_set=global_test_loader, model=copy.deepcopy(global_model)))

    for round in range(args.epochs+1):
        global_model.train()
        start_t = time.time()
        loss1, loss2, local_acc1, local_acc2, protos = train_round_parallel(args, global_model, local_clients, round, metadata=val_loader)
        end_t = time.time()
        train_loss.append(loss1)
        print("Dataset: {}, Train Rule: {}".format(args.dataset, args.train_rule))
        print("Train Loss: {}, {}".format(loss1, loss2))
        print("Local Accuracy on Local Data: {}%, {}%".format(local_acc1, local_acc2))
        local_accs1.append(local_acc1)
        local_accs2.append(local_acc2)
        acc = test_classification(device, global_model, global_test_loader)
        test_acc.append(acc)
        print("Global Model Accuracy: {}%".format(acc))
        print("Global Training Time: {} s".format(end_t-start_t))


