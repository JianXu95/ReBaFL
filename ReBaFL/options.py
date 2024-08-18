
import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments
    parser.add_argument('--epochs', type=int, default=200,
                        help="number of training epochs")
    parser.add_argument('--num_users', type=int, default=50,
                        help="number of users: n")
    parser.add_argument('--frac', type=float, default=1.0,
                        help='the fraction of clients: C')
    parser.add_argument('--local_epoch', type=int, default=5,
                        help="the number of local epochs")
    parser.add_argument('--local_iter', type=int, default=1,
                        help="the number of local iterations")
    parser.add_argument('--local_bs', type=int, default=50,
                        help="local batch size: b")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_g', type=float, default=1.0,
                        help='learning rate in server for some methods')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum')
    parser.add_argument('--train_rule', type=str, default='FedAvg',
                        help='the training rule for personalized FL')
    parser.add_argument('--agg_g', type=int, default=0,
                        help='weighted average for personalized FL')
    parser.add_argument('--lam', type=float, default=0.5,
                        help='coefficient for reg term')
    parser.add_argument('--eps', type=float, default=1e-2,
                        help='coefficient for reg term')
    parser.add_argument('--mu', type=float, default=1.0,
                        help='coefficient for reg term')
    parser.add_argument('--local_size', type=int, default=1000,
                        help='number of samples for each client')
    parser.add_argument('--n_way', type=int, default=3,
                        help='number of class for each client')
    parser.add_argument('--k_shot', type=int, default=100,
                        help='number of samples per class for each client')
    parser.add_argument('--concept_shift', type=int, default=0,
                        help='label concept shift')
    parser.add_argument('--imb_ratio', type=int, default=5,
                        help='label concept shift')
    parser.add_argument('--ft', type=int, default=0,
                        help='fine tune classifier in PS')
    parser.add_argument('--max_grad_norm', type=int, default=10,
                        help='max_grad_norm')

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar',
                        help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--device', default='cuda:0', help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--dir', type=int, default=0,
                        help='Default set to 0. Set to 1 for Dir non-IID.')
    parser.add_argument('--lt', type=int, default=0,
                        help='Default set to 0. Set to 1 for global long tail.')
    parser.add_argument('--noniid_s', type=int, default=20,
                        help='Default set to 0.2 Set to 1.0 for IID.')
    parser.add_argument('--noniid_beta', type=float, default=0.1,
                        help='Default set to 0.5, Set to 10.0 for ~IID.')
    parser.add_argument('--unbalance', type=int, default=0,
                        help='whether to use unequal data splits for non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--return_idx', type=int, default=0,
                        help='return index of mini-batch data')
    parser.add_argument('--stopping_rounds', type=int, default=200,
                        help='rounds of early stopping')
    parser.add_argument('--log', type=int, default=0,
                        help='rounds of early stopping')
    parser.add_argument('--user_list', type=list, default=[],
                        help='selected users')
    parser.add_argument('--straggling', type=int, default=1,
                        help='state change period')
    parser.add_argument('--interval', type=int, default=10,
                        help='state change period')
    parser.add_argument('--relabel', type=int, default=0,
                        help='state change period')
    # parser.add_argument('--verbose', type=int, default=1, help='verbose')
    # parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()
    return args
