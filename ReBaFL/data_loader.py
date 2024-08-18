import numpy as np
from numpy.core.fromnumeric import trace
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch
import pdb
import os
import glob
from shutil import copyfile
import json


## --------------------------------------------------
## dataset split
## --------------------------------------------------
# MNIST
def mnist_iid(dataset, num_users, *args, **kwargs):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    np.random.seed(2021)
    for i in range(num_users):
        select_set = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - select_set)
        dict_users[i] = list(select_set)
    return dict_users


def mnist_noniid(dataset, num_users, *args, **kwargs):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    np.random.seed(2022)
    shard_per_user = 3
    imgs_per_shard = int(len(dataset) / (num_users * shard_per_user))
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    
    idxs_dict = {}
    for i in range(len(dataset)):
        label = dataset.targets[i].item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    num_classes = len(np.unique(dataset.targets))
    rand_set_all = []
    if len(rand_set_all) == 0:
        for i in range(num_users):
            x = np.random.choice(np.arange(num_classes), shard_per_user, replace=False)
            rand_set_all.append(x)

    # divide and assign
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            # pdb.set_trace()
            x = np.random.choice(idxs_dict[label], imgs_per_shard, replace=False)
            rand_set.append(x)
        dict_users[i] = np.concatenate(rand_set)

    for key, value in dict_users.items():
        assert(len(np.unique(torch.tensor(dataset.targets)[value]))) == shard_per_user

    return dict_users


def mnist_noniid_s(dataset, num_users, *args, **kwargs):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs
    s = 0.0
    num_samples = len(dataset)
    num_per_user = int(num_samples/num_users)
    num_imgs_iid = int(num_per_user * s)
    num_imgs_noniid = num_per_user - num_imgs_iid
    dict_users = {i: np.array([]) for i in range(num_users)}
    labels = dataset.targets.numpy()
    idxs = np.arange(len(dataset.targets))
    # iid labels
    idxs_labels = np.vstack((idxs, labels))
    iid_length = int(s*len(labels))
    iid_idxs = idxs_labels[0,:iid_length]
    # noniid labels
    noniid_idxs_labels = idxs_labels[:,iid_length:]
    idxs_noniid = noniid_idxs_labels[:, noniid_idxs_labels[1, :].argsort()]
    noniid_idxs = idxs_noniid[0, :]
    num_shards, num_imgs = num_users*2, int(num_imgs_noniid/2)
    idx_shard = [i for i in range(num_shards)]
    all_idxs = [int(i) for i in iid_idxs]
    np.random.seed(2021)
    for i in range(num_users):
        # allocate iid idxs
        # selected_set = set(np.random.choice(all_idxs, num_imgs_iid, replace=False))
        selected_set = set(all_idxs[:num_imgs_iid])
        all_idxs = list(set(all_idxs) - selected_set)
        dict_users[i] = np.concatenate((dict_users[i], np.array(list(selected_set))), axis=0)
        # allocate noniid idxs
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], noniid_idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
        # start = i*num_imgs_noniid
        # dict_users[i] = np.concatenate((dict_users[i], noniid_idxs[start:start+num_imgs_noniid]), axis=0)
        dict_users[i] = dict_users[i].astype(int)
        # np.random.shuffle(dict_users[i])
    return dict_users


def mnist_noniid_ss(dataset, num_users, noniid_s=20, local_size=1000, train=True):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 50,000 training imgs
    np.random.seed(2022)
    s = noniid_s/100
    num_per_user = local_size
    num_classes = len(np.unique(dataset.targets))
    # noniid_labels = int(num_classes/10)*np.random.randint(2, 6, num_users)
    # noniid_labels_list = [[0,1,2,3],[4,5,6],[7,8,9]]
    # noniid_labels_list = [[0,1,2], [2,3,4], [4,5,6], [6,7,8], [8,9,0]]
    # noniid_labels_list = [[0,1], [2,3], [4,5], [6,7], [8,9]]
    noniid_labels_list = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]

    # num_imgs = [300, 600, 1200]
    # num_per_user = [0 for _ in range(num_users)]
    # for i in range(num_users):
    #     num_per_user[i] = num_imgs[i%3]
    # -------------------------------------------------------
    # divide the first dataset
    num_imgs_iid = int(num_per_user*s)
    num_imgs_noniid = num_per_user - num_imgs_iid
    dict_users = {i: np.array([]) for i in range(num_users)}
    num_samples = len(dataset)
    num_per_label_total = int(num_samples/num_classes)
    labels1 = np.array(dataset.targets)
    idxs1 = np.arange(len(dataset.targets))
    # iid labels
    idxs_labels = np.vstack((idxs1, labels1))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # label available
    label_list = [i for i in range(num_classes)]
    # number of imgs has allocated per label
    label_used = [0 for i in range(num_classes)]
    iid_per_label = int(num_imgs_iid/num_classes)
    iid_per_label_last = num_imgs_iid - (num_classes-1)*iid_per_label

    first_idx_label = [np.min(np.where(idxs_labels[1,:]==i)) for i in range(num_classes)]

    np.random.seed(2022)
    for i in range(num_users):
        # allocate iid idxs
        label_cnt = 0
        for y in label_list:
            label_cnt = label_cnt + 1
            iid_num = iid_per_label
            start = first_idx_label[y]+label_used[y]
            if label_cnt == num_classes:
                iid_num = iid_per_label_last
            if (label_used[y]+iid_num)>num_per_label_total:
                start = first_idx_label[y]
                label_used[y] = 0
            dict_users[i] = np.concatenate((dict_users[i], idxs[start:start+iid_num]), axis=0)
            label_used[y] = label_used[y] + iid_num
        # allocate noniid idxs
        # num_labels = len(noniid_labels_list[(i%5)])
        # rand_label = np.random.choice(label_list, num_labels, replace=False)
        # rand_label = np.random.choice(label_list, 5, replace=False)
        rand_label = noniid_labels_list[i%10]
        noniid_labels = len(rand_label)
        noniid_per_num = int(num_imgs_noniid/noniid_labels)
        noniid_per_num_last = num_imgs_noniid - noniid_per_num*(noniid_labels-1)
        label_cnt = 0
        for y in rand_label:
            label_cnt = label_cnt + 1
            noniid_num = noniid_per_num
            start = first_idx_label[y]+label_used[y]
            if label_cnt == noniid_labels:
                noniid_num = noniid_per_num_last
            if (label_used[y]+noniid_num)>num_per_label_total:
                start = first_idx_label[y]
                label_used[y] = 0
            dict_users[i] = np.concatenate((dict_users[i], idxs[start:start+noniid_num]), axis=0)
            label_used[y] = label_used[y] + noniid_num
        dict_users[i] = dict_users[i].astype(int)
    return dict_users


def mnist_noniid_lt(dataset, num_users, imgs_shot=100, n_way=3, ratio=1):
    """
    Sample non-I.I.D client data from MNIST dataset, Latent distribution
    :param dataset:
    :param num_users:
    :return:
    """
    # 50,000 training imgs --
    # num_class = 2 # in default
    num_samples = len(dataset)
    num_classes = len(np.unique(dataset.targets))
    # num_imgs = int(num_samples/(num_class*num_users)) # indefault 500
    num_per_label_total = int(num_samples/num_classes)
    dict_users = {i: np.array([]) for i in range(num_users)}
    labels = np.array(dataset.targets)
    idxs = np.arange(len(labels))
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # label available
    label_list = [i for i in range(num_classes)]
    # number of imgs has allocated per label
    label_used = [0 for i in range(num_classes)]
    
    # divide and assign 2 shards/client
    np.random.seed(2022)
    # num_class = np.random.randint(2, 9, num_users)
    local_labels = n_way
    imgs_label = imgs_shot # 100 in default
    groups = 5
    user_per_group = int(num_users/groups)
    user_labels = []
    label_list_np = np.array(label_list)
    for i in range(num_users):
        start = (local_labels*(i//user_per_group))%num_classes
        if start+local_labels<=num_classes:
            user_labels.append(label_list_np[start:start+local_labels])
        else:
            user_labels.append(np.concatenate((label_list_np[start:num_classes], label_list_np[0:local_labels+start-num_classes]), axis=0))
    
    num_imgs = [int(imgs_label*local_labels) for i in range(num_users)]
    # num_imgs = [int(num_samples/num_users) for i in range(num_users)]
    # num_imgs = int(0.2*num_samples/num_users)*np.random.randint(2, 9, num_users)
    for i in range(num_users):
        num_per_label = int(num_imgs[i]/local_labels)
        num_per_label_last = num_imgs[i] - num_per_label*(local_labels-1)
        if i < user_per_group:
            num_per_label = ratio*imgs_label
            num_per_label_last = num_per_label
        # rand_label = np.random.choice(label_list, num_labels[i], replace=False)
        rand_label = user_labels[i]
        label_cnt = 0
        for y in rand_label:
            label_cnt = label_cnt + 1
            start = y*num_per_label_total+label_used[y]
            if label_cnt == local_labels:
                num_per_label = num_per_label_last
            if (label_used[y]+num_per_label)>num_per_label_total:
                start = y*num_per_label_total
                label_used[y] = 0
            dict_users[i] = np.concatenate((dict_users[i], idxs[start:start+num_per_label]), axis=0)
            label_used[y] = label_used[y] + num_per_label
    return dict_users


def mnist_noniid_dir(dataset, num_users, noniid_beta=1.0, train=True):
    """
    Sample non-I.I.D client data from CIFAR dataset via Dirichlet
    :param dataset:
    :param num_users:
    :return:
    """
    # 50,000 training imgs
    beta = noniid_beta
    num_samples = len(dataset)
    num_classes = num_classes = len(np.unique(dataset.targets))
    min_size = 0
    min_require_size = 150 if train else 30 #0.1*int(num_samples/num_users)
    dict_users = {i: np.array([]) for i in range(num_users)}
    idx_batchs = [[] for _ in range(num_users)]
    labels = np.array(dataset.targets)
    idxs = np.arange(len(dataset.targets))
    np.random.seed(2022)
    # imgs_per_label = int(num_users*len(dataset.targets)/(num_classes*100))
    while min_size < min_require_size:
        idx_batchs = [[] for _ in range(num_users)]
        for k in range(num_classes):
            idx_k = np.where(labels == k)[0]
            # np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, num_users)) #+ (0.1/num_users)
            # proportions = np.array([p * (len(idx_j) < (num_samples / num_users)) for p, idx_j in zip(proportions, idx_batchs)])
            proportions = np.array([p for p, idx_j in zip(proportions, idx_batchs)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batchs = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batchs, np.split(idx_k, proportions))]
        min_size = min([len(idx_j) for idx_j in idx_batchs])
    for i in range(num_users):
        np.random.shuffle(idx_batchs[i])
        dict_users[i] = idx_batchs[i]

    return dict_users


def mnist_noniid_dir_balance(dataset, num_users, noniid_beta=1.0, local_size=1000, train=True):
    """
    Sample non-I.I.D client data from CIFAR dataset via Dirichlet
    :param dataset:
    :param num_users:
    :return:
    """
    # 50,000 training imgs
    beta = noniid_beta
    num_samples = len(dataset)
    num_classes = num_classes = len(np.unique(dataset.targets))
    min_size = 0
    dict_users = {i: np.array([]) for i in range(num_users)}
    labels = np.array(dataset.targets)
    idxs = np.arange(len(dataset.targets))
    np.random.seed(2022)
    # imgs_per_label = int(num_users*len(dataset.targets)/(num_classes*100))
    idx_batchs = [[] for _ in range(num_users)]
    idx_class = [np.where(labels == k)[0] for k in range(num_classes)]
    label_used = [0 for i in range(num_classes)]
    for i in range(num_users):
        proportions = np.random.dirichlet(np.repeat(beta, num_classes)) #+ (0.01/num_classes)
        proportions = proportions / proportions.sum()
        data_sizes = ((proportions) * local_size).astype(int)[:-1].tolist()
        data_sizes.append(local_size - sum(data_sizes)) 
        np.random.seed(520)
        for k in range(num_classes):
            start = label_used[k]
            if (label_used[k]+data_sizes[k])>len(idx_class[k]):
                start = 0
                label_used[k] = 0
            idx_batchs[i] += idx_class[k][start:start+data_sizes[k]].tolist()
            label_used[k] += data_sizes[k]
            # idx_batchs[i] += np.random.choice(idx_class[k], data_sizes[k], replace=False).tolist()
    for i in range(num_users):
        np.random.shuffle(idx_batchs[i])
        dict_users[i] = idx_batchs[i]

    return dict_users


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users


## Extended-MNIST
def emnist_noniid_ss(dataset, num_users, train=True, noniid_s=20, local_size=1000):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 62 classes 62*4 =248 ~ 250
    np.random.seed(2022)
    s = noniid_s/100
    num_per_user = local_size if train else 600
    noniid_labels_list = [[i for i in range(10)],[i for i in range(10, 36)],[i for i in range(36,62)]]
    num_classes = len(np.unique(dataset.targets))
    # -------------------------------------------------------
    # divide the first dataset
    num_imgs_iid = int(num_per_user * s)
    num_imgs_noniid = num_per_user - num_imgs_iid
    dict_users = {i: np.array([]) for i in range(num_users)}
    num_samples = len(dataset)
    num_per_label_total = int(num_samples/num_classes)
    labels1 = np.array(dataset.targets)
    idxs1 = np.arange(len(dataset.targets))
    # iid labels
    idxs_labels = np.vstack((idxs1, labels1))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    start_idxs = [0 for i in range(num_classes)]
    total_idxs = [0 for i in range(num_classes)]
    for i in range(num_classes):
        start_idxs[i] = np.where(idxs_labels[1,:]==i)[0][0]
        total_idxs[i] = len(np.where(idxs_labels[1,:]==i)[0])
    # label available
    label_list = [i for i in range(num_classes)]
    # number of imgs has allocated per label
    label_used = [0 for i in range(num_classes)]
    iid_per_label = int(num_imgs_iid/num_classes)
    iid_per_label_last = num_imgs_iid - (num_classes-1)*iid_per_label
    noniid_num_imgs = num_per_user - num_imgs_iid

    np.random.seed(2022)
    for i in range(num_users):
        # allocate iid idxs
        label_cnt = 0
        for y in label_list:
            label_cnt = label_cnt + 1
            start = start_idxs[y]+label_used[y]
            iid_num = iid_per_label
            if label_cnt == num_classes:
                iid_num = iid_per_label_last
            if (label_used[y]+iid_num)>total_idxs[y]:
                start = start_idxs[y]
                label_used[y] = 0
            dict_users[i] = np.concatenate((dict_users[i], idxs[start:start+iid_num]), axis=0)
            label_used[y] = label_used[y] + iid_num
        # allocate noniid idxs
        rand_label = noniid_labels_list[i%3]
        noniid_labels = noniid_labels_list[i%3]
        noniid_labels_num = len(noniid_labels)
        noniid_per_num = int(noniid_num_imgs/noniid_labels_num)
        noniid_per_num_last = noniid_num_imgs - noniid_per_num*(noniid_labels_num-1)
        # rand_label = np.random.choice(label_list, noniid_labels, replace=False)
        label_cnt = 0
        for y in rand_label:
            label_cnt = label_cnt + 1
            start = start_idxs[y]+label_used[y]
            noniid_num = noniid_per_num
            if label_cnt == noniid_labels_num:
                noniid_num = noniid_per_num_last
            if (label_used[y]+noniid_num)>total_idxs[y]:
                start = start_idxs[y]
                label_used[y] = 0
            dict_users[i] = np.concatenate((dict_users[i], idxs[start:start+noniid_num]), axis=0)
            label_used[y] = label_used[y] + noniid_per_num
        dict_users[i] = dict_users[i].astype(int)
    return dict_users


def emnist_noniid_dir(dataset, num_users, train=True, noniid_beta=1.0):
    """
    Sample non-I.I.D client data from CIFAR dataset via Dirichlet
    :param dataset:
    :param num_users:
    :return:
    """
    # 50,000 training imgs
    beta = noniid_beta
    num_samples = len(dataset)
    num_classes = num_classes = len(np.unique(dataset.targets))
    min_size = 0
    min_require_size = 100 #0.1*int(num_samples/num_users)
    dict_users = {i: np.array([]) for i in range(num_users)}
    idx_batchs = [[] for _ in range(num_users)]
    labels = np.array(dataset.targets)
    idxs = np.arange(len(dataset.targets))
    max_imgs_per_label = 2000 if train else 400
    imgs_per_label = int(max_imgs_per_label)
    np.random.seed(2022)
    while min_size < min_require_size:
        idx_batchs = [[] for _ in range(num_users)]
        for k in range(num_classes):
            idx_k = np.where(labels == k)[0][:imgs_per_label]
            # np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, num_users))
            # proportions = np.array([p * (len(idx_j) < num_samples / num_users) for p, idx_j in zip(proportions, idx_batchs)])
            proportions = np.array([p for p, idx_j in zip(proportions, idx_batchs)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batchs = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batchs, np.split(idx_k, proportions))]
        min_size = min([len(idx_j) for idx_j in idx_batchs])
    for i in range(num_users):
        np.random.shuffle(idx_batchs[i])
        dict_users[i] = idx_batchs[i]

    return dict_users


# CIFAR
def cifar_iid_smallset(dataset, num_data):
    """
    Sample I.I.D. client data from CIFAR dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    # num_items = int(num_data) 
    # all_idxs = [i for i in range(len(dataset))]

    # np.random.seed(2021)
    # data_idx = set(np.random.choice(all_idxs, num_items, replace=False))
    # return data_idx

    num_samples = len(dataset)
    num_classes = len(np.unique(dataset.targets))
    num_per_label_total = int(num_samples/num_classes)
    labels = np.array(dataset.targets)
    idxs = np.arange(len(labels))
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # label available
    label_list = [i for i in range(num_classes)]
    
    np.random.seed(2022)
    num_class = num_classes
    num_imgs = num_data
    num_per_label = int(num_imgs/num_class)
    rand_label = np.random.choice(label_list, num_class, replace=False)
    data_idx = np.array([])
    for y in rand_label:
        start = y*num_per_label_total
        data_idx = np.concatenate((data_idx, idxs[start:start+num_per_label]), axis=0)
    data_idx = data_idx.astype(int)
    return data_idx


def cifar_iid(dataset, num_users, *args, **kwargs):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    # num_items = int(len(dataset)/num_users)
    # dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    # np.random.seed(2021)
    # for i in range(num_users):
    #     select_set = set(np.random.choice(all_idxs, num_items,
    #                                          replace=False))
    #     all_idxs = list(set(all_idxs) - select_set)
    #     dict_users[i] = list(select_set)
    # return dict_users
    np.random.seed(2022)
    num_classes = len(np.unique(dataset.targets))
    shard_per_user = num_classes
    imgs_per_shard = int(len(dataset) / (num_users * shard_per_user))
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs_dict = {}
    for i in range(len(dataset)):
        label = dataset.targets[i]
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)
    
    rand_set_all = []
    if len(rand_set_all) == 0:
        for i in range(num_users):
            x = np.random.choice(np.arange(num_classes), shard_per_user, replace=False)
            rand_set_all.append(x)

    # divide and assign
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            # pdb.set_trace()
            x = np.random.choice(idxs_dict[label], imgs_per_shard, replace=False)
            rand_set.append(x)
        dict_users[i] = np.concatenate(rand_set)

    for key, value in dict_users.items():
        assert(len(np.unique(torch.tensor(dataset.targets)[value]))) == shard_per_user

    return dict_users


def cifar_noniid(dataset, num_users, *args, **kwargs):
    """
    Sample non-I.I.D client data from CIFAR dataset
    :param dataset:
    :param num_users:
    :return:
    """
    np.random.seed(2022)
    shard_per_user = 5
    imgs_per_shard = int(len(dataset) / (num_users * shard_per_user))
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    
    idxs_dict = {}
    for i in range(len(dataset)):
        label = dataset.targets[i]
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    num_classes = len(np.unique(dataset.targets))
    rand_set_all = []
    if len(rand_set_all) == 0:
        for i in range(num_users):
            x = np.random.choice(np.arange(num_classes), shard_per_user, replace=False)
            rand_set_all.append(x)

    # divide and assign
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            # pdb.set_trace()
            x = np.random.choice(idxs_dict[label], imgs_per_shard, replace=False)
            rand_set.append(x)
        dict_users[i] = np.concatenate(rand_set)

    label_sets = {}
    for key, value in dict_users.items():
        label_set = np.unique(torch.tensor(dataset.targets)[value])
        assert(len(label_set)) == shard_per_user
        label_sets[i] = [y for y in label_set]

    return dict_users, label_sets


def cifar_noniid_s(dataset, num_users, *args, **kwargs):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 50,000 training imgs
    shards = 2
    s = 0.0
    # num_per_user = 500
    # num_samples = int(num_per_user*num_users)
    num_samples = len(dataset)
    num_per_user = int(num_samples/num_users)
    num_imgs_iid = int(num_per_user * s)
    num_imgs_noniid = num_per_user - num_imgs_iid
    dict_users = {i: np.array([]) for i in range(num_users)}
    labels = np.array(dataset.targets)
    idxs = np.arange(len(dataset.targets))
    # iid labels
    idxs_labels = np.vstack((idxs, labels))
    iid_length = int(s*num_samples)
    iid_idxs = idxs_labels[0,:iid_length]
    # noniid labels
    noniid_idxs_labels = idxs_labels[:,iid_length:]
    idxs_noniid = noniid_idxs_labels[:, noniid_idxs_labels[1, :].argsort()]
    noniid_idxs = idxs_noniid[0, :]
    num_shards, num_imgs = num_users*shards, int(num_imgs_noniid/shards)
    idx_shard = [i for i in range(num_shards)]
    all_idxs = [int(i) for i in iid_idxs]
    np.random.seed(2022)
    for i in range(num_users):
        # allocate iid idxs
        # selected_set = set(np.random.choice(all_idxs, num_imgs_iid, replace=False))
        selected_set = set(all_idxs[:num_imgs_iid])
        all_idxs = list(set(all_idxs) - selected_set)
        dict_users[i] = np.concatenate((dict_users[i], np.array(list(selected_set))), axis=0)
        # allocate noniid idxs
        rand_set = set(np.random.choice(idx_shard, shards, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], noniid_idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
        # start = i*num_imgs_noniid
        # dict_users[i] = np.concatenate((dict_users[i], noniid_idxs[start:start+num_imgs_noniid]), axis=0)
        dict_users[i] = dict_users[i].astype(int)
        # np.random.shuffle(dict_users[i])
    return dict_users


def cifar_noniid_ss(dataset, num_users, noniid_s=20, local_size=1000, ratio=1, train=True):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 50,000 training imgs
    np.random.seed(2022)
    s = noniid_s/100
    num_per_user = local_size if train else 500
    num_classes = len(np.unique(dataset.targets))
    # noniid_labels = int(num_classes/10)*np.random.randint(2, 6, num_users)
    # noniid_labels_list = [[0,1,2], [2,3,4], [3,4,5,6,7], [5,6,7,8,9], [0,1,2,3,4,5,6,7]]
    # noniid_labels_list = [[0,1,2], [2,3,4], [4,5,6], [6,7,8], [8,9,0]]
    noniid_labels_list = [[0,1,2,3,4], [5,6,7,8,9]]
    # noniid_labels_list = [[0,1],[2,3],[4,5,6],[7,8,9],[5,6,7,8,9]]
    # -------------------------------------------------------
    # divide the first dataset
    num_imgs_iid = int(num_per_user*s)
    num_imgs_noniid = num_per_user - num_imgs_iid
    dict_users = {i: np.array([]) for i in range(num_users)}
    num_samples = len(dataset)
    num_per_label_total = int(num_samples/num_classes)
    labels1 = np.array(dataset.targets)
    idxs1 = np.arange(len(dataset.targets))
    # iid labels
    idxs_labels = np.vstack((idxs1, labels1))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # label available
    label_list = [i for i in range(num_classes)]
    # number of imgs has allocated per label
    # label_used = [2000 for i in range(num_classes)] if train else [500 for i in range(num_classes)]
    label_used = [0 for i in range(num_classes)]
    iid_per_label = int(num_imgs_iid/num_classes)
    iid_per_label_last = num_imgs_iid - (num_classes-1)*iid_per_label

    first_idx_label = [np.min(np.where(idxs_labels[1,:]==i)) for i in range(num_classes)]

    label_sets = {}
    np.random.seed(2022)
    for i in range(num_users):
        # allocate iid idxs
        label_cnt = 0
        for y in label_list:
            label_cnt = label_cnt + 1
            iid_num = iid_per_label
            start = first_idx_label[y]+label_used[y]
            if label_cnt == num_classes:
                iid_num = iid_per_label_last
            if (label_used[y]+iid_num)>num_per_label_total:
                start = first_idx_label[y]
                label_used[y] = 0
            dict_users[i] = np.concatenate((dict_users[i], idxs[start:start+iid_num]), axis=0)
            label_used[y] = label_used[y] + iid_num
        # allocate noniid idxs
        # num_labels = len(noniid_labels_list[(i%5)])
        # rand_label = np.random.choice(label_list, num_labels, replace=False)
        rand_label = np.random.choice(label_list, 2, replace=False)
        # rand_label = noniid_labels_list[(i%2)]
        label_sets[i] = [y for y in rand_label]
        noniid_labels = len(rand_label)
        # num_imgs_noniid = num_imgs[i%5] if train else local_size
        noniid_per_num = int(num_imgs_noniid/noniid_labels)
        noniid_per_num_last = num_imgs_noniid - noniid_per_num*(noniid_labels-1)
        label_cnt = 0
        for y in rand_label:
            label_cnt = label_cnt + 1
            noniid_num = noniid_per_num
            start = first_idx_label[y]+label_used[y]
            if label_cnt == noniid_labels:
                noniid_num = noniid_per_num_last
                
            if i%5==0:
                noniid_num = ratio*noniid_num

            if (label_used[y]+noniid_num)>num_per_label_total:
                start = first_idx_label[y]
                label_used[y] = 0
            dict_users[i] = np.concatenate((dict_users[i], idxs[start:start+noniid_num]), axis=0)
            label_used[y] = label_used[y] + noniid_num
        dict_users[i] = dict_users[i].astype(int)
    return dict_users, label_sets


def cifar_noniid_few(dataset, num_users, imgs_shot=100, n_way=3, ratio=10, *args, **kwargs):
    """
    Sample non-I.I.D client data from CIFAR10 dataset, few shot
    :param dataset:
    :param num_users:
    :return:
    """
    # 50,000 training imgs --
    # num_class = 2 # in default
    num_samples = len(dataset)
    num_classes = len(np.unique(dataset.targets))
    # num_imgs = int(num_samples/(num_class*num_users)) # indefault 500
    num_per_label_total = int(num_samples/num_classes)
    dict_users = {i: np.array([]) for i in range(num_users)}
    labels = np.array(dataset.targets)
    idxs = np.arange(len(labels))
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # label available
    label_list = [i for i in range(num_classes)]
    # number of imgs has allocated per label
    label_used = [0 for i in range(num_classes)]
    
    # divide and assign 2 shards/client
    np.random.seed(2022)
    # num_class = np.random.randint(2, 9, num_users)
    local_labels = n_way
    imgs_label = imgs_shot # 100 in default
    groups = 5
    user_per_group = int(num_users/groups)
    user_labels = []
    label_list_np = np.array(label_list)
    for i in range(num_users):
        start = (local_labels*(i//user_per_group))%num_classes
        if start+local_labels<=num_classes:
            user_labels.append(label_list_np[start:start+local_labels])
        else:
            user_labels.append(np.concatenate((label_list_np[start:num_classes], label_list_np[0:local_labels+start-num_classes]), axis=0))
    
    num_imgs = [int(imgs_label*local_labels) for i in range(num_users)]
    # num_imgs = [int(num_samples/num_users) for i in range(num_users)]
    # num_imgs = int(0.2*num_samples/num_users)*np.random.randint(2, 9, num_users)
    for i in range(num_users):
        num_per_label = int(num_imgs[i]/local_labels)
        num_per_label_last = num_imgs[i] - num_per_label*(local_labels-1)
        if i < user_per_group:
            num_per_label = ratio*imgs_label
            num_per_label_last = num_per_label
        # rand_label = np.random.choice(label_list, num_labels[i], replace=False)
        rand_label = user_labels[i]
        label_cnt = 0
        for y in rand_label:
            label_cnt = label_cnt + 1
            start = y*num_per_label_total+label_used[y]
            if label_cnt == local_labels:
                num_per_label = num_per_label_last
            if (label_used[y]+num_per_label)>num_per_label_total:
                start = y*num_per_label_total
                label_used[y] = 0
            dict_users[i] = np.concatenate((dict_users[i], idxs[start:start+num_per_label]), axis=0)
            label_used[y] = label_used[y] + num_per_label
    return dict_users


def cifar_noniid_lt(dataset, num_users, ratio=10, shards=3, *args, **kwargs):
    """
    Sample non-I.I.D client data from CIFAR10 dataset, long tail
    :param dataset:
    :param num_users:
    :return:
    """
    # 50,000 training imgs --
    num_samples = len(dataset)
    classes = np.unique(dataset.targets)
    num_classes = len(np.unique(dataset.targets))
    img_max = int(num_samples/(num_classes)) # indefault 500
    img_num_per_cls = []
    imb_factor = 1.0/ratio
    for cls_idx in range(num_classes):
        num = img_max * (imb_factor**(1.0*cls_idx / (num_classes - 1.0)))
        img_num_per_cls.append(int(num))
    dict_users = {i: np.array([]) for i in range(num_users)}

    labels = np.array(dataset.targets)
    data_idxs = np.arange(len(labels))
    new_data = []
    new_labels = []

    for the_class, the_img_num in zip(classes, img_num_per_cls):
        idx = np.where(labels == the_class)[0]
        # np.random.shuffle(idx)
        selec_idx = idx[:the_img_num]
        new_data.append(data_idxs[selec_idx])
        new_labels.append(labels[selec_idx])
    new_data = np.hstack(new_data)
    new_labels = np.hstack(new_labels)

    # sort labels
    idxs_labels = np.vstack((new_data, new_labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    
    # divide and assign 2 shards/client
    np.random.seed(2022)
    num_imgs_noniid = int(len(new_data)/num_users)
    noniid_idxs = idxs_labels[0, :]
    num_shards, num_imgs = num_users*shards, int(num_imgs_noniid/shards)
    idx_shard = [i for i in range(num_shards)]
    np.random.seed(2022)
    for i in range(num_users):
        # allocate noniid idxs
        rand_set = set(np.random.choice(idx_shard, shards, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], noniid_idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
        # start = i*num_imgs_noniid
        # dict_users[i] = np.concatenate((dict_users[i], noniid_idxs[start:start+num_imgs_noniid]), axis=0)
        dict_users[i] = dict_users[i].astype(int)
        # np.random.shuffle(dict_users[i])

    return dict_users


def cifar_noniid_lt1(dataset, num_users, ratio=10, beta=0.5, train=True, *args, **kwargs):
    """
    Sample non-I.I.D client data from CIFAR10 dataset, long tail
    :param dataset:
    :param num_users:
    :return:
    """
    # 50,000 training imgs --
    num_samples = len(dataset)
    classes = np.unique(dataset.targets)
    num_classes = len(np.unique(dataset.targets))
    img_max = int(num_samples/(num_classes)) # indefault 500
    img_num_per_cls = []
    imb_factor = 1.0/ratio
    for cls_idx in range(num_classes):
        num = img_max * (imb_factor**(1.0*cls_idx / (num_classes - 1.0)))
        img_num_per_cls.append(int(num))
    dict_users = {i: np.array([]) for i in range(num_users)}

    labels = np.array(dataset.targets)
    data_idxs = np.arange(len(labels))
    new_data = []
    new_labels = []

    for the_class, the_img_num in zip(classes, img_num_per_cls):
        idx = np.where(labels == the_class)[0]
        # np.random.shuffle(idx)
        selec_idx = idx[:the_img_num]
        new_data.append(data_idxs[selec_idx])
        new_labels.append(labels[selec_idx])
    new_data = np.hstack(new_data)
    new_labels = np.hstack(new_labels)
    
    min_size = 0
    min_require_size = 5 #0.1*int(num_samples/num_users)
    dict_users = {i: np.array([]) for i in range(num_users)}
    idx_batchs = [[] for _ in range(num_users)]
    labels = new_labels #np.array(dataset.targets)
    np.random.seed(2022)
    while min_size < min_require_size:
        idx_batchs = [[] for _ in range(num_users)]
        for k in range(num_classes):
            idx_k = np.where(labels == k)[0]
            # np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, num_users)) #+ (0.01/num_users)
            # proportions = np.array([p * (len(idx_j) < num_samples / num_users) for p, idx_j in zip(proportions, idx_batchs)])
            proportions = np.array([p for p, idx_j in zip(proportions, idx_batchs)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batchs = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batchs, np.split(idx_k, proportions))]
        min_size = min([len(idx_j) for idx_j in idx_batchs])
    for i in range(num_users):
        np.random.shuffle(idx_batchs[i])
        dict_users[i] = idx_batchs[i]

    return dict_users


def cifar_noniid_dir(dataset, num_users, noniid_beta=0.1, train=True):
    """
    Sample non-I.I.D client data from CIFAR dataset via Dirichlet
    :param dataset:
    :param num_users:
    :return:
    """
    # 50,000 training imgs
    beta = noniid_beta
    num_samples = len(dataset)
    num_classes = len(np.unique(dataset.targets))
    min_size = 0
    min_require_size = 10 #150 if train else 30 #0.1*int(num_samples/num_users)
    dict_users = {i: np.array([]) for i in range(num_users)}
    idx_batchs = [[] for _ in range(num_users)]
    labels = np.array(dataset.targets)
    idxs = np.arange(len(dataset.targets))
    np.random.seed(2022)
    while min_size < min_require_size:
        idx_batchs = [[] for _ in range(num_users)]
        for k in range(num_classes):
            idx_k = np.where(labels == k)[0]
            # np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, num_users)) #+ (0.1/num_users)
            # proportions = np.array([p * (len(idx_j) < num_samples / num_users) for p, idx_j in zip(proportions, idx_batchs)])
            proportions = np.array([p for p, idx_j in zip(proportions, idx_batchs)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batchs = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batchs, np.split(idx_k, proportions))]
        min_size = min([len(idx_j) for idx_j in idx_batchs])
    for i in range(num_users):
        np.random.shuffle(idx_batchs[i])
        dict_users[i] = idx_batchs[i]

    return dict_users


def cifar_iid_im_dir(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR dataset via Dirichlet
    :param dataset:
    :param num_users:
    :return:
    """
    # 50,000 training imgs
    beta = 0.5
    num_samples = len(dataset)
    num_classes = len(np.unique(dataset.targets))
    min_size = 0
    min_require_size = 0.1*int(num_samples/num_users)
    dict_users = {i: np.array([]) for i in range(num_users)}
    idx_batchs = [[] for _ in range(num_users)]
    labels = np.array(dataset.targets)
    # idxs = np.arange(len(dataset.targets))
    np.random.seed(2022)
    while min_size < min_require_size:
        proportions = np.random.dirichlet(np.repeat(beta, num_users))
        proportions = proportions/proportions.sum()
        min_size = np.min(proportions*num_samples)
    proportions = (np.cumsum(proportions)*num_samples).astype(int)[:-1]
    idxs = np.random.permutation(num_samples)
    idx_batchs = np.split(idxs,proportions)
    for i in range(num_users):
        np.random.shuffle(idx_batchs[i])
        dict_users[i] = idx_batchs[i]

    return dict_users


def cifar100_noniid_ss(dataset, num_users, noniid_s=20, local_size=1000, train=True):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 50,000 training imgs
    np.random.seed(2022)
    s = noniid_s/100
    num_per_user = local_size if train else 500
    num_classes = len(np.unique(dataset.targets))
    sup_classes, names = cifar100_superclass_label_pair()
    noniid_labels_list = sup_classes
    # noniid_labels_list = [[int(i) for i in range(20)], [int(i) for i in range(20,40)], [int(i) for i in range(40,60)], 
    #                                                    [int(i) for i in range(60,80)], [int(i) for i in range(80,100)]]
    # -------------------------------------------------------
    # divide the first dataset
    num_imgs_iid = int(num_per_user*s)
    num_imgs_noniid = num_per_user - num_imgs_iid
    dict_users = {i: np.array([]) for i in range(num_users)}
    num_samples = len(dataset)
    num_per_label_total = int(num_samples/num_classes)
    labels1 = np.array(dataset.targets)
    idxs1 = np.arange(len(dataset.targets))
    # iid labels
    idxs_labels = np.vstack((idxs1, labels1))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # label available
    label_list = [i for i in range(num_classes)]
    # number of imgs has allocated per label
    label_used = [0 for i in range(num_classes)]
    iid_per_label = int(num_imgs_iid/num_classes)
    iid_per_label_last = num_imgs_iid - (num_classes-1)*iid_per_label

    np.random.seed(2022)
    for i in range(num_users):
        # allocate iid idxs
        label_cnt = 0
        for y in label_list:
            label_cnt = label_cnt + 1
            iid_num = iid_per_label
            start = y*num_per_label_total+label_used[y]
            if label_cnt == num_classes:
                iid_num = iid_per_label_last
            if (label_used[y]+iid_num)>num_per_label_total:
                start = y*num_per_label_total
                label_used[y] = 0
            dict_users[i] = np.concatenate((dict_users[i], idxs[start:start+iid_num]), axis=0)
            label_used[y] = label_used[y] + iid_num
        # allocate noniid idxs
        # rand_label = noniid_labels_list[i%20]
        rand_label = np.random.choice(label_list, 20, replace=False)
        noniid_labels = len(rand_label)
        noniid_per_num = int(num_imgs_noniid/noniid_labels)
        noniid_per_num_last = num_imgs_noniid - noniid_per_num*(noniid_labels-1)
        label_cnt = 0
        for y in rand_label:
            label_cnt = label_cnt + 1
            noniid_num = noniid_per_num
            start = y*num_per_label_total+label_used[y]
            if label_cnt == noniid_labels:
                noniid_num = noniid_per_num_last
            if (label_used[y]+noniid_num)>num_per_label_total:
                start = y*num_per_label_total
                label_used[y] = 0
            dict_users[i] = np.concatenate((dict_users[i], idxs[start:start+noniid_num]), axis=0)
            label_used[y] = label_used[y] + noniid_num
        dict_users[i] = dict_users[i].astype(int)
    return dict_users



## transform

def _rotate_image(image, angle):
    if angle is None:
        return image

    image = transforms.functional.rotate(image, angle=angle)
    return image

def get_transform_mnist(angle=None):
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(lambda x: _rotate_image(x, angle)),
                    ])
    return transform

def get_transform_cifar_train(angle=None):
    transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),  
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
                transforms.Lambda(lambda x: _rotate_image(x, angle)),
                    ])
    return transform

def get_transform_cifar_test(angle=None):
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
                transforms.Lambda(lambda x: _rotate_image(x, angle)),
                    ])
    return transform

def mnist_rotated():
    angle_list = [30 * x for x in range(5)]

    train_datasets = [datasets.MNIST('data', train=True, download=True, transform=get_transform_mnist(angle_list[index])) for index in range(5)]

    test_datasets = [datasets.MNIST('data', train=False, download=True, transform=get_transform_mnist(angle_list[index])) for index in range(5)]

    print("Roatated-MNIST Data Loading...")
    return train_datasets, test_datasets


def fmnist_rotated():
    angle_list = [30 * x for x in range(5)]

    train_datasets = [datasets.FashionMNIST('data', train=True, download=True, transform=get_transform_mnist(angle_list[index])) for index in range(5)]

    test_datasets = [datasets.FashionMNIST('data', train=False, download=True, transform=get_transform_mnist(angle_list[index])) for index in range(5)]

    print("Roatated-FashionMNIST Data Loading...")
    return train_datasets, test_datasets


def cifar_rotated():
    angle_list = [30 * x for x in range(5)]

    train_datasets = [datasets.CIFAR10('data', train=True, download=True, transform=get_transform_cifar_train(angle_list[index])) for index in range(5)]

    test_datasets = [datasets.CIFAR10('data', train=False, download=True, transform=get_transform_cifar_test(angle_list[index])) for index in range(5)]

    print("Roatated-CIFAR10 Data Loading...")
    return train_datasets, test_datasets

## --------------------------------------------------
## loading dataset
## --------------------------------------------------
def mnist():
    trainset = datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

    testset = datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    print("MNIST Data Loading...")
    return trainset, testset


def fmnist():
    trainset = datasets.FashionMNIST('data', train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))

    testset = datasets.FashionMNIST('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    print("FashionMNIST Data Loading...")
    return trainset, testset


def emnist():
    trainset = datasets.EMNIST('data', 'byclass', train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))

    testset = datasets.EMNIST('data', 'byclass', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    print("EMNIST Data Loading...")
    return trainset, testset


def svhn():
    trainset = datasets.SVHN('data', split='train', download=True,
                       transform=transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0.5))
           ]))

    testset = datasets.SVHN('data', split='test', download=True,
                        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ]))
    print("SVHN Data Loading...")
    return trainset, testset


def cifar10():
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.ToPILImage(mode='RGB'),
        transforms.Resize([224, 224]),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])

    transform_test = transforms.Compose([
        transforms.ToPILImage(mode='RGB'),
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(
        root='data', train=True, download=True, transform=transform_train)  #
    testset = datasets.CIFAR10(
        root='data', train=False, download=True, transform=transform_test)
    print("CIFAR10 Data Loading...")
    return trainset, testset


def cifar100():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR100(
        root='data', train=True, download=True, transform=transform_train)  #
    testset = datasets.CIFAR100(
        root='data', train=False, download=True, transform=transform_test)
    print("CIFAR100 Data Loading...")
    return trainset, testset

def cifar100_superclass_label_pair():
    CIFAR100_LABELS_LIST = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm'
    ]
    sclass = []
    sclass.append(['beaver', 'dolphin', 'otter', 'seal', 'whale'])                      #aquatic mammals
    sclass.append(['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'])               #fish
    sclass.append(['orchid', 'poppy', 'rose', 'sunflower', 'tulip'])                    #flowers
    sclass.append(['bottle', 'bowl', 'can', 'cup', 'plate'])                            #food
    sclass.append(['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'])              #fruit and vegetables
    sclass.append(['clock', 'keyboard', 'lamp', 'telephone', 'television'])             #household electrical devices
    sclass.append(['bed', 'chair', 'couch', 'table', 'wardrobe'])                       #household furniture
    sclass.append(['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'])           #insects
    sclass.append(['bear', 'leopard', 'lion', 'tiger', 'wolf'])                         #large carnivores
    sclass.append(['bridge', 'castle', 'house', 'road', 'skyscraper'])                  #large man-made outdoor things
    sclass.append(['cloud', 'forest', 'mountain', 'plain', 'sea'])                      #large natural outdoor scenes
    sclass.append(['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'])            #large omnivores and herbivores
    sclass.append(['fox', 'porcupine', 'possum', 'raccoon', 'skunk'])                   #medium-sized mammals
    sclass.append(['crab', 'lobster', 'snail', 'spider', 'worm'])                       #non-insect invertebrates
    sclass.append(['baby', 'boy', 'girl', 'man', 'woman'])                              #people
    sclass.append(['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'])               #reptiles
    sclass.append(['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'])                  #small mammals
    sclass.append(['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'])  #trees
    sclass.append(['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'])            #vehicles 1
    sclass.append(['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'])             #vehicles 2
    
    labels_pair = [[cid for cid in range(100) if CIFAR100_LABELS_LIST[cid] in sclass[gid]] for gid in range(20)]

    return labels_pair, sclass

def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.

    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]


def label_binary(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.

    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([ 0,  0, 0,  0,  0,  1,  1,  1, 1,  1])
    return coarse_labels[targets]


class DatasetSplit(Dataset):
    """
    An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset, index=None, return_idx=False, label_set=None, relabel=False):
        self.dataset = dataset
        self.idxs = [int(i) for i in index] if index is not None else [int(i) for i in range(len(dataset))]
        self.return_index = return_idx
        self.label_set = label_set
        self.relabel= relabel
        if self.relabel:
            # print(label_set)
            num_classes = len(np.unique(dataset.targets))
            cls_labels = self.label_set if label_set is not None else [y for y in range(num_classes)]
            self.y_relabel = {y: y if y in cls_labels else num_classes for y in range(num_classes)}

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if self.relabel:
            label = self.y_relabel[label]
        if self.return_index:
            return image, label, item
        else:
            return image, label



def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    train_dataset = []
    test_dataset = []
    user_groups_train = {}
    user_groups_test = {}
    train_loader = []
    test_loader = []
    global_test_loader = []
    val_idx = []
    val_loader = []
    val_size = 300

    if args.dataset in ['cifar', 'cifar10']:
        train_dataset, test_dataset = cifar10()
        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups_train = cifar_iid(train_dataset, args.num_users)
            user_groups_test = cifar_iid(test_dataset, args.num_users)
            val_idx = cifar_iid_smallset(test_dataset, val_size)
            print('IID Data Loading---')
        else:
            # Sample Non-IID user data from Mnist
            if args.dir:
                # Chose euqal splits for every user
                user_groups_train = cifar_noniid_dir(train_dataset, args.num_users, args.noniid_beta, train=True)
                user_groups_test = cifar_noniid_dir(test_dataset, args.num_users, args.noniid_beta, train=False)
            elif args.lt:
                user_groups_train = cifar_noniid_lt1(train_dataset, args.num_users, args.imb_ratio, train=True)
                user_groups_test = cifar_noniid_lt1(test_dataset, args.num_users, args.imb_ratio, train=False)
                # user_groups_train = cifar_noniid_im(train_dataset, args.num_users, train=True)
                # user_groups_test = cifar_noniid_im(test_dataset, args.num_users, train=False)
            else:
                user_groups_train, user_label_train = cifar_noniid_ss(train_dataset, args.num_users, args.noniid_s, args.local_size, train=True)
                user_groups_test, user_label_test = cifar_noniid_ss(test_dataset, args.num_users, args.noniid_s, args.local_size, train=False)
            val_idx = cifar_iid_smallset(test_dataset, val_size)
            print('non-IID Data Loading---')

        # train_dataset.targets = label_binary(train_dataset.targets)
        # test_dataset.targets = label_binary(test_dataset.targets)

    elif args.dataset == 'cifar100':
        train_dataset, test_dataset = cifar100()
        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups_train = cifar_iid(train_dataset, args.num_users)
            user_groups_test = cifar_iid(test_dataset, args.num_users)
            val_idx = cifar_iid_smallset(test_dataset, val_size)
            print('IID Data Loading---')
        else:
            # Sample Non-IID user data from Mnist
            if args.dir:
                # Chose uneuqal splits for every user
                # raise NotImplementedError()
                user_groups_train = cifar_noniid_dir(train_dataset, args.num_users, args.noniid_beta, train=True)
                user_groups_test = cifar_noniid_dir(test_dataset, args.num_users, args.noniid_beta, train=False)
            else:
                # Chose euqal splits for every user
                user_groups_train = cifar100_noniid_ss(train_dataset, args.num_users, args.noniid_s, args.local_size, train=True)
                user_groups_test = cifar100_noniid_ss(test_dataset, args.num_users, args.noniid_s, args.local_size, train=False)
            val_idx = cifar_iid_smallset(test_dataset, val_size)
            print('non-IID Data Loading---')

    elif args.dataset == 'cinic':
        train_dataset, test_dataset = cinic10()
        # sample training data amongst users
        if args.iid:
            # Sample IID user data from CINIC-10
            user_groups_train = cifar_iid(train_dataset, args.num_users)
            user_groups_test = cifar_iid(test_dataset, args.num_users)
            val_idx = cifar_iid_smallset(test_dataset, val_size)
        else:
            # Sample Non-IID user data from CINIC-10
            if args.dir:
                raise NotImplementedError()
                # Chose uneuqal splits for every user
                # user_groups_train = cifar_noniid_dir(train_dataset, args.num_users, train=True)
                # user_groups_test = cifar_noniid_dir(test_dataset, args.num_users, train=False)
            else:
                # Chose euqal splits for every user
                user_groups_train = cifar_noniid_ss(train_dataset, args.num_users, args.noniid_s, args.local_size, train=True)
                user_groups_test = cifar_noniid_ss(test_dataset, args.num_users, args.noniid_s, args.local_size, train=False)
            val_idx = cifar_iid_smallset(test_dataset, val_size)

    elif args.dataset in ['mnist', 'fmnist']:
        if args.dataset == 'mnist':
            train_dataset, test_dataset = mnist()
        if args.dataset == 'fmnist':
            train_dataset, test_dataset = fmnist()
        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups_train = mnist_iid(train_dataset, args.num_users)
            user_groups_test = mnist_iid(test_dataset, args.num_users)
            val_idx = cifar_iid_smallset(test_dataset, val_size)
            print('IID Data Loading---')
        else:
            # Sample Non-IID user data from Mnist
            if args.dir:
                # Chose uneuqal splits for every user
                user_groups_train = mnist_noniid_dir(train_dataset, args.num_users, args.noniid_beta, train=True)
                user_groups_test = mnist_noniid_dir(test_dataset, args.num_users, args.noniid_beta, train=False)
            elif args.lt:
                user_groups_train = cifar_noniid_lt(train_dataset, args.num_users, args.imb_ratio, train=True)
                user_groups_test = cifar_noniid_lt(test_dataset, args.num_users, args.imb_ratio, train=False)
            else:
                # Chose euqal splits for every user
                user_groups_train = mnist_noniid_ss(train_dataset, args.num_users, args.noniid_s, args.local_size, train=True)
                user_groups_test = mnist_noniid_ss(test_dataset, args.num_users, args.noniid_s, args.local_size, train=False)
            val_idx = cifar_iid_smallset(test_dataset, val_size)
            print('non-IID Data Loading---')

    elif args.dataset == 'emnist':
        train_dataset, test_dataset = emnist()
        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups_train = mnist_iid(train_dataset, args.num_users)
            user_groups_test = mnist_iid(test_dataset, args.num_users)
            val_idx = cifar_iid_smallset(test_dataset, val_size)
            print('IID Data Loading---')
        else:
            # Sample Non-IID user data from Mnist
            if args.dir:
                # Chose uneuqal splits for every user
                user_groups_train = emnist_noniid_dir(train_dataset, args.num_users, train=True, noniid_beta=args.noniid_beta)
                user_groups_test = emnist_noniid_dir(test_dataset, args.num_users, train=False, noniid_beta=args.noniid_beta)
            else:
                user_groups_train = emnist_noniid_ss(train_dataset, args.num_users, train=True, noniid_s=args.noniid_s)
                user_groups_test = emnist_noniid_ss(test_dataset, args.num_users, train=False, noniid_s=args.noniid_s)

            val_idx = cifar_iid_smallset(test_dataset, val_size)
            print('non-IID Data Loading---')

    elif args.dataset == 'rotated_mnist':
        train_datasets, test_datasets = mnist_rotated()
        train_dataset = train_datasets[0]
        test_dataset = test_datasets[0]
        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups_train = mnist_iid(train_dataset, args.num_users)
            user_groups_test = mnist_iid(test_dataset, args.num_users)
            val_idx = cifar_iid_smallset(test_dataset, val_size)
            print('IID Data Loading---')
        else:
            # Sample Non-IID user data from Mnist
            if args.dir:
                # Chose uneuqal splits for every user
                user_groups_train = mnist_noniid_dir(train_dataset, args.num_users, args.noniid_beta, train=True)
                user_groups_test = mnist_noniid_dir(test_dataset, args.num_users, args.noniid_beta, train=False)
            else:
                # Chose euqal splits for every user
                user_groups_train = mnist_noniid_ss(train_dataset, args.num_users, args.noniid_s, args.local_size)
                user_groups_test = mnist_noniid_ss(test_dataset, args.num_users, args.noniid_s, args.local_size)
            val_idx = cifar_iid_smallset(test_dataset, val_size)
            print('non-IID Data Loading---')

    elif args.dataset == 'rotated_fmnist':
        train_datasets, test_datasets = fmnist_rotated()
        train_dataset = train_datasets[0]
        test_dataset = test_datasets[0]
        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups_train = mnist_iid(train_dataset, args.num_users)
            user_groups_test = mnist_iid(test_dataset, args.num_users)
            val_idx = cifar_iid_smallset(test_dataset, val_size)
            print('IID Data Loading---')
        else:
            # Sample Non-IID user data from Mnist
            if args.dir:
                # Chose uneuqal splits for every user
                user_groups_train = mnist_noniid_dir(train_dataset, args.num_users, args.noniid_beta, train=True)
                user_groups_test = mnist_noniid_dir(test_dataset, args.num_users, args.noniid_beta, train=False)
            else:
                # Chose euqal splits for every user
                user_groups_train = mnist_noniid_ss(train_dataset, args.num_users, args.noniid_s, args.local_size)
                user_groups_test = mnist_noniid_ss(test_dataset, args.num_users, args.noniid_s, args.local_size)
            val_idx = cifar_iid_smallset(test_dataset, val_size)
            print('non-IID Data Loading---')

    elif args.dataset == 'rotated_cifar':
        train_datasets, test_datasets = cifar_rotated()
        train_dataset = train_datasets[0]
        test_dataset = test_datasets[0]
        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups_train = cifar_iid(train_dataset, args.num_users)
            user_groups_test = cifar_iid(test_dataset, args.num_users)
            val_idx = cifar_iid_smallset(test_dataset, val_size)
            print('IID Data Loading---')
        else:
            # Sample Non-IID user data from Mnist
            if args.dir:
                # Chose euqal splits for every user
                user_groups_train = cifar_noniid_dir(train_dataset, args.num_users, args.noniid_beta, train=True)
                user_groups_test = cifar_noniid_dir(test_dataset, args.num_users, args.noniid_beta, train=False)
            elif args.lt:
                user_groups_train = cifar_noniid_lt(train_dataset, args.num_users, args.imb_ratio, train=True)
                user_groups_test = cifar_noniid_lt(test_dataset, args.num_users, args.imb_ratio, train=False)
            else:
                user_groups_train = cifar_noniid_ss(train_dataset, args.num_users, args.noniid_s, args.local_size, train=True)
                user_groups_test = cifar_noniid_ss(test_dataset, args.num_users, args.noniid_s, args.local_size, train=False)
            val_idx = cifar_iid_smallset(test_dataset, val_size)
            print('non-IID Data Loading---')

    ## --------------------------------------------------------------------------------------------------------
    ## data allocation
    if args.dataset in ['mnist', 'fmnist', 'emnist', 'cifar','cifar10', 'cifar100']:
        for idx in range(args.num_users):
            loader1 = DataLoader(DatasetSplit(train_dataset, user_groups_train[idx], args.return_idx),
                                batch_size=args.local_bs, shuffle=True, drop_last=True)
            loader2 = DataLoader(DatasetSplit(test_dataset, user_groups_test[idx]),
                                batch_size=args.local_bs, shuffle=False, drop_last=True)
            train_loader.append(loader1)
            test_loader.append(loader2)

        global_test_loader = DataLoader(test_dataset, batch_size=args.local_bs, shuffle=False)
        val_loader = DataLoader(DatasetSplit(test_dataset, val_idx),
                                batch_size=args.local_bs, shuffle=True)

    elif args.dataset in ['rotated_mnist', 'rotated_fmnist','rotated_cifar']:
        for idx in range(args.num_users):
            cluster_idx = int(idx%5)
            loader1 = DataLoader(DatasetSplit(train_datasets[cluster_idx], user_groups_train[idx], args.return_idx),
                                batch_size=args.local_bs, shuffle=True)
            loader2 = DataLoader(DatasetSplit(test_datasets[cluster_idx], user_groups_test[idx]),
                                batch_size=args.local_bs, shuffle=False)
            train_loader.append(loader1)
            test_loader.append(loader2)

        global_test_loader = DataLoader(test_datasets[0], batch_size=args.local_bs, shuffle=False)
        val_loader = DataLoader(DatasetSplit(test_datasets[0], val_idx),
                                batch_size=args.local_bs, shuffle=True)

    else:
        raise NotImplementedError()


    return train_loader, test_loader, global_test_loader, val_loader


if __name__ =='__main__':
    pass
