import copy
import torch
import types
import math
import numpy as np
from scipy import stats
# import networkx as nx
from torch import nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import cm
import matplotlib.pyplot as plt



def tSNE_visualization(test_embeddings, test_targets, dataset, train_rule, path, idx):
    tsne = TSNE(2, verbose=1)
    tsne_proj = tsne.fit_transform(test_embeddings)
    # Plot those points as a scatter plot and label them based on the pred labels
    cmap = cm.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(8,8))
    num_categories = 10
    for lab in range(num_categories):
        indices = test_targets==lab
        ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
    ax.legend(fontsize='large', markerscale=2)
    plt.savefig('{}/{}_{}_tsne_{}.pdf'.format(path, dataset, train_rule, idx))
    # plt.show()
    return


class LabelSmoothingLoss(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def pairwise(data):
    """ Simple generator of the pairs (x, y) in a tuple such that index x < index y.
    Args:
    data Indexable (including ability to query length) containing the elements
    Returns:
    Generator over the pairs of the elements of 'data'
    """
    n = len(data)
    for i in range(n):
        for j in range(i, n):
            yield (data[i], data[j])


def get_protos(protos):
    """
    Returns the average of the feature embeddings of samples from per-class.
    """
    protos_mean = {}
    for [label, proto_list] in protos.items():
        proto = 0 * proto_list[0]
        for i in proto_list:
            proto += i
        protos_mean[label] = proto / len(proto_list)

    return protos_mean


def get_covariance(feature_dict):
    """
    Returns the average of the feature embeddings of samples from per-class.
    """
    covariance = 0
    for [label, feature_list] in feature_dict.items():
        proto = 0 * feature_list[0]
        covar = 0 * torch.matmul(feature_list[0].reshape(-1,1), feature_list[0].reshape(1,-1))
        for f in feature_list:
            proto += f
            covar += torch.matmul(f.reshape(-1,1), f.reshape(1,-1)) 
        f_mean = proto / len(feature_list)
        covariance += covar - len(feature_list)*torch.matmul(f_mean.reshape(-1,1), f_mean.reshape(1,-1))

    return covariance


def get_global_protos(model, dataset, device):
    train_data = iter(dataset)
    protos_list = {}
    for inputs, labels in train_data:
        inputs, labels = inputs.to(device), labels.to(device)
        features, outputs = model(inputs)
        protos = features.clone().detach()
        for i in range(len(labels)):
            if labels[i].item() in protos_list.keys():
                protos_list[labels[i].item()].append(protos[i,:])
            else:
                protos_list[labels[i].item()] = [protos[i,:]]
    global_protos = get_protos(protos_list)
    return global_protos


def get_protos_average(protos1, protos2, lam=0):
    """
    Returns the average of the feature embeddings of samples from per-class.
    """
    protos_avg = {}
    for [label, proto] in protos1.items():
        protos_avg[label] = lam*protos1[label] + (1-lam)*protos2[label]
    return protos_avg


def protos_aggregation(local_protos_list, local_sizes_list, global_protos_old=None):
    if local_protos_list is None or local_protos_list[0] is None:
        return None
    agg_protos_label = {}
    agg_sizes_label = {}
    for idx in range(len(local_protos_list)):
        local_protos = local_protos_list[idx]
        local_sizes = local_sizes_list[idx]
        if len(local_protos)<2:
            continue
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
                agg_sizes_label[label].append(local_sizes[label])
            else:
                agg_protos_label[label] = [local_protos[label]]
                agg_sizes_label[label] = [local_sizes[label]]

    for [label, proto_list] in agg_protos_label.items():
        sizes_list = agg_sizes_label[label]
        proto = 0 * proto_list[0]
        for i in range(len(proto_list)):
            proto += sizes_list[i] * proto_list[i]
        agg_protos_label[label] = proto / sum(sizes_list)

    if global_protos_old is not None:
        for label in global_protos_old.keys():
            # sim = F.cosine_similarity(agg_protos_label[label], global_protos_old[label])
            if label in agg_protos_label:
                sim = 1.0
                global_protos_old[label] = sim*agg_protos_label[label] + (1-sim)*global_protos_old[label]
        agg_protos_label = global_protos_old

    return agg_protos_label


def classwise_feature_aggregation(local_features, local_sizes_list):
    agg_protos_label = {}
    agg_sizes_label = {}
    for idx in range(len(local_features)):
        local_protos = local_features[idx]
        local_sizes = local_sizes_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label].mean(dim=0))
                agg_sizes_label[label].append(local_sizes[label])
            else:
                agg_protos_label[label] = [local_protos[label].mean(dim=0)]
                agg_sizes_label[label] = [local_sizes[label]]

    for [label, proto_list] in agg_protos_label.items():
        sizes_list = agg_sizes_label[label]
        proto = 0 * proto_list[0]
        for i in range(len(proto_list)):
            proto += sizes_list[i] * proto_list[i]
        agg_protos_label[label] = proto / sum(sizes_list)

    return agg_protos_label


def feature_augmentation():
    pass


def entropy(probs, base=None):
    n_labels = probs.shape[0]
    if n_labels <= 1:
        return 0.0
    ent = 0.0
    # Compute entropy
    # base = e if base is None else base
    for i in range(n_labels):
        ent -= probs[i] * torch.log(probs[i]+1e-8)
    return ent


def average_tensor_weighted(w, avg_weight):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    weight = torch.tensor(avg_weight)
    agg_w = weight/(weight.sum(dim=0))
    for key in w_avg.keys():
        w_avg[key] = torch.zeros_like(w_avg[key]).float()
        for i in range(len(w)):
            w_avg[key] += agg_w[i]*w[i][key]
    return w_avg


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def average_weights_mom(w, avg_weight, w_old, beta=0.5, **kwargs):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    weight = torch.tensor(avg_weight)
    agg_w = weight/(weight.sum(dim=0))
    for key in w_avg.keys():
        w_avg[key] = torch.zeros_like(w_avg[key])
        for i in range(len(w)):
            w_avg[key] += agg_w[i]*w[i][key]
        w_avg[key] = (1-beta)*w_old[key] + beta*w_avg[key]
    return w_avg


def average_weights_server_rate(w, avg_weight, w_old, beta=0.5, **kwargs):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    weight = torch.tensor(avg_weight)
    agg_w = weight/(weight.sum(dim=0))
    for key in w_avg.keys():
        w_avg[key] = torch.zeros_like(w_avg[key])
        for i in range(len(w)):
            w_avg[key] += agg_w[i]*w[i][key]
        w_avg[key] = (1+beta)*w_avg[key] - beta*w_old[key]
    return w_avg


def average_weights_weighted(w, avg_weight):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    weight = torch.tensor(avg_weight)
    agg_w = weight/(weight.sum(dim=0))
    for key in w_avg.keys():
        w_avg[key] = torch.zeros_like(w_avg[key]).float()
        for i in range(len(w)):
            w_avg[key] += agg_w[i]*w[i][key]
        # w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def variance_weights(w, wkeys=None):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    keys= wkeys if wkeys is not None else w_avg.keys()
    wg = [torch.cat(tuple([w[i][key].flatten() for key in keys]), dim=-1) for i in range(len(w))]
    var = torch.stack(wg).var(dim=0,unbiased=False).sum()

    return var


def cls_moving_avg(model, w, lam=0.0):
    keys = model.classifier_weight_keys
    w_old = copy.deepcopy(model.state_dict())
    w_new = copy.deepcopy(w)
    for key in w_new.keys():
        if key in keys:
            w_new[key] = lam*w_old[key]+(1-lam)*w_new[key]
    return w_new


def agg_classifier_weighted(w, local_sizes, keys, idx):
    """
    Returns the average of the weights.
    """
    w_0 = copy.deepcopy(w[idx])
    wg = [torch.cat(tuple([w[i][key].flatten() for key in keys]), dim=-1) for i in range(len(w))]
    for key in keys:
        w_0[key] = torch.zeros_like(w_0[key])
    wc = 0
    for i in range(len(w)):
        # wi = max(0, torch.cosine_similarity(wg[i], wg[idx], dim=-1).item())
        # wi = torch.exp(-100.0*torch.square(wg[i]-wg[idx]).sum()).item()
        kl_i = F.kl_div(local_sizes[idx].log(), local_sizes[i], reduction='none').sum()
        wi = torch.exp(-10.0 * kl_i)
        # if idx==1:
        #     print(wi)
        wc += wi
        for key in keys:
            w_0[key] += wi*w[i][key]
    for key in keys:
        w_0[key] = torch.div(w_0[key], wc)
    return w_0


def agg_classifier_weighted_p(w, avg_weight, keys, idx):
    """
    Returns the average of the weights.
    """
    w_0 = copy.deepcopy(w[idx])
    for key in keys:
        w_0[key] = torch.zeros_like(w_0[key])
    wc = 0
    for i in range(len(w)):
        wi = avg_weight[i]
        wc += wi
        for key in keys:
            w_0[key] += wi*w[i][key]
    for key in keys:
        w_0[key] = torch.div(w_0[key], wc)
    return w_0

# --------------------------------------------------------------------- #
# Gradient access

def grad_of(tensor):
    """ Get the gradient of a given tensor, make it zero if missing.
    Args:
    tensor Given instance of/deriving from Tensor
    Returns:
    Gradient for the given tensor
    """
    # Get the current gradient
    grad = tensor.grad
    if grad is not None:
        return grad
    # Make and set a zero-gradient
    grad = torch.zeros_like(tensor)
    tensor.grad = grad
    return grad


def grads_of(tensors):
    """ Iterate of the gradients of the given tensors, make zero gradients if missing.
    Args:
    tensors Generator of/iterable on instances of/deriving from Tensor
    Returns:
    Generator of the gradients of the given tensors, in emitted order
    """
    return (grad_of(tensor) for tensor in tensors)

# ---------------------------------------------------------------------------- #
# "Flatten" and "relink" operations

def relink(tensors, common):
    """ "Relink" the tensors of class (deriving from) Tensor by making them point to another contiguous segment of memory.
    Args:
    tensors Generator of/iterable on instances of/deriving from Tensor, all with the same dtype
    common  Flat tensor of sufficient size to use as underlying storage, with the same dtype as the given tensors
    Returns:
    Given common tensor
    """
    # Convert to tuple if generator
    if isinstance(tensors, types.GeneratorType):
        tensors = tuple(tensors)
    # Relink each given tensor to its segment on the common one
    pos = 0
    for tensor in tensors:
        npos = pos + tensor.numel()
        tensor.data = common[pos:npos].view(*tensor.shape)
        pos = npos
    # Finalize and return
    common.linked_tensors = tensors
    return common


def flatten(tensors):
    """ "Flatten" the tensors of class (deriving from) Tensor so that they all use the same contiguous segment of memory.
    Args:
    tensors Generator of/iterable on instances of/deriving from Tensor, all with the same dtype
    Returns:
    Flat tensor (with the same dtype as the given tensors) that contains the memory used by all the given Tensor (or derived instances), in emitted order
    """
    # Convert to tuple if generator
    if isinstance(tensors, types.GeneratorType):
        tensors = tuple(tensors)
    # Common tensor instantiation and reuse
    common = torch.cat(tuple(tensor.view(-1) for tensor in tensors))
    # Return common tensor
    return relink(tensors, common)

# ---------------------------------------------------------------------------- #

def get_gradient(model):
    """ Get (optionally make each parameter's gradient) a reference to the flat gradient.
    Returns:
      Flat gradient (by reference: future calls to 'set_gradient' will modify it)
    """
    # Flatten (make if necessary)
    gradient = flatten(grads_of(model.parameters()))
    return gradient

def set_gradient(model, gradient):
    """ Overwrite the gradient with the given one.
    Args:
      gradient Given flat gradient
    """
    # Assignment
    grad_old = get_gradient(model)
    grad_old.copy_(gradient)

def get_gradient_values(model):
    """ Get (optionally make each parameter's gradient) a reference to the flat gradient.
    Returns:
      Flat gradient (by reference: future calls to 'set_gradient' will modify it)
    """

    gradient = torch.cat([torch.reshape(param.grad, (-1,)) for param in model.parameters()]).clone().detach()
    return gradient

def set_gradient_values(model, gradient):
    """ Overwrite the gradient with the given one.
    Args:
      gradient Given flat gradient
    """
    cur_pos = 0
    for param in model.parameters():
        param.grad = torch.reshape(torch.narrow(gradient, 0, cur_pos, param.nelement()), param.size()).clone().detach()
        cur_pos = cur_pos + param.nelement()

def get_parameter_values(model):
    """ Get (optionally make each parameter's gradient) a reference to the flat gradient.
    Returns:
      Flat gradient (by reference: future calls to 'set_gradient' will modify it)
    """

    parameter = torch.cat([torch.reshape(param.data, (-1,)) for param in model.parameters()]).clone().detach()
    return parameter

def set_parameter_values(model, parameter):
    """ Overwrite the gradient with the given one.
    Args:
      gradient Given flat gradient
    """
    cur_pos = 0
    for param in model.parameters():
        param.data = torch.reshape(torch.narrow(parameter, 0, cur_pos, param.nelement()), param.size()).clone().detach()
        cur_pos = cur_pos + param.nelement()
# ---------------------------------------------------------------------------- #

def elementwise_distance(data, point):
    '''
    :param data: tensor of gradients
    :param point: tensor of gradient
    :return: distance tensor
    '''
    return torch.norm(data-point,dim=1)


def pairwise_distance_faster(gradients):
    device = gradients[0][0].device
    n = gradients.shape[0]
    distances = torch.zeros((n,n),device=device)
    for gid_x, gid_y in pairwise(tuple(range(n))):
        dist = gradients[gid_x].sub(gradients[gid_y]).norm().item()
        if not math.isfinite(dist):
            dist = math.inf
        distances[gid_x][gid_y] = dist
        distances[gid_y][gid_x] = dist
    return distances