import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, D_in=784, H=128, num_classes=10):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, num_classes)
 
        self.classifier_weight_keys = ['linear2.weight', 'linear2.bias',]

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = x.view(-1, 28 * 28)
        x = self.linear1(x)
        x = F.leaky_relu(x)
        y = self.linear2(x)
        return x, y

    def feature2logit(self, x):
        return self.linear2(x)


class MLPSyn(nn.Module):
    def __init__(self, D_in=60, H=30, num_classes=10):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(MLPSyn, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, num_classes)
 
        self.classifier_weight_keys = ['linear2.weight', 'linear2.bias',]

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = x.view(-1, 60)
        x = self.linear1(x)
        x = F.leaky_relu(x)
        y = self.linear2(x)
        return x, y

    def feature2logit(self, x):
        return self.linear2(x)