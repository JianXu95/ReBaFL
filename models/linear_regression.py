import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self, D_in=1000):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(in_features=D_in, out_features=1, bias=False)
    def forward(self, x):
        out = self.linear(x)
        return 0, out

class LR(nn.Module):
    def __init__(self, D_in=60, num_classes=10):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(LR, self).__init__()
        self.cls = nn.Linear(D_in, num_classes, bias=False)
 
        self.classifier_weight_keys = ['cls.weight', #'cls.bias',
                                                              ]

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = x.view(-1, 60)
        y = self.cls(x)
        return x, y

    def feature2logit(self, x):
        return self.cls(x)
