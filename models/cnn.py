import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 3 * 3, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.num_flat_features(x))  # or: x.view(x.size(0), -1), x.size(0) = batch_size
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x, x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class CifarCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CifarCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        # self.fc2 = nn.Linear(128, 128)
        self.cls = nn.Linear(128, num_classes, bias=False)
        # self.cls.weight.data = F.normalize(self.cls.weight.data, p=2, dim=0)
        # self.dropout = nn.Dropout(0.5)
        # self.beta = nn.Parameter(torch.ones(1).cuda(), requires_grad=True)
        self.proj = torch.nn.Sequential(
            nn.Linear(128, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128, bias=True)
        )
        self.base_weight_keys = [
                                'conv1.weight', 'conv1.bias',
                                'conv2.weight', 'conv2.bias',
                                'conv3.weight', 'conv3.bias',
                                'fc1.weight', 'fc1.bias',
                                # 'fc2.weight', 'fc2.bias',
                                ]
        self.classifier_weight_keys = [
                                # 'fc1.weight', 'fc1.bias',
                                # 'fc2.weight', 'fc2.bias',
                                'cls.weight', 'cls.bias',
                                # 'fc3.weight', 'fc3.bias',
                                ]

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = x.view(-1, self.num_flat_features(x))  # or: x.view(x.size(0), -1), x.size(0) = batch_size
        x = (self.fc1(x))
        y = self.cls(x)#*self.beta
        # x = F.normalize(x)
        f = self.proj(x)
        return f, y
    
    def feature2logit(self, x):
        return self.cls(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class CifarCNN1(nn.Module):
    def __init__(self, num_classes=10):
        super(CifarCNN1, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        # self.fc2 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.cls = nn.Linear(num_classes, num_classes, bias=False)
        V = torch.sqrt(torch.tensor(num_classes)/(torch.tensor(num_classes)-1))*(torch.eye(num_classes)-(1/num_classes)*(torch.ones([num_classes,num_classes])))
        self.cls.weight.data = V
        # self.dropout = nn.Dropout(0.5)
        self.beta = nn.Parameter(torch.ones(1).cuda(), requires_grad=True)
        self.base_weight_keys = [
                                'conv1.weight', 'conv1.bias',
                                'conv2.weight', 'conv2.bias',
                                'conv3.weight', 'conv3.bias',
                                'fc1.weight', 'fc1.bias',
                                'fc2.weight', 'fc2.bias',
                                ]
        self.classifier_weight_keys = [
                                # 'fc1.weight', 'fc1.bias',
                                # 'fc2.weight', 'fc2.bias',
                                'cls.weight', 'cls.bias',
                                # 'fc3.weight', 'fc3.bias',
                                ]

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = x.view(-1, self.num_flat_features(x))  # or: x.view(x.size(0), -1), x.size(0) = batch_size
        # x = F.relu(self.fc1(x))
        # x = self.fc1(x)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=-1)
        y = self.cls(x)*self.beta
        return x, y
    
    def feature2logit(self, x):
        return self.cls(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class CifarNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 5, padding=0)
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, num_classes, bias=False)
        self.cls = self.fc3
        # self.dropout = nn.Dropout(0.5)
        self.base_weight_keys = [
                                'conv1.weight', 'conv1.bias',
                                'conv2.weight', 'conv2.bias',
                                # 'conv3.weight', 'conv3.bias',
                                'fc1.weight', 'fc1.bias',
                                'fc2.weight', 'fc2.bias',
                                ]
        self.classifier_weight_keys = [
                                # 'fc1.weight', 'fc1.bias',
                                # 'fc2.weight', 'fc2.bias',
                                'fc3.weight', 'fc3.bias',
                                ]

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))  # or: x.view(x.size(0), -1), x.size(0) = batch_size
        # x = F.relu(self.fc1(x))
        # x = self.fc1(x)
        x = F.leaky_relu(self.fc1(x))
        # x = self.dropout(x)
        y = F.leaky_relu(self.fc2(x))
        # x = self.dropout(x)
        # x = F.normalize(x, p=2, dim=-1)
        y = self.fc3(y)
        return x, y
    
    def feature2logit(self, x):
        return self.fc3(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Cifar100CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(Cifar100CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes, bias=True)
        # self.fc3 = nn.Linear(84, 10)
        # self.dropout = nn.Dropout(0.5)
        self.base_weight_keys = [
                                'conv1.weight', 'conv1.bias',
                                'conv2.weight', 'conv2.bias',
                                'conv3.weight', 'conv3.bias',
                                'fc1.weight', 'fc1.bias',
                                # 'fc2.weight', 'fc2.bias',
                                ]
        self.classifier_weight_keys = [
                                # 'fc1.weight', 'fc1.bias',
                                'fc2.weight', 'fc2.bias',
                                # 'fc3.weight', 'fc3.bias',
                                ]

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = x.view(-1, self.num_flat_features(x))  # or: x.view(x.size(0), -1), x.size(0) = batch_size
        # x = F.relu(self.fc1(x))
        # x = self.fc1(x)
        x = F.leaky_relu(self.fc1(x))
        # x = self.dropout(x)
        # x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        y = self.fc2(x)
        # y = F.relu(self.fc2(x))
        return x, y
    
    def feature2logit(self, x):
        return self.fc2(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class CNN_FMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_FMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=1)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.cls = nn.Linear(128, num_classes, bias=True)
        self.reshape = nn.Linear(128, 128)
        self.base_weight_keys = ['conv1.weight', 'conv1.bias',
                            'conv2.weight', 'conv2.bias',
                            'fc1.weight', 'fc1.bias',]
        self.classifier_weight_keys = [
                                        # 'fc2.weight', 'fc2.bias',
                                        'cls.weight', 'cls.bias',
                                       ]

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))  # or: x.view(x.size(0), -1), x.size(0) = batch_size
        # x = F.relu(self.fc1(x))
        x = F.leaky_relu(self.fc1(x))
        # feature_norm = torch.norm(x,dim=1)
        # weight_norm = torch.norm(self.fc2.parameters(),dim=0)
        y = self.cls(x)
        f = self.reshape(x)
        f = F.normalize(f)
        return f, y

    def feature2logit(self, x):
        return self.cls(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class CNN_FMNIST1(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_FMNIST1, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=1)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes, bias=True)
        self.base_weight_keys = ['conv1.weight', 'conv1.bias',
                            'conv2.weight', 'conv2.bias',
                            'fc1.weight', 'fc1.bias',]
        self.classifier_weight_keys = [
                                        'fc2.weight', 'fc2.bias',
                                        # 'cls.weight', 'cls.bias',
                                       ]

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))  # or: x.view(x.size(0), -1), x.size(0) = batch_size
        # x = F.relu(self.fc1(x))
        x = F.leaky_relu(self.fc1(x))
        # feature_norm = torch.norm(x,dim=1)
        # weight_norm = torch.norm(self.fc2.parameters(),dim=0)
        y = self.fc2(x)
        return x, y

    def feature2logit(self, x):
        return self.fc2(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def conv3x3(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        # nn.BatchNorm2d(out_channels, track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class CNNCifar_FedBABU(nn.Module):
    def __init__(self):
        super(CNNCifar_FedBABU, self).__init__()
        in_channels = 3
        num_classes = 10
        
        hidden_size = 64
        
        self.features = nn.Sequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size)
        )
        
        self.linear = nn.Linear(hidden_size*2*2, num_classes)
        self.classifier_weight_keys = ['linear.weight', 'linear.bias',]

    def forward(self, x):
        features = self.features(x)
        features = features.view((features.size(0), -1))
        logits = self.linear(features)
        
        return features, logits
    
    def extract_features(self, x):
        features = self.features(x)
        features = features.view((features.size(0), -1))
        
        return features