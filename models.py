import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from IPython import embed
from statisticalbias.radial_layers.variational_bayes import SVI_Linear, SVIConv2D, SVIMaxPool2D
import torchvision

class Model(Module):
    def __init__(self, loss_fn=mean_squared_error, hyperparameters=None):
        self.hyperparameters=hyperparameters
        self.loss_fn=loss_fn

class LinearRegressor(Model):
    def __init__(self, loss_fn=mean_squared_error, hyperparameters=None):
        super().__init__(loss_fn, hyperparameters)

    def train(self, dataset, train_idx=None, weights=None):
        if train_idx is None:
            train_idx= np.arange(dataset.n_points)
        x_train= dataset.x[train_idx]
        y_train= dataset.y[train_idx]
        self.model = LinearRegression().fit(x_train, y_train, weights)

    def predict(self, dataset, test_idx=None):
        if test_idx is None:
            test_idx= np.arange(dataset.n_points)
        return self.model.predict(dataset.x[test_idx])

    def plot_regressor(self, dataset, label, show=False):
        y_pred=self.predict(dataset)
        sns.lineplot(x=dataset.x.squeeze(), y=y_pred.squeeze(),
                     label= label)
        if show==True:
            plt.show()

class RadialBNN(nn.Module):
    def __init__(self, channels):
        super(RadialBNN, self).__init__()
        prior = {"name": "gaussian_prior",
                 "sigma": 0.25,
                 "mu": 0}
        initial_rho = -4
        self.conv1 = SVIConv2D(1, channels, [5,5], "radial", prior, initial_rho, "he")
        self.conv2 = SVIConv2D(channels, channels * 2, [5, 5], "radial", prior, initial_rho, "he")
        self.fc_in_dim = 32 * channels
        self.fc1 = SVI_Linear(self.fc_in_dim, 128, initial_rho, "he", "radial", prior)
        self.fc2 = SVI_Linear(128, 10, initial_rho, "he", "radial", prior)
        self.maxpool = SVIMaxPool2D((2,2))

    def forward(self, x):
        x = F.relu(self.maxpool(self.conv1(x)))
        x = F.relu(self.maxpool(self.conv2(x)))
        variational_samples = x.shape[1]
        x = x.view(-1, variational_samples, self.fc_in_dim)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        output= F.log_softmax(x, dim=2)
        return output.squeeze()


class CNN_dropout(nn.Module):
    def __init__(self, p, channels):
        super().__init__()
        self.p = p
        self.conv1 = nn.Conv2d(1, channels, kernel_size=5)
        self.conv2 = nn.Conv2d(channels, 2*channels, kernel_size=5)
        self.fc_in_dim = 32 * channels
        self.fc1 = nn.Linear(self.fc_in_dim, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(F.dropout2d(self.conv1(x), p=self.p, training=True), 2))
        x = F.relu(F.max_pool2d(F.dropout2d(self.conv2(x), p=self.p, training=True), 2))
        x = x.view(-1, self.fc_in_dim)
        x = F.relu(F.dropout(self.fc1(x), p=self.p, training=True))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



class MLP_dropout(nn.Module):
    def __init__(self, n_layers, p, n_in, n_out, n_neurons=50):
        super().__init__()
        self.p = p
        self.n_layers= n_layers
        self.n_in= n_in
        self.n_out= n_out
        self.in_layer = nn.Linear(n_in, n_neurons)
        self.hidden_layer= nn.Linear(n_neurons, n_neurons)
        self.out_layer = nn.Linear(n_neurons, n_out)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(F.dropout(self.in_layer(x), p=self.p, training=True))
        for _ in range(self.n_layers-1):
            x = F.relu(F.dropout(self.hidden_layer(x), p=self.p, training=True))
        x = self.out_layer(x)
        return F.log_softmax(x, dim=1)

def MLP_dropout_fun(n_layers, p=0.5, n_in=1, n_out=1, n_neurons=50):
    modules=[]
    modules.append(nn.Linear(n_in, n_neurons))
    modules.append(nn.Dropout(p))
    modules.append(nn.ReLU())
    for i in range(n_layers-1):
        modules.append(nn.Linear(n_neurons, n_neurons))
        modules.append(nn.Dropout(p))
        modules.append(nn.ReLU())
    modules.append(nn.Linear(n_neurons, n_out))
    network = nn.Sequential(*modules)
    return network


class ResNet(nn.Module):
    def __init__(self, n=18, n_out=10):
        super().__init__()
        if n==18:
            self.model= torchvision.models.resnet18()
        elif n==50:
            self.model= torchvision.models.resnet50()
        self.model.fc= torch.nn.Linear(512, n_out)

    def forward(self, x):
        x= self.model(x)
        return F.log_softmax(x, dim=1)

def initialize_model(model_arch, likelihood, n_layers=3):

    if model_arch=="radial_bnn":
        model=RadialBNN(16)
    elif model_arch=="mlp":
        model= MLP_dropout(n_layers=n_layers, p=0.2, n_in=784, n_out=10)
        if likelihood=="regression":
            model= MLP_dropout_fun(n_layers=n_layers, p=0.2, n_in=1, n_out=1)
    elif model_arch=="cnn":
        model= CNN_dropout(p=0.2, channels=1)
    elif model_arch=="resnet":
        model= torchvision.models.resnet18()
    return model

#
#
# ################################################################################################
#
#

#
# class MLP_test(nn.Module):
#     def __init__(self, n_layers, n_in, n_out, n_neurons=50):
#         super().__init__()
#         self.n_layers= n_layers
#         self.n_in= n_in
#         self.n_out= n_out
#         self.in_layer = nn.Linear(n_in, n_neurons)
#         self.hidden_layer= nn.Linear(n_neurons, n_neurons)
#         self.out_layer = nn.Linear(n_neurons, n_out)
#
#     def forward(self, x):
#         x = x.view(x.shape[0], -1)
#         x = nn.Tanh()(self.in_layer(x))
#         for _ in range(self.n_layers-1):
#             x = nn.Tanh()(self.hidden_layer(x))
#         x = self.out_layer(x)
#         return F.log_softmax(x, dim=1)
#
#
#
#
#
# class BasicBlock(nn.Module):
#     def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
#         super(BasicBlock, self).__init__()
#
#         self.bn1 = nn.BatchNorm2d(in_planes)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
#
#         self.bn2 = nn.BatchNorm2d(out_planes)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.droprate = dropRate
#         self.equalInOut = (in_planes == out_planes)
#         self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None
#
#     def forward(self, x):
#         if not self.equalInOut:
#             x = self.relu1(self.bn1(x))
#         else:
#             out = self.relu1(self.bn1(x))
#
#         if self.equalInOut:
#             out = self.relu2(self.bn2(self.conv1(out)))
#         else:
#             out = self.relu2(self.bn2(self.conv1(x)))
#
#         if self.droprate > 0:
#             out = F.dropout(out, p=self.droprate, training=self.training)
#
#         out = self.conv2(out)
#
#         if not self.equalInOut:
#             return torch.add(self.convShortcut(x), out)
#         else:
#             return torch.add(x, out)
#
#
# class NetworkBlock(nn.Module):
#     def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
#         super(NetworkBlock, self).__init__()
#         self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
#
#     def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
#         layers = []
#
#         for i in range(nb_layers):
#             layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.layer(x)
#
#
# class WideResNet(nn.Module):
#
#     def __init__(self, depth, widen_factor, num_classes, num_channel=3, dropRate=0.3, feature_extractor=False):
#         super(WideResNet, self).__init__()
#
#         nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
#         assert ((depth - 4) % 6 == 0)
#         n = (depth - 4) // 6
#         block = BasicBlock
#
#         # 1st conv before any network block
#         self.conv1 = nn.Conv2d(num_channel, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
#         self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
#         self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
#         self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
#         # global average pooling and classifier
#         self.bn1 = nn.BatchNorm2d(nChannels[3])
#         self.relu = nn.ReLU(inplace=True)
#         self.fc = nn.Linear(nChannels[3], num_classes)
#
#         self.nChannels = nChannels[3]
#         self.feature_extractor = feature_extractor
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 m.bias.data.zero_()
#
#     def forward(self, x):
#         out = self.features(x)
#
#         if self.feature_extractor:
#             return out
#
#         return self.fc(out)
#
#
#     def features(self, x):
#         out = self.conv1(x)
#         out = self.block1(out)
#         out = self.block2(out)
#         out = self.block3(out)
#         out = self.relu(self.bn1(out))
#         out = F.avg_pool2d(out, 8)
#         out = out.view(-1, self.nChannels)
#         return out
