import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from IPython import embed
import seaborn as sns
import torchvision
import pandas as pd
from torch.utils.data import Subset
from datasets import ActiveMNIST
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torchvision.transforms import ToTensor

from radial_layers.variational_bayes import SVI_Linear, SVIConv2D, SVIMaxPool2D
from radial_layers.loss import Elbo

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

class BNN_variational(Module):
    """
    Classification convolutional network in "Statistical bias in active learning paper"
    takes as input batch_size x n_training_var x n_channels x H x W
    and outputs batch_size x n_training_var x n_classes
    """
    def __init__(self):
        super(BNN_variational, self).__init__()
        initial_rho = -4
        initial_mu_std = ("he")
        variational_distribution = "radial"
        prior = {
            "name": "gaussian_prior",
            "sigma": 0.25,
            "mu": 0,
        }

        self.conv1= SVIConv2D(
            in_channels=1,
            out_channels=16,
            kernel_size=[5,5],
            variational_distribution=variational_distribution,
            prior= prior,
            initial_rho= initial_rho,
            mu_std= initial_mu_std
        )

        self.conv2 = SVIConv2D(
            in_channels=16,
            out_channels=32,
            kernel_size=[5,5],
            variational_distribution=variational_distribution,
            prior= prior,
            initial_rho=  initial_rho,
            mu_std= initial_mu_std
        )

        self.max_pool= SVIMaxPool2D(
            kernel_size= (2,2)
        )


        self.fc1= SVI_Linear(
            in_features= 512,
            out_features=128,
            initial_rho= initial_rho,
            initial_mu= initial_mu_std,
            variational_distribution= variational_distribution,
            prior= prior
        )

        self.fc2= SVI_Linear(
            in_features= 128,
            out_features=10,
            initial_rho= initial_rho,
            initial_mu= initial_mu_std,
            variational_distribution= variational_distribution,
            prior= prior
        )

    def forward(self, x):
        # Input has shape [examples, samples, n_channels, height, width]
        #Conv part
        out = F.relu(self.conv1(x))
        out = self.max_pool(out)
        out = F.relu(self.conv2(out))
        out = self.max_pool(out)

        # MLP part
        s = out.shape
        out = torch.reshape(out, (s[0], s[1], -1))
        out = F.relu(self.fc1(out))
        out = F.log_softmax(self.fc2(out), dim=-1)
        return out

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
        return F.log_softmax(x, dim=2)

