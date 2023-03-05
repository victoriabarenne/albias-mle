import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from IPython import embed

from statisticalbias.radial_layers.variational_bayes import SVI_Linear, SVIConv2D, SVIMaxPool2D


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
