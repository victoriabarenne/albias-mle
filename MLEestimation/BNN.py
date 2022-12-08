import torch
import torch.nn as nn
import torchbnn as bnn
import torch.optim as optim
from IPython import embed
from snelson import snelson1d

class BNNSnelson(nn.Module):
    def __init__(self, prior_mu, prior_sigma, n_hidden, n_neurons):
        super().__init__()
        self.n_hidden=n_hidden
        self.n_neurons=n_neurons
        self.in_layer=bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=1, out_features=self.n_neurons)
        self.hidden_layer=bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=self.n_neurons, out_features=self.n_neurons)
        self.out_layer=bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=self.n_neurons, out_features=1)
        self.activation=nn.ReLU()

    def forward(self, x):
        out = self.activation(self.in_layer(x))
        for i in range(self.n_hidden-1):
            out= self.activation(self.hidden_layer(out))
        out = self.out_layer(out)
        return out

class NNSnelson(nn.Module):
    def __init__(self, n_hidden, n_neurons):
        super().__init__()
        self.n_hidden= n_hidden
        self.n_neurons= n_neurons
        self.activation= nn.ReLU()

        # First layer
        self.model= nn.Sequential(nn.Linear(in_features=1, out_features=self.n_neurons),
                                  self.activation)
        # Hidden layers
        for i in range(self.n_hidden-1):
            self.model.append(nn.Linear(in_features=self.n_neurons, out_features=self.n_neurons))
            self.model.append(self.activation)

        # Output layers
        self.model.append(nn.Linear(in_features=self.n_neurons, out_features=1))


    def forward(self, x):
        out = self.model(x)
        return out

def train_bnn(model, n_epochs, x, y):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    mse_loss = nn.MSELoss()
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    kl_weight = 0.01

    for epoch in range(n_epochs):
        pre = model(x)
        mse = mse_loss(pre, y)
        kl = kl_loss(model)
        cost = mse + kl_weight * kl

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

    predictions= model(x)
    print('- MSE : %2.2f, KL : %2.2f' % (nn.MSELoss()(y, predictions), kl.item()))

def train_nn(model, n_epochs, x, y):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    mse_loss = nn.MSELoss()


    for epoch in range(n_epochs):
        pre = model(x)
        mse = mse_loss(pre, y)

        optimizer.zero_grad()

        mse.backward()
        optimizer.step()

    predictions= model(x)
    print('- MSE : %2.2f' % (nn.MSELoss()(y, predictions)))

x, y = snelson1d("./data")
x, y= torch.from_numpy(x), torch.from_numpy(y)
x, y= x.to(torch.float32), y.to(torch.float32)

simple_network= NNSnelson(1, 50)
complex_network=NNSnelson(3,50)

embed()