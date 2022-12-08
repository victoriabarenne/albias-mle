from snelson import snelson1d
from IPython import embed
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch.autograd.functional import jacobian
import numpy as np
from BNN import BNNSnelson, NNSnelson, train_nn, train_bnn
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn
# import faulthandler; faulthandler.enable()


plot_snelson=False


## Plotting the Snelson dataset
x, y = snelson1d("./data")
x, y= torch.from_numpy(x), torch.from_numpy(y)
x, y= x.to(torch.float32), y.to(torch.float32)
if plot_snelson:
    sns.scatterplot(x=x.squeeze(), y=y.squeeze())
    plt.title("Snelson training dataset")
    plt.show()



## Discrete model structure selection
step_size= 0.01
F= 1
B= 0
n_epochs= 1000
K= 1
n_neurons=50
sigma_likelihood=0.5
prior_sigma=0.1
prior_mu=0.0

# simple_model= BNNSnelson(prior_mu=prior_mu, prior_sigma=prior_sigma, n_hidden=1, n_neurons=50)
# complex_model= BNNSnelson(prior_mu=prior_mu, prior_sigma=prior_sigma, n_hidden=3, n_neurons=50)

simple_model= NNSnelson(n_hidden=1, n_neurons=50)
complex_model= NNSnelson(n_hidden=3, n_neurons=50)

model=simple_model

optimizer = optim.Adam(model.parameters(), lr=0.01)
mse_loss = nn.MSELoss()
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
kl_weight = 0.01
train_nn(model,3000, x, y)
embed()

for epoch in range(n_epochs):
    train(model, 1, x, y)
    if (epoch>B)&(epoch%F==0):
        # compute the hessian
        output= model(x)

        # compute the MAP for the weights
        # get the log q(D|M)
        for k in range(K):
            # gradient descent step for continuous_hyperparameters of M
            pass







train_nn(simple_model, 10000, x, y)
predictions_simple= simple_model(x)
train(complex_model, 10000, x, y)
predictions_complex= complex_model(x)

sns.scatterplot(x=x.detach().squeeze(), y=predictions_simple.detach().squeeze(), color="green")
sns.scatterplot(x=x.detach().squeeze(), y=predictions_complex.detach().squeeze(), color="red")
sns.scatterplot(x=x.detach().squeeze(), y=y.squeeze(), color="blue")
plt.show()
embed()

