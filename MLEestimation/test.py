import numpy as np
from sklearn import datasets
from datasets import snelson1d
from IPython import embed
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch.autograd.functional import jacobian
import numpy as np
from BNN import BNNSnelson
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn
import faulthandler; faulthandler.enable()


import torch
import torch.nn as nn
import torch.optim as optim

import torchbnn as bnn
import matplotlib.pyplot as plt

# x = torch.linspace(-2, 2, 500)
# y = x.pow(3) - x.pow(2) + 3*torch.rand(x.size())
x, y = snelson1d("./data")
x, y= torch.from_numpy(x), torch.from_numpy(y)
x, y= x.to(torch.float32), y.to(torch.float32)


plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()

model = nn.Sequential(
    bnn.BayesLinear(prior_mu=0, prior_sigma=1, in_features=1, out_features=50),
    nn.ReLU(),
    bnn.BayesLinear(prior_mu=0, prior_sigma=1, in_features=50, out_features=50),
    nn.ReLU(),
    bnn.BayesLinear(prior_mu=0, prior_sigma=1, in_features=50, out_features=50),
    nn.ReLU(),
    bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=50, out_features=1),
)

mse_loss = nn.MSELoss()
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
kl_weight = 0.01

optimizer = optim.Adam(model.parameters(), lr=0.01)

kl_weight = 0.1

for step in range(5000):
    pre = model(x)
    mse = mse_loss(pre, y)
    kl = kl_loss(model)
    cost = mse + kl_weight * kl

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

print('- MSE : %2.2f, KL : %2.2f' % (mse.item(), kl.item()))
#
# x_test = torch.linspace(-2, 2, 500)
# y_test = x_test.pow(3) - x_test.pow(2) + 3*torch.rand(x_test.size())


x_test, y_test = snelson1d("./data")
x_test, y_test= torch.from_numpy(x_test), torch.from_numpy(y_test)
x_test, y_test= x_test.to(torch.float32), y_test.to(torch.float32)



plt.xlabel(r'$x$')
plt.ylabel(r'$y$')

plt.scatter(x_test.data.numpy(), y_test.data.numpy(), color='k', s=2)

y_predict = model(x_test)
plt.scatter(x=x_test.data.numpy(), y=y_predict.data.numpy(), color="red", label='First Prediction')

y_predict = model(x_test)
plt.scatter(x_test.data.numpy(), y_predict.data.numpy(), color="blue", label='Second Prediction')

y_predict = model(x_test)
plt.scatter(x_test.data.numpy(), y_predict.data.numpy(), color="green",  label='Third Prediction')

plt.legend()

plt.show()

