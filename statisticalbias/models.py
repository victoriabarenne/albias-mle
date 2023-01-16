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
from datasets import ActiveUnbalancedMNIST
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
        initial_rho = -4  # This is a reasonable value, but not very sensitive.
        initial_mu_std = (
            "he"  # Uses Kaiming He init. Or pass float for a Gaussian variance init.
        )
        variational_distribution = "radial"  # You can use 'gaussian' to do normal MFVI.
        prior = {
            "name": "gaussian_prior",
            "sigma": 0.25,
            "mu": 0,
        }  # Just a unit Gaussian prior.

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
        # print("Conv1", out.shape)
        out = self.max_pool(out)
        # print("Maxpool1", out.shape)
        out = F.relu(self.conv2(out))
        # print("Conv2", out.shape)
        out = self.max_pool(out)
        # print("Maxpool2", out.shape)

        # MLP part
        s = out.shape
        out = torch.reshape(out, (s[0], s[1], -1))
        # print("linear input", out.shape)
        out = F.relu(self.fc1(out))
        # print("linear1", out.shape)
        out = F.log_softmax(self.fc2(out), dim=-1)
        # print("output", out.shape)
        return out



def train(model, dataset, train_loader, device, variational_samples, loss, optimizer, bias_correction):
    '''
    Trains model for one epoch
    '''
    model.train()
    model.to(device)
    epoch_loss= 0
    for batch_id, (data, target, acq_prob) in enumerate(train_loader):
        # zero the parameter gradients
        data, target = data.to(device), target.to(device)
        # change input to variational
        data = data.unsqueeze(1)
        data = data.expand((-1, variational_samples, -1, -1, -1))
        data= data.float()
        optimizer.zero_grad()
        output = model(data)
        output = output.squeeze()

        def loss_helper(prediction, target):
            nll_loss, kl_loss = loss.compute_loss(prediction, target)
            #TODO: regulazing term can be changed? (to other than 1/10)
            return nll_loss + kl_loss / 10

        M= len(dataset.train)
        N= dataset.n_points
        # TODO: Or is N= len(dataset.available??)
        if bias_correction== "none":
            batch_loss = loss_helper(output, target).mean(0)
        elif bias_correction=="pure":
            weight= 1/(N*acq_prob)+(M-torch.arange(1, M+1))/N
            batch_loss= (weight*loss_helper(output, target)).mean(0)
        elif bias_correction== "lure":
            weight= 1 + (N-M)/(N-torch.arange(1, M+1))*(1/((N-torch.arange(1,M+1)+1)*acq_prob)-1)
            batch_loss= (weight*loss_helper(output, target)).mean(0)

        batch_loss.backward()
        optimizer.step()

        # print statistics
        epoch_loss += batch_loss.item()
    return epoch_loss


def train_all(model, dataset, train_loader, val_loader, device, learning_rate, variational_samples, batch_size, n_epochs, early_stopping, bias_correction):
    loss = Elbo(binary=False, regression=False)
    loss.set_model(model, batch_size)
    loss.set_num_batches(len(train_loader))

    train_losses = []
    val_losses=[]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)

    for epoch in range(n_epochs):
        epoch_loss= train(model, dataset, train_loader, device, variational_samples, loss, optimizer, bias_correction)
        train_losses.append(epoch_loss)
        val_loss = validate(model, val_loader, device, variational_samples, loss, bias_correction)
        val_losses.append(val_loss)
        if epoch==0:
            best_loss= val_loss
            patience=0
        elif epoch>0:
            if val_loss>best_loss:
                patience+=1
            elif val_loss<=best_loss:
                best_loss=val_loss

        if patience>=early_stopping:
            print(f"Early stopping at {epoch} epochs")
            break
        print(f'Epoch {epoch+1}: Training loss {epoch_loss:.6f}, Validation loss {val_loss:.6f}')

    return train_losses, val_losses

def validate(model, val_loader, device, variational_samples, loss, bias_correction):
    model.eval()
    model.to(device)
    val_loss= 0
    for batch_id, (data, target, acq_prob) in enumerate(val_loader):
        # zero the parameter gradients
        data, target = data.to(device), target.to(device)
        # change input to variational
        data = data.unsqueeze(1)
        data = data.expand((-1, variational_samples, -1, -1, -1))

        output = model(data)
        output = output.squeeze()

        def loss_helper(prediction, target):
            nll_loss, kl_loss = loss.compute_loss(prediction, target)
            #TODO: regulazing term can be changed? (to other than 1/10)
            return nll_loss + kl_loss / 10

        M = len(dataset.train)
        N = dataset.n_points
        if bias_correction== "none":
            batch_loss = loss_helper(output, target).mean(0)
        elif bias_correction=="pure":
            weight= 1/(N*acq_prob)+(M-torch.arange(1, M+1))/N
            batch_loss= (weight*loss_helper(output, target)).mean(0)
        elif bias_correction== "lure":
            weight= 1 + (N-M)/(N-torch.arange(1, M+1))*(1/((N-torch.arange(1,M+1)+1)*acq_prob)-1)
            batch_loss= (weight*loss_helper(output, target)).mean(0)

        val_loss += batch_loss.item()
        return val_loss


def compute_score(model, dataset, batch_size, device, acquisition_var_samples):
    """
    Returns score for each available point in dataset,
    mutual information between the output and the disrtribution theta (of the model)
    """
    _, available_loader= dataset.get_dataloaders(batch_size)
    model.eval()
    model.to(device)
    scores= np.array([])
    for idx, (data, target, weight) in enumerate(available_loader):
        # Input preprocessing
        data= data.unsqueeze(1)
        data = data.expand((-1, acquisition_var_samples, -1, -1, -1))

        # Output
        output= model(data)
        output= torch.permute(output, (1, 0, 2))

        # Calculating the average entropy
        average_entropy_i= -((output*output.exp()).sum(2)).mean(0)
        # Calculating the entropy average
        mean_samples_i_c= (output.exp().sum(0)).log()- torch.log(torch.tensor([acquisition_var_samples]))
        entropy_average_i= -(mean_samples_i_c*mean_samples_i_c.exp()).sum(1)

        score= entropy_average_i- average_entropy_i
        scores= np.concatenate((scores, score.detach().numpy()))
    return scores

batch_size=64
dataset= ActiveUnbalancedMNIST(noise_ratio= 0.1, p_train=0.25, random_state=2) # unbalanced MNIST
loss_fn= nn.NLLLoss()
n_epochs=100
early_stopping_epochs= 20
device= "cpu" # or "cpu"
log_interval=100
learning_rate=5*10e-4
variational_samples=8
bias_correction="none"
acquisition_var_samples=100
T= 10000
points_per_acquisition= 1
initial_pool=10
goal_points=70
training_frequency=1

# 1) Initialize using 10 random samples from the training data pool
initial_idx= np.random.randint(low=0, high= len(dataset.available), size=initial_pool)
prob= np.repeat(1/len(dataset.available), repeats= initial_pool)
dataset.observe(initial_idx, prob)
n_acquired=0

while ((len(dataset.queries)<goal_points)):
    # 2) Train model using Dtrain and R_tilde loss (unmodified loss)
    #TODO: do we reinitialize the model in between each aquisition of points??
    model = BNN_variational()
    val_loader = torch.utils.data.DataLoader(dataset.validation, batch_size=batch_size, shuffle=True)
    train_loader, _ = dataset.get_dataloaders(64)
    train_all(model, dataset, train_loader, val_loader, device, learning_rate, variational_samples, batch_size,
              n_epochs, early_stopping_epochs, bias_correction)

    # 3) Select the next acquisition point
    print("Computing the scores for all available points")
    scores= compute_score(model, dataset, batch_size, device, acquisition_var_samples)
    q_proposal= torch.nn.Softmax(dim=0)(T*torch.from_numpy(scores))
    print(torch.any(torch.isnan(q_proposal)))
    print(q_proposal.max(), q_proposal.min())
    id= np.random.choice(np.arange(len(scores)), size= points_per_acquisition, replace=True, p = q_proposal.detach().numpy())
    prob= q_proposal[id].item()
    dataset.observe(id, prob, idx_absolute=False)

    n_acquired+=1

    # 4) Train models using only the training data using no bias-correction, pure, and lure correction every 3 aquisition rounds
    if n_acquired==training_frequency:
        model_none, model_pure, model_lure= BNN_variational(), BNN_variational(), BNN_variational()
        train_loader, _ = dataset.get_dataloaders(64)
        train_all(model_none, dataset, train_loader, val_loader, device, learning_rate, variational_samples, batch_size, n_epochs, early_stopping_epochs, bias_correction="none")
        train_all(model_pure, dataset, train_loader, val_loader, device, learning_rate, variational_samples, batch_size, n_epochs, early_stopping_epochs, bias_correction="pure")
        train_all(model_lure, dataset, train_loader, val_loader, device, learning_rate, variational_samples, batch_size, n_epochs, early_stopping_epochs, bias_correction="lure")
        n_acquired = 0


