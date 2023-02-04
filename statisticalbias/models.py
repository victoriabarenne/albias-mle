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
        out = self.max_pool(out)
        out = F.relu(self.conv2(out))
        out = self.max_pool(out)

        # MLP part
        s = out.shape
        out = torch.reshape(out, (s[0], s[1], -1))
        out = F.relu(self.fc1(out))
        out = F.log_softmax(self.fc2(out), dim=-1)
        return out

#### Model Training, Evaluation and Testing

def train(model, train_loader, device, variational_samples, loss, optimizer, bias_correction, N):
    '''
    Trains model for one epoch
    N: len(dataset.all)
    '''
    model.train()
    model.to(device)
    epoch_loss= 0
    m_id=0

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

        M = len(train_loader.dataset)
        m= torch.arange(m_id, m_id+len(acq_prob))+1
        m_id= m_id+len(acq_prob)
        if bias_correction== "none":
            batch_loss = loss_helper(output, target).mean(0)
        elif bias_correction=="pure":
            weight= 1/(N*acq_prob)+(M-m)/N
            batch_loss= (weight*loss_helper(output, target)).mean(0)
        elif bias_correction== "lure":
            weight= 1 + (N-M)/(N-torch.arange(1, M+1))*(1/((N-m+1)*acq_prob)-1)
            batch_loss= (weight*loss_helper(output, target)).mean(0)

        batch_loss.backward()
        optimizer.step()

        # print statistics
        epoch_loss += batch_loss.item()
    return epoch_loss


def train_all(model, train_loader, val_loader, device, learning_rate, variational_samples, batch_size, n_epochs, early_stopping, bias_correction, N):
    loss = Elbo(binary=False, regression=False)
    loss.set_model(model, batch_size)
    loss.set_num_batches(len(train_loader))

    train_losses = []
    val_losses=[]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)

    for epoch in range(n_epochs):
        #TODO M,N, what are they?
        # N = dataset.n_points
        # TODO: Or is N= len(dataset.available??)
        epoch_loss= train(model, train_loader, device, variational_samples, loss, optimizer, bias_correction, N)
        train_losses.append(epoch_loss)
        val_loss = evaluate(model, val_loader, device, variational_samples, loss, "none", len(val_loader.dataset))
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

    return train_losses, val_losses, best_loss

def evaluate(model, eval_loader, device, variational_samples, loss, bias_correction, N):
    model.eval()
    model.to(device)
    eval_loss= 0
    M = len(eval_loader.dataset)
    m_id=0
    for batch_id, (data, target, acq_prob) in enumerate(eval_loader):
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

        m= torch.arange(m_id, m_id+len(data))+1
        m_id= m_id+len(data)
        if bias_correction== "none":
            batch_loss = loss_helper(output, target).mean(0)
        elif bias_correction=="pure":
            weight= 1/(N*acq_prob)+(M-m)/N
            batch_loss= (weight*loss_helper(output, target)).mean(0)
        elif bias_correction== "lure":
            weight= 1 + (N-M)/(N-m)*(1/((N-m+1)*acq_prob)-1)
            batch_loss= (weight*loss_helper(output, target)).mean(0)
        eval_loss += batch_loss.item()
    return eval_loss/len(eval_loader)


def testing(model, testing_loader, device, variational_samples, loss, N):
    model.eval()
    model.to(device)
    test_loss_none, test_loss_pure, test_loss_lure= 0,0,0
    M = len(testing_loader.dataset)
    m_id=0
    for batch_id, (data, target, acq_prob) in enumerate(testing_loader):
        data, target = data.to(device), target.to(device)
        data = data.unsqueeze(1)
        data = data.expand((-1, variational_samples, -1, -1, -1))
        output = model(data)
        output = output.squeeze()

        def loss_helper(prediction, target):
            nll_loss, kl_loss = loss.compute_loss(prediction, target)
            return nll_loss

        m= torch.arange(m_id, m_id+len(data))+1
        m_id= m_id+len(data)
        batch_loss_none = loss_helper(output, target).mean(0)
        weight_pure= 1/(N*acq_prob)+(M-m)/N
        batch_loss_pure= (weight_pure*loss_helper(output, target)).mean(0)
        weight_lure= 1 + (N-M)/(N-m)*(1/((N-m+1)*acq_prob)-1)
        batch_loss_lure= (weight_lure*loss_helper(output, target)).mean(0)
        print(batch_loss_none, batch_loss_pure, batch_loss_pure)
        test_loss_none += batch_loss_none.item()
        test_loss_pure += batch_loss_pure.item()
        test_loss_lure += batch_loss_lure.item()
    return test_loss_none/len(testing_loader), test_loss_pure/len(testing_loader), test_loss_lure/len(testing_loader)


def compute_score(model, dataset, batch_size, device, acquisition_var_samples):
    """
    Returns score for each available point in dataset,
    mutual information between the output and the distribution theta (of the model)
    """
    available_loader= dataset.get_availableloader(batch_size)
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
loss_fn= nn.NLLLoss()
n_epochs=100
early_stopping_epochs= 20
device= "cpu" # or "cpu"
log_interval=100
learning_rate=5*10e-4
variational_samples=8
variational_samples_test=8
acquisition_var_samples=100
T= 10000
points_per_acquisition= 1
initial_pool=10
goal_points=70
training_frequency=1

#0) Initialize model and training/testing datasets
dataset= ActiveMNIST(noise_ratio= 0.1, p_train=0.25, train=True, unbalanced= True, random_state=2) # unbalanced MNIST
dataset_testing= ActiveMNIST(noise_ratio=0, p_train=0.1, train=False, unbalanced= False, random_state=1)
val_loader = torch.utils.data.DataLoader(dataset.validation, batch_size=batch_size, shuffle=True)
N= dataset.n_points
N_test= dataset_testing.n_points

# 1) Initialize using 10 random samples from the training data pool
initial_idx= np.random.randint(low=0, high= len(dataset.available), size=initial_pool)
prob= np.repeat(1/len(dataset.available), repeats= initial_pool)
dataset.observe(initial_idx, prob)
n_acquired=0

while ((len(dataset.queries)<goal_points)):
    # 2) Train model using Dtrain and R_tilde loss
    #TODO: do we reinitialize the model in between each aquisition of points??
    model = BNN_variational()
    train_loader= dataset.get_trainloader(64)
    train_all(model, train_loader, val_loader, device, learning_rate, variational_samples, batch_size,
              n_epochs, early_stopping_epochs, "none", N)

    # 3) Select the next acquisition point
    print("Computing the scores for all available points")
    scores= compute_score(model, dataset, batch_size, device, acquisition_var_samples)
    q_proposal= torch.nn.Softmax(dim=0)(T*torch.from_numpy(scores))
    id= np.random.choice(np.arange(len(scores)), size= points_per_acquisition, replace=True, p = q_proposal.detach().numpy())
    prob= q_proposal[id].item()
    dataset.observe(id, prob, idx_absolute=False)

    n_acquired+=1

    # 4) Train models using only the training data using no bias-correction, pure, and lure correction every 3 aquisition rounds
    if n_acquired==training_frequency:
        model_none, model_pure, model_lure= BNN_variational(), BNN_variational(), BNN_variational()
        train_loader = dataset.get_trainloader(64)
        train_loss_none, eval_loss_none, _= train_all(model_none, train_loader, val_loader, device, learning_rate, variational_samples, batch_size, n_epochs, early_stopping_epochs, "none", N)
        train_loss_pure, eval_loss_pure, _= train_all(model_pure, train_loader, val_loader, device, learning_rate, variational_samples, batch_size, n_epochs, early_stopping_epochs, "pure", N)
        train_loss_lure, eval_loss_lure, _= train_all(model_lure, train_loader, val_loader, device, learning_rate, variational_samples, batch_size, n_epochs, early_stopping_epochs, "lure", N)
        n_acquired = 0

        embed()

        # 5) Evaluate the models using the test set (to get the bias)
        print("Computing the scores for all available points")
        scores = compute_score(model, dataset_testing, batch_size, device, acquisition_var_samples)
        q_proposal = torch.nn.Softmax(dim=0)(T * torch.from_numpy(scores))
        # Note: Important that M<N, otherwise we get a zero in the denominator of the lure weights
        idx = np.random.choice(np.arange(len(scores)), size=len(scores)-1, replace=False,
                               p=q_proposal.detach().numpy())  # This will pick out every single index, but in approx decreasing probability
        prob = q_proposal[idx].numpy()
        dataset_testing.observe(idx, prob)

        testing_loader = dataset_testing.get_trainloader(batch_size)
        loss = Elbo(binary=False, regression=False)
        loss.set_model(model_none, batch_size)
        loss.set_num_batches(len(testing_loader))
        testing_loss=[]
        for model in [model_none, model_pure, model_lure]:
            testing_loss.append(testing(model, testing_loader, device, variational_samples_test, loss, N_test))

