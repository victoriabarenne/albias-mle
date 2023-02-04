import torch.nn as nn
from MLEestimation.datasets import snelson_dataloader
from IPython import embed
import torch
from laplace import Laplace
import matplotlib.pyplot as plt
import numpy as np


def NN(n_hidden_layers, n_neurons=50, n_in=1, n_out=1):
    modules=[]
    modules.append(nn.Linear(n_in, n_neurons))
    modules.append(nn.Tanh())
    for i in range(n_hidden_layers-1):
        modules.append(nn.Linear(n_neurons,n_neurons))
        modules.append(nn.Tanh())
    modules.append(nn.Linear(n_neurons,n_out))

    network = nn.Sequential(*modules)
    return network


def train_epoch_model(model, train_loader, optimizer, loss_fn, device):
    '''
    Trains model for one epoch
    N: len(dataset.all)
    '''
    model.train()
    model.to(device)
    epoch_loss= 0

    for batch_id, (input, target) in enumerate(train_loader):
        input, target = input.to(device), target.to(device)
        input= input.view(input.size(0), -1)
        target= target.squeeze()
        # input, target= input.to(torch.float32), target.to(torch.float32)
        optimizer.zero_grad()
        output = model(input)
        output = output.squeeze()

        batch_loss= loss_fn(output,target).mean(0)
        batch_loss.backward()
        optimizer.step()

        epoch_loss += batch_loss.item()

    return epoch_loss/len(train_loader)


def train_model(model, train_loader, n_epochs, likelihood: str, learning_rate, device):
    if likelihood=="regression":
        loss_fn= nn.MSELoss(reduction="none")
    elif likelihood=="classification":
        loss_fn= nn.NLLLoss(reduction="none")
    training_losses= []

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
    for epoch in range(n_epochs):
        epoch_loss= train_epoch_model(model, train_loader, optimizer, loss_fn, device)
        training_losses.append(epoch_loss)
        print(f'Epoch {epoch+1}: Training loss {epoch_loss:.6f}')

    return training_losses