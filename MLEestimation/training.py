import torch.nn as nn
import torch
from laplace import Laplace
import numpy as np
from torch.nn.utils import parameters_to_vector
from laplace.curvature.backpack import BackPackGGN
from copy import deepcopy

def train_epoch_model(model, train_loader, optimizer, loss_fn, device, hyperparameters, likelihood):
    '''
    Trains model for one epoch
    N: len(dataset.all)
    '''
    model.train()
    model.to(device)
    epoch_loss= 0
    epoch_perf= 0
    N= len(train_loader.dataset)

    for batch_id, (input, target) in enumerate(train_loader):
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input)
        # output = output.squeeze()

        prior_precision = hyperparameters[0].exp()
        if likelihood=="regression":
            sigma_noise = hyperparameters[1].exp()
            crit= sigma_noise**2/N
        elif likelihood== "classification":
            temperature= hyperparameters[1]
            crit= 1/N
            output= output/temperature

        theta= parameters_to_vector(model.parameters())
        batch_loss= loss_fn(output,target) + prior_precision*(theta@theta)*crit

        if likelihood=="regression":
            epoch_perf+= torch.sum((output-target)**2).item()
        elif likelihood=="classification":
            epoch_perf+= torch.sum(torch.argmax(output.detach(), dim=-1)==target).item()

        batch_loss.backward(retain_graph= True)
        optimizer.step()

        epoch_loss += batch_loss.item()

    return epoch_loss/len(train_loader), epoch_perf/N


def train_model(model, train_loader, n_epochs, likelihood: str, learning_rate, device, hyperparameters):
    if likelihood=="regression":
        loss_fn= nn.MSELoss(reduction="mean")
    elif likelihood=="classification":
        loss_fn= nn.CrossEntropyLoss(reduction="mean")
    training_losses= []
    perf_metrics= []
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
    for epoch in range(n_epochs):
        epoch_loss, epoch_perf= train_epoch_model(model, train_loader, optimizer, loss_fn, device, hyperparameters, likelihood)
        training_losses.append(epoch_loss)
        perf_metrics.append(epoch_perf)
        print(f'Epoch {epoch+1}: Training loss {epoch_loss:.6f} and Epoch performance {epoch_perf:.6f}')

    return training_losses, perf_metrics


def mle_training(model, train_loader, likelihood, n_epochs, lr_param, lr_hyper, F=1, B=0, K=1, hessian_structure="kron", device= "cpu", prior_init=1., sigma_init=1., temperature_init=1.):
    # 0) Initialize model, its training optimizer and loss
    if likelihood=="regression":
        loss_fn= nn.MSELoss(reduction="mean")
    elif likelihood=="classification":
        loss_fn= nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_param, amsgrad=True)

    # 1) Initialize hyperparameters that appear in the marginal likelihood:
    # observation noise of likelihood (log_sigma), variances of gaussian prior (log_prior)
    prior= torch.tensor([prior_init], dtype= torch.float32)
    temperature= torch.tensor([temperature_init], dtype= torch.float32, requires_grad=True)
    sigma= torch.tensor([sigma_init], dtype= torch.float32)
    log_prior, log_sigma= torch.log(prior), torch.log(sigma)
    log_prior.requires_grad, log_sigma.requires_grad= True, True

    hyperparameters = [log_prior, log_sigma] if likelihood=="regression" else [log_prior, temperature]
    hyper_optimizer = torch.optim.Adam(hyperparameters, lr=lr_hyper)

    # Tracking of training loss, marginal likelihood and hyperparameters
    training_losses=[]
    perf_metrics=[]
    neg_margliks=[]
    priors=[]
    if likelihood=="regression":
        sigmas= []
    elif likelihood=="classification":
        temperatures= []

    # 2) Run the optimization of both model weights and hyperparameters
    best_marglik = np.inf
    # scheduler= torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000)
    for epoch in range(1,n_epochs+1):
        epoch_training_loss, epoch_metric= train_epoch_model(model, train_loader, optimizer, loss_fn, device, hyperparameters, likelihood)
        training_losses.append(epoch_training_loss)
        perf_metrics.append(epoch_metric)
        # scheduler.step()
        if (epoch>B)&(epoch%F==0):
            la = Laplace(model, likelihood,
                         hessian_structure= hessian_structure,
                         sigma_noise=log_sigma.exp(),
                         prior_precision=log_prior.exp(),
                         temperature= temperature,
                         subset_of_weights='all',
                         backend= BackPackGGN)
            la.fit(train_loader)
            for k in range(K):
                hyper_optimizer.zero_grad()
                noise= log_sigma.exp() if likelihood=="regression" else None
                neg_marglik = -la.log_marginal_likelihood(log_prior.exp(), noise)
                neg_marglik.backward(retain_graph=True)
                hyper_optimizer.step()
            noise = log_sigma.exp() if likelihood == "regression" else None
            neg_marglik= -la.log_marginal_likelihood(log_prior.exp(), noise)
            neg_margliks.append(neg_marglik.item())
            priors.append(log_prior.exp().item())
            sigmas.append(log_sigma.exp().item()) if likelihood=="regression" else temperatures.append(temperature.item())

            if neg_margliks[-1]< best_marglik:
                best_marglik= deepcopy(neg_margliks[-1])
                best_prior= deepcopy(la.prior_precision.detach().numpy())
                if likelihood=="regression":
                    best_sigma= deepcopy(la.sigma_noise.detach().numpy())
                elif likelihood=="classification":
                    best_temperature= deepcopy(la.temperature.detach().numpy())
            print(f"Epoch {epoch}: Training Loss {epoch_training_loss} Log MargLikelihood: {neg_marglik}")
        else:
            print(f"Epoch {epoch}: Training Loss {training_losses[-1]:.6f} Log MargLikelihood: not updated")

    if likelihood=="regression":
        print(f"Best Log marginal likelihood:{best_marglik} with observation noise {best_sigma} and prior variance {best_prior}")
    elif likelihood=="classification":
        print(f"Best Log marginal likelihood:{best_marglik} with temperature {best_temperature} and prior variance {best_prior}")

    if likelihood=="regression":
        all_hyperparameters= np.concatenate((np.array(priors).reshape(-1,1), np.array(sigmas).reshape(-1,1)), -1)
    elif likelihood=="classification":
        all_hyperparameters= np.concatenate((np.array(priors).reshape(-1,1), np.array(temperatures).reshape(-1,1)), -1)

    return la, model, neg_margliks, training_losses, perf_metrics, all_hyperparameters

