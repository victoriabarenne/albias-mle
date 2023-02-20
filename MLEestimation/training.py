import torch.nn as nn
import torch
from laplace import Laplace
import numpy as np
from torch.nn.utils import parameters_to_vector
from laplace.curvature.backpack import BackPackGGN
from copy import deepcopy
import wandb
def train_epoch_model(model, train_loader, optimizer, loss_fn, device, hyperparameters, likelihood):
    '''
    Trains model for one epoch
    N: len(dataset.all)
    '''
    model.train()
    model.to(device)
    epoch_loss= 0
    epoch_metric= 0
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
            epoch_metric+= torch.sum((output-target)**2).item()
        elif likelihood=="classification":
            epoch_metric+= torch.sum(torch.argmax(output.detach(), dim=-1)==target).item()

        batch_loss.backward(retain_graph= True)
        optimizer.step()

        epoch_loss += batch_loss.item()

    return epoch_loss/len(train_loader), epoch_metric/N


def train_model(model, train_loader, val_loader, n_epochs, early_stopping, likelihood: str, learning_rate, device, hyperparameters):
    path_to_project= "/Users/victoriabarenne/ALbias"
    if likelihood=="regression":
        loss_fn= nn.MSELoss(reduction="mean")
    elif likelihood=="classification":
        loss_fn= nn.CrossEntropyLoss(reduction="mean")
    best_loss= np.inf
    patience=0
    best_model= model

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
    for epoch in range(n_epochs):
        epoch_loss, epoch_metric= train_epoch_model(model, train_loader, optimizer, loss_fn, device, hyperparameters, likelihood)
        val_loss, val_metric = evaluate(model, val_loader, device, likelihood)

        if val_loss > best_loss:
            patience += 1
        elif val_loss <= best_loss:
            best_loss = deepcopy(val_loss)
            torch.save(model.state_dict(), path_to_project + "/best_model.pth")
            patience = 0
        if patience >= early_stopping:
            print(f"Early stopping at {epoch} epochs")
            break

        print(f"Epoch {epoch + 1:0>3d} eval: Val nll: {val_loss:.4f}, Val Performance: {val_metric}")
    best_model.load_state_dict(torch.load(path_to_project + "/best_model.pth"))
    return best_model


def evaluate(model, val_loader, device, likelihood):
    model.train()
    model.to(device)
    loss= 0
    metric=0

    if likelihood=="regression":
        loss_fn= nn.MSELoss(reduction="mean")
    elif likelihood=="classification":
        loss_fn= nn.CrossEntropyLoss(reduction="mean")

    for batch_id, (data, target) in enumerate(val_loader):
        # zero the parameter gradients
        data, target = data.to(device), target.to(device)
        output = model(data)
        batch_loss = loss_fn(output, target)
        loss += batch_loss.item()
        if likelihood=="regression":
            metric+= torch.sum((output-target)**2).item()
        elif likelihood=="classification":
            metric+= torch.sum(torch.argmax(output.detach(), dim=-1)==target).item()

    return loss / len(val_loader), metric / len(val_loader.dataset)


def mle_training(model, train_loader, test_loader, likelihood, n_epochs, lr_param, lr_hyper, F=1, B=0, K=1, hessian_structure="kron", device= "cpu", prior_init=1., sigma_init=1., temperature_init=1.):
    path_to_project= "/Users/victoriabarenne/ALbias"
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
    if likelihood=="regression":
        hyper_optimizer = torch.optim.Adam(hyperparameters, lr=lr_hyper)
    elif likelihood=="classification":
        hyper_optimizer = torch.optim.Adam([hyperparameters[0]], lr=lr_hyper)


    # 2) Run the optimization of both model weights and hyperparameters
    best_marglik = np.inf
    best_model=model

    # scheduler= torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000)
    for epoch in range(1,n_epochs+1):
        epoch_training_loss, epoch_training_metric= train_epoch_model(model, train_loader, optimizer, loss_fn, device, hyperparameters, likelihood)
        epoch_testing_loss, epoch_testing_metric= evaluate(model, test_loader, device, likelihood)
        wandb.log(data= {f"test_loss":epoch_testing_loss}, step=epoch)
        wandb.log(data= {f"test_metric":epoch_testing_metric}, step=epoch)

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
            wandb.log(data={f"neg_margliks": neg_marglik.item()}, step=epoch)
            wandb.log(data={f"priors": log_prior.exp().item()}, step=epoch)
            if likelihood=="regression":
                wandb.log(data={f"sigmas": log_sigma.exp().item()}, step=epoch)

            if neg_marglik< best_marglik:
                torch.save(model.state_dict(), path_to_project + "/best_model.pth")
                best_marglik= deepcopy(neg_marglik.item())
                best_prior= deepcopy(la.prior_precision.detach().numpy())
                if likelihood=="regression":
                    best_sigma= deepcopy(la.sigma_noise.detach().numpy())
            print(f"Epoch {epoch}: Training Loss {epoch_training_loss} Log MargLikelihood: {neg_marglik}")
        else:
            print(f"Epoch {epoch}: Training Loss {epoch_training_loss:.6f} Log MargLikelihood: not updated")

    best_model.load_state_dict(torch.load(path_to_project + "/best_model.pth"))

    best_hyperparameters= [best_prior] if likelihood=="classification" else [best_prior, best_sigma]
    return la, best_model, best_hyperparameters