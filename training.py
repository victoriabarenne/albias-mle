from torch.nn.utils import parameters_to_vector
from laplace.curvature.backpack import BackPackGGN
from laplace.curvature.asdl import AsdlGGN
import torch
from statisticalbias.radial_layers.loss import Elbo
from laplace import Laplace
from models import initialize_model

import numpy as np
import math
from copy import deepcopy
import wandb

from IPython import embed


def train_epoch(model, dataset, train_loader, loss_helper, optimizer, bias_correction, hyperparameters, likelihood="classification", device="cpu"):
    '''
    Trains model for one epoch
    N: len(dataset.all)
    '''
    model.train()
    model.to(device)
    epoch_loss, epoch_metric, epoch_loglik= 0, 0, 0
    N= dataset.n_points
    M= len(train_loader.dataset)
    m= torch.arange(1, M+1).to(device)

    for batch_id, (data, target, acq_prob) in enumerate(train_loader):
        optimizer.zero_grad()
        data, target, acq_prob = data.to(device), target.to(device), acq_prob.to(device)
        output = model(data)  # shape batch_size x var_samples x n_output for radial_bnn, #batch_size x n_output otherwise
        raw_loss = loss_helper(output, target).squeeze()
        theta= parameters_to_vector(model.parameters()).to(device)
        m_iter= m[batch_id*train_loader.batch_size: batch_id*train_loader.batch_size + len(acq_prob)]

        if bias_correction== "none":
            weight= 1
        elif bias_correction=="pure":
            weight= 1/(N*acq_prob)+(M-m_iter)/N
        elif bias_correction== "lure":
            weight= 1 + (N-M)/(N-m_iter)*(1/((N-m_iter+1)*acq_prob)-1)

        prior_precision = hyperparameters[0].exp().to(device)
        if likelihood== "regression":
            sigma_noise = hyperparameters[1].exp().to(device)
            loglik= -(weight*raw_loss).sum(0)/(2*sigma_noise**2)
            # loglik= -(weight*raw_loss).mean(0)/(2*sigma_noise**2)
        elif likelihood == "classification":
            loglik= -(weight*raw_loss).sum(0)
            # loglik= -(weight*raw_loss).mean(0)
        batch_loss= -loglik + 0.5*prior_precision*(theta@theta)*len(target)/M
        # batch_loss= -loglik + 0.5*prior_precision*(theta@theta)

        if likelihood=="classification":
            epoch_metric += torch.sum(torch.argmax(output, dim=1) == target).item()
        elif likelihood=="regression":
            epoch_metric+= raw_loss.sum().item()

        batch_loss.backward(retain_graph= False)
        optimizer.step()

        # print statistics
        epoch_loss += batch_loss.item()
        epoch_loglik+= loglik.item()

def evaluate(la, model, test_loader, model_arch, likelihood="classification", variational_samples=8, device="cpu"):
    # model.train()
    model.eval()
    model.to(device)
    test_loss, test_metric, test_marglik= 0,0,0
    N_test= len(test_loader.dataset)
    if likelihood=="regression":
        loss_helper= torch.nn.MSELoss(reduction="none")
    elif likelihood=="classification":
        loss_helper= torch.nn.CrossEntropyLoss(reduction="none")
        # loss_helper= torch.nn.NLLLoss(reduction="none")

    for batch_id, (data, target, acq_prob) in enumerate(test_loader):
        data, target, acq_prob = data.to(device), target.to(device), acq_prob.to(device)
        if (model_arch=="mlp") or (model_arch=="cnn"):
            output = torch.stack([model(data) for _ in range(variational_samples)])
            prediction = torch.logsumexp(output, dim=0) - math.log(variational_samples)

        raw_loss= loss_helper(prediction, target).squeeze()
        test_loss += raw_loss.mean().item()
        if likelihood=="regression":
            test_metric+= torch.sum((output-target)**2).item()
        elif likelihood=="classification":
            test_metric+= torch.sum(torch.argmax(prediction.detach(), dim=-1)==target).item()

    if likelihood=="regression":
        neg_marglik= test_loss/(2*la.sigma_noise**2)/len(test_loader) +0.5*(la.scatter+la.log_det_ratio)
    if likelihood=="classification":
        neg_marglik= test_loss/len(test_loader)+0.5*(la.scatter+la.log_det_ratio)

    return test_loss/len(test_loader), test_metric/len(test_loader.dataset), -neg_marglik


def train_all(model, dataset, train_loader, n_epochs, learning_rate, bias_correction, model_arch, early_stopping, hyperparameters, likelihood, variational_samples, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)

    if likelihood == "regression":
        loss_helper = torch.nn.MSELoss(reduction="none")
    elif likelihood == "classification":
        # loss_helper= torch.nn.NLLLoss(reduction="none")
        loss_helper = torch.nn.CrossEntropyLoss(reduction="none")

    for epoch in range(n_epochs):
        train_loss, train_acc= train_epoch(model, dataset, train_loader, loss_helper, optimizer, bias_correction, hyperparameters, likelihood, device)
        print(f"Epoch {epoch+1:0>3d} eval: Train loss: {train_loss:.4f}, Train metric: {train_acc}")
    return model

def train_epoch_marglik(la, dataset, train_loader, loss_helper, hyper_optimizer, bias_correction, likelihood, device):
    N= dataset.n_points
    M = len(train_loader.dataset)
    m = torch.arange(1, M + 1).to(device)
    epoch_loss=  0

    assert ((M < N) or (bias_correction == "none"))
    assert (M <= N)
    for batch_id, (data, target, acq_prob) in enumerate(train_loader):
        hyper_optimizer.zero_grad()
        data, target, acq_prob = data.to(device), target.to(device), acq_prob.to(device)
        output = la.model(data)  # shape batch_size x var_samples x n_output for radial_bnn, #batch_size x n_output otherwise
        raw_loss = loss_helper(output, target).squeeze()
        theta= parameters_to_vector(la.model.parameters()).to(device)
        m_iter= m[batch_id*train_loader.batch_size: batch_id*train_loader.batch_size + len(acq_prob)]

        if bias_correction == "none":
            weight = 1
        elif bias_correction == "pure":
            weight = 1 / (N * acq_prob) + (M - m_iter) / N
        elif bias_correction == "lure":
            weight = 1 + (N - M) / (N - m_iter) * (1 / ((N - m_iter + 1) * acq_prob) - 1)
            weight= torch.nan_to_num(weight, nan=1)
        if likelihood == "regression":
            # loglik= -0.5*(weight*raw_loss).mean(0)/(la.sigma_noise**2)- 0.5*len(target) * torch.log(la.sigma_noise * math.sqrt(2 * math.pi))
            loglik= -0.5*(weight*(raw_loss/(la.sigma_noise**2)+ len(target)* torch.log(la.sigma_noise * math.sqrt(2 * math.pi)))).mean(0)
        elif likelihood=="classification":
            loglik= -(weight*raw_loss).mean(0)
        batch_loss_marglik= -loglik + 0.5 * (la.scatter + la.log_det_ratio)
        batch_loss_marglik.backward(retain_graph=True)
        hyper_optimizer.step()
        epoch_loss += batch_loss_marglik.item()

        return epoch_loss/len(train_loader)



def train_model_marglik(dataset, n_epochs,
                        batch_size_train, model_arch, likelihood, n_layers,
                        lr_param, lr_hyper,
                        F, B, K,
                        hessian_structure,
                        bias_correction,
                        device="cpu"):
    model = initialize_model(model_arch, likelihood, n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_param)
    best_loss= np.inf

    # 3) Initialise hyperparameters and optimizer
    prior = torch.tensor([1.], dtype=torch.float32)
    temperature = torch.tensor([1.], dtype=torch.float32, requires_grad=True)
    sigma = torch.tensor([1.], dtype=torch.float32)
    log_prior, log_sigma = torch.log(prior), torch.log(sigma)
    log_prior.requires_grad, log_sigma.requires_grad = True, True
    hyperparameters = [log_prior, log_sigma] if likelihood == "regression" else [log_prior]
    hyper_optimizer = torch.optim.Adam(hyperparameters, lr=lr_hyper)

    # 4) Loss helper
    if likelihood == "regression":
        loss_helper = torch.nn.MSELoss(reduction="none")
    elif likelihood == "classification":
        loss_helper = torch.nn.CrossEntropyLoss(reduction="none")
        # loss_helper= torch.nn.NLLLoss(reduction="none")
    backend = AsdlGGN if likelihood == "classification" else BackPackGGN

    for n in range(n_epochs):
        train_loader= dataset.get_loader("train", batch_size_train, model_arch, laplace= False)
        train_epoch(model, dataset, train_loader, loss_helper, optimizer, bias_correction, hyperparameters, likelihood)

        if ((n%F==0) and n>B):
            la = Laplace(model, likelihood,
                         hessian_structure=hessian_structure,
                         sigma_noise=log_sigma.exp().to(device),
                         prior_precision=log_prior.exp().to(device),
                         temperature=temperature.to(device),
                         subset_of_weights='all',
                         backend=backend)
            train_loader= dataset.get_loader("train", batch_size_train, model_arch, laplace= True)
            la.fit(train_loader)
            train_loader= dataset.get_loader("train", batch_size_train, model_arch, laplace= False)
            for k in range(K):
                epoch_loss= train_epoch_marglik(la, dataset, train_loader, loss_helper, hyper_optimizer, bias_correction, likelihood, device)
                if epoch_loss< best_loss:
                    torch.save(model.state_dict(), "/Users/victoriabarenne" + "/best_model.pth")
                    best_marglik = deepcopy(epoch_loss)
    model.load_state_dict(torch.load("/Users/victoriabarenne" + "/best_model.pth"))
    return model, la, best_marglik


def bias_corrected_marglik(la, dataset, train_loader, bias_correction, likelihood, prior_precision, sigma_noise=None, device="cpu"):
    N = dataset.n_points
    M = len(train_loader.dataset)
    m = torch.arange(1, M + 1).to(device)
    assert ((M < N) or (bias_correction == "none"))
    assert (M <= N)
    weighted_loss = 0
    if prior_precision is not None:
        la.prior_precision= prior_precision
    if sigma_noise is not None:
        if la.likelihood != 'regression':
            raise ValueError('Can only change sigma_noise for regression.')
        la.sigma_noise=sigma_noise

    if likelihood == "regression":
        loss_helper = torch.nn.MSELoss(reduction="none")
    elif likelihood == "classification":
        # loss_helper = torch.nn.NLLLoss(reduction="none")
        loss_helper= torch.nn.CrossEntropyLoss(reduction="none")

    for batch_id, (data, target, acq_prob) in enumerate(train_loader):
        output = la.model(data)
        raw_loss = loss_helper(output, target)

        m_iter = m[batch_id * train_loader.batch_size: batch_id * train_loader.batch_size + len(acq_prob)]
        if bias_correction == "none":
            weight = 1
        elif bias_correction == "pure":
            weight = 1 / (N * acq_prob) + (M - m_iter) / N

        elif bias_correction == "lure":
            weight = 1 + (N - M) / (N - m_iter) * (1 / ((N - m_iter + 1) * acq_prob) - 1)
            weight= torch.nan_to_num(weight, nan=1)
        weighted_loss -= (weight * raw_loss).sum()
    if likelihood == "regression":
        log_likelihood = 0.5 * weighted_loss / (la.sigma_noise ** 2) - len(train_loader.dataset) * torch.log(
            la.sigma_noise * math.sqrt(2 * math.pi))
    elif likelihood == "classification":
        log_likelihood = weighted_loss
    bias_corrected_marg_lik = log_likelihood - 0.5 * (la.scatter + la.log_det_ratio)
    wandb.log({"likelihood_loss": log_likelihood})
    wandb.log({"scatter_term": la.scatter})
    wandb.log({"log_det_ratio": la.log_det_ratio})



    return bias_corrected_marg_lik

def compute_score(model, dataset, batch_size, device, acquisition_var_samples, model_arch):
    """
    Returns score for each available point in dataset,
    mutual information between the output and the distribution theta (of the model)
    """
    available_loader= dataset.get_loader(type= "available", batch_size=batch_size, laplace=False, variational_samples=acquisition_var_samples, model_arch=model_arch)
    model.eval()
    model.to(device)
    scores= np.array([])
    for idx, (data, target, weight) in enumerate(available_loader):
        data, target, weight= data.to(device), target.to(device), weight.to(device)
        # Input preprocessing
        output= model(data)
        if model_arch=="radial_bnn":
            output= torch.permute(output, (1, 0, 2)) #var_samples x batch_size x output_dim
        else:
            output= output.unsqueeze(0)

        # Calculating the average entropy
        average_entropy_i= -((output*output.exp()).sum(2)).mean(0) # batch_size
        # Calculating the entropy average
        mean_samples_i_c= (output.exp().sum(0)).log()- torch.log(torch.tensor([acquisition_var_samples]).to(device)) # batch_size x output_dim
        entropy_average_i= -(mean_samples_i_c*mean_samples_i_c.exp()).sum(1) # batch_size

        score= entropy_average_i- average_entropy_i
        scores= np.concatenate((scores, score.cpu().detach().numpy()))
        scores=torch.from_numpy(scores)

        if torch.any(torch.isnan(scores)):
            scores = torch.nan_to_num(scores, nan=0.0)
        scores[scores<0]=0
    return scores

