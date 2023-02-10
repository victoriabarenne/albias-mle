from MLEestimation.models import MLP
import torch
import matplotlib.pyplot as plt
import numpy as np
from MLEestimation.training import train_model, train_epoch_model, mle_training

import os
import pandas as pd
import json
from MLEestimation.datasets import snelson_dataloader

def mle_plotting_saving_snelson(run_name, n_layers, n_epochs, lr_param, lr_hyper, F, B, K, hessian_structure):
    # Snelson dataset
    path_to_data = "/Users/victoriabarenne/ALbias/Experiments/MLEestimation/data/"
    snelson_train = snelson_dataloader(path_to_data, 64)

    #model
    model= MLP(n_hidden_layers=n_layers, n_in=1, n_out=1)


    path_to_run = f"/Users/victoriabarenne/ALbias/{run_name}/"
    if not os.path.exists(path_to_run):
        os.mkdir(path_to_run)

    la, model, neg_margliks, training_losses, perf_metrics, all_hyperparameters= mle_training(model, snelson_train, "regression", n_epochs, lr_param, lr_hyper, F, B, K, hessian_structure=hessian_structure,
                                                                                              device="cpu", prior_init=1., sigma_init=1., temperature_init=1.)
    x, y= snelson_train.dataset[:][0], snelson_train.dataset[:][1]
    x_test = torch.arange(-5, 9, 0.1).view(-1, 1)
    f_mu, f_var = la(x_test)
    f_mu, f_sigma = f_mu.squeeze().detach().cpu().numpy(), f_var.squeeze().sqrt().cpu().numpy()
    pred_std = np.sqrt(f_sigma ** 2 + la.sigma_noise.item() ** 2)

    plt.scatter(x, y, alpha=0.8, s=10)
    plt.plot(x_test, f_mu)
    plt.fill_between(x_test.squeeze(), f_mu - pred_std, f_mu + pred_std, alpha=0.3, color="blue")
    plt.title(f"{n_layers}-layer model")
    plt.savefig(path_to_run + "fit.png")
    plt.show()

    plt.plot(neg_margliks)
    plt.title(f"Negative log marginal likelihood for {n_layers}-layer model (lowest: {min(neg_margliks):.2f})")
    plt.savefig(path_to_run + "neg_margliks.png")
    plt.show()

    plt.plot(training_losses)
    plt.title(f"Training loss for {n_layers}-layer model (lowest: {min(training_losses):.4f})")
    plt.savefig(path_to_run + "losses.png")
    plt.show()

    plt.plot(perf_metrics)
    plt.title(f"MSE {n_layers}-layer model")
    plt.savefig(path_to_run + "metrics.png")
    plt.show()

    df = pd.DataFrame(data={"losses": training_losses, "margliks": neg_margliks, "metrics": perf_metrics, "prior_prec": all_hyperparameters[:,0], "sigma": all_hyperparameters[:,1]})
    df.to_csv(path_to_run + "tracking.csv", sep=',', index=False)

    dict = {"n_epochs": n_epochs, 'lr_param': lr_param, "lr_hyper": lr_hyper,
            "n_layers": n_layers, "hessian_structure": hessian_structure,
            "prior_prec_init": 1., "sigma_noise_init": 1.,
            "temperature": 1.}
    json.dump(dict, open(path_to_run + "hyperparameters.json", 'w'))


def mle_plotting(run_name, model, train_loader, likelihood, n_epochs, lr_param, lr_hyper, F, B, K, hessian_structure, device="cpu", prior_init=1., sigma_init=1., temperature_init=1.):

    path_to_run = f"/Users/victoriabarenne/ALbias/{run_name}/"
    if not os.path.exists(path_to_run):
        os.mkdir(path_to_run)

    la, model, neg_margliks, training_losses, perf_metrics, all_hyperparameters= mle_training(model, train_loader, likelihood, n_epochs, lr_param, lr_hyper,
                                                                                              F, B, K, hessian_structure,
                                                                                              device, prior_init, sigma_init, temperature_init)


    plt.plot(neg_margliks)
    plt.title(f"Negative log marginal likelihood (lowest: {min(neg_margliks):.2f})")
    plt.savefig(path_to_run + "neg_margliks.png")
    plt.show()

    plt.plot(training_losses)
    plt.title(f"Training loss (lowest: {min(training_losses):.4f})")
    plt.savefig(path_to_run + "losses.png")
    plt.show()

    plt.plot(perf_metrics)
    if likelihood=="regression":
        plt.title(f"Mean Squared Error")
    elif likelihood=="classification":
        plt.title(f"Accuracy")
    plt.savefig(path_to_run + "metrics.png")
    plt.show()

    df = pd.DataFrame(data={"losses": training_losses, "margliks": neg_margliks, "metrics": perf_metrics, "prior_prec": all_hyperparameters[:,0], "sigma": all_hyperparameters[:,1]})
    df.to_csv(path_to_run + "tracking.csv", sep=',', index=False)

    dict = {"n_epochs": n_epochs, 'lr_param': lr_param, "lr_hyper": lr_hyper, "hessian_structure": hessian_structure,
            "prior_prec_init": prior_init, "sigma_noise_init": sigma_init,
            "temperature": temperature_init}
    json.dump(dict, open(path_to_run + "hyperparameters.json", 'w'))

