from torch.nn.utils import parameters_to_vector
from laplace.curvature.backpack import BackPackGGN
from laplace.curvature.asdl import AsdlGGN
import torch
from statisticalbias.radial_layers.loss import Elbo
from laplace import Laplace

import numpy as np
import math
from copy import deepcopy
import wandb
from models import MLP_dropout, CNN_dropout, RadialBNN, ResNet, MLP_dropout_fun, initialize_model
from datasets import ActiveMNIST, SnelsonDataset
from IPython import embed
import torchvision
import matplotlib.pyplot as plt
from training import train_epoch, evaluate, compute_score, train_epoch_marglik, train_model_marglik

track=True
run_name= "test_snelson10"
dataset_name="snelson"
# batch_size_train, batch_size_test= 128, 512
batch_size_train, batch_size_test= 32, 64
likelihood= "regression"
lr_param= 0.001
lr_hyper=0.01
n_epochs= 500
variational_samples, variational_samples_acquisition= 8, 100
device= "cpu"
hessian_structure="kron"
model_arch="mlp" # or "radial_bnn", "mlp_complex"
initial_pool=30
goal_points=70

B, K, F= 0, 1, 1
prior_init, temperature_init, sigma_init= 1., 1., 1.
freq_evaluation=3
temperature=15000
points_per_acquisition=1
noise_ratio_training=0.1
unbalanced= False
n_layers= 3

if track:
    wandb.init(
        project="MLE_ALBias",
        name=run_name,
        config={
            "learning_rate_param": lr_param,
            "learning_rate_hyper": lr_hyper,
            "architecture": model_arch,
            "dataset": dataset_name,
            "dataset_unbalanced": unbalanced,
            "noise_ratio_training": noise_ratio_training,
            "initial_pool": initial_pool,
            "goal_points": goal_points,
            "hessian_structure": hessian_structure,
            "epochs": n_epochs,
            "n_layers": n_layers,
            "batch_size_train": batch_size_train,
            "batch_size_test": batch_size_test,
            "hyper_training_frequency": F,
            "hyper_burn_in": B,
            "hyper_epochs":K,
            "likelihood": likelihood,
            "frequency_bias_evaluation":freq_evaluation,
            "points_per_acquisition": points_per_acquisition,
            "temperature": temperature,
            "prior_init": prior_init,
            "sigma_init": sigma_init,
            "var_samples": variational_samples,
            "var_samples_acquisition": variational_samples_acquisition
        }
    )


#0) Initialize pool dataset, validation dataset and test dataset
if likelihood=="classification":
    dataset= ActiveMNIST(noise_ratio= noise_ratio_training, p_train=0.15, train=True, unbalanced= unbalanced, random_state=2) # unbalanced MNIST
    dataset_testing= ActiveMNIST(noise_ratio=0, p_train=1, train=False, unbalanced= unbalanced, random_state=1)
    # val_loader= dataset.get_loader(type= "validation", batch_size=batch_size_test, model_arch=model_arch, laplace= False, variational_samples=variational_samples)
    test_loader= dataset_testing.get_loader(type= "available", batch_size=512, laplace=False, model_arch=model_arch, variational_samples=variational_samples)

if likelihood=="regression":
    dataset = SnelsonDataset("/Users/victoriabarenne/ALBias/Experiments/MLEestimation/data/", random_state=1)
    test_loader= dataset.get_loader(type="validation", batch_size= 32, laplace= False, model_arch=model_arch, variational_samples=variational_samples)


# 1) Initialize using 10 random samples from the training data pool
initial_idx= np.random.randint(low=0, high= len(dataset.available), size=initial_pool)
prob= np.repeat(1/len(dataset.available), repeats= initial_pool)
dataset.observe(initial_idx, prob)
n_acquired= 0

while (len(dataset.train)<goal_points):
    model_acquisition, la_acquisition, _= train_model_marglik(dataset, n_epochs, batch_size_train, model_arch, likelihood, n_layers,
                                           lr_param, lr_hyper, F, B, K,
                                           hessian_structure, "none")
    if (n_acquired + freq_evaluation) % freq_evaluation == 0:
        x= torch.arange(-1,5,0.1).unsqueeze(1)
        y= model_acquisition(x)
        plt.scatter(x.numpy(), y.detach().numpy())
        plt.show()

        model_none, la_none, best_marglik_none = train_model_marglik(dataset, n_epochs, batch_size_train, model_arch, likelihood, n_layers,
                                                lr_param, lr_hyper, F, B, K,
                                                hessian_structure, "none")
        model_pure, la_pure, best_marglik_pure = train_model_marglik(dataset, n_epochs, batch_size_train, model_arch, likelihood, n_layers,
                                                lr_param, lr_hyper, F, B, K,
                                                hessian_structure, "pure")
        model_lure, la_lure, best_marglik_lure = train_model_marglik(dataset, n_epochs, batch_size_train, model_arch, likelihood, n_layers,
                                                lr_param, lr_hyper, F, B, K,
                                                hessian_structure, "lure")

        test_loss_none, test_metric_none, test_marglik_none= evaluate(la_none, model_none, test_loader, model_arch, likelihood, device="cpu")
        test_loss_pure, test_metric_pure, test_marglik_pure= evaluate(la_pure, model_pure, test_loader, model_arch, likelihood, device="cpu")
        test_loss_lure, test_metric_lure, test_marglik_lure= evaluate(la_lure, model_lure, test_loader, model_arch, likelihood, device="cpu")
    if track:
        wandb.log(data={f"test_nll_none": test_loss_none}, step=len(dataset.queries))
        wandb.log(data={f"test_acc_none": test_metric_none}, step=len(dataset.queries))
        wandb.log(data={f"test_marglik_none": test_marglik_none}, step=len(dataset.queries))

        wandb.log(data={f"test_nll_pure": test_loss_pure}, step=len(dataset.queries))
        wandb.log(data={f"test_acc_pure": test_metric_pure}, step=len(dataset.queries))
        wandb.log(data={f"test_marglik_pure": test_marglik_pure}, step=len(dataset.queries))

        wandb.log(data={f"test_nll_lure": test_loss_lure}, step=len(dataset.queries))
        wandb.log(data={f"test_acc_lure": test_metric_lure}, step=len(dataset.queries))
        wandb.log(data={f"test_marglik_lure": test_marglik_lure}, step=len(dataset.queries))


    scores = compute_score(model_acquisition, dataset, batch_size_test, device, variational_samples_acquisition, model_arch)
    q_proposal = torch.nn.Softmax(dim=0)(temperature * scores)
    id = torch.multinomial(q_proposal, num_samples=points_per_acquisition, replacement=False)
    prob = q_proposal[id]
    # TODO: implement such that we don't observe the same point twice
    dataset.observe(id.item(), prob.item(), idx_absolute=False)
    print(f"Acquired new point: {dataset.queries[-1]} with probability {prob.item()}")
    n_acquired += 1

x= torch.arange(-1,5,0.1).unsqueeze(1)
y= model_none(x)
plt.scatter(x.numpy(), y.detach().numpy())
plt.show()

y= model_pure(x)
plt.scatter(x.numpy(), y.detach().numpy())
plt.show()

y= model_lure(x)
plt.scatter(x.numpy(), y.detach().numpy())
plt.show()
embed()



