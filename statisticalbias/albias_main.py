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
from radial_layers.loss import Elbo
from models import BNN_variational, RadialBNN
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import wandb
import torch.nn.functional as F
import math
from sacred import Experiment
from sacred.observers import FileStorageObserver
from copy import deepcopy

# torch.cuda.empty_cache()

def train_epoch(model, dataset, train_loader, loss, optimizer, bias_correction, variational_samples=8, device="cpu"):
    '''
    Trains model for one epoch
    N: len(dataset.all)
    '''
    model.train()
    model.to(device)
    epoch_loss= 0
    N= dataset.n_points
    M= len(train_loader.dataset)
    m= torch.arange(1, M+1).to(device)
    accuracy=0

    for batch_id, (data, target, acq_prob) in enumerate(train_loader):
        optimizer.zero_grad()
        # zero the parameter gradients
        data, target, acq_prob = data.to(device), target.to(device), acq_prob.to(device)
        # change input to variational
        data = data.unsqueeze(1)
        data = data.expand((-1, variational_samples, -1, -1, -1))
        data= data.float()
        output = model(data)
        output = output.squeeze()

        def loss_helper(prediction, target):
            nll_loss, kl_loss = loss.compute_loss(prediction, target)
            #TODO: regulazing term can be changed? (to other than 1/10)
            return nll_loss + kl_loss / 10

        m_iter= m[batch_id*train_loader.batch_size: batch_id*train_loader.batch_size + len(acq_prob)]
        raw_loss= loss_helper(output, target)

        if bias_correction== "none":
            weight= 1
        elif bias_correction=="pure":
            weight= 1/(N*acq_prob)+(M-m_iter)/N
        elif bias_correction== "lure":
            weight= 1 + (N-M)/(N-m_iter)*(1/((N-m_iter+1)*acq_prob)-1)
        batch_loss= (weight*raw_loss).mean(0)
        accuracy+= torch.sum(torch.argmax(output.mean(dim=1), dim= 1)==target).item()

        batch_loss.backward()
        optimizer.step()

        # print statistics
        epoch_loss += batch_loss.item()
    # return epoch_loss/ len(train_loader)
    return epoch_loss, accuracy/len(train_loader.dataset)


def train_all(model, dataset, train_loader, val_loader, n_epochs, learning_rate, bias_correction, early_stopping=20, variational_samples=8, device="cpu"):
    path_to_project= "/Users/victoriabarenne/ALbias"
    loss = Elbo(binary=False, regression=False)
    loss.set_model(model, train_loader.batch_size)
    loss.set_num_batches(len(train_loader))

    train_losses, val_losses, train_accuracies, val_accuracies= np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
    best_loss= np.inf
    patience=0
    best_model=model

    for epoch in range(n_epochs):
        train_loss, train_acc= train_epoch(model, dataset, train_loader, loss, optimizer, bias_correction, variational_samples, device)
        val_loss, _, val_acc = evaluate(model, val_loader, device, variational_samples, "none")
        train_losses= np.append(train_losses, train_loss)
        train_accuracies= np.append(train_accuracies, train_acc)
        val_losses= np.append(val_losses, val_loss)
        val_accuracies= np.append(val_accuracies, val_acc)
        # wandb.log({f"{logging_name}_tl": train_loss, f"{logging_name}_ta": train_acc,
        #            f"{logging_name}_vl": val_loss, f"{logging_name}_va": val_acc})

        if val_loss>best_loss:
            patience+=1
        elif val_loss<=best_loss:
            best_loss= deepcopy(val_loss)
            torch.save(model.state_dict(), path_to_project+ "/best_model.pth")
            patience = 0

        if patience>=early_stopping:
            print(f"Early stopping at {epoch} epochs")
            break
        print(f"Epoch {epoch+1:0>3d} eval: Val nll: {val_loss:.4f}, Val Accuracy: {val_acc}")
    # return train_losses, val_losses, train_accuracies, val_accuracies
    best_model.load_state_dict(torch.load(path_to_project+ "/best_model.pth"))
    return best_model

def evaluate(model, eval_loader, device, variational_samples, bias_correction):
    model.train()
    model.to(device)

    val_nll= 0
    val_weighted_nll= 0
    correct=0

    N= len(eval_loader.dataset)
    M = len(eval_loader.dataset)
    m= torch.arange(1, M+1).to(device)

    loss = Elbo(binary=False, regression=False)
    loss.set_model(model, eval_loader.batch_size)
    loss.set_num_batches(len(eval_loader))

    for batch_id, (data, target, acq_prob) in enumerate(eval_loader):
        # zero the parameter gradients
        data, target, acq_prob = data.to(device), target.to(device), acq_prob.to(device)
        # change input to variational
        data = data.unsqueeze(1)
        data = data.expand((-1, variational_samples, -1, -1, -1))
        output = model(data)
        output = output.squeeze()

        #DEAL WITH RAW LOSS
        #TODO: review raw loss
        #prediction= log(mean(exp(output), dim=-1)) ie mean over the variational samples
        prediction = torch.logsumexp(output, dim=1) - math.log(variational_test_train)
        raw_loss= F.nll_loss(prediction, target, reduction="none")
        m_iter= m[batch_id*eval_loader.batch_size: batch_id*eval_loader.batch_size + len(acq_prob)]

        if bias_correction== "none":
            weight=1
        elif bias_correction=="pure":
            weight= 1/(N*acq_prob)+(M-m_iter)/N
        elif bias_correction== "lure":
            weight= 1 + (N-M)/(N-m_iter)*(1/((N-m_iter+1)*acq_prob)-1)

        val_nll+=raw_loss.sum().item()
        val_weighted_nll += (weight*raw_loss).sum().item()

        if torch.any(torch.isnan(torch.tensor([val_nll]))):
            print(weight)
            print(torch.isnan(torch.tensor([val_nll])).sum())

        class_prediction = prediction.max(1, keepdim=True)[1].squeeze()
        correct += (class_prediction==target).sum().item()
    # return eval_loss/len(eval_loader)
    return val_nll/len(eval_loader.dataset), val_weighted_nll/len(eval_loader.dataset), correct/len(eval_loader.dataset)


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
        data, target, weight= data.to(device), target.to(device), weight.to(device)
        # Input preprocessing
        data= data.unsqueeze(1)
        data = data.expand((-1, acquisition_var_samples, -1, -1, -1))

        # Output
        output= model(data)
        output= torch.permute(output, (1, 0, 2)) #var_samples x batch_size x output_dim

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

track=False
project_name="ALbias_fixed"
run_name= "test_cluster_5"
batch_size=128
batch_size_evaluate= 512
batch_size_scores= 32
n_epochs=100 #100
early_stopping_epochs= 20
device= "cpu" # or "cpu"
learning_rate=5e-4
variational_test_train=8
variational_scoring=100 #should be 100
temperature= 10000
points_per_acquisition= 1
initial_pool=10
goal_points=70 #70
training_frequency=3 #3
B=0

if track ==True:
    wandb.init(
        project=project_name,
        name= run_name,
        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "architecture": "CNN",
            "dataset": "Unbalanced MNIST",
            "epochs": n_epochs,
            "early_stopping_epochs":early_stopping_epochs,
            "variational_test_train":variational_test_train,
            "variational_scoring": variational_scoring,
            "batch_size":batch_size,
            "batch_sze_evaluate": batch_size_evaluate,
            "batch_size_scores": batch_size_scores,
            "temperature": temperature,
            "points_per_acquisition":points_per_acquisition,
            "initial_pool": initial_pool,
            "goal_points": goal_points,
            "training_frequency": training_frequency
        }
    )

#0) Initialize model and training/testing datasets
dataset= ActiveMNIST(noise_ratio= 0.1, p_train=0.15, train=True, unbalanced= True, random_state=2) # unbalanced MNIST
dataset_testing= ActiveMNIST(noise_ratio=0, p_train=1, train=False, unbalanced= True, random_state=1)
val_loader = torch.utils.data.DataLoader(dataset.validation, batch_size=batch_size_evaluate, shuffle=True)
df_testing= pd.DataFrame(columns=["none", "pure", "lure"])
N_test= dataset_testing.n_points
train_losses, val_losses, train_accuracies, val_accuracies= np.array([]), np.array([]), np.array([]), np.array([])

# 1) Initialize using 10 random samples from the training data pool
initial_idx= np.random.randint(low=0, high= len(dataset.available), size=initial_pool)
prob= np.repeat(1/len(dataset.available), repeats= initial_pool)
dataset.observe(initial_idx, prob)
n_acquired= 0

while ((len(dataset.queries)<goal_points)):
    # 2) Train model using Dtrain and R_tilde loss
    model_acquisition = RadialBNN(16)
    train_loader= dataset.get_trainloader(batch_size)
    model_acquisition= train_all(model_acquisition, dataset, train_loader, val_loader, n_epochs, learning_rate, "none",
              early_stopping_epochs, variational_test_train, device)
    dataset_testing.restart()
    test_loader= dataset_testing.get_availableloader(batch_size_evaluate)
    (acquisition_test_nll, _ , acquisition_test_accuracy,
     ) = evaluate(model_acquisition, test_loader, device, variational_test_train, "pure")
    if track==True:
        wandb.log(data= {f"acquisition_test_nll":acquisition_test_nll}, step=len(dataset.queries))
        wandb.log(data= {f"acquisition_test_accuracy":acquisition_test_accuracy}, step=len(dataset.queries))

    # 4) Train models using only the training data using no bias-correction, pure, and lure correction every 3 aquisition rounds
    if (n_acquired==training_frequency) & (len(dataset.queries) > B):
        model_none, model_pure, model_lure= RadialBNN(16), RadialBNN(16), RadialBNN(16)
        train_loader = dataset.get_trainloader(batch_size)
        model_none = train_all(model_none, dataset, train_loader, val_loader, n_epochs, learning_rate, "none",
                               early_stopping_epochs, variational_test_train, device)
        model_pure= train_all(model_pure, dataset, train_loader, val_loader, n_epochs, learning_rate, "pure",
                                                      early_stopping_epochs, variational_test_train, device)
        model_lure= train_all(model_lure, dataset, train_loader, val_loader, n_epochs, learning_rate, "lure",
                                                      early_stopping_epochs, variational_test_train, device)
        n_acquired = 0
        # 5) Evaluate the models using the test set (to get the bias)
        def my_function(model, name):
            # dataset_testing.restart()
            # scores = compute_score(model, dataset_testing, batch_size, device, variational_scoring)
            # q_proposal = torch.nn.Softmax(dim=0)(temperature * scores)
            # idx = torch.multinomial(q_proposal, num_samples=len(scores) - 1, replacement=False)
            # prob = scores[idx]
            # dataset_testing.observe(idx, prob)
            testing_loader = dataset_testing.get_availableloader(batch_size_evaluate)
            test_nll, _, test_acc = evaluate(model, testing_loader, device, variational_test_train, "none")
            if track==True:
                wandb.log(data={f"test_nll_{name}": test_nll}, step=len(dataset.queries))
                wandb.log(data={f"test_acc_{name}": test_acc}, step=len(dataset.queries))
            return test_nll, test_acc


        test_nll_none, test_acc_none= my_function(model_none, "none")
        test_nll_pure, test_acc_pure= my_function(model_pure, "pure")
        test_nll_lure, test_acc_lure= my_function(model_lure, "lure")


    # 3) Select the next acquisition point
    scores= compute_score(model_acquisition, dataset, batch_size_scores, device, variational_scoring)
    q_proposal= torch.nn.Softmax(dim=0)(temperature * scores)
    print(q_proposal.sum())
    id = torch.multinomial(q_proposal, num_samples=points_per_acquisition, replacement=False)
    prob = q_proposal[id]
    #TODO: implement such that we don't observe the same point twice
    dataset.observe(id.item(), prob.item(), idx_absolute=False)
    print(f"Acquired new point: {dataset.queries[-1]} with probability {prob.item()}")
    n_acquired+=1
    np.save("/Users/victoriabarenne/ALbias/weights5.npy", dataset.train.weights)
    np.save("/Users/victoriabarenne/ALbias/idx5.npy", dataset.queries)

embed()
