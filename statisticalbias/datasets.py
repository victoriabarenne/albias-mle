import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
import seaborn as sns
import random
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Normalize, Compose
import torch

class Dataset:
    def __init__(self, n_points, random_state=1):
        self.random_state=random_state
        self.n_points= n_points
        self.x, self.y= self.generate_data()
        if self.x.ndim==1:
            self.x=self.x.reshape(-1,1)
        self.D = self.x.shape[1:]

class WeightedSubset(Dataset):
    def __init__(self, dataset, idx, weights=None, unsqueeze=False, transform= None):
        if weights is not None:
            assert(len(idx)==len(weights))
        elif weights is None:
            weights= torch.ones(size=(len(idx),1)).squeeze()
        self.data = dataset.data[idx]
        if unsqueeze:
            self.data = self.data.float().unsqueeze(1)
        self.targets = dataset.targets[idx]
        self.weights = weights
        assert((len(self.data)==len(self.targets))&(len(self.data) == len(idx)))
        self.transform= transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if self.transform:
            data= self.transform(self.data[item].float())
        else:
            data= self.data[item].float()
        if self.weights is None:
            weights= 0
        else:
            weights= self.weights[item]

        return data, self.targets[item], weights

class Subset(Dataset):
    def __init__(self, dataset, idx, unsqueeze=False, transform= None):
        self.data = dataset.data[idx]
        if unsqueeze:
            self.data = self.data.float().unsqueeze(1)
        self.targets = dataset.targets[idx]
        assert((len(self.data)==len(self.targets))&(len(self.data) == len(idx)))
        self.transform= transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        if self.transform:
            data= self.transform(self.data[item].float())
        else:
            data= self.data[item].float()

        return data, self.targets[item]

class ActiveLearningDataset(Dataset):
    def __init__(self, n_points, random_state):
        super(ActiveLearningDataset, self).__init__(n_points, random_state)
        self.labeled= np.zeros(self.n_points, dtype=int)
        self.queries = list()

    def restart(self):
        self.labeled= np.zeros(self.n_points, dtype=int)
        self.queries = np.array([], dtype=int)

    def observe(self, idx):
        self.labeled[idx]=1
        if isinstance(idx, int):
            idx= np.array([idx])
        elif idx.ndim==0:
            idx = np.array([idx])

        self.queries= np.concatenate((self.queries, idx), axis=0).astype(int)


class SinusoidalData2D(ActiveLearningDataset):
    def __init__(self, n_points, random_state):
        super(SinusoidalData2D, self).__init__(n_points, random_state)
        self.type= "regression"

    def generate_data(self):
        def density(x):
            if -1.2 <= x < -0.8:
                return 0.12
            elif 0 <= x <= 0.5 or 1 <= x <= 1.5:
                return 0.95
            else:
                return 0

        state = np.random.get_state()
        np.random.seed(self.random_state)
        Y=np.random.uniform(low=-1.5, high=1.5, size=(self.n_points,))
        U=np.random.uniform(size=(self.n_points,))
        np.random.set_state(state)
        W= [density(y) for y in Y]
        X=np.zeros(shape=self.n_points)
        X[U<=W]=Y[U<=W]
        X=X.reshape(-1,1)
        y=np.maximum(0,X)*(np.absolute(X)**1.5+0.25*np.sin(20*X))
        return X, y



class ActiveMNIST(ActiveLearningDataset):
    def __init__(self, noise_ratio=0.1, p_train=0.25, train=True, unbalanced=True, random_state= None):
        self.mnist= torchvision.datasets.MNIST(root="",
                                                     train=train,
                                                     transform= Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]),
                                                     download=True)

        if random_state is not None:
            state = np.random.get_state()
            np.random.seed(random_state)
            state_torch= torch.random.get_rng_state()
            torch.random.manual_seed(random_state)

        if noise_ratio>0:
            idx = np.random.choice(np.arange(len(self.mnist)), int(noise_ratio*len(self.mnist)), replace=False)
            random_labels = torch.randint(high=10, size=(len(idx),))
            self.mnist.targets[idx] = random_labels

        # Select classes proportional to ratio
        if unbalanced:
            class_balance = [1, 0.5, 0.5, 0.2, 0.2, 0.2, 0.1, 0.1, 0.01, 0.01]
        else:
            class_balance= np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        ratio= sum(class_balance)/10
        idx_available = np.array([], dtype=int)
        idx_val= np.array([], dtype=int)
        for target in np.arange(10):
            class_idx = np.where(self.mnist.targets == target)[0]
            if train:
                idx_all = np.random.choice(class_idx, size=int(p_train * class_balance[target] * len(class_idx) + 100* class_balance[target]/ratio),
                                             replace=False)
                idx_available = np.concatenate((idx_available, idx_all[int(100*class_balance[target]/ratio):]), axis=0)
                idx_val= np.concatenate((idx_val, idx_all[:int(100*class_balance[target]/ratio)]), axis=0)
            else:
                idx_all = np.random.choice(class_idx, size=int(p_train * class_balance[target] * len(class_idx)),
                                           replace=False)
                idx_available = np.concatenate((idx_available, idx_all[100:]), axis=0)
        self.idx_available= idx_available

        if train== True:
            self.idx_val = idx_val
            self.validation = WeightedSubset(self.mnist, self.idx_val, unsqueeze=True)


        self.all = WeightedSubset(self.mnist, idx_available, unsqueeze=True)
        self.available = WeightedSubset(self.mnist, idx_available, unsqueeze=True)
        self.train = WeightedSubset(self.mnist, np.array([], dtype=int), unsqueeze=True)
        self.n_points= len(self.all)

        # Reset the seed
        if random_state is not None:
            np.random.set_state(state)
            torch.random.set_rng_state(state_torch)

        self.labeled= np.zeros(len(self.all), dtype=int)
        self.queries = np.array([], dtype=int)
        self.acq_prob = torch.tensor([])


    def get_loader(self, type: str, batch_size, model_arch, laplace, variational_samples= None):
        if type=="train":
            initial_dataset= self.train
        elif type=="available":
            initial_dataset= self.available
        elif type=="validation":
            initial_dataset=self.available
        if laplace:
            if model_arch== "radial_bnn":
                dataset = Subset(initial_dataset, np.arange(len(initial_dataset)), unsqueeze=True)
                dataset.data = dataset.data.expand((-1, variational_samples, -1, -1, -1))
            elif model_arch=="mlp":
                dataset= Subset(initial_dataset, np.arange(len(initial_dataset)), unsqueeze=False)
                dataset.data= dataset.data.view(dataset.data.shape[0], -1)
            elif model_arch=="cnn":
                dataset= Subset(initial_dataset, np.arange(len(initial_dataset)), unsqueeze=False)
            elif model_arch=="resnet":
                dataset= Subset(initial_dataset, np.arange(len(initial_dataset)), unsqueeze=False, transform= torchvision.transforms.Resize(224))
                dataset.data = dataset.data.expand((-1, 3, -1, -1)) # to the right number of channels for the resnet network
        else:
            if model_arch=="radial_bnn":
                dataset= WeightedSubset(initial_dataset, np.arange(len(initial_dataset)), initial_dataset.weights, unsqueeze= True)
                dataset.data = dataset.data.expand((-1, variational_samples, -1, -1, -1))
            elif model_arch=="mlp":
                dataset= WeightedSubset(initial_dataset, np.arange(len(initial_dataset)), initial_dataset.weights, unsqueeze= False)
                dataset.data= dataset.data.view(dataset.data.shape[0], -1)
            elif model_arch=="cnn":
                dataset= WeightedSubset(initial_dataset, np.arange(len(initial_dataset)), initial_dataset.weights, unsqueeze= False)
            elif model_arch == "resnet":
                dataset = WeightedSubset(initial_dataset, np.arange(len(initial_dataset)), initial_dataset.weights, unsqueeze=False, transform=torchvision.transforms.Resize(224))
                dataset.data = dataset.data.expand((-1, 3, -1, -1))  # to the right number of channels for the resnet network

        dataset.targets= dataset.targets.long()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader


    def observe(self, idx, prob, idx_absolute=True):
        self.labeled[idx] = 1
        if isinstance(idx, int):
            idx = np.array([idx])
        elif idx.ndim == 0:
            idx = np.array([idx])

        if isinstance(prob, float):
            prob = np.array([prob])
        elif prob.ndim == 0:
            prob = np.array([prob])

        if idx_absolute == False:
            # idx is the idx in all available data: available[idx]= probs not all[idx]=probs
            idx = np.arange(len(self.all))[self.labeled == 0][idx]

        self.queries = np.concatenate((self.queries, idx), axis=0).astype(int)
        self.acq_prob = np.concatenate((self.acq_prob, prob), axis=0)

        self.train = WeightedSubset(self.all, self.queries, torch.from_numpy(self.acq_prob))
        self.available = WeightedSubset(self.all, np.where(self.labeled == 0)[0])
        print(f"Acquired point {idx} with probability {prob}. Train dataset: {len(self.train)} points")

    def restart(self):
        self.labeled= np.zeros(len(self.all), dtype=int)
        self.queries = np.array([], dtype=int)
        self.acq_prob = torch.tensor([])
        self.available = WeightedSubset(self.mnist, self.idx_available, unsqueeze=True)
        self.train = WeightedSubset(self.mnist, np.array([], dtype=int), unsqueeze=True)






