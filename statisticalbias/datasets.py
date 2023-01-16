import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
import seaborn as sns
import random
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import ToTensor, Normalize
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


class ActiveUnbalancedMNIST(ActiveLearningDataset):
    def __init__(self, noise_ratio=0.1, p_train=0.25, random_state= None):
        mnist_unbalanced= torchvision.datasets.MNIST(root="",
                                                     train=True,
                                                     transform= ToTensor(),
                                                     download=False)

        if random_state is not None:
            state = np.random.get_state()
            np.random.seed(random_state)
            state_torch= torch.random.get_rng_state()
            torch.random.manual_seed(random_state)

        idx = np.random.choice(np.arange(60000), int(noise_ratio*60000), replace=False)
        random_labels = torch.randint(high=10, size=(len(idx),))
        mnist_unbalanced.targets[idx] = random_labels

        # Select classes proportional to ratio
        class_balance = [1, 0.5, 0.5, 0.2, 0.2, 0.2, 0.1, 0.1, 0.01, 0.01]
        idx = np.array([], dtype=int)
        for target in np.arange(10):
            class_idx = np.where(mnist_unbalanced.targets == target)[0]
            idx_all = np.random.choice(class_idx, size=int(p_train * class_balance[target] * len(class_idx)),
                                         replace=False)
            idx = np.concatenate((idx, idx_all), axis=0)

        idx_val = np.isin(idx, np.random.choice(idx, 1000, replace=False))


        self.validation = WeightedSubset(mnist_unbalanced, np.where(idx_val == True)[0], unsqueeze=True)
        self.all = WeightedSubset(mnist_unbalanced, np.where(idx_val == False)[0], unsqueeze=True)
        self.available = WeightedSubset(mnist_unbalanced, np.where(idx_val == False)[0], unsqueeze=True)
        self.train = WeightedSubset(mnist_unbalanced, np.array([], dtype=int), torch.tensor([]), unsqueeze=True)
        self.n_points= len(self.all)

        # Reset the seed
        if random_state is not None:
            np.random.set_state(state)
            torch.random.set_rng_state(state_torch)

        self.labeled= np.zeros(len(self.all), dtype=int)
        self.queries = np.array([], dtype=int)
        self.acq_prob = torch.tensor([])


    def get_dataloaders(self, batch_size):
        train_loader = DataLoader(self.train, batch_size=batch_size, shuffle=True)
        available_loader = DataLoader(self.available, batch_size=batch_size, shuffle=True)
        return train_loader, available_loader

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
        print(len(self.train), len(self.available))



# ################################################################################################################################################
# class MNISTdataset(ActiveLearningDataset):
#     def __init__(self, noise_ratio= 0.1, p_train=0.25, random_state=None):
#         self.all = torchvision.datasets.MNIST(root="",
#                                                  train=True,
#                                                  transform=ToTensor(),
#                                                  download=False)
#         self.random_state= random_state
#
#         ## Creating Unbalanced MNIST
#         # Noise to 10% of the training labels
#         if self.random_state is not None:
#             state = np.random.get_state()
#             np.random.seed(self.random_state)
#             state_torch= torch.random.get_rng_state()
#             torch.random.manual_seed(self.random_state)
#             state_torch_modified= torch.random.get_rng_state()
#
#         idx = np.random.choice(np.arange(60000), int(noise_ratio*60000), replace=False)
#         random_labels = torch.randint(high=10, size=(len(idx),))
#         self.all.targets[idx] = random_labels
#
#         # Select classes proportional to ratio
#         class_balance = [1, 0.5, 0.5, 0.2, 0.2, 0.2, 0.1, 0.1, 0.01, 0.01]
#         idx = np.array([], dtype=int)
#         for target in np.arange(10):
#             class_idx = np.where(self.all.targets == target)[0]
#             idx_all = np.random.choice(class_idx, size=int(p_train * class_balance[target] * len(class_idx)),
#                                          replace=False)
#             idx = np.concatenate((idx, idx_all), axis=0)
#
#         idx_val = np.isin(idx, np.random.choice(idx, 1000, replace=False))
#         self.val = Subset(self.all, np.where(idx_val == True)[0])
#         self.all = Subset(self.all, np.where(idx_val == False)[0])
#         self.n_points= len(self.all)
#         self.available= Subset(self.all, np.arange(len(self.all)))
#         self.train= Subset(self.all, np.array([]))
#
#         # Reset the seed
#         if self.random_state is not None:
#             np.random.set_state(state)
#             torch.random.set_rng_state(state_torch)
#
#         self.labeled= np.zeros(len(self.all), dtype=int)
#         self.queries = np.array([], dtype=int)
#         self.acq_prob = np.array([])
#
#     def get_dataloaders(self, batch_size):
#         train_loader= torch.utils.data.DataLoader(self.train, batch_size=batch_size, shuffle=True)
#         available_loader= torch.utils.data.DataLoader(self.available, batch_size=batch_size, shuffle=True)
#         return train_loader, available_loader
#
#     def observe(self, idx, prob, idx_absolute= True):
#         self.labeled[idx]=1
#         if isinstance(idx, int):
#             idx= np.array([idx])
#         elif idx.ndim==0:
#             idx = np.array([idx])
#
#         if isinstance(prob, float):
#             prob= np.array([prob])
#         elif prob.ndim==0:
#             prob = np.array([prob])
#
#         if idx_absolute==False:
#             #idx is the idx in all available data: available[idx]= probs not all[idx]=probs
#             idx= np.arange(len(self.all))[self.labeled==0][idx]
#
#         self.queries= np.concatenate((self.queries, idx), axis=0).astype(int)
#         self.acq_prob= np.concatenate((self.acq_prob, prob), axis=0)
#
#         idx_train = np.isin(np.arange(len(self.all)), self.queries)
#         self.train= Subset(self.all, np.where(idx_train==True)[0])
#         self.available= Subset(self.all, np.where(idx_train==False)[0])
#         print(len(self.train), len(self.available))
#
#
#
