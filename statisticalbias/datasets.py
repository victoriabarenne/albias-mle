import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
import seaborn as sns
import random


class Dataset:
    def __init__(self, n_points, random_state=1):
        self.random_state=random_state
        self.n_points= n_points
        self.x, self.y= self.generate_data()
        if self.x.ndim==1:
            self.x=self.x.reshape(-1,1)
        self.D = self.x.shape[1:]

    #TODO: add plotting for clustering data
    def plot_data(self, show=False):
        if self.type== "regression":
            sns.scatterplot(x=self.x.squeeze(), y=self.y.squeeze())
        if self.type== "clustering":
            pass
        if show==True:
            plt.show()



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

    #TODO: add plot for clustering data
    def plot_labeled(self, show=False):
        if self.type== "regression":
            sns.scatterplot(x=self.x[self.labeled==0].squeeze(), y=self.y[self.labeled==0].squeeze())
            sns.scatterplot(x=self.x[self.queries].squeeze(), y=self.y[self.queries].squeeze(),
                            color="red", marker="P", s=150)
        if self.type== "clustering":
            pass
        if show==True:
            plt.show()


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

