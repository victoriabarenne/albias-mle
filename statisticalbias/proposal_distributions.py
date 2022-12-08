import numpy as np
from IPython import embed

class ProposalDistribution:
    def __init__(self, hyperparameters=None):
        self.hyperparameters=hyperparameters
    def propose_query(self, dataset):
        pass

class UniformProposal(ProposalDistribution):
    def __init__(self, hyperparameters):
        super(UniformProposal, self).__init__(hyperparameters)

    def propose_query(self, dataset):
        id_query = np.random.choice(np.where(dataset.labeled == 0)[0])
        p_query = 1 / np.sum(dataset.labeled == 0)
        return id_query, p_query

class MyProposal(ProposalDistribution):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        self.eps=hyperparameters[0]

    def propose_query(self, dataset):
        X_al= dataset.x[dataset.queries].reshape(1,-1)
        distance = np.sum(np.absolute(dataset.x[dataset.labeled == 0]- X_al), axis=1)
        u = np.random.uniform(0, 1)
        id = np.arange(dataset.n_points)[dataset.labeled==0][np.argmax(distance)]
        if u>self.eps:
             prob= (1-self.eps)+self.eps/dataset.n_points
        elif u <= self.eps:
            id_rand = np.random.choice(dataset.n_points)
            if id_rand==id:
                prob = (1 - self.eps) + self.eps / dataset.n_points
            else:
                prob= self.eps / dataset.n_points
            id = id_rand

        return id, prob
