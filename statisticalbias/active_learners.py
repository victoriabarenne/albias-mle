import numpy as np
from IPython import embed
import random
from proposal_distributions import ProposalDistribution, UniformProposal
import seaborn as sns
import matplotlib.pyplot as plt

class ActiveLearner:
    def __init__(self, dataset, model=None):
        self.model=model
        self.dataset=dataset
        self.n_pool=self.dataset.n_points

    def query(self, M, n_initial=1, b=1):
        pass

    def plot_al(self):
        plot

class RandomSampler(ActiveLearner):
    def __init__(self, dataset):
        super(RandomSampler, self).__init__(dataset)
        self.name="Random sampler"

    def query(self, M):
        n_labeled = np.sum(self.dataset.labeled)
        if self.n_pool-n_labeled>=M:
            idx = np.random.choice(np.where(self.dataset.labeled == 0)[0], M, replace=False)
            probs= np.full((M,1), 1/(self.n_pool-n_labeled))
        else:
            print(f"There are only {self.n_pool-n_labeled} unlabeled points left "
                  f"and you are trying to label {M} new points.")
        self.dataset.observe(idx)
        return probs


class BoltzmanSampler(ActiveLearner):
    def __init__(self, dataset, model):
        super(BoltzmanSampler, self).__init__(dataset, model)
        self.name="Boltzman sampler"

    def query(self, M, n_initial=1):

        #Random initial pool
        if np.sum(self.dataset.labeled)==0:
            initial_idx= np.random.choice(range(self.n_pool), n_initial)
            self.dataset.observe(initial_idx)
        elif np.sum(self.dataset.labeled)>0:
            n_initial=0

        #Selection of new queries
        for i in range(n_initial, M):
            self.model.train_epoch(self.dataset.queries)
            y_al = self.dataset.y[self.dataset.queries].reshape(1,-1)
            x_al = self.dataset.x[self.dataset.queries].reshape(1, -1)
            y_train_pred= self.model.predict(np.arange(0, self.n_pool)).reshape(-1,1)
            distance= (y_train_pred-y_al)**2
            distance= (self.dataset.x.reshape(-1,1)-x_al)**2

            if np.max(np.min(distance, axis=1))>0:
                id= np.argmax(np.min(distance, axis=1))
            else:
                id= np.random.choice(np.where(self.dataset.labeled==0)[0])
            self.dataset.observe(id)

        return None



class ProposalDistributionSampler(ActiveLearner):
    def __init__(self, dataset, proposal, model=None):
        super(ProposalDistributionSampler, self).__init__(dataset, model)
        self.proposal=proposal
        self.name="Proposal distribution Sampler"

    def query(self, M, n_initial=1):

        #Random initial pool
        probs= np.zeros(M)
        start=n_initial
        if np.sum(self.dataset.labeled)==0:
            initial_idx= np.random.choice(range(self.n_pool), n_initial)
            self.dataset.observe(initial_idx)
            probs[0:n_initial] = 1 / np.array(range(self.n_pool, self.n_pool - n_initial, -1))
            start=2*n_initial
        for i in range(start-n_initial, M):
            id, probs[i]= self.proposal.propose_query(self.dataset)
            self.dataset.observe(id)

        return probs.reshape(-1,1)
