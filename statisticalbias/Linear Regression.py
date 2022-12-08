import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
from models import LinearRegressor
import active_learners
from active_learners import ActiveLearner, BoltzmanSampler, ProposalDistributionSampler
from proposal_distributions import MyProposal, ProposalDistribution
import random
import seaborn as sns
from datasets import sample_points

def simulate_bias(dataset, total_budget:list, al_learner: ActiveLearner, n_initial=1, plot=False):
    n_pool=len(dataset)
    bias= {"biased": np.zeros(len(total_budget)), "pure": np.zeros(len(total_budget)) , "lure": np.zeros(len(total_budget))}
    lm_true=LinearRegressor(dataset)
    R_hat = np.mean(lm_true.evaluate_test(dataset[:,0:-1].reshape(n_pool, -1), dataset[:,-1].reshape(-1,1)))

    for i,M in enumerate(total_budget):
        idx_al, probs_al= al_learner.query(M, n_initial=1, b=1)
        dataset_al=dataset[idx_al,:]
        loss_al= lm_true.evaluate_test(dataset_al[:,:-1],
                                       dataset_al[:,-1].reshape(-1,1))
        bias["biased"][i]= np.mean(loss_al)

        w_pure = 1 / (n_pool * probs_al) + (M - np.arange(1, M + 1)) / n_pool
        bias["pure"][i]= np.mean(w_pure* loss_al)

        # if not isinstance(al_learner, ProposalDistributionSampler):
        #     probs_al=np.flip(probs_al)
        #     # for non Proposal distribution active learners, the probs output is: [1/(N-M+1), 1/(N-M+2),..., 1/(N-1), 1/N]
        #     # so that np.mean(w_pure*loss)=np.mean(loss)
        #     # for RLURE we need to have probs [1/N, 1/(N-1), ..., 1/(N-M+2), 1/(N-M+1)] as output so we can just flip the vector

        w_lure = 1 + (n_pool-M) * (1/((n_pool - np.arange(1, M + 1) + 1)*probs_al) - 1) / (n_pool-np.arange(1, M+1))
        bias["lure"][i]= np.mean(w_lure * loss_al)

    bias["biased"], bias["pure"], bias["lure"]= bias["biased"]-R_hat, bias["pure"]-R_hat, bias["lure"]-R_hat

    if plot==True:
        sns.lineplot(x=total_budget, y=bias["biased"], label="Biased")
        sns.lineplot(x=total_budget, y=bias["pure"], label="Rpure")
        sns.lineplot(x=total_budget, y=bias["lure"], label="Rlure")
        plt.title(f"Bias for all estimators with the {al_learner.__class__.__name__}")
        plt.show()

    return bias


np.random.seed(1)
N = 101
m = 1
M = 20
dataset_pool = sample_points(N)
plt.scatter(dataset_pool[:,0], dataset_pool[:,1])
plt.show()
my_regressor=LinearRegressor(dataset=dataset_pool)
boltzman_learner= BoltzmanSampler(dataset=dataset_pool, model=my_regressor)
my_proposal=MyProposal(hyperparameters=[0.1])
proposal_learner= ProposalDistributionSampler(dataset=dataset_pool, model=None, proposal=my_proposal)
total_budget=list(range(1,N))

bias_boltzman=simulate_bias(dataset_pool, total_budget, boltzman_learner, n_initial=1, plot=True)
bias_proposal=simulate_bias(dataset_pool, total_budget, proposal_learner, n_initial=1, plot=True)
embed()

def simulate_bias_average(dataset, total_budget, al_learner: ActiveLearner, n_iter=100, n_initial=1, plot=False):
    average_bias= {"biased": np.zeros(len(total_budget)), "pure": np.zeros(len(total_budget)) , "lure": np.zeros(len(total_budget))}

    if not isinstance(al_learner, ProposalDistributionSampler):
        print("Is not a proposal distribution sampler")
    for i in range(n_iter):
        bias= simulate_bias(dataset, total_budget, al_learner, n_initial, plot=False)
        average_bias["biased"]+=bias["biased"]
        average_bias["pure"]+=bias["pure"]
        average_bias["lure"]+=bias["lure"]
    average_bias["biased"], average_bias["pure"], average_bias["lure"] =average_bias["biased"]/n_iter, average_bias["pure"]/n_iter, average_bias["lure"]/n_iter

    if plot==True:
        sns.lineplot(x=total_budget, y=average_bias["biased"], label="Biased")
        sns.lineplot(x=total_budget, y=average_bias["pure"], label="Rpure")
        sns.lineplot(x=total_budget, y=average_bias["lure"], label="Rlure")
        plt.title(f"Bias for all estimators with the {al_learner.__class__.__name__} over {n_iter} iterations")
        plt.show()

    return average_bias

avg_bias_boltzman=simulate_bias_average(dataset_pool, total_budget, boltzman_learner, n_iter=500, n_initial=1, plot=True)
avg_bias_proposal=simulate_bias_average(dataset_pool, total_budget, proposal_learner, n_iter=500, n_initial=1, plot=True)

embed()

