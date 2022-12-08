import numpy as np
from IPython import embed
import torch
import torchvision
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import random
import copy
import seaborn as sns
import torchvision.datasets as datasets
import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
from models import LinearRegressor
from active_learners import ActiveLearner, BoltzmanSampler, ProposalDistributionSampler, RandomSampler
from proposal_distributions import MyProposal, ProposalDistribution
from datasets import SinusoidalData2D
from estimators import TrueEstimator, RpureEstimator, RlureEstimator, BiasedEstimator
import pandas as pd
from sklearn.metrics import mean_squared_error

#np.random.seed(1)
N = 101
m = 1
M = 10
dataset_pool = SinusoidalData2D(n_points=N, random_state=1)

def figure1(M):

    # dataset_pool.plot_data(show=False)

    # True linear regression on the whole dataset
    true_linear_regressor= LinearRegressor(dataset_pool)
    true_linear_regressor.train(np.arange(0, dataset_pool.n_points))
    true_linear_regressor.plot_regressor(label="True estimator on Dpool")

    # Random Sampling on the dataset
    random_learner=RandomSampler(dataset=dataset_pool)
    random_learner.query(M)
    #dataset_pool.plot_labeled()
    #linear_regressor.plot_regressor(dataset_pool.queries, label="Regressor trained on randomly sampled points")

    #Boltzman
    # dataset_pool.restart()
    # boltzman_learner= BoltzmanSampler(dataset=dataset_pool, model=true_linear_regressor)
    # boltzman_learner.query(M)
    # #dataset_pool.plot_labeled()
    # #linear_regressor.plot_regressor(dataset_pool.queries, label="Regressor trained on Boltzman AL points", show=False)

    #Rpure and Rlure
    dataset_pool.restart()
    my_proposal=MyProposal(hyperparameters=[0.1])
    proposal_learner= ProposalDistributionSampler(dataset=dataset_pool, model=None, proposal=my_proposal)
    acq_probs= proposal_learner.query(M)
    dataset_pool.plot_labeled()
    acq_weights_pure, R_pure= RpureEstimator(dataset_pool, mean_squared_error).evaluate_risk(true_linear_regressor, acq_probs)
    acq_weights_lure, R_lure= RlureEstimator(dataset_pool, mean_squared_error).evaluate_risk(true_linear_regressor, acq_probs)

    linear_regressor= LinearRegressor(dataset_pool)
    linear_regressor.train(dataset_pool.queries)
    linear_regressor.plot_regressor(label="Unweighted estimator on AL points", show=False)
    linear_regressor.train(dataset_pool.queries, weights= acq_weights_pure.squeeze())
    linear_regressor.plot_regressor(label="Unbiased Rpure", show=False)
    linear_regressor.train(dataset_pool.queries, weights= acq_weights_lure.squeeze())
    linear_regressor.plot_regressor(label="Unbiased Rlure", show=False)
    plt.title(f"Linear Regression trained on {M} actively sampled points")
    plt.show()
    y_true= dataset_pool.y[dataset_pool.queries]
    y_pred= true_linear_regressor.predict(dataset_pool.queries)
    loss = np.array([mean_squared_error(y_true[i], y_pred[i]) for i in range(0, len(y_true))]).reshape(-1,1)
    weights= pd.DataFrame(np.concatenate([acq_probs, acq_weights_pure.reshape(-1,1), acq_weights_lure.reshape(-1,1), loss],
                                         axis=1), columns=["Acquisition probability", "Rpure weights", "Rlure weights", "Loss"])
    weights.to_csv(f"Weights_{M}samples.csv")


def figure2(M, n_iter):
    # True linear regression on the whole dataset
    true_linear_regressor= LinearRegressor(dataset_pool)

    # Linear regressor to train at each iteration
    linear_regressor = LinearRegressor(dataset_pool)

    #Rpure and Rlure
    dataset_pool.restart()
    my_proposal = MyProposal(hyperparameters=[0.1])
    proposal_learner = ProposalDistributionSampler(dataset=dataset_pool, model=None, proposal=my_proposal)
    y_predictions_pure, y_predictions_lure= np.zeros(shape=(N,1)), np.zeros(shape=(N,1))
    y_predictions_biased= np.zeros(shape=(N,1))
    for n in range(n_iter):
        dataset_pool.restart()
        acq_probs= proposal_learner.query(M)

        acq_weights_pure, R_pure= RpureEstimator(dataset_pool, mean_squared_error).evaluate_risk(true_linear_regressor, acq_probs)
        acq_weights_lure, R_lure= RlureEstimator(dataset_pool, mean_squared_error).evaluate_risk(true_linear_regressor, acq_probs)
        linear_regressor.train(dataset_pool.queries)
        y_predictions_biased+= linear_regressor.predict(np.arange(dataset_pool.n_points))
        linear_regressor.train(dataset_pool.queries, acq_weights_pure)
        y_predictions_pure += linear_regressor.predict(np.arange(dataset_pool.n_points))
        linear_regressor.train(dataset_pool.queries, acq_weights_lure)
        y_predictions_lure += linear_regressor.predict(np.arange(dataset_pool.n_points))

    y_predictions_pure=y_predictions_pure/n_iter
    y_predictions_lure=y_predictions_lure/n_iter
    y_predictions_biased=y_predictions_biased/n_iter

    dataset_pool.plot_data()
    true_linear_regressor.plot_regressor(label="True estimator on Dpool")
    sns.lineplot(x=dataset_pool.x.squeeze(), y=y_predictions_pure.squeeze(), label="Unbiased Rpure")
    sns.lineplot(x=dataset_pool.x.squeeze(), y=y_predictions_lure.squeeze(), label="Unbiased Rlure")
    sns.lineplot(x=dataset_pool.x.squeeze(), y=y_predictions_biased.squeeze(), label="Biased")
    plt.title(f"Regressor averaged over {n_iter} iterations of sampling points")
    plt.xlim(xmin=-1.5, xmax= 1.5)
    plt.ylim(ymin=-0.5, ymax=2.5)
    plt.show()


dataset_pool.restart()
true_linear_regressor = LinearRegressor(dataset_pool)
true_linear_regressor.train(np.arange(0, dataset_pool.n_points))

R_hat = TrueEstimator(dataset_pool, mean_squared_error).evaluate_risk(true_linear_regressor)
R_tilde = np.zeros(dataset_pool.n_points - 1)
R_pure = np.zeros(dataset_pool.n_points - 1)
R_lure = np.zeros(dataset_pool.n_points - 1)

# Query 101 points
my_proposal = MyProposal(hyperparameters=[0.1])
proposal_learner = ProposalDistributionSampler(dataset=dataset_pool, model=None, proposal=my_proposal)

#Ploting bias as a function of budget
def simulate_budget():
    dataset_pool.restart()
    true_linear_regressor= LinearRegressor(dataset_pool)
    true_linear_regressor.train(np.arange(0, dataset_pool.n_points))

    R_hat= TrueEstimator(dataset_pool, mean_squared_error).evaluate_risk(true_linear_regressor)
    R_tilde= np.zeros(dataset_pool.n_points-1)
    R_pure = np.zeros(dataset_pool.n_points-1)
    R_lure = np.zeros(dataset_pool.n_points-1)

    #Query 101 points
    my_proposal = MyProposal(hyperparameters=[0.1])
    proposal_learner = ProposalDistributionSampler(dataset=dataset_pool, model=None, proposal=my_proposal)
    acq_probs = proposal_learner.query(N)
    for M in range(1, N):
        R_tilde[M-1]= BiasedEstimator(dataset_pool, mean_squared_error).evaluate_risk(true_linear_regressor,
                                                                                      query_idx=np.arange(M))
        _, R_pure[M-1] = RpureEstimator(dataset_pool,
                                        mean_squared_error).evaluate_risk(true_linear_regressor, acq_probs,
                                                                          query_idx=np.arange(M))
        _, R_lure[M-1] = RlureEstimator(dataset_pool,
                                        mean_squared_error).evaluate_risk(true_linear_regressor, acq_probs,
                                                                          query_idx=np.arange(M))
    return R_hat-R_tilde, R_hat-R_pure, R_hat-R_lure

def average_simulate_budget(n_iter=500):
    bias_tilde= np.zeros(shape=(dataset_pool.n_points-1, n_iter))
    bias_pure= np.zeros(shape=(dataset_pool.n_points-1, n_iter))
    bias_lure= np.zeros(shape=(dataset_pool.n_points-1, n_iter))
    for i in range(n_iter):
        print(i)
        bias_tilde[:,i], bias_pure[:,i], bias_lure[:,i]= simulate_budget()

    mean_R_tilde, std_R_tilde = np.mean(bias_tilde, axis=1), np.std(bias_tilde, axis=1)
    mean_R_pure, std_R_pure = np.mean(bias_pure, axis=1), np.std(bias_pure, axis=1)
    mean_R_lure, std_R_lure = np.mean(bias_lure, axis=1), np.std(bias_lure, axis=1)
    x= np.arange(1, dataset_pool.n_points)
    plt.plot(x, mean_R_tilde, 'b--', label='R_tilde')
    plt.fill_between(x, mean_R_tilde - std_R_tilde, mean_R_tilde + std_R_tilde, color='b', alpha=0.2)
    plt.plot(x, mean_R_pure, 'r--', label='R_pure')
    plt.fill_between(x, mean_R_pure - std_R_pure, mean_R_pure + std_R_pure, color='r', alpha=0.2)
    plt.plot(x, mean_R_lure, 'g--', label='R_lure')
    plt.fill_between(x, mean_R_lure - std_R_lure, mean_R_lure + std_R_lure, color='g', alpha=0.2)
    plt.legend(title=f"Bias over {n_iter} iterations")
    plt.show()



    embed()

embed()
true_linear_regressor.evaluate_loss(np.arange(0,N))
average_simulate_budget(100)