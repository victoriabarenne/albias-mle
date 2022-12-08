import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from IPython import embed
import seaborn as sns


class Model:
    def __init__(self, dataset, loss_fn=mean_squared_error, hyperparameters=None):
        self.dataset=dataset
        self.hyperparameters=hyperparameters
        self.loss_fn=loss_fn

    def train(self, train_idx):
        pass

    def predict(self, test_idx):
        pass

    def evaluate_test(self, test_idx):
        y_pred=self.predict(test_idx)
        y_test= self.dataset.y[test_idx]
        assert(y_pred.shape==y_test.shape)
        return np.array([self.loss_fn(y_test[i], y_pred[i]) for i in range(len(y_pred))])


class LinearRegressor(Model):
    def __init__(self, dataset, loss_fn=mean_squared_error, hyperparameters=None):
        super().__init__(dataset, loss_fn, hyperparameters)
        self.train(np.arange(0,self.dataset.n_points))

    def train(self, train_idx, weights=None):
        x_train= self.dataset.x[train_idx]
        y_train= self.dataset.y[train_idx]
        self.model = LinearRegression().fit(x_train, y_train, weights)

    def predict(self, test_idx):
        return self.model.predict(self.dataset.x[test_idx])

    def plot_regressor(self, label, show=False):
        y_pred=self.predict(np.arange(0, self.dataset.n_points))
        sns.lineplot(x=self.dataset.x.squeeze(), y=y_pred.squeeze(),
                     label= label)
        if show==True:
            plt.show()

    def evaluate_loss(self, test_idx, weights=None):
        y_true= self.dataset.y[test_idx]
        y_pred= self.predict(test_idx)
        loss=np.array([self.loss_fn(y_true[i], y_pred[i]) for i in np.arange(0,len(test_idx))])
        if weights is not None:
            assert(weights.shape==test_idx.shape)
            loss=loss*weights
        return loss, np.mean(loss)






# class BNN(nn.Module, Model):
#     def __init__(self, mu_prior, sigma_prior):
#         self.sigma= sigma_prior
#         self.mu= mu_prior
#
#
#         self.conv_layer1 = nn.Sequential(
#             bnn.BayesConv2d(prior_mu=self.mu, prior_sigma=self.sigma, in_channels=1, out_channels=16, kernel_size=5),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2)
#         )
#
#         self.conv_layer2 = nn.Sequential(
#             bnn.BayesConv2d(prior_mu=self.mu, prior_sigma=self.sigma, in_channels=16, out_channels=32, kernel_size=5),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2)
#         )
#
#         self.fc_layer = nn.Sequential(
#             bnn.BayesLinear(prior_mu=self.mu, prior_sigma=self.sigma, in_features=??, out_features=128),
#             nn.ReLU(),
#             bnn.BayesLinear(prior_mu=self.mu, prior_sigma=self.sigma, in_features=128, out_features=10)
#         )
#
#     def forward(self, x):
#         out = self.conv_layer1(x)
#         out=self.conv_layer2(out)
#         out = out.view(-1, ??)
#         out = self.fc_layer(out)
#         return out
#
# model = nn.Sequential(
#     bnn.BayesConv2d(prior_mu=mu, prior_sigma=sig, in_channels=1, out_channels=16, kernel_size=5),
#     nn.ReLU(),
#     nn.MaxPool2d(2,2),
#
#     bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=4, out_features=100),
#     nn.ReLU(),
#     bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=100, out_features=3),
# )

# class First(object):
#     def __init__(self):
#         super(First, self).__init__()
#         print("first")
#
# class Second(object):
#     def __init__(self):
#         super(Second, self).__init__()
#         print("second")
#
# class Third(First, Second):
#     def __init__(self):
#         super(Third, self).__init__()
#         print("third")
#
# x_train = np.arange(10).reshape(-1,1)
# y_train=2.3*x_train+1+5*np.random.rand(10,1)
# dataset=np.concatenate([x_train,y_train], axis=1)
#
# lm=LinearRegressor(dataset)

