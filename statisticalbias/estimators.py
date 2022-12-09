import numpy as np

class RiskEstimator:
    def __init__(self, dataset, loss_fn):
        self.dataset=dataset
        self.loss_fn=loss_fn

    def evaluate_risk(self, model):
        pass

class TrueEstimator(RiskEstimator):
    def __init__(self, dataset, loss_fn):
        super(TrueEstimator, self).__init__(dataset, loss_fn)

    def evaluate_risk(self, model):
        y_true= self.dataset.y
        y_pred=model.predict(np.arange(0, self.dataset.n_points))
        loss= [self.loss_fn(y_true[i], y_pred[i]) for i in range(0,len(y_true))]
        assert(len(loss)==len(y_true))
        return np.mean(loss)

class BiasedEstimator(RiskEstimator):
    def __init__(self, dataset, loss_fn):
        super(BiasedEstimator, self).__init__(dataset, loss_fn)

    def evaluate_risk(self, model, query_idx=None):
        if query_idx is None:
            query_idx=np.arange(len(self.dataset.queries))
        y_true= self.dataset.y[self.dataset.queries[query_idx]]
        y_pred= model.predict(self.dataset.queries[query_idx])
        loss= np.array([self.loss_fn(y_true[i], y_pred[i]) for i in range(0, len(y_true))]).reshape(-1,1)
        assert (loss.shape == y_true.shape)
        return np.mean(loss)


class RpureEstimator(RiskEstimator):
    def __init__(self, dataset, loss_fn):
        super(RpureEstimator, self).__init__(dataset, loss_fn)

    def get_weights(self, acq_probs):
        M=len(acq_probs)
        N= self.dataset.n_points
        m= np.arange(1,M+1).reshape(-1,1)
        acq_weights = (1 / (N * acq_probs) + (M - m) / N)
        return acq_weights.squeeze()

    def evaluate_risk(self, model, acq_probs, query_idx=None):
        if query_idx is None:
            query_idx=np.arange(len(self.dataset.queries))
        y_true= self.dataset.y[self.dataset.queries[query_idx]]
        y_pred= model.predict(self.dataset.queries[query_idx])
        loss= np.array([self.loss_fn(y_true[i], y_pred[i]) for i in range(0, len(y_true))])

        acq_weights= self.get_weights(acq_probs[query_idx])
        return np.mean(loss*acq_weights)

class RlureEstimator(RiskEstimator):
    def __init__(self, dataset, loss_fn):
        super(RlureEstimator, self).__init__(dataset, loss_fn)

    def get_weights(self, acq_probs):
        M=len(acq_probs)
        N= self.dataset.n_points
        m= np.arange(1,M+1).reshape(-1,1)
        acq_weights = 1 + (1 / ((N - m + 1) * acq_probs) - 1) * (N - M) / (N - m)
        return acq_weights.squeeze()

    def evaluate_risk(self, model, acq_probs, query_idx=None):
        if query_idx is None:
            query_idx=np.arange(len(self.dataset.queries))
        y_true= self.dataset.y[self.dataset.queries[query_idx]]
        y_pred=model.predict(self.dataset.queries[query_idx])
        loss= np.array([self.loss_fn(y_true[i], y_pred[i]) for i in range(0,len(y_true))])

        acq_weights= self.get_weights(acq_probs[query_idx])
        return np.mean(loss*acq_weights)
