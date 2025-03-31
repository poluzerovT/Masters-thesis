import numpy as np
import pandas as pd
from abc import abstractmethod
from scipy import optimize as opt

class Strategy:
    def __init__(self, name, period, offset):
        self.name = name
        self.period = period
        self.offset = offset
      
    @abstractmethod
    def fit(self, x_hist):
        pass

    def predict(self, x_now):
        if not hasattr(self, '_w'):
            raise Exception('not fitted yet')
        return self._w
    
    @property
    def mean_roi_est(self):
        """
        Estimated portfolio return mean
        """
        return self._mean_roi_est
    @property
    def var_roi_est(self):
        """
        Estimated variance return portfolio
        """
        return self._var_roi_est
    def __repr__(self):
        return f'{self.name}'
    def __str__(self):
        return self.__repr__()

def mu_cov_estimate(df, period, max_leng=None):
    ret = (df.diff(period) / df.shift(period)).dropna()
    if max_leng:
        ret = ret.tail(max_leng)
    if ret.shape[0] < 2:
        raise Exception(f'too short dataset: {df.shape} with period {period}')
    mu_hat = ret.mean(axis=0).values
    cov_hat = ret.cov().values
    return mu_hat, cov_hat

class SingleAsset(Strategy):
    def __init__(self, name, period, offset, idx):
        super().__init__(name, period, offset)
        self.idx = idx

    def fit(self, x_hist):
        self._n = x_hist.shape[1]
        w = np.zeros((1, self._n))
        w[0, self.idx] = 1
        self._w = w

        mu, cov = mu_cov_estimate(pd.DataFrame(x_hist), self.period)

        self._mean_roi_est = (self._w @ mu).item()
        self._var_roi_est = (self._w @ cov @ self._w.T).item()
        return self
    
def markowitz_solver(mu_hat, cov_hat, risk_tolerance):
    n = mu_hat.shape[0]

    def objective(x, mu, cov, risk_tolerance):
        x = x.reshape((-1, 1))
        return x.T @ cov @ x - risk_tolerance * x.T @ mu

    def constraint(x):
        return np.abs(x).sum() - 1

    eq_cons = {
        'type': 'eq',
        'fun': constraint
    }

    bounds = [(-1, 1) for i in range(n)]
    x0 = np.zeros(n)
    x0[0] = 1

    sol = opt.minimize(
        fun=objective,
        x0=x0,
        args=(mu_hat, cov_hat, risk_tolerance),
        method='SLSQP',
        bounds=bounds,
        constraints=[eq_cons]
    )
    return sol.x.reshape((1, -1)), sol

class MarkowitzClassic(Strategy):
    def __init__(self, name, period, offset, risk_tolerance):
        super().__init__(name, period, offset)
        self.risk_tolerance = risk_tolerance

    def fit(self, x_hist):
        mu, cov = mu_cov_estimate(pd.DataFrame(x_hist), self.period)
        w, meta = markowitz_solver(mu, cov, self.risk_tolerance)
        self._w = w

        self._mean_roi_est = (self._w @ mu).item()
        self._var_roi_est = (self._w @ cov @ self._w.T).item()
        return self
    
class MarkowitzMartingal(Strategy):
    pass
class MarkowitzTimeSeries(Strategy):
    pass

if __name__ == '__main__':
    d = {}
    d[MarkowitzClassic('abc', 1, 1, 1)] = 1
    print(d)