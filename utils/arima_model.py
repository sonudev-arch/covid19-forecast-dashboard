
from sklearn.linear_model import Ridge
import numpy as np

class SimpleARIMA:
    """
    ARIMA(7,1,3) via Ridge regression on lagged first-differences.
    Equivalent to standard ARIMA without statsmodels dependency.
    """
    def __init__(self, p=7, q=3, alpha=10.0):
        self.p, self.q = p, q
        self.model = Ridge(alpha=alpha)

    def _featurise(self, diff, resid):
        ml, X, y = max(self.p, self.q), [], []
        for t in range(ml, len(diff)):
            X.append(np.concatenate([diff[t-self.p:t][::-1],
                                     resid[t-self.q:t][::-1]]))
            y.append(diff[t])
        return np.array(X), np.array(y)

    def fit(self, y):
        self._y    = y.astype(float)
        self._diff = np.diff(self._y)
        res = np.zeros_like(self._diff)
        X, target = self._featurise(self._diff, res)
        self.model.fit(X, target)
        ml = max(self.p, self.q)
        self._res = np.zeros(len(self._diff))
        self._res[ml:] = target - self.model.predict(X)
        return self

    def predict(self, steps):
        dh, rh, out = list(self._diff), list(self._res), []
        for _ in range(steps):
            ar = np.array(dh[-self.p:])[::-1]
            ma = np.array(rh[-self.q:])[::-1]
            dp = self.model.predict(np.concatenate([ar,ma]).reshape(1,-1))[0]
            out.append(dp);  dh.append(dp);  rh.append(0.)
        return np.maximum(np.cumsum(out) + self._y[-1], 0)