import numpy as np
import pandas as pd
from scipy.optimize import minimize
class HoltWinters:
    """
    Additive Holt-Winters: triple exponential smoothing.
    Season length = 7 (weekly COVID reporting cycle).
    Parameters (α, β, γ) optimised by Nelder-Mead SSE minimisation.
    """
    def __init__(self, m=7):
        self.m = m

    def _hw(self, y, a, b, g):
        m, L = self.m, np.mean(y[:self.m])
        T = (np.mean(y[m:2*m]) - np.mean(y[:m])) / m
        S = [y[i] - L for i in range(m)]
        fitted = []
        for t in range(len(y)):
            if t == 0:
                fitted.append(L + T + S[0]); continue
            Lp, Tp = L, T
            L = a*(y[t]-S[t%m]) + (1-a)*(Lp+Tp)
            T = b*(L-Lp)        + (1-b)*Tp
            S[t%m] = g*(y[t]-L) + (1-g)*S[t%m]
            fitted.append(L + T + S[t%m])
        return np.array(fitted), L, T, S

    def _sse(self, params, y):
        a, b, g = params
        if not (0 < a < 1 and 0 < b < 1 and 0 < g < 1):
            return 1e12
        f, *_ = self._hw(y, a, b, g)
        return np.sum((y - f)**2)

    def fit(self, y):
        y = y.astype(float)
        res = minimize(self._sse, x0=[0.3,0.1,0.3], args=(y,),
                       method='Nelder-Mead', options={'maxiter':3000})
        self.alpha, self.beta, self.gamma = np.clip(res.x, 0.001, 0.999)
        self._fitted, self._L, self._T, self._S = self._hw(y, self.alpha, self.beta, self.gamma)
        self._y = y
        return self

    def predict(self, steps):
        L, T, S, n = self._L, self._T, list(self._S), len(self._y)
        preds = [L + h*T + S[(n+h-1)%self.m] for h in range(1, steps+1)]
        return np.maximum(preds, 0)