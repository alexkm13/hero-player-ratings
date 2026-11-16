"""Count model used by the A3Z action-incidence experiments."""
import numpy as np
from scipy import sparse
from scipy.optimize import minimize


class PoissonRidge:
    def __init__(self, penalties):
        self.penalties = np.asarray(penalties, float)

    def fit(self, x: sparse.csr_matrix, y: np.ndarray, hours: np.ndarray):
        assert x.shape[1] == len(self.penalties)
        assert np.all(hours > 0)
        init = np.r_[np.log(y.sum() / hours.sum()), np.zeros(x.shape[1])]

        def objective(theta):
            eta = theta[0] + x @ theta[1:]
            mu = hours * np.exp(eta)
            residual = mu - y
            loss = (np.sum(mu - y * eta) +
                    .5 * np.dot(self.penalties * theta[1:], theta[1:]))
            grad = np.r_[residual.sum(),
                np.asarray(x.T @ residual).ravel() + self.penalties * theta[1:]]
            return float(loss), grad

        result = minimize(objective, init, method='L-BFGS-B', jac=True,
                          options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-4})
        self.theta = result.x
        self.fit_info = {'success': bool(result.success), 'message': str(result.message),
                         'iterations': int(result.nit), 'objective': float(result.fun)}
        if not result.success:
            raise RuntimeError(self.fit_info)
        return self

    def predict(self, x: sparse.csr_matrix, hours: np.ndarray) -> np.ndarray:
        return hours * np.exp(self.theta[0] + x @ self.theta[1:])
