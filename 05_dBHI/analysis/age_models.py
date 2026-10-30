"""Ridge age model shared by the main and child-centred analyses."""

import numpy as np
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-2, 4, 25)
OBSERVED_REPEATS = 5
# The submitted analysis used one CV repeat per label permutation; see the README.
PERMUTATION_REPEATS = 1
N_PERM = 1000
SEED = 0


def ridge():
    """Standardise inside the fold, then RidgeCV with the 25 log-spaced penalties."""
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))


def oos(X, y, seed=0, rep=OBSERVED_REPEATS):
    """Return r and the mean held-out age prediction across CV repeats (10-fold)."""
    rk = RepeatedKFold(n_splits=10, n_repeats=rep, random_state=seed)
    P = np.zeros((rep, len(y)))
    k = 0
    for tr, te in rk.split(X):
        P[k // 10, te] = ridge().fit(X[tr], y[tr]).predict(X[te])
        k += 1
    p = P.mean(0)
    return stats.pearsonr(p, y)[0], p
