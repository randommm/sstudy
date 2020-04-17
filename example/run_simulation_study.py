import numpy as np
import time
from scipy import stats
from sklearn.linear_model import LinearRegression, Lasso
from sstudy import do_simulation_study

from db_structure import Result, db

no_simulations = 10

to_sample = dict(
    data_distribution = [0,1],
    no_instances = [100, 1000],
    method = ['ols', 'lasso'],
)

def func(
    data_distribution,
    no_instances,
    method,
    ):

    x = stats.norm.rvs(0, 2, size=(no_instances + 10000, 10))
    beta = stats.norm.rvs(0, 2, size=(10, 1))
    eps = stats.norm.rvs(0, 3, size=(no_instances + 10000, 1))
    if data_distribution == 0:
        y = np.matmul(x, beta) + eps
    if data_distribution == 1:
        y = np.matmul(x[:,:5], beta[:5]) + eps

    y_train = y[:no_instances]
    y_test = y[no_instances:]
    x_train = x[:no_instances]
    x_test = x[no_instances:]

    start_time = time.time()
    if method == 'ols':
        reg = LinearRegression()
    elif method == 'lasso':
        reg = Lasso(alpha=0.1)
    reg.fit(x_train, y_train)
    score = reg.score(x_test, y_test)
    elapsed_time = time.time() - start_time

    return dict(
        score = score,
        elapsed_time = elapsed_time,
    )

def sample_filter(
    data_distribution,
    no_instances,
    method,
    ):

    return True

do_simulation_study(to_sample, func, db, Result,
    max_count=no_simulations,
    sample_filter=sample_filter)
