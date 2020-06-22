import numpy as np
import time
from scipy import stats
from sstudy import do_simulation_study
from nnlocallinear import NNPredict
import torch
from db_structure import Result, db

to_sample = dict(
    hidden_size = np.linspace(50, 500, 14),
    seed = range(1200),
    dropout = [False, True],
    batch_normalization = [False, True],
)

def func(
    hidden_size,
    seed,
    dropout,
    batch_normalization,
    ):

    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.manual_seed(10)
    reg = NNPredict(
        hidden_size = hidden_size,
        num_layers = 2,
        batch_initial = 100,
        dropout_rate = 0.5 if dropout else 0.0,
        batch_normalization = batch_normalization,
        optim_lr = 0.7,
        verbose=2,
        es_max_epochs = 100,
    )

    x_train = stats.norm.rvs(size=(2000, 2))
    y_train = np.cos(x_train[:,[0]]) + np.sin(x_train[:,[1]])
    y_train += stats.norm.rvs(size=(2000, 1))

    x_test = x_train[:1000]
    y_test = y_train[:1000]
    x_train = x_train[1000:]
    y_train = y_train[1000:]

    reg.fit(x_train, y_train)
    mse_train = - reg.score(x_train, y_train)
    mse_test = - reg.score(x_test, y_test)

    return dict(
        mse_train = mse_train,
        mse_test = mse_test,
    )

def sample_filter(
    hidden_size,
    seed,
    dropout,
    batch_normalization,
    ):

    return True

do_simulation_study(to_sample, func, db, Result,
    max_count=1,
    sample_filter=sample_filter)
