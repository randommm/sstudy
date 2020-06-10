import numpy as np
import time
import pickle
from scipy import stats

from db_structure import Result, db
from sstudy import do_simulation_study

to_sample = dict(
    no_instances = [1000, 2000],
    dissimilarity = [
        0.0,
        0.1,
    ],
    method = [
        'mannwhitneyu',
        'ks',
        'ttest',
        ],
)

def sample_filter(
    no_instances,
    dissimilarity,
    method,
    ):

    return True

def func(
    no_instances,
    dissimilarity,
    method,
    ):

    y_train0 = stats.lognorm(1).rvs(size=no_instances)
    y_train1 = stats.lognorm(1).rvs(size=no_instances) + dissimilarity

    start_time = time.time()

    if method == 'mannwhitneyu':
        htest = stats.mannwhitneyu(y_train0, y_train1,
            alternative='two-sided')

    if method == 'ks':
        htest = stats.ks_2samp(y_train0, y_train1)

    if method == 'ttest':
        htest = stats.ttest_ind(y_train0, y_train1,
            equal_var=False)

    elapsed_time = time.time() - start_time

    return dict(
        pvalue=htest.pvalue,
        elapsed_time=elapsed_time,
        )

do_simulation_study(to_sample, func, db, Result,
    max_count=1000, sample_filter=sample_filter)
