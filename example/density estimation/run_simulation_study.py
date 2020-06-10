import numpy as np
import time
from scipy import stats
from sklearn.linear_model import LinearRegression, Lasso
from sstudy import do_simulation_study
import npcompare as npc

from db_structure import Result, db
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KernelDensity

no_simulations = 30

to_sample = dict(
    no_instances = [100, 200],
    #method = ['npcompare', 'kde'],
    method = ['kde'],
)

def func(
    no_instances,
    method,
    ):

    dists = [stats.beta(1.3, 1.3), stats.beta(1.1, 3.0),
        stats.beta(5.0, 1.1), stats.beta(1.5, 4.0)]
    probs = [.2,.25,.35,.2]

    obs = stats.multinomial(no_instances, probs).rvs().flatten()
    obs = [dists[i].rvs(size=obs[i]) for i in range(4)]
    obs = np.random.permutation(np.hstack(obs))

    eval_grid = np.linspace(0,1,10000,endpoint=False)[1:]
    true_density = [probs[i]*dists[i].pdf(eval_grid) for i in range(4)]
    true_density = np.array(true_density).sum(0)

    start_time = time.time()
    if method == 'npcompare':
        npcobj = npc.EstimateBFS(obs, 5)
        npcobj.sampleposterior(10_000, njobs=3, nchains=3,
            control=dict(adapt_delta=0.99, max_treedepth=15)
        )
        est_densities = npcobj.evaluate(eval_grid)

    elif method == 'kde':
        params_for_kde_cv = {'bandwidth': np.logspace(-2, 3, 100)}
        cv = ShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
        grid = GridSearchCV(KernelDensity(), params_for_kde_cv, cv=cv)
        grid.fit(obs[:,None])

        #obtain estimated density at some points
        est_densities = grid.best_estimator_.score_samples
        est_densities = est_densities(eval_grid[:,None])
        est_densities = np.exp(est_densities).flatten()

    loss = ((est_densities - true_density)**2).mean()
    elapsed_time = time.time() - start_time

    return dict(
        loss = loss,
        elapsed_time = elapsed_time,
    )

def sample_filter(
    no_instances,
    method,
    ):

    if method == 'kde':
        return 500

    return True

do_simulation_study(to_sample, func, db, Result,
    max_count=no_simulations,
    sample_filter=sample_filter)
