import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from db_structure import Result

cls = ['-', 'dotted', "--", "-.", ":"]
clw = [1.4, 1.9, 2.5]
colors = ['red', 'magenta', 'green', 'blue', 'yellow', 'purple', 'pink',
    'gray', 'brown', 'orange', 'black']

def ecdf_plot(data, ax, label, bandwidth=True, *args, **kwargs):
    data = np.array(data).flatten()
    alpha_grid = np.linspace(0, 1, 1000).reshape((-1, 1))
    vals = data <= alpha_grid
    mean = vals.mean(1)
    stderror = vals.std(1) / np.sqrt(len(data))
    alpha_grid = alpha_grid.flatten()

    if bandwidth:
        ax.fill_between(alpha_grid, mean+stderror*2,
            mean-stderror*2, alpha=.5, *args, **kwargs)
    ax.plot(alpha_grid, mean, label=label, *args, **kwargs)

for ptype in ['null', 'power']:
    if ptype == 'null':
        dissimilarity = 0.0
    else:
        dissimilarity = 0.1

    df = pd.DataFrame(list(Result.select().where(
        Result.dissimilarity==dissimilarity,
        ).dicts()))

    method = df.method.copy()
    method[method == "mannwhitneyu"] = "M-Whitney"
    method[method == "ks"] = "K-Smirnov"
    method[method == "ttest"] = "Welch"
    df['method'] = method

    ax = plt.figure(figsize=[8.4, 7.8]).add_subplot(111)
    ax.plot(np.linspace(0, 1, 10000), np.linspace(0, 1, 10000))

    k = 0
    for i, no_instances in enumerate(np.sort(np.unique(df.no_instances))):
        for j, method in enumerate(np.sort(np.unique(df.method))):
            label = "{} test with {} instances"
            label = label.format(method, no_instances)

            pvals = df
            pvals = pvals[pvals.no_instances == no_instances]
            pvals = pvals[pvals.method == method]
            pvals = pvals.pvalue

            ecdf_plot(pvals, ax, label=label, color=colors[k],
               lw=clw[j], ls=cls[i], bandwidth=ptype=='power')
            k += 1

        ax.set_xlabel("p-value")
        ax.set_ylabel("Probability")
        legend = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)

    filename = ptype+".pdf"
    with PdfPages(filename) as ps:
        ps.savefig(ax.get_figure(), bbox_inches='tight')
    plt.close(ax.get_figure())
