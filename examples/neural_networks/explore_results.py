import numpy as np
import pandas as pd
from scipy import stats
from db_structure import Result, db
import pickle
import itertools
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Load data
df = pd.DataFrame(list(Result.select().dicts()))
del df['id']
df.drop_duplicates(['hidden_size', 'seed', 'dropout',
    'batch_normalization'], inplace=True)
del df['seed']

# Print summary
def mpse(data):
    mean = data.mean()
    std_error = np.std(data) / np.sqrt(len(data))
    return "{0:.3f} ({1:.3f})".format(mean, std_error)
to_group = ['hidden_size', 'dropout', 'batch_normalization']
gdf = df.groupby(to_group).agg(mpse)
count = df.groupby(to_group).count().iloc[:,-1]
gdf['nsim'] = count
print(gdf)
#print(gdf.to_latex())

# Plot results
cls = ["-", ":", "-.", "--"]
clw = [1.0, 2.0, 1.5, 3.0, 0.5, 4.0]
clws = list(itertools.product(clw, cls))

for batch_n in [True, False]:
    for dropout in [True, False]:
        fig, ax = plt.subplots(figsize=(6,3))
        for ptype in ['train', 'test']:
            label = "MSE on {} dataset".format(ptype)
            mean = []
            stderror = []
            hidden_sizes = np.sort(np.unique(df.hidden_size))
            for hidden_size in hidden_sizes:
                vals = df[df.hidden_size == hidden_size]
                vals = vals[vals.dropout == dropout]
                vals = vals[vals.batch_normalization == batch_n]
                vals = vals['mse_'+ptype]
                mean.append(vals.mean())
                stderror.append(np.std(vals) / np.sqrt(len(vals)))
            mean = np.array(mean)
            stderror = np.array(stderror)

            multip = stats.norm.ppf(1-0.05)
            ax.fill_between(hidden_sizes, mean+stderror*multip,
                    mean-stderror*multip, alpha=.5)
            ax.plot(hidden_sizes, mean, label=label)

        ax.set_xlabel("Network input value")
        ax.set_ylabel("MSE")
        title = "{} droupout and {} batch normalization"
        title = title.format(
            "With" if dropout else "Without",
            "with" if batch_n else "without",
        )
        ax.set_title(title)
        #ax.set_ylim(1.06, 1.23)
        legend = ax.legend(bbox_to_anchor=(0., -0.32, 1., .102), loc=3,
                   ncol=3, mode="expand", borderaxespad=0.)

        filename = "sim_study_nn_dropout_{}_batch_normalization_{}.pdf"
        filename = filename.format(dropout, batch_n)
        with PdfPages(filename) as ps:
            ps.savefig(fig, bbox_inches='tight')
        plt.close(ax.get_figure())


