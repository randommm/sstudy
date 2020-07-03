import numpy as np
import pandas as pd
import pickle
from scipy import stats

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from db_structure import Result, db

df = pd.DataFrame(list(Result.select()
.dicts()))

del df["id"]
del df["elapsed_time"]

method = df.method.copy()
method[method == "vaecompare_median"] = "vaecompare"
method[method == "mannwhitneyu"] = "M-Whitney"
method[method == "ks"] = "K-Smirnov"
method[method == "ttest"] = "Welch"
df['method'] = method

df.rename(columns={"no_instances": "n instances"}, inplace=True)

to_group = ['dissimilarity', 'method', 'n instances']

df["e1_power_1"] = df['pvalue'] <= 0.01
df["e1_power_5"] = df['pvalue'] <= 0.05
df["e1_power_10"] = df['pvalue'] <= 0.10
count = df.groupby(to_group).count().iloc[:,0]

def mpse(data):
    mean = data.mean()
    std_error = np.std(data) / np.sqrt(len(data))
    return "{0:.3f} ({1:.3f})".format(mean, std_error)

grouped = df.groupby(to_group).agg(mpse)
grouped["count"] = count

grouped.rename(columns={"count": "n sim"}, inplace=True)
grouped.rename(columns={"pvalue": "avg p-value"}, inplace=True)

grouped1 = grouped[[x[0] == 0.0 for x in grouped.index]]
grouped2 = grouped[[x[0] == 0.1 for x in grouped.index]]

grouped1.rename(columns={"e1_power_1": "Error (α=1%)"}, inplace=True)
grouped1.rename(columns={"e1_power_5": "Error (α=5%)"}, inplace=True)
grouped1.rename(columns={"e1_power_10": "Error (α=10%)"}, inplace=True)
grouped2.rename(columns={"e1_power_1": "Power (α=1%)"}, inplace=True)
grouped2.rename(columns={"e1_power_5": "Power (α=5%)"}, inplace=True)
grouped2.rename(columns={"e1_power_10": "Power (α=10%)"}, inplace=True)

print(grouped1)
print(grouped2)

grouped1.reset_index(inplace=True)
grouped2.reset_index(inplace=True)

del grouped1["dissimilarity"]
del grouped1["n sim"]
del grouped1["Error (α=10%)"]
del grouped2["dissimilarity"]
del grouped2["n sim"]
del grouped2["Power (α=10%)"]

print(grouped1.to_latex(index=False))
print(grouped2.to_latex(index=False))
