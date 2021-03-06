import numpy as np
import pandas as pd
from db_structure import Result, db

df = pd.DataFrame(list(Result.select().dicts()))
del df['id']
del df['elapsed_time']

df.groupby(['no_instances', 'method']).mean()

def mpse(data):
    mean = data.mean()
    std_error = np.std(data) / np.sqrt(len(data))
    return "{0:.3f} ({1:.3f})".format(mean, std_error)

gdf = df.groupby(['no_instances', 'method']).agg(mpse)

count = df.groupby(['no_instances', 'method']).count().iloc[:,-1]
gdf['no simulations'] = count

print(gdf)
print(gdf.to_latex(multirow=True))
