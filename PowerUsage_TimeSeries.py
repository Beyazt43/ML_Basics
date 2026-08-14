import os

import matplotlib.pyplot as plt
import pandas as pd

# %matplotlib inline

energy = pd.read_csv("data/energy.csv", parse_dates=["timestamp"])

energy.set_index("timestamp", inplace=True)
energy = energy.reindex(
    pd.date_range(
        energy.index.min(),
        energy.index.max(),
        freq="h",
    )
)
# print(energy.head())

"""energy.plot(y="load", subplots=True, figsize=(15, 8), fontsize=12)
plt.xlabel("timestamp", fontsize=12)
plt.ylabel("load", fontsize=12)
plt.show()"""

# plot the first week of July 2014, by providing it as input to the energy in [from date]: [to date] pattern:
energy["2014-07-01":"2014-07-07"].plot(
    y="load", subplots=True, figsize=(15, 8), fontsize=12
)
plt.xlabel("timestamp", fontsize=12)
plt.ylabel("load", fontsize=12)
plt.show()
