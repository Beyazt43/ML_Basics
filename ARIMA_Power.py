import datetime as dt
import math
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels
from IPython.display import Image
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

energy = pd.read_csv("data/energy.csv", parse_dates=["timestamp"])
# indexes were 0,1,2.. but with this line "timestamps" became the index
energy.set_index("timestamp", inplace=True)
# pd.date_range() generates a sequence of timestamps.
# freq = h means the timestamps will in one hour intervals
energy = energy.reindex(
    pd.date_range(
        energy.index.min(),
        energy.index.max(),
        freq="h",
    )
)

pd.options.display.float_format = "{:,.2f}".format
np.set_printoptions(precision=2)
warnings.filterwarnings("ignore")  # specify to ignore warning messages

# print(energy.head(10))

"""energy.plot(y="load", subplots=True, figsize=(15, 8), fontsize=12)
plt.xlabel("timestamp", fontsize=12)
plt.ylabel("load", fontsize=12)
plt.show()"""

# To training
train_start_dt = "2014-11-01 00:00:00"
test_start_dt = "2014-12-30 00:00:00"

"""energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][
    ["load"]
].rename(columns={"load": "train"}).join(
    energy[test_start_dt:][["load"]].rename(columns={"load": "test"}), how="outer"
).plot(y=["train", "test"], figsize=(15, 8), fontsize=12)
plt.xlabel("timestamp", fontsize=12)
plt.ylabel("load", fontsize=12)
plt.show()"""

train = energy.copy()[
    (energy.index >= train_start_dt) & (energy.index < test_start_dt)
][["load"]]
test = energy.copy()[energy.index >= test_start_dt][["load"]]

print("Training data shape: ", train.shape)
print("Test data shape: ", test.shape)

# scaling them to be in (0,1) range, this is a common practice
scaler = MinMaxScaler()
train["load"] = scaler.fit_transform(train)
train.head(10)

"""energy[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][
    ["load"]
].rename(columns={"load": "original load"}).plot.hist(bins=100, fontsize=12)
train.rename(columns={"load": "scaled load"}).plot.hist(bins=100, fontsize=12)
plt.show()"""

test["load"] = scaler.transform(test)
test.head()

"""SARIMAX() and passing in the model parameters: p, d, and q parameters, and P, D, and Q parameters.What are all these parameters for?
In an ARIMA model there are 3 parameters that are used to help model the major aspects of a time series: seasonality, trend, and noise.
These parameters are:

p: the parameter associated with the auto-regressive aspect of the model, which incorporates past values.
d: the parameter associated with the integrated part of the model, which affects the amount of differencing to apply to a time series.
q: the parameter associated with the moving-average part of the model.

Note: If your data has a seasonal aspect - which this one does - , we use a seasonal ARIMA model (SARIMA).
In that case you need to use another set of parameters: P, D, and Q which describe the same associations as p, d, and q,
but correspond to the seasonal components of the model."""

# Specify the number of steps to forecast ahead
HORIZON = 3
print("Forecasting horizon:", HORIZON, "hours")
# (p, d, q)
order = (4, 1, 0)
seasonal_order = (1, 1, 0, 24)

model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
results = model.fit()

print(results.summary())

# evaluating the model
test_shifted = test.copy()

for t in range(1, HORIZON + 1):
    test_shifted["load+" + str(t)] = test_shifted["load"].shift(-t, freq="h")

test_shifted = test_shifted.dropna(how="any")
test_shifted.head(5)


training_window = 720  # dedicate 30 days (720 hours) for training

train_ts = train["load"]
test_ts = test_shifted

history = [x for x in train_ts]
history = history[(-training_window):]

predictions = list()

order = (2, 1, 0)
seasonal_order = (1, 1, 0, 24)

print(test.shape)
print(test_ts.shape)
print(len(test))
print(len(test_ts))


for t in range(test_ts.shape[0]):
    model = SARIMAX(endog=history, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    yhat = model_fit.forecast(steps=HORIZON)
    predictions.append(yhat)
    obs = list(test_ts.iloc[t])
    # move the training window
    history.append(obs[0])
    history.pop(0)
    print(test_ts.index[t])
    print(t + 1, ": predicted =", yhat, "expected =", obs)

print(len(predictions))
# comparing the predictions to the actual load

eval_df = pd.DataFrame(
    predictions, columns=["t+" + str(t) for t in range(1, HORIZON + 1)]
)
eval_df["timestamp"] = test.index[0 : len(test.index) - HORIZON]
eval_df = pd.melt(eval_df, id_vars="timestamp", value_name="prediction", var_name="h")
eval_df["actual"] = np.array(np.transpose(test_ts.iloc[:, 1:])).ravel()
eval_df[["prediction", "actual"]] = scaler.inverse_transform(
    eval_df[["prediction", "actual"]]
)
print(eval_df.head())

if HORIZON > 1:
    eval_df["APE"] = (eval_df["prediction"] - eval_df["actual"]).abs() / eval_df[
        "actual"
    ]
    print(eval_df.groupby("h")["APE"].mean())

mape_value = (
    np.mean(np.abs((eval_df["actual"] - eval_df["prediction"]) / eval_df["actual"]))
    * 100
)

print(
    "Multi-step forecast MAPE: ",
    mape(eval_df["prediction"], eval_df["actual"]) * 100,
    "%",
)

if HORIZON == 1:
    ## Plotting single step forecast
    eval_df.plot(
        x="timestamp", y=["actual", "prediction"], style=["r", "b"], figsize=(15, 8)
    )

else:
    ## Plotting multi step forecast
    plot_df = eval_df[(eval_df.h == "t+1")][["timestamp", "actual"]]
    for t in range(1, HORIZON + 1):
        plot_df["t+" + str(t)] = eval_df[(eval_df.h == "t+" + str(t))][
            "prediction"
        ].values

    fig = plt.figure(figsize=(15, 8))
    ax = plt.plot(plot_df["timestamp"], plot_df["actual"], color="red", linewidth=4.0)
    ax = fig.add_subplot(111)
    for t in range(1, HORIZON + 1):
        x = plot_df["timestamp"][(t - 1) :]
        y = plot_df["t+" + str(t)][0 : len(x)]
        ax.plot(
            x, y, color="blue", linewidth=4 * math.pow(0.9, t), alpha=math.pow(0.8, t)
        )

    ax.legend(loc="best")

plt.xlabel("timestamp", fontsize=12)
plt.ylabel("load", fontsize=12)
plt.show()
