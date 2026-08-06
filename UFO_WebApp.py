import numpy as np
import pandas as pd

# How to save your Scikit-learn model as a file that can be used to make predictions within a web application.
# Once the model is saved, you'll learn how to use it in a web app built in Flask.

ufos = pd.read_csv("data/ufos.csv")
print(ufos.head())
