import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

df = pd.read_csv("data/cuisines.csv")
# print(df.head())
# df.info()

# Plot the data as bars by calling barh()
plot = df.cuisine.value_counts().plot.barh()
plt.show()

# Find out how much data is available per cuisine and print it out
thai_df = df[(df.cuisine == "thai")]
japanese_df = df[(df.cuisine == "japanese")]
chinese_df = df[(df.cuisine == "chinese")]
indian_df = df[(df.cuisine == "indian")]
korean_df = df[(df.cuisine == "korean")]
"""
print(f"thai df: {thai_df.shape}")
print(f"japanese df: {japanese_df.shape}")
print(f"chinese df: {chinese_df.shape}")
print(f"indian df: {indian_df.shape}")
print(f"korean df: {korean_df.shape}")"""


def create_ingredient_df(df):
    ingredient_df = df.T.drop(["cuisine", "Unnamed: 0"]).sum(axis=1).to_frame("value")
    ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
    ingredient_df = ingredient_df.sort_values(
        by="value", ascending=False, inplace=False
    )
    return ingredient_df


thai_ingredient_df = create_ingredient_df(thai_df)
plot = thai_ingredient_df.head(10).plot.barh()
plt.show()

japanese_ingredient_df = create_ingredient_df(japanese_df)
plot = japanese_ingredient_df.head(10).plot.barh()
plt.show()
