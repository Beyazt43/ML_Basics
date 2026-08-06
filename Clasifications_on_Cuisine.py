import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

df = pd.read_csv("data/cuisines.csv")
# print(df.head())
# df.info()

# Plot the data as bars by calling barh()
# plot = df.cuisine.value_counts().plot.barh()
# plt.show()

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

chinese_ingredient_df = create_ingredient_df(chinese_df)
chinese_ingredient_df.head(10).plot.barh()

indian_ingredient_df = create_ingredient_df(indian_df)
indian_ingredient_df.head(10).plot.barh()

korean_ingredient_df = create_ingredient_df(korean_df)
korean_ingredient_df.head(10).plot.barh()

feature_df = df.drop(["cuisine", "Unnamed: 0", "rice", "garlic", "ginger"], axis=1)
labels_df = df.cuisine  # .unique()
feature_df.head()

# use SMOTE - "Synthetic Minority Over-sampling Technique" - to balance the dataset
oversample = SMOTE()
transformed_feature_df, transformed_label_df = oversample.fit_resample(
    feature_df, labels_df
)

print(f"new label count: {transformed_label_df.value_counts()}")
print(f"old label count: {df.cuisine.value_counts()}")

# saving my balanced data, including labels and features, into a new dataframe that can be exported into a file
transformed_df = pd.concat(
    [transformed_label_df, transformed_feature_df], axis=1, join="outer"
)

transformed_df.head()
transformed_df.info()
transformed_df.to_csv("data/cleaned_cuisines.csv")
