from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

pumpkins = pd.read_csv("data/US-pumpkins.csv")


columns_to_select = ["City Name", "Package", "Variety", "Origin", "Item Size", "Color"]
pumpkins = pumpkins.loc[:, columns_to_select]

# dropping null values
pumpkins.dropna(inplace=True)

palette = {
    "ORANGE": "orange",
    "WHITE": "wheat",
}

sns.catplot(
    data=pumpkins,
    y="Variety",
    hue="Color",
    kind="count",
    palette=palette,
)
plt.show()


# ordinal encoding
item_size_categories = [["sml", "med", "med-lge", "lge", "xlge", "jbo", "exjbo"]]
ordinal_features = ["Item Size"]
ordinal_encoder = OrdinalEncoder(categories=item_size_categories)

# categorical encoding
categorical_features = ["City Name", "Package", "Variety", "Origin"]
categorical_encoder = OneHotEncoder(sparse_output=False)

# ColumnTransformer is used to combine multiple encoders into a single step and apply them to the appropriate columns
ct = ColumnTransformer(
    transformers=[
        ("ord", ordinal_encoder, ordinal_features),
        ("cat", categorical_encoder, categorical_features),
    ]
)

ct.set_output(transform="pandas")
encoded_features = ct.fit_transform(pumpkins)

label_encoder = LabelEncoder()
encoded_label = label_encoder.fit_transform(pumpkins["Color"])

# Once we have encoded the features and the label, we can merge them into a new dataframe encoded_pumpkins
encoded_pumpkins = encoded_features.assign(Color=encoded_label)

# plotting o visualize the relationships between Item Size, Variety and Color in a categorical plot.
# To better plot the data we'll be using the encoded Item Size column and the unencoded Variety column

palette = {
    "ORANGE": "orange",
    "WHITE": "wheat",
}
pumpkins["Item Size"] = encoded_pumpkins["ord__Item Size"]

g = sns.catplot(
    data=pumpkins,
    x="Item Size",
    y="Color",
    row="Variety",
    kind="box",
    orient="h",
    sharex=False,
    margin_titles=True,
    height=1.8,
    aspect=4,
    palette=palette,
)
g.set(xlabel="Item Size", ylabel="").set(xlim=(0, 6))
g.set_titles(row_template="{row_name}")

plt.show()

palette = {0: "orange", 1: "wheat"}
sns.swarmplot(
    x="Color",
    y="ord__Item Size",
    data=encoded_pumpkins,
    palette=palette,
    hue="Color",
    legend=False,
)

plt.show()
