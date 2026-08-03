from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

pumpkins = pd.read_csv("data/US-pumpkins.csv")


columns_to_select = ["City Name", "Package", "Variety", "Origin", "Item Size", "Color"]
pumpkins = pumpkins.loc[:, columns_to_select]

# dropping null values
pumpkins.dropna(inplace=True)

"""palette = {
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
plt.show()"""


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
"""
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

plt.show()"""
# Linear regression predicts a number. Logistic regression predicts a class.
# now we can use the encoded_pumpkins dataframe to train a logistic regression model to predict the color of a pumpkin based on its features

X = encoded_pumpkins[encoded_pumpkins.columns.difference(["Color"])]
y = encoded_pumpkins["Color"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(classification_report(y_test, predictions))
print("Predicted labels: ", predictions)
print("F1-score: ", f1_score(y_test, predictions))

# a confusion matrix for better comprehension of the model's performance
# remember that it is       0	1
#                       0	TN	FP
#                       1	FN	TP
confusion_matrix(y_test, predictions)
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

# Precision = tp / (tp + fp)
# Recall = tp / (tp + fn)
# F1-score = (2 * precision * recall)/(precision + recall)
# Accuracy: (TP + TN)/(TP + TN + FP + FN)

# How to do the ROC curve
y_scores = model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:, 1])

fig = plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], "k--")
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

#  use Scikit-learn's roc_auc_score API to compute the actual 'Area Under the Curve' (AUC)
auc = roc_auc_score(y_test, y_scores[:, 1])
print(auc)
