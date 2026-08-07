import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

cuisines_df = pd.read_csv("data/cleaned_cuisines.csv")

cuisines_label_df = cuisines_df["cuisine"]
# print(cuisines_label_df.head())
# we split it into 2 dataframes, one for the features and one for the labels

# Drop that Unnamed: 0 column and the cuisine column, calling drop()
cuisines_feature_df = cuisines_df.drop(["Unnamed: 0", "cuisine"], axis=1)
# print(cuisines_feature_df.head())

# time to split the data
X_train, X_test, y_train, y_test = train_test_split(
    cuisines_feature_df, cuisines_label_df, test_size=0.3, random_state=42
)
# we're going to need to make the result reproducible

# Since you are using the multiclass case, you need to choose what scheme to use and what solver to set
# Use LogisticRegression with a multiclass setting and the liblinear solver to train.
# turns out in the new version of sklearn you don't have to specify the multiclass setting
lr = OneVsRestClassifier(LogisticRegression(solver="liblinear", max_iter=1000))
model = lr.fit(X_train, np.ravel(y_train))

accuracy = model.score(X_test, y_test)
print("Accuracy is {}".format(accuracy))

# testing an individual line fo ingredients
print(f"ingredients: {X_test.iloc[50][X_test.iloc[50] != 0].keys()}")
print(f"cuisine: {y_test.iloc[50]}")

test = X_test.iloc[50].values.reshape(-1, 1).T
proba = model.predict_proba(test)
classes = model.classes_
resultdf = pd.DataFrame(data=proba, columns=classes)

topPrediction = resultdf.T.sort_values(by=[0], ascending=[False])
print(topPrediction.head())

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
