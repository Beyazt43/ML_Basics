import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

cuisines_df = pd.read_csv("data/cleaned_cuisines.csv")
cuisines_features_df = cuisines_df.drop(["Unnamed: 0", "cuisine"], axis=1)
cuisines_label_df = cuisines_df["cuisine"]

X_train, X_test, y_train, y_test = train_test_split(
    cuisines_features_df, cuisines_label_df, test_size=0.3
)


C = 10
# Create different classifiers.
classifiers = {
    "Linear SVC": SVC(kernel="linear", C=C, probability=True, random_state=0)
}

n_classifiers = len(classifiers)

for index, (name, classifier) in enumerate(classifiers.items()):
    classifier.fit(X_train, np.ravel(y_train))

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
    print(classification_report(y_test, y_pred))
