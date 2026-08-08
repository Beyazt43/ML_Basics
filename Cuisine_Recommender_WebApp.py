import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC

data = pd.read_csv("data/cleaned_cuisines.csv")
# print(data.head())

# Remove the first two unnecessary columns and save the remaining data as 'X':
X = data.iloc[:, 2:]
# print(X.head())

# Save the labels as 'y':
y = data[["cuisine"]]
# print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = SVC(kernel="linear", C=10, probability=True, random_state=0)
model.fit(X_train, y_train.values.ravel())
