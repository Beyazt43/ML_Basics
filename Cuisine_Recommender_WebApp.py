import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
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

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# This dataset has 380 ingredients listed, so you need to notate that number in FloatTensorType
initial_type = [("float_input", FloatTensorType([None, 380]))]
options = {id(model): {"nocl": True, "zipmap": False}}

onx = convert_sklearn(model, initial_types=initial_type, options=options)
with open("./model.onnx", "wb") as f:
    f.write(onx.SerializeToString())
