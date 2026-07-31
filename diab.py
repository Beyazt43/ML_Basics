import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, model_selection

"""
The built-in diabetes dataset includes 442 samples of data around diabetes, with 10 feature variables, some of which include:

age: age in years
bmi: body mass index
bp: average blood pressure
s1 tc: T-Cells (a type of white blood cells)

"""
# By convention x contains the features, and y contains the target variable which we are trying to predict

# Remember, this is supervised learning, and we need a named 'y' target.
X, y = datasets.load_diabetes(return_X_y=True)
"""print(X.shape)
print(X[0])
# what you're getting is a tuple of the features for the first sample, which is a 10-dimensional vector. (rows, columns)
# tuple elements are ordered, tuple ( 1 , 2 , 3 ) ≠ ( 3 , 2 , 1 ) , but set { 1 , 2 , 3 } = { 3 , 2 , 1 }

# The input return_X_y=True signals that X will be a data matrix, and y will be the regression target.
"""
X = X[:, 2]
X = X.reshape((-1, 1))
print(X.shape)
print(X[0])

# select a portion of this dataset to plot by selecting the 3rd column of the dataset.
# You can do this by using the : operator to select all rows, and then selecting the 3rd column (BMI) using the index (2)

# You can also reshape the data to be a 2D array - as required for plotting - by using reshape(n_rows, n_columns).
# If one of the parameter is -1, the corresponding dimension is calculated automatically.

# Time to train

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.33
)
# train_test_split(X, y, test_size=0.33): 2/3 of the data is used for training, and 1/3 is used for testing. This is a common split ratio.
# X_train and y_train will have the same number of rows just like X_test and y_test will have the same number of rows. The number of columns in X_train and X_test will be the same as the number of features in the dataset.

# Create linear regression object
model = linear_model.LinearRegression()  # creates an empty LinearRegression object, Linear regression tries to find the best-fitting straight line through your data. y = mx + b
model.fit(X_train, y_train)
# learning happens here, learns the relationship between the features and the target variable by finding the best-fitting line through the training data.

y_pred = model.predict(X_test)
# predictions are done with the test data, which the model has never seen before, will help us evaluate

# Here comes the modeling:
plt.scatter(X_test, y_test, color="black")
plt.plot(X_test, y_pred, color="blue", linewidth=3)
plt.xlabel("Scaled BMIs")
plt.ylabel("Disease Progression")
plt.title("A Graph Plot Showing Diabetes Progression Against BMI")
plt.show()
