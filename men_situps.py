import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, model_selection
from sklearn.linear_model import LinearRegression

"""This dataset has multiple targets: 'It consists of three exercise (data) and three physiological (target) variables
collected from twenty middle-aged men in a fitness club'."""


X, y = datasets.load_linnerud(return_X_y=True, as_frame=False)
# (20, 3) - 20 samples of exercise types Chins, Situps and Jumps, 3 features Weight, Waist and Pulse.

"""print(X.shape)
print(X[3])"""

# choosing the first feature (Chins) to plot against the target variables (Weight, Waist and Pulse)
X = X[:, 0]
X = X.reshape((-1, 1))

# you can use print(y[:, 0].shape) to see the shape of the first target variable (Weight).


# print(y.shape)

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.33
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

order = np.argsort(X_test[:, 0])
# It does not sort the data itself. Instead, it returns the indices that would sort the data.
# When you use the same order on multiple arrays that represent the same observations, they all get rearranged in the same way,
# so the correspondence between features and targets is preserved.

plt.scatter(X_test[:, 0], y_test[:, 0], color="black", label="Actual")

plt.plot(X_test[order, 0], y_pred[order, 0], color="red", label="Predicted")

plt.xlabel("Chins")
plt.ylabel("Weight")
plt.legend()
plt.show()
