import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# How to save your Scikit-learn model as a file that can be used to make predictions within a web application.
# Once the model is saved, you'll learn how to use it in a web app built in Flask.

ufos = pd.read_csv("data/ufos.csv")
ufos.head()

ufos = pd.DataFrame(
    {
        "Seconds": ufos["duration (seconds)"],
        "Country": ufos["country"],
        "Latitude": ufos["latitude"],
        "Longitude": ufos["longitude"],
    }
)

# Removes rows with missing values (only those four columns) directly from the original DataFrame without returning a new one
# This means that the changes are made to the existing DataFrame itself
ufos.dropna(inplace=True)
# Only keep rows where the 'Seconds' column is between 1 and 60 (inclusive)
ufos = ufos[(ufos["Seconds"] >= 1) & (ufos["Seconds"] <= 60)]

# Encoding the countries into numerical values using LabelEncoder
# This is necessary because machine learning models typically require numerical input
ufos["Country"] = LabelEncoder().fit_transform(ufos["Country"])
# print(ufos.head())

# Now we get to training
Selected_features = ["Seconds", "Latitude", "Longitude"]
X = ufos[Selected_features]
y = ufos["Country"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# A dataset with 1 million rows might converge in 50 iterations.
# A dataset with 500 rows might need 1000 iterations.
# It depends on how difficult the optimization problem is, not on the number of samples.
# increasing the iteration limit from the default 100 to 1000 increased my accuracy from 0.9601 to 9702
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(classification_report(y_test, predictions))
print("Predicted labels: ", predictions)
print("Accuracy: ", accuracy_score(y_test, predictions))

# how many iterations it took to converge
print(model.n_iter_)

# Pickle the model
model_filename = "ufo-model.pkl"
pickle.dump(model, open(model_filename, "wb"))

model = pickle.load(open("./ufo-model.pkl", "rb"))
print(model.predict([[50, 44, -12]]))
