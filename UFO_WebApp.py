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

model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(classification_report(y_test, predictions))
print("Predicted labels: ", predictions)
print("Accuracy: ", accuracy_score(y_test, predictions))
