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
from sklearn.svm import SVC

cuisines_df = pd.read_csv("data/cleaned_cuisines.csv")
