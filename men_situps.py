import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, model_selection

"""This dataset has multiple targets: 'It consists of three exercise (data) and three physiological (target) variables
collected from twenty middle-aged men in a fitness club'."""


X, y = datasets.load_linnerud(return_X_y=True, as_frame=False)
# (20, 3) - 20 samples of exercise types Chins, Situps and Jumps, 3 features Weight, Waist and Pulse.

print(X.shape)
print(X[3])
