import pandas as pd
import numpy as np
import math
import sys
import random

fileName = "labels.csv"
data = pd.read_csv('labels.csv', delimiter=';',header = None)
X = data.as_matrix()
print(X)
X = np.repeat(X,20,axis=0)

np.savetxt("train_labels.csv", X, delimiter=",", fmt="%s")
