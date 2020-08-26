import numpy as np 
from sklearn import datasets
from sklearn.datasets import  load_boston
from sklearn import preprocessing
# create boston datasets 
boston = load_boston()
X, y = boston.data, boston.target

print(X[:, :3].mean(axis = 0))

print(X[:, :3].std(axis = 0))

# we should scale data 
X_2 = preprocessing.scale(X[:, :3])
print(X_2.mean(axis = 0))
print(X_2.std(axis=0))

import pandas as pd
import matplotlib.pyplot as plt  
