import numpy as np
import pandas as pd
import os
# Loading the data & use cols as columns name 

cols = ['population ', 'profit']
data = pd.read_csv('ex1data1.txt', sep=",", header =  None)
data.columns = cols


X = data.iloc[:, 0]
y = data.iloc[:, 1]
m =  y.size  # the number of trainint example 
