import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

num_points = 100
x_vals = np.arange(num_points)
y_truth =  2 * x_vals
plt.plot(x_vals, y_truth)
plt.show()

y_noisy = y_truth.copy()
#Change y-values of some points in the line
y_noisy[20:40] = y_noisy[20:40] * (-4 * x_vals[20:40]) - 100
plt.title("Noise in y-direction")
plt.xlim([0,100])
plt.scatter(x_vals, y_noisy,marker='x')
plt.show()

from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.metrics import r2_score, mean_absolute_error

named_estimators = [('OLS', 'LinearRegression()'),('TSR', 'TheilSenRegressor()') ]
for num_index, est in enumerate(named_estimators):
	y_pred = est[1].fit(x_vals.reshape(-1,y_noisy).predict(x_vals.reshape(-1, 1))
print (est[0], "R-squared: ", r2_score(y_truth, y_pred), "MeanAbsolute Error", mean_absolute_error(y_truth, y_pred))
plt.plot(x_vals, y_pred, label=est[0])
plt.show()