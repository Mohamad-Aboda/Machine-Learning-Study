import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import datasets 
iris = datasets.load_iris()
diabetes = datasets.load_digits()
digits = datasets.load_digits()
#print(iris.DESCR) # view handy info about the data 
print(iris.data)
print(iris.data.shape)
iris.data[0, :]
print(iris.feature_names)
print(iris.target_names)
print(iris.target)
print(iris.target.shape)
print(iris.target[0])


# use pandas 
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
print(iris_df.head())
plt.hist(iris_df['sepal length (cm)'], bins = 30)
plt.style.use('fivethirtyeight')
plt.show()

for class_number in np.unique(iris.target):
	plt.figure(1)
	plt.hist(iris_df['sepal length (cm)'].iloc[np.where(iris.target == class_number)[0]])
	plt.show()


