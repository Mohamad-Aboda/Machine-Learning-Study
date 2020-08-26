import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import datasets 
iris = datasets.load_iris()

X = iris.data[:, :2]
y =  iris.target
# split the data 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
		test_size = 0.25, random_state=1
	)	
from sklearn.svm import SVC
clf = SVC(kernel='linear', random_state=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# i want to know the score
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
