import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, random_state = 0)

knn_3_clf = KNeighborsClassifier(n_neighbors=3)
knn_5_clf = KNeighborsClassifier(n_neighbors=5)

knn_3_scores = cross_val_score(knn_3_clf, X_train, y_train, cv=10)
knn_5_scores = cross_val_score(knn_5_clf, X_train, y_train, cv=10)

print(f"knn_3 mean scores : {knn_3_scores.mean()}")
print(f"knn_3 std scores: {knn_3_scores.std()}")

print(f"knn_5 mean scores : {knn_5_scores.mean()}")
print(f"knn_5 std scores: {knn_5_scores.std()}")

# we can run a loop to choose the number k 
all_scores = []
for n_neighbors in range(3, 9, 1):
	knn_clf = KNeighborsClassifier(n_neighbors = n_neighbors)
	try_score = cross_val_score(knn_clf, X_train, y_train, cv=10).mean()
	all_scores.append((n_neighbors, try_score))
print(all_scores)
sorted(all_scores, key = lambda x:x[1], reverse = True)

print(all_scores)
#  4 is a good choice 

