import numpy as np 
import pandas as pd 
from  sklearn import datasets

iris = datasets.load_iris()
X = iris.data[iris.target < 2]
y = iris.target[iris.target < 2]

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score

X_train, X_test , y_train, y_test = train_test_split(X, y, stratify=y, random_state=7)

from sklearn.svm import SVC 
from sklearn.model_selection import cross_val_score

svc_clf = SVC(kernel='linear').fit(X_train, y_train)
svc_scores = cross_val_score(svc_clf, X_train, y_train, cv=4)
print(svc_scores)
print(svc_scores.mean())


from sklearn.svm import SVR
svr_clf = SVR(kernel='linear').fit(X_train, y_train)
from sklearn.metrics import make_scorer

def for_scorer(y_test, orig_y_pred):
	y_pred = np.rint(orig_y_pred).astype(np.int)
	return accuracy_score(y_test, y_pred)

svr_class_scorer = make_scorer(for_scorer, greater_is_better = True)
cross_val_score(svr_clf, X_train, y_train, cv=10).mean()
print(svr_scorer.mean())