import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import auc
iris = datasets.load_iris()
X = iris.data[:, :2] #load the iris data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
random_state=1)
#train the model
clf = LogisticRegression(random_state = 1)
clf.fit(X_train, y_train)
#predict with Logistic Regression
y_pred = clf.predict(X_test)
#examine the model accuracy
print(accuracy_score(y_test,y_pred))

from sklearn.metrics import mean_squared_error