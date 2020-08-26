from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, 
		test_size=0.25, random_state=7
	)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, 
		test_size=0.25, random_state=7
	)	

svc_clf = SVC(kernel='linear', random_state=7)
svc_clf.fit(X_train2, y_train2)

lr_clf = LogisticRegression(random_state=7)
lr_clf.fit(X_train2, y_train2)

svc_pred = svc_clf.predict(X_test2)
lr_pred = lr_clf.predict(X_test2)

print(f"Accuracy of SVC: {accuracy_score(y_test2, svc_pred)}")
print(f"Accuracy of LR: {accuracy_score(y_test2, lr_pred)}")
# in this model svc performes better 
y_original_pred = svc_clf.predict(X_test)
print(f"Accuracy of SVC on originl Test set: {accuracy_score(y_test, y_original_pred)}")
# accuracu now is low 


from sklearn.model_selection import cross_val_score
svc_scores = cross_val_score(svc_clf, X_train, y_train, cv=4)
print(svc_scores)

print(f"Average SVC scores: {svc_scores.mean()}")
print(f'Standard Deviation of SVC scores: {svc_scores.std()}')


lr_scores = cross_val_score(lr_clf, X_train, y_train, cv = 4)
print(f'Average LogisticRegression scores: {lr_scores.mean()}')
print(f'Standard Deviation of LogisticRegression: {lr_scores.std()}')