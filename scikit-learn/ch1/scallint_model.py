import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn import model_selection
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score

def build_model(n_components):
	X = df[features]
	y = df[target]

	fa_obj = FactorAnalysis(n_components = n_components, 
		random_state = 10, max_iter = 10000
		)
	x_new = fa_obj.fit_transform(X, y)
	X = pd.DataFrame(n_new)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
	model = LogisticRegression(solver='liblinear', max_iter = 1000)
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	pred_results = pd.DataFrame({'y_test': y_test, 'y_pred':y_pred})
	acc = accuracy_score(y_test, y_pred)
	prec = precision_score(y_test, y_pred)
	recall = recall_score(y_test, y_pred)

	return {'fact_a_obj':fa_obj, 
		'transform_x':X, 'accuracy':acc, 
		'precision': pred, 'recall': recall
	}