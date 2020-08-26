import numpy as np 
from sklearn.datasets import load_boston

boston = load_boston()
X, y = boston.data, boston.target.reshape(-1, 1)

from sklearn import preprocessing
new_target = preprocessing.binarize(y, threshold=boston.target.mean())
print(new_target[:5])

print(y[:5] > y.mean())
# using Binarizer class 
binar = preprocessing.Binarizer(y.mean())
new_target = binar.fit_transform(y)
new_target[:5]

# sparse matrix
from scipy.sparse import coo
spar = coo.coo_matrix(np.random.binomial(1, .25, 100))
# print(preprocessing.binarize(spar, threshold=-1)) # error 
# binarization makes features takes 0 or 1 depends on the thresholde value 
