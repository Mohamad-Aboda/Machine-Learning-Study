# import numpy as np 
# arr = np.array([[1, 2], [3, 3], [32, 4]])
# print(arr)
# print(arr.shape)
# arr = arr.reshape(-1, 1)
# print(arr.shape)

import numpy as np 
from sklearn import  datasets

iris = datasets.load_iris()
X = iris.data 
y = iris.target 

from sklearn import  preprocessing
cat_encoder = preprocessing.OneHotEncoder()
cat_encoder = cat_encoder.fit_transform(y.reshape(-1, 1)).toarray()[:5]
print(cat_encoder)