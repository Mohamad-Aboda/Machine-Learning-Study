from sklearn import datasets
import numpy as np 

iris = datasets.load_iris()
iris_x = iris.data 
masking_array = np.random.binomial(1, .25, iris_x.shape).astype(bool)
iris_x[masking_array] = np.nan 
print(masking_array[:5])
print(iris_x[:5])

# handling missing values 
from sklearn.impute import SimpleImputer

impute = SimpleImputer(missing_values=np.nan, strategy='mean')
iris_x_prime = impute.fit_transform(iris_x)

print(iris_x_prime[:5])


# fill missing with -1 as a test 
iris_x[np.isnan(iris_x)] = -1 
print(iris_x[:5])

new_imputer = SimpleImputer(missing_values=-1)
iris_x_prime = new_imputer.fit_transform(iris_x)
print(iris_x_prime[:5])


# using pandas to impute missing values 
import pandas as pd 
iris_x_prime = np.where(pd.DataFrame(iris_x).isna(), -1, iris_x)
iris_x_prime[:5]