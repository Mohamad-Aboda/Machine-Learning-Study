import numpy as np 
a = np.arange(10)
print(a)
print(a.shape)
# the number of dimination 
print(a.ndim)
a = a.reshape((5, 2))
print(a)
print(a.shape)
print(a.ndim)
# this means added one to each element of the array
a = a + 1
print(a)

ar2 = np.arange(10)
ar2 =  ar2 * ar2
print(ar2)
print(ar2.shape)
print(ar2.ndim)
ar2 = ar2.reshape((5, 2))
print(ar2)
print(ar2.ndim)
print(ar2.shape)
arr3 = ar2 + a
print(arr3)
arr3 = arr3 - a
print(arr3)
zero = np.zeros((5, 2))
print(zero)
one = np.ones((5, 2))
print(one)
one = np.ones((5 ,2), dtype = np.int)
print(one)
# empty means i don't know what his value is?? 
unknown = np.empty((2, 3), dtype = np.float)
print(unknown)
print(a[0][0])
print(unknown[0][0])
print(a[0, :])
print(a)
print(a[:, 0])
print(a[2:4, :])
print(a > 5)
print(a[a > 5])
print(a[a < 0]) # [] cause we start from zero 
print(a[a > -1].shape)  # 10 all numbers is > -1
print(a.sum())
print(a.sum(axis = 1))
print(a)
print(a.sum(axis = 0))
print(a.mean()) # the mean for all numbers 
print(a.mean(axis = 0)) # the mean for all sum for each col
print(a.mean(axis = 1)) # the mean for all sum for each row 



import numpy as np 
arr3 = np.array([np.nan, 0, 1, 2, 3, np.nan])
print(np.isnan(arr3))
print(arr3[~np.isnan(arr3)])
print(arr3[np.isnan(arr3)])
arr3[np.isnan(arr3)] = 0
# scikit learn accepts only 2D numpy arrays of real numbers with no missing values 
