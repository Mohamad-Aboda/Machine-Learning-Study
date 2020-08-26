import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import math
# plt.plot(np.arange(10), np.arange(10))
# plt.show()

# plt.plot(np.arange(10), np.exp(np.arange(10)))
# plt.show()
# steps 1) subplt  2) plt.(...)  3) plt.show()

plt.subplot(224)
plt.plot(np.arange(10), np.exp(np.arange(10)))
plt.subplot(221)
plt.scatter(np.arange(10), np.exp(np.arange(10)))
plt.subplot(222)
plt.scatter(np.arange(10), np.exp(np.arange(10)))
plt.subplot(223)
plt.scatter(np.arange(10), np.exp(np.arange(10)))
plt.show()

# plot real data
from sklearn import datasets
iris = datasets.load_iris()
data = iris.data
target = iris.target

plt.figure(figsize = (12, 5))
plt.subplot(121)
plt.scatter(data[:, 0], data[:, 1], c= target)


plt.subplot(122)
plt.scatter(data[:, 2], data[:, 3], c=target)
plt.show()