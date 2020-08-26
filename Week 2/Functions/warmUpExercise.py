import os
import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


def warmUpExercise():
	A = np.eye(5)
	return A

def plotData(x, y):
	fig = pyplot.figure()
	pyplot.plot(x, y, 'ro', ms = 10, mec = 'k')
	pyplot.xlabel('Population of City in 10,000s')
	pyplot.ylabel('Profit in $10,000')

# now you can run the functionn 