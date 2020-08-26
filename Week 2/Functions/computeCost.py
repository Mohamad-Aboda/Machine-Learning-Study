

# h is the linear function 
# h = theta0 + theta1 * x   
# h = (theta)T.X  >> the same as dot product 
h = np.dot(X, theta)
# j = h - y[i] 
J = (1/(2 * m)) * np.sum(np.square(np.dot(X, theta) - y))
    
