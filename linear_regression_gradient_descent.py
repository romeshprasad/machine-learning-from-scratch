import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

#loading a file
dt = genfromtxt("D3.csv", delimiter=",")
X = dt[:, 0:3] #splitting X and y
y= dt[:, 3]
print(y[0:10])
print(y.shape) #looking at the shape of y as we have to compute the error using
               # this y

# gradient descent algorithm for multiple variables
def multiple_gradient_algorithm(X,y):
    learning_rate = 0.1
    iterations = 2000
    cost = []
    m = X.shape[0]  # number of samples
    vector_x = np.c_[np.ones((len(X), 1)), X]
    #print(vector_x)
    number_of_features = vector_x.shape[1]  # 1 means columns
    print(number_of_features)
    theta = np.random.randn(number_of_features, 1)
    theta = np.ones(number_of_features)
    print(theta)
    print(theta.shape)
    m = vector_x.shape[0]  # 0 means rows
    y_pred = np.dot(vector_x, theta) #initialize y_pred
    y_pred = np.squeeze(y_pred)
    print(y_pred.shape)

    for i in range(iterations):
        theta[0] = theta[0] - 1/m * learning_rate * sum(y_pred-y)
        for j in range(1, number_of_features):
            gradient = 1/m * sum((y_pred-y) * vector_x[:,j])
            theta[j] = theta[j] - learning_rate * 1/m * gradient #output of this loop is theta
        y_pred = np.dot(vector_x, theta)
        cost_function = (1 /2* m) * sum([val ** 2 for val in (y_pred - y)])
        cost.append(cost_function)
        #print(theta, cost_function, i)
        print("theta {} , cost {}, iterations {} ".format(theta, cost_function, i))

    return theta, cost_function, cost
    #print(theta,cost_function)


theta, cost_function, cost = multiple_gradient_algorithm(X,y)
print(theta)
print(cost)
print(cost_function)

#plt.plot(x)
plt.plot(cost, color="black")
plt.xlabel("Number of iterations")
plt.ylabel("Cost")


# a function to compute gradient descent for single feature

def gradient_descent(x,y):
    m_coef, b_coef = 0 , 0
    n = len(x)
    learning_rate = 0.01
    cost = []
    iterations = 10

    for i in range(iterations):
        y_pred = b_coef + m_coef * x #since we have only one x or feature
        cost_function = (1/n) * sum([val**2 for val in (y_pred - y)]) #cost function
        cost.append(cost_function) 
        md = (1/2*n)*sum(x*(y_pred-y)) #slope derivative
        bd = (1/2*n)*sum(y_pred-y) #bias derivative
        b_coef = b_coef - learning_rate * bd
        m_coef = m_coef - learning_rate * md

        print("m {} , b {} , cost {}, iterations {} ".format(m_coef, b_coef, cost_function, i))

    return b_coef, m_coef, cost, cost_function


