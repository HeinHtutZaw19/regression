#Gradient Descent for Linear Regression
# f_wb = wx + b
# loss = (y-f_wb)**2 + b

import numpy as np 

#Initialize the parameters
x = np.random.randn(10, 1)
y = 6*x + np.random.rand()

#Iteratively find out what these parameters are
w = 0.0
b = 0.0

#Hyperparameter
alpha = 0.01

#Create gradient descent function
def descend(x, y, w, b, alpha):
    dl_dw = 0.0
    dl_db = 0.0
    N = x.shape[0]

    for xi, yi in zip(x,y):
        dl_dw += -2*xi*(yi-(w*xi+b))
        dl_db += -2*(yi-(w*xi+b))

    w = w - alpha * (1/N) * dl_dw
    b = b - alpha * (1/N) * dl_db

    return w,b

for epoch in range(400):
    w,b = descend(x,y,w,b,alpha)
    f_wb = w*x+b
    loss = np.divide(np.sum((y-f_wb)**2,axis = 0),x.shape[0])
    print(f'{epoch} loss is {loss}, parameters w:{w}, b:{b}')
