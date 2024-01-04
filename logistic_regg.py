import numpy as np
import matplotlib.pyplot as plt

# given data for classification problem ...
X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])


def sigmoid_function (z):
    pass

def compute_cost_logistic (w_arr, x_arr, y_arr, b) :
    m,n = x_arr.shape
    cost =0.0

    for i in range(m):
        x_arr_i = x_arr[i]
        z_i =( np.dot(x_arr_i, w_arr) +b - y_arr[i])







