import numpy as np
import matplotlib.pyplot as plt
import copy, math


# given data for classification problem ...
# X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
# y_train = np.array([0, 0, 0, 1, 1, 1])

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])






# sigmoid function
def sigmoid_func(z):
    d =1/(1+np.exp(-z))
    return d

def logistic_cost_func (w_arr,bi,  X_arr, y_arr):
    cost_ =0
    m,n = X_arr.shape
    for i in range(m):
        z_i = np.dot(w_arr, X_arr[i] ) +bi
        z_i_sig = sigmoid_func(z_i)
        j = (-y_arr[i])*np.log(z_i_sig) -(1-y_arr[i]) * np.log(1-z_i_sig)
        cost_ = cost_ + j

    return cost_

def derivative_function (x_arr, y, w_arr, b, learning_rate, m):
    dj_arr_new = np.zeros(len(w_arr))
    for i in range(len(w_arr)):
        dj_w_i = w_arr[i] - (learning_rate*(sigmoid_func(np.dot(w_arr, x_arr) + b) - y)*x_arr[i])/m
        dj_arr_new[i] =  dj_w_i

    temp_b = b - (learning_rate*(sigmoid_func(np.dot(w_arr, x_arr) + b) - y))/m

    return [dj_arr_new, temp_b]


def gradient_descent_function(X_arr, y_arr, w_arr, b,learning_rate, iteration) :
    m,n = X_arr.shape
    cost =0
    for itr in range(iteration):

        for i in range(m):
            x_arr_i = X_arr[i]
            # get value from derivative  function and get w_arr and b new value
            [w_arr_new, b_new] = derivative_function(x_arr_i, y_arr[i], w_arr, b, learning_rate,m )

            # update the w_arr and b values

            for k in range(len(w_arr_new)) :
                w_arr[k]= w_arr_new[k]

            b = b_new

        cost = logistic_cost_func(w_arr, b, X_arr, y_arr)
        print(f"cost value {itr} : {cost} ")

    return [w_arr, b, cost]


[w, b, cost] =gradient_descent_function(X_train, y_train, [0,0], 0, 0.1, 10000)

print(w,b, cost)



print(X_train.shape[0], len(y_train))

# show it on graph

x1 = X_train[:,0]
x2 = X_train[:,1]


x3 = np.arange(7)
y3=[]
for j in range(7):
    yi = w[0]*x3[j] + w[1]*x3[j] + b
    y3.append(yi)

print(y3)
plt.plot(x3, y3)
plt.scatter(x1, x2)
plt.show()
















