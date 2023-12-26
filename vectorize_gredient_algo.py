import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt("data_studen.csv", delimiter=',')

(r,c) = data.shape
print(r,c)
w_arr = np.zeros(c-1)


x_train = data[: , 0 : c-1]
y_train = data[ : , c-1]
print(x_train)
print(y_train)

def cost_function (x_mat, y_arr, b, w_arr, m ) :
    cost =0
    for i in range(len(y_arr)):
        x_mar_i= x_mat[i]
        err = np.dot(x_mar_i, w_arr) +b -y_arr[i]
        err = err**2
        cost = cost + err

    return cost/(2*m)


# print(cost_function(x_train, y_train, 1, [1,2,3], 11))

def derivatives (x_arr,w_arr, y, b, m, learning_rate ):
    temp_w_arr = np.zeros(len(x_arr))
    for i in range(len(w_arr)) :
        ddw = ((learning_rate * x_arr[i])*(np.dot(w_arr, x_arr) + b - y)) / m
        temp_w_arr[i] = w_arr[i] - ddw

    ddb = (learning_rate*(np.dot(w_arr, x_arr) + b - y)) / m
    temp_b = b - ddb

    return [temp_b, temp_w_arr]

def gredient_descent_algorithm (x_mat, y_arr, w_arr, b , learning_rate, m, itr) :
    cost = []
    for i in range(itr):

        for j in range(len(y_arr)) :
            x_mat_i = x_mat[j]
            [temp_b , w_arr_new]= derivatives(x_mat_i, w_arr, y_arr[j], b,m, learning_rate)

            # update b and w_arr
            for k in range(len(w_arr)):
                w_arr[k]= w_arr_new[k]

            b = temp_b

        # store the cost value

        cost_j = cost_function(x_train, y_arr, b, w_arr, m)
        cost.append(cost_j)

    return [b, w_arr, cost]


[b, w_arrr, cost] = gredient_descent_algorithm(x_train, y_train, w_arr, 0, 0.001, r,  100)

print(b, w_arrr, cost)

it = np.arange(100)

plt.plot(it, cost)
plt.show()
print(x_train[1])
pp = x_train[:,1]
plt.scatter(pp, y_train)

plt.show()










