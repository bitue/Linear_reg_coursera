import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("data_studen.csv", delimiter=',')
print(data)
r,c = data.shape
print(r,c)

x_train = data[: , 0 : c-1]
y_train = data[ : , c-1]
print(type(x_train))
print(y_train)

# cost function
def cost_function (w_arr, x_arr, y, b, m) :
    cost = 0
    for i in range(len(w_arr)) :
        cost_i = (w_arr[i]*x_arr[i] + b -y) **2 /2*m
        cost = cost +  cost_i
    return cost

# write Gradient Disent
def gredient_discent(x_arr,  y_arr,w_arr, b, learning_rate, iterations, trained_number):
    j_arr =[] # cost function arr
    for itr in range(iterations):
        for i in range(trained_number):
            dxy =0
            x_arr_i = x_arr[i]
            xx =0
            for j in range(len(x_arr_i)) :
                xx = xx + x_arr_i[j]*w_arr[j]
            dxy = (xx - y_arr[i] + b ) * (learning_rate / trained_number)

            # write the temp w values and finally update them
            temp_w =[]
            for k in range(len(x_arr_i)) :
                temp_w_k = w_arr[k] - dxy * x_arr_i[k]
                temp_w.append(temp_w_k)

            # update the temp_b value
            temp_b = b - dxy

            # update the w_arr values and temp_b values

            b = temp_b

            for w in range(len(temp_w)):
                w_arr[w] = temp_w[w]

            j = cost_function(w_arr, x_arr_i, y_arr[i], b , trained_number)

            print(f" iteration : {itr} \n  cost {j} weights {w_arr} and b {b} \n")
    return w_arr




y = gredient_discent(x_train, y_train, [0,0,0,0], 0, 0.01, 1000 , r)
print(y)






















