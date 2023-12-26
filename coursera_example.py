import copy, math
import numpy as np
import matplotlib.pyplot as plt
X_train = np.array([[1, 5, 1, 4], [1, 3, 2, 4], [8, 2, 1, 3]])
y_train = np.array([9, 2, 7])

def cost_function_j (x_arr, y_arr, b , m , w_arr):
    cost =0
    for i in range(len(y_arr)):
        x_arr_i = x_arr[i]
        cost_i = np.dot(x_arr_i, w_arr ) + b - y_arr[i]
        cost_i = cost_i**2
        cost = cost + cost_i

    return  cost /( 2*m)

# j =cost_function_j(X_train, y_train, 785.1811367994083, 3,  [ 0.39133535, 18.75376741, -53.36032453, -26.42131618] )
# print(j)

def derivative(w_arr, x_arr, y , b , m, learning_rate ):
    temp_w_arr = np.zeros(len(x_arr))
    #print(temp_w_arr,"first print derivative")
    for i in range(len(x_arr)) :
        dot_product_w =( np.dot(x_arr, w_arr) +b -y)
        dot_product_w = dot_product_w * x_arr[i]*learning_rate
        dot_product_w = dot_product_w/m
        temp_w_i = w_arr[i] - dot_product_w
        temp_w_arr[i] = temp_w_i
    dot_product_b = ((np.dot(x_arr, w_arr) + b -y)) * learning_rate / m
    temp_b = b - dot_product_b
    #print(temp_w_arr, " print tempw_arr")

    return [temp_b, temp_w_arr]

def gradient_desent (x_arr, y_arr, w_arr, b, learning_rate, iteration, trained_number):
    cost_arr =[]
    for itr in range(iteration):

        for i in range(len(y_arr)):
            x_arr_i = x_arr[i]
            [temp_b , w_arr_new ]= derivative(w_arr,x_arr_i, y_arr[i],b, trained_number, learning_rate)

        # update w_arr
        for i in range(len(w_arr)):
            w_arr[i]= w_arr_new[i]
        # update b
        b= temp_b

        j = cost_function_j(x_arr, y_train, b, 3,w_arr)
        # print(j)
        cost_arr.append(j)



    return [w_arr, b, cost_arr]

[w, b, cost] = gradient_desent(X_train, y_train, [0,0,0,0], 0, 0.01, 100, 3)




print(w , " weight \n")
print(b , " b \n")
print(cost , "cost \n")

itr = np.arange(100)


# plot the data set visualize



plt.plot(itr, cost)
plt.scatter(X_train[:, 1], y_train)

plt.show()










