
import numpy as np

# Load the data
data = np.loadtxt("data_studen.csv", delimiter=',')
r, c = data.shape

# Extract features and labels
x_train = data[:, 0:c-1]
y_train = data[:, c-1]

# Cost function
def cost_function(w_arr, x_arr, y, b, m):
    cost = 0
    for i in range(len(x_arr)):
        cost_i = (np.dot(w_arr, x_arr[i]) + b - y[i]) ** 2 / (2 * m)
        cost += cost_i
    return cost

# Gradient Descent
def vec(x_arr, y_arr, w_arr, b, learning_rate, iterations, trained_number):
    j_arr = []  # Cost function array
    for itr in range(iterations):
        dw = np.zeros(x_train.shape[1])
        db = 0
        for i in range(trained_number):
            x_arr_i = x_arr[i]
            y_i = y_arr[i]
            y_hat = np.dot(w_arr, x_arr_i) + b

            dw += np.dot((y_hat - y_i), x_arr_i)
            db += (y_hat - y_i)

        dw /= trained_number
        db /= trained_number

        w_arr -= learning_rate * dw
        b -= learning_rate * db

        j = cost_function(w_arr, x_arr, y_arr, b, trained_number)
        j_arr.append(j)

        print(f"Iteration: {itr}, Cost: {j}, Weights: {w_arr}, Bias: {b}\n")

    return [w_arr, b, j_arr]

# Run gradient descent
result = vec(x_train, y_train, [0, 0, 0, 0], 0, 0.01, 1000, r)
print(result)

