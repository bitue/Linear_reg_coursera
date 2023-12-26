import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('marks.csv', delimiter=',')  # Specify delimiter if not comma-separated
x_train =[]
y_train =[]
m = data.shape[0]
for i in range (m):
    x_train.append(data[i][0])
    y_train.append(data[i][1])

plt.scatter(x_train, y_train)
plt.show()

# calculate the cost function
def cost_function(w,b,x_arr, y_arr):
    j_cost =0
    for k in range(len(x_arr)):
        p =((w*x_arr[k] + b) - y_arr[k])**2
        j_cost = j_cost + p

    return j_cost/(2*m)

print(f"know the cost function of random value {cost_function(2,1.5,x_train, y_train)}")

# know the w , b for using grad decent algorithm

def gradient_decent_algo(w, b, x_arr, y_arr, learning_rate, trained_number):
    for _ in range(trained_number):
        for j in range(len(x_arr)):
            temp_w = w - ((learning_rate*(w*x_arr[j]+b - y_arr[j]) ))*x_arr[j] /trained_number
            temp_b = b - (learning_rate*(w*x_arr[j]+b - y_arr[j]) )/trained_number

            #update the w and b
            # if (w== temp_w ):
            #     print(f"the moment trained success {i}")
            #     break
            w = temp_w
            b = temp_b


    return [w ,b ]


prms = gradient_decent_algo(0,0, x_train, y_train, 0.01, 1000)
print(f'after trained the cost is {cost_function(prms[0], prms[1], x_train, y_train)}')

# draw the linear model

x_min_value = int(min(x_train))
x_max_value =int(max(x_train))

target_value = []
features = []

for i in range(x_max_value-x_min_value):
    y_pre = prms[0]*i + prms[1]
    target_value.append(y_pre)
    features.append(i)


# show the predicted curve
# print(x_max_value)

plt.plot(features, target_value)
plt.scatter(x_train, y_train)
plt.title(f"cost value is {cost_function(prms[0],prms[1],x_train, y_train)} and (w,b) = ({prms[0]}, {prms[1]}) ")
plt.xlabel("Study Hours")
plt.ylabel("Math Marks")
plt.show()


















