import numpy as np

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(X_train.shape)
x_tr = X_train[0, :]
print(x_tr)
b_init = 785.1811367994083

f = np.dot(w_init,x_tr) + b_init
print(f, "predictions")

#
# a =np.arange(6)
# print(a.shape, a.dtype, a)
#
# a = np.arange(10)
# print(f"a         = {a}")
#
# #access 5 consecutive elements (start:stop:step)
# c = a[2:7:1];     print("a[2:7:1] = ", c)
#
# # access 3 elements separated by two
# c = a[2:7:2];     print("a[2:7:2] = ", c)
#
# # access all elements index 3 and above
# c = a[3:];        print("a[3:]    = ", c)
#
# # access all elements below index 3
# c = a[:3];        print("a[:3]    = ", c)
#
# # access all elements
# c = a[:];         print("a[:]     = ", c)
#
# r = np.arange(10)
# rr= np.mean(r)
# print(rr)
#
#
# a = np.array([ 1, 2, 3, 4])
# b = np.array([-1,-2, 3, 4])
# ww= a+b
# print(ww)
#
# d = np.array([[0,0], [0,0]])
# print(d.shape)
# uu = np.arange(6).reshape(2,3)
#
# print(uu)
# print(uu.shape)
#
# a = np.arange(20).reshape(-1, 10)
# print("========")
# print(a.shape)
# print(a[0,2:7:1])
# print(f"a = \n{a}")
