import numpy as np
import matplotlib.pyplot as plt

M = 4
m = np.zeros(M)
gamma = np.eye(M,M)

w1 = np.random.multivariate_normal(m, gamma)
w2 = np.random.multivariate_normal(m, gamma)
w3 = np.random.multivariate_normal(m, gamma)
w4 = np.random.multivariate_normal(m, gamma)

x = np.arange(-1, 1, 0.01)
x2 = x ** 2
x3 = x ** 3
one = np.ones(x.shape, dtype=float)
xs = np.array([one, x, x2, x3])

y1 = np.dot(w1.T, xs)
y2 = np.dot(w2.T, xs)
y3 = np.dot(w3.T, xs)
y4 = np.dot(w4.T, xs)

plt.plot(x, y1, color='b')
plt.plot(x, y2, color='g')
plt.plot(x, y3, color='r')
plt.plot(x, y4, color='c')
print('学習前のモデルからの3次関数のサンプル')
plt.show()

x_2 = np.arange(-1, 1, 0.1)
x2_2 = x_2 ** 2
x3_2 = x_2 ** 3
one_2 = np.ones(x_2.shape, dtype=float)
xs_2 = np.array([one_2, x_2, x2_2, x3_2])

print(w1.shape, xs_2.shape)
mean = np.dot(w1.T, xs_2)
lam = 10.0
y_2 = np.random.normal(mean, 1/lam)

plt.plot(x, y1, color='b')
plt.scatter(x_2, y_2)
plt.show()
