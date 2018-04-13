import numpy as np
import matplotlib.pyplot as plt

# answer
def answer(s, e):
    x = np.linspace(s,e)
    y = np.sin(x)
    return x, y

# variable
def variable():
    M = 4
    N = 10
    lam = 10.0
    m = np.zeros(M)
    l_lam = np.eye(M,M)
    return M, N, lam, m, l_lam

def x_to_vec_x(x, d=M):
    ones = np.ones(x.shape)
    x_vec = np.array([ones, x])
    for i in range(2, M):
        tmp = x ** i
        x_vec = np.append(x_vec, np.array([tmp]), axis=0)
    return x_vec

# train_data
def train_data(N=N):
    train_x = np.random.uniform(0, 7, N)
    train_y = np.sin(train_x)
    train_x_vec = x_to_vec_x(train_x)
    return train_x, train_y, train_x_vec

# calcurate
def input_var(D=50):
    D = D
    input_x = np.linspace(0,7,D)
    input_x_vec = x_to_vec_x(input_x)
    return input_x, input_x_vec

def linear_regression():
    ynxn = 0
    xnxn = 0
    for i in range(N):
        tmp1 = train_y[i] * train_x_vec[:,i]
        ynxn += tmp1
        tmp2 = np.dot(np.reshape(np.array(train_x_vec[:,i]), (M,1)), np.reshape(np.array(train_x_vec[:,i]), (1,M)))
        xnxn += tmp2

    post_l_lam = lam * xnxn + l_lam
    inv_post_l_lam = np.linalg.inv(post_l_lam)
    tmp_post_m = lam * ynxn + np.dot(l_lam, m)
    post_m = np.dot(inv_post_l_lam, tmp_post_m)

    new_lam_mat = 1/lam + np.dot(np.dot(input_x_vec.T, inv_post_l_lam), input_x_vec)
    new_lam = np.diag(new_lam_mat)

    output_y_u = np.dot(post_m, input_x_vec)
    output_y_u_plam = output_y_u + np.sqrt(new_lam)
    output_y_u_mlam = output_y_u - np.sqrt(new_lam)

    return output_y_u, output_y_u_plam, output_y_u_mlam


def plotting():
    plt.plot(train_x, train_y, "o")
    plt.plot(x, y, color='r')
    output = linear_regression()
    plt.plot(input_x, output[0], color='b')
    plt.plot(input_x, output[1], linestyle='--', color='c')
    plt.plot(input_x, output[2], linestyle='--', color='c')
    plt.xlim([-0.5, 7])
    plt.ylim([-3, 3])
    plt.show()

ans_x, ans_y = answer()
M, N, lam, m, l_lam = variable()


