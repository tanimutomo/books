import numpy as np
import matplotlib.pyplot as plt
import argparse

# answer
def get_answer(s, e):
    x = np.linspace(s,e)
    y = np.sin(x)
    return x, y

# variable
def variable(M):
    D = 50
    m = np.zeros(M)
    l_lam = np.eye(M,M)
    return D, m, l_lam

def x_to_vec_x(x, D):
    ones = np.ones(x.shape)
    if args.M != 1:
        x_vec = np.array([ones, x])
        for i in range(2, args.M):
            tmp = x ** i
            x_vec = np.append(x_vec, np.array([tmp]), axis=0)
    else:
        x_vec = np.ones((1,D))
    return x_vec 

# train_data
def train_data(D):
    train_x = np.random.uniform(0, 7, args.N)
    train_y = np.sin(train_x)
    train_x_vec = x_to_vec_x(train_x, D)
    return train_x, train_y, train_x_vec

# calcurate
def input_var(D):
    input_x = np.linspace(0,7,D)
    input_x_vec = x_to_vec_x(input_x, D)
    return input_x, input_x_vec

def linear_regression():
    ynxn = 0
    xnxn = 0
    for i in range(args.N):
        tmp1 = train_y[i] * train_x_vec[:,i]
        tmp2 = np.dot(train_x_vec[:,i].reshape(args.M,1), train_x_vec[:,i].reshape(1,args.M))
        ynxn += tmp1
        xnxn += tmp2

    post_l_lam = args.s_lam * xnxn + l_lam
    inv_post_l_lam = np.linalg.inv(post_l_lam)
    tmp_post_m = args.s_lam * ynxn + np.dot(l_lam, m)
    post_m = np.dot(inv_post_l_lam, tmp_post_m)

    new_lam_mat = 1/args.s_lam + np.dot(np.dot(input_x_vec.T, inv_post_l_lam), input_x_vec)
    new_lam = np.diag(new_lam_mat)

    output_y_u = np.dot(post_m, input_x_vec)
    output_y_u_plam = output_y_u + np.sqrt(new_lam)
    output_y_u_mlam = output_y_u - np.sqrt(new_lam)

    return output_y_u, output_y_u_plam, output_y_u_mlam


def plotting():
    plt.plot(train_x, train_y, "o")
    plt.plot(ans_x, ans_y, color='r')
    output = linear_regression()
    plt.plot(input_x, output[0], color='b')
    plt.plot(input_x, output[1], linestyle='--', color='c')
    plt.plot(input_x, output[2], linestyle='--', color='c')
    plt.xlim([-0.5, 7])
    plt.ylim([-3, 3])
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='linear regression')
    parser.add_argument('--M', type=int, default=4,
                        help='input dimention')
    parser.add_argument('--N', type=int, default=10,
                        help='number of train data')
    parser.add_argument('--s_lam', type=float, default=10.0,
                        help='accracy parameter')
    parser.add_argument('--start_data', type=int, default=0,
                        help='start point of graph')
    parser.add_argument('--end_data', type=int, default=7,
                        help='end point of graph')
    args = parser.parse_args()

    ans_x, ans_y = get_answer(0, 7)
    D, m, l_lam = variable(args.M)
    train_x, train_y, train_x_vec = train_data(D)
    input_x, input_x_vec = input_var(D)
    output = linear_regression()
    plotting()

