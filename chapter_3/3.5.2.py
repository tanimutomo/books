import numpy as np
import matplotlib.pyplot as plt

import argparse


def x_to_vec_x(x, M):
    ones = np.ones(x.shape)
    x_vec = np.array([ones, x])
    for i in range(2, M):
        tmp = x ** i
        x_vec = np.append(x_vec, np.array([tmp]), axis=0)
    return x_vec


def create_dataset(N, function=np.sin):
    train_x = np.random.uniform(0, 7, N)
    train_y = function(train_x)

    train_y += 0.5 * np.random.randn(*train_y.shape)

    return train_x, train_y


def estimate_posterior(train_data, N, M, lam):
    ynxn = 0
    xnxn = 0

    m = np.zeros(M)
    l_lam = np.eye(M,M)

    train_x_vec, train_y = train_data

    for i in range(N):
        tmp1 = train_y[i] * train_x_vec[:,i]
        ynxn += tmp1
        tmp2 = np.dot(train_x_vec[:,i].reshape(M,1), train_x_vec[:,i].reshape(1,M))
        xnxn += tmp2

    post_l_lam = lam * xnxn + l_lam
    inv_post_l_lam = np.linalg.inv(post_l_lam)
    tmp_post_m = lam * ynxn + np.dot(l_lam, m)
    post_m = np.dot(inv_post_l_lam, tmp_post_m)

    return post_m, post_l_lam


def estimate_predictive(input_x_vec, post_m, post_l_lam, lam):
    inv_post_l_lam = np.linalg.inv(post_l_lam)

    new_lam_mat = 1/lam + np.dot(np.dot(input_x_vec.T, inv_post_l_lam), input_x_vec)
    new_lam = np.diag(new_lam_mat)

    output_y_u = np.dot(post_m, input_x_vec)
    output_y_u_plam = output_y_u + np.sqrt(new_lam)
    output_y_u_mlam = output_y_u - np.sqrt(new_lam)

    return output_y_u, output_y_u_plam, output_y_u_mlam


def plotting(ground_truth, training_data, prediction):
    plt.plot(training_data[0], training_data[1], "o")
    plt.plot(ground_truth[0], ground_truth[1], color='r')
    plt.plot(ground_truth[0], prediction[0], color='b')
    plt.plot(ground_truth[0], prediction[1], linestyle='--', color='c')
    plt.plot(ground_truth[0], prediction[2], linestyle='--', color='c')
    plt.xlim([-0.5, 7])
    plt.ylim([-3, 3])
    plt.show()


def main(args):
    M = args.M
    N = args.N
    D = args.D
    lam = args.lam

    function = np.sin

    train_x, train_y = create_dataset(N, function)
    train_x_vec = x_to_vec_x(train_x, M)

    input_x = np.linspace(0, 7, 100)
    y = function(input_x)
    input_x_vec = x_to_vec_x(input_x, M)

    m, l = estimate_posterior((train_x_vec, train_y), N, M, lam)

    output = estimate_predictive(input_x_vec, m, l, lam)

    plotting((input_x, y), (train_x, train_y), output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ArgParser')
    parser.add_argument('--M', type=int, default=4,
                        help='M')
    parser.add_argument('--N', type=int, default=10,
                        help='N')
    parser.add_argument('--D', type=int, default=50,
                        help='D')
    parser.add_argument('--lam', type=float, default=10.0,
                        help='lambda')

    args = parser.parse_args()

    main(args)
