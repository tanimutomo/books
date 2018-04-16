import numpy as np
import matplotlib.pyplot as plt

import argparse

import utils
from blr import BayesLinearRegression



def main(args):
    M = args.M
    N = args.N
    lam = args.lam
    function = np.sin

    train_x, train_y = utils.create_dataset(N, function)

    xs = np.linspace(0, 7, 100)
    ys = function(xs)

    model = BayesLinearRegression(M, lam)
    
    post_m, post_l_lam = model.calculate_posterior(train_x, train_y)
    output = model.calculate_predictive(xs, post_m, post_l_lam)

    utils.plotting((xs, ys), (train_x, train_y), output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ArgParser')
    parser.add_argument('--M', type=int, default=4,
                        help='M')
    parser.add_argument('--N', type=int, default=10,
                        help='N')
    parser.add_argument('--lam', type=float, default=10.0,
                        help='lambda')

    args = parser.parse_args()

    main(args)
