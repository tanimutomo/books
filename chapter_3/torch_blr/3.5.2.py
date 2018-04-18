import argparse
import utils
from blr import BayesianLinearRegression

def main(args):
    M = args.M
    N = args.N
    s_lam = args.s_lam

    train_data, ground_truth = utils.create_dataset(N)

    model = BayesianLinearRegression(args.M, args.s_lam)

    train_x_vec = model.mapping(train_data[0])
    gt_x_vec = model.mapping(ground_truth[0])
    post_m, post_l_lam = model.calcurate_posterior(train_x_vec, train_data[1])
    prediction = model.calcurate_predictive(gt_x_vec)

    utils.plotting(ground_truth, train_data, prediction)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='linear regression')
    parser.add_argument('--M', type=int, default=4,
                        help='input dimention')
    parser.add_argument('--N', type=int, default=10,
                        help='number of train data')
    parser.add_argument('--s_lam', type=float, default=10.0,
                        help='accracy parameter')
    args = parser.parse_args()

    main(args)
