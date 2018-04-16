import numpy as np

class BayesLinearRegression(object):
    def __init__(self, M, lam):
        self.M = M
        self.lam = lam

        self.prior_m = np.zeros((1, self.M))
        self.prior_l_lam = np.identity(self.M)

    def transfer(self, x):
        '''
        tmp = []
        for i in range(self.M):
            x = x.reshape(-1, 1)
            x = np.power(x, i)
            tmp.append(x)
        features = np.concatenate(tmp, axis=1)
        '''
        features = np.concatenate([
            np.power(x.reshape(-1, 1), i) for i in range(self.M)],
            axis=1)
        return features

    def calculate_posterior(self, xs, ys, update=False):
        xs = self.transfer(xs)
        ynxn = 0
        xnxn = 0

        for x, y in zip(xs, ys):
            x = x.reshape(1, -1)
            ynxn += y * x
            xnxn += np.dot(x.T, x)

        post_l_lam = self.lam * xnxn + self.prior_l_lam

        inv_post_l_lam = np.linalg.inv(post_l_lam)

        post_m = self.lam * ynxn + np.dot(self.prior_m, self.prior_l_lam)
        post_m = np.dot(post_m, inv_post_l_lam)

        if update:
            self.prior_m = post_m
            self.prior_l_lam = post_l_lam

        return post_m, post_l_lam

    def calculate_predictive(self, xs, post_m=None, post_l_lam=None):
        if post_m is None and post_l_lam is None:
            post_m = self.prior_m
            post_l_lam = self.prior_l_lam

        xs = self.transfer(xs)
        inv_post_l_lam = np.linalg.inv(post_l_lam)

        new_lam_mat = 1 / self.lam + np.dot(np.dot(xs, inv_post_l_lam), xs.T)
        new_lam = np.diag(new_lam_mat)

        output_y_u = np.dot(xs, post_m.T).reshape(-1)
        output_y_u_plam = output_y_u + np.sqrt(new_lam)
        output_y_u_mlam = output_y_u - np.sqrt(new_lam)

        return output_y_u, output_y_u_plam, output_y_u_mlam
