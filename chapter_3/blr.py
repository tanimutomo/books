import numpy as np

class BayesianLinearRegression(object):
    def __init__(self, M, s_lam):
        self.M = M
        self.s_lam = s_lam
        self.prior_m = np.zeros(M)
        self.prior_l_lam = np.eye(M,M)
    
    def mapping(self, x):
        x = x.reshape(1, -1)
        x_vec = np.concatenate([
            np.power(x, i) for i in range(self.M)],
            axis=0)
        return x_vec

    def calcurate_posterior(self, tx, ty):
        tx = tx.T
        ynxn = 0
        xnxn = 0
        for x, y in zip(tx, ty):
            x = x.reshape(1, -1)
            ynxn += y * x
            xnxn += np.dot(x.T, x)
        post_l_lam = self.s_lam * xnxn + self.prior_l_lam
        inv_post_l_lam = np.linalg.inv(post_l_lam)
        post_m = self.s_lam * ynxn + np.dot(self.prior_l_lam, self.prior_m)
        post_m = np.dot(inv_post_l_lam, post_m.T)

        self.prior_m = post_m
        self.prior_l_lam = post_l_lam
        
        return post_m, post_l_lam
    
    def calcurate_predictive(self, xs, post_m=None, post_l_lam=None):
        if post_m is None and post_l_lam is None:
            post_m = self.prior_m
            post_l_lam = self.prior_l_lam

        inv_post_l_lam = np.linalg.inv(post_l_lam)

        post_s_lam = 1 / self.s_lam + np.dot(np.dot(xs.T, inv_post_l_lam), xs)
        post_s_lam = np.diag(post_s_lam).reshape(-1, 1)

        output_y_u = np.dot(xs.T, post_m)
        output_y_u_plam = output_y_u + np.sqrt(post_s_lam)
        output_y_u_mlam = output_y_u - np.sqrt(post_s_lam)

        return output_y_u, output_y_u_plam, output_y_u_mlam

