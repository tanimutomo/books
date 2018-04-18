import torch

class BayesianLinearRegression(object):
    def __init__(self, M, s_lam):
        self.M = M
        self.s_lam = s_lam
        self.prior_m = torch.zeros(M).view(-1,1)
        self.prior_l_lam = torch.eye(M)
    
    def mapping(self, x):
        x_vec = torch.ones(len(x)).view(1, -1)
        for i in range(1, self.M):
            tmp = torch.pow(x, i).view(1, -1)
            x_vec = torch.cat((x_vec, tmp), 0)
        return x_vec
            
    def calcurate_posterior(self, tx, ty):
        tx = torch.t(tx)
        ynxn = 0
        xnxn = 0
        for x, y in zip(tx, ty):
            x = x.view(-1, 1)
            ynxn += y * x
            xnxn += torch.mm(x, torch.t(x))
        post_l_lam = self.s_lam * xnxn + self.prior_l_lam
        inv_post_l_lam = torch.inverse(post_l_lam)
        post_m = self.s_lam * ynxn + torch.mm(self.prior_l_lam, self.prior_m)
        post_m = torch.mm(inv_post_l_lam, post_m)

        self.prior_m = post_m
        self.prior_l_lam = post_l_lam
        
        return post_m, post_l_lam
    
    def calcurate_predictive(self, xs, post_m=None, post_l_lam=None):
        if post_m is None and post_l_lam is None:
            post_m = self.prior_m
            post_l_lam = self.prior_l_lam

        inv_post_l_lam = torch.inverse(post_l_lam)

        post_s_lam = 1 / self.s_lam + torch.mm(torch.mm(torch.t(xs), inv_post_l_lam), xs)
        post_s_lam = torch.diag(post_s_lam).view(-1, 1)

        output_y_u = torch.mm(torch.t(xs), post_m).view(-1)
        output_y_u_plam = output_y_u + torch.sqrt(post_s_lam).view(-1)
        output_y_u_mlam = output_y_u - torch.sqrt(post_s_lam).view(-1)

        return output_y_u, output_y_u_plam, output_y_u_mlam

