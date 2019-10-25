import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from utils import Normal, float_tensor, logitexp, sample_DAG, sample_bipartite, Flatten, one_hot
from torch.distributions import Categorical


class RegressionFNP(nn.Module):
    """
    Functional Neural Process for regression
    """
    def __init__(self, dim_x=1, dim_y=1, dim_h=50, transf_y=None, n_layers=1, use_plus=True, num_M=100,
                 dim_u=1, dim_z=1, fb_z=0.):
        '''
        :param dim_x: Dimensionality of the input
        :param dim_y: Dimensionality of the output
        :param dim_h: Dimensionality of the hidden layers
        :param transf_y: Transformation of the output (e.g. standardization)
        :param n_layers: How many hidden layers to use
        :param use_plus: Whether to use the FNP+
        :param num_M: How many points exist in the training set that are not part of the reference set
        :param dim_u: Dimensionality of the latents in the embedding space
        :param dim_z: Dimensionality of the  latents that summarize the parents
        :param fb_z: How many free bits do we allow for the latent variable z
        '''
        super(RegressionFNP, self).__init__()

        self.num_M = num_M
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_h = dim_h
        self.dim_u = dim_u
        self.dim_z = dim_z
        self.use_plus = use_plus
        self.fb_z = fb_z
        self.transf_y = transf_y
        # normalizes the graph such that inner products correspond to averages of the parents
        self.norm_graph = lambda x: x / (torch.sum(x, 1, keepdim=True) + 1e-8)

        self.register_buffer('lambda_z', float_tensor(1).fill_(1e-8))

        # function that assigns the edge probabilities in the graph
        self.pairwise_g_logscale = nn.Parameter(float_tensor(1).fill_(math.log(math.sqrt(self.dim_u))))
        self.pairwise_g = lambda x: logitexp(-.5 * torch.sum(torch.pow(x[:, self.dim_u:] - x[:, 0:self.dim_u], 2), 1,
                                                             keepdim=True) / self.pairwise_g_logscale.exp()).view(x.size(0), 1)
        # transformation of the input
        init = [nn.Linear(dim_x, self.dim_h), nn.ReLU()]
        for i in range(n_layers - 1):
            init += [nn.Linear(self.dim_h, self.dim_h), nn.ReLU()]
        self.cond_trans = nn.Sequential(*init)
        # p(u|x)
        self.p_u = nn.Linear(self.dim_h, 2 * self.dim_u)
        # q(z|x)
        self.q_z = nn.Linear(self.dim_h, 2 * self.dim_z)
        # for p(z|A, XR, yR)
        self.trans_cond_y = nn.Linear(self.dim_y, 2 * self.dim_z)

        # p(y|z) or p(y|z, u)
        self.output = nn.Sequential(nn.Linear(self.dim_z if not self.use_plus else self.dim_z + self.dim_u, self.dim_h),
                                    nn.ReLU(), nn.Linear(self.dim_h, 2 * dim_y))

    def forward(self, XR, yR, XM, yM, kl_anneal=1.):
        X_all = torch.cat([XR, XM], dim=0)
        H_all = self.cond_trans(X_all)

        # get U
        pu_mean_all, pu_logscale_all = torch.split(self.p_u(H_all), self.dim_u, dim=1)
        pu = Normal(pu_mean_all, pu_logscale_all)
        u = pu.rsample()

        # get G
        G = sample_DAG(u[0:XR.size(0)], self.pairwise_g, training=self.training)

        # get A
        A = sample_bipartite(u[XR.size(0):], u[0:XR.size(0)], self.pairwise_g, training=self.training)

        # get Z
        qz_mean_all, qz_logscale_all = torch.split(self.q_z(H_all), self.dim_z, 1)
        qz = Normal(qz_mean_all, qz_logscale_all)
        z = qz.rsample()

        cond_y_mean, cond_y_logscale = torch.split(self.trans_cond_y(yR), self.dim_z, 1)
        pz_mean_all = torch.mm(self.norm_graph(torch.cat([G, A], dim=0)), cond_y_mean + qz_mean_all[0:XR.size(0)])
        pz_logscale_all = torch.mm(self.norm_graph(torch.cat([G, A], dim=0)), cond_y_logscale + qz_logscale_all[0:XR.size(0)])

        pz = Normal(pz_mean_all, pz_logscale_all)

        pqz_all = pz.log_prob(z) - qz.log_prob(z)

        # apply free bits for the latent z
        if self.fb_z > 0:
            log_qpz = - torch.sum(pqz_all)

            if self.training:
                if log_qpz.item() > self.fb_z * z.size(0) * z.size(1) * (1 + 0.05):
                    self.lambda_z = torch.clamp(self.lambda_z * (1 + 0.1), min=1e-8, max=1.)
                elif log_qpz.item() < self.fb_z * z.size(0) * z.size(1):
                    self.lambda_z = torch.clamp(self.lambda_z * (1 - 0.1), min=1e-8, max=1.)

            log_pqz_R = self.lambda_z * torch.sum(pqz_all[0:XR.size(0)])
            log_pqz_M = self.lambda_z * torch.sum(pqz_all[XR.size(0):])

        else:
            log_pqz_R = torch.sum(pqz_all[0:XR.size(0)])
            log_pqz_M = torch.sum(pqz_all[XR.size(0):])

        final_rep = z if not self.use_plus else torch.cat([z, u], dim=1)

        mean_y, logstd_y = torch.split(self.output(final_rep), 1, dim=1)
        logstd_y = torch.log(0.1 + 0.9 * F.softplus(logstd_y))

        mean_yR, mean_yM = mean_y[0:XR.size(0)], mean_y[XR.size(0):]
        logstd_yR, logstd_yM = logstd_y[0:XR.size(0)], logstd_y[XR.size(0):]

        # logp(R)
        pyR = Normal(mean_yR, logstd_yR)
        log_pyR = torch.sum(pyR.log_prob(yR))

        # logp(M|S)
        pyM = Normal(mean_yM, logstd_yM)
        log_pyM = torch.sum(pyM.log_prob(yM))

        obj_R = (log_pyR + log_pqz_R) / float(self.num_M)
        obj_M = (log_pyM + log_pqz_M) / float(XM.size(0))

        obj = obj_R + obj_M

        loss = - obj

        return loss

    def predict(self, x_new, XR, yR, sample=True):

        H_all = self.cond_trans(torch.cat([XR, x_new], 0))

        # get U
        pu_mean_all, pu_logscale_all = torch.split(self.p_u(H_all), self.dim_u, dim=1)
        pu = Normal(pu_mean_all, pu_logscale_all)
        u = pu.rsample()

        A = sample_bipartite(u[XR.size(0):], u[0:XR.size(0)], self.pairwise_g, training=False)

        pz_mean_all, pz_logscale_all = torch.split(self.q_z(H_all[0:XR.size(0)]), self.dim_z, 1)
        cond_y_mean, cond_y_logscale = torch.split(self.trans_cond_y(yR), self.dim_z, 1)
        pz_mean_all = torch.mm(self.norm_graph(A), cond_y_mean + pz_mean_all)
        pz_logscale_all = torch.mm(self.norm_graph(A), cond_y_logscale + pz_logscale_all)
        pz = Normal(pz_mean_all, pz_logscale_all)

        z = pz.rsample()
        final_rep = z if not self.use_plus else torch.cat([z, u[XR.size(0):]], dim=1)

        mean_y, logstd_y = torch.split(self.output(final_rep), 1, dim=1)
        logstd_y = torch.log(0.1 + 0.9 * F.softplus(logstd_y))

        init_y = Normal(mean_y, logstd_y)
        if sample:
            y_new_i = init_y.sample()
        else:
            y_new_i = mean_y

        y_pred = y_new_i

        if self.transf_y is not None:
            if torch.cuda.is_available():
                y_pred = self.transf_y.inverse_transform(y_pred.cpu().data.numpy())
            else:
                y_pred = self.transf_y.inverse_transform(y_pred.data.numpy())

        return y_pred


class ClassificationFNP(nn.Module):
    """
    Functional Neural Process for classification with the LeNet-5 architecture
    """
    def __init__(self, dim_x=(1, 28, 28), dim_y=10, use_plus=True, num_M=1, dim_u=32, dim_z=64, fb_z=1.0):
        '''
        :param dim_x: Dimensionality of the input
        :param dim_y: Dimensionality of the output
        :param use_plus: Whether to use the FNP+
        :param num_M: How many points exist in the training set that are not part of the reference set
        :param dim_u: Dimensionality of the latents in the embedding space
        :param dim_z: Dimensionality of the  latents that summarize the parents
        :param fb_z: How many free bits do we allow for the latent variable z
        '''
        super(ClassificationFNP, self).__init__()

        self.num_M = num_M
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_u = dim_u
        self.dim_z = dim_z
        self.use_plus = use_plus
        self.fb_z = fb_z
        # normalizes the graph such that inner products correspond to averages of the parents
        self.norm_graph = lambda x: x / (torch.sum(x, 1, keepdim=True) + 1e-8)

        self.register_buffer('lambda_z', float_tensor(1).fill_(1e-8))

        # function that assigns the edge probabilities in the graph
        self.pairwise_g_logscale = nn.Parameter(float_tensor(1).fill_(math.log(math.sqrt(self.dim_u))))
        self.pairwise_g = lambda x: logitexp(-.5 * torch.sum(torch.pow(x[:, self.dim_u:] - x[:, 0:self.dim_u], 2), 1,
                                                             keepdim=True) / self.pairwise_g_logscale.exp()).view(x.size(0), 1)

        # transformation of the input
        self.cond_trans = nn.Sequential(nn.Conv2d(self.dim_x[0], 20, 5), nn.ReLU(), nn.MaxPool2d(2),
                                        nn.Conv2d(20, 50, 5), nn.ReLU(), nn.MaxPool2d(2), Flatten(),
                                        nn.Linear(800, 500))

        # p(u|x)
        self.p_u = nn.Sequential(nn.ReLU(), nn.Linear(500, 2 * self.dim_u))
        # q(z|x)
        self.q_z = nn.Sequential(nn.ReLU(), nn.Linear(500, 2 * self.dim_z))
        # for p(z|A, XR, yR)
        self.trans_cond_y = nn.Linear(self.dim_y, 2 * self.dim_z)

        # p(y|z) or p(y|z, u)
        self.output = nn.Sequential(nn.ReLU(),
                                    nn.Linear(self.dim_z if not self.use_plus else self.dim_z + self.dim_u, dim_y))

    def forward(self, XM, yM, XR, yR, kl_anneal=1.):
        X_all = torch.cat([XR, XM], dim=0)
        H_all = self.cond_trans(X_all)

        # get U
        pu_mean_all, pu_logscale_all = torch.split(self.p_u(H_all), self.dim_u, dim=1)
        pu = Normal(pu_mean_all, pu_logscale_all)
        u = pu.rsample()

        # get G
        G = sample_DAG(u[0:XR.size(0)], self.pairwise_g, training=self.training)

        # get A
        A = sample_bipartite(u[XR.size(0):], u[0:XR.size(0)], self.pairwise_g, training=self.training)

        # get Z
        qz_mean_all, qz_logscale_all = torch.split(self.q_z(H_all), self.dim_z, 1)
        qz = Normal(qz_mean_all, qz_logscale_all)
        z = qz.rsample()

        cond_y_mean, cond_y_logscale = torch.split(self.trans_cond_y(one_hot(yR, n_classes=self.dim_y)), self.dim_z, 1)
        pz_mean_all = torch.mm(self.norm_graph(torch.cat([G, A], dim=0)), cond_y_mean + qz_mean_all[0:XR.size(0)])
        pz_logscale_all = torch.mm(self.norm_graph(torch.cat([G, A], dim=0)), cond_y_logscale + qz_logscale_all[0:XR.size(0)])

        pz = Normal(pz_mean_all, pz_logscale_all)

        pqz_all = pz.log_prob(z) - qz.log_prob(z)

        # apply free bits for the latent z
        if self.fb_z > 0:
            log_qpz = - torch.sum(pqz_all)

            if self.training:
                if log_qpz.item() > self.fb_z * z.size(0) * z.size(1) * (1 + 0.05):
                    self.lambda_z = torch.clamp(self.lambda_z * (1 + 0.1), min=1e-8, max=1.)
                elif log_qpz.item() < self.fb_z * z.size(0) * z.size(1):
                    self.lambda_z = torch.clamp(self.lambda_z * (1 - 0.1), min=1e-8, max=1.)

            log_pqz_R = self.lambda_z * torch.sum(pqz_all[0:XR.size(0)])
            log_pqz_M = self.lambda_z * torch.sum(pqz_all[XR.size(0):])

        else:
            log_pqz_R = torch.sum(pqz_all[0:XR.size(0)])
            log_pqz_M = torch.sum(pqz_all[XR.size(0):])

        final_rep = z if not self.use_plus else torch.cat([z, u], dim=1)

        logits_all = self.output(final_rep)

        pyR = Categorical(logits=logits_all[0:XR.size(0)])
        log_pyR = torch.sum(pyR.log_prob(yR))

        pyM = Categorical(logits=logits_all[XR.size(0):])
        log_pyM = torch.sum(pyM.log_prob(yM))

        obj_R = (log_pyR + log_pqz_R) / float(self.num_M)
        obj_M = (log_pyM + log_pqz_M) / float(XM.size(0))

        obj = obj_R + obj_M

        loss = - obj

        return loss

    def get_pred_logits(self, x_new, XR, yR, n_samples=100):
        H_all = self.cond_trans(torch.cat([XR, x_new], 0))

        # get U
        pu_mean_all, pu_logscale_all = torch.split(self.p_u(H_all), self.dim_u, dim=1)
        pu = Normal(pu_mean_all, pu_logscale_all)

        qz_mean_R, qz_logscale_R = torch.split(self.q_z(H_all[0:XR.size(0)]), self.dim_z, 1)

        logits = float_tensor(x_new.size(0), self.dim_y, n_samples)
        for i in range(n_samples):
            u = pu.rsample()

            A = sample_bipartite(u[XR.size(0):], u[0:XR.size(0)], self.pairwise_g, training=False)

            cond_y_mean, cond_y_logscale = torch.split(self.trans_cond_y(one_hot(yR, n_classes=self.dim_y)), self.dim_z, 1)
            pz_mean_M = torch.mm(self.norm_graph(A), cond_y_mean + qz_mean_R)
            pz_logscale_M = torch.mm(self.norm_graph(A), cond_y_logscale + qz_logscale_R)
            pz = Normal(pz_mean_M, pz_logscale_M)

            z = pz.rsample()

            final_rep = z if not self.use_plus else torch.cat([z, u[XR.size(0):]], dim=1)

            logits[:, :, i] = F.log_softmax(self.output(final_rep), 1)

        logits = torch.logsumexp(logits, 2) - math.log(n_samples)

        return logits

    def predict(self, x_new, XR, yR, n_samples=100):
        logits = self.get_pred_logits(x_new, XR, yR, n_samples=n_samples)
        return torch.argmax(logits, 1)
