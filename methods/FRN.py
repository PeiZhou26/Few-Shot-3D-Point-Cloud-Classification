# This code is modified from https://github.com/Tsingularity/FRN
import os
import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from PIL import Image


class FRN(MetaTemplate):
    def __init__(self, model_func, n_way, n_support):
        super(FRN, self).__init__(model_func, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.r = nn.Parameter(torch.zeros(2),requires_grad=True)

    def get_recon_dist(self,query,support,alpha,beta,Woodbury=True):
    # query: way*query_shot*resolution, d
    # support: way, shot*resolution , d
        # correspond to kr/d in the paper
        reg = support.size(1)/support.size(2)
        # correspond to lambda in the paper
        lam = 0.1*alpha.exp()+1e-6
        # correspond to gamma in the paper
        rho = beta.exp()
        st = support.permute(0,2,1) # way, d, shot*resolution
        # correspond to Equation 8 in the paper
        sst = support.matmul(st) # way, shot*resolution, shot*resolution
        m_inv = (sst+torch.eye(sst.size(-1)).to(sst.device).unsqueeze(0).mul(lam)).inverse() # way, shot*resolution, shot*resolutionsf 
        hat = st.matmul(m_inv).matmul(support) # way, d, d
        Q_bar = query.matmul(hat).mul(rho) # way, way*query_shot*resolution, d
        # dist = (Q_bar-query.unsqueeze(0)).pow(2).sum(2).permute(1,0) # way*query_shot*resolution, way
        dist = (Q_bar-query.unsqueeze(0)).pow(2).view(self.n_way*self.n_query,self.feat_dim,self.n_way).sum(1)
        return dist


    def set_forward(self, x, is_feature=False):
        alpha = self.r[0]
        beta = self.r[1]
        z_support, z_query = self.parse_feature(x, is_feature)
        z_support = z_support.contiguous().view(self.n_way,self.feat_dim*self.n_support,1)
        z_query = z_query.contiguous().view(self.n_way*self.feat_dim*self.n_query,1)
        recon_dist = self.get_recon_dist(query=z_query,support=z_support,alpha=alpha,beta=beta)
        scores = recon_dist.neg()
        return scores

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())
        scores = self.set_forward(x)
        return self.loss_fn(scores, y_query)


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)