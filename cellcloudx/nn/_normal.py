import torch, torch.nn as nn
from torch_scatter import scatter, scatter_add, scatter_mean    
import numpy as np

class LipschitzNorm(nn.Module):
    def __init__(self, att_norm = 4, recenter = False, scale_individually = True, eps = 1e-12):
        super(LipschitzNorm, self).__init__()
        self.att_norm = att_norm
        self.eps = eps
        self.recenter = recenter
        self.scale_individually = scale_individually

    def forward(self, x, att, alpha, index):
        att_l, att_r = att
        
        if self.recenter:
            mean = scatter(src = x, index = index, dim=0, reduce='mean')
            x = x - mean


        norm_x = torch.norm(x, dim=-1) ** 2
        max_norm = scatter(src = norm_x, index = index, dim=0, reduce = 'max').view(-1,1)
        max_norm = torch.sqrt(max_norm[index] + norm_x)  # simulation of max_j ||x_j||^2 + ||x_i||^2

        
        # scaling_factor =  4 * norm_att , where att = [ att_l | att_r ]         
        if self.scale_individually == False:
            norm_att = self.att_norm * torch.norm(torch.cat((att_l, att_r), dim = -1))
        else:
            norm_att = self.att_norm * torch.norm(torch.cat((att_l, att_r), dim=-1), dim = -1)

        alpha = alpha / ( norm_att * max_norm + self.eps )
        return alpha

class NeighborNorm(nn.Module):
    def __init__(self, scale = 1, eps = 1e-12):
        super(NeighborNorm, self).__init__()
        self.scale = scale
        self.eps = eps

    def forward(self, x, edge_index):
        index_i, index_j = edge_index[0], edge_index[1]
        mean = scatter(src = x[index_j], index = index_i, dim=0, reduce='mean')
        reduced_x = (x - mean)**2

        # is it 'add' or 'mean'??
        # std = torch.sqrt(scatter(src = reduced_x[edge_index[0]], index = edge_index[1] , dim = dim, reduce = 'add')+ eps) # we add the eps for numerical instabilities (grad of sqrt at 0 gives nan)
        std = torch.sqrt(scatter(src = reduced_x[index_j], index = index_i, dim=0, reduce = 'mean')+ self.eps) # we add the eps for numerical instabilities (grad of sqrt at 0 gives nan)
        out =  self.scale * x / std
        infs = torch.isinf(out)
        out[infs] = x[infs]
        return out 

class PairNorm(nn.Module):
    r"""Applies PairNorm normalization layer over aa batch of nodes
    """
    def __init__(self, s = 1):
        super(PairNorm, self).__init__()
        self.s = s

    def forward(self, x, batch=None):

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x_c = x - scatter_mean(x, batch, dim=0)[batch]
        out = self.s * x_c / scatter_mean((x_c * x_c).sum(dim=-1, keepdim=True),
                             batch, dim=0).sqrt()[batch]

        return out

class Quadratic_LipschitzNorm(nn.Module):
    def __init__(self, eps = 1e-12):
        super(Quadratic_LipschitzNorm, self).__init__()
        self.eps = eps

    def forward(self, Q, K, V, alpha, index):
        
        Q_F = torch.norm(Q,dim=[0,2]) # frobenious norm of Q, multihead
        K_2 = torch.norm(K,dim = -1)
        V_2 = torch.norm(V,dim = -1)
        K_inf_2 = scatter(src= K_2, index = index, dim=0, reduce = 'max')
        V_inf_2 = scatter(src= V_2, index = index, dim=0, reduce = 'max')
        
        uv = Q_F * K_inf_2 
        uw = Q_F * V_inf_2
        vw = K_inf_2 * V_inf_2

        max_over_norms = torch.max(torch.stack([uv,uw,vw]),dim=0).values 
        alpha = alpha / (max_over_norms[index] + self.eps)
        # dsdas
        
        return alpha
