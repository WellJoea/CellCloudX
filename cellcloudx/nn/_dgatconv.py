########################################################
# DeepGAT for Node Classification
########################################################


# import libraries 

from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, OptTensor)
from torch import Tensor
import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros


import pandas as pd
import numpy as np
import torch
# coefficient = pd.read_csv("data/pubmed4.csv",header=None) #WTF
from ._normal import LipschitzNorm

class dGATConv(MessagePassing):
   
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.5,
                 add_self_loops: bool = True, bias: bool = True, norm = None, nlayers = 0,**kwargs,):
        super().__init__(aggr='add', node_dim=0,  **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.num_nodes, self.num_features, self.degrees = None, None, None
        self.norm = LipschitzNorm(scale_individually=False) if norm is None else norm
        self.layer = nlayers
        if isinstance(in_channels, int):
            self.lin_s = Linear(in_channels, heads * out_channels, bias=False)
            self.lin_d = self.lin_s
        else:
            self.lin_s = Linear(in_channels[0], heads * out_channels, False)
            self.lin_d = Linear(in_channels[1], heads * out_channels, False)
        self.att_s = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_d = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_s.weight)
        glorot(self.lin_d.weight)
        glorot(self.att_s)
        glorot(self.att_d)
        zeros(self.bias)


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None, **kargs):
        r"""

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        self.num_nodes, self.num_features = x.shape[0], x.shape[1]
        self.edge_index = edge_index
        

        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None

        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = x_r = self.lin_s(x).view(-1, H, C)    # Theta parameter: lin_l, lin_r

            alpha_l = (x_l * self.att_s).sum(dim=-1)
            alpha_r = (x_r * self.att_d).sum(dim=-1)
 

        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_s(x_l).view(-1, H, C)        # Theta parameter: lin_l, lin_r
            alpha_l = (x_l * self.att_s).sum(dim=-1)   
            if x_r is not None:
                x_r = self.lin_d(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_d).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)

        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), size=size)

        alpha = self._alpha
        self._alpha = None
        self.edge_index = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        if self.norm is not None:
            alpha = self.norm(x_j, att = (self.att_s, self.att_d), alpha = alpha, index = index)

        #layerwise
        # layers = self.layer
    
        # edge = self.edge_index
        # edge, _ = remove_self_loops(edge)
        # edge, _ = add_self_loops(edge, num_nodes=N)

        # lss = torch.tensor(np.array(coefficient.loc[layers,:])).repeat(N,1).\
        #                 reshape(N,N)[edge[0, :],edge[1, :]].float().cuda().reshape(np.shape(alpha)[0],1)
        #WTF SB

        lss =1
        alpha = F.leaky_relu(alpha*lss, self.negative_slope)


        alpha = softmax(alpha, index, ptr, size_i)
        

        self._alpha = alpha
        

        return x_j * F.dropout(alpha, p=self.dropout, training=self.training).unsqueeze(-1)
      


    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


# from normalizations.lipschitznorm import LipschitzNorm
# from normalizations.neighbornorm import NeighborNorm
# class DeepGAT(nn.Module):
#     def __init__(self, idim, hdim, odim, heads, num_layers, dropout, norm = None , ogb=False):

#         super(DeepGAT, self).__init__()
#         self.num_layers = num_layers
#         self.norm_name = norm
#         self.ogb= ogb
#         self.dropout = dropout
#         # Normalization methods
#         if self.norm_name == "pairnorm":
#             self.norm = None
#             self.pairnorm = PairNorm(scale_individually=False) 
#         elif self.norm_name == "pairnorm-si":
#             self.norm = None
#             self.pairnorm = PairNorm(scale_individually=True)
#         elif self.norm_name == "lipschitznorm":
#             self.norm = LipschitzNorm(scale_individually=False)
#         elif self.norm_name == "lipschitznorm-si":
#             self.norm = LipschitzNorm(scale_individually=True)
#         elif self.norm_name == "neighbornorm":
#             self.norm = NeighborNorm(scale = 0.1)
#         else:
#             self.norm = None

#         self.layers = nn.ModuleList([DeepGATConv(hdim, hdim, heads=heads, dropout=self.dropout
#         ,norm=self.norm,nlayers = 0)])
#         for i in range(1,num_layers):
#             self.layers.append(DeepGATConv(heads * hdim, hdim, heads=heads, dropout=self.dropout,norm=self.norm,nlayers = i))
        
#         self.lin = nn.Linear(idim, hdim)
#         self.fc1 = nn.Linear(heads * hdim, odim)


        
#     def forward(self, batched_data):
#         # x, edge_index = batched_data.x, batched_data.edge_index
#         if self.ogb:
#             x, edge_index = batched_data.x, batched_data.adj_t
#         else:
#             x, edge_index = batched_data.x, batched_data.edge_index



#         h = self.lin(x)
#         # h = x
#         for i, layer in enumerate(self.layers):
#             h = F.dropout(h, p=self.dropout
#             , training = self.training)
#             h = layer(h, edge_index)
#             if i < self.num_layers - 1:
#                 h = F.elu(h)
#             if self.norm_name == "pairnorm" or self.norm_name == "pairnorm-si":
#                 h = self.pairnorm(h)

            
#         # return F.log_softmax(h, dim=1)  
#         return F.log_softmax(self.fc1(h), dim=1),self.fc1(h)

