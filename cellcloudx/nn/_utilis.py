import numpy as np
import pandas as pd
import random
import os

from scipy.sparse import issparse
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import torch
from torch_geometric.utils import scatter, segment
from torch_geometric.utils.num_nodes import maybe_num_nodes
from typing import Optional, Tuple, Union, Any
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.utils as tgu
from torch_geometric.utils import remove_self_loops

from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
    torch_sparse,
)

import numpy as np

from scipy.sparse import issparse, vstack
import torch 
from torch_geometric.data import Data
from scipy.sparse import issparse
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class loadAData():
    def __init__(self, adata, groupby=None, 
                 use_rep = 'X', 
                 basis='spatial', 
                 edge_key = None, 
                 add_self_loops=True,
                 weight_temp = 1,
                 batch_size = 1,
                 weights =None,
                 validate=True):
        if use_rep == 'X':
            Xs = adata.X
        elif use_rep == 'raw':
            Xs = adata.raw.X
        else :
            Xs = adata.obsm[use_rep]

        assert adata.shape[0] == adata.obs_names.unique().shape[0], 'adata.obs_names is not unique'
        edge_key = f'{basis}_edges' if edge_key is None else edge_key
        assert edge_key in adata.uns.keys(), f'{edge_key} not in adata.uns.keys()'
        self.adata = adata
        self.Xs = Xs
        self.edge_data = adata.uns[edge_key]
        self.batch_size = batch_size
        self.validate = validate
        self.add_self_loops = add_self_loops
        self.weight_temp = weight_temp
        self.groupby=groupby
        self.groups = None
        self.cell_order = adata.obs_names.copy().values

        self.weights = [weights] if isinstance(weights, str) else weights

        if not groupby is None:
            try:
                groups = adata.obs[groupby].cat.remove_unused_categories().cat.categories
            except:
                groups = adata.obs[groupby].unique()
            self.groups = groups

    def _to_Data(self, X, src, dst, edges_weight=None, num_nodes=None):
        if num_nodes is None:
            num_nodes = X.shape[0]
        else:
            assert num_nodes == X.shape[0]

        edge_index = torch.LongTensor(np.vstack([src, dst]).astype(np.int64))
        if not edges_weight is None:
            edges_weight  = torch.FloatTensor(edges_weight).pow(self.weight_temp)

        edge_index, edges_weight = self_loop_check(edge_index, edges_weight,
                                                    add_self_loops=self.add_self_loops,
                                                    num_nodes=num_nodes )
        Xs = torch.FloatTensor(X.toarray() if issparse(X) else X)
        data = Data(x=Xs, 
                    edge_index=edge_index,
                    edge_weight = (None if edges_weight is None else edges_weight[:,0]),
        )
        data.validate(raise_on_error=self.validate)
        return data

    def to_Data(self, iX, iedge_data, src_int, dsr_int):
        src = src_int[iedge_data['src']]
        dst = dsr_int[iedge_data['dst']]

        edges_weight = None if self.weights is None else iedge_data[self.weights].values
        return self._to_Data(iX, src, dst, edges_weight=edges_weight)

    def to_listData(self, Datalist, drop_data_cins=True, shuffle=False):
        loader = DataLoader(Datalist, batch_size=self.batch_size, shuffle=shuffle)
        cellids =[]
        for batch in loader:
            cellids.append(batch.cellins)
            if drop_data_cins:
                del batch.cellins #error
        return loader, cellids

    def get_sData(self, iGroup):
        idins = self.adata.obs[self.groupby] == iGroup
        icells= self.cell_order[idins]
        icell_map = pd.Series(np.arange(icells.shape[0]), index=icells).astype(np.int64)
        ix = self.Xs[idins,:]
        iedge_data = self.edge_data[((self.edge_data['src_name'] == iGroup) & (self.edge_data['dst_name'] == iGroup))]

        iData = self.to_Data(ix, iedge_data, icell_map, icell_map)
        iData.group = iGroup
        return [ iData, icells]

    def _get_nevEdge(self, sGroup, dGroup, kns=10, pos_edge_index =None, seed = 200504):
        sSize = (self.adata.obs[self.groupby] == sGroup).sum()
        dSize = (self.adata.obs[self.groupby] == dGroup).sum()

        if pos_edge_index is None:
            assert not kns is None
            if sGroup != sGroup:
                edge_size = kns* (sSize + dSize) #directed-graph
            else:
                edge_size = kns* sSize
        else:
            edge_size = pos_edge_index.shape[1] *2 

        rng = np.random.default_rng(seed=[1, seed])
        snev = rng.choice(sSize, size=edge_size, replace=True, shuffle=False)
        rng = np.random.default_rng(seed=[2, seed])
        dnev = rng.choice(dSize, size=edge_size, replace=True, shuffle=False)

        if sGroup != sGroup:
            dnev = dnev + sSize
        nev_edge = set(zip(snev, dnev))

        if not pos_edge_index is None:
            pos_edge = set(zip(pos_edge_index[0], pos_edge_index[1]))
            nev_edge = (nev_edge - pos_edge)
            nev_edge = np.array(list(nev_edge)) 
            ind = np.lexsort((nev_edge[:,1], nev_edge[:,0]))  # set is disordered
            nev_edge = nev_edge[ind]
    
            if len(nev_edge)>len(pos_edge):
                rng = np.random.default_rng(seed=[3, seed])
                nev_edge = rng.choice(nev_edge, size=len(pos_edge), replace=False, shuffle=False)
        else:
            nev_edge = np.array(list(nev_edge))

        return nev_edge.T

    def get_nevEdge(self, sGroup, dGroup, edge_size=None, 
                    keep_self=False, pos_edge_index =None, seed = 200504):
        sSize = (self.adata.obs[self.groupby] == sGroup).sum()
        dSize = (self.adata.obs[self.groupby] == dGroup).sum()
        edge_size = edge_size if edge_size else pos_edge_index.shape[1] *2
    
        if keep_self:
            all_edges = sSize + dSize
            rng = np.random.default_rng(seed=[1, seed])
            snev = rng.choice(all_edges, size=edge_size, replace=True, shuffle=False)
            rng = np.random.default_rng(seed=[2, seed])
            dnev = rng.choice(all_edges, size=edge_size, replace=True, shuffle=False)
            nev_edge = set(zip(snev, dnev))

        else:
            rng = np.random.default_rng(seed=[1, seed])
            snev = rng.choice(sSize, size=edge_size, replace=True, shuffle=False)
            rng = np.random.default_rng(seed=[2, seed])
            dnev = rng.choice(dSize, size=edge_size, replace=True, shuffle=False)

            dnev = dnev + sSize
            nev_edge = set(zip(snev, dnev))

        if not pos_edge_index is None:
            pos_edge = set(zip(pos_edge_index[0], pos_edge_index[1]))
            nev_edge = (nev_edge - pos_edge)
            nev_edge = np.array(list(nev_edge)) 
            ind = np.lexsort((nev_edge[:,1], nev_edge[:,0]))  # set is disordered
            nev_edge = nev_edge[ind]
    
            if len(nev_edge)>len(pos_edge):
                rng = np.random.default_rng(seed=[3, seed])
                nev_edge = rng.choice(nev_edge, size=len(pos_edge), replace=False, shuffle=False)
        else:
            nev_edge = np.array(list(nev_edge))

        return nev_edge.T

    def get_pData(self, sGroup, dGroup, undirect=True, 
                  keep_self = False,
                  get_nev = False, seed = 200504):
        idins = (self.adata.obs[self.groupby] ==sGroup).values
        idind = (self.adata.obs[self.groupby] ==dGroup).values
        icells= np.concatenate( [self.cell_order[idins], self.cell_order[idind]]) 

        icell_map = pd.Series(np.arange(icells.shape[0]), index=icells).astype(np.int64)
        if issparse(self.Xs):
            ix = vstack([self.Xs[idins,:], self.Xs[idind,:]])
        else:
            ix = np.concatenate([self.Xs[idins,:], self.Xs[idind,:]], axis=0)

        iedge_idx = ((self.edge_data['src_name'] == sGroup) & (self.edge_data['dst_name'] == dGroup))
        if undirect:
            iedge_idx = iedge_idx  | ((self.edge_data['src_name'] == dGroup) & (self.edge_data['dst_name'] == sGroup))
        if keep_self:
            iedge_idx = iedge_idx  | ((self.edge_data['src_name'] == sGroup) & (self.edge_data['dst_name'] == sGroup))
            iedge_idx = iedge_idx  | ((self.edge_data['src_name'] == dGroup) & (self.edge_data['dst_name'] == dGroup))
    
        iedge_data = self.edge_data[(iedge_idx)]
        iData = self.to_Data(ix, iedge_data, icell_map, icell_map)
        iData.group = (sGroup, dGroup)

        if get_nev:
            nev_index = self.get_nevEdge(sGroup, dGroup,
                                         keep_self = keep_self,
                                            pos_edge_index = iData.edge_index.numpy(),
                                            seed = seed)
            iData.nev_index = torch.tensor(nev_index, dtype=torch.long)
        return [ iData, icells]

    def get_sDatas(self, shuffle=False):
        Datalist = []
        cellids = []
        if self.groups is None:
            cell_map = pd.Series(np.arange(self.adata.shape[0]), index=self.cell_order).astype(np.int64)
            iData = self.to_Data(self.Xs, self.edge_data, cell_map, cell_map)
            iData.cell = self.cell_order
            Datalist = [iData]
            cellids = [self.cell_order]
        else:
            for iGroup in self.groups:
                iData, icells = self.get_sData(iGroup)
                Datalist.append(iData)
                cellids.append(icells)

        loader = DataLoader(Datalist, batch_size=self.batch_size, shuffle=shuffle)
        if shuffle:
            return [loader, None]
        else:
            cellids = cellids
            return [loader, cellids]

    def get_pDatas(self, undirect=True, keep_self=False, pGroups=None,
                   get_nev = False, seed=491001, shuffle=False):
        assert not self.groupby is None
        if pGroups is None:
            pGroups = set(zip(self.edge_data['src_name'], self.edge_data['dst_name']))
            pGroups = [ each for each in pGroups if each[0] != each[1] ]
        assert len(pGroups) > 0, 'No pairs of groups'

        Datalist = []
        cellids = []
        for i,pgroup in enumerate(pGroups):
            iData, icells = self.get_pData(*pgroup, undirect=undirect, get_nev=get_nev, keep_self=keep_self, seed=i+seed)
            Datalist.append(iData)
            cellids.append(icells)

        loader = DataLoader(Datalist, batch_size=self.batch_size, shuffle=shuffle)
        if shuffle:
            return [loader, None]
        else:
            cellids = cellids
            return [loader, cellids]

def self_loop_check(edge_index, edge_weight=None, num_nodes=None, fill_value='mean', add_self_loops=True):
    if add_self_loops:
        edge_index, edge_weight = tgu.remove_self_loops( edge_index, edge_weight)
        edge_index, edge_weight = tgu.add_self_loops(edge_index,
                                                edge_weight,
                                                fill_value=fill_value,
                                                num_nodes=num_nodes)
    else:
        edge_index, edge_weight = tgu.remove_self_loops( edge_index, edge_weight)
    return edge_index, edge_weight

def Activation(active, negative_slope=0.01):
    if active is None:
        return nn.Identity()
    elif active == 'relu':
        return nn.ReLU()
    elif active in ['leaky_relu', 'lrelu']:
        return nn.LeakyReLU(negative_slope)
    elif active == 'elu':
        return nn.ELU()
    elif active == 'selu':
        return nn.SELU()
    elif active == 'tanh':
        return torch.tanh
    elif active == 'sigmoid':
        return nn.Sigmoid()
    elif active == 'softmax':
        return nn.Softmax(dim=1)
    elif active == 'softplus':
        return nn.Softplus()
    elif active == 'srelu':
        return shiftedReLU.apply
    elif active == 'linear':
        return nn.Identity()
    else:
        return active

def seed_torch(seed=200504):
    if not seed is None:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.mps.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.enabled = False
        # try:
        #     torch.use_deterministic_algorithms(True)
        # except:
        #     pass

def glorot(value: Any, gain: float = 1.):
    if isinstance(value, Tensor):
        stdv = np.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-gain*stdv, gain*stdv)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            glorot(v, gain=gain)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            glorot(v, gain=gain)

def ssoftmax(
    src: Tensor,
    index: Optional[Tensor] = None,
    ptr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
    sift : Optional[float] = 0.,
    temp : Optional[float] = 1.,
    dim: int = 0,
) -> Tensor:

    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor, optional): The indices of elements for applying the
            softmax. (default: :obj:`None`)
        ptr (LongTensor, optional): If given, computes the softmax based on
            sorted inputs in CSR representation. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        dim (int, optional): The dimension in which to normalize.
            (default: :obj:`0`)

    :rtype: :class:`Tensor`

    Examples:
        >>> src = torch.tensor([1., 1., 1., 1.])
        >>> index = torch.tensor([0, 0, 1, 2])
        >>> ptr = torch.tensor([0, 2, 3, 4])
        >>> softmax(src, index)
        tensor([0.5000, 0.5000, 1.0000, 1.0000])

        >>> softmax(src, None, ptr)
        tensor([0.5000, 0.5000, 1.0000, 1.0000])

        >>> src = torch.randn(4, 4)
        >>> ptr = torch.tensor([0, 4])
        >>> softmax(src, index, dim=-1)
        tensor([[0.7404, 0.2596, 1.0000, 1.0000],
                [0.1702, 0.8298, 1.0000, 1.0000],
                [0.7607, 0.2393, 1.0000, 1.0000],
                [0.8062, 0.1938, 1.0000, 1.0000]])
    """
    if ptr is not None:
        dim = dim + src.dim() if dim < 0 else dim
        size = ([1] * dim) + [-1]
        count = ptr[1:] - ptr[:-1]
        ptr = ptr.view(size)
        src_max = segment(src.detach(), ptr, reduce='max')
        src_max = src_max.repeat_interleave(count, dim=dim)
        out = ((src - src_max)/temp).exp()
        out_sum = segment(out, ptr, reduce='sum') + 1e-16
        out_sum = out_sum.repeat_interleave(count, dim=dim)
    elif index is not None:
        N = maybe_num_nodes(index, num_nodes)
        src_max = scatter(src.detach(), index, dim, dim_size=N, reduce='max')
        out = (src - src_max.index_select(dim, index))/temp
        out = out.exp()
        out_sum = scatter(out, index, dim, dim_size=N, reduce='sum') + 1e-16
        out_sum = out_sum.index_select(dim, index)
    else:
        raise NotImplementedError

    return out / (sift + out_sum)

class shiftedReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, ):
        out = inp
        out[out <= 0.05] = 0
        ctx.save_for_backward(out)
        # out = torch.zeros_like(inp) #.cuda()
        # out[inp <= 0.05] = 0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inp, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[inp <= 0 ] = 0
        return grad_input
