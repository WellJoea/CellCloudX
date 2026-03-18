import numpy as np
from tqdm import tqdm

import torch as th
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from scipy.sparse import issparse

from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter
import torch.nn as nn
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn import GCNConv, GATConv

import torch.nn.functional as F
# from ._gats import GATConv
from ._utilis import seed_torch, self_loop_check, loadAData, seed_torch


def GCNnn(adata,              
            basis='spatial',
            use_rep = 'X',
            add_embed = 'GATE',
            label = None,
            mask = None,
            mask_w = 1.0,

            add_self_loops=True,
            weight_temp = 1,
            edge_weight = None,
            validate= True,

            hidden_dims=[512, 30], n_epochs=500, lr=0.001, key_added='GATE',
            gradient_clipping=5.,  weight_decay=0.0001, verbose=True, 
            seed=491001, save_loss=False, save_reconstrction=False, 
            device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')):

    seed_torch(seed)
    data, new_order, num_classes = toLData(adata,  basis=basis, use_rep = use_rep,
                                        label = label, mask = mask, validate=validate )
    in_dim = data.x.size(1)
    
    # model = GATE(hidden_dims = [in_dim, *hidden_dims], num_classes=num_classes).to(device)
    model = GCNnet(hidden_dims = [in_dim, *hidden_dims], num_classes=num_classes).to(device)
    print(model)
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    # scheduler = th.optim.lr_scheduler.StepLR(optimizer, 
    #                                             step_size=step_size, 
    #                                             gamma=step_gamma)

    pbar = tqdm(range(n_epochs), total=n_epochs, colour='red', desc='GCN training')
    
    mean_loss = 0
    for char in pbar:
        model.train()
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data.x, data.edge_index)
        # loss_mse = F.mse_loss(data.x, out)
        # loss = loss_mse
        # loss = criterion(out[data.mask], data.y[data.mask]) 
        loss = F.cross_entropy(out[data.mask], data.y[data.mask])
        loss.backward()

        th.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()

        loss_prt = { 'loss_mes': f'{loss.item():.4f}'}
        mean_loss += loss.item()
        pbar.set_postfix(loss_prt)

    mean_loss /= (n_epochs*1)
    loss_prt = {'loss': f'{mean_loss :.6f}'}
    pbar.set_postfix(loss_prt)
    pbar.close()

    model.eval()
    with th.no_grad():
        out = model(data.x, data.edge_index)

        # prob = F.softmax(out, dim=1)
        prob = out
        pred_score, pred = th.max(prob, dim=1)

        test_correct = pred[data.mask] == data.y[data.mask] 
        test_acc = int(test_correct.sum()) / int(data.mask.sum()) 
        print(f'Test Accuracy: {test_acc:.4f}')

    adata.obsm[key_added] = out.cpu().detach().numpy()
    if not num_classes is None:
        adata.uns[f'{key_added}_label_order'] = new_order
        adata.uns[f'{key_added}_prob'] = prob.cpu().detach().numpy()
        adata.obs[f'{key_added}_pred'] = new_order[pred.cpu().detach().numpy()]
        adata.obs[f'{key_added}_predscore'] = pred_score.cpu().detach().numpy()

def GATEnn(adata,              
            basis='spatial',
            use_rep = 'X',
            add_embed = 'GATE',
            label = None,
            mask = None,
            mask_w = 1.0,

            add_self_loops=True,
            weight_temp = 1,
            edge_weight = None,
            validate= True,

            hidden_dims=[512, 30], n_epochs=500, lr=0.001, key_added='GATE',
            gradient_clipping=5.,  weight_decay=0.0001, verbose=True, 
            seed=491001, save_loss=False, save_reconstrction=False, 
            device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')):

    seed_torch(seed)
    data, new_order, num_classes = toLData(adata,  basis=basis, use_rep = use_rep,
                                        label = label, mask = mask, validate=validate )
    in_dim = data.x.size(1)
    
    model = GATE(hidden_dims = [in_dim, *hidden_dims], num_classes=num_classes).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # scheduler = th.optim.lr_scheduler.StepLR(optimizer, 
    #                                             step_size=step_size, 
    #                                             gamma=step_gamma)

    pbar = tqdm(range(n_epochs), total=n_epochs, colour='red', desc='GATE training')
    
    mean_loss = 0
    for char in pbar:
        model.train()
        optimizer.zero_grad()
        data = data.to(device)
        z, out, logits  = model(data.x, data.edge_index)
        loss_mse = F.mse_loss(data.x, out)
        loss = loss_mse

        if not num_classes is None:
            loss_crs = F.cross_entropy(logits[data.mask], data.y[data.mask])
            loss = loss_mse + mask_w * loss_crs
        loss.backward()

        th.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()

        if  num_classes is None:
            loss_prt = { 'loss_mes': f'{loss_mse.item():.4f}'}
        else:
            loss_prt = { 'loss_mes': f'{loss_mse.item():.4f}', 'loss_crs': f'{loss_crs.item():.4f}' }
  
        mean_loss += loss.item()
        pbar.set_postfix(loss_prt)
    mean_loss /= (n_epochs*1)
    loss_prt = {'loss': f'{mean_loss :.6f}'}
    pbar.set_postfix(loss_prt)
    pbar.close()

    model.eval()
    with th.no_grad():
        z, out, logits = model(data.x, data.edge_index)

        if not num_classes is None:
            prob = F.softmax(logits, dim=1)
            pred_score, pred = th.max(prob, dim=1)

            test_correct = pred[data.mask] == data.y[data.mask] 
            test_acc = int(test_correct.sum()) / int(data.mask.sum()) 
            print(f'Test Accuracy: {test_acc:.4f}')

    adata.obsm[key_added] = z.cpu().detach().numpy()
    adata.obsm['Rec'] = out.cpu().detach().numpy()

    if not num_classes is None:
        adata.uns[f'{key_added}_label_order'] = new_order
        adata.uns[f'{key_added}_logits'] = logits
        adata.uns[f'{key_added}_prob'] = prob.cpu().detach().numpy()
        adata.obs[f'{key_added}_pred'] = new_order[pred.cpu().detach().numpy()]
        adata.obs[f'{key_added}_predscore'] = pred_score.cpu().detach().numpy()


def label_binary(lables, mask):
    lables = pd.Series(np.array(lables))
    mask = np.array(mask)
    
    m_labels, m_order = pd.factorize(lables[mask])
    n_labels = np.ones(lables.shape[0]) * len(m_order)
    n_labels[mask] = m_labels
    
    n_labels = np.int64(n_labels)
    m_order = np.array(m_order)
    return n_labels, m_order

def toLData(adata,  basis='spatial', use_rep = 'X',
             label = None, mask = None, validate=True ):
    edge_infor = adata.uns[f'{basis}_edges']
    edge_index = th.LongTensor(edge_infor[['src_id', 'dst_id']].values.T.astype(np.int64))

    if use_rep == 'X':
        X = adata.X
    elif use_rep in adata.obsm.keys():
        X = adata.obsm[use_rep]
    Xs = th.FloatTensor(X.toarray() if issparse(X) else X)

    if label is None:
        data = Data(x=Xs, edge_index=edge_index,)
        new_order, n_class = None, None 
    else:
        if not mask is None:
            mask = adata.obs[mask]
        new_labels, new_order = label_binary( adata.obs[label], mask = mask)
        data = Data(x=Xs, edge_index=edge_index, y=th.asarray(new_labels), mask = th.asarray(mask))
        n_class = len(new_order)
        
    data.validate(raise_on_error=validate)
    return data, new_order, n_class

class GCNnet(torch.nn.Module):
    def __init__(self, hidden_dims, num_classes=None):
        super().__init__()
        [in_dim, num_hidden, out_dim] = hidden_dims
        self.conv1 = GCNConv(in_dim, num_hidden)
        self.conv2 = GCNConv(num_hidden, out_dim)
        self.classifier = GCNConv(out_dim, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.classifier(x, edge_index)
        return x

class GATE(th.nn.Module):
    def __init__(self, hidden_dims, num_classes=None):
        super(GATE, self).__init__()

        [in_dim, num_hidden, out_dim] = hidden_dims
        self.conv1 = GATConvl(in_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv2 = GATConvl(num_hidden, out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv3 = GATConvl(out_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv4 = GATConvl(num_hidden, in_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)

        if num_classes is not None:
            self.classifier = th.nn.Linear(out_dim, num_classes)
        else:
            self.classifier = None

    def forward(self, features, edge_index):

        h1 = F.elu(self.conv1(features, edge_index))
        h2 = self.conv2(h1, edge_index, attention=False)
        self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
        self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
        self.conv4.lin_src.data = self.conv1.lin_src.transpose(0, 1)
        self.conv4.lin_dst.data = self.conv1.lin_dst.transpose(0, 1)
        h3 = F.elu(self.conv3(h2, edge_index, attention=True,
                              tied_attention=self.conv1.attentions))
        h4 = self.conv4(h3, edge_index, attention=False)

        logits = None
        if self.classifier is not None:
            logits = self.classifier(h2)
            
        return h2, h4 , logits

class GATConvl(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, 
                 heads: int = 1, 
                 concat: bool = True,
                 negative_slope: float = 0.2, 
                 dropout: float = 0.0,
                 gat_temp: float = 1,
        
                 add_self_loops: bool = True, 
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.gat_temp = gat_temp


        self.lin_src = nn.Parameter(torch.zeros(size=(in_channels, out_channels)))
        nn.init.xavier_normal_(self.lin_src.data, gain=1.414)
        self.lin_dst = self.lin_src

        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))
        nn.init.xavier_normal_(self.att_src.data, gain=1.414)
        nn.init.xavier_normal_(self.att_dst.data, gain=1.414)

        self._alpha = None
        self.attentions = None

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None, attention=True, tied_attention = None):

        H, C = self.heads, self.out_channels
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = torch.mm(x, self.lin_src).view(-1, H, C)
        else:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        if not attention:
            return x[0].mean(dim=1)

        if tied_attention == None:
            alpha_src = (x_src * self.att_src).sum(dim=-1)
            alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
            alpha = (alpha_src, alpha_dst)
            self.attentions = alpha
        else:
            alpha = tied_attention


        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        # if self.bias is not None:
        #     out += self.bias

        if isinstance(return_attention_weights, bool):
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
        # alpha = torch.sigmoid(alpha)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha 
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

