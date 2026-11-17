#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
***********************************************************
* @File    : _GRAC.py
* @Author  : Wei Zhou                                     *
* @Date    : 2023/12/01 05:43:30                          *
* @E-mail  : welljoea@gmail.com                           *
* @Version : --                                           *
* You are using the program scripted by Wei Zhou.         *
* Please give me feedback if you find any problems.       *
* Please let me know and acknowledge in your publication. *
* Thank you!                                              *
* Best wishes!                                            *
***********************************************************
'''

import torch
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from torch_geometric.data import Data
import torch_geometric.utils as tgu
from torch_geometric.utils import remove_self_loops
from tqdm import tqdm
import torch.nn.functional as F

from ..nn._GATEconv import GATEConv
# from ..integration.STAligner._STALIGNER import STAligner as GATEConv
# from ..integration.STAGATE_pyG._STAGATE import STAGATE1 as GATEConv
from ..nn._utilis import seed_torch, self_loop_check, loadAData
from ..nn._loss import loss_recon, loss_structure, loss_mse, loss_gae,loss_contrast,loss_soft_cosin

from ..tools._sswnn import SSWNN, sswnn_match
from ..tools._spatial_edges import spatial_edges


class GRAC():
    def __init__(self, save_model=True, save_latent=True, save_x=True,
                  save_alpha=True, shuffle=False):
        self.save_model = save_model
        self.save_latent = save_latent
        self.save_x = save_x
        self.save_alpha = save_alpha

        self.sswnn_match = sswnn_match
        self.loadAData = loadAData
        self.shuffle = shuffle

    def train(self,
              adata,
              basis='spatial',
              edge_key = None,
              add_embed = 'glatent',
              add_key = 'GRAC',
              use_rep = 'X',
              groupby = None,
              groupadd = None,
              gconvs ='gatv3',
              edge_weight = None,
              do_intergration = True,
              use_cross_edges = False,
              hidden_dims=[512, 48], 
              Heads=1,

              Concats=False,
              validate = True,
              add_self_loops = True,
              share_weights = True,
              weight_norml2=False,
              residual_norml2 =False,
              residual=False,
              bias=False,

              tied_attr={},
              layer_attr={},

              lr=5e-4, 
              weight_decay = 1e-4,
              gradient_clipping = 5.,
              n_epochs=500,
              e_epochs=0,
              u_epoch=0,
              Lambda = 0,
              Beta = 0,
              Gamma = 0,
              gat_temp = 1,
              loss_temp = 1,
              weight_temp = 1,
              
              norm_latent = False,
              device=None,
              seed=491001,

              root =None,
              regist_pair = None,
              full_pair=False,
              step=1,

              use_dpca = False,
              dpca_npca = 50,
              ckd_method ='hnsw',
              ckdsp_method = None,
              m_neighbor= 6,
              e_neighbor = 30,
              s_neighbor = 30,
              o_neighbor = 30,
              lower = 0.01,
              upper = 0.9,
              line_width=0.5,
              line_alpha=0.5,
              line_sample=None,
              drawmatch = False,
              point_size=1,
              n_nn = 10,

              **kargs):
        print('Computing GRAL...')
        seed_torch(seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)

        lDa = self.loadAData(adata,
                                groupby=groupby,
                                basis=basis,
                                use_rep=use_rep,
                                add_self_loops=add_self_loops,
                                edge_key=edge_key,
                                weight_temp=weight_temp,
                                weights=edge_weight, 
                                validate=validate)
        loader,_ = lDa.get_sDatas(shuffle=self.shuffle)
        in_dim = loader.dataset[0].x.size(1)
        model = GATEConv(
                    [in_dim] + hidden_dims,
                    Heads=Heads,
                    Concats=Concats,
                    bias=bias,
                    share_weights = share_weights,
                    gconvs=gconvs,
                    tied_attr=tied_attr,
                    gat_temp=gat_temp,
                    layer_attr=layer_attr,
                    weight_norml2=weight_norml2,
                    residual_norml2=residual_norml2,
                    residual=residual,
                    **kargs).to(device)

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=lr,
                                     weight_decay=weight_decay)

        pbar = tqdm(range(n_epochs), total=n_epochs, colour='red', desc='embedding  ')
        mean_loss = 0
        for char in pbar:
            for batch in loader:
                model.train()
                optimizer.zero_grad()
                batch = batch.to(device)
                H, X_ = model(batch.x, batch.edge_index,
                               edge_weight=batch.edge_weight) #, edge_attr = None

                loss = model.loss_gae(batch.x, X_, H, batch.edge_index,
                                    #  edge_weight=batch.edge_weight, 
                                       Lambda=Lambda, Gamma=Gamma)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()
                mean_loss += loss.item()
            pbar.set_postfix(loss = f'{loss.item():.8f}')
        mean_loss /= (n_epochs*len(loader))
        pbar.set_postfix(loss = f'{mean_loss :.8f}')
        pbar.close()

        with torch.no_grad():
            loader, cellidx = lDa.get_sDatas(shuffle=False)
            cellidx = np.concatenate(cellidx, axis=0)
            cellidx = pd.Series(np.arange(cellidx.shape[0]), index=cellidx)
            cellord = cellidx.loc[adata.obs_names].values
            print(cellord.shape, np.arange(adata.shape[0]).shape, 
                 (cellord- np.arange(adata.shape[0])).sum())
        
            Hs, Xs_  = [], []
            for batch in loader:
                batch = batch.to(device)
                H, X_ = model(batch.x, batch.edge_index, edge_weight=batch.edge_weight) #, edge_attr = None
                Hs.append(H.cpu().detach())
                Xs_.append(X_.cpu().detach())
            Hs = torch.cat(Hs, dim=0).numpy()[cellord]
            Xs_ = torch.cat(Xs_, dim=0)
        adata.obsm[add_embed] = Hs.copy()

        if do_intergration and ( not groupby is None):
            assert (adata.obs[groupby].unique().shape[0]>1) and (e_epochs>0)
            if use_cross_edges:
                mean_loss = 0
                pbar = tqdm(range(e_epochs), total=e_epochs, colour='blue', desc='integrating')
                for epoch in pbar:
                    if epoch % u_epoch == 0:
                        loader_cross, _ = lDa.get_pDatas(get_nev=True, keep_self=False, seed=seed+epoch, 
                                                         undirect=True, shuffle=self.shuffle)
                    for batch in loader_cross:
                        model.train()
                        optimizer.zero_grad()
                        batch = batch.to(device)
                        H, X_ = model(batch.x, batch.edge_index,
                                    edge_weight=batch.edge_weight) #, edge_attr = None

                        loss = model.loss_gae(batch.x, X_, H, batch.edge_index,
                                            #  edge_weight=batch.edge_weight, 
                                              Lambda=Lambda, Gamma=Gamma) + \
                                model.loss_contrast(H, batch.edge_index, batch.nev_index,
                                                        temperature=loss_temp, 
                                                        edge_weight = 1,
                                                        Beta= Beta)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                        optimizer.step()
                    pbar.set_postfix(loss = f'{loss.item():.8f}')
                mean_loss /= (e_epochs*len(loader_cross))
                pbar.set_postfix(loss = f'{mean_loss :.8f}')
                pbar.close()

            else:
                if basis in adata.obsm.keys():
                    position = adata.obsm[basis]
                else:
                    position = None
                u_epoch = u_epoch or 100
                pos_edges = [ torch.LongTensor(cellid[i][batch.edge_index]) for i, batch in enumerate(loader) ]
                pos_edges = torch.cat(pos_edges, dim=1).type(torch.LongTensor)
                Xr = torch.cat( [batch.x for batch in loader ], dim=0).type(torch.FloatTensor)

                if not edge_weight is None:
                    edge_weight = torch.cat( [batch.edge_weight for batch in loader ], dim=0).type(torch.FloatTensor)
                    edge_weight = edge_weight.to(device)
                else:
                    edge_weight = None

                Xr = Xr.to(device)
                pos_edges = pos_edges.to(device)

                pbar = tqdm(range(e_epochs), total=e_epochs, colour='blue',  desc='integrating')
                for epoch in pbar:

                    if epoch % u_epoch == 0:
                        if groupadd is None:
                            groups = adata.obs[groupby]
                            iroot = root[0]
                        else:
                            if (epoch % (u_epoch * 2) == 0):
                                groups = adata.obs[groupby]
                                iroot = root[0]
                            else:
                                groups = adata.obs[groupadd]
                                iroot = root[1]

                        hData = Hs.detach().cpu()
                        Hnorm = F.normalize(torch.FloatTensor(hData), dim=1).numpy() if norm_latent else hData.numpy()
                        pmnns, ssnn_scr, rqsid = self.sswnn_match(
                                                    Hnorm, groups, position=position,
                                                    ckd_method=ckd_method, 
                                                    sp_method = ckdsp_method,
                                                    use_dpca = use_dpca,
                                                    dpca_npca = dpca_npca,
                                                    root=iroot, regist_pair=regist_pair,
                                                    full_pair=full_pair, step=step,
                                                    m_neighbor=m_neighbor, 
                                                    e_neighbor =e_neighbor, 
                                                    s_neighbor =s_neighbor,
                                                    o_neighbor =o_neighbor,
                                                    lower = lower, upper = upper,
                                                    point_size=point_size,
                                                    drawmatch=drawmatch,  line_sample=line_sample,
                                                    line_width=line_width, line_alpha=line_alpha)
                        pmnns = torch.tensor(pmnns, dtype=torch.long)
                        ssnn_scr = torch.tensor(ssnn_scr, dtype=torch.float32)

                        nmnns = GRAC.nnnself(Hnorm, 
                                                groups, 
                                                root=0,
                                                kns=n_nn, 
                                                seed = [epoch, seed], 
                                                exclude_edge_index = list(map(tuple, pos_edges.cpu().detach().numpy().T)))

                        # print('pos.mnn: ', pmnns.shape, 'nev.sample: ', nmnns.shape)
                        # edge_mnn = edge_index
                        # edge_mnn = edge_mnn.to(device)

                        pmnns, nmnns, ssnn_scr =  pmnns.to(device), nmnns.to(device), ssnn_scr.to(device)

                    model.train()
                    optimizer.zero_grad()
                    Hs, Xs_ = model(Xr, pos_edges, edge_weight = edge_weight)

                    loss = model.loss_gae(Xr, Xs_, Hs, pos_edges, Lambda=Lambda, Gamma=Gamma) + \
                        model.loss_contrast(Hs, pmnns, nmnns,
                                                temperature=loss_temp, 
                                                edge_weight = 1,
                                                Beta= Beta)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                    optimizer.step()

                    pbar.set_postfix(loss = f'{loss.item():.10f}; sswnn pairs:{pmnns.size(1)}')
                    # if epoch % p_epoch == 0 or epoch==e_epochs-1:
                    #     print(f'total loss: {loss}')
                pbar.close()

        # eval
        model.eval()
        Hs, Xs_  = [], []
        # loader, cellidx = lDa.get_sDatas(shuffle=False)
        # cellidx = np.concatenate(cellidx, axis=0)
        # cellidx = pd.Series(np.arange(cellidx.shape[0]), index=cellidx)
        # cellord = cellidx.loc[adata.obs_names].values
        
        for batch in loader:
            batch_data = batch.cpu()
            H, X_ = model.cpu()(batch_data.x, batch_data.edge_index, edge_weight=batch.edge_weight ) #, edge_attr = None
            Hs.append(H.detach())
            Xs_.append(X_.detach())
        Hs = torch.cat(Hs, dim=0)
        Xs_ = torch.cat(Xs_, axis=0)

        self.Lambda = Lambda
        print(f'finished: added to `.obsm["{add_key}"]`')
        print(f'          added to `.layers["{add_key}"]`')
        adata.obsm[add_key] = Hs.numpy()[cellord]
        adata.layers[add_key] = Xs_.numpy()[cellord]

        if self.save_model:
            self.model = model
        else:
            return model

    @torch.no_grad()
    def infer(self, data, edge_attr=None,  model = None, device=None):
        model = self.model if model is None else model
        device = next(model.parameters()).device if device is None else device
        data = data.to(device)
        edge_attr = edge_attr.to(device) if edge_attr is not None else None

        model.eval()
        H, X_ = model(data.x, data.edge_index, edge_attr=edge_attr)
        # loss = model.loss_gae(X, X_, H, edge_index)
        # print(f'infer loss: {loss}')
        return H, X_

    @staticmethod
    def mnnpairs(hData, groups, root=None, regist_pair=None, full_pair=False, 
                 step=1, keep_self=True,
                   knn_method='hnsw',cross=True, edge_nn=15, set_ef=50, **kargs):
        rqid = np.unique(groups)
        if len(rqid) ==1 :
            full_pair = False
            root=0
            regist_pair = [(0,0)]
            keep_self=True,
    
        mnnk = SSWNN()
        mnnk.build(hData, 
                groups, 
                hData=None,
                method=knn_method,
                root=root,
                regist_pair=regist_pair,
                step=step,
                full_pair=full_pair,
                keep_self=keep_self)
        mnn_idx = mnnk.pairmnn(knn=edge_nn, cross=cross, 
                               return_dist=False,
                               set_ef=set_ef, **kargs)
        # import matplotlib.pyplot as plt
        # radiu_trim = trip_edges(mnn_idx[:,2], filter = 'std2q')
        # plt.hist(mnn_idx[:,2], bins=100)
        # plt.axvline(radiu_trim, color='black', label=f'radius: {radiu_trim :.3f}')
        # plt.show()
        # mnn_idx = mnn_idx[mnn_idx[:,2]<=radiu_trim,:]
        mnn_idx = torch.tensor(mnn_idx[:,:2].T, dtype=torch.long)

        if len(rqid) ==1 :
            mnn_idx, _ = remove_self_loops(torch.tensor(mnn_idx, dtype=torch.long))
        return mnn_idx

    @staticmethod
    def nnnself(hData, groups, root=0, kns=10, seed = 491001, exclude_edge_index = None):
        nnnk = SSWNN()
        nnnk.build(hData, groups,
                    splocs=None,
                    root=root)
        nnn_idx = nnnk.negative_self(kns=kns+1, seed = seed, 
                                     exclude_edge_index = exclude_edge_index)
        nnn_idx, _ = remove_self_loops(torch.tensor(nnn_idx.T, dtype=torch.long))
        return nnn_idx

    @staticmethod
    def nnnhself(hData, groups, root=0, kns=None, seed = 491001, exclude_edge_index = None):
        nnnk = nnalign()
        nnnk.build(hData, 
                groups, 
                hData=None,
                root=root)
        nnn_idx = nnnk.negative_hself(exclude_edge_index, kns=kns, seed = seed)
        nnn_idx, _ = remove_self_loops(torch.tensor(nnn_idx, dtype=torch.long))
        return nnn_idx