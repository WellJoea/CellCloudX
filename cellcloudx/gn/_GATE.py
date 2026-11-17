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
import torch.nn.functional as F

import numpy as np
import pandas as pd
from scipy.sparse import issparse

from ._GATEconv import GATEConv, GACEConv
from ._utilis import seed_torch, self_loop_check, loadAData


from tqdm import tqdm
torch.set_default_dtype(torch.float32)

class GATE():
    def __init__(self, save_model=True, save_latent=True, save_x=True, 
                 weight_attent=True, save_alpha=True,edge_weight = None,
                 weight_temp = 1, add_self_loops = True, validate = True,
                Lambda = 1, 
                Gamma = 0,):
        self.save_model = save_model
        self.save_latent = save_latent
        self.save_x = save_x
        self.save_alpha = save_alpha
        self.weight_attent = weight_attent
        self.add_self_loops = add_self_loops
        self.weight_temp =  weight_temp
        self.validate = validate
        self.edge_weight = edge_weight
        self.Lambda = Lambda
        self.Gamma = Gamma

    def train(self,
              adata,
              basis='spatial',
              use_rep = 'X',
              add_embed = 'GATE',

              groupby = None,
              #images = None,
              gconvs=['gatv3','gatv3'],

              gcn2_alpha = 0.5,
              hidden_dims=[512, 48],
              Heads=1,
              Concats=False,

              share_weights = True,
   
              bias=False,
              shuffle=False,

              lr=1e-3,
              weight_decay = 1e-4,
              gradient_clipping = 5,
              n_epochs=500,
              p_epoch=None,

              gat_temp = 1,

              use_scheduler=False,
              step_size=250,
              step_gamma=0.5,
              device=None,
              seed=491001,
              verbose=1,
              **kargs):
        seed_torch(seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
        p_epoch = p_epoch or max(n_epochs/10, 10)

        lDa = loadAData(adata,
                                groupby=groupby,
                                basis=basis,
                                use_rep=use_rep,
                                add_self_loops=self.add_self_loops,
                                weight_temp=self.weight_temp,
                                weights=self.edge_weight, 
                                validate=self.validate)
        loader,_ = lDa.get_sDatas(shuffle=shuffle)

        in_dim = loader.dataset[0].x.size(1)
        model = GACEConv(
                    [in_dim] + hidden_dims,
                    Heads=Heads,
                    Concats=Concats,
                    bias=bias,
                    share_weights = share_weights,
                    gconvs=gconvs,
                    gat_temp=gat_temp,
                    gcn2_alpha=gcn2_alpha,
                    **kargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=lr,
                                     weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=step_size, 
                                                    gamma=step_gamma)

        pbar = tqdm(range(n_epochs), total=n_epochs, colour='red', desc='graph embedding')
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
                                       Lambda=self.Lambda, Gamma=self.Gamma)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()
                loss_prt = { 'loss': f'{loss.item():.6f}' }
                if use_scheduler:
                    loss_prt['lr'] = scheduler.get_last_lr()[0]
                    scheduler.step()
                mean_loss += loss.item()
            pbar.set_postfix(loss_prt)
        mean_loss /= (n_epochs*len(loader))
        loss_prt = {'loss': f'{mean_loss :.6f}', 'lr': scheduler.get_last_lr()[0]}
        pbar.set_postfix(loss_prt)
        pbar.close()

        self.model = model
        self.infer(adata, groupby=groupby, use_rep=use_rep, basis=basis,
                     add_embed = add_embed, model = model, device='cpu', verbose=verbose)

    @torch.no_grad()
    def infer(self, adata, groupby = None, use_rep='X',  basis='spatial',
             add_embed = 'GATE', model = None, device='cpu', verbose=2):
        device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
                    if device is None else torch.device(device))

        lDa = loadAData(adata,
                                groupby=groupby,
                                basis=basis,
                                use_rep=use_rep,
                                add_self_loops=self.add_self_loops,
                                weight_temp=self.weight_temp,
                                weights=self.edge_weight, 
                                validate=self.validate)
        loader, cellidx = lDa.get_sDatas(shuffle=False)
        cellidx = np.concatenate(cellidx, axis=0)
        cellidx = pd.Series(np.arange(cellidx.shape[0]), index=cellidx)
        cellord = cellidx.loc[adata.obs_names].values
        Model = model.cpu().to(device)
        Model.eval()
        
        with torch.no_grad():        
            Hs, Xs_, Loss  = [], [], []
            for batch in loader:
                batch = batch.to(device)
                H, X_ = Model(batch.x, batch.edge_index, edge_weight=batch.edge_weight) #, edge_attr = None
                loss = Model.loss_gae(batch.x, X_, H, batch.edge_index,
                                        #  edge_weight=batch.edge_weight, 
                                            Lambda=self.Lambda, Gamma=self.Gamma)
                Hs.append(H.cpu().detach())
                Xs_.append(X_.cpu().detach())
                Loss.append(loss.item())
    
            Hs = torch.cat(Hs, dim=0).numpy()[cellord]
            Xs_ = torch.cat(Xs_, dim=0).numpy()[cellord]
        adata.obsm[add_embed] = Hs.view()
        adata.layers['deconv'] = Xs_.view()

        verbose and  print(f'final mean loss: {np.mean(Loss)}')
        verbose and  print(f'finished: added to `.obsm["{add_embed}"]`')
        verbose and  print(f'          added to `.layers["deconv"]`')