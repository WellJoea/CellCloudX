import torch as th
import torch.nn as nn
import torch.nn.functional as Fun

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, TransformerConv
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering, DBSCAN
from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd

from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.sparse as ssp
import scipy as sci

try:
    import pykeops
    pykeops.set_verbose(False)
    from pykeops.torch import LazyTensor
except:
    pass
from ..tools._neighbors import Neighbors


class SpatialClusterRefiner:
    def __init__(self, 
                 X, F, labels, label_mask=None,
                 knn=15, 
                 kd_method = 'sknn',
                 radius = None,
                 chunks = None,
                 batch_size = 1,

                 temp=[1.0,1.0], beta = 0.75, alpha= [1,1],
                 gnn_layers=[128, 128, 64, 64],
                 gnn_type='transformer',
    
                 n_epochs=1000, lr=0.001,
                 weight_decay=5e-4,

                 sup_soft=True,
                 p_gl  = 1.0,
                 p_boundary = 0.15, 
                 p_ent = 0.0,
                 p_edge = 0.0,
                 p_cont = 0.0,
                 p_pseudo = 0.0,    
                 
                 label_noise =0.1,
                 warmup = 30,  
                 boundary_threshold=0.5,
                 pseudo_threshold = 0.8,
                 # === ADDED END ===

                 margin = None,
                 use_sp_w=True,

                 contrastive_temp=0.1,
                 min_cluster_size=10,
                 min_samples=5,
                 eps=1.5,
                 dtype =th.float32,
                 device=None, 
                 verbose=1):
        self.knn = knn
        self.kd_method = kd_method
        self.radius = radius
        self.temp = temp
        self.beta = beta
        self.alpha = alpha

        self.gnn_layers = gnn_layers
        self.gnn_type = gnn_type

        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.label_noise = label_noise
        self.warmup = warmup
        self.sup_soft = sup_soft
        self.p_gl = p_gl
        self.p_ent = p_ent
        self.p_edge = p_edge
        self.p_cont = p_cont
        self.margin = margin
        self.use_sp_w = use_sp_w

        self.p_pseudo = p_pseudo
        self.p_boundary = p_boundary
        self.pseudo_threshold = pseudo_threshold

        self.X = np.array(X)
        self.F = np.array(F)
        self.label_mask = np.array(label_mask) if label_mask is not None else np.ones_like(labels, dtype=bool)
        self.chunks = min(chunks, self.X.shape[0]) if bool(chunks) else self.X.shape[0]
        self.batch_size = batch_size

        self.Y, self.Y_orders = label_binary(labels, mask=self.label_mask)
        self.num_classes = len(self.Y_orders)
        self.order_dict  = dict(zip(self.Y_orders, range(self.num_classes)))

        self.contrastive_temp = contrastive_temp
        self.boundary_threshold = boundary_threshold
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.eps = eps
        self.device = device or th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.verbose = verbose
        
    def fit(self, **kargs):
        self.build_graph()
        self.build_model()
        self.prediction()
        self.labels, self.scores = self.smooth_labels(**kargs)

        # corrected_preds = self._correct_boundaries(preds, embeddings)
        # final_labels, new_cluster_ids = self._split_disconnected_clusters(corrected_preds)
        # cluster_info = self._generate_cluster_report(final_labels, embeddings)

    def build_graph(self, ):
        X = self.X
        F = self.F
        Y = self.Y
        M = self.label_mask

        sx = centerlize(X)
        sf = center_normalize(F)
        sx1, sf1 = scaler(X), scaler(F)

        self.sins = random_split_torch(X.shape[0], n=self.chunks)
        Datalist = []
        mean_radiu = 0
        for i, sin in enumerate(self.sins):
            iX = X[sin]
            iY = Y[sin]
            iM = M[sin]
            # iF = F[sin]

            src_x, dst_x, dist_x = coord_edges(iX,  knn = self.knn, radius=self.radius, method=self.kd_method)
            src, dst, dist = src_x, dst_x, dist_x
            mean_radiu += (dist[dist>0]).sum()
            
            spatial_dist = np.linalg.norm(sx[sin][src] - sx[sin][dst], axis=1) ** 2
            # spatial_weight = np.exp(-spatial_dist / (np.median(spatial_dist[spatial_dist>0]) * self.temp[0]) )
            feature_dist = np.linalg.norm(sf[sin][src] - sf[sin][dst], axis=1) **2
            # feature_weight = np.exp(-feature_dist / (np.median(feature_dist[feature_dist>0]) * self.temp[1]) )
            # edge_weight = (self.beta * spatial_weight +  (1- self.beta)* feature_weight)

            edge_index = th.tensor(np.array([dst,  src]), dtype=th.long)
            x = th.tensor(np.hstack([self.alpha[0] * sx1[sin],
                                     self.alpha[1] * sf1[sin]]), 
                        dtype=self.dtype)
            # wa = th.tensor(edge_weight, dtype=self.dtype)
            # ws = th.tensor(spatial_weight, dtype=self.dtype)

            wa = th.tensor(spatial_dist, dtype=self.dtype)
            ws = th.tensor(feature_dist, dtype=self.dtype)
            y  = th.tensor(iY, dtype=th.long)
            mask = th.tensor(iM)
            idata = Data(x=x, y=y, mask=mask, edge_index=edge_index, edge_attr=wa, edge_sp = ws ).to('cpu')
            Datalist.append(idata)

        spdist = th.cat([idata.edge_attr for idata in Datalist])
        ftdist = th.cat([idata.edge_sp for idata in Datalist])
        sigmap = th.median(spdist[spdist>0]) * self.temp[0]
        sigmaf = th.median(ftdist[ftdist>0]) * self.temp[1]

        if self.verbose:
            all_edges  = sum([idata.edge_index.shape[1] for idata in Datalist])
            mean_radiu = mean_radiu/(all_edges - X.shape[0])
            mean_nodes = np.mean([ i.shape[0] for i in self.sins ])
            mean_edges = all_edges/X.shape[0]-1
            print(f'sigma2: {sigmap:.4e}, {sigmaf:.4e}\n'
                  f'n_chunk:{len(Datalist)}, mean_nodes:{int(mean_nodes)} \n'
                  f'nodes: {X.shape[0]}, edges: {all_edges}, mean radiu: {mean_radiu:.3e}, mean edges: {mean_edges:.1f}.')

        if self.verbose:
            spdist = th.exp(-spdist/sigmap).numpy()
            ftdist = th.exp(-ftdist/sigmaf).numpy()
            edge_weight = self.beta * spdist + (1-self.beta) * ftdist

            fig, axs = plt.subplots(1,3, figsize=(8.5,3))
            axs[0].hist(spdist, bins=100)
            axs[1].hist(ftdist, bins=100)
            axs[2].hist(edge_weight, bins=100)
            plt.show()

        for i in range(len(Datalist)):
            spdist = th.exp( -Datalist[i].edge_attr/sigmap )
            ftdist = th.exp( -Datalist[i].edge_sp/sigmaf )
            edge_weight = self.beta * spdist + (1-self.beta) * ftdist
            Datalist[i].edge_attr = edge_weight
            Datalist[i].edge_sp = spdist

            # fig, axs = plt.subplots(1,3, figsize=(9,3.5))
            # axs[0].hist(spdist, bins=100)
            # axs[1].hist(ftdist, bins=100)
            # axs[2].hist(edge_weight, bins=100)
            # plt.show()
        self.gdata = DataLoader(Datalist, batch_size=self.batch_size, shuffle=False)

    def build_model(self):
        n_epochs, lr, weight_decay = self.n_epochs, self.lr, self.weight_decay
        margin, p_gl, p_ent, p_edge, p_cont = self.margin, self.p_gl, self.p_ent, self.p_edge, self.p_cont
        p_pseudo, p_boundary = self.p_pseudo, self.p_boundary

        indim = self.gdata.dataset[0].x.size(1)
        model = MultiTaskGNN([ indim ] + self.gnn_layers, self.num_classes).to(self.device)
        if self.verbose:
            print(model)

        optimizer = th.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

        use_constraint = p_cont > 0 
        use_edge = p_edge > 0
        pbar = tqdm(range(n_epochs), desc="Multi-task GNN")

        for epoch in pbar:
            for data in self.gdata:
                data = data.to(self.device)
        
                (src, dst), y, mask = data.edge_index, data.y, data.mask
                ws = data.edge_sp if self.use_sp_w else data.edge_attr 
                M, indim = data.x.shape
   
                label_idx = th.where(mask)[0]
                unlabel_idx = th.where(~mask)[0]

                model.train()
                optimizer.zero_grad()
                logits, h, bij, z= model(data.x, data.edge_index, data.edge_attr,
                                        use_edge = use_edge,
                                        use_constraint=use_constraint)
                # logits_max = logits.max(dim=1, keepdim=True).values.detach()
                # stable_logits = logits - logits_max
                probs = Fun.softmax(logits, dim=1)

                if self.sup_soft:
                    if  (epoch >= self.warmup):
                        y1h = Fun.one_hot(y[label_idx], num_classes=logits.size(1)).float()
                        y_boot = (1-self.label_noise)*y1h + self.label_noise*probs[label_idx].detach()
                        loss_sup = -(y_boot * logits[label_idx].log_softmax(1)).sum(1).mean()
                    else:
                        y1h = Fun.one_hot(y[label_idx], num_classes=logits.size(1)).float()
                        y_boot = y1h
                        loss_sup = -(y_boot * logits[label_idx].log_softmax(1)).sum(1).mean()
                else:
                    loss_sup = Fun.cross_entropy(logits[label_idx], y[label_idx])

                if use_edge:
                    den = ws.sum().clamp_min(1e-8)
                    diff_p = (probs[src] - probs[dst]).pow(2).sum(dim=1)  # [E]
                    loss_gl = (ws * (1.0 - bij) * diff_p).sum()/ den
                    # loss_gl = th.tensor(0)
                    
                    h_hat = Fun.normalize(h, p=2, dim=1, eps=1e-10)
                    d2 = (h_hat[src] - h_hat[dst]).pow(2).sum(dim=1) 
                    with th.no_grad():
                        neg = (bij > 0.5)
                        pool = d2[neg] if neg.any() else d2
                        k = int(max(1, np.ceil(0.75 * pool.numel())))
                        m2_inst = th.kthvalue(pool, k).values
                        if not hasattr(self, "m2_ema"): self.m2_ema = m2_inst
                        self.m2_ema = 0.9 * self.m2_ema + 0.1 * m2_inst
                    m2 = self.m2_ema
                    # push = (ws * bij * th.nn.functional.softplus((self.m2_ema - d_ij) / 0.5)).sum()/M
                    pull = (ws * (1.0 - bij) * d2).sum() / den
                    push = (ws * bij * th.sigmoid((m2 - d2) / 0.5)).sum() / den
                    loss_edge = pull + push
                    # loss_edge = th.tensor(0)
                else:
                    p_diff = ((probs[src] - probs[dst])**2).sum(dim=1)
                    loss_gl = (ws * p_diff).sum() / M
                    loss_edge = th.tensor(0.0, device=self.device)

                if (p_ent >0) and (unlabel_idx.numel() > 0):
                    p_unlabel = probs[unlabel_idx]
                    loss_ent = -(p_unlabel * th.log(p_unlabel + 1e-10)).sum(dim=1).mean()
                else:
                    loss_ent = th.tensor(0.0, device=self.device)

                if use_constraint:
                    aug_data = self._create_augmented_view(data)
                    _, z_aug = model(aug_data.x, aug_data.edge_index, aug_data.edge_attr,
                                    use_constraint=use_constraint)
                    # loss_contrast = self._contrastive_loss_keops(z, z_aug)
                    loss_contrast = self._contrastive_loss_keops_stable(z, z_aug)
                else:
                    loss_contrast = th.tensor(0.0, device=self.device)

                loss_pseudo = th.tensor(0.0, device=self.device)
                loss_boundary = th.tensor(0.0, device=self.device)
                if (p_pseudo > 0) and (epoch >= self.warmup) and (unlabel_idx.numel() > 0):
                    with th.no_grad():
                        max_probs, pseudo_labels = th.max(probs[unlabel_idx], dim=1)
                        pseudo_mask = max_probs > self.pseudo_threshold
                        
                        pseudo_idx = unlabel_idx[pseudo_mask]
                        pseudo_targets = pseudo_labels[pseudo_mask]

                    if pseudo_idx.numel() > 0:
                        loss_pseudo = Fun.cross_entropy(logits[pseudo_idx], pseudo_targets)

                if (p_boundary >0):
                    with th.no_grad():
                        max_probs, _ = th.max(probs, dim=1)
                        bound_mask = max_probs > self.boundary_threshold
                    boundary_mask = ~mask[src] & mask[dst] & bound_mask[dst]
                    if (boundary_mask.sum() > 0):
                        # student_logits = logits[dst[boundary_mask]]
                        # teacher_labels = y[src[boundary_mask]]
                        student_logits = logits[src[boundary_mask]]
                        teacher_labels = y[dst[boundary_mask]]
                        loss_boundary = Fun.cross_entropy(student_logits, teacher_labels)

                loss = (loss_sup + p_gl * loss_gl + 
                        p_pseudo * loss_pseudo +
                        p_boundary * loss_boundary + 
                        p_ent* loss_ent +
                        p_edge * loss_edge + 
                        p_cont * loss_contrast)
                
                loss.backward()
                th.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                # scheduler.step() #pass
            
            if epoch % 20 == 0 or epoch == n_epochs - 1:
                # E = data.edge_index.size(1)
                # N = data.x.size(0)
                # print(f"[batch] N={N}, E={E}, xdim={data.x.size(1)}")
                # print(th.cuda.memory_summary(abbreviated=True))
        
                model.eval()
                with th.no_grad():
                    preds = logits.argmax(dim=1)
                    acc = (preds[label_idx] == y[label_idx]).float().mean().item()
                
            metrics = {
                'Loss': f'{loss.item():.4f}',
                'Sup': f'{loss_sup.item():.4f}',
                'GL': f'{loss_gl.item():.4f}',
                'EC': f'{loss_edge.item():.4f}',
                'bound': f'{loss_boundary.item():.4f}',
                'Ent': f'{loss_ent.item():.4f}',
                # 'Cont': f'{loss_contrast.item():.4f}',
                'Pse': f'{loss_pseudo.item():.4f}',
                'Acc': f'{acc:.4f}'
            } # check last
            pbar.set_postfix(metrics)
        self.model = model

    def prediction(self, model=None, gdata=None, sins=None, device=None):
        if model is None:
            model = self.model
        if gdata is None:
            gdata = self.gdata
        if device is None:
            device = self.device
        if sins is None:
            sins = self.sins

        N = sum([ ins.shape[0] for ins in sins])
        probs, embeddings = None, None
        model = model.to(device)
        model.eval()
        with th.no_grad():
            for ins, data in zip(sins, gdata):
                data = data.to(device)
                logits = model(data.x, data.edge_index, data.edge_attr)[0]
                iemb  = model.embed(data.x, data.edge_index, data.edge_attr).cpu().numpy()
                iprob = Fun.softmax(logits, dim=1).cpu().numpy()

                if probs is None:
                    K = iprob.shape[1]
                    D = iemb.shape[1]
                    probs = np.empty((N, K), dtype=np.float64)
                    embeddings   = np.empty((N, D), dtype=np.float64)
                probs[ins] = iprob
                embeddings[ins] = iemb

        labels = np.argmax(probs, axis=1)
        labels = self.Y_orders[labels.astype(np.int64)]
        scores = np.max(probs, axis=1)
        self.probs = probs
        self.embeddings = embeddings
        self.labels_0 = labels
        self.scores_0 = scores

    def smooth_labels(self, Y=None, X=None, probs=None, mask=None, method='all', 
                    knn=25, radius = None,  temp=1.0,
                    df_gamma=5,  max_cg_iter=200,
                    potts_iter =10, potts_gamma=0.3, potts_tol = 0,
                    mode_iters = 3,
                    threshold=0.3,
                    ):
        kd_method = self.kd_method
        if Y is None:
            Y = self.Y
        if X is None:
            X = self.X
        if probs is None:
            probs = self.probs
        if mask is None:
            mask = self.label_mask
        assert Y[mask].min() >=0 and Y[mask].max() < self.num_classes

        C = self.num_classes
        N = X.shape[0]

        [src, dst, dist] = coord_edges(X, knn=knn, radius=radius, method=kd_method, keep_loops= True)
        dist2  = dist ** 2
        simga2 = np.median(dist2[dist2>0]) * temp
        W = np.exp(-dist2 / simga2)
        W = ssp.csr_array((W, (src, dst)), shape=(X.shape[0], X.shape[0]))
        W = (W + W.T)/2.0

        if type(method) == str:
            if method == 'all':
                methods = ['aggr', 'mode', 'diffusion', 'potts']
            else:
                methods = [method]
        else:
            assert type(method) == list
            methods = method

        labela, scorea = [np.argmax(probs, axis=1)], [np.max(probs, axis=1)]
        for method in methods:
            if self.verbose:
                print('do:', method)
            if method == 'aggr':
                labels = np.argmax(probs, axis=1)
                scores = np.max(probs, axis=1)
                remark0 = (scores >= threshold)
                remarky =  remark0 & mask
                labels[remarky] = Y[remarky]
                scores[remarky] = 1.0

                WS =  W.multiply( ~remark0[:, None] )
                if True:
                    WS.setdiag(1)
                else:
                    WS.setdiag((remark0).astype(WS.dtype))
                D = WS.sum(1)[:, None]
                D[D==0] = 1.0
                WS = WS.multiply(1.0/D)

                probs1 = WS @ probs
                labels_1 = np.argmax(probs1, axis=1)
                scores_1 = np.max(probs1, axis=1)

                remark1 = scores_1 > scores
                labels[remark1] = labels_1[remark1]
                scores[remark1] = scores_1[remark1]

            elif method == 'mode':
                labels = np.argmax(probs, axis=1)
                scores = np.max(probs, axis=1)
                remarky = (scores >= threshold) & mask
                labels[remarky] = Y[remarky]

                D = W.sum(1)[:, None]
                D[D==0] = 1.0
                WS = W.multiply(1.0/D)

                for _ in range(mode_iters):
                    votes = np.zeros((N, C), dtype=np.float32)
                    for k in range(C):
                        votes[:, k] = WS.dot((labels == k).astype(np.float32))
                    labels = votes.argmax(1).astype(np.int32)
                    labels[remarky] = Y[remarky]
                scores = votes.max(1)

            elif method == 'diffusion':
                labels = np.argmax(probs, axis=1)
                scores = np.max(probs, axis=1)
                maskf = mask & (scores >= threshold)
                probf = np.eye(C, dtype=np.float32)[labels]

                D = ssp.csr_array((W.sum(1), (np.arange(N), np.arange(N))), 
                                shape=(N, N), dtype=np.float32)
                L = D - W
                A =  (ssp.identity(N, dtype=np.float32) + df_gamma * L).tocsr()  # SPD
                if maskf is not None:
                    rk = (~maskf).astype(A.dtype)
                    A = A.multiply(rk[:, None])
                    Ad = A.diagonal()
                    Ad[maskf]= 1.0
                    A.setdiag(Ad)
                    A.eliminate_zeros()   

                    # A = A.tolil()
                    # idx = np.where(maskf)[0]
                    # A[idx, :] = 0.0
                    # A[idx, idx] = 1.0
                    # A = A.tocsr()

                prob1 = np.empty((N, C), dtype=np.float32)
                for ic in tqdm(range(C), desc='diffusion with CG', disable= not bool(self.verbose)):
                    b = probs[:, ic].copy()
                    if (probf is not None)  and (maskf is not None):
                        b[maskf] = probf[maskf, ic]
                    xk, _ = ssp.linalg.cg(A, b, x0=b, maxiter=max_cg_iter)
                    prob1[:, ic] = xk

                prob1 = np.clip(prob1, 1e-10, None)
                prob1 /= prob1.sum(1, keepdims=True)
                labels = prob1.argmax(1)
                scores = prob1.max(1)

            elif method == 'potts':
                labels = np.argmax(probs, axis=1)
                scores = np.max(probs, axis=1)

                maskf = mask & (scores >= threshold)
                labels[maskf] = Y[maskf]

                D = W.sum(1)[:, None]
                unary = -np.log(np.clip(probs, 1e-10, 1.0)) 
                one_hot = np.zeros((N, C), dtype=np.float32)
                for it in tqdm(range(potts_iter), desc='potts', disable= not bool(self.verbose)):
                    neigh_vote = np.zeros((N, C), dtype=np.float32)
                    for ic in range(C):
                        one_hot[:, ic] = (labels == ic)
                    neigh_vote = W.dot(one_hot)  # (N,K)
                    # pairwise cost for choosing k: lam * (d - neigh_vote[:,k])
                    pair_cost = potts_gamma * (D - neigh_vote)

                    local_E = unary + pair_cost
                    labelsn = local_E.argmin(1).astype(np.int64)
                    labelsn[maskf] = labels[maskf]

                    diff = (labelsn != labels).sum()/N
                    labels = labelsn

                    if diff <= potts_tol:
                        break
                scores = np.exp(-local_E).max(1)
            else:
                raise ValueError(f'Unknown method: {method}')

            # labels = self.Y_orders[labels.astype(np.int64)]
            labela.append(labels)
            scorea.append(scores)
        labela = np.array(labela, dtype=np.int64).T
        scorea = np.array(scorea).T
        column = ['raw', *methods]
        if self.verbose:
            N, K = labela.shape
            simis = np.ones((K,K))
            for i in range(K):
                for j in range(i):
                    ism = (labela[:,i] == labela[:,j]).sum()/N
                    simis[i,j] = ism
                    simis[j,i] = ism
            simis = pd.DataFrame(simis, columns=column, index=column)
            simis['sum'] = simis.sum(1)
            print(simis)
        
        labelf = sci.stats.mode(labela[:,1:], axis=1).mode.astype(np.int64)
        scoref = scorea[:,1:].mean(1)

        labela = pd.DataFrame(self.Y_orders[labela], columns=column)
        labela['labels'] = self.Y_orders[labelf]
                                        
        scorea = pd.DataFrame(scorea, columns=[f'{i}_scores'for i in column])
        scorea['scores'] = scoref
        return labela, scorea

    def smooth_labels0(self, Y=None, X=None, probs=None, mask=None, method='all', 
                        knn=25, radius = None,  temp=1.0,
                    df_gamma=5,  max_cg_iter=200,
                    potts_iter =10, potts_gamma=0.3, potts_tol = 0,
                    mode_iters = 3,
                    threshold=0.7,
                    ):
        kd_method = self.kd_method
        if Y is None:
            Y = self.Y
        if X is None:
            X = self.X
        if probs is None:
            probs = self.probs
        if mask is None:
            mask = self.label_mask
        assert Y[mask].min() >=0 and Y[mask].max() < self.num_classes

        C = self.num_classes
        N = X.shape[0]

        [src, dst, dist] = coord_edges(X, knn=knn, radius=radius, method=kd_method, keep_loops= True)
        dist2  = dist ** 2
        simga2 = np.median(dist2[dist2>0]) * temp
        W = np.exp(-dist2 / simga2)
        W = ssp.csr_array((W, (src, dst)), shape=(X.shape[0], X.shape[0]))
        W = (W + W.T)/2.0

        if type(method) == str:
            if method == 'all':
                methods = ['aggr', 'mode', 'diffusion', 'potts']
            else:
                methods = [method]
        else:
            assert type(method) == list
            methods = method

        labela, scorea = [], []
        for method in methods:
            if self.verbose:
                print('do:', method)
            if method == 'aggr':
                labels = np.argmax(probs, axis=1)
                scores = np.max(probs, axis=1)
                remark0 = (scores < threshold)
                remarky =  remark0 & mask
                labels[remarky] = Y[remarky]
                scores[remarky] = 1.0

                WS =  W.multiply( remark0[:, None] )
                if True:
                    WS.setdiag(1)
                else:
                    WS.setdiag((~remark0).astype(WS.dtype))
                D = WS.sum(1)[:, None]
                D[D==0] = 1.0
                WS = WS.multiply(1.0/D)

                probs1 = WS @ probs
                labels_1 = np.argmax(probs1, axis=1)
                scores_1 = np.max(probs1, axis=1)

                remark1 = scores_1 > scores
                labels[remark1] = labels_1[remark1]
                scores[remark1] = scores_1[remark1]

            elif method == 'mode':
                labels = np.argmax(probs, axis=1)
                scores = np.max(probs, axis=1)
                remarky = (scores < threshold) & mask
                labels[remarky] = Y[remarky]

                D = W.sum(1)[:, None]
                D[D==0] = 1.0
                WS = W.multiply(1.0/D)

                for _ in range(mode_iters):
                    votes = np.zeros((N, C), dtype=np.float32)
                    for k in range(C):
                        votes[:, k] = WS.dot((labels == k).astype(np.float32))
                    labels = votes.argmax(1).astype(np.int32)
                    labels[remarky] = Y[remarky]
                scores = votes.max(1)

            elif method == 'diffusion':
                labels = np.argmax(probs, axis=1)
                scores = np.max(probs, axis=1)
                maskf = mask | (scores >= threshold)
                probf = np.eye(C, dtype=np.float32)[labels]

                D = ssp.csr_array((W.sum(1), (np.arange(N), np.arange(N))), 
                                shape=(N, N), dtype=np.float32)
                L = D - W
                A =  (ssp.identity(N, dtype=np.float32) + df_gamma * L).tocsr()  # SPD
                if maskf is not None:
                    rk = (~maskf).astype(A.dtype)
                    A = A.multiply(rk[:, None])
                    Ad = A.diagonal()
                    Ad[maskf]= 1.0
                    A.setdiag(Ad)
                    A.eliminate_zeros()   

                    # A = A.tolil()
                    # idx = np.where(maskf)[0]
                    # A[idx, :] = 0.0
                    # A[idx, idx] = 1.0
                    # A = A.tocsr()

                prob1 = np.empty((N, C), dtype=np.float32)
                for ic in tqdm(range(C), desc='diffusion with CG', disable= not bool(self.verbose)):
                    b = probs[:, ic].copy()
                    if (probf is not None)  and (maskf is not None):
                        b[maskf] = probf[maskf, ic]
                    xk, _ = ssp.linalg.cg(A, b, x0=b, maxiter=max_cg_iter)
                    prob1[:, ic] = xk

                prob1 = np.clip(prob1, 1e-10, None)
                prob1 /= prob1.sum(1, keepdims=True)
                labels = prob1.argmax(1)
                scores = prob1.max(1)

            elif method == 'potts':
                labels = np.argmax(probs, axis=1)
                scores = np.max(probs, axis=1)

                maskf = mask & (scores >= 1- threshold)
                labels[maskf] = Y[maskf]

                D = W.sum(1)[:, None]
                unary = -np.log(np.clip(probs, 1e-10, 1.0)) 
                one_hot = np.zeros((N, C), dtype=np.float32)
                for it in tqdm(range(potts_iter), desc='potts', disable= not bool(self.verbose)):
                    neigh_vote = np.zeros((N, C), dtype=np.float32)
                    for ic in range(C):
                        one_hot[:, ic] = (labels == ic)
                    neigh_vote = W.dot(one_hot)  # (N,K)
                    # pairwise cost for choosing k: lam * (d - neigh_vote[:,k])
                    pair_cost = potts_gamma * (D - neigh_vote)

                    local_E = unary + pair_cost
                    labelsn = local_E.argmin(1).astype(np.int64)
                    labelsn[maskf] = labels[maskf]

                    diff = (labelsn != labels).sum()/N
                    labels = labelsn

                    if diff <= potts_tol:
                        break
                scores = np.exp(-local_E).max(1)
            else:
                raise ValueError(f'Unknown method: {method}')

            # labels = self.Y_orders[labels.astype(np.int64)]
            labela.append(labels)
            scorea.append(scores)
        labela = np.array(labela, dtype=np.int64).T
        scorea = np.array(scorea).T

        if self.verbose:
            N, K = labela.shape
            simis = np.ones((K,K))
            for i in range(K):
                for j in range(i):
                    ism = (labela[:,i] == labela[:,j]).sum()/N
                    simis[i,j] = ism
                    simis[j,i] = ism
            simis = pd.DataFrame(simis, columns=methods, index=methods)
            simis['sum'] = simis.sum(1)
            print(simis)
        
        labels = pd.DataFrame(self.Y_orders[labela], columns=methods)
        labels['labels'] = self.Y_orders[sci.stats.mode(labela, axis=1).mode.astype(np.int64)]
                                         
        scores = pd.DataFrame(scorea, columns=methods)
        scores['scores'] = scorea.mean(1)
        return labels, scores

    def interpolation(self, Y, F=None, knn=None, radius=None, kd_method=None, temp=None):
        Ys = label_onehot( self.labels)
        assert Ys.shape[1] == self.num_classes

        W = self.knn_weight(Y, F = F, knn=knn, radius=radius, kd_method=kd_method, temp=temp)
        D = W.sum(1)
        D[D == 0] = 1
        W = W / D[:, None]
        W.eliminate_zeros()

        Ys = W @ Ys
        label = self.Y_orders[Ys.argmax(1)]
        score = Ys.max(1)
        return label, score

    def knn_weight(self, Y, F=None, beta=None, knn = None, radius=None, kd_method=None, temp=None):
        beta = self.beta if beta is None else beta
        knn  = self.knn if knn is None else knn
        radius = self.radius if radius is None else radius
        kd_method = self.kd_method if kd_method is None else kd_method

        src_x, dst_x, dist_x = coord_edges(self.X, Y, knn = knn, radius=radius, method=kd_method)
        src, dst, dist = src_x, dst_x, dist_x

        if self.verbose:
            k_nodes, counts = np.unique(dst, return_counts=True)
            mean_neig = np.mean(counts)
            mean_radiu = np.mean(dist)
            print(f'nodes: {Y.shape[0]}, edges: {len(dst)}\n'
                f'mean edges: {mean_neig :.3f}.\n'
                f'mean distance: {mean_radiu :.3e}.')

        N, M = self.X.shape[0], Y.shape[0]
        sx = centerlize(np.r_[self.X, Y])

        spatial_dist = np.linalg.norm(sx[:N][src] - sx[N:][dst], axis=1) ** 2
        spatial_weight = np.exp(-spatial_dist / (np.median(spatial_dist) * temp[0]) )

        if F is not None:
            sf = center_normalize(np.r_[self.F, F])
            feature_dist = np.linalg.norm(sf[:N][src] - sf[N:][dst], axis=1) **2
            feature_weight = np.exp(-feature_dist / (np.median(feature_dist) * temp[1]) )
            edge_weight = (beta * spatial_weight +  (1- beta)* feature_weight)

            if self.verbose:
                fig, axs = plt.subplots(2,2, figsize=(7,6))
                axs[0,0].hist(spatial_dist, bins=100)
                axs[0,1].hist(feature_dist, bins=100)
                axs[1,0].hist(spatial_weight, bins=100)
                axs[1,1].hist(feature_weight, bins=100)
                # axs[1,2].hist(edge_weight, bins=100)
                plt.show()
        
        else:
            edge_weight = spatial_weight
        
        W = ssp.csr_array((edge_weight, (dst, src)), shape=( M, N))
        return W

    def _create_augmented_view(self, data):
        aug_data = data.clone()
        noise = th.randn_like(aug_data.x) * 0.1
        aug_data.x = aug_data.x + noise
        drop_mask = th.rand_like(aug_data.edge_attr) < 0.2
        aug_data.edge_attr[drop_mask] = 0
        return aug_data
    
    def _contrastive_loss(self, z, z_aug):
        sim_matrix = th.mm(z, z_aug.t()) / self.contrastive_temp
        sim_matrix = th.exp(sim_matrix)
        pos_sim = th.diag(sim_matrix)
        neg_sim = sim_matrix.sum(dim=1) - pos_sim
        loss = -th.log(pos_sim / (pos_sim + neg_sim + 1e-12)).mean()
        return loss

    def _contrastive_loss_keops(self, z, z_aug):
        # z = z.contiguous()
        # z_aug = z_aug.contiguous()
        
        z_i = LazyTensor(z.unsqueeze(1))
        z_aug_j = LazyTensor(z_aug.unsqueeze(0))

        sim_matrix = (z_i | z_aug_j) / self.contrastive_temp
        sim_matrix = sim_matrix.exp()

        pos_sim = (z * z_aug).sum(dim=1) / self.contrastive_temp
        pos_sim = pos_sim.exp()

        all_sim = sim_matrix.sum(dim=1).squeeze() + 1e-12
        loss = -th.log(pos_sim / all_sim).mean()
        return loss

    def _contrastive_loss_keops_stable(self, z, z_aug):
        N, D = z.shape
        invT = 1.0 / self.contrastive_temp

        Zi = LazyTensor(z.view(N, 1, D))
        Zj = LazyTensor(z_aug.view(1, N, D))
        logits_ij = (Zi * Zj).sum(dim=2) * invT 

        pos_logits = (z * z_aug).sum(dim=1) * invT

        log_denom = logits_ij.logsumexp(dim=1).view(N)  # (N,)
        loss = -(pos_logits - log_denom).mean()
        return loss

    def _correct_boundaries(self, probs, embeddings):
        preds = probs.argmax(1)
        corrected = preds.copy()

        max_probs = np.max(probs, axis=1)
        boundary_mask = max_probs < self.boundary_threshold
        

        spatial_grad = self._compute_spatial_gradient()
        spatial_boundary_mask = spatial_grad > np.percentile(spatial_grad, 75)
        
        feature_sim = pairwise_distances(self.F, metric='cosine')
        feature_boundary_mask = feature_sim.mean(axis=1) > np.percentile(feature_sim.mean(axis=1), 75)
        
        combined_boundary_mask = boundary_mask | spatial_boundary_mask | feature_boundary_mask

        src, dst = self.gdata.edge_index.cpu().numpy()
        for i in np.where(combined_boundary_mask)[0]:
            neighbors = dst[src == i]
            if len(neighbors) > 0:
                neighbor_labels = preds[neighbors]
                corrected[i] = np.bincount(neighbor_labels).argmax()
        
        return corrected
    
    def _compute_spatial_gradient(self):
        grads = []
        for idim in range(self.X.shape[1]):
            grad = np.gradient(self.X[:, idim]) ** 2
            grads.append(grad)
        return np.sqrt(np.sum(grads, axis=0))
    
    def _split_disconnected_clusters(self, preds):
        final_labels = preds.copy()
        new_cluster_id = self.num_classes
        
        # 1. 基于空间连通性的拆分
        for cluster_id in range(self.num_classes):
            cluster_mask = preds == cluster_id
            cluster_points = self.X[cluster_mask]
            
            if len(cluster_points) < self.min_cluster_size:
                continue
            
            # 使用DBSCAN检测不连续区域
            db = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            sub_labels = db.fit_predict(cluster_points)
            
            # 如果发现多个连通分量
            unique_sub_labels = np.unique(sub_labels)
            if len(unique_sub_labels) > 1:
                for sub_id in unique_sub_labels:
                    if sub_id == -1:  # 噪声点保持原标签
                        continue
                    point_indices = np.where(cluster_mask)[0]
                    sub_mask = sub_labels == sub_id
                    final_labels[point_indices[sub_mask]] = new_cluster_id
                    new_cluster_id += 1
        
        # 2. 基于特征差异的二次拆分
        for cluster_id in range(self.num_classes, new_cluster_id):
            cluster_mask = final_labels == cluster_id
            cluster_features = self.F[cluster_mask]
            
            if len(cluster_features) < self.min_cluster_size * 2:
                continue
            
            # 使用高斯混合模型检测特征子群
            gmm = GaussianMixture(n_components=2, covariance_type='diag')
            sub_labels = gmm.fit_predict(cluster_features)
            
            # 如果子群差异显著
            if gmm.bic(cluster_features) < gmm.aic(cluster_features):
                point_indices = np.where(cluster_mask)[0]
                for sub_id in range(2):
                    sub_mask = sub_labels == sub_id
                    final_labels[point_indices[sub_mask]] = new_cluster_id
                    new_cluster_id += 1
        
        return final_labels, list(range(self.num_classes, new_cluster_id))
    
    def _identify_boundaries(self, labels):
        src, dst = self.gdata.edge_index.cpu().numpy()
        boundary_mask = np.zeros(len(labels), dtype=bool)
        
        for i in range(len(labels)):
            neighbors = dst[src == i]
            if len(neighbors) > 0:
                neighbor_labels = labels[neighbors]
                if not np.all(neighbor_labels == labels[i]):
                    boundary_mask[i] = True
        
        return boundary_mask

class MultiTaskGNN(nn.Module):
    def __init__(self, dims, num_classes, gnn_type='gcn', slope=0.3, dropout=0.3):
        super().__init__()
        if gnn_type == 'gcn':
            ConvLayer = GCNConv
        elif gnn_type == 'gat':
            ConvLayer = lambda in_c, out_c: GATConv(in_c, out_c, heads=1, concat=False)
        elif gnn_type == 'sage':
            ConvLayer = SAGEConv
        elif gnn_type == 'transformer':
            ConvLayer = TransformerConv
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")

        self.act = th.nn.LeakyReLU(slope)
        self.act = th.nn.GELU()

        self.convs = nn.ModuleList()        
        for i in range(len(dims)-1):
            self.convs.append(ConvLayer(dims[i], dims[i+1]))
        
        self.classifier = ConvLayer(dims[-1], num_classes)

        H=dims[-1]
        self.projection_head = nn.Sequential(
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, H)
        )
        self.edge_mlp=th.nn.Sequential(
            th.nn.Linear(H, H//2), 
            th.nn.LeakyReLU(slope),
            th.nn.Linear(H//2,1)
        )
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None, use_edge=False, use_constraint=False):
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = Fun.leaky_relu(x)
            x = Fun.dropout(x, p=self.dropout, training=self.training)
        h = x
        logits = self.classifier(x, edge_index, edge_weight)
        
        if use_edge:
            row, col = edge_index
            # hij = th.cat([x[row], x[col], (x[row]-x[col]).abs()], dim=1)
            # hij = x[row] * x[col]
            bij = (x[row]-x[col]).abs()
            bij = th.sigmoid(self.edge_mlp(bij)).squeeze(-1)
        else:
            bij = None

        if use_constraint:
            z = self.projection_head(x)
            z = Fun.normalize(z, p=2, dim=1)
        else:
            z = None
        return logits, h, bij, z
    
    def embed(self, x, edge_index, edge_weight=None):
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = Fun.leaky_relu(x)
        return x
            
def knnGraph(X, F, knn=None, radius=None, method='sknn', fpfh=None, 
            temp=[1.0,1.0], beta = 0.5, alpha= [1,1], verbose=1, use_delaunay=False, 
            adapt_bandwidth=True, dtype=th.float64,
            normal=True, normal_F = True, n_jobs = -1):
    
    if use_delaunay:
        W = delaunay_adjacency(X)
        src, dst = W.nonzero()
        dist = np.linalg.norm(X[src] - X[dst], axis=1)
    else:
        [src, dst, dist] = coord_edges(X, knn=knn, radius=radius,
                                        method=method, n_jobs=n_jobs)

    k_nodes, counts = np.unique(dst, return_counts=True)
    mean_neig = np.mean(counts)
    mean_radiu = np.mean(dist)

    if verbose:
        print(f'nodes: {X.shape[0]}, edges: {len(dst)}\n'
            f'mean edges: {mean_neig :.3f}.\n'
            f'mean distance: {mean_radiu :.3e}.')

    # fpfh = FPFH(X, knn=41)
    if normal:
        # sx = StandardScaler().fit_transform(X)
        sx = centerlize(X)[0]
        # sx = center_normalize(fpfh)
    else:
        sx = X
    if normal_F:
        # sf = centerlize(F)[0]
        sf = center_normalize(F)
        # sf = StandardScaler().fit_transform(F)
    else:
        sf = F

    dist_x = np.linalg.norm(sx[src] - sx[dst], axis=1)**2
    dist_f = np.linalg.norm(sf[src] - sf[dst], axis=1)**2

    if adapt_bandwidth:
        sigma_x = sigma_square(sx, sx, xp=np)
        sigma_z = sigma_square_cos(sf, sf, xp=np)
    else:
        sigma_x = sigma_z = 1.0

    dist_x *= temp[0]/sigma_x
    dist_f *= temp[1]/sigma_z
    # W = np.exp(-dist - dist_f)
    
    W = beta * np.exp(- dist_x) + (1- beta) * np.exp(- dist_f)
    Z = np.hstack([alpha[0] * sx, alpha[1] * sf])

    if verbose:
        print(f'X: {sigma_x.min() :.3e}, {dist_x.min() :.3e}, {dist_x.max() :.3e}, {dist_x.mean() :.3e}\n'
              f'F: {sigma_z.min() :.3e}, {dist_f.min() :.3e}, {dist_f.max() :.3e}, {dist_f.mean() :.3e}\n'
              f'W: {W.min() :.3e}, {W.max() :.3e}, {W.mean() :.3e}'
        )

    edge_index = th.tensor(np.array([dst, src]), dtype=th.long)
    W = th.tensor(W, dtype=dtype)
    Z = th.tensor(Z, dtype=dtype)

    data = Data(x=Z, edge_index=edge_index, edge_attr=W)
    return data

def coord_edges(coordx, coordy=None,
                knn=50,
                radius=None,
                
                max_neighbor = int(1e4),
                method='sknn' ,
                keep_loops= True,
                n_jobs = -1):
    if coordy is None:
        coordy = coordx
    
    cknn = Neighbors( method=method ,metric='euclidean', n_jobs=n_jobs)
    cknn.fit(coordx, radius_max= None,max_neighbor=max_neighbor)
    distances, indices = cknn.transform(coordy, knn=knn, radius = radius)

    src = np.concatenate(indices, axis=0).astype(np.int64)
    dst = np.repeat(np.arange(len(indices)), list(map(len, indices))).astype(np.int64)
    dist = np.concatenate(distances, axis=0)

    if (coordy is None) and (not keep_loops):
        mask = src != dst
        src = src[mask]
        dst = dst[mask]
        dist = dist[mask]

    return [src, dst, dist]

def label_binary( lables, lable_dict = None, mask=None, na_value=-1):
    lables = pd.Series(np.array(lables))
    if mask is None:
        mask = np.ones(lables.shape[0], dtype=np.bool_)
    mask = np.array(mask)

    if lable_dict is None:
        m_order = np.unique(lables[mask])
        lable_dict = dict(zip(m_order, range(m_order.shape[0])))
    else:
        m_order = np.array(list(lable_dict.keys()))
        # assert set(lables[mask]).issubset(m_order)

    n_labels = lables.map(lable_dict).fillna(na_value).values
    n_labels[~mask] = na_value
    n_labels = np.int64(n_labels)
    m_order = np.array(m_order)
    return n_labels, m_order

def label_onehot(Y):
    Y = np.array(Y)
    assert Y.min() >=0
    dist = np.ones(Y.shape[0])
    src  = np.arange(Y.shape[0])
    dst  = Y
    return ssp.csr_array((dist, (src, dst)),  shape=(len(Y), max(Y)+1 ))

def scaler( X):
    return (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

def centerlize(X, Xm=None, Xs=None):
    if ssp.issparse(X): 
        X = X.toarray()
    X = X.copy()
    N,D = X.shape
    Xm = np.mean(X, 0)

    X -= Xm
    Xs = np.sqrt(np.sum(np.square(X))/(N*D/2)) if Xs is None else Xs
    X /= Xs

    return X

def normalize(X):
    if ssp.issparse(X): 
        X = X.toarray()

    X = X.copy()
    l2x = np.linalg.norm(X, ord=None, axis=1, keepdims=True)
    l2x[l2x == 0] = 1
    return X/l2x #*((self.DF/2.0)**0.5)

def center_normalize(X):
    if ssp.issparse(X): 
        X = X.toarray()

    X = X.copy()
    X -= X.mean(axis=0, keepdims=True)
    l2x = np.linalg.norm(X, ord=None, axis=1, keepdims=True)
    l2x[l2x == 0] = 1
    return X/l2x 

def sigma_square_cos(X, Y, xp =th):
    [N, D] = X.shape
    [M, D] = Y.shape
    # sigma2 = (M*np.trace(np.dot(np.transpose(X), X)) + 
    #           N*np.trace(np.dot(np.transpose(Y), Y)) - 
    #           2*np.dot(np.sum(X, axis=0), np.transpose(np.sum(Y, axis=0))))
    sigma2 = (M*xp.sum(X * X) + 
                N*xp.sum(Y * Y) - 
                2* xp.sum(xp.sum(X, 0) * xp.sum(Y, 0)))
    sigma2 /= (N*M) * 2 # TODO: D
    return sigma2

def sigma_square( X, Y, xp=th):
    sigma2 = (xp.mean(X * X) + 
                xp.mean(Y * Y) - 
                2* xp.mean( xp.mean(X, 0) * xp.mean(Y, 0) ))
    return sigma2

def random_split_torch(N,n = None, m=None,  seed=42):
    th.manual_seed(seed)
    if m is None:
        m = N // n
    indices = th.randperm(N)
    chunk_sizes = [N // m + (1 if i < N % m else 0) for i in range(m)]
    chunks = th.split(indices, chunk_sizes)
    return chunks

def random_split_numpy(N, n = None, m=None, seed=42):
    np.random.seed(seed)
    if m is None:
        m = N // n
    indices = np.random.permutation(N)
    chunk_sizes = [N // m + (1 if i < N % m else 0) for i in range(m)]
    chunks = []
    start = 0
    for size in chunk_sizes:
        chunks.append(indices[start:start+size])
        start += size
    return chunks

def delaunay_adjacency(points):
    from scipy.spatial import Delaunay    
    from scipy.sparse import csr_matrix
    tri = Delaunay(points)
    simplices = tri.simplices
    
    n = points.shape[0]
    adj_matrix = np.zeros((n, n))
    for simplex in simplices:
        for i in range(3):
            for j in range(i+1, 3):
                u = simplex[i]
                v = simplex[j]
                adj_matrix[u, v] = 1
                adj_matrix[v, u] = 1
    return csr_matrix(adj_matrix)