import torch as th
import numpy as np
import pandas as pd
from tqdm import tqdm

import scipy.sparse as ssp

from ..tools._neighbors import Neighbors

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

import torch.nn.functional as Fun
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt

from ..tools._neighbors import Neighbors

def knnGraph(X, F, knn=None, radius=None, method='sknn', fpfh=None, 
            temp=[1.0,1.0], beta = 0.5, alpha= [1,1], verbose=1, use_delaunay=False, 
            adapt_bandwidth=True, dtype=th.float64,
            n_jobs = -1):
    
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

    sx = scaler(X)
    sf = scaler(F)

    dist_x = np.linalg.norm(sx[src] - sx[dst], axis=1) **2
    dist_f = np.linalg.norm(sf[src] - sf[dst], axis=1) **2

    if adapt_bandwidth:
        # sigma_x = sigma_square(sx, sx, xp=np)
        # sigma_z = sigma_square_cos(sf, sf, xp=np)
        sigma_x = np.median(dist_x)
        sigma_z = np.median(dist_f)
    else:
        sigma_x = sigma_z = 1.0

    dist_x /= temp[0]*sigma_x
    dist_f /= temp[1]*sigma_z
    weig_x  = np.exp(-dist_x)
    weig_f  = np.exp(-dist_f)

    # W = np.exp(-dist - dist_f)
    W = beta * weig_x + (1- beta) * weig_f
    Z = np.hstack([alpha[0] * sx, alpha[1] * sf])

    fig, axs = plt.subplots(2,2, figsize=(10,8))
    axs[0,0].hist(dist_x, bins=100)
    axs[0,1].hist(dist_f, bins=100)
    axs[1,0].hist(weig_x, bins=100)
    axs[1,1].hist(weig_f, bins=100)
    # axs[1,2].hist(edge_weight, bins=100)
    plt.show()

    if verbose:
        print(f'X: {sigma_x :.3e}, {dist_x.min() :.3e}, {dist_x.max() :.3e}, {dist_x.mean() :.3e}\n'
              f'F: {sigma_z :.3e}, {dist_f.min() :.3e}, {dist_f.max() :.3e}, {dist_f.mean() :.3e}\n'
              f'W: {W.min() :.3e}, {W.max() :.3e}, {W.mean() :.3e}'
        )

    edge_index = th.tensor(np.array([dst, src]), dtype=th.long)
    W = th.tensor(W, dtype=dtype)
    Z = th.tensor(Z, dtype=dtype)

    data = Data(x=Z, edge_index=edge_index, edge_attr=W)
    return data


def knnGraph0(X, F, knn=None, radius=None, method='sknn', fpfh=None, 
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


# -------------------------
# GCN model
# -------------------------

class SemiSupervisedGCN(th.nn.Module):
    def __init__(self, in_dim, hidden=[64], num_classes=5, slope= 0.2, dropout=0.5):
        super().__init__()
        if isinstance(hidden, int):
            hidden = [hidden]

        layers = []
        dims = [in_dim] + hidden 
        for in_c, out_c in zip(dims[:-1], dims[1:]):
            layers.append(GCNConv(in_c, out_c))
        layers.append(GCNConv(dims[-1], num_classes))

        self.convs = th.nn.ModuleList(layers)
        self.dropout = dropout
        self.act = th.nn.LeakyReLU(slope)

    def forward(self, x, edge_index, edge_weight=None, return_embedding=False):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            # x = Fun.relu(x)
            x = self.act(x)
            x = Fun.dropout(x, p=self.dropout, training=self.training)
        hidden = x
        logit = self.convs[-1](hidden, edge_index, edge_weight)
        if return_embedding:
            return logit, hidden
        return logit
    
# -------------------------
# Training routine
# -------------------------
def train_gcn(X, F, labels, label_mask=None, data = None, 
                knn=15, radius=None, beta=0.5, alpha= [1,1], temp=[1.0,1.0], 
                use_delaunay=False,  adapt_bandwidth=True, 
                hidden = [128, 64, 64],
                n_epochs=200, lr=0.01, weight_decay=5e-4,
                dtype=th.float32, device=None,  gradient_clipping=10.,
                smooth_lambda=1.0, ent_lambda=0.1, mc_dropout=False):
    """
    labels: int array of shape (M,) with -1 for unlabeled, otherwise 0..K-1
    labeled_mask: boolean array shape (M,)
    """
    device = th.device('cuda' if th.cuda.is_available() else 'cpu') if device is None else device
    if data is None:
        data = knnGraph(X, F, knn = knn, radius=radius, alpha=alpha, beta=beta,
                        temp=temp, use_delaunay=use_delaunay, adapt_bandwidth=adapt_bandwidth )
    data.x = data.x.to(dtype)
    data.edge_attr = data.edge_attr.to(dtype)
    data = data.to(device)

    Y, Y_orders = label_binary(labels, mask=label_mask)

    M, indim = data.x.shape
    num_classes = int(np.max(Y) + 1)
    model = SemiSupervisedGCN(in_dim=indim, 
                              hidden=hidden , #+ [indim], 
                              num_classes=num_classes, 
                              dropout=0.5).to(device)
    print(model)
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    Y = th.tensor(Y, dtype=th.long).to(device)
    labeled_idx = th.where(th.tensor(label_mask))[0]
    unlabeled_idx = th.where(~th.tensor(label_mask))[0]

    pbar = tqdm(range(n_epochs), total=n_epochs, colour='red', desc='GNN')
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        # data = data.to(device)

        logits = model(data.x, data.edge_index, data.edge_attr)
        probs = Fun.softmax(logits, dim=1)

        loss_sup = Fun.cross_entropy(logits[labeled_idx], Y[labeled_idx])

        # smoothness loss: Tr(Y^T L Y) approximated via neighbors
        # compute pairwise smoothness using edge list
        # sum_{(i,j) in E} w_ij ||p_i - p_j||^2
        p = probs
        row, col = data.edge_index
        w = data.edge_attr #.squeeze()
        loss_gl = (w * ((p[row] - p[col])**2).sum(dim=1)).sum() * 0.5 / M

        # entropy loss on unlabeled: encourage confident predictions
        loss_ent = -(p[unlabeled_idx] * th.log(p[unlabeled_idx] + 1e-12)).sum(dim=1).mean()

        loss = loss_sup + smooth_lambda * loss_gl + ent_lambda * loss_ent
        loss.backward()

        # th.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()

        if epoch % 20 == 0 or epoch == n_epochs - 1:
            # compute acc on labeled for monitoring
            model.eval()
            with th.no_grad():
                preds = probs.argmax(dim=1)
                acc = (preds[labeled_idx] == Y[labeled_idx]).to(dtype).mean()
        loss_prt = {'Loss': f'{loss.item():.4f}', 'Sup': f'{loss_sup.item():.4f}',
                    "GL": f'{loss_gl.item():.4f}', 'Ent': f'{loss_ent.item():.4f}',
                    'LabeledAcc': f'{acc:.4f}'}
        pbar.set_postfix(loss_prt)
    pbar.close()

    # final predictions and MC-dropout uncertainty if requested
    model.eval()
    if mc_dropout:
        model.train()  # enable dropout
        T = 20
        prob_mean, prob_sq, emb = 0,0,0
        with th.no_grad():
            for t in range(T):
                logits_t, emb_t = model(data.x, data.edge_index, data.edge_attr, return_embedding=True)
                probs_t = Fun.softmax(logits_t, dim=1).cpu().numpy()
                prob_mean += probs_t
                prob_sq   += probs_t ** 2
                emb       += emb_t.cpu().numpy()

        probs = prob_mean/T
        var_prob = prob_sq/T - probs.pow(2)
        emb = emb/T
    else:
        with th.no_grad():
            logits, emb = model(data.x, data.edge_index, data.edge_attr, return_embedding=True )
            probs = Fun.softmax(logits, dim=1).cpu().numpy()
            emb = emb.cpu().numpy()
            var_prob = None

    preds = Y_orders[probs.argmax(axis=1)]
    predictive_entropy = - (probs * np.log(probs + 1e-12)).sum(axis=1)
    return preds, emb, probs, var_prob, predictive_entropy 

def coord_edges(coordx, coordy=None,
                knn=50,
                radius=None,
                
                max_neighbor = int(1e4),
                method='sknn' ,
                keep_loops= False,
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

def label_binary(lables, mask=None, na_value=-1):
    lables = pd.Series(np.array(lables))

    if mask is None:
        mask = np.ones(lables.shape[0], dtype=np.bool_)
    mask = np.array(mask)
    
    m_labels, m_order = pd.factorize(lables[mask])
    n_labels = np.ones(lables.shape[0]) * na_value
    n_labels[mask] = m_labels
    
    n_labels = np.int64(n_labels)
    m_order = np.array(m_order)
    return n_labels, m_order

def centerlize(X, Xm=None, Xs=None):
    if ssp.issparse(X): 
        X = X.toarray()
    X = X.copy()
    N,D = X.shape
    Xm = np.mean(X, 0)

    X -= Xm
    Xs = np.sqrt(np.sum(np.square(X))/(N*D/2)) if Xs is None else Xs
    X /= Xs

    return [X, Xm, Xs]

def scaler( X):
    return (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

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