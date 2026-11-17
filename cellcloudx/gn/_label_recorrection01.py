import torch as th
import torch.nn as nn
import torch.nn.functional as Fun

from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, TransformerConv
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering, DBSCAN
from scipy.spatial import Delaunay, cKDTree
import numpy as np
import pandas as pd

from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.sparse as ssp

try:
    import pykeops
    pykeops.set_verbose(False)
    from pykeops.torch import LazyTensor
except:
    pass
from ..tools._neighbors import Neighbors

class SpatialClusterRefiner:
    def __init__(self, 
                 knn=15, 
                 radius = None,
                 temp=[1.0,1.0], beta = 0.5, alpha= [1,1],
                 gnn_layers=[128, 128, 64, 64],
                 gnn_type='transformer',

                 
                 contrastive_temp=0.1,
                 boundary_threshold=0.7,
                 min_cluster_size=10,
                 min_samples=5,
                 eps=1.5,
                 dtype =th.float32,
                 device=None, 
                 verbose=1):
        self.knn = knn
        self.radius = radius
        self.temp = temp
        self.beta = beta
        self.alpha = alpha

        self.gnn_layers = gnn_layers
        self.gnn_type = gnn_type
        self.contrastive_temp = contrastive_temp
        self.boundary_threshold = boundary_threshold
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.eps = eps
        self.device = device or th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.verbose = verbose
        
    def fit_predict(self, X, F, labels, label_mask=None, n_epochs=500, lr=0.001,
                     weight_decay=5e-4, smooth_lambda=1.0, 
                     ent_lambda=0.1,
                     edge_lambda=0.0,
                     cont_lambda=0.0):
        self.X = self._normalize(X)
        self.F = self._normalize(F)
        self.labels = labels
        self.label_mask = label_mask if label_mask is not None else np.ones_like(labels, dtype=bool)

        self.Y, self.Y_orders = label_binary(labels, mask=label_mask)
        self.num_classes = len(self.Y_orders)

        # self.unique_labels = np.unique(labels[self.label_mask])
        # self.num_classes = len(self.unique_labels)

        self.gdata = self._build_multimodal_graph()
        self.model = self._train_multitask_gnn(n_epochs, lr, weight_decay, 
                            smooth_lambda=smooth_lambda,
                            edge_lambda=edge_lambda,
                            ent_lambda=ent_lambda, cont_lambda=cont_lambda)

        # preds, embeddings = self._get_predictions()
        # corrected_preds = self._correct_boundaries(preds, embeddings)
        # final_labels, new_cluster_ids = self._split_disconnected_clusters(corrected_preds)
        # cluster_info = self._generate_cluster_report(final_labels, embeddings)
        
        # return final_labels, cluster_info
    
    def _normalize(self, data):
        return (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
    
    def _build_multimodal_graph(self):
        src_x, dst_x, dist_x = coord_edges(self.X,  knn = self.knn, radius=self.radius )
        src, dst, dist = src_x, dst_x, dist_x

        if self.verbose:
            k_nodes, counts = np.unique(dst, return_counts=True)
            mean_neig = np.mean(counts)
            mean_radiu = np.mean(dist)
            print(f'nodes: {self.X.shape[0]}, edges: {len(dst)}\n'
                f'mean edges: {mean_neig :.3f}.\n'
                f'mean distance: {mean_radiu :.3e}.')


        # src_f, dst_f, dist_f = coord_edges(self.F,  knn = self.knn + 1)
        # edge = np.c_[ np.r_[src_x, src_f], np.r_[dst_x, dst_f] ]
        # edge = np.unique(edge, axis=0)
        # src, dst = edge[:, 0], edge[:, 1]

        spatial_dist = np.linalg.norm(self.X[src] - self.X[dst], axis=1)
        spatial_weight = np.exp(-spatial_dist / (np.median(spatial_dist) * self.temp[0]) )
        
        feature_dist = np.linalg.norm(self.F[src] - self.F[dst], axis=1)
        feature_weight = np.exp(-feature_dist / (np.median(feature_dist) * self.temp[1]) )
        edge_weight = (self.beta * spatial_weight +  (1- self.beta)* feature_weight)
        
        if self.verbose:
            fig, axs = plt.subplots(2,2, figsize=(7,6))
            axs[0,0].hist(spatial_dist, bins=100)
            axs[0,1].hist(feature_dist, bins=100)
            axs[1,0].hist(spatial_weight, bins=100)
            axs[1,1].hist(feature_weight, bins=100)
            # axs[1,2].hist(edge_weight, bins=100)
            plt.show()

        edge_index = th.tensor(np.array([dst,  src]), dtype=th.long)
        x = th.tensor(np.hstack([self.alpha[0] * self.X, self.alpha[1] *self.F]), 
                      dtype=self.dtype)
        wa = th.tensor(edge_weight, dtype=self.dtype)
        ws = th.tensor(spatial_weight, dtype=self.dtype)
        return Data(x=x, edge_index=edge_index, edge_attr=wa, edge_sp = ws ).to('cpu')

    def _train_multitask_gnn(self, n_epochs, lr, weight_decay, 
                             margin = 1.0,
                             smooth_lambda=1.0,  
                             ent_lambda=0.1, 
                             edge_lambda=0.1,
                             cont_lambda=0.1):

        M, indim = self.gdata.x.shape
        model = MultiTaskGNN([ indim ] + self.gnn_layers, self.num_classes).to(self.device)
        print(model)

        optimizer = th.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

        data = self.gdata.clone().to(self.device)
        src, dst = data.edge_index
        ws = data.edge_attr
        ws = data.edge_sp


        label_tensor = th.tensor(self.Y, dtype=th.long).to(self.device)
        label_idx = th.where(th.tensor(self.label_mask))[0]
        unlabel_idx = th.where(~th.tensor(self.label_mask))[0]

        use_constraint = cont_lambda >0 
        use_edge = edge_lambda > 0
        if use_constraint:
            aug_data = self._create_augmented_view(data)
  
        pbar = tqdm(range(n_epochs), desc="Multi-task GNN")
        for epoch in pbar:
            model.train()
            optimizer.zero_grad()

            logits, h, bij, z= model(data.x, data.edge_index, data.edge_attr,
                                    use_edge = use_edge,
                                    use_constraint=use_constraint)
            # logits_max = logits.max(dim=1, keepdim=True).values.detach()
            # stable_logits = logits - logits_max
            probs = Fun.softmax(logits, dim=1)

            loss_sup = Fun.cross_entropy(logits[label_idx], label_tensor[label_idx])

            if use_edge:
                diff_p = (probs[src] - probs[dst]).pow(2).sum(dim=1)  # [E]
                loss_gl = (ws * diff_p).sum()/ M
                # loss_gl = th.tensor(0)

                d_ij = (h[src] - h[dst]).pow(2).sum(dim=1) 
                d_ij = (th.clamp(d_ij, min=1e-8)).sqrt()
                pull = (ws * (1.0 - bij) * d_ij).mean()
                margin = th.quantile(d_ij[bij > 0.5], 0.75)
                push = (ws * bij * Fun.relu(margin - d_ij)).mean()
                loss_edge = pull + push
                # loss_edge = th.tensor(0)
            else:
                p_diff = probs[src] - probs[dst]
                loss_gl = (ws * (p_diff**2).sum(dim=1)).sum() / M
                loss_edge = th.tensor(0)

            p_unlabel = probs[unlabel_idx]
            loss_ent = -(p_unlabel * th.log(p_unlabel + 1e-12)).sum(dim=1).mean()

            if use_constraint:
                _, z_aug = model(aug_data.x, aug_data.edge_index, aug_data.edge_attr, use_constraint=use_constraint)
                # loss_contrast = self._contrastive_loss_keops(z, z_aug)
                loss_contrast = self._contrastive_loss_keops_stable(z, z_aug)
            else:
                loss_contrast = th.tensor(0)

            loss = (loss_sup + smooth_lambda * loss_gl + 
                    edge_lambda * loss_edge + 
                    cont_lambda * loss_contrast +
                    ent_lambda* loss_ent)
            
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.parameters(), 8.0)
            optimizer.step()
            # scheduler.step() #pass
            
            if epoch % 20 == 0 or epoch == n_epochs - 1:
                model.eval()
                with th.no_grad():
                    preds = logits.argmax(dim=1)
                    acc = (preds[label_idx] == label_tensor[label_idx]).float().mean().item()
                
            metrics = {
                'Loss': f'{loss.item():.4f}',
                'Sup': f'{loss_sup.item():.4f}',
                'GL': f'{loss_gl.item():.4f}',
                # 'EC': f'{loss_edge.item():.4f}',
                # 'Cont': f'{loss_contrast.item():.4f}',
                'Ent': f'{loss_ent.item():.4f}',
                'Acc': f'{acc:.4f}'
            }
            pbar.set_postfix(metrics)
        
        return model

    def _get_predictions(self):
        self.model.eval()
        data = self.gdata.clone().to(self.device)
        
        with th.no_grad():
            logits = self.model(data.x, data.edge_index, data.edge_attr)[0]
            embeddings = self.model.embed(data.x, data.edge_index, data.edge_attr).cpu().numpy()
            probs = Fun.softmax(logits, dim=1).cpu().numpy()
            # preds = probs.argmax(axis=1)
        return probs, embeddings

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
            th.nn.Linear(3*H, H), 
            th.nn.LeakyReLU(slope),
            th.nn.Linear(H,1)
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
            hij = th.cat([x[row], x[col], (x[row]-x[col]).abs()], dim=1)
            bij = th.sigmoid(self.edge_mlp(hij)).squeeze(-1)
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