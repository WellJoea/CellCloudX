import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import scipy as sci
import scipy.sparse as ssp

from ._neighbors import Neighbors

from ._cluster import leiden_graph

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering

def cluster(X, F,  k=6, alpha=[0.5, 0.5],  beta=[0.1,5,], 
            temp=2, knn=None, method= ['kmeans', 'leiden'],
            pnt = 64,
            resolution=0.01, radius=None, verbose=1, **kargs):
    X = X.toarray() if ssp.issparse(X) else X
    F = F.toarray() if ssp.issparse(F) else F

    sx = StandardScaler().fit_transform(X)
    sf = center_normalize(F)


    if method[0] == 'kmeans':
        print('KMeans')
        import os
        os.environ["OPENBLAS_NUM_THREADS"] = str(int(pnt))
        Z = np.hstack([np.sqrt(alpha[0]) * sx, np.sqrt(alpha[1]) * sf]).astype(np.float32)
        km = KMeans(n_clusters=k, random_state=0).fit(Z)
        labels_km = km.labels_
    else:
        labels_km = None

    if method[1] == 'leiden':
        print('Leiden')

        [src, dst, dist] = coord_edges(X, knn=knn, radius=radius,)
        Z = np.hstack([np.sqrt(beta[0]) * sx, np.sqrt(beta[1]) * sf]).astype(np.float32)
        D2 = ((Z[src] - Z[dst])**2).sum(1)
        temp = temp / D2.mean() 
        A = np.exp(-D2* temp )

        A = ssp.csr_array((A, (dst, src)), shape=(X.shape[0], X.shape[0])).astype(np.float32)
        if not A.has_sorted_indices:
            A.sort_indices()
        A.eliminate_zeros()

        # A = knnGraph( X, F, knn=knn, radius=radius, 
        #             temp=beta, # alpha=beta, use_delaunay=use_delaunay,
        #             normal=False, normal_F = False,  )
    
        # D = np.sum(A, axis=1)
        # D[D==0] = 1
        # A = A / D[:, None]

        if verbose:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(7, 3.))
            ax[0].set_title(f'D2 mean: {D2.mean() :.3e}')
            ax[0].hist(D2, bins=100)
            ax[1].set_title('A')
            ax[1].hist(A.data, bins=100)
            plt.show()

        import igraph as ig
        adj = A.tocsr()
        sources, targets = adj.nonzero()

        weights = adj[sources, targets]
        if isinstance(weights, np.matrix):
            weights = weights.A1
        G = ig.Graph(directed=None)
        G.add_vertices(adj.shape[0])
        G.add_edges(list(zip(sources, targets)))
        G.es['weight'] = weights

        cluster = leiden_graph(G, resolution=resolution, **kargs)
    else:
        cluster = None

    return labels_km, cluster

def Leiden(X, F, #C, mask=None, 
                alpha = 0.5,
                temp = [1.0,1.0],
                knn=None, radius=None, 
                normal=True, normal_F = True,
                resolution=0.002,
                use_delaunay=False, 
                verbose= 1, **kargs
                ):

    X = X.toarray() if ssp.issparse(X) else X
    F = F.toarray() if ssp.issparse(F) else F
    # mask = mask if mask is None else np.array(mask)
    
    W = knnGraph( X, F, knn=knn, radius=radius, 
                temp=temp, alpha=alpha, use_delaunay=use_delaunay,
                normal=normal, normal_F = normal_F,  )
    
    # Y, classes = onehots(C, mask=mask, normal=False) 
    # N, L = Y.shape
    # D_half = np.power(np.sum(W, axis=1), -0.5)
    # S = (W * D_half.reshape(-1, 1)) * D_half.reshape(1, -1)
    D = np.sum(W, axis=1)
    D[D==0] = 1
    S = W / D[:, None]

    if verbose:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(7, 3.))
        ax[0].set_title('W')
        ax[0].hist(W.data, bins=100)
        ax[1].set_title('S')
        ax[1].hist(S.data, bins=100)
        plt.show()

    import igraph as ig
    adj = S.tocsr()
    sources, targets = adj.nonzero()

    weights = adj[sources, targets]
    if isinstance(weights, np.matrix):
        weights = weights.A1
    G = ig.Graph(directed=None)
    G.add_vertices(adj.shape[0])
    G.add_edges(list(zip(sources, targets)))
    G.es['weight'] = weights

    cluster = leiden_graph(G, resolution=resolution, **kargs)
    return cluster, G

    
    # partition = ig.Graph.community_multilevel(G, weights='weight')

def label_propagation0(X, F, C, mask=None, 
                alpha = 0.5,
                temp = [1.0,1.0],
                knn=None, radius=None, 
                normal=True, normal_F = True,
                gamma =0.7, maxiter=500, verbose= 1):

    X = X.toarray() if ssp.issparse(X) else X
    F = F.toarray() if ssp.issparse(F) else F 
    mask = mask if mask is None else np.array(mask)
    
    W = knnGraph( X, F, knn=knn, radius=radius, 
                temp=temp, alpha=alpha,
                normal=normal, normal_F = normal_F,  )
    
    Y, classes = onehots(C, mask=mask, normal=False) 
    N, L = Y.shape
    # D_half = np.power(np.sum(W, axis=1), -0.5)
    # S = (W * D_half.reshape(-1, 1)) * D_half.reshape(1, -1)
    D = np.sum(W, axis=1)
    D[D==0] = 1
    S = W / D[:, None]

    if verbose:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].set_title('W')
        ax[0].hist(W.data, bins=100)
        ax[1].set_title('S')
        ax[1].hist(S.data, bins=100)
        plt.show()
    
    def solve_with_cg(A, Y, tol=1e-6, maxiter=500):
        """A: sparse matrix (csr), Y: numpy array (M x C)"""
        N, L = Y.shape
        F_star = np.zeros_like(Y)
        for c in tqdm(range(L), desc='Solving with CG'):
            f, info = ssp.linalg.cg(A, Y[:, c],  maxiter=maxiter)
            # if info != 0:
            #     print(f"CG did not converge for column {c}, maxiter={info}")
            F_star[:, c] = f
        return F_star
    A =  ssp.identity(N) - gamma * S
    F = solve_with_cg( A, Y, tol=1e-6, maxiter=maxiter)

    # # A = th.eye(len(C)) - gamma * S
    # # F_star = th.linalg.inv(A) @ Y

    prob = F
    pred_score = np.max(prob, axis=1)
    pred = np.argmax(prob, axis=1)
    pred_labels = classes[pred]

    if np.squeeze(C).ndim == 1:
        test_correct = pred_labels[mask] == C[mask]
        test_acc = int(test_correct.sum()) / int(mask.sum()) 
        print(f'Maks Accuracy: {test_acc:.4f}')

    return pred_labels, pred_score, F

def label_propagation(X, C, F=None, mask=None, 
                beta = 0.5,
                temp = 1, temp_f =1.0,
                knn=None, radius=None, 
                gamma =0.7, maxiter=500, verbose= 1):

    X = X.toarray() if ssp.issparse(X) else X
    F = F if F is None else (F.toarray() if ssp.issparse(F) else F) 
    mask = mask if mask is None else np.array(mask)
    
    W = knn_weight( X, F=F, knn=knn, radius=radius, 
                temp=temp,  temp_f=temp_f, beta=beta, verbose=verbose)
    
    Y, classes = onehots(C, mask=mask, normal=False) 
    N, L = Y.shape
    # D_half = np.power(np.sum(W, axis=1), -0.5)
    # S = (W * D_half.reshape(-1, 1)) * D_half.reshape(1, -1)
    D = np.sum(W, axis=1)
    D[D==0] = 1
    S = W / D[:, None]

    if verbose:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].set_title('W')
        ax[0].hist(W.data, bins=100)
        ax[1].set_title('S')
        ax[1].hist(S.data, bins=100)
        plt.show()
    
    def solve_with_cg(A, Y, tol=1e-6, maxiter=500):
        """A: sparse matrix (csr), Y: numpy array (M x C)"""
        N, L = Y.shape
        F_star = np.zeros_like(Y)
        for c in tqdm(range(L), desc='Solving with CG'):
            f, info = ssp.linalg.cg(A, Y[:, c],  maxiter=maxiter)
            # if info != 0:
            #     print(f"CG did not converge for column {c}, maxiter={info}")
            F_star[:, c] = f
        return F_star
    A =  ssp.identity(N) - gamma * S
    F = solve_with_cg( A, Y, tol=1e-6, maxiter=maxiter)

    # # A = th.eye(len(C)) - gamma * S
    # # F_star = th.linalg.inv(A) @ Y

    prob = F
    pred_score = np.max(prob, axis=1)
    pred = np.argmax(prob, axis=1)
    pred_labels = classes[pred]

    if np.squeeze(C).ndim == 1:
        test_correct = pred_labels[mask] == C[mask]
        test_acc = int(test_correct.sum()) / int(mask.sum()) 
        print(f'Maks Accuracy: {test_acc:.4f}')

    return pred_labels, pred_score, F

def label_diffusion(X, F, C, mask=None,
                alpha = 0.5,
                temp = [1.0,1.0],
                knn=None, radius=None, 
                normal=True, normal_F = True,
                gamma =0.7, maxiter=500, verbose= 1):
    X = X.toarray() if ssp.issparse(X) else X
    F = F.toarray() if ssp.issparse(F) else F
    mask = np.ones(X.shape[0], dtype=bool) if mask is None else np.array(mask)
    # mask = mask if mask is None else np.array(mask)

    W = knnGraph( X, F, mask=mask, knn=knn, radius=radius, 
                temp=temp, alpha=alpha,
                normal=normal, normal_F = normal_F,  )

    D = np.sum(W, axis=1)
    D[D==0] = 1
    S = W / D[:, None]

    if verbose:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].set_title('W')
        ax[0].hist(W.data, bins=100)
        ax[1].set_title('S')
        ax[1].hist(S.data, bins=100)
        plt.show()

    Y, classes = onehots(C, mask=mask, normal=False) 
    F =  S @ Y[mask]
    # pbar = tqdm(range(maxiter), total=maxiter,)
    # for it in pbar:
    #     F_old = F
    #     F = gamma * (S @ F_old)
    #     F[mask] = Y[mask]
        
    #     diff = np.linalg.norm(F - F_old) / np.linalg.norm(F_old + 1e-8)
    #     pbar.set_postfix({'diff': f"{diff:.4e}"})
    #     if diff < tol:
    #         if verbose:
    #             print(f"Converged at iteration {it}, diff={diff:.6f}")
    #         break
    Y[~mask] = F
    prob = Y
    pred_score = np.max(prob, axis=1)
    pred = np.argmax(prob, axis=1)
    pred_labels = classes[pred]

    return pred_labels, pred_score, F

def knnGraph(X,  F, mask=None, knn=None, radius=None, method='sknn', 
            temp=[1.0,1.0], alpha=0.5, verbose=1, use_delaunay=False,
            normal=True, normal_F = True, n_jobs = -1):

    if normal:
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X)
    else:
        X = X
    if normal_F:
        F = center_normalize(F)
    else:
        F = F
    
    if mask is not None:
        X1 = X[mask]
        F1 = F[mask]

        X2 = X[~mask]
        F2 = F[~mask]

        [src, dst, dist] = coord_edges(X1, coordy=X2, knn=knn, radius=radius,
                                        method=method, n_jobs=n_jobs)
    else:
        X1 = X
        F1 = F
        X2 = X
        F2 = F
        if use_delaunay:
            W = delaunay_adjacency(X1)
            src, dst = W.nonzero()
            dist = np.linalg.norm(X1[src] - X1[dst], axis=1)
        else:
            [src, dst, dist] = coord_edges(X1, knn=knn, radius=radius,
                                            method=method, n_jobs=n_jobs)
            # dist1 = np.linalg.norm(X1[src] - X1[dst], axis=1)
            # dist2 =  ((X1[src] - X1[dst])**2).sum(1)
            # print(dist, dist1, dist2)
    k_nodes, counts = np.unique(dst, return_counts=True)
    mean_neig = np.mean(counts)
    mean_radiu = np.mean(dist)

    if verbose:
        print(f'nodes: {X2.shape[0]}, edges: {len(dst)}\n'
            f'mean edges: {mean_neig :.3f}.\n'
            f'mean distance: {mean_radiu :.3e}.')

    # dist_f = ((F1[src] - F2[dst])**2).sum(1)
    dist_f = np.linalg.norm(F1[src] - F2[dst], axis=1)

    # dist   = dist**2 * temp[0] #/ dist.max()
    dist   *= temp[0]
    dist_f *= temp[1] #/dist_f.mean()

    if verbose:
        print(f'dist: {dist.min() :.3e}, {dist.max() :.3e}, {dist.mean() :.3e}\n'
                f'dist_f: {dist_f.min() :.3e}, {dist_f.max() :.3e}, {dist_f.mean() :.3e}')
    W = np.exp(-dist - dist_f)
    # W = alpha* np.exp(-dist) + (1-alpha) * np.exp(-dist_f)
    
    W = ssp.csr_array((W, (dst, src)), shape=(X2.shape[0], X1.shape[0]))
    if not W.has_sorted_indices:
        W.sort_indices()
    W.eliminate_zeros()
    return W

def knn_weight(X, F=None, knn=None, radius=None, method='sknn', fpfh=None, 
            temp=1.0, temp_f=1.0, beta = 0.5, alpha= [1,1], verbose=1, use_delaunay=False, 
            adapt_bandwidth=True,
            n_jobs = -1):
    use_F = not F is None
    if use_delaunay:
        W = delaunay_adjacency(X)
        src, dst = W.nonzero()
        dist = np.linalg.norm(X[src] - X[dst], axis=1)
    else:
        [src, dst, dist] = knn_edges(X, knn=knn, radius=radius,
                                        method=method, n_jobs=n_jobs)

    k_nodes, counts = np.unique(dst, return_counts=True)
    mean_neig = np.mean(counts)
    mean_radiu = np.mean(dist)

    if verbose:
        print(f'nodes: {X.shape[0]}, edges: {len(dst)}\n'
            f'mean edges: {mean_neig :.3f}.\n'
            f'mean distance: {mean_radiu :.3e}.')

    sx = scaler(X)
    dist_x  = np.linalg.norm(sx[src] - sx[dst], axis=1) **2
    sigma_x = np.median(dist_x) if adapt_bandwidth else 1.0 
    # sigma_x = sigma_square(sx, sx, xp=np)
    dist_x /= temp*sigma_x
    weig_x  = np.exp(-dist_x)

    if use_F:
        sf = scaler(F)
        dist_f = np.linalg.norm(sf[src] - sf[dst], axis=1) **2
        sigma_z = np.median(dist_f) if adapt_bandwidth else 1.0
        # sigma_z = sigma_square_cos(sf, sf, xp=np)
        dist_f /= temp_f*sigma_z
        weig_f  = np.exp(-dist_f)

        # W = np.exp(-dist - dist_f)
        W = beta * weig_x + (1- beta) * weig_f
        # Z = np.hstack([alpha[0] * sx, alpha[1] * sf])

        fig, axs = plt.subplots(2,2, figsize=(7.5, 6))
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
    else:
        W =  weig_x
        fig, axs = plt.subplots(1,2, figsize=(7.5,3))
        axs[0].hist(dist_x, bins=100)
        axs[1].hist(dist_f, bins=100)
        plt.show()

        if verbose:
            print(f'X: {sigma_x :.3e}, {dist_x.min() :.3e}, {dist_x.max() :.3e}, {dist_x.mean() :.3e}\n'
                  f'W: {W.min() :.3e}, {W.max() :.3e}, {W.mean() :.3e}'
            )
    W = ssp.csr_array((W, (dst, src)), shape=(X.shape[0], X.shape[0]))
    if not W.has_sorted_indices:
        W.sort_indices()
    W.eliminate_zeros()
    return W


def knn_edges(coordx, coordy=None,
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

def onehots(C,  mask= None, normal=False):
    if mask is None:
        mask = np.ones(C.shape[0], dtype=bool)
    if np.squeeze(C).ndim == 1:
        Cm = np.squeeze(np.array(C)[mask])
        order = np.unique(Cm)
        D = np.zeros((C.shape[0], len(order)))
        for i, k in enumerate(order):
            D[C == k, i] = 1
    else:
        Cm = np.array(C)[mask]
        Om = np.sum(Cm, axis=0) > 0

        if isinstance(C, pd.DataFrame):
            order = C.columns.values[Om]
        else:
            order = np.arange(Om.sum())
        
        D = np.array(C)[:, Om]

    D = D.astype(np.float64)
    if normal:
        D = D / D.sum(axis=1)[:, np.newaxis]
    return  D, order


def coord_edges0(coordx, 
                knn=50,
                radius=None,
                
                max_neighbor = int(1e4),
                method='sknn' ,
                keep_loops= False,
                n_jobs = -1):

    coordy = coordx
    cknn = Neighbors( method=method ,metric='euclidean', n_jobs=n_jobs)
    cknn.fit(coordx, radius_max= None,max_neighbor=max_neighbor)
    distances, indices = cknn.transform(coordy, knn=knn, radius = radius)

    src = np.concatenate(indices, axis=0).astype(np.int64)
    dst = np.repeat(np.arange(len(indices)), list(map(len, indices))).astype(np.int64)
    dist = np.concatenate(distances, axis=0)

    if not keep_loops:
        mask = src != dst
        src = src[mask]
        dst = dst[mask]
        dist = dist[mask]

    return [src, dst, dist]

def centerlize(X, Xm=None, Xs=None):
    if ssp.issparse(X): 
        X = X.toarray()

    N,D = X.shape
    Xm = np.mean(X, 0)

    X -= Xm
    Xs = np.sqrt(np.sum(np.square(X))/(N*D/2)) if Xs is None else Xs
    X /= Xs

    return [X, Xm, Xs]

def normalize(X):
    if ssp.issparse(X): 
        X = X.toarray()

    l2x = np.linalg.norm(X, ord=None, axis=1, keepdims=True)
    l2x[l2x == 0] = 1
    return X/l2x #*((self.DF/2.0)**0.5)

def center_normalize(X):
    if ssp.issparse(X): 
        X = X.toarray()

    X -= X.mean(axis=0, keepdims=True)
    l2x = np.linalg.norm(X, ord=None, axis=1, keepdims=True)
    l2x[l2x == 0] = 1
    return X/l2x 

def scaler( X):
    return (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)