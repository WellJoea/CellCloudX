import anndata as ad
import matplotlib.pyplot as plt
from skimage import filters
import collections
import torch as th
import numpy as np
import pandas as pd
from tqdm import tqdm

import scipy as sci
import scipy.sparse as ssp

from ._neighbors import Neighbors
from ._exp_edges import exp_similarity
from ._outlier import Invervals
from ._search import searchidx

from ..plotting._imageview import drawMatches
from ..utilis._arrays import list_iter
from ..utilis._arrays import list_iter, vartype
from ..alignment.ccflib.xmm import kernel_xmm_k



class LabelPropagator:
    def __init__(self, X, F, C, mask=None, mask_value=None, method='label_propagation', 
                 alpha = 0.5,
                 temp = [1.0,1.0],
                 knn=None, radius=None, 
                 normal=True, normal_F = True,
                 spatial_weight=0.7,  diffusion_time=5.0, reg=0.1, verbose=1):

        self.X = X.toarray() if ssp.issparse(X) else X
        self.F = F.toarray() if ssp.issparse(F) else F
        self.C = C
        self.classes = np.unique(C)
        self.n_class = len(self.classes)

        self.method = method
        self.spatial_weight = spatial_weight
        self.alpha = alpha
        self.temp = temp

        self.diffusion_time = diffusion_time
        self.reg = reg
        self.verbose = verbose

        self.label_binary(C, mask=mask, mask_value=mask_value)
        self.knnGraph( knn=knn, radius=radius, 
                        temp=temp,
                        normal=normal, normal_F = normal_F,  )

        D_half = np.power(np.sum(self.W, axis=1), -0.5)
        self.S = (self.W * D_half.reshape(-1, 1)) * D_half.reshape(1, -1)

        # indices = self.W._indices()
        # values = self.W._values()
        # row, col = indices[0], indices[1]
        # values = values * D_half[row] * D_half[col]
        # S1 = th.sparse_coo_tensor(indices, values, self.W.shape)
        # S2 =  th.diag(D_half) @ self.W.to_dense() @ th.diag(D_half)
        # print( (self.S.to_dense() - S2).sum(),22222222 )

    def label_diffusion(self, gamma =0.3, tol=1e-3, maxiter=100):
        N = len(self.C)
        Y = np.zeros((N, self.n_class), dtype=np.float64)
        for i, cls in enumerate(self.classes):
            Y[ (self.mask & (self.C == cls) ), i] = 1

        F = Y.copy()
        pbar = tqdm(range(maxiter), total=maxiter,)
        for it in pbar:
            F_old = F
            F = gamma * (self.S @ F_old)
            F[self.mask] = Y[self.mask]
            
            diff = np.linalg.norm(F - F_old) / np.linalg.norm(F_old + 1e-8)
            pbar.set_postfix({'diff': f"{diff:.4e}"})
            if diff < tol:
                if self.verbose:
                    print(f"Converged at iteration {it}, diff={diff:.6f}")
                break

        pbar.close()
        prob = F
        pred_score = np.max(prob, axis=1)
        pred = np.argmax(prob, axis=1)
        pred_labels = self.classes[pred]

        test_correct = pred_labels[self.mask] == self.C[self.mask]
        test_acc = int(test_correct.sum()) / int(self.mask.sum()) 
        print(f'Maks Accuracy: {test_acc:.4f}')

        return pred_labels, pred_score, F


    def label_propagation(self, gamma =0.3, maxiter=500):
        N = len(self.C)
        Y = np.zeros((N, self.n_class), dtype=np.float64)
        for i, cls in enumerate(self.classes):
            Y[ (self.mask & (self.C == cls) ), i] = 1

        def solve_with_cg(A, Y, tol=1e-6, maxiter=500):
            """A: sparse matrix (csr), Y: numpy array (M x C)"""
            M, C = Y.shape
            F_star = np.zeros_like(Y)
            for c in tqdm(range(C), desc='Solving with CG'):
                f, info = ssp.linalg.cg(A, Y[:, c],  maxiter=maxiter)
                # if info != 0:
                #     print(f"CG did not converge for column {c}, maxiter={info}")
                F_star[:, c] = f
            return F_star
        A =  ssp.identity(N) - gamma * self.S
        F_star = solve_with_cg( A, Y, tol=1e-6, maxiter=maxiter)

        # A = th.eye(len(self.C)) - gamma * self.S
        # F_star = th.linalg.inv(A) @ Y

        prob = F_star
        pred_score = np.max(prob, axis=1)
        pred = np.argmax(prob, axis=1)
        pred_labels = self.classes[pred]

        test_correct = pred_labels[self.mask] == self.C[self.mask]
        test_acc = int(test_correct.sum()) / int(self.mask.sum()) 
        print(f'Maks Accuracy: {test_acc:.4f}')

        return pred_labels, pred_score, F_star

    def label_binary(self, C, mask=None, mask_value = -1):
        lables = pd.Series(np.array(C))

        if mask is None:
            mask = lables != mask_value
        else:
            mask = mask
        mask = np.array(mask)
        
        m_labels, m_order = pd.factorize(lables[mask])
    
        self.C = lables.values
        self.mask = mask
        self.n_class = len(m_order)
        self.classes = m_order.values

    def knnGraph(self, knn=None, radius=None, method='sknn', 
                temp=[1.0,1.0],
                normal=True, normal_F = True, n_jobs = -1):

        if normal:
            X = centerlize(self.X)[0]
        else:
            X = self.X
        if normal_F:
            F = center_normalize(self.F)
        else:
            F = self.F

        N = X.shape[0]
        [src, dst, dist] = coord_edges(X, knn=knn, radius=radius,
                                        method=method, n_jobs=n_jobs)
        
        k_nodes, counts = np.unique(dst, return_counts=True)
        mean_neig = np.mean(counts)
        mean_radiu = np.mean(dist)

        if self.verbose:
            print(f'nodes: {N}, edges: {len(dst)}\n'
                f'mean edges: {mean_neig :.3e}.\n'
                f'mean distance: {mean_radiu :.3e}.')

        dist_f = ((F[src] - F[dst])**2).sum(1)

        dist   *= temp[0] #/ dist.max()
        dist_f *= temp[1] #/dist_f.mean()

        if self.verbose:
            print(f'dist: {dist.min() :.3e}, {dist.max() :.3e}, {dist.mean() :.3e}\n'
                  f'dist_f: {dist_f.min() :.3e}, {dist_f.max() :.3e}, {dist_f.mean() :.3e}')
        # W = np.exp(-dist - dist_f)
        W = self.alpha* np.exp(-dist) + (1-self.alpha) * np.exp(-dist_f)
 
        W = ssp.csr_array((W, (dst, src)), shape=(N,N))
        if not W.has_sorted_indices:
            W.sort_indices()
        W.eliminate_zeros()
        self.W = W


    def onehots(self, C, normal=False):
        if np.squeeze(C).ndim == 1:
            C = np.squeeze(C)
            order = np.unique(C)
            D = np.zeros((C.shape[0], len(order)))
            for i, k in enumerate(order):
                D[C == k, i] = 1
        else:
            if isinstance(C, pd.DataFrame):
                order = C.columns.values
            else:
                order = np.arange(C.shape[1])
            
            D = np.array(C)

        if normal:
            D = D / D.sum(axis=1)[:, np.newaxis]
        return D, order

    def label_vector(self, lables, mask_value = -1):
        lables = pd.Series(np.array(lables))
        mask = lables != mask_value
        
        m_labels, m_order = pd.factorize(lables[mask])
        n_labels = np.ones(lables.shape[0]) * len(m_order)
        n_labels[mask] = m_labels
        
        n_labels = np.int64(n_labels)
        m_order = np.array(m_order)
        return n_labels, m_order

    def sparse_svd(self, S, k=100): #PASS error
        import scipy.sparse as sp
        S_coo = S.coalesce()
        values = S_coo.values().cpu().numpy()
        indices = S_coo.indices().cpu().numpy()
        m, n = S.shape
        S_sp = sp.coo_matrix((values, (indices[0], indices[1])), shape=(m, n))

        U, Q, Vh = sp.linalg.svds(S_sp, k=k,  which='LM',)
        print((S.to_dense().cpu().numpy() , U @ np.diag(Q) @ Vh), Q)
        return (
            th.asarray(U.copy(), device=self.device, dtype=th.float64),
            th.asarray(Q.copy(), device=self.device, dtype=th.float64),
            th.asarray(Vh.copy(), device=self.device, dtype=th.float64),
        )

def spsparse_to_thsparse(X):
    import torch as th
    XX = X.tocoo()
    values = XX.data
    indices = np.vstack((XX.row, XX.col))
    i = th.LongTensor(indices)
    v = th.tensor(values, dtype=th.float64)
    shape = th.Size(XX.shape)
    return th.sparse_coo_tensor(i, v, shape)

def thsparse_to_spsparse(X):
    XX = X.to_sparse_coo().coalesce()
    values = XX.values().detach().cpu().numpy()
    indices = XX.indices().detach().cpu().numpy()
    shape = XX.shape
    return ssp.csr_array((values, indices), shape=shape)

def coord_edges(coordx, 
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

