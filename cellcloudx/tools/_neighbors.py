import numpy as np
from scipy.sparse import issparse, csr_array, diags, csc_array
import random
import os
from sklearn.metrics.pairwise import paired_cosine_distances

import matplotlib.pyplot as plt
class Neighbors():
    '''
    https://github.com/erikbern/ann-benchmarks
    '''
    def __init__(self, method='hnsw',
                  metric='euclidean',
                  device_index =None,
                  n_jobs=-1):
        self.method = method
        self.metric = metric
        self.n_jobs = n_jobs
        self.device_index = device_index if device_index is not None else 0
        self.ckd = None

        self.ckdmethod = None
        if self.method == 'hnsw':
            import hnswlib
            self.ckdmethod = hnswlib.Index
        elif self.method == 'annoy':
            from annoy import AnnoyIndex
            self.ckdmethod = AnnoyIndex
        elif self.method == 'sknn':
            from sklearn.neighbors import NearestNeighbors as sknn
            self.ckdmethod = sknn
        elif self.method == 'cunn':
            from cuml.common.device_selection import using_device_type, set_global_device_type, get_global_device_type
            from cuml.neighbors import NearestNeighbors as cunn
            import cupy as cp
            set_global_device_type("gpu")
            self.using_device_type = using_device_type
            self.set_global_device_type = set_global_device_type
            self.set_global_device_type("gpu")
            self.get_global_device_type = get_global_device_type
            self.ckdmethod = cunn
            self.cp = cp
        elif self.method == 'cuKDTree': #TODO
            from cupyx.scipy.spatial import KDTree
            self.ckdmethod = KDTree
        elif self.method == 'faiss':
            import faiss
            self.ckdmethod = faiss.IndexFlatL2
        elif self.method == 'faisscu':
            try:
                import faiss
                self.ckdmethod = faiss.IndexFlatL2
            except:
                print('faiss is not installed. '
                      'Please install it by `pip install faiss-gpu-cu11` or `pip install faiss-gpu-cu12`')

        elif self.method == 'pynndescent':
            from pynndescent import NNDescent
            self.ckdmethod = NNDescent

    def fit(self, data, hnsw_space='l2',
            seed=200504,
            radius_max= None,
            max_neighbor=None,
            algorithm='auto',
            metric = 'minkowski',
            p = 2,
            max_elements=None, ef_construction=200, M=20,
            annoy_n_trees=70, pynndescent_n_neighbors=50):
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        data_labels = np.arange(data.shape[0])
        if self.method == 'hnsw':
            ckd = self.ckdmethod(space=hnsw_space, dim=data.shape[1])
            ckd.init_index(max_elements = data.shape[0] if max_elements is None else max_elements, 
                            ef_construction = ef_construction, M = M,  random_seed =seed)
            ckd.add_items(data, data_labels, num_threads = self.n_jobs)

        elif self.method == 'annoy':
            ckd = self.ckdmethod(data.shape[1], metric=self.metric)
            ckd.set_seed(seed)
            for i in np.arange(data.shape[0]):
                ckd.add_item(i,data[i,:])
            ckd.build(annoy_n_trees)

        elif self.method in ['sknn']:
            ckd = self.ckdmethod(radius=radius_max ,
                        n_neighbors=max_neighbor, 
                        p=p,
                        n_jobs=self.n_jobs,
                        algorithm=algorithm, metric=metric)
            ckd.fit(data)
        elif self.method in ['cunn']:
            with self.cp.cuda.Device(self.device_index):
                data = self.cp.array(data)
            with self.using_device_type("gpu"):
                ckd = self.ckdmethod(
                            n_neighbors=max_neighbor, 
                            p=p,
                            algorithm=algorithm, metric=metric)
                ckd.fit(data)
        elif  self.method == 'faiss':
            ckd = self.ckdmethod(data.shape[1])
            ckd.add(data)
        elif  self.method == 'pynndescent':
            ckd = self.ckdmethod(data, metric=self.metric,
                                    n_jobs=self.n_jobs,
                                    n_neighbors=pynndescent_n_neighbors, 
                                    random_state=seed)
            ckd.prepare()

        elif self.method == 'cKDTree':
            from scipy.spatial import cKDTree
            ckd = cKDTree(data)

        elif self.method == 'KDTree':
            from sklearn.neighbors import KDTree
            ckd = KDTree(data,metric=self.metric)

        elif self.method == 'ngtpy':
            import ngtpy
            ngtpy.create(b"tmp", data.shape[1])
            index = ngtpy.Index(b"tmp")
            index.batch_insert(data)
            index.save()
            results = index.search(query, 3)
            for i, (id, distance) in enumerate(results) :
                print(str(i) + ": " + str(id) + ", " + str(distance))
                object = index.get_object(id)
                print(object)

        self.ckd = ckd


    def transform(self, data,  ckd=None, knn=20, set_ef=60, 
                  radius = None, 
                  search_k=-1, sort_dist=False,
                  include_distances=True):
        ckd = self.ckd if ckd is None else ckd
        if self.method == 'hnsw':
            ckd.set_ef(max(set_ef, knn+10)) #ef should always be > k
            ckd.set_num_threads(self.n_jobs)
            labels, distances = ckd.knn_query(data, k = knn, num_threads=self.n_jobs)
            ckdout = [np.sqrt(distances), labels] if ckd.space =='l2' else [distances, labels]

        elif self.method == 'annoy':
            ckdo_ind = []
            ckdo_dist = []
            for i in np.arange(data.shape[0]):
                holder = ckd.get_nns_by_vector(data[i,:],
                                                knn,
                                                search_k=search_k,
                                                include_distances=include_distances)
                ckdo_ind.append(holder[0])
                ckdo_dist.append(holder[1])
            ckdout = [np.asarray(ckdo_dist),np.asarray(ckdo_ind)]

        elif self.method == 'sknn': 
            # distance neighbors
            if not radius is None:
                sort_dist = False
                distances, indices = ckd.radius_neighbors(data, radius, return_distance=True)
                ckdout = [distances, indices]
            else:
                distances, indices = ckd.kneighbors(data, knn, return_distance=True)
                ckdout = [distances, indices]

        elif self.method == 'cunn':
            with self.cp.cuda.Device(self.device_index):
                data = self.cp.array(data)
            with self.using_device_type("gpu"):
                distances, indices = ckd.kneighbors(data, knn, return_distance=True)
                ckdout = [distances, indices]

        elif self.method == 'pynndescent':
            ckdout = ckd.query(data, k=knn)
            ckdout = [ckdout[1], ckdout[0]]

        elif self.method == 'faiss':
            D, I = ckd.search(data, knn)
            D[D<0] = 0
            ckdout = [np.sqrt(D), I]
        elif self.method == 'cKDTree':
            ckdout = ckd.query(x=data, k=knn, p=2, workers=self.n_jobs)
        elif self.method == 'KDTree':
            ckdout = ckd.query(data, k=knn)

        if sort_dist and ((ckdout[0][:,1:] - ckdout[0][:,:-1]).min()<0):
            idxsort = ckdout[0].argsort(axis=1)
            ckdout[0] = np.take_along_axis(ckdout[0], idxsort, 1)
            ckdout[1] = np.take_along_axis(ckdout[1], idxsort, 1)
        return list(ckdout)

    @staticmethod
    def neighbors(cdkout, n_obs=None, n_neighbors=15, self_weight = 0):
        n_obs = n_obs or cdkout[0].shape[0]
        adj = Neighbors.fuzzy_connectivities(None, knn_indices=cdkout[1], knn_dists=cdkout[0],
                                             n_obs= n_obs, #adatai.shape[0],
                                             random_state=None,
                                             n_neighbors=n_neighbors)
        if self_weight:
            adj = Neighbors.set_diagonal(adj, val=self_weight)
        dist = Neighbors.translabel(cdkout, return_type='sparse', rsize=n_obs)
        return dist, adj

    @staticmethod
    def translabel(ckdout, rsize=None, rlabel=None, qlabel=None, return_type='raw'):
        nnidx = ckdout[1]
        # minrnum = np.int64(nnidx.max()- nnidx.min()+1)
        minrnum = np.unique(nnidx).shape[0]
        if not rlabel is None:
            assert len(rlabel) >= minrnum
        if not rsize is None:
            assert rsize >= minrnum
        if not qlabel is None:
            assert len(qlabel) == len(nnidx)
        
        if return_type == 'raw':
            if  ( rlabel is None):
                return [ckdout[0], nnidx]
            else:
                rlabel = np.asarray(rlabel)
                return [ckdout[0], rlabel[nnidx]]
        elif return_type in ['lists', 'sparse', 'sparseidx']:
            try:
                src = nnidx.flatten('C')
                dst = np.repeat(np.arange(nnidx.shape[0]), nnidx.shape[1])
                dist = ckdout[0].flatten('C')
            except:
                src = np.concatenate(nnidx, axis=0)
                dst = np.repeat(np.arange(len(nnidx)), list(map(len, nnidx)))
                dist = np.concatenate(ckdout[0], axis=0)

            if return_type in ['sparse', 'sparseidx']:
                rsize = rsize or (None if rlabel is None else len(rlabel)) or minrnum ## set fixed value
                if return_type == 'sparseidx':
                    dist = np.ones_like(dst)
                adj = csr_array((dist, (dst, src)), shape=(nnidx.shape[0], rsize))
                if not adj.has_sorted_indices:
                    adj.sort_indices()
                adj.eliminate_zeros()
                return adj
            else:
                if not rlabel is None:
                    src = np.asarray(rlabel)[src]
                if not qlabel is None:
                    dst = np.asarray(qlabel)[dst]
                return [src, dst, dist]
        else:
            raise ValueError('return_type must be one of "raw", "lists", "sparse", "sparseidx"')
    
    @staticmethod
    def fuzzy_connectivities(X, knn_indices=None, knn_dists=None,
                                    n_obs=None,
                                    random_state=200504, metric=None, # euclidean 
                                    n_neighbors=15, set_op_mix_ratio=1.0,
                                    local_connectivity=1.0):
        from scipy.sparse import coo_matrix
        from umap.umap_ import fuzzy_simplicial_set
        if X is None:
            X = coo_matrix(([], ([], [])), shape=(n_obs, 1))
        connectivities = fuzzy_simplicial_set(X, n_neighbors, random_state, metric,
                                            knn_indices=knn_indices, knn_dists=knn_dists,
                                            set_op_mix_ratio=set_op_mix_ratio,
                                            local_connectivity=local_connectivity)
        if isinstance(connectivities, tuple):
            connectivities = connectivities[0]

        connectivities= connectivities.tocsr()
        connectivities.eliminate_zeros()
        return connectivities

    @staticmethod
    def set_diagonal(mtx, val=1, inplace=False):
        assert mtx.shape[0] == mtx.shape[1], "Matrix must be square"
        if issparse(mtx):
            diamtx = diags(val- mtx.diagonal(), dtype=mtx.dtype)
            mtx = mtx + diamtx
            mtx.sort_indices()
            mtx.eliminate_zeros()
            return mtx

        elif isinstance(mtx, np.ndarray):
            mtx = mtx if inplace else mtx.copy()
            np.fill_diagonal(mtx, val)
            return mtx

def mtx_similarity(mtxa, mtxb, method = 'cosine', kdt_method = 'sknn', kdt_metric='euclidean',
                   self_weight = 0, n_neighbors = 15, pairidx = None, eps = 1e-8, n_jobs=-1, 
                   chunck=None):
    if method == 'cosine':
        if pairidx is None:
            l2a =  np.linalg.norm(mtxa, ord=None, axis=1)[:, np.newaxis]
            l2b =  np.linalg.norm(mtxb, ord=None, axis=1)[:, np.newaxis]
            l2a[l2a< eps] = eps
            l2b[l2b< eps] = eps
            mtxa = mtxa / l2a
            mtxb = mtxb / l2b
            return mtxa @ mtxb.T
        else:
            from sklearn.metrics.pairwise import paired_cosine_distances
            src = np.int64(pairidx[0])
            dst = np.int64(pairidx[1])
            cdist = paired_cosine_distances(mtxa[src], mtxb[dst])
            return 1-cdist
            # l2a =  np.linalg.norm(mtxa, ord=None, axis=1)[:, np.newaxis]
            # l2b =  np.linalg.norm(mtxb, ord=None, axis=1)[:, np.newaxis]
            # l2a[l2a< eps] = eps
            # l2b[l2b< eps] = eps
            # mtxa = mtxa / l2a
            # mtxb = mtxb / l2b
            # if chunck is None:
            #     return np.sum(mtxa[pairidx[0]] * mtxb[pairidx[1]], axis=1)
            # else:
            #     src = np.int64(pairidx[0])
            #     dst = np.int64(pairidx[1])
            #     split_range = np.arange(0, src.shape[0]+chunck, chunck).clip(0, src.shape[0])
            #     simis = []
            #     for idx in range(len(split_range)-1):
            #         islice = slice(split_range[idx], split_range[idx+1])
            #         isimi = np.sum(mtxa[src[islice]] * mtxb[dst[islice]], axis=1)
            #         simis.append(isimi)
            #     simis = np.concatenate(simis)
            #     return simis

    elif method == 'pearson':
        mtxa = mtxa - mtxa.mean(1)[:, None]
        mtxb = mtxb - mtxb.mean(1)[:, None]
        stda = np.sqrt(np.sum(np.square(mtxa), axis=1))
        stdb = np.sqrt(np.sum(np.square(mtxb), axis=1))
        stda[stda< eps] = eps
        stdb[stdb< eps] = eps
        mtxa = mtxa/stda[:, None]
        mtxb = mtxb/stdb[:, None]
        if pairidx is None:
            return mtxa @ mtxb.T
        else:
            return np.sum(mtxa[pairidx[0]] * mtxb[pairidx[1]], axis=1)

    elif method == 'fuzzy':
        if pairidx is None:
            if  mtxb is None:
                mtx = mtxa
            else:
                mtx = np.concatenate([mtxa, mtxb], axis=0)
            snn = Neighbors(method=kdt_method, metric=kdt_metric, n_jobs=n_jobs)
            snn.fit(mtx)
            cdkout = snn.transform(mtx, knn=n_neighbors)
            dist, adj = snn.neighbors(cdkout, n_neighbors=n_neighbors, self_weight=self_weight)
            return adj, dist
        else:
            src = pairidx[0]
            dst = pairidx[1]
            dist = np.linalg.norm(mtxa[src]-mtxa[dst], axis=1)
            # to redce complexity, we assumen edges come to the same matrix
            # TODO
            idx, counts = np.unique(dst, return_counts=True)
            assert (idx.shape[0] == mtxa.shape[0]) and len(set(counts)) == 1
            n_obs = mtxa.shape[0]
            n_nei = np.int64(counts[0])

            sort_idx = np.argsort(dst)
            ckd_idx = src[sort_idx].reshape(mtxa.shape[0], -1, order='C')
            ckd_dist = dist[sort_idx].reshape(mtxa.shape[0], -1, order='C')
            assert np.all(ckd_idx.shape == (n_obs, n_nei))
            dist, adj = Neighbors.neighbors([ckd_dist, ckd_idx], 
                                            n_neighbors=min(n_nei, n_neighbors), 
                                            self_weight=self_weight)
            return adj, dist

def edge_neighbors(adata,
                    method='annoy', n_jobs= -1,
                    metric='euclidean', 
                    use_rep = None,
                    edges = None,
                    n_pcs=100, 
                    key_added = None,
                    inplace = True,
                    n_neighbors=15):
    import scanpy as sc
    adata = adata if inplace else adata.copy()

    X = sc.tl._utils._choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs, silent=True)
    adj, dist = mtx_similarity(X, None, kdt_method = method, n_neighbors = n_neighbors, 
                                method = 'fuzzy',
                                pairidx = edges, n_jobs=n_jobs)

    if not edges is None:
        n_neighbors = min(n_neighbors, np.median(np.unique(edges[1], return_counts=True)[1]))

    if key_added is None:
        key_added = 'neighbors'
        conns_key = 'connectivities'
        dists_key = 'distances'
    else:
        conns_key = key_added + '_connectivities'
        dists_key = key_added + '_distances'

    adata.obsp[dists_key] = dist
    adata.obsp[conns_key] = adj
    adata.uns[key_added] = {'connectivities_key':conns_key, 
                            'distances_key':dists_key,
                            'params':{'n_neighbors': n_neighbors,
                                       'method': 'umap',
                                       'metric': metric,
                                       'use_rep': use_rep,
                                       'n_pcs':n_pcs,
                            }}
    if not inplace:
        return adata

def fuzzy_connectivities(*args, kargs):
    return Neighbors.fuzzy_connectivities(*args, **kargs)