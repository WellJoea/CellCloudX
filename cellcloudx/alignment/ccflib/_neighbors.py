import numpy as np
from scipy.sparse import issparse, csr_array, diags, csc_array
import random
import os

class ko_knn():
    def __init__(self, X, Y=None, K=60):
        import pykeops
        pykeops.set_verbose(False)
        from pykeops.torch import LazyTensor
        import torch as th
        self.xp = th
        self.LazyTensor = LazyTensor

        self.X = X
        self.Y = Y
        self.K = K
        self.N, self.D = X.shape

        if Y is None:
            self.Y = X
            self.M = self.N
        else:
            self.M = Y.shape[0]
            self.Y = Y
        self.dtype = X.dtype
        self.device = X.device

    def knn(self, return_distances=False, return_sparse=False,):
        Xi = self.LazyTensor(self.X[:, None, :])
        Yj = self.LazyTensor(self.Y[None, :, :])
        D_ij = ((Xi - Yj) ** 2).sum(-1)
        I = D_ij.argKmin(self.K, dim=0)
        self.src, self.dst = self.kidx(I)

        if return_distances:
            D = self.kdist().view(*I.shape)
            # D = D.view(*I.shape).sort(1)
            # I = I[th.arange(I.shape[0])[:, None], D.indices]
            # D = D.values

        if return_sparse:
            I = self.tosparse()
            if return_distances:
                D = self.tosparse(D)

        if return_distances:
            return I, D
        else:
            return I

    def kidx(self, I):
        M, K = I.shape
        src = I.flatten(0).to(self.xp.int64)
        dst = self.xp.repeat_interleave(self.xp.arange(M, dtype=self.xp.int64), K, dim=0).to(I.device)
        return (src, dst)

    def kdist(self):
        return ((self.X[self.src, :] - self.Y[self.dst, :])**2).sum(-1)

    def tosparse(self, value=None, src=None, dst=None, dtype=None, device=None):
        if value is None:
            if not src is None:
                value = self.xp.ones(src.shape[0])
            elif not dst is None:
                value = self.xp.ones(dst.shape[0])
            else:
                value = self.xp.ones(self.src.shape[0])

        src = self.src if src is None else src
        dst = self.dst if dst is None else dst
        value = value if value.ndim == 1 else value.flatten()
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        V = self.xp.sparse_coo_tensor( 
                    self.xp.vstack([dst, src]), value, 
                    size=(self.M, self.N), 
                    dtype=dtype,
                    device=device)
        V = V.coalesce()
        return V

    def resparse(self, I, D):
        return self.tosparse(D, *self.kidx(I))

class Neighbors():

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
        elif self.method == 'konn': #TODO
            import pykeops
            pykeops.set_verbose(False)
            from pykeops.torch import LazyTensor
            self.LazyTensor = LazyTensor
        elif self.method == 'cuKDTree': #TODO
            from cupyx.scipy.spatial import KDTree
            self.ckdmethod = KDTree
        elif self.method == 'faiss':
            import faiss
            self.ckdmethod = faiss.IndexFlatL2
        elif self.method == 'faisscu':
            error = 'faiss is not installed. Please install it by `pip install faiss-gpu-cu11` or `pip install faiss-gpu-cu12`'
            try:
                import faiss
                faiss.GpuIndex
            except (ImportError, AttributeError):
                raise ImportError(error)

            def ckdmethod(d):
                # res = faiss.StandardGpuResources()
                # ckd = faiss.GpuIndexFlatL2(res, d)

                ckd = faiss.IndexFlatL2(d)
                ckd = faiss.index_cpu_to_all_gpus(ckd)
                return ckd
            self.ckdmethod = ckdmethod
    
        elif self.method == 'pynndescent':
            from pynndescent import NNDescent
            self.ckdmethod = NNDescent
        else:
            raise ValueError('Invalid method: {}'.format(self.method))

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

        elif  self.method == 'faisscu':
            ckd = self.ckdmethod(data.shape[1])
            ckd.add(data.cpu())

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

        elif self.method in ['faiss']:
            D, I = ckd.search(data, knn)
            D[D<0] = 0
            ckdout = [np.sqrt(D), I]
        elif self.method in ['faisscu']:
            D, I = ckd.search(data.cpu(), knn)
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
        return ckdout

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