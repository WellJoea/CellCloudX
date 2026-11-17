import numpy as np
from sklearn.base import ClusterMixin, BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.sparse import csr_matrix, spdiags

class trans_labels():
    def __init__(self, ref_data, method='hnsw', n_jobs=-1):
        self.ref_data = ref_data
        self.nbrs = cc.tl.Neighbors(method=method, n_jobs=n_jobs)
        self.nbrs.fit(self.ref_data)

    def neighbor(self, query_data, knn=5, set_ef=60):
        self.ckdout = self.nbrs.transform(query_data, knn=knn, set_ef=set_ef )
        self.knn= knn
        return self.ckdout

    def trim_neighbor(self, query_data, knn=1, set_ef=60, 
                      radius=None, show_plot=True, doknn=True,
                      CI = 0.95, kernel='poi'):
        if not hasattr(self, 'ckdout') and doknn:
            self.neighbor(query_data, knn=knn, set_ef=set_ef)

        (distances, indices) = self.ckdout
        src = np.concatenate(indices, axis=0)
        dst = np.repeat(np.arange(indices.shape[0]), list(map(len, indices)))
        dist = np.concatenate(distances, axis=0)

        if radius is None:
            radius = cc.tl.Invervals(dist, CI = CI, tailed ='two', kernel=kernel)[-1]

        keep_idx = (dist<=radius)
        dst_k = dst[keep_idx]
        src_k = src[keep_idx]
        dist_k = dist[keep_idx]

        self.dst_k =dst_k 
        self.src_k =src_k
        self.dist_k =dist_k
        self.radius = radius

        if show_plot:
            bins = max(knn,2)
            nnodes = distances.shape[0]
            keep_idx = (dist<=radius)
            _, countr = np.unique(dst, return_counts=True)
            _, countk = np.unique(dst_k, return_counts=True)
            mean_neir = np.mean(countr)
            mean_neik = np.mean(countk)
            nnodes = distances.shape[0]
            knodes = len(set(dst_k))

            fig, ax = plt.subplots(1, 2, figsize=((2+0.5)*3, 3))
            ax[0].hist(dist, histtype='barstacked', bins=100, facecolor='r', alpha=1)
            if not radius is None:
                ax[0].axvline(radius, color='black', 
                                label=f'radius: {radius :.3f}\nnodes: {nnodes}\nkeep nodes: {knodes}')
                ax[0].legend()
            ax[0].set_title(f'distance distribution')

            ax[1].hist(countr, bins=bins, facecolor='b', 
                        label=f'mean neighbors:{mean_neir :.3f}' )
            ax[1].hist(countk, bins=bins, facecolor='r', 
                        label=f'mean keep neighbors:{mean_neik :.3f}' )

            ax[1].legend()
            ax[1].set_title(f'mean neighbor distribution')

            plt.tight_layout()
            plt.show()

    def predict_label(self, ref_label, agg = 'mean'):
        assert len(ref_label) == self.ref_data.shape[0]
        labels = np.array(ref_label)[self.src_k] # knn = 1
        # TO DO, torchc scatter
        return labels

        # keep_idx =contains & (adata.obs['annot'] !='Clear Label')
        # ref_data = adata[keep_idx, :].obsm['align3d1']

        # transl = trans_labels(ref_data)
        # transl.trim_neighbor(inter_points_sub, knn=1)


# https://github.com/albert-espin/snn-clustering

def snn_dissimilarity_func(graph : csr_matrix, n_neighbors : int, *args, **kwargs) -> csr_matrix:
    """Default SNN dissimilarity function

    Computes the dissimilarity between two points in terms of shared nearest neighbors

    Args:
        graph (scipy.sparse.csr_matrix): sparse matrix with dimensions (n_samples, n_samples),
         where the element ij represents the distance between the point i and j 
        n_neighbors (int): number of neighbors in the k-neighborhood search
    """ 

    graph.data[graph.data > 0] = 1
    n_samples = graph.shape[0]

    # Add the point as its own neighbor
    #graph += spdiags(np.ones(n_samples), diags=0, m=n_samples, n=n_samples)
    graph.setdiag(1)
    matrix = graph * graph.transpose()
    matrix.sort_indices()

    # The lower the "closer"
    matrix.data = n_neighbors - matrix.data

    return matrix

class SNN(ClusterMixin, BaseEstimator):
    def __init__(
        self,
        n_neighbors=7,
        eps=5,
        min_samples=5,
        algorithm="auto",
        leaf_size=30,
        metric="euclidean",
        p=None,
        metric_params=None,
        dissimilarity_func=snn_dissimilarity_func,
        n_jobs=None,
    ):
        """Shared Nearest Neighbor clustering  algorithm for finding clusters or different sizes, shapes and densities in
        noisy, high-dimensional datasets.


        The algorithm can be seen as a variation of DBSCAN which uses neighborhood similairty as a metric.
        It does not have a hard time detecting clusters of different densities as DBSCAN, and keeps its advantages.


        Parameters
        ----------
        n_neighbors : int, optional
            The number of neighbors to construct the neighborhood graph, including the point itself. By default 7

        eps : int, optional
            The minimum number of neighbors two points have to share in order to be
            connected by an edge in the neighborhood graph. This value has to be smaller
            than n_neighbors. By default 5

        min_samples : int, optional
            The number of samples (or total weight) in a neighborhood for a point
            to be considered as a core point. This includes the point itself, by default 5

        algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
            The algorithm to be used by the NearestNeighbors module
            to compute pointwise distances and find nearest neighbors.
            See NearestNeighbors module documentation for details., by default "auto"

        leaf_size : int, optional
            [description], by default 30

        metric : str, or callable
            The metric to use when calculating distance between instances in a
            feature array. If metric is a string or callable, it must be one of
            the options allowed by :func:`sklearn.metrics.pairwise_distances` for
            its metric parameter.
            If metric is "precomputed", X is assumed to be a distance matrix and
            must be square. X may be a :term:`Glossary <sparse graph>`, in which
            case only "nonzero" elements may be considered neighbors for DBSCAN.
            Default to "euclidean"

        p : int, optional
            The power of the Minkowski metric to be used to calculate distance
            between points. If None, then ``p=2`` (equivalent to the Euclidean
            distance).

        metric_params : [type], optional
            Additional keyword arguments for the metric function., by default None

        n_jobs : int, optional
            The number of parallel jobs to run.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details. Default None.

        dissimilarity_func: Callable, optional
            A function that receives two inputs: scipy.sparse.csr_matrix with the k-neighbors distance and the n_neighbors attribute;
            and returns another csr_matrix

        Attributes
        ----------
        
        neigh : sklearn.neighbors.NearestNeighbors 
        
        dbscan : sklearn.cluster.DBSCAN

        labels_ : ndarray of shape (n_samples)
            Cluster labels for each point in the dataset given to fit().
            Noisy samples are given the label -1.
            
        components_ : ndarray of shape (n_core_samples, n_features)

        Copy of each core sample found by training.

        core_samples_indices_ : ndarray of shape (n_core_samples,)
            Indices of core samples.

        dissimilarity_matrix : scipy.sparse.csr_matrix 
            containing the dissimilarity between points

        References
        ----------

        Ertöz, L., Steinbach, M., & Kumar, V. (2003, May). Finding clusters of different sizes, shapes, and densities in noisy, high dimensional data. In Proceedings of the 2003 SIAM international conference on data mining (pp. 47-58). Society for Industrial and Applied Mathematics.
        Ertoz, Levent, Michael Steinbach, and Vipin Kumar. "A new shared nearest neighbor clustering algorithm and its applications." Workshop on clustering high dimensional data and its applications at 2nd SIAM international conference on data mining. 2002.


        """

        if eps <= 0:
            raise ValueError("Eps must be positive.")
        if eps >= n_neighbors and dissimilarity_func == snn_dissimilarity_func:
            raise  ValueError("Eps must be smaller than n_neighbors.")

        self.eps = eps
        self.min_samples = min_samples
        self.n_jobs = n_jobs
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.dissimilarity_func = dissimilarity_func
        self.neigh = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            n_jobs=self.n_jobs,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            p=self.p,
            metric_params=self.metric_params,
        )

        # Reasoning behind eps=self.n_neighbors - self.eps:
        # In DBSCAN, eps is an upper bound of the distance between two points.
        # In terms of similarity, it would an "lower bound" on the similarity
        # or, once again, upper bound on the difference between the max similarity value and
        # the similarity between two points
        self.dbscan = DBSCAN(
            eps=self.n_neighbors - self.eps,
            min_samples=self.min_samples,
            metric="precomputed",
            n_jobs=self.n_jobs,
        )

    def fit(self, X, y=None, sample_weight=None):
        """Perform SNN clustering from features or distance matrix

        First calls NearestNeighbors to construct the neighborhood graph considering the params
        n_neighbors, n_jobs, algorithm, leaf_size, metric, p, metric_params

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``metric='precomputed'``. If a sparse matrix is provided, it will
            be converted into a sparse ``csr_matrix``.

        y : Ignored
            Not used, present here for API consistency by convention.
        sample_weight : array-like of shape (n_samples,), default=None
            Weight of each sample, such that a sample with a weight of at least
            ``min_samples`` is by itself a core sample; a sample with a
            negative weight may inhibit its eps-neighbor from being core.
            Note that weights are absolute, and default to 1.

        Returns
        -------
        self : object
            Returns a fitted instance of self.
        """

        self.dissimilarity_matrix = self.neighborhood_dissimilarity_matrix(X)

        self.dbscan.fit(self.dissimilarity_matrix, sample_weight=sample_weight)

        return self

    @property
    def labels_(self):
        return self.dbscan.labels_
    
    @property
    def components_(self):
        return self.dbscan.components_

    @property
    def core_sample_indices_(self):
        return self.dbscan.core_sample_indices_


    def neighborhood_dissimilarity_matrix(self, X) -> csr_matrix:
        """Neighborhood similarity matrix

        Computes the sparse neighborhood similarity matrix

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``metric='precomputed'``. If a sparse matrix is provided, it will
            be converted into a sparse ``csr_matrix``.

        Returns
        -------
        csr_matrix
            Sparse matrix of shape (n_samples, n_samples)
        """

        self.neigh.fit(X)
        graph = self.neigh.kneighbors_graph(X, mode="distance")
        dissimilarity_matrix = self.dissimilarity_func(graph, self.n_neighbors)
        dissimilarity_matrix[dissimilarity_matrix<0] = 0 #add 
        return dissimilarity_matrix