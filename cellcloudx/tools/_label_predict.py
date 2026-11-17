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



"""Spatial label refinement on an arbitrary 2D/3D point cloud.

Dependencies: numpy, scipy. CPU implementation; no mesh or training required.

    from label_refine import refine_labels
    new_label, score = refine_labels(points, label)
    new_label, score, info = refine_labels(points, label, return_info=True)

The original labels define a fixed unary term. A distance-limited symmetric
kNN graph defines a joint multiclass graph-TV relaxation of the Potts model.
No points or small components are deleted. Scores are local conditional model
probabilities, NOT calibrated probabilities of biological correctness.
"""

import warnings

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.special import softmax



def _spatial_graph(points, k, max_distance, axis_scale, feature,
                   feature_scale, workers):
    """Return symmetric CSR weights, normalized to median positive degree 1."""
    n = len(points)
    if n < 2:
        return sparse.csr_matrix((n, n)), max_distance
    x = (points - points.mean(axis=0)) / axis_scale
    k = min(k, n - 1)
    distance, index = cKDTree(x).query(x, k=k + 1, workers=workers)
    # Self is not necessarily the first result when coordinates are duplicated.
    distance[index == np.arange(n)[:, None]] = np.inf
    order = np.argsort(distance, axis=1, kind="stable")[:, :k]
    distance = np.take_along_axis(distance, order, axis=1)
    index = np.take_along_axis(index, order, axis=1)
    h = distance[:, -1].copy()
    positive = h[h > 0]
    reference = float(np.median(positive)) if positive.size else 1.0
    h = np.clip(h, 0.25 * reference, 1.5 * reference)
    radius = 1.5 * reference if max_distance is None else max_distance
    row = np.repeat(np.arange(n), k)
    col = index.ravel()
    d = distance.ravel()
    keep = d <= radius
    row, col, d = row[keep], col[keep], d[keep]
    weight = np.exp(-0.5 * (d / h[row]) * (d / h[col]))
    if feature is not None and row.size:
        # Accumulate by column to avoid allocating an (N*k, n_features) array.
        fd2 = np.zeros(row.size)
        for f in feature.T:
            fd2 += (f[row] - f[col]) ** 2
        if feature_scale is None:
            positive = fd2[fd2 > 0]
            scale2 = float(np.median(positive)) if positive.size else 1.0
        else:
            scale2 = feature_scale ** 2
        weight *= np.exp(-0.5 * fd2 / scale2)
    graph = sparse.csr_matrix((weight, (row, col)), shape=(n, n))
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    degree = np.asarray(graph.sum(axis=1)).ravel()
    if np.any(degree > 0):
        graph.data /= np.median(degree[degree > 0])
    return graph, radius


def _simplex(v):
    """Euclidean projection of each row onto {u >= 0, sum(u) = 1}."""
    ordered = np.sort(v, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1) - 1.0
    count = np.sum(ordered > cumulative / np.arange(1, v.shape[1] + 1), axis=1)
    threshold = cumulative[np.arange(len(v)), count - 1] / count
    return np.maximum(v - threshold[:, None], 0.0)


def _graph_tv(graph, cost, original, fixed, smooth, max_iter, tol):
    """Chambolle-Pock solver; stop on the primal-dual gap per point."""
    n, classes = cost.shape
    u = np.zeros_like(cost)
    u[np.arange(n), original] = 1.0
    if not graph.nnz or smooth == 0 or classes == 1:
        return u, 0, True, 0.0
    edge = sparse.triu(graph, k=1, format="coo")
    m = len(edge.data)
    incidence = sparse.csr_matrix(
        (np.tile([-1.0, 1.0], m),
         (np.repeat(np.arange(m), 2),
          np.column_stack((edge.row, edge.col)).ravel())),
        shape=(m, n),
    )
    transpose = incidence.T.tocsr()
    limit = (0.5 * smooth * edge.data)[:, None]
    step = 0.99 / np.sqrt(2.0 * np.max(np.diff(graph.indptr)))
    dual = np.zeros((m, classes))
    extrapolated = u.copy()
    fixed_index = np.flatnonzero(fixed)
    gap = np.inf
    for iteration in range(1, max_iter + 1):
        dual += step * (incidence @ extrapolated)
        np.clip(dual, -limit, limit, out=dual)
        gradient = cost + transpose @ dual
        updated = _simplex(u - step * gradient)
        updated[fixed_index] = 0.0
        updated[fixed_index, original[fixed_index]] = 1.0
        extrapolated = 2.0 * updated - u
        u = updated
        if iteration % 20 == 0 or iteration == max_iter:
            primal = np.sum(cost * u) + np.sum(limit * np.abs(incidence @ u))
            lower = np.min(gradient, axis=1)
            lower[fixed_index] = gradient[fixed_index, original[fixed_index]]
            gap = max(0.0, float(primal - lower.sum()) / n)
            if gap <= tol:
                break
    return u, iteration, gap <= tol, gap


def refine_labels(points, label, *, k=20, smooth=3.0, confidence=0.8,
                  max_distance=None, axis_scale=1.0, feature=None,
                  feature_scale=None, fixed=None, max_iter=500, tol=1e-4,
                  workers=-1, return_info=False):
    """Correct scattered labels and regularize boundaries jointly.

    Parameters
    ----------
    points : array-like, (N, 2) or (N, 3)
        Finite coordinates; their order and positions are unchanged.
    label : array-like, (N,)
        String or numeric labels, including pandas Series/Categorical inputs.
        Missing labels are not supported. Output is a NumPy array retaining
        the input label values and dtype (not the pandas categorical metadata).
    k : int, default=20
        Number of candidate neighbors per point. Symmetrization can increase
        a node's final degree. Larger values may join across thin structures.
    smooth : float >= 0, default=3.0
        Graph-TV/Potts strength. Larger values remove more fragments but may
        erase thin/rare regions. Zero returns the original labels exactly.
        Edge weights have median positive weighted degree 1; this is not
        physical volume/area normalization.
    confidence : float or (N,) array, default=0.8
        Assumed reliability of each ORIGINAL label. For C > 1, values must
        satisfy 1/C <= confidence < 1. Remaining mass is shared equally among
        the other C-1 classes. It stays fixed throughout optimization. Use
        higher values, e.g. 0.95-0.99, to protect trusted labels, or fixed for
        exact anchors. This is an assumed prior, not estimated accuracy.
    max_distance : positive float or None, default=None
        Maximum edge length AFTER division by axis_scale. At axis_scale=1 it
        is in the original coordinate units. None uses 1.5 times the median
        positive kth-neighbor distance. No long edge is added to connect an
        isolated point. An explicit limit is preferable for known cavities.
    axis_scale : positive float or (D,) array, default=1.0
        Coordinates used for neighbor search are points/axis_scale. For 3D,
        (1, 1, 2) makes a z separation count half as much. This changes graph
        connectivity; it neither moves points nor reconstructs missing slices.
    feature : array-like, (N, F), optional
        Finite, already standardized low-dimensional expression/image features.
        Dissimilar features weaken an edge. No model is trained. Prefer a
        small embedding; sparse full gene matrices should be reduced first.
    feature_scale : positive float or None, default=None
        Gaussian feature distance scale. None uses the median positive feature
        distance of candidate edges. Smaller values protect feature boundaries
        more strongly; larger values make feature differences less influential.
    fixed : bool array, (N,), optional
        True fixes that point to its original label. Its returned score is 1
        by the imposed constraint, not by independently measured certainty.
    max_iter : positive int, default=500
        Maximum iterations of the convex relaxation. If the gap remains above
        tol, return the current approximation and issue a RuntimeWarning.
    tol : positive float, default=1e-4
        Primal-dual gap tolerance per point. Increasing max_iter improves solver
        convergence; it is not a substitute for increasing smooth.
    workers : int, default=-1
        cKDTree workers; -1 uses all available CPU threads.
    return_info : bool, default=False
        Also return class-wise conditional scores and solver diagnostics.

    Returns
    -------
    new_label : ndarray, (N,)
        Labels from the graph-TV relaxation, rounded jointly with argmax.
        Exact ties prefer the original label. No points are deleted.
    score : ndarray, (N,)
        In [0, 1], the conditional score of the RETURNED label:
        softmax_c(log(q_i(c)) + smooth * sum_j w_ij [new_label_j == c]).
        It conditions on final neighbor labels; it is not a calibrated
        correctness probability. An isolated point retains its original label
        and prior score. With only one observed class the score is 1.
    info : dict, only if return_info=True
        classes: class values for columns of probabilities;
        probabilities: (N,C) conditional model scores, not TV variables;
        margin: selected score minus the largest alternative score (may be
        negative: TV rounding need not maximize the local conditional score);
        changed: (N,) bool; iterations, converged, duality_gap, max_distance.

    Notes
    -----
    The convex relaxation has objective
        sum_i,c D_i(c)*U_i(c) + smooth/2 * sum_{i<j} w_ij*|U_i-U_j|_1,
    with U_i in the probability simplex and D=-log(q). Its rounded multiclass
    solution is approximate. Convergence refers to the relaxation, not the
    discrete global optimum or biological correctness. Use reliable evidence
    to distinguish real scattered cells from erroneous labels.
    Memory is O(N*k*C); each solver iteration is O(N*k*C + N*C*log(C)).

    Examples
    --------
    >>> new_label, score = refine_labels(points, label)  # doctest: +SKIP
    >>> new_label, score, info = refine_labels(  # doctest: +SKIP
    ...     points, label, smooth=4.0, return_info=True)
    """
    x = np.asarray(points, dtype=float)
    labels = np.asarray(label)
    if x.ndim != 2 or x.shape[1] not in (2, 3) or not np.isfinite(x).all():
        raise ValueError("points must be a finite (N, 2) or (N, 3) array.")
    n = len(x)
    if labels.ndim != 1 or len(labels) != n:
        raise ValueError("label must be a one-dimensional array of length N.")
    if labels.dtype.kind in "fc" and not np.isfinite(labels).all():
        raise ValueError("label cannot contain missing or infinite values.")
    if labels.dtype.kind == "O" and any(
        v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v))
        for v in labels
    ):
        raise ValueError("label cannot contain missing values.")
    try:
        classes, original = np.unique(labels, return_inverse=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("Use nonmissing string labels or homogeneous numeric labels.") from exc
    c = len(classes)
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer.")
    if not np.isfinite(smooth) or smooth < 0:
        raise ValueError("smooth must be finite and >= 0.")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and > 0.")
    if max_distance is not None and (not np.isfinite(max_distance) or max_distance <= 0):
        raise ValueError("max_distance must be finite and > 0, or None.")
    if feature_scale is not None and (not np.isfinite(feature_scale) or feature_scale <= 0):
        raise ValueError("feature_scale must be finite and > 0, or None.")
    axes = np.broadcast_to(np.asarray(axis_scale, dtype=float), (x.shape[1],))
    if not np.isfinite(axes).all() or np.any(axes <= 0):
        raise ValueError("axis_scale must be positive and finite.")
    reliability = np.broadcast_to(np.asarray(confidence, dtype=float), (n,))
    if not np.isfinite(reliability).all() or np.any((reliability < 0) | (reliability > 1)):
        raise ValueError("confidence must be finite and in [0, 1].")
    if c > 1 and np.any((reliability < 1.0 / c) | (reliability >= 1)):
        raise ValueError("For C>1, confidence must be >=1/C and <1; use fixed for anchors.")
    anchors = np.zeros(n, dtype=bool) if fixed is None else np.asarray(fixed)
    if anchors.shape != (n,) or anchors.dtype.kind != "b":
        raise ValueError("fixed must be a boolean array of length N.")
    if feature is not None:
        feature = np.asarray(feature, dtype=float)
        if feature.ndim != 2 or feature.shape[0] != n or not np.isfinite(feature).all():
            raise ValueError("feature must be a finite (N, F) array.")
    rows = np.arange(n)
    if c <= 1:
        probabilities = np.ones((n, c))
        score = np.ones(n)
        updated = original.copy()
        iterations, converged, gap, radius = 0, True, 0.0, max_distance
    else:
        prior = np.broadcast_to(((1.0 - reliability) / (c - 1))[:, None], (n, c)).copy()
        prior[rows, original] = reliability
        log_prior = np.log(prior)
        cost = -log_prior
        cost -= cost.min(axis=1, keepdims=True)
        if smooth == 0:
            graph, radius = sparse.csr_matrix((n, n)), max_distance
        else:
            graph, radius = _spatial_graph(
                x, k, max_distance, axes, feature, feature_scale, workers)
        u, iterations, converged, gap = _graph_tv(
            graph, cost, original, anchors, smooth, max_iter, tol)
        updated = u.argmax(axis=1)
        ties = u[rows, original] >= u.max(axis=1) - 1e-10
        updated[ties] = original[ties]
        updated[anchors] = original[anchors]
        isolated = np.diff(graph.indptr) == 0
        updated[isolated] = original[isolated]
        membership = sparse.csr_matrix((np.ones(n), (rows, updated)), shape=(n, c))
        support = (graph @ membership).toarray()
        probabilities = softmax(log_prior + smooth * support, axis=1)
        probabilities[anchors] = 0.0
        probabilities[rows[anchors], original[anchors]] = 1.0
        score = probabilities[rows, updated]
        if not converged:
            warnings.warn(
                "graph-TV reached max_iter; duality gap per point is %.3g "
                "(tol=%.3g). Increase max_iter if needed." % (gap, tol),
                RuntimeWarning, stacklevel=2)
    result = labels.copy()
    if n:
        result[:] = classes[updated]
    if not return_info:
        return result, score
    if c > 1:
        second = np.partition(probabilities, -2, axis=1)[:, -2]
        best = probabilities.max(axis=1)
        alternative = np.where(updated == probabilities.argmax(axis=1), second, best)
        margin = score - alternative
    else:
        margin = score.copy()
    info = dict(classes=classes, probabilities=probabilities, margin=margin,
                changed=result != labels, iterations=iterations,
                converged=converged, duality_gap=gap, max_distance=radius)
    return result, score, info




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