import numpy as np
from sklearn.neighbors import KDTree
from numba import njit, prange
import time
import warnings
from ...tools._neighbors import Neighbors

def calc_normals( pc, viewpoint=None, inds=None, knn = None, radius=None, kd_method='sknn' ):
    """
    Calculate normals for a point cloud
    
    Args:
        pc: (N, D) point cloud coordinates
        viewpoint: (D,) viewpoint for orientation
        inds: Precomputed neighbor indices
        knn: Number of neighbors for normal estimation
        radius: Radius for neighbor search
        
    Returns:
        normals: (N, D) normal vectors
    """

    N, dim = pc.shape
    normals = np.zeros((N, dim))
    if viewpoint is None:
        viewpoint = np.zeros(dim)
    
    if inds is None:
        dists, inds = get_neighbors(pc, knn=knn or 11, radius=radius, kd_method=kd_method )
    for i in range(N):
        indN = inds[i, 1:]
        if len(indN) == 0:
            normals[i] = np.array([0.0, 0.0, 1.0]) if dim == 3 else np.array([0.0, 1.0])
            continue

        X = pc[indN]
        Y = X - np.mean(X, axis=0)
        cov = np.matmul(Y.T, Y)/(len(indN))
        S, Q = np.linalg.eigh(cov)
        normal = Q[:, np.argmin(S)]

        vv = viewpoint - pc[i]
        # Re-orient normal vectors
        if np.dot(normal, vv) < 0:
            normal = -normal
        normals[i] = normal
    return normals

def get_neighbors(coordx, coordy=None, knn = 11, radius = None, max_neighbor = int(1e4), kd_method='sknn', n_jobs = -1):
    """
    Find neighbors for each point using KDTree
    
    Args:
        points: (N, D) point cloud
        knn: Number of neighbors to find
        radius: Search radius
        
    Returns:
        dists: (N, knn) distances to neighbors
        indices: (N, knn) indices of neighbors
    """
    if coordy is None:
        coordy = coordx
    
    cknn = Neighbors( method=kd_method ,metric='euclidean', n_jobs=n_jobs)
    cknn.fit(coordx, radius_max= None,max_neighbor=max_neighbor)
    distances, indices = cknn.transform(coordy, knn=knn , radius = radius)
    return [distances, indices]

def FPFH(points, normals=None, knn=31, radius=None, n_bins=11, viewpoint=None, kd_method='sknn', eps=1e-10):
    """
    Compute Fast Point Feature Histogram (FPFH) features
    
    Args:
        points: (N, D) point cloud coordinates (2D or 3D)
        normals: (N, D) normal vectors (optional)
        knn: Number of neighbors for feature calculation
        radius: Search radius for neighbors
        n_bins: Number of bins per feature
        viewpoint: Viewpoint for normal orientation
        
    Returns:
        fpfh_features: (N, 3*n_bins) FPFH feature vectors
    """
    N, D = points.shape
    total_bins = 3 * n_bins
    
    # Get neighbors
    dists, indices = get_neighbors(points, knn=knn, radius = radius, kd_method=kd_method)

    # Compute normals if not provided
    if normals is None:
        normals = calc_normals(points, viewpoint=viewpoint, inds=indices)
    
    # Normalize normals safely
    norms = np.linalg.norm(normals, axis=1)
    valid_mask = norms > 1e-10
    if not np.all(valid_mask):
        warnings.warn(f"Warning: Found {np.sum(~valid_mask)} invalid normals")
        normals = normals.copy()
        normals[valid_mask] /= norms[valid_mask, None]
        normals[~valid_mask] = [0, 0, 1] if D == 3 else [0, 1]
    else:
        normals = normals / norms[:, None]

    # Precompute SPFH features
    spfh = np.zeros((N, total_bins), dtype=np.float32)

    # Compute SPFH features
    compute_spfh_numba(points, normals, indices[:,1:], dists[:,1:], spfh, n_bins, eps=eps)
    # Compute FPFH features
    fpfh_features = np.zeros((N, total_bins), dtype=np.float32)
    compute_fpfh_numba(spfh, indices[:,1:], dists[:,1:], fpfh_features, eps=eps)

    return fpfh_features

@njit(parallel=True, fastmath=True)
def compute_spfh_numba(points, normals, indices, dists, spfh, n_bins, eps = 1e-10):
    """
    Compute Simplified Point Feature Histogram (SPFH) using Numba
    
    Args:
        points: (N, D) point coordinates
        normals: (N, D) normal vectors
        indices: (N, K) neighbor indices
        dists: (N, K) distances to neighbors
        spfh: (N, 3*n_bins) output array for SPFH features
        n_bins: Number of bins per feature
        dim: Dimensionality (2 or 3)
    """
    N, D = points.shape
    bin_size = 1.0 / n_bins
    
    for i in prange(points.shape[0]):
        center = points[i]
        normal_center = normals[i]
        neighbors = indices[i]
        distances = dists[i]

       # Skip if no neighbors
        if len(neighbors) == 0:
            continue
            
        hist_alpha = np.zeros(n_bins)
        hist_phi = np.zeros(n_bins)
        hist_theta = np.zeros(n_bins)
        valid_count = 0
        
        for j in range(len(neighbors)):
            neighbor_idx = neighbors[j]
            # Skip invalid indices
            if neighbor_idx < 0:
                continue
            dist = distances[j]
            # Skip invalid distances
            if dist < eps:
                continue
                
            p = points[neighbor_idx]
            n = normals[neighbor_idx]
            
            # Compute vector between points
            diff = p - center
            diff_normalized = diff / dist
            
            # Compute phi and alpha features
            phi = np.dot(diff_normalized, normal_center)
            alpha = np.dot(diff_normalized, n)
            
            # Compute theta feature (only for 3D)
            if D == 3:
                cross = np.cross(normal_center, diff_normalized)
                cross_norm = np.linalg.norm(cross)
                
                if cross_norm > eps:
                    cross_normalized = cross / cross_norm
                    w = np.cross(cross, cross_normalized) 
                    dot_u_nj = np.dot(n, normal_center)
                    dot_w_nj = np.dot(n, cross_normalized) #w
                    theta = np.arctan2(dot_w_nj, dot_u_nj)
                else:
                    theta = 0.0
            else:  # 2D case
                # For 2D, use angle between normals
                dot_val = np.dot(normal_center, n)
                dot_val = max(-1.0, min(1.0, dot_val))
                theta = np.arccos(dot_val)
            
            # Normalize features to [0, 1]
            alpha_norm = max(0.0, min(1.0, (alpha + 1) * 0.5))
            phi_norm = max(0.0, min(1.0, (phi + 1) * 0.5))
            theta_norm = max(0.0, min(1.0, (theta + np.pi) / (2 * np.pi))) # 2d check
            
            # Update histograms
            bin_idx_alpha = min(int(alpha_norm / bin_size), n_bins - 1)
            bin_idx_phi = min(int(phi_norm / bin_size), n_bins - 1)
            bin_idx_theta = min(int(theta_norm / bin_size), n_bins - 1)
            
            hist_alpha[bin_idx_alpha] += 1
            hist_phi[bin_idx_phi] += 1
            hist_theta[bin_idx_theta] += 1
            valid_count += 1
        
        # Normalize histograms if valid points found
        if valid_count > 0:
            hist_alpha /= valid_count
            hist_phi /= valid_count
            hist_theta /= valid_count
            
            spfh[i, :n_bins] = hist_alpha
            spfh[i, n_bins:2*n_bins] = hist_phi
            spfh[i, 2*n_bins:] = hist_theta

    
@njit(parallel=True, fastmath=True)
def compute_fpfh_numba(spfh, indices, dists, fpfh, eps = 1e-10):
    """
    Compute FPFH features from SPFH using Numba
    
    Args:
        spfh: (N, B) SPFH features
        indices: (N, K) neighbor indices
        dists: (N, K) distances to neighbors
        fpfh_features: (N, B) output array for FPFH features
        knn: Number of neighbors used
    """

    for i in prange(spfh.shape[0]):
        neighbors = indices[i]
        distances = dists[i]
        
        if len(neighbors) == 0:
            fpfh[i] = spfh[i]
            continue
            
        weight_sum = 0.0
        weighted_spfh = np.zeros(spfh.shape[1])
        valid_count = 0
        
        for j in range(len(neighbors)):
            neighbor_idx = neighbors[j]
            # Skip invalid indices
            if neighbor_idx < 0:
                continue
            dist = distances[j]
            # Skip invalid distances
            if dist < eps:
                continue
                
            weight = 1.0 / dist
            weighted_spfh += spfh[neighbor_idx] * weight
            weight_sum += weight
            valid_count += 1

        if valid_count > 0 and weight_sum > eps:
            weighted_spfh /= weight_sum
            # fpfh[i] = spfh[i] + weighted_spfh * (1.0 / valid_count) # check
            fpfh[i] = spfh[i] + weighted_spfh
        else:
            fpfh[i] = spfh[i]
        
        # L1 normalization
        feature_sum = np.sum(fpfh[i])
        if feature_sum > eps:
            fpfh[i] /= feature_sum