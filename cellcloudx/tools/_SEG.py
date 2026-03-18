import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as ssp
from tqdm import tqdm

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

def _compute_rbf_basis(X, n_centers=50, length_scale=None, seed=200504):
    """
    Construct RBF basis matrix Φ(X) for a single dataset.

    Parameters:
    -----------
    X : ndarray (N, d)
        Input coordinates
    n_centers : int
        Number of RBF centers
    length_scale : float, optional
        Length scale parameter. If None, estimated automatically
    seed : int
        Random seed for selecting centers

    Returns:
    --------
    Phi : ndarray (N, K)
        RBF basis matrix
    """
    X = np.asarray(X, dtype=float)
    N, d = X.shape
    n_centers = min(n_centers, N)

    # Randomly select centers
    rng = np.random.RandomState(seed)
    centers = X[rng.choice(N, n_centers, replace=False)]

    # Estimate length scale if not provided
    if length_scale is None:
        nbrs = NearestNeighbors(n_neighbors=1).fit(centers)
        dist, _ = nbrs.kneighbors(X)
        length_scale = np.median(dist)
        if not np.isfinite(length_scale) or length_scale <= 0:
            length_scale = 1.0

    # Compute RBF kernel matrix efficiently
    # Using broadcasting: (N, K, d) - (K, d) -> (N, K, d)
    Phi = X[:, None, :] - centers[None, :, :]
    Phi = np.sum(Phi ** 2, axis=2)
    Phi = np.exp(-Phi / (2.0 * length_scale ** 2 + 1e-12))

    # Column-wise standardization for numerical stability
    Phi_centered = Phi - np.mean(Phi, axis=0, keepdims=True)
    Phi_std = np.std(Phi_centered, axis=0, keepdims=True, ddof=1)
    Phi_std[Phi_std == 0] = 1.0
    Phi = Phi_centered / Phi_std

    return Phi


def _solve_ridge_regression(H, Y, penalty_vector, eps=1e-8):
    """
    Solve multi-output Ridge regression:
      (H^T H + diag(penalty) + eps*I) Beta = H^T Y

    Parameters:
    -----------
    H : ndarray (N, p)
        Design matrix
    Y : ndarray (N, G)
        Response matrix
    penalty_vector : ndarray (p,)
        L2 penalty coefficients for each parameter
    eps : float
        Numerical stability term

    Returns:
    --------
    Beta : ndarray (p, G)
        Regression coefficients
    """
    H = np.asarray(H, dtype=float)
    Y = np.asarray(Y, dtype=float)

    N, p = H.shape
    _, G = Y.shape

    penalty_vector = np.asarray(penalty_vector, dtype=float)
    assert penalty_vector.shape[0] == p, "Penalty vector dimension mismatch"

    # Compute normal matrix H^T H
    HtH = H.T @ H
    HtY = H.T @ Y

    # Add regularization
    A = HtH + np.diag(penalty_vector) + eps * np.eye(p)

    # Solve linear system for all genes simultaneously
    try:
        Beta = np.linalg.solve(A, HtY)
    except np.linalg.LinAlgError:
        # Fallback to least squares if matrix is ill-conditioned
        Beta = np.linalg.lstsq(A, HtY, rcond=None)[0]

    return Beta


def _compute_mahalanobis_moment(X, W, inv_covariance, eps=1e-12):
    """
    Compute weighted Mahalanobis second moment for a batch.

    Parameters:
    -----------
    X : ndarray (N, d)
        Spatial coordinates
    W : ndarray (N, B)
        Non-negative weights for B genes
    inv_covariance : ndarray (d, d)
        Inverse covariance matrix
    eps : float
        Numerical stability term

    Returns:
    --------
    sigma2 : ndarray (B,)
        Weighted Mahalanobis second moments for each gene
    """
    X = np.asarray(X, dtype=float)
    W = np.asarray(W, dtype=float)

    N, d = X.shape
    N_check, B = W.shape
    assert N == N_check, "Dimension mismatch between X and W"

    # Compute weighted sum and centroids
    weight_sum = np.sum(W, axis=0) + eps  # (B,)
    centroids = (X.T @ W) / weight_sum[None, :]  # (d, B)

    # Compute Mahalanobis distances efficiently
    # delta: (N, d, B) = X - centroids
    delta = X[:, :, None] - centroids[None, :, :]

    # Compute Mahalanobis distance: (X - μ)^T Σ^{-1} (X - μ)
    # Using einsum for efficiency
    tmp = np.einsum('ij,njb->nib', inv_covariance, delta)  # (N, d, B)
    dist_sq = np.einsum('njb,njb->nb', delta, tmp)         # (N, B)

    # Weighted average of squared distances
    sigma2 = np.sum(W * dist_sq, axis=0) / weight_sum

    return sigma2


class SpatialExpansionGene:
    """Gene Spatial Expansion Model"""

    def __init__(self, n_centers=1000, lambda_spatial=1e-2, lambda_slice=5e-4,
                 tau_resid=1.0, use_softplus=True, mahalanobis=True,
                 global_basis=False, rbf_beta=None,
                 use_ortho=False, batch_size=256,
                 baseline_mode='dim', w_intensity=1.0, w_spatial=1.0,
                 eps=1e-8, random_seed=0):
        """
        Initialize GSS++ model.

        Parameters:
        -----------
        n_centers : int
            Number of RBF centers per slice
        lambda_spatial : float
            L2 regularization strength for spatial basis coefficients
        lambda_slice : float
            L2 regularization strength for slice effects
        tau_resid : float
            Temperature parameter for softplus transformation
        use_softplus : bool
            If True, use softplus transformation; otherwise use ReLU
        mahalanobis : bool
            If True, use Mahalanobis distance; otherwise use Euclidean
        baseline_mode : str
            Baseline calculation mode: 'dim' or 'median'
        w_intensity : float
            Weight for intensity scaling component
        w_spatial : float
            Weight for spatial scaling component
        global_basis : bool
            If True, use global RBF basis; otherwise per-slice basis
        use_ortho : bool
            If True, orthogonalize basis w.r.t. slice effects
        batch_size : int
            Batch size for gene-wise computations
        eps : float
            Numerical stability term
        random_seed : int
            Random seed for reproducibility
        """
        self.n_centers = n_centers
        self.lambda_spatial = lambda_spatial
        self.lambda_slice = lambda_slice
        self.tau_resid = tau_resid
        self.use_softplus = use_softplus
        self.global_basis = global_basis
        self.rbf_beta = rbf_beta
        self.use_ortho = use_ortho
        self.batch_size = batch_size

        self.mahalanobis = mahalanobis
        self.baseline_mode = baseline_mode
        self.w_intensity = w_intensity
        self.w_spatial = w_spatial
        self.eps = eps
        self.random_seed = random_seed

        # Initialize attributes
        self.results_ = None

    def _construct_block_diagonal_basis(self, Xs, ):
        """
        Construct block-diagonal RBF basis matrix.

        Parameters:
        -----------
        Xs : list of ndarray
            List of coordinate arrays for each slice
        Returns:
        --------
        Phi : ndarray
            Block-diagonal basis matrix
        slice_sizes : list
            Number of basis functions per slice
        """
        S = len(Xs)

        if self.global_basis:
            # Construct global basis matrix
            X_all = np.concatenate(Xs, axis=0)
            Phi = _compute_rbf_basis(
                X_all,
                n_centers=self.n_centers * S,
                length_scale=self.rbf_beta,
                seed=self.random_seed
            )
            slice_sizes = [Phi.shape[1]] * S
            return Phi, slice_sizes

        # Construct per-slice basis matrices
        slice_matrices = []
        slice_sizes = []

        for s in range(S):
            X_s = np.asarray(Xs[s], dtype=float)
            Phi_s = _compute_rbf_basis(
                X_s,
                n_centers=self.n_centers,
                length_scale=self.rbf_beta,
                seed=self.random_seed + s
            )
            slice_matrices.append(Phi_s)
            slice_sizes.append(Phi_s.shape[1])

        # Construct block-diagonal matrix
        total_basis = sum(slice_sizes)
        total_points = sum(X.shape[0] for X in Xs)
        Phi = np.zeros((total_points, total_basis), dtype=float)

        row_offset = 0
        col_offset = 0

        for s in range(S):
            rows = Xs[s].shape[0]
            cols = slice_sizes[s]
            Phi[row_offset:row_offset + rows, col_offset:col_offset + cols] = slice_matrices[s]
            row_offset += rows
            col_offset += cols

        return Phi, slice_sizes

    def _orthogonalize_basis(self, Phi, slice_indices, S):
        """
        Orthogonalize basis matrix w.r.t. slice dummy variables.

        Parameters:
        -----------
        Phi : ndarray
            Basis matrix to orthogonalize
        slice_indices : ndarray
            Array indicating slice membership for each point
        S : int
            Number of slices

        Returns:
        --------
        Phi_ortho : ndarray
            Orthogonalized basis matrix
        """
        N = len(slice_indices)

        # Create slice dummy matrix (drop first slice as reference)
        Z_full = np.zeros((N, S), dtype=float)
        Z_full[np.arange(N), slice_indices] = 1.0
        Z = Z_full[:, 1:]  # (N, S-1)

        # Create design matrix J = [1 | Z]
        ones = np.ones((N, 1), dtype=float)
        J = np.concatenate([ones, Z], axis=1)

        # Compute projection: P = J (J^T J)^(-1) J^T
        JtJ = J.T @ J + self.eps * np.eye(J.shape[1])
        JtJ_inv = np.linalg.inv(JtJ)
        projection_coeff = JtJ_inv @ (J.T @ Phi)
        Phi_ortho = Phi - J @ projection_coeff

        return Phi_ortho

    def _compute_inverse_covariances(self, Xs):
        """
        Compute inverse covariance matrices for each slice.

        Parameters:
        -----------
        Xs : list of ndarray
            List of coordinate arrays

        Returns:
        --------
        inv_covariances : list of ndarray
            List of inverse covariance matrices
        """
        inv_covariances = []

        for X_s in Xs:
            if self.mahalanobis:
                # Compute covariance matrix
                X_centered = X_s - np.mean(X_s, axis=0)
                cov = np.cov(X_centered.T, ddof=1)
                # Regularize and invert
                cov += self.eps * np.eye(cov.shape[0])
                inv_cov = np.linalg.inv(cov)
            else:
                # Use identity matrix for Euclidean distance
                inv_cov = np.eye(X_s.shape[1])

            inv_covariances.append(inv_cov)

        return inv_covariances

    def _compute_spatial_moments(self, Xs, Phi_ortho, Beta,
                                inv_covariances,
                                batch_size=500):
        """
        计算 sigma2[s,g]：每片每基因的加权空间二阶矩

        Xs         : list of (N_s, d)
        Phi_ortho  : (N_total, Ktot) 正交化后的空间基
        Beta       : (p, G) 回归系数
        inv_covariances : list of (d,d) 协方差逆
        """
        S = len(Xs)

        # 取出空间系数块
        spatial_start_idx = 1 + (S - 1)
        spatial_coeff = Beta[spatial_start_idx:, :]   # (Ktot, G)
        Ktot, G = spatial_coeff.shape

        sigma2 = np.zeros((S, G), dtype=float)

        row_offset = 0
        for s in range(S):
            X_s = np.asarray(Xs[s], dtype=float)
            N_s = X_s.shape[0]
            inv_cov = inv_covariances[s]

            # ★ 对每个 slice，只按行截取 Phi_ortho；列保持完整
            Phi_s = Phi_ortho[row_offset:row_offset + N_s, :]   # (N_s, Ktot)

            for g_start in range(0, G, batch_size):
                g_end = min(G, g_start + batch_size)

                # 空间场预测
                S_pred = Phi_s @ spatial_coeff[:, g_start:g_end]   # (N_s, B)

                # 转为非负权重
                if self.use_softplus:
                    if self.tau_resid is None or self.tau_resid <= 0:
                        W = np.log1p(np.exp(S_pred))
                    else:
                        W = np.log1p(np.exp(S_pred / self.tau_resid))
                else:
                    W = np.maximum(S_pred, 0.0)

                W += 1e-12

                sigma2_batch = _compute_mahalanobis_moment(X_s, W, inv_cov, self.eps)
                sigma2[s, g_start:g_end] = sigma2_batch

            row_offset += N_s

        return sigma2

    def _compute_spatial_moments0(self, Xs, G,
                                 Phi_ortho, Beta,
                                 slice_sizes, inv_covariances,
                                 batch_size=500):
        """
        Compute spatial second moments (sigma2) for each slice and gene.

        Parameters:
        -----------
        Xs : list of ndarray
            List of coordinate arrays
        Fs : list of ndarray
            List of expression matrices
        Phi_ortho : ndarray
            Orthogonalized design matrix
        Beta : ndarray
            Regression coefficients
        slice_sizes : list
            Number of basis functions per slice
        inv_covariances : list
            Inverse covariance matrices
        batch_size : int
            Batch size for gene-wise computation

        Returns:
        --------
        sigma2 : ndarray (S, G)
            Spatial second moments
        """
        S = len(Xs)

        # Extract spatial coefficients (skip intercept and slice effects)
        spatial_start_idx = 1 + (S - 1)
        spatial_coeff = Beta[spatial_start_idx:, :]  # (K_total, G)

        sigma2 = np.zeros((S, G), dtype=float)
        row_offset = 0
        col_offset = 0

        for s in range(S):
            X_s = Xs[s]
            N_s = X_s.shape[0]
            inv_cov = inv_covariances[s]

            # Extract relevant columns from design matrix
            if self.global_basis:
                Phi_s = Phi_ortho[row_offset:row_offset + N_s, :]
                coeff_s = spatial_coeff
            else:
                K_s = slice_sizes[s]
                Phi_s = Phi_ortho[row_offset:row_offset + N_s, col_offset:col_offset + K_s]
                coeff_s = spatial_coeff[col_offset:col_offset + K_s, :]

            # Process genes in batches
            for g_start in range(0, G, batch_size):
                g_end = min(G, g_start + batch_size)

                # Predict spatial field
                S_pred = Phi_s @ coeff_s[:, g_start:g_end]  # (N_s, batch_size)

                # Transform to non-negative weights
                if self.use_softplus:
                    if self.tau_resid is None or self.tau_resid <= 0:
                        W = np.log1p(np.exp(S_pred))
                    else:
                        W = np.log1p(np.exp(S_pred / self.tau_resid))
                else:
                    W = np.maximum(S_pred, 0.0)

                W += 1e-12  # Prevent all zeros

                # Compute Mahalanobis moments
                sigma2_batch = _compute_mahalanobis_moment(X_s, W, inv_cov, self.eps)
                sigma2[s, g_start:g_end] = sigma2_batch

            row_offset += N_s
            if not self.global_basis:
                col_offset += slice_sizes[s]

        return sigma2

    def fit(self, Xs, Fs, normal_xs=True):
        """
        Fit GSS++ model to data.

        Parameters:
        -----------
        Xs : list of ndarray
            List of coordinate arrays [(N_s, d)]
        Fs : list of ndarray
            List of expression matrices [(N_s, G)]

        Returns:
        --------
        self : object
            Fitted model
        """
        
        # Validate inputs
        S = len(Xs)
        Ns = [ix.shape[0] for ix in Xs]
        Nt = np.cumsum([0] + Ns)
        assert S == len(Fs), "Mismatch in number of slices"

        X_all = np.concatenate(Xs, axis=0)
        N_total, d = X_all.shape
        if normal_xs:
            X_all =centerlize(X_all)
            Xs =  [X_all[ Nt[s]:Nt[s + 1]] for s in range(S)]

        # Create slice indices
        slice_indices = np.concatenate([
            np.full(coords.shape[0], s, dtype=int)
            for s, coords in enumerate(Xs)
        ])

        # Concatenate all data
        F_all = np.concatenate(Fs, axis=0)
        _, G = F_all.shape

        # 1. Construct basis matrix
        Phi, slice_sizes = self._construct_block_diagonal_basis(Xs)
        # 2. Orthogonalize basis if requested
        if self.use_ortho:
            Phi_ortho = self._orthogonalize_basis(Phi, slice_indices, S)
        else:
            Phi_ortho = Phi

        # 3. Construct final design matrix H = [1 | Z | Φ_ortho]
        # Create slice dummy matrix
        Z_full = np.zeros((N_total, S), dtype=float)
        Z_full[np.arange(N_total), slice_indices] = 1.0
        Z = Z_full[:, 1:]  # Drop first slice as reference

        ones = np.ones((N_total, 1), dtype=float)
        H = np.concatenate([ones, Z, Phi_ortho], axis=1)

        # 4. Construct regularization vector
        p = H.shape[1]
        penalty = np.zeros(p, dtype=float)
        penalty[1:1 + (S - 1)] = self.lambda_slice  # Slice effects
        penalty[1 + (S - 1):] = self.lambda_spatial  # Spatial coefficients

        # 5. Solve Ridge regression
        self.Beta = _solve_ridge_regression(H, F_all, penalty, self.eps)

        # 6. Extract slice effects and compute intensity scaling
        alpha_tilde = self.Beta[1:1 + (S - 1), :]  # (S-1, G)
        alpha_full = np.zeros((S, G), dtype=float)
        alpha_full[1:, :] = alpha_tilde

        # Apply sum-to-zero constraint
        alpha_full -= np.mean(alpha_full, axis=0, keepdims=True)

        # Compute z-scores for intensity scaling
        A_std = np.std(alpha_full, axis=0, keepdims=True, ddof=1)
        A_std[A_std == 0] = 1.0
        A_z = alpha_full / A_std

        # 7. Compute spatial moments
        inv_covariances = self._compute_inverse_covariances(Xs)
        sigma2 = self._compute_spatial_moments(
            Xs,
            Phi_ortho, self.Beta,
            #slice_sizes, 
            inv_covariances,
            self.batch_size
        )

        # 8. Compute spatial scaling
        if self.baseline_mode == 'dim':
            baseline = float(d)
            E_raw = np.log((sigma2 + self.eps) / (baseline + self.eps))
        elif self.baseline_mode == 'median':
            baseline = np.median(sigma2, axis=0, keepdims=True)
            E_raw = np.log((sigma2 + self.eps) / (baseline + self.eps))
        else:
            raise ValueError(f"Unknown baseline mode: {self.baseline_mode}")

        # Compute z-scores for spatial scaling
        E_std = np.std(E_raw, axis=0, keepdims=True, ddof=1)
        E_std[E_std == 0] = 1.0
        E_z = E_raw / E_std

        # 9. Compute total scaling score
        total_score = self.w_intensity * A_z + self.w_spatial * E_z

        # Store results
        self.results_ = {
            'A': alpha_full,
            'Az': A_z,
            'E': E_raw,
            'Ez': E_z,
            'S': sigma2,
            'AE': total_score,
            'H': H,
            'Beta': self.Beta,
            'slice_sizes': slice_sizes
        }

        return self

    def get_results(self):
        """Return computed scaling scores and related metrics."""
        if self.results_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.results_

# Example usage and demonstration
def SEG(Xs, Fs, f_thred = 0, n_centers=1000, lambda_spatial=1e-2, lambda_slice=5e-4,
        baseline_mode='dim', normal_xs = True, rbf_beta=None, batch_size=500,
        seed=491001, global_basis=False, use_ortho=False, **kargs):
    """
    Demonstrate SEG model usage.
    Parameters
    ----------
    # Generate synthetic data
    # np.random.seed(42)
    # Ns = [1000, 1500, 2000, 3000]
    # Xs = [np.random.random((N, 3)) for N in Ns]
    # Fs = [np.random.random((N, 50)) for N in Ns]

    """
    if f_thred >0:
        Nl = len(Xs)
        Ng = Fs[0].shape[1]
        results = {
            'A': [],
            'Az': [],
            'E': [],
            'Ez': [],
            'S': [],
            'AE': [],
        }

        for ig in tqdm(range(Ng)):
            ixs, ifs = [], []
            for il in range(Nl):
                ik = (Fs[il][:, ig] > f_thred)
                ixs.append(Xs[il][ik])
                ifs.append(Fs[il][ik][:,[ig]])
            model = SpatialExpansionGene(
                n_centers=n_centers,
                lambda_spatial=lambda_spatial,
                lambda_slice=lambda_slice,
                global_basis=global_basis,
                baseline_mode=baseline_mode,
                use_ortho=use_ortho,
                rbf_beta = rbf_beta,
                batch_size = batch_size,
                random_seed=seed,
                **kargs
            )
            model.fit(ixs, ifs, normal_xs = normal_xs)
            iresult = model.get_results()
            for k in results.keys():
                results[k].append(iresult[k])
        for k in results.keys():
            results[k] = np.concatenate(results[k], axis=1)
        return results
    else:
        model = SpatialExpansionGene(
            n_centers=n_centers,
            lambda_spatial=lambda_spatial,
            lambda_slice=lambda_slice,
            global_basis=global_basis,
            baseline_mode=baseline_mode,
            use_ortho=use_ortho,
            rbf_beta = rbf_beta,
            batch_size = batch_size,
            random_seed=seed,
            **kargs
        )

        model.fit(Xs, Fs, normal_xs = normal_xs)
        results = model.get_results()
        return results
