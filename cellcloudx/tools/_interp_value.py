import scipy as sci
import numpy as np
import pandas as pd
import pickle

from ..utilis._arrays import list_iter, vartype
from ..alignment.m3lib.xmm import kernel_xmm_k
from ._interp_shape import interpolate_slices, apply_slice_z
from joblib import Parallel, delayed


class _LUFactor:
    """Keep native LU solves in memory and serialize reusable sparse factors."""

    def __init__(self, A):
        from scipy.sparse.linalg import splu

        self.factor = splu(A)

    def solve(self, rhs):
        from scipy.sparse.linalg import spsolve_triangular

        if self.factor is not None:
            result = self.factor.solve(rhs)
        else:
            L, U, scale, rows, columns = self.factors
            result = spsolve_triangular(L, rhs[rows], lower=True, unit_diagonal=True,
                                       overwrite_A=True, overwrite_b=True)
            result = spsolve_triangular(U, result, lower=False, unit_diagonal=True,
                                       overwrite_A=True, overwrite_b=True)
            result *= scale.reshape((-1,) + (1,) * (result.ndim - 1))
            result = result[columns]
        return result

    def __getstate__(self):
        from scipy.sparse import diags

        if self.factor is not None:
            f = self.factor
            U = f.U
            scale = 1.0 / U.diagonal()
            L, U = f.L, (U @ diags(scale)).tocsr()
            L.setdiag(1.0)
            U.setdiag(1.0)
            L.sort_indices()
            U.sort_indices()
            factors = (L, U, scale, np.argsort(f.perm_r), f.perm_c)
        else:
            factors = self.factors
        return factors

    def __setstate__(self, factors):
        self.factor = None
        self.factors = factors


class SliceInterp:
    """Build target coordinates once and interpolate new values repeatedly.

    Values follow the row order of points; pandas column names are preserved.
    method='sdf' reuses the shape grid and requires return_info support in
    _interp_shape.py. prob/vtk/kdt/sci retain their original value backends.
    Automatic target slice counts are strictly less than max_slice.
    record=True stores each value group in values and values_itp; a repeated
    group replaces its previous record. get_interps() combines all records;
    the latest recorded value wins when groups share a column name.
    SDF options: mapping='grid'/'centroid', label_mode='prob' (default)/'sdf'.
    SDF numeric predictions stay within the finite observed channel range;
    this does not guarantee preservation of extrema or variance.
    Only points, points_itp, values, values_itp and reusable interpolation
    geometry are retained. Bounded interpolation caches are reused between
    calls; save(keep_cache=False) excludes them from the file. Merged outputs
    and intermediate volumes are not recorded. clear_cache() releases runtime caches.
    sample_cache bounds retained corner maps and local spline weights in MiB;
    once full, new maps are computed without evicting reusable cached maps.
    Call clear_cache() after reducing this budget to release existing maps.
    """

    def __init__(self, points, method="sdf", n_points=21,
                 kernel="linear", n_slices=None, max_slice=500,
                 shape_method="pspline", shape_options=None, z_new=None,
                 keep_observed=True, mapping="grid", label_mode="prob", values=None,
                 anisotropy=0.0, level_scale=None, solver="auto",
                 factor_cache=4, sample_cache=128):
        self.points = np.asarray(points, dtype=float).copy()
        if (self.points.ndim != 2 or self.points.shape[1] != 3
                or not len(self.points) or not np.isfinite(self.points).all()):
            raise ValueError("points must be a nonempty finite (N, 3) array.")
        self.z_obs = np.unique(self.points[:, 2])
        if len(self.z_obs) < 2:
            raise ValueError("At least two observed z planes are required.")
        if z_new is not None and n_slices is not None:
            raise ValueError("Specify z_new or n_slices, not both.")
        if mapping not in ("grid", "centroid"):
            raise ValueError("mapping must be 'grid' or 'centroid'.")
        if label_mode not in ("sdf", "prob"):
            raise ValueError("label_mode must be 'sdf' or 'prob'.")
        if isinstance(max_slice, bool) or int(max_slice) != max_slice or max_slice < 2:
            raise ValueError("max_slice must be an integer >= 2.")
        if not np.isfinite(anisotropy) or not 0 <= anisotropy < 1:
            raise ValueError("anisotropy must be in [0, 1).")
        if level_scale is not None and (not np.isfinite(level_scale) or level_scale <= 0):
            raise ValueError("level_scale must be positive or None.")
        if solver not in ("auto", "lu", "cg"):
            raise ValueError("solver must be auto/lu/cg.")
        if isinstance(factor_cache, bool) or int(factor_cache) != factor_cache or factor_cache < 0:
            raise ValueError("factor_cache must be a nonnegative integer.")
        if not np.isfinite(sample_cache) or sample_cache < 0:
            raise ValueError("sample_cache must be a nonnegative size in MiB.")
        self.anisotropy, self.level_scale = float(anisotropy), level_scale
        self.solver, self.factor_cache = solver, int(factor_cache)
        self.sample_cache = float(sample_cache)
        self.method = method
        self.n_points = n_points
        self.kernel = kernel
        self.n_slices = n_slices
        self.max_slice = int(max_slice)
        self.shape_method = shape_method
        self.shape_options = dict(shape_options or {})
        if set(self.shape_options) & {"points", "z_obs", "z_new", "method", "return_info", "return_z_weights"}:
            raise ValueError("Pass shape_method separately; shape_options contains reserved keys.")
        self.keep_observed, self.mapping, self.label_mode = keep_observed, mapping, label_mode
        if z_new is None:
            self.z_new = None
        else:
            z = np.asarray(z_new, dtype=float)
            if z.ndim != 1 or not len(z) or not np.isfinite(z).all():
                raise ValueError("z_new must be a nonempty finite 1D array.")
            self.z_new = np.unique(z)
        self.points_itp = None
        self.values_itp = {}
        self.values = {}
        self.shape_info = None
        self._cache = {}
        self._rows = [np.flatnonzero(self.points[:, 2] == z) for z in self.z_obs]
        if values is not None:
            self.interp_value(values)

    def get_nslice(self):
        """Return target plane count; n_slices includes retained observations."""
        from scipy.spatial import cKDTree

        if self.z_new is None:
            n = self.n_slices
            if n is None:
                d = [cKDTree(self.points[r, :2]).query(self.points[r, :2], k=2)[0][:, 1]
                     for r in self._rows if len(r) > 1]
                d = np.concatenate(d) if d else np.empty(0)
                d = d[np.isfinite(d) & (d > 0)]
                if not len(d):
                    raise ValueError("No positive XY spacing; provide denser observed slices.")
                n = max(len(self.z_obs), int(np.round(np.ptp(self.z_obs) / np.median(d))) + 1)
                n = min(n, self.max_slice - 1)
            if isinstance(n, bool) or int(n) != n or n < 2:
                raise ValueError("n_slices must be an integer >= 2.")
            n = int(n)
            if self.keep_observed:
                if n < len(self.z_obs):
                    raise ValueError("n_slices must cover all observed planes; increase max_slice if automatic.")
                # Allocate extra planes by gap length, keeping the requested total.
                gaps = np.diff(self.z_obs)
                extra = (n - len(self.z_obs)) * gaps / gaps.sum()
                counts = np.floor(extra).astype(int)
                order = np.argsort(-(extra - counts), kind="stable")
                counts[order[:n - len(self.z_obs) - counts.sum()]] += 1
                self.z_new = np.r_[np.concatenate([
                    np.linspace(a, b, c + 2)[:-1]
                    for a, b, c in zip(self.z_obs[:-1], self.z_obs[1:], counts)
                ]), self.z_obs[-1]]
            else:
                self.z_new = np.linspace(self.z_obs[0], self.z_obs[-1], n)
        else:
            if self.z_new.ndim != 1 or not len(self.z_new) or not np.isfinite(self.z_new).all():
                raise ValueError("z_new must be a nonempty finite 1D array.")
            if self.keep_observed:
                self.z_new = np.union1d(self.z_new, self.z_obs)
        if self.z_new[0] < self.z_obs[0] or self.z_new[-1] > self.z_obs[-1]:
            raise ValueError("z_new must lie inside the observed z range.")
        self.n_slices = len(self.z_new)
        return self.n_slices

    def interpolate_slices(self):
        """Build shared geometry once and align its metadata with actual output."""
        if self.points_itp is None:
            self._cache.clear()
            self.get_nslice()
            Zs = [self.points[r, :2] for r in self._rows]
            sampled, self.shape_info = interpolate_slices(
                Zs, self.z_obs, self.z_new, method=self.shape_method,
                return_info=True, return_z_weights=False, **self.shape_options,
            )
            targets = list(sampled)
            if self.keep_observed:
                for i, z in enumerate(self.z_new):
                    k = np.searchsorted(self.z_obs, z)
                    if k < len(self.z_obs) and self.z_obs[k] == z:
                        targets[i] = self.points[self._rows[k], :2]
            self.slice_offsets = np.r_[0, np.cumsum([len(p) for p in targets])]
            self.points_itp = np.concatenate([
                np.c_[p, np.full(len(p), z)] for p, z in zip(targets, self.z_new)
            ])
            self.shape_info = self._shape_state(self.shape_info)
            # Share the effective weights so geometry can be rebuilt without a copy.
            self.shape_options["reliability"] = self.shape_info["z_operator"]["reliability"]
        return self.points_itp

    @staticmethod
    def _shape_state(info):
        """Keep only geometry used by value interpolation, without target grids."""
        from scipy.ndimage import label

        if info is None:
            state = None
        else:
            state = {key: info[key] for key in (
                "x", "y", "dx", "dy", "dxy", "masks_obs", "sdf_obs",
                "z_operator", "centroid_safe", "centers_obs", "centers_new",
            ) if key in info}
            if ("centroid_safe" not in state
                    and all(key in info for key in ("x", "y", "masks_obs", "masks_new"))):
                state["centroid_safe"] = all(
                    label(m)[1] <= 1 for m in list(info["masks_obs"]) + list(info["masks_new"])
                )
                for key, masks in (("centers_obs", info["masks_obs"]),
                                   ("centers_new", info["masks_new"])):
                    centers = []
                    for mask in masks:
                        iy, ix = np.where(mask)
                        if len(ix):
                            centers.append([info["x"][ix].mean(), info["y"][iy].mean()])
                        else:
                            centers.append([0.0, 0.0])
                    state[key] = np.asarray(centers)
        return state

    def _geometry(self, k):
        """Cache conservative splatting and optional SDF-aware conductances."""
        from collections import OrderedDict
        from scipy.sparse import coo_matrix, diags
        from scipy.sparse.csgraph import connected_components
        from scipy.spatial import cKDTree

        config = (getattr(self, "anisotropy", 0.0), getattr(self, "level_scale", None))
        if self._cache.get("config") != config:
            self._cache.clear()
            self._cache["config"] = config
        if k not in self._cache:
            info = self.shape_info
            mask = info["masks_obs"][k]
            ids = np.full(mask.shape, -1, dtype=int)
            ids[mask] = np.arange(mask.sum())
            p = self.points[self._rows[k], :2]
            uv = (p - [info["x"][0], info["y"][0]]) / [info["dx"], info["dy"]]
            base = np.floor(uv).astype(int)
            frac = uv - base
            row, col, val = [], [], []
            for a, b in ((0, 0), (1, 0), (0, 1), (1, 1)):
                ix, iy = base[:, 0] + a, base[:, 1] + b
                w = (frac[:, 0] if a else 1 - frac[:, 0]) * (frac[:, 1] if b else 1 - frac[:, 1])
                good = (ix >= 0) & (ix < mask.shape[1]) & (iy >= 0) & (iy < mask.shape[0])
                src = np.flatnonzero(good)
                dst = ids[iy[good], ix[good]]
                good = (dst >= 0) & (w[src] > 0)
                row.extend(dst[good]); col.extend(src[good]); val.extend(w[src[good]])
            S = coo_matrix((val, (row, col)), shape=(int(mask.sum()), len(p))).tocsr()
            mass = np.asarray(S.sum(0)).ravel()
            missing = np.flatnonzero(mass == 0)
            if len(missing) and mask.any():
                iy, ix = np.where(mask)
                xy = np.c_[info["x"][ix], info["y"][iy]]
                nearest = cKDTree(xy).query(p[missing])[1]
                S += coo_matrix((np.ones(len(missing)), (nearest, missing)), shape=S.shape).tocsr()
                mass[missing] = 1.0
            inv = np.zeros_like(mass)
            np.divide(1.0, mass, out=inv, where=mass > 0)
            S = (S @ diags(inv)).tocsr()

            # Positive symmetric edge weights preserve the discrete maximum principle.
            phi = info["sdf_obs"][k]
            anisotropy = getattr(self, "anisotropy", 0.0)
            level_scale = getattr(self, "level_scale", None)
            if anisotropy:
                gy, gx = np.gradient(phi, info["dy"], info["dx"])
                norm = np.hypot(gx, gy)
                nx = np.divide(gx, norm, out=np.zeros_like(gx), where=norm > 0)
                ny = np.divide(gy, norm, out=np.zeros_like(gy), where=norm > 0)
            else:
                nx = np.zeros_like(phi)
                ny = np.zeros_like(phi)
            row, col, val = [], [], []
            for a, b, pa, pb, na, nb, step in (
                (ids[:, :-1], ids[:, 1:], phi[:, :-1], phi[:, 1:], nx[:, :-1], nx[:, 1:], info["dx"]),
                (ids[:-1, :], ids[1:, :], phi[:-1, :], phi[1:, :], ny[:-1, :], ny[1:, :], info["dy"]),
            ):
                good = (a >= 0) & (b >= 0)
                weight = 1 - anisotropy * (na ** 2 + nb ** 2) / 2
                if level_scale is not None:
                    weight *= np.exp(-0.5 * ((pa - pb) / level_scale) ** 2)
                weight = np.maximum(weight, 1e-8) / step ** 2
                row.extend(a[good]); col.extend(b[good]); val.extend(weight[good])
            if anisotropy:
                # Diagonal directions distinguish tangent from normal at oblique walls.
                # Both intervening axis neighbors must be inside: do not bridge corners.
                step = np.hypot(info["dx"], info["dy"])
                for left, right, sign, mid_a, mid_b in (
                    ((slice(None, -1), slice(None, -1)), (slice(1, None), slice(1, None)), 1,
                     ids[:-1, 1:], ids[1:, :-1]),
                    ((slice(None, -1), slice(1, None)), (slice(1, None), slice(None, -1)), -1,
                     ids[:-1, :-1], ids[1:, 1:]),
                ):
                    a, b = ids[left], ids[right]
                    good = (a >= 0) & (b >= 0) & (mid_a >= 0) & (mid_b >= 0)
                    na = (sign * info["dx"] * nx[left] + info["dy"] * ny[left]) / step
                    nb = (sign * info["dx"] * nx[right] + info["dy"] * ny[right]) / step
                    weight = 1 - anisotropy * (na ** 2 + nb ** 2) / 2
                    if level_scale is not None:
                        weight *= np.exp(-0.5 * ((phi[left] - phi[right]) / level_scale) ** 2)
                    weight = np.maximum(weight, 1e-8) / step ** 2
                    row.extend(a[good]); col.extend(b[good]); val.extend(weight[good])
            A = coo_matrix((val, (row, col)), shape=(int(mask.sum()), int(mask.sum()))).tocsr()
            A = A + A.T
            L = diags(np.asarray(A.sum(1)).ravel()) - A
            _, components = connected_components(A, directed=False)
            self._cache[k] = dict(
                S=S, L=L.tocsr(), components=components, factors=OrderedDict(),
                ids=ids, samples={}, pattern_keys=set(),
            )
        return self._cache[k]

    @staticmethod
    def _robust_polish(u, mean, mass, laplacian, solve, tau, tol, maxiter, use_cg=False):
        """Exact-Huber Newton polishing with component shifts and exact line search.

        No objective ridge, threshold change, or permanent per-gene factor cache.
        Returns the polished fields and maximum number of Newton iterations used.
        """
        from inspect import signature
        from scipy.optimize import brentq
        from scipy.sparse import diags
        from scipy.sparse.csgraph import connected_components
        from scipy.sparse.linalg import cg, spsolve

        u = u.copy()
        tau = np.broadcast_to(tau, u.shape[1:])
        known = mass > 0
        count, labels = connected_components(laplacian, directed=False)
        components = [np.flatnonzero(labels == k) for k in range(count)]
        absolute, degree = abs(laplacian), laplacian.diagonal()
        eps, maximum = np.finfo(float).eps, 0
        for col in range(u.shape[1]):
            origin = np.zeros(len(mass))
            for rows in components:
                observed = rows[known[rows]]
                order = np.argsort(mean[observed, col])
                cumulative = np.cumsum(mass[observed][order])
                index = order[np.searchsorted(cumulative, .5*cumulative[-1])]
                origin[rows] = mean[observed[index], col]
            target = (mean[:, col]-origin)/tau[col]
            field = (u[:, col]-origin)/tau[col]
            for iteration in range(maxiter+1):
                # Certify in the original units, including representable roundoff.
                current = field*tau[col]+origin
                residual = current-mean[:, col]
                diffusion = laplacian @ current
                gradient = mass*np.clip(residual, -tau[col], tau[col])+diffusion
                scale = max(mass.max()*tau[col], np.max(np.abs(diffusion)))
                rounding = 64*eps*(absolute @ np.abs(current) + mass*(tau[col]
                    + (np.abs(residual) <= tau[col])*(np.abs(current)+np.abs(mean[:, col]))))
                if np.all(np.abs(gradient) <= tol*scale+rounding):
                    u[:, col] = current
                    break
                if iteration == maxiter:
                    raise RuntimeError(f"Robust Huber polishing did not converge for column {col}.")
                # Solve each constant graph null mode exactly before Newton.
                for rows in components:
                    observed = rows[known[rows]]
                    locations = target[observed]-field[observed]
                    weights = mass[observed]
                    order = np.argsort(locations)
                    cumulative = np.cumsum(weights[order])
                    median = locations[order[np.searchsorted(cumulative, .5*cumulative[-1])]]
                    def derivative_shift(shift):
                        return weights @ np.clip(shift-locations, -1., 1.)
                    lower, upper = median-1., median+1.
                    if abs(derivative_shift(lower)) <= .01*tol*weights.max():
                        shift = lower
                    elif abs(derivative_shift(upper)) <= .01*tol*weights.max():
                        shift = upper
                    elif lower == upper:
                        shift = median
                    else:
                        shift = brentq(derivative_shift, lower, upper,
                                       xtol=.01*tol, rtol=8*eps, maxiter=100)
                    field[rows] += shift
                residual = field-target
                diffusion = laplacian @ field
                gradient = mass*np.clip(residual, -1., 1.)+diffusion
                active = known & (np.abs(residual) <= 1.+.1*tol
                                 + 64*eps*(np.abs(field)+np.abs(target)))
                diagonal = mass*active
                for rows in components:
                    if not np.any(active[rows]):
                        anchor = rows[np.argmax(mass[rows])]
                        diagonal[anchor] = mass[anchor]
                hessian = (laplacian+diags(diagonal)).tocsc()
                if use_cg:
                    options = dict(maxiter=max(1000, min(10000, 10*len(mass))), atol=0.)
                    options["rtol" if "rtol" in signature(cg).parameters else "tol"] = min(1e-10, tol*.01)
                    direction, status = cg(hessian, -gradient,
                                           M=diags(1./hessian.diagonal()), **options)
                else:
                    direction, status = spsolve(hessian, -gradient), 0
                if status != 0 or not np.isfinite(direction).all() or gradient @ direction >= 0:
                    direction = -solve(gradient[:, None])[:, 0]
                if not np.isfinite(direction).all() or gradient @ direction >= 0:
                    direction = -gradient/(mass+degree)
                curvature = max(float(direction @ (laplacian @ direction)), 0.)
                slope = float(direction @ diffusion)
                def derivative_step(step):
                    return (mass*np.clip(residual+step*direction, -1., 1.)) @ direction + slope + step*curvature
                upper = 1.
                for search in range(60):
                    if derivative_step(upper) >= 0:
                        break
                    upper *= 2.
                else:
                    raise RuntimeError("Could not bracket the robust Huber directional minimum.")
                step = brentq(derivative_step, 0., upper, xtol=8*eps, rtol=8*eps, maxiter=100)
                field += step*direction
            maximum = max(maximum, iteration)
        return u, maximum

    @staticmethod
    def _robust_complete(solution, rhs, mass, laplacian, solve, threshold,
                         maxiter, tol, stats, use_cg=False):
        """Fit Huber fidelity with shared solves and active-set polishing.

        Center each connected component without changing the objective. The
        shared-factor split is a fast path, not a guaranteed finite-step solver.
        Original-gradient tolerances include an additive floating-error bound.
        """
        from scipy.sparse.csgraph import connected_components

        known = mass > 0
        mean = np.zeros_like(rhs)
        np.divide(rhs, mass[:, None], out=mean, where=known[:, None])
        amplitude = np.max(np.abs(mean[known]), axis=0)
        count, labels = connected_components(laplacian, directed=False)
        origins = np.empty((count, rhs.shape[1]))
        for component in range(count):
            observed = known & (labels == component)
            origins[component] = np.median(mean[observed], axis=0)
            mean[observed] -= origins[component]
        if isinstance(threshold, (bool, np.bool_)):
            degree = laplacian.diagonal()
            neighbor_mass = degree * known - laplacian @ known.astype(float)
            valid = known & (neighbor_mass > 0)
            tau = np.full(rhs.shape[1], np.inf)
            if valid.sum() >= 3:
                # Use observed neighbors only: a fitted impulse's harmonic
                # tails must not become a tiny, artificial noise estimate.
                neighbors = degree[:, None] * mean - laplacian @ mean
                residual = mean[valid] - neighbors[valid] / neighbor_mass[valid, None]
                center = np.median(residual, axis=0)
                scale = 1.4826 * np.median(np.abs(residual - center), axis=0)
                floor = 64 * np.finfo(float).eps * amplitude
                tau = np.where(scale > floor, 2.5 * scale, np.inf)
            stats["auto_fallback"] += int((~np.isfinite(tau)).sum())
        else:
            tau = np.full(rhs.shape[1], float(threshold))
        active = np.flatnonzero(np.isfinite(tau))
        stats["robust_fits"] += len(active)
        if len(active):
            if len(active) == rhs.shape[1]:
                active = slice(None)
            mean, tau = mean[:, active], tau[active]
            b = mass[:, None] * mean
            u = solve(b)
            absolute = abs(laplacian)
            limit = min(16, maxiter)
            for iteration in range(limit + 1):
                residual = u - mean
                clipped = np.clip(residual, -tau, tau)
                lap = laplacian @ u
                scale = np.maximum(np.max(mass) * tau, np.max(np.abs(lap), axis=0))
                rounding = 64 * np.finfo(float).eps * (
                    absolute @ np.abs(u) + mass[:, None]
                    * ((np.abs(residual) <= tau) * (np.abs(u) + np.abs(mean)) + tau))
                lap += mass[:, None] * clipped
                pending = np.flatnonzero(~np.all(np.abs(lap) <= tol * scale + rounding, axis=0))
                if not len(pending):
                    break
                if iteration < limit:
                    u[:, pending] = solve(b[:, pending] + mass[:, None]
                                          * (residual[:, pending] - clipped[:, pending]))
            if len(pending):
                u[:, pending], extra = SliceInterp._robust_polish(
                    u[:, pending], mean[:, pending], mass, laplacian, solve,
                    tau[pending], tol, maxiter - iteration, use_cg=use_cg,
                )
                stats["polish_fits"] = stats.get("polish_fits", 0) + len(pending)
                stats["polish_iterations"] = max(stats.get("polish_iterations", 0), extra)
                iteration += extra
            solution[:, active] = u + origins[:, active][labels]
            stats["max_iterations"] = max(stats["max_iterations"], iteration)
        return solution

    def _complete(self, values, k, value_smooth=0.0, zero_weight=1.0,
                  value_robust=False, robust_maxiter=200, robust_tol=1e-5, stats=None):
        """Complete grouped channels with bounded LU caching or preconditioned CG.

        value_smooth=0 retains hard finite observations. Positive smoothing
        uses the mass-lumped system (diag(mass) + value_smooth*dxy**2*L)u
        = sums; zero_weight changes finite zero observations only in this
        optional soft-fidelity solve. Optional Huber fidelity acts on aggregated
        grid observations. The fast path shares factors; stalled channels use
        temporary weighted systems without growing the factor cache.
        """
        from hashlib import blake2b
        from scipy.sparse import diags
        from scipy.sparse.linalg import cg
        from inspect import signature

        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[0] != len(self._rows[k]):
            raise ValueError("Completion values must have one row per source-slice point.")
        g = self._geometry(k)
        S, L = g["S"], g["L"]
        finite = np.isfinite(values)
        shared = finite.all() and (value_smooth == 0 or zero_weight == 1)
        reliability = np.ones((len(values), 1)) if shared else finite.astype(float)
        if value_smooth > 0 and zero_weight != 1:
            reliability[finite & (values == 0)] = zero_weight
        mass = S @ reliability
        sums = S @ (np.where(finite, values, 0.0) * reliability)
        fields = np.full(sums.shape, np.nan)
        if value_robust and stats is None:
            stats = dict(robust_fits=0, auto_fallback=0, max_iterations=0)
        if value_smooth == 0:
            np.divide(sums, mass, out=fields, where=mass > 0)
        groups = {}
        for j in range(mass.shape[1]):
            if value_smooth == 0:
                pattern = np.packbits(mass[:, j] > 0).tobytes()
            else:
                pattern = np.ascontiguousarray(mass[:, j]).tobytes()
            key = (float(value_smooth), blake2b(pattern, digest_size=16).digest())
            groups.setdefault(key, []).append(j)
        if shared:
            groups[key] = list(range(values.shape[1]))
        factors = g["factors"]
        seen = g["pattern_keys"]
        for key in groups:
            if len(seen) <= 8:
                seen.add(key)
        solver = getattr(self, "solver", "auto")
        cache_size = getattr(self, "factor_cache", 4)
        for key, columns in groups.items():
            j = 0 if shared else columns[0]
            known = mass[:, j] > 0
            supported = np.isin(g["components"], np.unique(g["components"][known]))
            if value_smooth == 0:
                unknown = supported & ~known
                if unknown.any():
                    rhs = -(L[unknown][:, known] @ fields[np.ix_(known, columns)])
                else:
                    rhs = None
            else:
                unknown = supported
                if unknown.any():
                    rhs = sums[np.ix_(unknown, columns)]
                else:
                    rhs = None
            if rhs is not None:
                use_cg = solver == "cg" or (solver == "auto" and len(seen) > 8 and key not in factors)
                if use_cg or key not in factors:
                    A = L[unknown][:, unknown].tocsc()
                    if value_smooth > 0:
                        A = (diags(mass[unknown, j]) + value_smooth * self.shape_info["dxy"] ** 2 * A).tocsc()
                if use_cg:
                    M = diags(1.0 / A.diagonal())
                    options = dict(maxiter=max(1000, min(10000, 10 * A.shape[0])))
                    if "rtol" in signature(cg).parameters:
                        options.update(rtol=1e-8, atol=0.0)
                    else:
                        options.update(tol=1e-8, atol=0.0)
                    def solve(rhs):
                        solution = np.empty_like(rhs)
                        for q in range(rhs.shape[1]):
                            solution[:, q], status = cg(A, rhs[:, q], M=M, **options)
                            if status != 0:
                                raise RuntimeError(f"CG completion failed on slice {k}, channel {columns[q]} (status={status}).")
                        return solution
                else:
                    if key in factors:
                        factor = factors.pop(key)
                    else:
                        factor = _LUFactor(A)
                    solve = factor.solve
                    if cache_size > 0:
                        factors[key] = factor
                        while len(factors) > cache_size:
                            factors.popitem(last=False)
                solution = solve(np.asarray(rhs))
                if value_robust:
                    laplacian = value_smooth * self.shape_info["dxy"] ** 2 * L[unknown][:, unknown]
                    solution = self._robust_complete(
                        solution, rhs, mass[unknown, j], laplacian, solve,
                        value_robust, robust_maxiter, robust_tol, stats, use_cg=use_cg,
                    )
                fields[np.ix_(unknown, columns)] = solution
        for j in range(values.shape[1]):
            if finite[:, j].any():
                fields[:, j] = np.clip(fields[:, j], values[finite[:, j], j].min(),
                                       values[finite[:, j], j].max())
        return fields

    def _sample(self, fields, k, t, mapping, signed=False, all_finite=False):
        """Sample with distance support in both numerator and denominator."""
        from scipy.ndimage import distance_transform_edt
        import warnings

        out = np.full((self.slice_offsets[t + 1] - self.slice_offsets[t], fields.shape[1]), np.nan)
        mass = np.zeros((len(out), 1)) if all_finite else np.zeros_like(out)
        if len(fields):
            info = self.shape_info
            g = self._geometry(k)
            if mapping == "centroid":
                if "centroid_safe" not in self._cache:
                    self._cache["centroid_safe"] = info["centroid_safe"]
                    if not self._cache["centroid_safe"]:
                        warnings.warn("mapping='centroid' requires a single connected component per slice; using 'grid'.",
                                      RuntimeWarning, stacklevel=2)
                if not self._cache["centroid_safe"]:
                    mapping = "grid"
            key = (t, mapping)
            samples = g["samples"].get(key)
            if samples is None:
                p = self.points_itp[self.slice_offsets[t]:self.slice_offsets[t + 1], :2].copy()
                mask = info["masks_obs"][k]
                if mapping == "centroid":
                    p += info["centers_obs"][k] - info["centers_new"][t]
                if "nearest" not in g:
                    g["nearest"] = distance_transform_edt(~mask, sampling=(info["dy"], info["dx"]),
                                                           return_indices=True)
                distance, indices = g["nearest"]
                uv = (p - [info["x"][0], info["y"][0]]) / [info["dx"], info["dy"]]
                base = np.floor(uv).astype(int)
                frac = uv - base
                samples = []
                for a, b in ((0, 0), (1, 0), (0, 1), (1, 1)):
                    raw_x, raw_y = base[:, 0] + a, base[:, 1] + b
                    ix = np.clip(raw_x, 0, mask.shape[1] - 1)
                    iy = np.clip(raw_y, 0, mask.shape[0] - 1)
                    w = (frac[:, 0] if a else 1 - frac[:, 0]) * (frac[:, 1] if b else 1 - frac[:, 1])
                    nearest_y, nearest_x = indices[:, iy, ix]
                    d = distance[iy, ix] + np.hypot((raw_x - ix) * info["dx"], (raw_y - iy) * info["dy"])
                    samples.append((g["ids"][nearest_y, nearest_x], d,
                                    w * np.exp(-d / info["dxy"])))
                size = sum(a.nbytes for corner in samples for a in corner)
                used = self._cache.get("sample_bytes", 0)
                spline_used = self._cache.get("spline", {}).get("used_bytes", 0)
                if used + spline_used + size <= getattr(self, "sample_cache", 128) * 1024 ** 2:
                    g["samples"][key] = samples
                    self._cache["sample_bytes"] = used + size
            sums = np.zeros_like(out)
            for ids, distance, weight in samples:
                if not weight.any():
                    continue
                if (weight != 0).all():
                    rows = slice(None)
                else:
                    rows = np.flatnonzero(weight)
                f = fields[ids[rows]]
                w = weight[rows, None]
                if signed:
                    # Class signed distances are negative outside the source domain.
                    d = distance[rows, None]
                    f = np.where(d > 0, np.minimum(f, -d), f)
                if all_finite:
                    sums[rows] += f * w
                    mass[rows] += w
                else:
                    valid = np.isfinite(f)
                    sums[rows] += np.where(valid, f, 0.0) * w
                    mass[rows] += valid * w
            np.divide(sums, mass, out=out, where=mass > 0)
        return out, np.clip(mass, 0, 1)

    def _transfer(self, fields, mapping, signed=False, keep_observed_values=None):
        """Fuse spatial samples along z using their point/channel support."""
        result = np.full((len(self.points_itp), fields[0].shape[1]), np.nan)
        info = self.shape_info
        options = info.get("z_operator", dict(method=self.shape_method)).copy()
        reliability = options.pop("reliability", None)
        all_finite = all(np.isfinite(f).all() for f in fields)
        preserve = self.keep_observed if keep_observed_values is None else keep_observed_values
        observed = np.isin(self.z_new, self.z_obs) if preserve else np.zeros(len(self.z_new), dtype=bool)
        for t in range(len(self.z_new)):
            rows = slice(self.slice_offsets[t], self.slice_offsets[t + 1])
            if rows.start != rows.stop and not observed[t]:
                active = np.arange(len(fields))
                if options["method"] == "linear":
                    hi = np.clip(np.searchsorted(self.z_obs, self.z_new[t], side="right"), 1, len(fields) - 1)
                    active = np.arange(hi - 1, hi + 1)
                for active in (active, np.arange(len(fields))):
                    samples = [self._sample(fields[k], k, t, mapping, signed=signed, all_finite=all_finite)
                               for k in active]
                    values = np.stack([v for v, c in samples])
                    supports = np.stack([c for v, c in samples])
                    if reliability is not None:
                        R = np.asarray(reliability)
                        if R.ndim <= 1:
                            supports *= np.broadcast_to(R, (len(fields),))[active, None, None]
                        else:
                            R = np.broadcast_to(R, info["masks_obs"].shape)
                            R = np.stack([
                                self._sample(R[k][info["masks_obs"][k]][:, None], k, t, mapping)[0]
                                for k in active
                            ])
                            np.multiply(supports, R, out=supports, where=supports > 0)
                    # Missing local observations still use the full reliable z range.
                    if len(active) == len(fields) or (np.isfinite(values).all() and (supports > 0).all()):
                        cache = None
                        if options["method"] == "pspline":
                            cache = self._cache.setdefault("spline", {})
                            cache["limit_bytes"] = max(0, getattr(self, "sample_cache", 128) * 1024 ** 2
                                                       - self._cache.get("sample_bytes", 0))
                        result[rows] = apply_slice_z(
                            values, self.z_obs[active], self.z_new[t:t + 1], reliability=supports,
                            _cache=cache, **options,
                        )[0]
                        break
        return result

    def interp_value(self, values, method=None, n_points=None, kernel=None,
                     record=True, value_type="auto", label_mode=None, mapping=None,
                     value_smooth=0.0, zero_weight=1.0, label_temperature=1.0,
                     value_batch=64, value_clip=True, value_robust=False,
                     keep_observed_values=None, robust_maxiter=200, robust_tol=1e-5,
                     **kargs):
        """Return xyz + values using cached geometry and recorded value groups.

        Numeric columns are continuous in auto mode; integer labels require
        categorical dtype or value_type='categorical'. value_smooth=0 retains
        hard observations; positive values use soft fidelity on the grid.
        zero_weight in (0,1] changes zero-observation reliability only in soft
        mode, and must be chosen under an explicit missing-detection assumption.
        value_batch bounds simultaneous numeric channels. value_clip bounds
        final numeric predictions after the shared z operator; no rescaling or
        distribution calibration is applied. value_robust=False keeps squared
        fidelity; True uses 2.5 times each source slice/gene's observed-neighbor
        contrast MAD as the Huber threshold (zero scale retains ordinary smoothing).
        A positive number specifies an absolute Huber threshold in expression
        units. Robust fitting requires positive value_smooth. Shared solves are
        followed by active-set polishing only for stalled channels, respecting
        the same robust_maxiter budget and robust_tol gradient test plus a
        floating-error bound. keep_observed_values=False evaluates continuous
        trends at retained source coordinates too; categorical values and the
        keep_observed geometry policy are unchanged. None retains old behavior.
        Only hard labels are returned,
        so positive label_temperature is retained for compatibility.
        The returned xyz + values table is not retained separately;
        get_interps() combines all records.
        """
        from scipy.ndimage import distance_transform_edt
        from pandas.api.types import is_numeric_dtype, is_bool_dtype

        if isinstance(values, pd.Series):
            source = values.to_frame(name=values.name if values.name is not None else "group_0")
        elif isinstance(values, pd.DataFrame):
            source = values
        else:
            v = np.asarray(values)
            if v.ndim == 1:
                v = v[:, None]
            if v.ndim != 2:
                raise ValueError("values must be 1D or 2D.")
            source = pd.DataFrame(v, columns=[f"group_{i}" for i in range(v.shape[1])])
        if len(source) != len(self.points) or not len(source.columns):
            raise ValueError("values must have one row per source point and at least one column.")
        if not source.columns.is_unique or any(c in ("x", "y", "z") for c in source.columns):
            raise ValueError("Value column names must be unique and distinct from x/y/z.")
        method = self.method if method is None else method
        n_points = self.n_points if n_points is None else n_points
        kernel = self.kernel if kernel is None else kernel
        label_mode = self.label_mode if label_mode is None else label_mode
        mapping = self.mapping if mapping is None else mapping
        if label_mode not in ("sdf", "prob") or mapping not in ("grid", "centroid"):
            raise ValueError("Invalid label_mode or mapping.")
        if (not np.isfinite(value_smooth) or value_smooth < 0
                or not np.isfinite(zero_weight) or not 0 < zero_weight <= 1):
            raise ValueError("value_smooth must be nonnegative; zero_weight must be in (0,1].")
        if not np.isfinite(label_temperature) or label_temperature <= 0:
            raise ValueError("label_temperature must be positive.")
        if isinstance(value_batch, bool) or int(value_batch) != value_batch or value_batch < 1:
            raise ValueError("value_batch must be a positive integer.")
        if value_smooth == 0 and zero_weight != 1:
            raise ValueError("zero_weight requires value_smooth > 0.")
        if not isinstance(value_robust, (bool, np.bool_)):
            if not np.isscalar(value_robust) or not np.isfinite(value_robust) or value_robust < 0:
                raise ValueError("value_robust must be False, True, or a positive Huber threshold.")
            value_robust = float(value_robust)
        if value_robust and value_smooth == 0:
            raise ValueError("value_robust requires value_smooth > 0.")
        if (isinstance(robust_maxiter, (bool, np.bool_)) or int(robust_maxiter) != robust_maxiter
                or robust_maxiter < 1 or not np.isfinite(robust_tol) or robust_tol <= 0):
            raise ValueError("robust_maxiter must be a positive integer; robust_tol must be positive.")
        if keep_observed_values is not None and not isinstance(keep_observed_values, (bool, np.bool_)):
            raise ValueError("keep_observed_values must be True, False or None.")
        preserve = self.keep_observed if keep_observed_values is None else keep_observed_values
        if preserve and not self.keep_observed:
            raise ValueError("keep_observed_values=True requires keep_observed=True geometry.")
        stats = dict(robust_fits=0, auto_fallback=0, max_iterations=0)
        self.interpolate_slices()
        try:
            if method == "sdf":
                if self.shape_info is None:
                    raise ValueError("method='sdf' requires the matching _interp_shape.py with return_info support.")
                if kargs:
                    raise TypeError("method='sdf' does not use backend options: " + ", ".join(kargs))
                out, kinds = {}, {}
                for col in source.columns:
                    kind = value_type.get(col, "auto") if isinstance(value_type, dict) else value_type
                    if kind == "auto":
                        if is_numeric_dtype(source[col].dtype) and not is_bool_dtype(source[col].dtype):
                            kind = "continuous"
                        else:
                            kind = "categorical"
                    if kind not in ("continuous", "categorical"):
                        raise ValueError("value_type must be auto/continuous/categorical or a column dict.")
                    kinds[col] = kind
                columns = [col for col in source.columns if kinds[col] == "continuous"]
                for start in range(0, len(columns), int(value_batch)):
                    cols = columns[start:start + int(value_batch)]
                    v = source[cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float, na_value=np.nan)
                    if np.isinf(v).any():
                        raise ValueError("Continuous values cannot contain infinity.")
                    fields = [self._complete(v[r], k, value_smooth=value_smooth, zero_weight=zero_weight,
                                             value_robust=value_robust, robust_maxiter=int(robust_maxiter),
                                             robust_tol=robust_tol, stats=stats)
                              for k, r in enumerate(self._rows)]
                    g = self._transfer(fields, mapping, keep_observed_values=preserve)
                    if value_clip:
                        for j in range(len(cols)):
                            finite = np.isfinite(v[:, j])
                            if finite.any():
                                g[:, j] = np.clip(g[:, j], v[finite, j].min(), v[finite, j].max())
                    out.update(zip(cols, g.T))
                for col in source.columns:
                    if kinds[col] == "categorical":
                        v = source[col]
                        valid = ~v.isna().to_numpy()
                        if isinstance(v.dtype, pd.CategoricalDtype):
                            classes = v.cat.categories[v.cat.categories.isin(v[valid])]
                        else:
                            classes = pd.Index(sorted(pd.unique(v[valid]),
                                                      key=lambda x: (type(x).__name__, repr(x))))
                        if not len(classes):
                            out[col] = np.nan
                        else:
                            codes = classes.get_indexer(v)
                            onehot = np.zeros((len(v), len(classes)))
                            onehot[np.flatnonzero(valid), codes[valid]] = 1.0
                            onehot[~valid] = np.nan
                            fields = [self._complete(onehot[r], k, value_smooth=value_smooth)
                                      for k, r in enumerate(self._rows)]
                            if label_mode == "sdf":
                                for k, f in enumerate(fields):
                                    mask = self.shape_info["masks_obs"][k]
                                    ok = np.isfinite(f).all(1)
                                    labels = np.full(mask.shape, -1, dtype=int)
                                    labels[mask] = np.where(ok, np.nan_to_num(f).argmax(1), -1)
                                    sdf = np.full_like(f, np.nan)
                                    for j in range(len(classes)):
                                        region = labels == j
                                        if region.any():
                                            # Include the object's outside in each class boundary.
                                            field = (distance_transform_edt(region, sampling=(self.shape_info["dy"], self.shape_info["dx"]))
                                                     - distance_transform_edt(~region, sampling=(self.shape_info["dy"], self.shape_info["dx"])))
                                        else:
                                            iy, ix = np.where(mask)
                                            if len(ix):
                                                diameter = np.hypot(np.ptp(ix) * self.shape_info["dx"], np.ptp(iy) * self.shape_info["dy"])
                                            else:
                                                diameter = 0.0
                                            field = np.full(mask.shape, -max(diameter, self.shape_info["dxy"]))
                                        sdf[ok, j] = field[mask][ok]
                                    fields[k] = sdf
                            g = self._transfer(fields, mapping, signed=label_mode == "sdf")
                            ok = np.isfinite(g).all(1)
                            if label_mode == "prob":
                                np.clip(g, 0, 1, out=g)
                                ok &= (g.sum(1) > 0)
                            labels = np.full(len(g), np.nan, dtype=object)
                            labels[ok] = classes.to_numpy()[g[ok].argmax(1)]
                            out[col] = labels
                out = pd.DataFrame(out, index=np.arange(len(self.points_itp)), columns=source.columns)
            elif method in ("prob", "vtk", "kdt", "sci"):
                if value_smooth != 0 or zero_weight != 1 or value_robust or keep_observed_values is not None:
                    raise ValueError("Value smoothing options apply only to method='sdf'.")
                keys = [f"group_{i}" for i in range(len(source.columns))]
                out = interp_value(self.points, source, self.points_itp, method=method, keys=keys,
                                   n_points=n_points, kernel=kernel, **kargs)
                out = pd.DataFrame(out).reset_index(drop=True)
                out.columns = source.columns
            else:
                raise ValueError("method must be sdf/prob/vtk/kdt/sci.")
            if self.keep_observed:
                restore = list(source.columns)
                if method == "sdf" and not preserve:
                    restore = [col for col in source.columns if kinds[col] == "categorical"]
                columns = source.columns.get_indexer(restore)
                for t, z in enumerate(self.z_new):
                    k = np.searchsorted(self.z_obs, z)
                    if len(columns) and k < len(self.z_obs) and self.z_obs[k] == z:
                        a, b = self.slice_offsets[t:t + 2]
                        out.iloc[a:b, columns] = source.iloc[self._rows[k], columns].to_numpy()
            if value_robust or (keep_observed_values is not None and not preserve):
                out.attrs["value_smoothing"] = dict(
                    value_smooth=float(value_smooth), value_robust=value_robust,
                    keep_observed_values=bool(preserve), **stats,
                )
            if record:
                key = tuple(out.columns)
                self.values.pop(key, None)
                self.values_itp.pop(key, None)
                self.values[key] = values
                self.values_itp[key] = out
            result = pd.concat([pd.DataFrame(self.points_itp, columns=list("xyz")), out], axis=1)
            result.attrs.update(out.attrs)
            return result
        except Exception:
            self._cache.clear()
            raise

    def clear_cache(self):
        """Release derived matrices, sparse factors and sampling caches."""
        self._cache.clear()

    def get_interps(self):
        """Return xyz and all recorded value groups, aligned by row position."""
        if self.points_itp is not None and len(self.values_itp):
            coords = pd.DataFrame(self.points_itp, columns=list("xyz"))
            values = pd.concat(list(self.values_itp.values()), axis=1)
            values = values.loc[:, ~values.columns.duplicated(keep="last")]
            interps = pd.concat([coords, values], axis=1)
        else:
            interps = None
        return interps

    def __getstate__(self):
        """Serialize minimal state unless save() explicitly requests caches."""
        state = {key: self.__dict__[key] for key in (
            "points", "points_itp", "values", "values_itp", "shape_info", "z_new",
            "method", "n_points", "kernel", "n_slices", "max_slice",
            "shape_method", "shape_options", "keep_observed", "mapping", "label_mode",
            "anisotropy", "level_scale", "solver", "factor_cache", "sample_cache",
        ) if key in self.__dict__}
        state["shape_info"] = self._shape_state(state.get("shape_info"))
        if state["shape_info"] is not None and "z_operator" in state["shape_info"]:
            state["shape_options"] = dict(state.get("shape_options", {}))
            state["shape_options"]["reliability"] = state["shape_info"]["z_operator"]["reliability"]
        if getattr(self, "_save_cache", False):
            state["_cache"] = self._cache
        return state

    def __setstate__(self, state):
        """Restore results and optional caches; rebuild inexpensive row indices."""
        cache = state.get("_cache", {})
        self.__dict__.update(state)
        state = self.__getstate__()
        self.__dict__.clear()
        self.__dict__.update(state)
        values = state.get("values", {})
        self.values = {}
        records = state.get("values_itp", {})
        for key, frame in records.items():
            columns = tuple(frame.columns)
            if key in values:
                self.values[columns] = values[key]
            elif columns in values:
                self.values[columns] = values[columns]
        self.values_itp = {tuple(frame.columns): frame for frame in records.values()}
        self.z_obs = np.unique(self.points[:, 2])
        self._rows = [np.flatnonzero(self.points[:, 2] == z) for z in self.z_obs]
        self._cache = cache
        self.sample_cache = state.get("sample_cache", 128.0)
        if cache and "sample_bytes" not in cache:
            cache["sample_bytes"] = sum(a.nbytes for g in cache.values() if isinstance(g, dict)
                                        for corners in g.get("samples", {}).values()
                                        for corner in corners for a in corner)
        if self.points_itp is not None:
            self.slice_offsets = np.r_[np.searchsorted(self.points_itp[:, 2], self.z_new), len(self.points_itp)]

    def save(self, path, keep_cache=False):
        """Save results and geometry; optionally retain caches for faster reuse.

        keep_cache=False omits derived matrices, factors and samples from the
        file without clearing runtime caches. True includes existing caches;
        loaded LU factors use sparse triangular solves without refactorization.
        """
        self._save_cache = bool(keep_cache)
        try:
            with open(path, "wb") as file:
                pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)
        finally:
            del self._save_cache

    @classmethod
    def load(cls, path):
        """Load a model saved with this class (trusted pickle files only)."""
        with open(path, "rb") as file:
            model = pickle.load(file)
        if not isinstance(model, cls):
            raise TypeError("The file does not contain a SliceInterp model.")
        return model


def interp_value(points, values, points_intp, method = 'prob', 
                 keys=None,
                   rescale=False,
                   radius = None, 
                   n_points=31,
                    null_strategy = 1,
                    sharp=2,
                    power=2,
                    temp=1,
                    null_value = None,
                    drop_aggna=True,
                    n_job = 3,
                    agg = None,
                    threshold=0,
                    knn_method= 'sknn',
                    kernel = 'linear'):
    '''
    null_strategy: MASK_POINTS = 0 , NULL_VALUE = 1 , CLOSEST_POINT = 2
    '''
    if method not in ('prob', 'sci', 'kdt', 'vtk'):
        raise ValueError("method must be prob/sci/kdt/vtk.")
    source = pd.DataFrame(values)
    values, keys = to_narray(values, points, keys=keys)
    points, points_intp = to_nparray(points, points_intp)

    dtype = np.array([
        'discrete' if isinstance(source.iloc[:, i].dtype, pd.CategoricalDtype)
        else vartype(source.iloc[:, i].to_numpy()) for i in range(len(keys))
    ])
    outs = {}
    dis_idx = dtype == 'discrete'
    con_idx = dtype == 'continuous'

    if method == 'prob':
        pitp = prob_interpolation(points, points_intp, n_points=n_points, method=knn_method,
                                  radius=radius, temp=temp)
        out = pitp(source, threshold=threshold,
                   null_value='_NONE' if null_value is None else null_value)[0]
        outs = {key: pd.to_numeric(out[:, i]) if con_idx[i] else out[:, i]
                for i, key in enumerate(keys)}
        return pd.DataFrame(outs, columns=keys)
    
    null_value = np.nan if null_value is None else null_value
    if np.any(dis_idx):
        values_s = values[:,dis_idx]
        values_l = np.arange(values_s.shape[0]).astype(np.int64)
        keys_s = np.array(keys)[dis_idx]

        if method == 'sci':
            import scipy as sci
            out_idx = sci.interpolate.griddata(points, values_l, points_intp,  rescale=rescale, 
                                               fill_value=null_value, method='nearest').astype(np.int64)
            outs_s = values_s[out_idx,:]
        elif method == 'kdt':
            outs_s = kd_interpolation(points, values_s, points_intp, n_points=n_points, radius=radius,
                                       agg=(agg or 'mode'), drop_aggna=drop_aggna, null_value=null_value)
        elif method == 'vtk':
            out_idx = vtk_interpolation(points, values_l, points_intp,
                                        keys = ['index'],
                                            radius = radius, 
                                            n_points=n_points,
                                            null_strategy = 2,
                                            sharp=sharp,
                                            power=power,
                                            null_value = null_value,
                                            kernel = 'voronoi')
            out_idx = out_idx['index'].astype(np.uint64).values
            outs_s = np.vstack([ values_s[:, i][out_idx] for i in range(values_s.shape[1]) ]).T

        outs.update(zip(keys_s, np.asarray(outs_s).T))

    if np.any(con_idx):
        values_s = np.asarray(values[:,con_idx], dtype=float)
        keys_s = np.array(keys)[con_idx]
        if method == 'sci':
            import scipy as sci
            outs_s = Parallel( n_jobs=n_job, verbose=0)( delayed(sci.interpolate.griddata)
                                (points, values_s[:,i], points_intp, rescale=rescale, 
                                 fill_value=null_value, method=kernel) 
                            for i in range(values_s.shape[1]))
            outs_s = np.vstack(outs_s).T
        elif method == 'kdt':
            outs_s = kd_interpolation(points, values_s, points_intp, n_points = n_points, radius=radius, 
                                        agg= (agg or 'mean'), drop_aggna=drop_aggna, null_value=null_value)
        elif method == 'vtk':
            outs_s = vtk_interpolation(points, values_s, points_intp,
                                        keys = keys_s,
                                            radius = radius, 
                                            n_points=n_points,
                                            null_strategy = null_strategy,
                                            sharp=sharp,
                                            power=power,
                                            null_value = null_value,
                                            kernel = kernel) 
        outs.update(zip(keys_s, np.asarray(outs_s).T))
    return pd.DataFrame(outs, index=np.arange(len(points_intp)), columns=keys)

def kd_interpolation(points, values, points_intp, n_points = None, radius=None, 
                     agg='mean', drop_aggna=True, null_value=np.nan):
    from scipy import spatial
    from scipy import stats
    points = np.asarray(points)
    values = np.asarray(values)

    if values.ndim ==1:
        values = values[:, np.newaxis]

    points_intp = np.asarray(points_intp)
    if agg == 'mean':
        aggfun = np.nanmean if drop_aggna else np.mean 
    elif agg == 'sum':
        aggfun = np.nansum if drop_aggna else np.sum
    elif agg == 'max':
        aggfun = np.nanmax if drop_aggna else np.max
    elif agg == 'min':
        aggfun = np.nanmin if drop_aggna else np.min
    elif agg == 'median':
        aggfun = np.nanmedian if drop_aggna else np.median
    elif agg == 'mode':
        def aggfun(x, **kargs):
            modes = [pd.Series(x[:, j]).mode(dropna=drop_aggna) for j in range(x.shape[1])]
            return [m.iloc[0] if len(m) else null_value for m in modes]
    elif callable(agg):
        aggfun = agg
    else:
        aggfun = eval(agg)

    p_tree = spatial.cKDTree(points)
    R, C = points_intp.shape[0], values.shape[1]
    dtype = float if np.issubdtype(values.dtype, np.number) else object
    values_intp = np.empty([R, C], dtype=dtype)
    values_intp[:] = null_value

    if not n_points is None:
        if isinstance(n_points, bool) or int(n_points) != n_points or n_points < 1:
            raise ValueError('n_points must be a positive integer.')
        n_points = min(int(n_points), len(points))
        if not n_points:
            raise ValueError('points must contain at least one source point.')
        qidx = p_tree.query(points_intp, k=n_points)
        qidx = (qidx[0].reshape(R, n_points), qidx[1].reshape(R, n_points))
        if not radius is None:
            qbool = ( qidx[0] <= radius )
            kinds = [ qidx[1][i][qbool[i]] for i in range(R) ]
        else:
            kinds = qidx[1]
    elif not radius is None:
        kinds = p_tree.query_ball_point(points_intp, radius)
    else:
        raise ValueError('Either n_points or radius must be specified.')

    for i in range(R):
        if len(kinds[i]):
            values_intp[i] = aggfun(values[kinds[i]], axis=0)
    return values_intp

def vtk_interpolation(points, values, points_intp,
                      keys = None,
                        radius = None, 
                        n_points=31,
                        null_strategy = 1,
                        sharp=2,
                        power=2,
                        null_value = np.nan,
                        null_to_closest = False,
                        kernel = 'linear'):
    '''
    null_strategy: MASK_POINTS = 0 , NULL_VALUE = 1 , CLOSEST_POINT = 2
    '''
    import vtk
    import pyvista as pv
    values, keys = to_narray(values, points, keys=keys)

    source_model = pv.PolyData(np.asarray(points))
    for i, ikey in enumerate(keys):
        source_model.point_data[ikey] = values[:,i]
    target_model = pv.PolyData(np.asarray(points_intp))

    locator = vtk.vtkPointLocator()
    locator.SetDataSet(source_model)
    locator.BuildLocator()

    if kernel.lower() == "shepard":
        kern = vtk.vtkShepardKernel()
        kern.SetPowerParameter(power)
    elif kernel.lower() == "gaussian":
        kern = vtk.vtkGaussianKernel()
        kern.SetSharpness(sharp)
    elif kernel.lower() == "linear":
        kern = vtk.vtkLinearKernel()
    elif kernel.lower() == "voronoi":
        kern = vtk.vtkVoronoiKernel()
    else:
        raise ValueError(
            "Available `kernel` are: shepard, gaussian, linear, voronoi."
        )

    if kernel.lower() != "voronoi":
        if radius is not None:
            kern.SetRadius(radius)
            kern.SetKernelFootprintToRadius()
        elif n_points:
            kern.SetNumberOfPoints(n_points)
            kern.SetKernelFootprintToNClosest()
        else:
            raise ValueError("Please set either radius or n_points")

    interpolator = vtk.vtkPointInterpolator()
    interpolator.SetInputData(target_model)
    interpolator.SetSourceData(source_model)
    interpolator.SetKernel(kern)
    interpolator.SetLocator(locator)
    interpolator.PassFieldArraysOff()
    interpolator.SetNullPointsStrategy(null_strategy)
    interpolator.SetNullValue(null_value)

    if kernel.lower() == "voronoi"  and null_to_closest :
        interpolator.SetNullPointsStrategyToClosestPoint()
    # interpolator.SetValidPointsMaskArrayName("ValidPointMask") #
    interpolator.Update()
    cpoly = interpolator.GetOutput()
    interpolated_model = pv.wrap(cpoly)
    inter_val = pd.DataFrame({ikey: interpolated_model[ikey] for ikey in keys})[keys]
    return inter_val

class onehot():
    def __init__(self):
        self.order = None
        self.order_dict = None
        self.num_classes = None

    def fit(self, X, order=None, null_value='None'):
        values = np.asarray(X, dtype=object)
        missing = pd.isna(values)
        if order is None:
            if isinstance(getattr(X, 'dtype', None), pd.CategoricalDtype):
                order = X.dtype.categories
            else:
                order = pd.unique(values[~missing])
        order = list(np.asarray(order, dtype=object)[~pd.isna(order)])
        self.null_index = -1
        if null_value is not None:
            if pd.isna(null_value) or null_value not in order:
                order.append(null_value)
            self.null_index = len(order) - 1 if pd.isna(null_value) else order.index(null_value)

        self.order = np.array(order, dtype=object)
        self.order_dict = {k: i for i, k in enumerate(order)}
        self.num_classes = len(order)

        return self.transform(X)

    def transform(self, X):
        X_id = np.array([self.null_index if pd.isna(k) else self.order_dict[k] for k in X])
        src = np.flatnonzero(X_id >= 0)
        dst = X_id[src]
        dist = np.ones(len(src))
        onehot = sci.sparse.csr_array((dist, (src, dst)), 
                                      shape=(len(X), self.num_classes))
        return onehot
    
    def fit_transform(self, X):
        return self.fit(X)
    
    def inverse_transform(self, X):
        return self.order[np.array(X).astype(np.int64)]

class prob_interpolation():
    def __init__(self, X, Y, sigma2=None, temp=1, method='sknn', 
                   radius = None, 
                   n_points=None):
        self.X = X
        self.Y = Y
        if (n_points is None) and (radius is None):
            n_points = 25
        self.W = self.gaussian_prob(sigma2=sigma2, temp=temp, method=method, 
                                    n_points=n_points, radius=radius)
    def gaussian_prob(self, sigma2=None, temp=1, method='sknn', n_points=31, 
                      radius=None):
        P, R, simga2 = kernel_xmm_k( self.Y, self.X,
                                    sigma2=sigma2, temp=temp, method=method, 
                                    knn=n_points, radius=radius)
        W = P/np.clip(P.sum(0), 1e-8, None) #N*M
        W = W.T
        W.eliminate_zeros()
        return W

    def smoothed_probs(self, weights, onehot_labels):
        return weights.dot(onehot_labels)

    def predict_labels(self, prob, threshold=0):
        if prob.shape[0] and prob.shape[1]:
            score = prob.max(1)
            if sci.sparse.issparse(score):
                score = score.toarray()
            score = np.asarray(score).ravel()
            index = np.asarray(prob.argmax(1)).ravel()
            index[(score <= 0) | (score < threshold)] = -1
        else:
            index = np.full(prob.shape[0], -1, dtype=int)
            score = np.zeros(prob.shape[0])
        return index, score

    def get_W_dict(self):
        sdzip = zip(self.W.nonzero()[0].tolist(),
                    self.W.nonzero()[1].tolist())
        W_dict = {}
        for s,d in sdzip:
            if s not in W_dict:
                W_dict[s] = [d]
            else:
                W_dict[s].append(d)
        self.W_dict = W_dict

    def predict_maxidx(self, ivalue, prob_label=None):
        if prob_label is None:
            W_mask = self.W.multiply(ivalue)
        else:
            # W_mask = prob_label[:, None] == ivalue[None,:]
            # W_mask = self.W * W_mask
            key_set = {}
            for i, k in enumerate(ivalue):
                if k not in key_set:
                    key_set[k] = [i]
                else:
                    key_set[k].append(i)

            incs = []
            for i, k in enumerate(prob_label):
                iv = list( set(key_set.get(k, [])) & set(self.W_dict.get(i, [])) )
                incs.append(iv)
            # incs = [ key_set.get(v, []) for v in prob_label ] #large array
            src = np.concatenate(incs).astype(int) if incs else np.empty(0, dtype=int)
            dst = np.repeat(np.arange(len(incs)), list(map(len, incs)))
            val = np.ones(len(src))

            W_mask = sci.sparse.csr_array((val, (dst, src)), 
                                          shape=self.W.shape)
            del src, dst, val, key_set, incs
            W_mask = self.W.multiply(W_mask)

        P_idx = np.asarray(W_mask.argmax(1)).ravel()
        P_didx = W_mask.max(1).toarray().flatten() <= 0.0
        P_idx[P_didx] = -1
        return P_idx

    def __call__(self, values, threshold=0.0, null_value='_NONE', predict_idx=False):
        values = pd.DataFrame(values)
        dtype = np.array([
            'discrete' if isinstance(values.iloc[:, i].dtype, pd.CategoricalDtype)
            else vartype(values.iloc[:, i].to_numpy()) for i in range(values.shape[1])
        ])

        outvalue = []
        outscore = []
        outsindx = []
        if ('discrete' in dtype) and predict_idx:
            self.get_W_dict()

        for i, itype in enumerate(dtype):
            ivalue = values.iloc[:, i]
            if itype == 'discrete':
                OH = onehot()
                IV = OH.fit(ivalue, null_value=null_value)
                P = self.smoothed_probs(self.W, IV)
                PI, PV = self.predict_labels(P, threshold=threshold)
                PL = np.full(len(PI), null_value, dtype=object)
                valid = PI >= 0
                PL[valid] = OH.inverse_transform(PI[valid])
                outvalue.append(PL)
                outscore.append(PV)
                if predict_idx:
                    PD = self.predict_maxidx(ivalue, prob_label=PL)
                    outsindx.append(PD)
            elif itype == 'continuous':
                IV = ivalue.to_numpy(dtype=float, na_value=np.nan)
                valid = np.isfinite(IV)
                sums = self.smoothed_probs(self.W, np.where(valid, IV, 0.0))
                mass = self.smoothed_probs(self.W, valid.astype(float))
                P = np.full(len(sums), np.nan)
                np.divide(sums, mass, out=P, where=mass > 0)
                outvalue.append(P)
                outscore.append(P)

                if predict_idx:
                    PD = self.predict_maxidx(np.where(valid, IV, 0.0))
                    outsindx.append(PD)
            else:
                raise ValueError('Unknown variable type')
        return (np.array(outvalue, dtype=object if 'discrete' in dtype else float).T,
                np.array(outscore).T, np.array(outsindx).T)


def to_nparray(*arrays, to_column=False):
    iarrs = []
    for iarr in arrays:
        if isinstance(iarr, (pd.Series, pd.DataFrame)):
            iarr = iarr.to_numpy()
        elif isinstance(iarr, (list)):
            iarr = np.array(iarr)

        if isinstance(iarr, np.ndarray):
            if to_column & (iarr.ndim == 1):
                iarr = iarr.reshape(-1,1)
            iarr = iarr
        else:
            raise ValueError('values must be a numpy array or pandas Series or DataFrame or list')

        iarrs.append(iarr)
    return iarrs
            

def to_narray(values, points, keys = None):
    if isinstance(values, np.ndarray):
        if (values.ndim == 1) and (values.shape[0]==points.shape[0]):
            values = values.reshape(-1,1)
        else:
            assert (values.ndim == 2) and (values.shape[0]==points.shape[0])
        keys_ = [ f'group_{i}'for i in range(values.shape[1]) ]
    elif isinstance(values, pd.Series):
        keys_= [values.name]
        values = np.array(values).reshape(-1,1)
    elif isinstance(values, pd.DataFrame):
        keys_ = values.columns.tolist()
        values = values.values
    else:
        raise ValueError('values must be a numpy array or pandas Series or DataFrame')
    if keys is None:
        keys = keys_
    else:
        assert len(keys) == values.shape[1]
    return values, keys

def gaussian_density(points, xy_grid=None, normal=True):
    from scipy.stats import gaussian_kde
    kernel = gaussian_kde(points.T)
    if xy_grid is None:
        values = points
    else:
        values = xy_grid
    z = kernel(values.T)

    if normal:
        min_z = np.min(z)
        max_z = np.max(z)

        # Scale between 0 and 1
        z = (z - min_z) / (max_z - min_z)
    return z

def gaussian_intp(points, xy_grid, value=None):
    from scipy.interpolate import RBFInterpolator
    if value is None:
        values = np.ones((points.shape[0]))
    else:
        values = np.asarray(value)
    kernel = RBFInterpolator(points, values)
    z = kernel(xy_grid)
    return z

def counts_ratio(x, y, xthred=10, ythred=10, EPS=1, state_method ='raw',):
    x = np.array(x)
    Y = np.array(y)
    idx0 = (y<=ythred) & (x<=xthred)
    X = x.copy()
    Y = y.copy()

    X[idx0] = 0
    Y[idx0] = 0
    if state_method == 'raw':
        ifc = (X+EPS)/(Y+EPS)
        ifc[idx0] = 0
        return  ifc

    elif state_method == 'log2fc':
        return np.log2(X+EPS) - np.log2(Y+EPS)

    elif state_method == 'inversal':
        ifc = (X+EPS)/(Y+EPS) - 1
        inv = (Y+EPS)/(X+EPS) - 1
        ifc[ifc<0] = -inv[ifc<0]
        return  ifc

    elif state_method == 'log2raw':
        ifc = (X+EPS)/(Y+EPS)
        ifc[idx0] = 0
        return  np.log2(ifc+EPS)
