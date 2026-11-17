import scipy as sci
import numpy as np
import pandas as pd
import pickle

from ..utilis._arrays import list_iter, vartype
from ..alignment.m3lib.xmm import kernel_xmm_k
from ._interp_shape import interpolate_slices, apply_slice_z
from joblib import Parallel, delayed


class SliceInterp:
    """Build target coordinates once and interpolate new values repeatedly.

    Values follow the row order of points; pandas column names are preserved.
    method='sdf' reuses the shape grid and requires return_info support in
    _interp_shape.py. prob/vtk/kdt/sci retain their original value backends.
    Automatic target slice counts are strictly less than max_slice.
    record=True stores each value group in values and values_itp; a repeated
    group replaces its previous record. get_interps() combines all records.
    SDF options: mapping='grid'/'centroid', label_mode='prob' (default)/'sdf'.
    SDF numeric predictions stay within the finite observed channel range;
    this does not guarantee preservation of extrema or variance.
    Only points, points_itp, values, values_itp and reusable interpolation
    geometry are retained. Temporary solver caches are released after each
    interp_value call; merged outputs and diagnostics are not recorded.
    """

    def __init__(self, points, method="sdf", n_points=21,
                 kernel="linear", n_slices=None, max_slice=500,
                 shape_method="pspline", shape_options=None, z_new=None,
                 keep_observed=True, mapping="grid", label_mode="prob", values=None,
                 anisotropy=0.0, level_scale=None, solver="auto",
                 factor_cache=4):
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
        self.anisotropy, self.level_scale = float(anisotropy), level_scale
        self.solver, self.factor_cache = solver, int(factor_cache)
        self.method = method
        self.n_points = n_points
        self.kernel = kernel
        self.n_slices = n_slices
        self.max_slice = int(max_slice)
        self.shape_method = shape_method
        self.shape_options = dict(shape_options or {})
        if set(self.shape_options) & {"points", "z_obs", "z_new", "method", "return_info"}:
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
            from inspect import signature

            self.get_nslice()
            self._hi = np.minimum(np.searchsorted(self.z_obs, self.z_new, side="right"), len(self.z_obs) - 1)
            self._lo = np.maximum(0, self._hi - 1)
            Zs = [self.points[r, :2] for r in self._rows]
            if "return_info" in signature(interpolate_slices).parameters:
                sampled, self.shape_info = interpolate_slices(
                    Zs, self.z_obs, self.z_new, method=self.shape_method,
                    return_info=True, **self.shape_options,
                )
            else:
                sampled = interpolate_slices(
                    Zs, self.z_obs, self.z_new, method=self.shape_method,
                    **self.shape_options,
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
            # After geometry exists, value interpolation uses z_operator's weights.
            self.shape_options.pop("reliability", None)
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
                "z_operator", "z_weights", "centroid_safe", "centers_obs", "centers_new",
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
                solver_stats=dict(lu=0, cg=0, reused=0, patterns=0, max_residual=0.0),
            )
        return self._cache[k]

    def _complete(self, values, k, value_smooth=0.0, zero_weight=1.0):
        """Complete grouped channels with bounded LU caching or preconditioned CG.

        value_smooth=0 retains hard finite observations. Positive smoothing
        uses (diag(mass) + value_smooth*dxy**2*L)u = sums; zero_weight only
        changes finite zero observations in this optional soft-fidelity solve.
        """
        from collections import OrderedDict
        from hashlib import blake2b
        from scipy.sparse import diags
        from scipy.sparse.linalg import cg, splu
        from inspect import signature

        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[0] != len(self._rows[k]):
            raise ValueError("Completion values must have one row per source-slice point.")
        if not np.isfinite(value_smooth) or value_smooth < 0:
            raise ValueError("value_smooth must be finite and nonnegative.")
        if not np.isfinite(zero_weight) or not 0 < zero_weight <= 1:
            raise ValueError("zero_weight must lie in (0, 1].")
        if value_smooth == 0 and zero_weight != 1:
            raise ValueError("zero_weight requires value_smooth > 0.")
        g = self._geometry(k)
        S, L = g["S"], g["L"]
        finite = np.isfinite(values)
        reliability = finite.astype(float)
        if value_smooth > 0 and zero_weight != 1:
            reliability[finite & (values == 0)] = zero_weight
        mass = S @ reliability
        sums = S @ (np.where(finite, values, 0.0) * reliability)
        fields = np.full(sums.shape, np.nan)
        if value_smooth == 0:
            np.divide(sums, mass, out=fields, where=mass > 0)
        groups = {}
        for j in range(values.shape[1]):
            if value_smooth == 0:
                pattern = np.packbits(mass[:, j] > 0).tobytes()
            else:
                pattern = np.ascontiguousarray(mass[:, j]).tobytes()
            key = (float(value_smooth), blake2b(pattern, digest_size=16).digest())
            groups.setdefault(key, []).append(j)
        factors = g["factors"]
        if not isinstance(factors, OrderedDict):
            factors = OrderedDict(factors)
            g["factors"] = factors
        stats = g.setdefault("solver_stats", dict(lu=0, cg=0, reused=0, patterns=0, max_residual=0.0))
        seen = g.setdefault("pattern_keys", set())
        for key in groups:
            if len(seen) <= 8:
                seen.add(key)
        stats["patterns"] = len(groups)
        solver = getattr(self, "solver", "auto")
        cache_size = getattr(self, "factor_cache", 4)
        for key, columns in groups.items():
            j = columns[0]
            known = mass[:, j] > 0
            supported = np.isin(g["components"], np.unique(g["components"][known]))
            if value_smooth == 0:
                unknown = supported & ~known
                if unknown.any():
                    A = L[unknown][:, unknown].tocsc()
                    rhs = -(L[unknown][:, known] @ fields[np.ix_(known, columns)])
                else:
                    A, rhs = None, None
            else:
                unknown = supported
                if unknown.any():
                    A = (diags(mass[unknown, j]) + value_smooth * self.shape_info["dxy"] ** 2 * L[unknown][:, unknown]).tocsc()
                    rhs = sums[np.ix_(unknown, columns)]
                else:
                    A, rhs = None, None
            if A is not None:
                use_cg = solver == "cg" or (solver == "auto" and len(seen) > 8 and key not in factors)
                if use_cg:
                    M = diags(1.0 / A.diagonal())
                    solution = np.empty_like(rhs)
                    options = dict(maxiter=max(1000, min(10000, 10 * A.shape[0])))
                    if "rtol" in signature(cg).parameters:
                        options.update(rtol=1e-8, atol=0.0)
                    else:
                        options.update(tol=1e-8, atol=0.0)
                    for q in range(rhs.shape[1]):
                        solution[:, q], status = cg(A, rhs[:, q], M=M, **options)
                        if status != 0:
                            raise RuntimeError(f"CG completion failed on slice {k}, channel {columns[q]} (status={status}).")
                    stats["cg"] += len(columns)
                else:
                    if key in factors:
                        factor = factors.pop(key)
                        stats["reused"] += 1
                    else:
                        factor = splu(A)
                        stats["lu"] += 1
                    solution = factor.solve(np.asarray(rhs))
                    if cache_size > 0:
                        factors[key] = factor
                        while len(factors) > cache_size:
                            factors.popitem(last=False)
                residual = np.linalg.norm(A @ solution - rhs, axis=0) / np.maximum(np.linalg.norm(rhs, axis=0), 1.0)
                stats["max_residual"] = max(stats["max_residual"], float(residual.max(initial=0.0)))
                fields[np.ix_(unknown, columns)] = solution
        for j in range(values.shape[1]):
            if finite[:, j].any():
                fields[:, j] = np.clip(fields[:, j], values[finite[:, j], j].min(),
                                       values[finite[:, j], j].max())
        return fields

    def _sample(self, fields, k, t, mapping, signed=False):
        """Sample with distance support in both numerator and denominator."""
        from scipy.ndimage import distance_transform_edt
        import warnings

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
        if key not in g["samples"]:
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
            g["samples"][key] = samples
            # Bound source/target corner arrays as well as sparse LU factors.
            while len(g["samples"]) > 4:
                del g["samples"][next(iter(g["samples"]))]
        sums = np.zeros((self.slice_offsets[t + 1] - self.slice_offsets[t], fields.shape[1]))
        mass = np.zeros_like(sums)
        conf = np.zeros_like(sums)
        for ids, distance, weight in g["samples"][key]:
            f = fields[ids]
            if signed:
                # Class signed distances are negative outside the source domain.
                f = np.where(distance[:, None] > 0, np.minimum(f, -distance[:, None]), f)
            valid = np.isfinite(f)
            sums += np.where(valid, f, 0.0) * weight[:, None]
            mass += valid * weight[:, None]
            conf += valid * weight[:, None]
        out = np.full_like(sums, np.nan)
        np.divide(sums, mass, out=out, where=mass > 0)
        return out, np.clip(conf, 0, 1)

    def _transfer(self, fields, mapping, signed=False):
        """Apply the shape's actual z operator to spatially sampled fields."""
        result = np.full((len(self.points_itp), fields[0].shape[1]), np.nan)
        confidence = np.zeros_like(result)
        info = self.shape_info
        options = info.get("z_operator", dict(method=self.shape_method)).copy()
        weights = info.get("z_weights")
        reliability = options.get("reliability")
        spatial = reliability is not None and np.asarray(reliability).ndim > 1
        finite_fields = all(np.isfinite(f).all() for f in fields)
        for t in range(len(self.z_new)):
            rows = slice(self.slice_offsets[t], self.slice_offsets[t + 1])
            if rows.start != rows.stop:
                if weights is not None and finite_fields:
                    active = np.flatnonzero(weights[t] != 0)
                else:
                    active = np.arange(len(fields))
                samples = [self._sample(fields[k], k, t, mapping, signed=signed) for k in active]
                values = np.stack([v for v, c in samples])
                supports = np.stack([c for v, c in samples])
                if spatial:
                    R = np.broadcast_to(np.asarray(reliability), info["masks_obs"].shape)
                    options["reliability"] = np.stack([
                        self._sample(R[k][info["masks_obs"][k]][:, None], k, t, mapping)[0]
                        for k in range(len(fields))
                    ])
                if weights is not None and np.isfinite(values).all():
                    out = np.tensordot(weights[t, active], values, axes=(0, 0))
                else:
                    if len(active) != len(fields):
                        values = np.stack([self._sample(f, k, t, mapping, signed=signed)[0]
                                           for k, f in enumerate(fields)])
                    out = apply_slice_z(values, self.z_obs, self.z_new[t:t + 1], **options)[0]
                if weights is not None:
                    w = np.abs(weights[t, active])
                    total = w.sum()
                    c = np.tensordot(w / total, supports, axes=(0, 0)) if total > 0 else np.zeros_like(out)
                else:
                    if options["method"] == "pchip":
                        lo = max(0, self._lo[t] - 1)
                        hi = min(len(fields), self._hi[t] + 2)
                        c = supports[lo:hi].min(0)
                    else:
                        # Spatial spline weights have no single global matrix.
                        c = supports.min(0)
                result[rows] = out
                confidence[rows] = np.where(np.isfinite(out), np.clip(c, 0, 1), 0)
        return result, confidence

    def interp_value(self, values, method=None, n_points=None, kernel=None,
                     record=True, value_type="auto", label_mode=None, mapping=None,
                     value_smooth=0.0, zero_weight=1.0, label_temperature=1.0,
                     value_batch=64, value_clip=True, **kargs):
        """Return xyz + values using cached geometry and recorded value groups.

        Numeric columns are continuous in auto mode; integer labels require
        categorical dtype or value_type='categorical'. value_smooth=0 retains
        hard observations; positive values use soft fidelity on the grid.
        zero_weight in (0,1] changes zero-observation reliability only in soft
        mode, and must be chosen under an explicit missing-detection assumption.
        value_batch bounds simultaneous numeric channels. value_clip bounds
        final numeric predictions after the shared z operator; no rescaling or
        distribution calibration is applied. label_temperature scales SDF
        softmax in units of observed XY spacing. The returned xyz + values
        table is not retained separately; get_interps() combines all records.
        """
        from scipy.ndimage import distance_transform_edt
        from scipy.special import softmax
        from pandas.api.types import is_numeric_dtype, is_bool_dtype

        if isinstance(values, pd.Series):
            source = values.to_frame(name=values.name if values.name is not None else "group_0")
        elif isinstance(values, pd.DataFrame):
            source = values.copy()
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
                    fields = [self._complete(v[r], k, value_smooth=value_smooth, zero_weight=zero_weight)
                              for k, r in enumerate(self._rows)]
                    g, _ = self._transfer(fields, mapping)
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
                            onehot = np.eye(len(classes))[np.maximum(codes, 0)]
                            onehot[~valid] = np.nan
                            fields = [self._complete(onehot[r], k) for k, r in enumerate(self._rows)]
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
                                            diameter = np.hypot(np.ptp(ix) * self.shape_info["dx"], np.ptp(iy) * self.shape_info["dy"])
                                            field = np.full(mask.shape, -max(diameter, self.shape_info["dxy"]))
                                        sdf[ok, j] = field[mask][ok]
                                    fields[k] = sdf
                            g, _ = self._transfer(fields, mapping, signed=label_mode == "sdf")
                            ok = np.isfinite(g).all(1)
                            p = np.full_like(g, np.nan)
                            if label_mode == "sdf":
                                p[ok] = softmax(g[ok] / (label_temperature * self.shape_info["dxy"]), axis=1)
                            else:
                                p[ok] = np.clip(g[ok], 0, 1)
                                mass = p[ok].sum(1)
                                rows = np.flatnonzero(ok)
                                good = mass > 0
                                p[rows[good]] /= mass[good, None]
                                p[rows[~good]] = np.nan
                                ok[rows[~good]] = False
                            labels = np.full(len(g), np.nan, dtype=object)
                            labels[ok] = classes.to_numpy()[p[ok].argmax(1)]
                            out[col] = labels
                out = pd.DataFrame(out, index=np.arange(len(self.points_itp)), columns=source.columns)
            elif method in ("prob", "vtk", "kdt", "sci"):
                if value_smooth != 0 or zero_weight != 1:
                    raise ValueError("value_smooth/zero_weight apply only to method='sdf'.")
                keys = [f"group_{i}" for i in range(len(source.columns))]
                out = interp_value(self.points, source, self.points_itp, method=method, keys=keys,
                                   n_points=n_points, kernel=kernel, **kargs)
                out = pd.DataFrame(out).reset_index(drop=True)
                out.columns = source.columns
            else:
                raise ValueError("method must be sdf/prob/vtk/kdt/sci.")
            if self.keep_observed:
                for t, z in enumerate(self.z_new):
                    k = np.searchsorted(self.z_obs, z)
                    if k < len(self.z_obs) and self.z_obs[k] == z:
                        a, b = self.slice_offsets[t:t + 2]
                        out.iloc[a:b] = source.iloc[self._rows[k]].to_numpy()
            if record:
                key = '.'.join(map(str, out.columns))
                self.values[key] = values
                self.values_itp[key] = out
            return pd.concat([pd.DataFrame(self.points_itp, columns=list("xyz")), out], axis=1)
        finally:
            self._cache.clear()

    def get_interps(self):
        """Return xyz and all recorded value groups, aligned by row position."""
        if self.points_itp is not None and len(self.values_itp):
            coords = pd.DataFrame(self.points_itp, columns=list("xyz"))
            values = pd.concat(list(self.values_itp.values()), axis=1)
            interps = pd.concat([coords, values], axis=1)
        else:
            interps = None
        return interps

    def __getstate__(self):
        """Save results and required geometry; derived indices are rebuilt on load."""
        state = {key: self.__dict__[key] for key in (
            "points", "points_itp", "values", "values_itp", "shape_info", "z_new",
            "method", "n_points", "kernel", "n_slices", "max_slice",
            "shape_method", "shape_options", "keep_observed", "mapping", "label_mode",
            "anisotropy", "level_scale", "solver", "factor_cache",
        ) if key in self.__dict__}
        state["shape_info"] = self._shape_state(state.get("shape_info"))
        if state.get("points_itp") is not None:
            state["shape_options"] = dict(state.get("shape_options", {}))
            state["shape_options"].pop("reliability", None)
        return state

    def __setstate__(self, state):
        """Restore minimal state, also discarding caches from older models."""
        self.__dict__.update(state)
        state = self.__getstate__()
        self.__dict__.clear()
        self.__dict__.update(state)
        self.z_obs = np.unique(self.points[:, 2])
        self._rows = [np.flatnonzero(self.points[:, 2] == z) for z in self.z_obs]
        self._cache = {}
        if self.points_itp is not None:
            self._hi = np.minimum(np.searchsorted(self.z_obs, self.z_new, side="right"), len(self.z_obs) - 1)
            self._lo = np.maximum(0, self._hi - 1)
            self.slice_offsets = np.r_[np.searchsorted(self.points_itp[:, 2], self.z_new), len(self.points_itp)]

    def save(self, path):
        """Save only reusable geometry and records, without temporary caches."""
        with open(path, "wb") as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

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
    values, keys = to_narray(values, points, keys=keys)
    points, points_intp = to_nparray(points, points_intp)

    dtype = np.array([ vartype(values[:,i]) for i in range(len(keys)) ])
    outs = {}
    dis_idx = dtype == 'discrete'
    con_idx = dtype == 'continuous'

    if method == 'prob':
        pitp = prob_interpolation(points, points_intp, n_points=n_points, method=knn_method,
                                  radius=radius, temp=temp)
        out = pitp(values, threshold=threshold)[0]
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
        if radius is None and not n_points:
            raise ValueError("Please set either radius or n_points")
        elif n_points:
            kern.SetNumberOfPoints(n_points)
            kern.SetKernelFootprintToNClosest()
        else:
            kern.SetRadius(radius)

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
        if order is None:
            try:
                order = X.cat.categories
            except:
                order = np.unique(X)
        if null_value is not None:
            order = np.r_[order, [null_value]]
        assert len(order) >= np.unique(X).shape[0]

        self.order = np.array(order)
        self.order_dict = {k: i for i, k in enumerate(order)}
        self.num_classes = len(order)

        return self.transform(X)

    def transform(self, X):
        X_id = np.array([self.order_dict[k] for k in X])
        src = np.arange(len(X))
        dst = X_id
        dist = np.ones(len(X))
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
        if threshold > 0:
            prob = prob.toarray().copy()
            prob[:,-1] = threshold
            score = prob.max(1)
        else:
            score = prob.max(1).toarray().flatten()
        return np.asarray(prob.argmax(1)).ravel(), np.asarray(score).ravel()

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
            src = np.concatenate(incs, axis=0)
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
        values = np.array(values)
        if values.ndim == 1:
            values = values.reshape(-1,1)
        dtype = np.array([ vartype(values[:,i]) for i in range(values.shape[1]) ])

        outvalue = []
        outscore = []
        outsindx = []
        if ('discrete' in dtype) and predict_idx:
            self.get_W_dict()

        for i, itype in enumerate(dtype):
            ivalue = values[:,i]
            if itype == 'discrete':
                OH = onehot()
                IV = OH.fit(ivalue, null_value=null_value)
                P = self.smoothed_probs(self.W, IV)
                PI, PV = self.predict_labels(P, threshold=threshold)
                PL = OH.inverse_transform(PI)
                outvalue.append(PL)
                outscore.append(PV)
                if predict_idx:
                    PD = self.predict_maxidx(ivalue, prob_label=PL)
                    outsindx.append(PD)
            elif itype == 'continuous':
                IV = ivalue
                P = self.smoothed_probs(self.W, IV)
                outvalue.append(P)
                outscore.append(P)

                if predict_idx:
                    PD = self.predict_maxidx(ivalue)
                    outsindx.append(PD)
            else:
                raise ValueError('Unknown variable type')
        return np.array(outvalue).T, np.array(outscore).T, np.array(outsindx).T


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