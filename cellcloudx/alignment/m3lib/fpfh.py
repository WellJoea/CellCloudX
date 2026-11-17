"""Classical point-cloud/image descriptors with optional masks and stain channels.

FPFH / FPFH_batch: arbitrary-dimensional geometry, radial and attribute features.
Only optional normal angles need 2D/3D coordinates; full-mask profiles support nD.
image_features / image_features_batch: multiscale Gaussian structure in nD fields.

Batch each_args is {parameter: [sample_0_value, sample_1_value, ...]}.
Lists follow input order; non-list values broadcast. Batch n_jobs defaults to 5.
Hard string labels and soft membership matrices are accepted directly.

return_info=False (default) skips metadata. Single-sample info has blocks and
valid only; batch info adds reusable shared_parameters. Point valid is (N,B),
ordered like the B block slices. Image valid marks observed pixels/sampled rows.
"""

import os
from collections.abc import Mapping
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

try:
    from numba import njit, prange
    # Aliased imports must not write module-name-dependent Numba disk caches.
    _CACHE = __name__.rsplit(".", 1)[-1] == os.path.splitext(os.path.basename(__file__))[0]
except ImportError:
    prange, _CACHE = range, False

    def njit(**options):
        return lambda function: function


# ---------------------------------------------------------------- primitives

def _label_input(label, count, name="label", normalize=True):
    """Hard categories (N,) or row-normalised soft categories (N,C)."""
    if hasattr(label, "detach"):
        label = label.detach().cpu().numpy()
    if hasattr(label, "toarray"):
        label = label.toarray()
    label = np.asarray(label)
    if label.ndim not in (1, 2) or label.shape[0] != count:
        raise ValueError(f"{name} must have shape (N,) or (N,C), N={count}.")
    if label.ndim == 1 or not normalize:
        return label
    probability = np.asarray(label, dtype=float)
    observed = np.isfinite(probability).all(1) & (probability >= 0).all(1)
    probability = np.where(observed[:, None], probability, 0.0)
    mass = probability.sum(1, keepdims=True)
    probability /= np.where(mass > 0, mass, 1.0)
    probability[mass[:, 0] <= 0] = np.nan
    return probability

def _label_observed(label):
    if label.ndim == 2:
        return np.isfinite(label).all(1) & (label.sum(1) > 0)
    if label.dtype.kind in "biufc":
        return np.isfinite(label)
    if label.dtype.kind in "US":
        return np.ones(len(label), dtype=bool)

    def observed(value):
        try:
            return value is not None and bool(value == value)
        except (TypeError, ValueError):
            return False
    return np.fromiter((observed(v) for v in label), dtype=bool, count=len(label))

def _label_mode(label, mode="auto"):
    if label.ndim == 2:
        if mode == "binary" and label.shape[1] != 2:
            raise ValueError("Binary soft labels need [background, foreground] columns.")
        return mode == "binary"
    if mode != "auto":
        return mode == "binary"
    return (label.dtype.kind in "biufc"
            and np.isin(label[_label_observed(label)], (0, 1, 255)).all())

def _label_informative(label, binary):
    value = label[_label_observed(label)]
    if binary:
        value = value[:, 1] if label.ndim == 2 else value != 0
    return bool(len(value) and np.any(value != value[0]))

def _label_histogram(label, observed, binary, safe, radial, weights, bins, eps,
                     return_info=False):
    """Same/different radial mass; soft similarity preserves category uncertainty."""
    lw = weights * observed[safe]
    lw /= np.maximum(lw.sum(1, keepdims=True), eps)
    value = (np.where(observed[:, None], label, 0.0) if label.ndim == 2 else
             label if label.dtype.kind in "US" else np.where(observed, label, 0))
    if binary:
        selected = value[safe, 1] if label.ndim == 2 else value[safe] != 0
    else:
        lw *= observed[:, None]
        if label.ndim == 1:
            selected = value[safe] == value[:, None]
        else:
            # Avoid materialising the (N,K,C) neighbour probability tensor.
            selected = np.empty(safe.shape)
            for k in range(safe.shape[1]):
                selected[:, k] = np.einsum("nc,nc->n", value, value[safe[:, k]])
            np.clip(selected, 0.0, 1.0, out=selected)
    hist = np.concatenate([_histogram(radial, lw * (1.0 - selected), bins),
                           _histogram(radial, lw * selected, bins)], axis=1)
    return hist, (lw.sum(1) > 0) if return_info else None


def _feature_array(feature, count, name="feature"):
    if feature is None:
        return None
    feature = np.asarray(feature, dtype=float)
    feature = feature[:, None] if feature.ndim == 1 else feature
    if feature.ndim != 2 or feature.shape[0] != count or feature.shape[1] == 0:
        raise ValueError(f"{name} must have shape (N,) or (N,C).")
    return feature


def _scales(values, name):
    values = np.atleast_1d(np.asarray(values, dtype=float))
    if len({f"{value:g}" for value in values}) != len(values):
        raise ValueError(f"{name} must have distinct scale names.")
    return values


def _binary_mask(mask):
    return np.asarray(mask, dtype=bool)


def _histogram(values, weights, bins):
    """Soft histogram of values in [0, 1], split linearly over two bins."""
    scaled = np.clip(np.nan_to_num(values, nan=0.0), 0.0, 1.0) * (bins - 1)
    lower = scaled.astype(np.intp)
    upper = np.minimum(lower + 1, bins - 1)
    fraction = scaled - lower
    row = np.arange(scaled.shape[0], dtype=np.intp)[:, None] * bins
    out = np.bincount((row + lower).ravel(), (weights * (1.0 - fraction)).ravel(),
                      minlength=scaled.shape[0] * bins)
    out += np.bincount((row + upper).ravel(), (weights * fraction).ravel(),
                       minlength=out.size)
    return out.reshape(scaled.shape[0], bins)


def _joint_histogram(radius, contrast, weights, bins):
    """Soft joint histogram over two axes, both rescaled to [0, 1]."""
    rad = np.clip(radius, 0.0, 1.0) * (bins - 1)
    con = np.clip(np.nan_to_num(contrast, nan=0.5), 0.0, 1.0) * (bins - 1)
    rb, cb = rad.astype(np.intp), con.astype(np.intp)
    rf, cf = rad - rb, con - cb
    base = np.arange(rad.shape[0], dtype=np.intp)[:, None] * (bins * bins)
    out = np.zeros(rad.shape[0] * bins * bins)
    for ro, rw in ((0, 1.0 - rf), (1, rf)):
        rindex = np.minimum(rb + ro, bins - 1) * bins
        for co, cw in ((0, 1.0 - cf), (1, cf)):
            cindex = np.minimum(cb + co, bins - 1)
            out += np.bincount((base + rindex + cindex).ravel(),
                               (weights * rw * cw).ravel(), minlength=out.size)
    return out.reshape(rad.shape[0], bins * bins)


def get_neighbors(coordx, coordy=None, knn=11, radius=None,
                  max_neighbor=10000, kd_method="sknn", n_jobs=-1):
    """Euclidean neighbours; knn includes self and radius also limits support."""
    coordx = np.asarray(coordx, dtype=float)
    coordy = coordx if coordy is None else np.asarray(coordy, dtype=float)
    if not len(coordx) or not len(coordy):
        return [np.empty((len(coordy), 0)), np.empty((len(coordy), 0), dtype=int)]
    if kd_method not in ("sknn", "scipy", "ckdtree"):
        from ...tools._neighbors import Neighbors
        model = Neighbors(method=kd_method, metric="euclidean", n_jobs=n_jobs)
        model.fit(coordx, radius_max=None, max_neighbor=max_neighbor)
        return model.transform(coordy, knn=knn, radius=radius)
    tree = cKDTree(coordx)
    if knn is None:
        counts = tree.query_ball_point(coordy, radius, workers=n_jobs, return_length=True)
        knn = max(1, int(counts.max()))
    needed = min(knn, len(coordx))
    if needed > max_neighbor:
        warnings.warn("max_neighbor truncates the neighbourhood.", RuntimeWarning, stacklevel=2)
    knn = min(needed, max_neighbor)
    distance, index = tree.query(coordy, k=knn, workers=n_jobs)
    distance, index = distance.reshape(len(coordy), knn), index.reshape(len(coordy), knn)
    if radius is not None:
        keep = distance <= radius
        distance, index = np.where(keep, distance, np.inf), np.where(keep, index, -1)
    return [distance, index]


def calc_normals(pc, viewpoint=None, inds=None, knn=None, radius=None,
                 kd_method="sknn", orient="centroid", chunk=50000, eps=1e-12):
    """Unit normals from the smallest local covariance eigenvector.

    A neighbourhood is accepted only when the smallest eigenvalue is small and
    well separated, so volumes and lines get zero rather than an arbitrary
    direction. orient="centroid" points normals at the centroid (inward for a
    closed shell), "outward" away from it, "as_is" keeps the eigenvector sign.
    """
    if orient not in ("centroid", "outward", "as_is"):
        raise ValueError("orient must be centroid, outward or as_is.")
    pc = np.asarray(pc, dtype=float)
    if len(pc) == 0:
        return np.zeros_like(pc)
    if inds is None:
        _, inds = get_neighbors(pc, knn=11 if knn is None else knn,
                                radius=radius, kd_method=kd_method)
    inds = np.asarray(inds)
    if inds.shape[1] == 0:
        return np.zeros_like(pc)
    reference = pc.mean(axis=0) if viewpoint is None else np.asarray(viewpoint, dtype=float)
    if orient == "as_is":
        warnings.warn("orient='as_is' keeps an arbitrary eigenvector sign.",
                      RuntimeWarning, stacklevel=2)
    direction = reference - pc
    if orient == "outward":
        direction = -direction
    dim, tiny = pc.shape[1], np.finfo(float).tiny
    out = np.zeros_like(pc)
    for start in range(0, len(pc), max(1, int(chunk))):
        stop = min(start + max(1, int(chunk)), len(pc))
        rows = np.arange(start, stop)
        ids = np.asarray(inds[start:stop], dtype=np.intp)
        valid = (ids >= 0) & (ids < len(pc)) & (ids != rows[:, None])
        delta = pc[np.where(valid, ids, 0)] - pc[start:stop, None, :]
        keep = valid & (np.einsum("nkd,nkd->nk", delta, delta) > eps ** 2)
        weight = keep.astype(float)
        mass = np.maximum(weight.sum(axis=1), 1.0)
        mean = np.einsum("nk,nkd->nd", weight, delta) / mass[:, None]
        centered = (delta - mean[:, None, :]) * weight[:, :, None]
        values, vectors = np.linalg.eigh(
            np.einsum("nkd,nke->nde", centered, centered) / mass[:, None, None])
        total = values.sum(axis=1)
        accept = ((keep.sum(axis=1) >= dim) & (total > tiny)
                  & (values[:, 0] / np.where(total > tiny, total, 1.0) <= 0.1)
                  & ((values[:, 1] - values[:, 0]) / np.maximum(values[:, -1], tiny) >= 0.05))
        normal = vectors[:, :, 0]
        if orient != "as_is":
            flip = np.einsum("nd,nd->n", normal, direction[start:stop]) < 0
            normal = np.where(flip[:, None], -normal, normal)
        length = np.linalg.norm(normal, axis=1)
        out[start:stop] = np.where((accept & (length > max(eps, tiny)))[:, None],
                                   normal / np.where(length > 0, length, 1.0)[:, None], 0.0)
    return out


# ---------------------------------------------------------------- angular kernels

@njit(parallel=True, cache=_CACHE)
def compute_spfh_numba(points, normals, indices, dists, extent, spfh, n_bins, eps=1e-10):
    """Per-point angular histogram (SPFH).

    3D: u = n_i, v = u x d, w = u x v; alpha = v.n_j, phi = u.d,
        theta = atan2(w.n_j, u.n_j).
    2D: t = (-n_y, n_x); psi = angle(tangent -> chord), theta = normal turn,
        third channel = distance / extent.
    """
    for i in prange(points.shape[0]):
        if np.dot(normals[i], normals[i]) <= eps:
            continue
        total = 0
        for k in range(indices.shape[1]):
            j = indices[i, k]
            distance = dists[i, k]
            if j < 0 or j >= len(points) or j == i or not np.isfinite(distance) or distance <= eps:
                continue
            u, nt = normals[i], normals[j]
            if np.dot(nt, nt) <= eps:
                continue
            direction = (points[j] - points[i]) / distance
            if points.shape[1] == 3:
                phi = np.dot(u, direction)
                v = np.cross(u, direction)
                length = np.sqrt(np.dot(v, v))
                if length <= eps:
                    continue
                v = v / length
                w = np.cross(u, v)
                alpha = np.dot(v, nt)
                theta = np.arctan2(np.dot(w, nt), np.dot(u, nt))
                b0 = min(n_bins - 1, max(0, int((alpha + 1.0) * 0.5 * n_bins)))
                b1 = min(n_bins - 1, max(0, int((phi + 1.0) * 0.5 * n_bins)))
                b2 = min(n_bins - 1, max(0, int((theta + np.pi) / (2.0 * np.pi) * n_bins)))
            else:
                t0, t1 = -u[1], u[0]
                psi = np.arctan2(t0 * direction[1] - t1 * direction[0],
                                 t0 * direction[0] + t1 * direction[1])
                theta = np.arctan2(u[0] * nt[1] - u[1] * nt[0],
                                   u[0] * nt[0] + u[1] * nt[1])
                ratio = min(1.0, max(0.0, distance / extent[i] if extent[i] > eps else 0.0))
                b0 = min(n_bins - 1, max(0, int((psi + np.pi) / (2.0 * np.pi) * n_bins)))
                b1 = min(n_bins - 1, max(0, int((theta + np.pi) / (2.0 * np.pi) * n_bins)))
                b2 = min(n_bins - 1, max(0, int(ratio * (n_bins - 1))))
            spfh[i, b0] += 1
            spfh[i, n_bins + b1] += 1
            spfh[i, 2 * n_bins + b2] += 1
            total += 1
        if total:
            spfh[i] /= total


@njit(parallel=True, cache=_CACHE)
def compute_fpfh_numba(spfh, indices, dists, fpfh, eps=1e-10):
    """FPFH = L1-normalised( SPFH_i + inverse-distance mean of neighbour SPFH )."""
    for i in prange(spfh.shape[0]):
        weighted = np.zeros(spfh.shape[1])
        weight_sum = 0.0
        for k in range(indices.shape[1]):
            j = indices[i, k]
            distance = dists[i, k]
            if j < 0 or j >= len(spfh) or j == i or not np.isfinite(distance) or distance <= eps:
                continue
            if np.sum(spfh[j]) <= eps:
                continue
            weight = 1.0 / distance
            weighted += spfh[j] * weight
            weight_sum += weight
        if weight_sum > eps:
            fpfh[i] = spfh[i] + weighted / weight_sum
        else:
            fpfh[i] = spfh[i]
        total = np.sum(fpfh[i])
        if total > eps:
            fpfh[i] /= total


# ---------------------------------------------------------------- point blocks

def _scale_blocks(cfg, dists, indices, radius, prefix):
    """Compute one support; metadata is built only when requested."""
    points, bins, eps = cfg["points"], cfg["n_bins"], cfg["eps"]
    count, dim = points.shape
    valid = ((indices >= 0) & (indices < count) & np.isfinite(dists)
             & (dists > eps) & (indices != np.arange(count)[:, None]))
    safe = np.where(valid, indices, 0)
    distance = np.where(valid, dists, 0.0)
    extent = distance.max(1) if distance.shape[1] else np.zeros(count)
    extent = np.where(extent > eps, extent, 1.0)
    radial_scale = extent if radius is None else np.full(count, float(radius))
    scale = radial_scale
    if cfg["scale_mode"] == "rms":
        scale = np.sqrt((distance ** 2).sum(1) / valid.sum(1).clip(min=1))
        scale = np.where(scale > eps, scale, radial_scale)
    elif cfg["scale_mode"] == "median":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            scale = np.nanmedian(np.where(valid, distance, np.nan), axis=1)
        scale = np.where(np.isfinite(scale) & (scale > eps), scale, radial_scale)
    radial = distance / radial_scale[:, None]
    weights = np.exp(-0.5 * radial ** 2) * valid
    weights /= np.maximum(weights.sum(1, keepdims=True), eps)
    delta = np.where(valid[:, :, None],
                     (points[safe] - points[:, None, :]) / scale[:, None, None], 0.0)
    mean = np.einsum("nk,nkd->nd", weights, delta)
    centered = delta - mean[:, None, :]
    eig = np.maximum(np.linalg.eigvalsh(
        np.einsum("nk,nkd,nke->nde", weights, centered, centered)), 0.0)[:, ::-1]
    trace = eig.sum(1)
    parts, blocks, flags, offset = [], {}, [], 0
    want_info = cfg["return_info"]
    geometry_valid = valid.any(1) if want_info else None

    def add(name, values, supported):
        nonlocal offset
        parts.append(values.astype(np.float32, copy=False))
        if want_info:
            blocks[name + prefix] = slice(offset, offset + values.shape[1])
            flags.append(supported)
        offset += values.shape[1]

    add("geometry", np.column_stack((eig / np.maximum(trace[:, None], eps), trace,
                                     np.linalg.norm(mean, axis=1))), geometry_valid)
    add("radial", np.sqrt(_histogram(radial, weights, bins)), geometry_valid)
    if cfg["mask_data"] is not None:
        answer = _mask_shape_features(cfg["mask_data"], points, radius, n_bins=bins,
                                      eps=eps, prepared=cfg["mask_data"], return_info=want_info)
        if want_info:
            values, metadata = answer
            for k, (name, section) in enumerate(metadata["blocks"].items()):
                add(name, values[:, section], metadata["valid"][:, k])
        else:
            parts.append(answer.astype(np.float32, copy=False))
            offset += answer.shape[1]
    normals = cfg["normals"]
    if normals is not None:
        kept = np.where(valid, indices, -1)
        spfh = np.zeros((count, 3 * bins), dtype=np.float32)
        compute_spfh_numba(points, normals, kept, dists, radial_scale, spfh, bins, eps)
        angular = np.zeros_like(spfh)
        compute_fpfh_numba(spfh, kept, dists, angular, eps)
        angular[~cfg["normal_valid"]] = 0
        if cfg["normal_gate"]:
            gap = eig[:, 1] - eig[:, 2] if dim == 3 else eig[:, 0] - eig[:, 1]
            angular *= np.clip(gap / np.maximum(eig[:, 0], eps), 0.0, 1.0)[:, None]
        add("normal", angular, (angular.sum(1) > 0) if want_info else None)
    label = cfg["label"]
    if label is not None:
        hist, supported = _label_histogram(label, cfg["label_finite"], cfg["label_binary"],
                                           safe, radial, weights, bins, eps, want_info)
        add("label", np.sqrt(hist) * cfg["label_informative"],
            supported & cfg["label_informative"] if want_info else None)
    feature = cfg["feature"]
    if feature is not None:
        b = cfg["feature_bins"]
        for channel in range(feature.shape[1]):
            center = feature[:, channel]
            value = feature[safe, channel]
            fw = weights * np.isfinite(value)
            mass = fw.sum(1)
            fw /= np.maximum(mass[:, None], eps)
            value = np.where(np.isfinite(value), value, 0.0)
            mu = np.sum(fw * value, axis=1)
            std = np.sqrt(np.maximum(np.sum(fw * (value - mu[:, None]) ** 2, axis=1), 0.0))
            observed = np.isfinite(center)
            filled = np.where(observed, center, mu)
            contrast = (value - filled[:, None]) / np.maximum(std[:, None], eps)
            hist = _joint_histogram(radial, (np.clip(contrast, -3, 3) + 3) / 6, fw, b)
            if cfg["hist_prior"] > 0:
                seen = mass > eps
                neff = 1.0 / np.sum(fw[seen] ** 2, axis=1)
                hist[seen] = ((neff[:, None] * hist[seen] + cfg["hist_prior"] / b ** 2)
                              / (neff[:, None] + cfg["hist_prior"]))
            contrast0 = (filled - mu) / np.maximum(std, eps)
            contrast0 /= np.hypot(1, contrast0)
            amplitude = std / np.maximum(np.sqrt(np.sum(fw * value ** 2, axis=1)), eps)
            stats = np.column_stack((contrast0, amplitude))
            stats[mass <= eps] = 0
            supported = observed & (mass > eps) if want_info else None
            add(f"feature_{channel}", np.column_stack((stats, np.sqrt(hist))), supported)
            if cfg["include_intensity"]:
                add(f"intensity_{channel}", np.column_stack((filled, mu)) / cfg["reference"][channel],
                    supported)
    return parts, blocks, flags


def FPFH(points, normals=None, knn=31, radius=None, n_bins=11, viewpoint=None,
         kd_method="sknn", eps=1e-10, *, label=None, feature=None, normal=None,
         return_info=False, label_mode="auto", scale_mode="max", normal_knn=None,
         normal_radius=None, orient="centroid", radii=None, feature_scale=None,
         include_intensity=True, mask=None, mask_options=None, feature_bins=5,
         hist_prior=0.0, normal_gate=True):
    """Local descriptors for (N,d) physical coordinates; d is unrestricted.

    knn/radius/radii   Neighbour cap (31), fixed support, or multiple supports.
                      knn includes self; None uses ball neighbours. Radii are physical.
    label/label_mode  Optional (N,) string/numeric categories or (N,C) soft labels.
                      auto: numeric masks use binary, strings/soft labels use region.
                      Region uses equality or probability dot products; no argmax.
    feature           Optional (N,) or (N,C) grey/stain values; NaN means missing.
    feature_scale     Shared C-channel reference for absolute intensity only; None=raw.
    include_intensity Keep absolute centre/mean (True). feature_bins=5 sets joint bins;
                      None follows n_bins. hist_prior=0 disables uniform shrinkage.
    normal/normals    Optional (N,d) normals or 'auto'; None skips angles. Use one name.
                      Only this branch needs d=2/3. normal_gate=True weights reliability.
    normal_knn,
    normal_radius     Auto-normal support; defaults to knn (or 11) and single radius.
    orient/viewpoint  centroid, outward or as_is; reference defaults to cloud centroid.
    mask/mask_options Full nD binary raster and spacing/origin/smoothing_sigma.
                      Needs fixed radius. Spacing uses array axes, origin physical axes.
    n_bins/scale_mode Histogram bins (11); geometry scale max, rms or median.
    kd_method/eps     Neighbour backend ('sknn') and numerical distance tolerance (1e-10).
    return_info       False returns (N,F) float32 only. True adds {blocks, valid}:
                      valid has shape (N, number_of_blocks), ordered like blocks.
    """
    points = np.asarray(points, dtype=float)
    count, dim = points.shape
    if radius is not None and radii is not None:
        raise ValueError("Use radius or radii, not both.")
    if normal is not None and normals is not None:
        raise ValueError("Use normal or normals, not both.")
    if label_mode not in ("auto", "binary", "region") or scale_mode not in ("max", "rms", "median"):
        raise ValueError("Unknown label_mode or scale_mode.")
    if not np.isfinite(hist_prior) or hist_prior < 0:
        raise ValueError("hist_prior must be finite and nonnegative.")
    supports = [radius] if radii is None else _scales(radii, "radii").tolist()
    normals = normal if normal is not None else normals
    if normals is not None:
        if dim not in (2, 3):
            raise ValueError("Normal-angle descriptors require 2D/3D coordinates.")
        if isinstance(normals, str):
            if normals != "auto":
                raise ValueError("normal must be an array, 'auto' or None.")
            normals = calc_normals(points, viewpoint=viewpoint,
                knn=knn if normal_knn is None else normal_knn,
                radius=radius if normal_radius is None else normal_radius,
                kd_method=kd_method, orient=orient, eps=eps)
        normals = np.asarray(normals, dtype=float)
        if normals.shape != points.shape:
            raise ValueError("normal must match points.")
        lengths = np.linalg.norm(normals, axis=1)
        normal_valid = np.isfinite(lengths) & (lengths > eps)
        normals = np.where(normal_valid[:, None], normals, 0.0)
        normals /= np.where(normal_valid, lengths, 1.0)[:, None]
    else:
        normal_valid = None
    label = None if label is None else _label_input(label, count)
    finite = None if label is None else _label_observed(label)
    binary = False if label is None else _label_mode(label, label_mode)
    informative = False if label is None else _label_informative(label, binary)
    feature = _feature_array(feature, count)
    reference = None if feature is None else (
        np.ones(feature.shape[1]) if feature_scale is None else np.asarray(feature_scale, dtype=float).ravel())
    if reference is not None and (reference.shape != (feature.shape[1],) or np.any(reference <= 0)):
        raise ValueError("feature_scale needs one positive reference per channel.")
    data = None
    if mask is not None:
        if supports[0] is None or np.ndim(mask) != dim:
            raise ValueError("mask needs fixed radii and matching spatial dimension.")
        data = _prepare_mask_shape(mask, **(mask_options or {}))
    cfg = dict(points=points, n_bins=n_bins, eps=eps, scale_mode=scale_mode,
               normals=normals, normal_valid=normal_valid, normal_gate=normal_gate,
               label=label, label_finite=finite, label_binary=binary, label_informative=informative,
               feature=feature, feature_bins=n_bins if feature_bins is None else feature_bins,
               hist_prior=hist_prior, include_intensity=include_intensity, reference=reference,
               mask_data=data, return_info=return_info)
    reuse = radii is not None and kd_method in ("sknn", "scipy", "ckdtree")
    if reuse:
        wide_dist, wide_ind = get_neighbors(points, knn=knn, radius=max(supports), kd_method=kd_method)
    parts, blocks, flags, offset = [], {}, [], 0
    for rad in supports:
        if reuse:
            width = wide_dist.shape[1]
            if knn is None and count:
                width = min(width, max(1, int((wide_dist <= rad).sum(1).max())))
            keep = wide_dist[:, :width] <= rad
            dist, ind = np.where(keep, wide_dist[:, :width], np.inf), np.where(keep, wide_ind[:, :width], -1)
        else:
            dist, ind = get_neighbors(points, knn=knn, radius=rad, kd_method=kd_method)
        prefix = "" if radii is None else f"@r={rad:g}"
        values, local, supported = _scale_blocks(cfg, dist, ind, rad, prefix)
        if return_info:
            blocks.update({name: slice(section.start + offset, section.stop + offset)
                           for name, section in local.items()})
            flags.extend(supported)
        parts.extend(values)
        offset += sum(value.shape[1] for value in values)
    result = np.concatenate(parts, axis=1)
    if not return_info:
        return result
    return result, {"blocks": blocks, "valid": np.column_stack(flags)}


# ---------------------------------------------------------------- images

def _sample_linear(array, coord):
    """Sample channel-last fields at in-bounds array-axis coordinates.

    Accumulate in float64 in the original corner order. Skip zero-weight
    corners so an unused NaN/Inf cannot contaminate an integer-grid sample.
    Bounds and XY/XYZ conversion belong to the calling wrapper.
    """
    low = np.floor(coord).astype(np.intp)
    fraction = coord - low
    values = np.zeros((len(coord), array.shape[-1]))
    for corner in range(1 << coord.shape[1]):
        weight, index = np.ones(len(coord)), []
        for axis, length in enumerate(array.shape[:-1]):
            upper = (corner >> axis) & 1
            index.append(np.minimum(low[:, axis] + upper, length - 1))
            weight *= fraction[:, axis] if upper else 1.0 - fraction[:, axis]
        keep = weight > 0
        values[keep] += weight[keep, None] * array[tuple(i[keep] for i in index)]
    return values


def _image_sample(array, point):
    """Multilinear interpolation; invalid points give NaN, never clamped."""
    coord = np.asarray(point, dtype=float)[:, ::-1]
    valid = np.isfinite(coord).all(axis=1)
    for axis, length in enumerate(array.shape[:-1]):
        valid &= (coord[:, axis] >= 0) & (coord[:, axis] <= length - 1)
    out = np.full((len(coord), array.shape[-1]), np.nan, dtype=np.float32)
    rows = np.flatnonzero(valid)
    if len(rows):
        values = _sample_linear(array, coord[rows])
        out[rows] = values
        valid[rows] &= np.isfinite(values).all(axis=1)
    return out, valid


def _gaussian_derivative(array, sigma, axis, other=None):
    order = [0] * array.ndim
    order[axis] = 1
    if other is not None:
        order[other] += 1
    return ndi.gaussian_filter(array, sigma, order=order, mode="reflect")


def _gaussian_eigenvalues(hessian, dim):
    """Ascending Hessian eigenvalues; closed form in 2D."""
    if dim == 2:
        a, b, c = hessian[..., 0, 0], hessian[..., 0, 1], hessian[..., 1, 1]
        half = (a - c) * 0.5
        root = np.sqrt(half * half + b * b)
        middle = (a + c) * 0.5
        return np.stack([middle - root, middle + root], axis=-1)
    return np.linalg.eigvalsh(hessian)


def image_features(image, point=None, label=None, scales=(1.0, 2.0, 4.0),
                   channel_axis=-1, spacing=None, return_info=False, eps=1e-10):
    """Multi-scale image descriptors; return a dense map or features at point.

    image         Scalar field or channel-last array, with any spatial dimension.
    point         Optional (N,d) indices in reversed array-axis order; None returns dense output.
                  Spacing never converts these indices to physical coordinates.
    label         Optional full binary mask matching the spatial image shape.
    scales        Physical Gaussian sigmas, default (1,2,4).
    channel_axis  Channel axis, default -1; None for grayscale volumes.
    spacing       Physical array-axis steps, e.g. YX/ZYX, default ones.
    return_info   True returns (features, {blocks, valid}); False builds no info.
    eps           Numerical tolerance, default 1e-10.

    Each channel/scale has mean, std, scaled gradient/Laplacian/Hessian
    eigenvalues, and DoG. Point output uses bilinear/trilinear sampling after
    full-image filtering (multilinear sampling in higher dimensions).
    Missing observations or outside points produce NaN.
    valid is the observed raster or finite sampled rows; it does not certify
    the availability of both mask sides for inside/outside contrast.
    """
    array = np.asarray(image, dtype=float)
    array = (array[..., None] if array.ndim == 2 or channel_axis is None
             else np.moveaxis(array, channel_axis, -1))
    shape = array.shape[:-1]
    dim, channels = len(shape), array.shape[-1]
    if not dim or 0 in array.shape:
        raise ValueError("image needs nonempty spatial axes.")
    step = _mask_step(spacing, dim, "spacing", 1.0)
    scales = _scales(scales, "scales")
    observed = np.isfinite(array).all(axis=-1)
    if not observed.any():
        raise ValueError("image has no finite observations across all channels.")
    complete = bool(observed.all())
    source = array if complete else np.where(observed[..., None], array, 0.0)
    domain = 1.0 if complete else observed.astype(float)
    mask = None if label is None else _binary_mask(label)
    if mask is not None and mask.shape != shape:
        raise ValueError("label must match the spatial image shape.")

    width = channels * (5 + dim)
    n_image = len(scales) * width
    n_mask = 0 if mask is None else 1 + len(scales) * (1 + channels)
    if point is None:
        result = np.empty(shape + (n_image + n_mask,), dtype=np.float32)
    else:
        point = np.asarray(point, dtype=float)
        if point.ndim != 2 or point.shape[1] != dim:
            raise ValueError(f"point must have shape (N, {dim}) in reversed array-axis order.")
        result = np.empty((len(point), n_image + n_mask), dtype=np.float32)

    blocks = {} if return_info else None
    offset = 0

    def put(field):
        nonlocal offset
        if point is None:
            result[..., offset] = field
        else:
            field = np.asarray(field, dtype=np.float32)
            if not complete:
                field = np.where(observed, field, np.nan)
            result[:, offset] = _image_sample(field[..., None], point)[0][:, 0]
        offset += 1

    filter_domain = np.ones((1,) * dim) if complete else domain
    cached_sigma, cached_domain, cached_values = None, None, [None] * channels
    for index, sigma in enumerate(scales):
        sig = tuple(float(sigma) / step)
        reuse = cached_sigma == sigma
        smooth_domain = cached_domain if reuse else ndi.gaussian_filter(filter_domain, sig, mode="reflect")
        denominator = np.maximum(smooth_domain, eps)
        first, second = [], {}
        for axis in range(dim):
            first.append(_gaussian_derivative(filter_domain, sig, axis))
            for other in range(axis, dim):
                second[axis, other] = _gaussian_derivative(filter_domain, sig, axis, other)
        wide_sig = tuple(2.0 * s for s in sig)
        wide_raw = ndi.gaussian_filter(filter_domain, wide_sig, mode="reflect")
        wide_domain = np.maximum(wide_raw, eps)
        keep_next = index + 1 < len(scales) and scales[index + 1] == 2 * sigma
        start = offset
        for channel in range(channels):
            values = source[..., channel]
            mean = (cached_values[channel] if reuse else
                    ndi.gaussian_filter(values, sig, mode="reflect")) / denominator
            variance = (ndi.gaussian_filter(values * values, sig, mode="reflect")
                        / denominator - mean * mean)
            index_gradient = []
            for axis in range(dim):
                derivative = _gaussian_derivative(values, sig, axis)
                index_gradient.append((derivative - mean * first[axis]) / denominator)
            physical = [g / step[axis] for axis, g in enumerate(index_gradient)]
            hessian = np.empty(shape + (dim, dim))
            for axis in range(dim):
                for other in range(axis, dim):
                    derivative = _gaussian_derivative(values, sig, axis, other)
                    # quotient rule for f = S/D, then index -> physical, then s^2
                    component = (derivative - mean * second[axis, other]
                                 - index_gradient[axis] * first[other]
                                 - index_gradient[other] * first[axis]) / denominator
                    hessian[..., axis, other] = hessian[..., other, axis] = (
                        component * sigma ** 2 / (step[axis] * step[other]))
            eigenvalues = _gaussian_eigenvalues(hessian, dim)
            wide_values = ndi.gaussian_filter(values, wide_sig, mode="reflect")
            cached_values[channel] = wide_values if keep_next else None
            fields = [mean, np.sqrt(np.maximum(variance, 0)),
                      sigma * np.sqrt(sum(g * g for g in physical)),
                      np.trace(hessian, axis1=-2, axis2=-1)]
            fields.extend(eigenvalues[..., axis] for axis in range(dim))
            fields.append(mean - wide_values / wide_domain)
            for field in fields:
                put(field)
        if return_info:
            blocks[f"image.sigma={sigma:g}"] = slice(start, offset)
        cached_sigma, cached_domain = (2 * sigma, wide_raw) if keep_next else (None, None)

    if mask is not None:
        boundary_observed = bool(mask.any() and (~mask).any())
        distance = (ndi.distance_transform_edt(mask, sampling=tuple(step))
                    - ndi.distance_transform_edt(~mask, sampling=tuple(step))
                    if boundary_observed else np.zeros(shape))
        put(distance / scales[0])
        for sigma in scales:
            sig = tuple(float(sigma) / step)
            inside = ndi.gaussian_filter(domain * mask, sig, mode="reflect")
            outside = ndi.gaussian_filter(domain * (~mask), sig, mode="reflect")
            put(inside if complete else ndi.gaussian_filter(mask.astype(float), sig, mode="reflect"))
            valid = (inside > eps) & (outside > eps)
            for channel in range(channels):
                values = source[..., channel]
                mean_in = (ndi.gaussian_filter(values * mask, sig, mode="reflect")
                           / np.maximum(inside, eps))
                mean_out = (ndi.gaussian_filter(values * (~mask), sig, mode="reflect")
                            / np.maximum(outside, eps))
                var_in = (ndi.gaussian_filter(values * values * mask, sig, mode="reflect")
                          / np.maximum(inside, eps) - mean_in ** 2)
                var_out = (ndi.gaussian_filter(values * values * (~mask), sig, mode="reflect")
                           / np.maximum(outside, eps) - mean_out ** 2)
                contrast = ((mean_in - mean_out)
                            / np.sqrt(np.maximum(var_in, 0) + np.maximum(var_out, 0) + eps))
                put(np.where(valid, contrast, 0))
        if return_info:
            blocks["mask"] = slice(n_image, offset)

    if point is None:
        result[~observed] = np.nan
    if not return_info:
        return result
    return result, {"blocks": blocks, "valid": observed if point is None
                    else np.isfinite(result).all(axis=1)}


# ---------------------------------------------------------------- mask shape

def _mask_step(value, dim, name, default):
    vector = np.broadcast_to(np.asarray(default if value is None else value, dtype=float), (dim,))
    if name == "spacing" and np.any(vector <= 0):
        raise ValueError("spacing must be positive.")
    return vector


def _boundary_mask(binary):
    """Voxels on either side of a 4/6-connected label change."""
    boundary = np.zeros_like(binary, dtype=bool)
    for axis in range(binary.ndim):
        if binary.shape[axis] < 2:
            continue
        lo = tuple(slice(None, -1) if a == axis else slice(None) for a in range(binary.ndim))
        hi = tuple(slice(1, None) if a == axis else slice(None) for a in range(binary.ndim))
        change = binary[lo] != binary[hi]
        boundary[lo] |= change
        boundary[hi] |= change
    return boundary


def _mask_sample(array, coord):
    """Float64 scalar-field sampling at validated array-axis coordinates."""
    return _sample_linear(array[..., None], coord)[:, 0]


def _prepare_mask_shape(mask, spacing=None, origin=None, smoothing_sigma=None):
    """Prepare a binary raster with a coordinate-order physical origin.

    SDF distances use opposite-label voxel centres. Curvature is the inward
    normal divergence; only boundary values are retained after preparation.
    """
    binary = _binary_mask(mask)
    if not binary.ndim or 0 in binary.shape:
        raise ValueError("mask needs nonempty spatial axes.")
    dim = binary.ndim
    step = _mask_step(spacing, dim, "spacing", 1.0)
    offset = _mask_step(origin, dim, "origin", 0.0)
    sigma = float(2.5 * step.max() if smoothing_sigma is None else smoothing_sigma)
    data = dict(mask=binary, spacing=step, origin=offset,
                shape=np.asarray(binary.shape), dim=dim, smoothing_sigma=sigma,
                boundary_observed=bool(binary.any() and not binary.all()))
    if not data["boundary_observed"]:
        return data
    sdf = ndi.distance_transform_edt(binary, sampling=step)
    sdf -= ndi.distance_transform_edt(~binary, sampling=step)
    smooth = ndi.gaussian_filter(sdf, sigma / step, mode="reflect", truncate=4.0)
    derivative = lambda a, axis: (np.gradient(a, step[axis], axis=axis)
                                   if binary.shape[axis] > 1 else np.zeros_like(a))
    gradient = [derivative(smooth, axis) for axis in range(dim)]
    length = np.sqrt(sum(g * g for g in gradient))
    normal = [g / np.maximum(length, np.finfo(float).eps) for g in gradient]
    curvature = sum(derivative(component, axis) for axis, component in enumerate(normal))
    boundary = _boundary_mask(binary)
    index = np.argwhere(boundary)
    flat = np.flatnonzero(boundary.ravel())
    margin = np.min(np.minimum((index + .5) * step,
                               (data["shape"] - .5 - index) * step), axis=1)
    physical = offset + index[:, ::-1] * step[::-1]
    data.update(sdf=sdf, gradient=gradient, boundary_phys=physical,
                boundary_valid=((margin > 4 * sigma + 3 * step.max())
                                & (length.ravel()[flat] >= .5)
                                & np.isfinite(curvature.ravel()[flat])),
                boundary_curvature=curvature.ravel()[flat],
                boundary_tree=cKDTree(physical) if len(index) else None)
    return data


def _mask_shape_features(mask, point, radius, n_bins=11, spacing=None, origin=None,
                         eps=1e-10, prepared=None, return_info=False):
    """Distance, three annular occupancies/angle profiles and mean curvature.

    Coordinate axes reverse mask array axes (XY versus YX in 2D). Each supported
    ring carries 1/3 of angular squared norm. Curvature is a boundary-window
    average, not exact point curvature. Outside 2D the angular axis is polar cosine.
    A boundary sample requires crop margin
    greater than 4*sigma+3*max(spacing). Optional info is only blocks and (N,4)
    validity for distance, occupancy, angles and curvature, in that order.
    """
    data = prepared if prepared is not None else (
        mask if isinstance(mask, dict) else _prepare_mask_shape(mask, spacing, origin))
    step, offset, dim, shape = data["spacing"], data["origin"], data["dim"], data["shape"]
    point = np.asarray(point, dtype=float)
    coord = (point - offset)[:, ::-1] / step
    inside = (np.isfinite(coord).all(1)
              & ((coord >= 0) & (coord <= shape - 1)).all(1))
    count, rings = len(point), 3
    features = np.zeros((count, 5 + rings * n_bins))
    validity = np.zeros((count, 4), dtype=bool) if return_info else None
    normal, gradient_norm, crop, sdf = (np.zeros((count, dim)), np.zeros(count),
                                        np.zeros(count), np.zeros(count))
    normal_valid = curvature_pre = np.zeros(count, dtype=bool)
    rows = np.flatnonzero(inside)
    if len(rows):
        c = coord[rows]
        crop[rows] = np.min(np.minimum((c + .5) * step, (shape - .5 - c) * step), axis=1)
    if data["boundary_observed"] and len(rows):
        sdf[rows] = _mask_sample(data["sdf"], coord[rows])
        distance_valid = inside & (np.abs(sdf) < crop - eps)
        features[distance_valid, 0] = sdf[distance_valid] / radius
        if return_info:
            validity[:, 0] = distance_valid
        g = np.column_stack([_mask_sample(v, coord[rows]) for v in data["gradient"]])
        gradient_norm[rows] = np.linalg.norm(g, axis=1)
        normal[rows] = g / np.maximum(gradient_norm[rows, None], eps)
        sigma = data["smoothing_sigma"]
        margin = 4 * sigma + 2 * step.max()
        normal_valid = (distance_valid & (gradient_norm >= .5)
                        & (crop > margin) & (np.abs(sdf) <= radius))
        curvature_pre = (normal_valid & (np.abs(sdf) <= step.max())
                         & (crop > margin + step.max()))
    for row in rows:
        centre = coord[row]
        axes = [np.arange(np.floor(c - radius / h), np.ceil(c + radius / h) + 1,
                          dtype=np.intp) for c, h in zip(centre, step)]
        indices = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, dim)
        delta = (indices - centre) * step
        distance = np.linalg.norm(delta, axis=1)
        keep = distance <= radius + eps
        indices, delta, distance = indices[keep], delta[keep], distance[keep]
        ring = np.clip(np.ceil(distance / radius * rings - eps).astype(np.intp) - 1,
                       0, rings - 1)
        total = np.bincount(ring, minlength=rings)
        observed = ((indices >= 0) & (indices < shape)).all(1)
        if not observed.any():
            continue
        available = np.bincount(ring[observed], minlength=rings)
        support = np.divide(available, total, out=np.zeros(rings), where=total > 0)
        ij = indices[observed]
        foreground = data["mask"][tuple(ij[:, a] for a in range(dim))]
        np.divide(np.bincount(ring[observed], weights=foreground, minlength=rings),
                  available, out=features[row, 1:4], where=available > 0)
        complete = (total > 0) & (support >= 1 - eps)
        if return_info:
            validity[row, 1] = complete.all()
        if curvature_pre[row]:
            near = np.asarray(data["boundary_tree"].query_ball_point(point[row], radius),
                              dtype=np.intp)
            if near.size:
                weight = np.exp(-.5 * (np.linalg.norm(
                    data["boundary_phys"][near] - point[row], axis=1) / (.5 * radius)) ** 2)
                selected = data["boundary_valid"][near]
                curvature_support = weight[selected].sum() / weight.sum() * support.min()
                near, weight = near[selected], weight[selected]
                if len(near) >= max(3, dim + 1):
                    weight /= weight.sum()
                    kappa = float(weight @ data["boundary_curvature"][near])
                    if curvature_support >= 1 - eps and abs(kappa) * sigma <= .5:
                        features[row, -1] = radius * kappa
                        if return_info:
                            validity[row, 3] = True
        if not normal_valid[row] or not complete.all():
            continue
        use = observed & (distance > eps)
        use[observed] &= foreground
        if not use.any():
            continue
        direction = delta[use] / distance[use, None]
        polar = np.clip(direction @ normal[row], -1, 1)
        if dim == 2:
            tangent = np.array((normal[row, 1], -normal[row, 0]))
            angle_bin = np.floor((np.arctan2(direction @ tangent, polar) + np.pi)
                                 / (2 * np.pi) * n_bins).astype(np.intp) % n_bins
        else:
            angle_bin = np.minimum(((polar + 1) * .5 * n_bins).astype(np.intp), n_bins - 1)
        hist = np.bincount(ring[use] * n_bins + angle_bin,
                           weights=np.exp(-.5 * (distance[use] / radius) ** 2),
                           minlength=rings * n_bins).reshape(rings, n_bins)
        mass = hist.sum(1, keepdims=True)
        hist = np.divide(hist, mass, out=np.zeros_like(hist), where=mass > eps)
        features[row, 4:4 + rings * n_bins] = np.sqrt(hist).ravel() / np.sqrt(rings)
        if return_info:
            validity[row, 2] = (mass[:, 0] > eps).all()
    if not return_info:
        return features
    blocks = dict(mask_distance=slice(0, 1), mask_occupancy=slice(1, 4),
                  mask_boundary=slice(4, 4 + rings * n_bins),
                  mask_curvature=slice(4 + rings * n_bins, 5 + rings * n_bins))
    return features, dict(blocks=blocks, valid=validity)


# ---------------------------------------------------------------- batches

def _batch_inputs(samples, each_args, shared):
    """Expand parameter columns: a Python list is per sample; all else broadcasts."""
    mapping = isinstance(samples, Mapping)
    keys = list(samples) if mapping else list(range(len(samples)))
    data = [samples[key] for key in keys]
    columns = each_args or {}
    if set(columns) & {"points", "image", "return_info", "n_jobs"}:
        raise ValueError("Data, return_info and n_jobs belong outside each_args.")
    for name, value in columns.items():
        if isinstance(value, list) and len(value) != len(data):
            raise ValueError(f"each_args[{name!r}] needs one value per sample.")
    options = []
    for i in range(len(data)):
        item = {name: value[i] if isinstance(value, list) else value
                for name, value in columns.items()}
        option = dict(shared)
        if "normal" in item:
            option.pop("normals", None)
        if "normals" in item:
            option.pop("normal", None)
        if item.get("mask_options") is not None:
            item["mask_options"] = dict(option.get("mask_options") or {}, **item["mask_options"])
        option.update(item)
        options.append(option)
    return keys, data, options, mapping


def _batch_same(values, what):
    if any(not np.array_equal(value, values[0]) for value in values[1:]):
        raise ValueError(f"All samples must share {what}.")
    return values[0] if values else None


def _batch_present(options, names):
    for name in names:
        _batch_same([option.get(name) is not None for option in options],
                    f"the presence of {name}")


def _batch_shared(options, excluded=()):
    """Keep only common configuration; never copy sample arrays into info."""
    common = {}
    if not options:
        return common
    for name, value in options[0].items():
        if name in excluded or not all(name in option for option in options):
            continue
        values = [option[name] for option in options]
        if all(isinstance(v, Mapping) for v in values):
            nested = _batch_shared(values)
            if nested:
                common[name] = nested
        elif all(np.array_equal(v, value) for v in values[1:]):
            common[name] = value
    return common


def _batch_dispatch(function, keys, data, options, mapping, shared,
                    n_jobs, return_info):
    workers = (os.cpu_count() or 1) if n_jobs == -1 else n_jobs
    if workers < 1:
        raise ValueError("n_jobs must be positive or -1.")

    def run(pair):
        return function(pair[0], return_info=return_info, **pair[1])

    if workers > 1 and len(data) > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(data))) as pool:
            results = list(pool.map(run, zip(data, options)))
    else:
        results = [run(pair) for pair in zip(data, options)]
    pack = (lambda values: dict(zip(keys, values))) if mapping else list
    if not return_info:
        _batch_same([value.shape[-1] for value in results], "feature width")
        return pack(results)
    values, infos = zip(*results) if results else ((), ())
    blocks = infos[0]["blocks"] if infos else {}
    if any(info["blocks"] != blocks for info in infos[1:]):
        raise ValueError("All samples must share feature blocks.")
    return pack(values), dict(blocks=blocks, valid=pack(info["valid"] for info in infos),
                              shared_parameters=shared)


def FPFH_batch(point_clouds, each_args=None, *, n_jobs=5, return_info=False, **kwargs):
    """Point-cloud descriptors with parameter columns, in sample order.

    each_args maps parameter names to lists: label=Ll/Lr, feature=Fr, etc.
    Lists have one value per cloud; any non-list value broadcasts. Use tuples
    or arrays for shared vectors, e.g. radii=(20,40). Outer kwargs are shared.
    Hard string/numeric labels and soft membership matrices are accepted.
    Single-valued parameters such as knn/radius may also use per-cloud lists.
    Channel counts and output layouts remain common. n_jobs defaults to 5.
    Optional info contains blocks, valid and common resolved shared_parameters.
    """
    shared = dict(kwargs)
    keys, data, options, mapping = _batch_inputs(point_clouds, each_args, shared)
    data = [np.asarray(points, dtype=float) for points in data]
    _batch_same([points.shape[1] for points in data], "point dimension")
    _batch_present(options, ("label", "feature", "mask"))
    _batch_same([(o.get("normal") is not None) or (o.get("normals") is not None)
                 for o in options], "the presence of normal descriptors")
    # These alter bin/column definitions rather than neighbourhood sampling.
    for name, default in (("n_bins", 11), ("feature_bins", 5),
                          ("include_intensity", True), ("radii", None)):
        _batch_same([option.get(name, default) for option in options], name)

    for key, points, option in zip(keys, data, options):
        if option.get("label") is not None:
            option["label"] = _label_input(option["label"], len(points),
                                               f"Sample {key!r}: label", normalize=False)
        option["feature"] = _feature_array(
            option.get("feature"), len(points), f"Sample {key!r}: feature")
    labels = [o["label"] for o in options if o.get("label") is not None]
    channels = _batch_same([o["feature"].shape[1] for o in options
                            if o["feature"] is not None], "feature channel count") or 0

    if labels:
        numeric = all(label.ndim == 1 and label.dtype.kind in "biuf" for label in labels)
        observed = [label[_label_observed(label)] for label in labels] if numeric else []
        binary = numeric and all(np.isin(value, (0, 1, 255)).all() for value in observed)
        for option in options:
            if option.get("label_mode", "auto") == "auto":
                option["label_mode"] = "binary" if binary else "region"
        _batch_same([option.get("label_mode", "auto") for option in options], "label semantics")

    if options and options[0].get("mask") is not None:
        sigma = 2.5 * max(_mask_step((o.get("mask_options") or {}).get("spacing"),
                                     np.ndim(o["mask"]), "spacing", 1.0).max() for o in options)
        for option in options:
            mask_options = dict(option.get("mask_options") or {})
            if mask_options.get("smoothing_sigma") is None:
                mask_options["smoothing_sigma"] = sigma
            option["mask_options"] = mask_options

    auto_radius = [isinstance(o.get("radius"), str) and o["radius"] == "auto" for o in options]
    if any(auto_radius):
        if not all(auto_radius) or any(o.get("radii") is not None for o in options):
            raise ValueError("radius='auto' must apply to all clouds without radii.")
        estimates = []
        for points, option in zip(data, options):
            knn = option.get("knn", 31)
            dist, ind = get_neighbors(points, knn=31 if knn is None else knn,
                                      kd_method=option.get("kd_method", "sknn"))
            valid = ((ind != np.arange(len(points))[:, None]) & np.isfinite(dist)
                     & (dist > option.get("eps", 1e-10)))
            farthest = np.where(valid, dist, 0).max(1) if dist.shape[1] else np.zeros(len(points))
            farthest = farthest[farthest > 0]
            if farthest.size:
                estimates.append(np.median(farthest))
        if not estimates:
            raise ValueError("radius='auto' needs positive neighbour distances.")
        for option in options:
            option["radius"] = float(np.median(estimates))

    auto_scale = [isinstance(o.get("feature_scale"), str) and o["feature_scale"] == "auto"
                  for o in options]
    if any(auto_scale):
        if not all(auto_scale) or not channels:
            raise ValueError("feature_scale='auto' needs features in all clouds.")
        squares = []
        for option in options:
            feature = option["feature"]
            finite = np.isfinite(feature)
            mass = finite.sum(0)
            squares.append(np.divide(np.sum(np.where(finite, feature, 0.0) ** 2, axis=0),
                                     mass, out=np.full(channels, np.nan), where=mass > 0))
        squares = np.asarray(squares)
        counts = np.isfinite(squares).sum(0)
        if np.any(counts == 0):
            raise ValueError("feature_scale='auto' needs observations in every channel.")
        reference = np.sqrt(np.nansum(squares, axis=0) / counts)
        reference[reference == 0] = 1.0
        for option in options:
            option["feature_scale"] = reference

    public = None
    if return_info:
        public = _batch_shared(options, {"label", "feature", "normal", "normals", "mask"})
        effective = [o.get("normal") if o.get("normal") is not None else o.get("normals")
                     for o in options]
        if effective and all(isinstance(n, str) and n == "auto" for n in effective):
            public["normal"] = "auto"
        if channels and all(o.get("feature_scale") is None for o in options):
            public["feature_scale"] = np.ones(channels)
    return _batch_dispatch(FPFH, keys, data, options, mapping, public, n_jobs, return_info)


def image_features_batch(images, each_args=None, *, n_jobs=5, return_info=False, **kwargs):
    """Image descriptors; each_args={'point': [q0,q1], 'spacing': [(dy,dx), ...]}.

    Every list is indexed in image order; scalars/tuples/arrays broadcast.
    Outer kwargs are shared defaults. Point lists may mix None (dense) and
    index arrays. channel_axis may vary; channels and output layout must agree.
    n_jobs defaults to 5; optional info adds common resolved shared_parameters.
    """
    shared = dict(kwargs)
    keys, data, options, mapping = _batch_inputs(images, each_args, shared)
    dimensions, widths = [], []
    for image, option in zip(data, options):
        array = np.asarray(image)
        axis = option.get("channel_axis", -1)
        if array.ndim == 2 or axis is None:
            dimensions.append(array.ndim)
            widths.append(1)
        else:
            moved = np.moveaxis(array, axis, -1)
            dimensions.append(moved.ndim - 1)
            widths.append(moved.shape[-1])
    _batch_same(dimensions, "spatial dimension")
    _batch_same(widths, "image channel count")
    _batch_same([o.get("scales", (1., 2., 4.)) for o in options], "scales")
    _batch_present(options, ("label",))
    public = _batch_shared(options, {"point", "label"}) if return_info else None
    return _batch_dispatch(image_features, keys, data, options, mapping, public, n_jobs, return_info)