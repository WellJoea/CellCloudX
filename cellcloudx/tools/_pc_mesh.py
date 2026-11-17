"""Pure-PyTorch/CUDA reconstruction of a registered z-stack point cloud.

The pipeline is: robust spacing/outlier estimation, weighted KDE on each
section, z interpolation, sign-constrained implicit smoothing, marching
tetrahedra, light Taubin smoothing, and projection to the zero level-set.
"""

from pathlib import Path
import math

import torch
import torch.nn.functional as nnf

"""Coarse-to-fine point-cloud to mesh reconstruction.

Required: numpy, scipy, scikit-image, pyvista
Optional: torch + https://github.com/tatsy/torchmcubes
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pyvista as pv
from scipy import sparse
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    binary_opening,
    gaussian_filter,
    generate_binary_structure,
)
from scipy.spatial import cKDTree


"""Coarse-to-fine point-cloud to mesh reconstruction.

Required: numpy, scipy, scikit-image, pyvista
Optional: torch + https://github.com/tatsy/torchmcubes
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pyvista as pv
from scipy import sparse
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    binary_opening,
    gaussian_filter,
    generate_binary_structure,
    label as ndi_label,
)
from scipy.spatial import cKDTree


def _rows(a, ids):
    return None if a is None else np.asarray(a)[ids]


def _inputs(X, label, feature, confidence, feature_mode):
    X = np.asarray(X, float)
    if X.ndim != 2 or X.shape[1] != 3 or len(X) < 4:
        raise ValueError("X must have shape (N, 3), N >= 4.")

    label = None if label is None else np.asarray(label).reshape(-1)
    if sparse.issparse(feature):
        if feature.shape[1] != 1:
            raise TypeError("Use a dense low-dimensional feature/embedding.")
        feature = feature.toarray().ravel()
    else:
        feature = None if feature is None else np.asarray(feature)
    confidence = None if confidence is None else np.asarray(confidence).reshape(-1)
    if label is not None and len(label) != len(X):
        raise ValueError("label must have length N.")
    if feature is not None and feature.shape[0] != len(X):
        raise ValueError("feature must have shape (N,) or (N, F).")
    if confidence is not None and len(confidence) != len(X):
        raise ValueError("confidence must have length N.")
    if feature_mode not in ("auto", "confidence", "attribute"):
        raise ValueError("feature_mode must be auto, confidence, or attribute.")

    valid = np.isfinite(X).all(1)
    X, label = X[valid], _rows(label, valid)
    feature, confidence = _rows(feature, valid), _rows(confidence, valid)
    _, keep = np.unique(X, axis=0, return_index=True)
    keep.sort()
    X, label = X[keep], _rows(label, keep)
    feature, confidence = _rows(feature, keep), _rows(confidence, keep)
    if len(X) < 4:
        raise ValueError("Fewer than four finite unique points remain.")

    if confidence is None and feature is not None and feature_mode != "attribute":
        f = feature.ravel() if feature.ndim == 1 or feature.shape[1] == 1 else None
        finite = None if f is None else f[np.isfinite(f)]
        if feature_mode == "confidence" and f is None:
            raise ValueError("confidence feature must be one-dimensional.")
        if finite is not None and len(finite) and finite.min() >= 0:
            confidence = f
        elif feature_mode == "confidence":
            raise ValueError("confidence feature must be nonnegative.")

    if confidence is None:
        confidence = np.ones(len(X))
    else:
        confidence = np.asarray(confidence, float)
        if np.any(np.isfinite(confidence) & (confidence < 0)):
            raise ValueError("confidence must be nonnegative.")
        ok = np.isfinite(confidence) & (confidence >= 0)
        fill = np.median(confidence[ok]) if ok.any() else 1.0
        confidence = np.where(ok, confidence, fill)
        positive = confidence[confidence > 0]
        if len(positive):
            lo, hi = np.quantile(positive, [0.02, 0.98])
            confidence = np.clip(confidence, lo, hi)
            confidence /= max(np.median(positive), np.finfo(float).eps)
        else:
            confidence[:] = 1
    return X, label, feature, confidence


def _factorize(label):
    try:
        return np.unique(label, return_inverse=True)
    except TypeError:
        levels, index, inverse = [], {}, np.empty(len(label), int)
        for i, value in enumerate(label):
            key = (type(value).__name__, repr(value))
            if key not in index:
                index[key] = len(levels)
                levels.append(value)
            inverse[i] = index[key]
        return np.asarray(levels, object), inverse


def _outlier_mask(X, confidence, k, ratio, n_iter, confidence_quantile):
    keep = np.ones(len(X), bool)
    n_drop = int(confidence_quantile * len(X))
    if n_drop and np.ptp(confidence) > np.finfo(float).eps:
        drop = np.argpartition(confidence, n_drop - 1)[:n_drop]
        keep[drop] = False

    if ratio is None:
        return keep
    for _ in range(n_iter):
        ids0 = np.flatnonzero(keep)
        if len(ids0) <= 4:
            break
        Xi = X[ids0]
        ki = min(max(2, int(k)), len(Xi) - 1)
        dist, ids = cKDTree(Xi).query(Xi, k=ki + 1, workers=-1)
        dk = dist[:, -1]
        local = np.median(dk[ids[:, 1:]], axis=1)
        good = dk <= ratio * np.maximum(local, np.finfo(float).eps)
        if good.all():
            break
        keep[ids0[~good]] = False
    return keep


def _local_radii(X, k, confidence, feature_power):
    tree = cKDTree(X)
    k = min(max(2, int(k)), len(X) - 1)
    dist = tree.query(X, k=k + 1, workers=-1)[0][:, 1:]
    radius = np.median(dist, axis=1)
    good = radius[np.isfinite(radius) & (radius > 0)]
    if not len(good):
        raise ValueError("Cannot estimate local point spacing.")
    lo, hi = np.quantile(good, [0.02, 0.98])
    radius = np.clip(np.nan_to_num(radius, nan=np.median(good)), lo, hi)
    radius *= np.clip(confidence ** feature_power, 0.5, 1.5)
    return tree, radius


def _grid(X, radius, coarse, voxel_size, voxel_scale, max_voxels):
    rmed = float(np.median(radius))
    h = np.asarray(rmed * voxel_scale if voxel_size is None else voxel_size, float)
    h = np.repeat(h, 3) if h.ndim == 0 or h.size == 1 else h.reshape(-1)
    if h.shape != (3,) or np.any(h <= 0):
        raise ValueError("voxel_size must be positive and scalar or length 3.")
    for _ in range(8):
        margin = coarse * np.quantile(radius, 0.99) + 2 * h
        origin, upper = X.min(0) - margin, X.max(0) + margin
        shape = np.maximum(3, np.ceil((upper - origin) / h).astype(int) + 1)
        nvox = int(np.prod(shape, dtype=np.int64))
        if nvox <= max_voxels:
            return origin, h, tuple(shape), rmed
        h *= 1.01 * (nvox / max_voxels) ** (1 / 3)
    raise RuntimeError("Cannot meet max_voxels; increase max_voxels.")


def _support_field(
    tree, radius, origin, spacing, shape, support_k, chunk_size,
    support_order=1,
):
    total = int(np.prod(shape, dtype=np.int64))
    field = np.empty(total, np.float32)
    order = min(max(1, int(support_order)), tree.n)
    k = min(max(order, int(support_k)), tree.n)
    yz = shape[1] * shape[2]
    for start in range(0, total, chunk_size):
        flat = np.arange(start, min(start + chunk_size, total), dtype=np.int64)
        ix, rem = divmod(flat, yz)
        iy, iz = divmod(rem, shape[2])
        q = np.column_stack((ix, iy, iz)) * spacing + origin
        dist, ids = tree.query(q, k=k, workers=-1)
        if k == 1:
            dist, ids = dist[:, None], ids[:, None]
        value = dist / radius[ids]
        field[start:start + len(flat)] = np.partition(
            value, order - 1, axis=1
        )[:, order - 1]
    return field.reshape(shape)


def _filter_mask_components(mask, min_voxels, keep_largest):
    if not keep_largest and min_voxels <= 1:
        return mask
    region, n_region = ndi_label(
        mask, structure=generate_binary_structure(3, 2)
    )
    if not n_region:
        return mask
    count = np.bincount(region.ravel())
    valid = np.zeros(len(count), bool)
    if keep_largest:
        valid[1 + np.argmax(count[1:])] = True
    else:
        valid[count >= min_voxels] = True
        valid[0] = False
    return valid[region]


def _marching_cubes(field, level, origin, spacing, backend, device):
    if backend not in ("auto", "skimage", "torch"):
        raise ValueError("backend must be auto, skimage, or torch.")
    if backend in ("auto", "torch"):
        try:
            import torch
            from torchmcubes import marching_cubes as torch_mc

            use_torch = backend == "torch" or (
                torch.cuda.is_available() and field.size >= 2_000_000
            )
            if use_torch:
                volume = torch.as_tensor(
                    np.ascontiguousarray(field), dtype=torch.float32, device=device
                )
                vertices, faces = torch_mc(volume, float(level))
                vertices = vertices.detach().cpu().numpy()
                faces = faces.detach().cpu().numpy()
                return origin + vertices * spacing, faces
        except Exception:
            if backend == "torch":
                raise

    from skimage.measure import marching_cubes

    vertices, faces, _, _ = marching_cubes(
        field, float(level), spacing=tuple(spacing),
        gradient_direction="descent", allow_degenerate=False,
    )
    return origin + vertices, faces


def _polydata(vertices, faces, min_component_faces, keep_largest):
    cells = np.c_[np.full(len(faces), 3), faces].astype(np.int64).ravel()
    mesh = pv.PolyData(np.asarray(vertices), cells).clean().triangulate()
    if keep_largest:
        mesh = mesh.connectivity(extraction_mode="largest").clean()
    elif min_component_faces > 1:
        conn = mesh.connectivity(extraction_mode="all")
        region = np.asarray(conn.cell_data["RegionId"])
        count = np.bincount(region)
        mask = count[region] >= min_component_faces
        if not mask.any():
            raise ValueError("All components were removed; lower min_component_faces.")
        mesh = conn.extract_cells(mask).extract_surface(
            algorithm="dataset_surface"
        ).triangulate().clean()
    if mesh.n_cells == 0:
        raise ValueError("Surface extraction returned an empty mesh.")
    mesh.clear_data()
    return mesh


def _extract(
    field, level, origin, spacing, backend, device,
    fill_holes, opening_iters, closing_iters,
    min_component_faces, keep_largest,
):
    inside = field <= level
    cleaned = inside
    if opening_iters:
        structure = generate_binary_structure(3, 2)
        cleaned = binary_opening(
            cleaned, structure=structure, iterations=opening_iters
        )
    if closing_iters:
        structure = generate_binary_structure(3, 2)
        cleaned = binary_closing(
            cleaned, structure=structure, iterations=closing_iters
        )
    if fill_holes:
        cleaned = binary_fill_holes(cleaned)

    changed = cleaned != inside
    work = field
    if changed.any():
        work = field.copy()
        delta = max(0.05, 0.25 * level)
        lower, upper = level - delta, level + delta
        work[cleaned & ~inside] = lower
        work[inside & ~cleaned] = upper
    vertices, faces = _marching_cubes(
        work, level, origin, spacing, backend, device
    )
    return _polydata(vertices, faces, min_component_faces, keep_largest)


def _smooth_displacement(vertices, displacement, sigma, k=16):
    key = np.floor(
        (vertices - vertices.min(0)) / max(0.65 * sigma, 1e-12)
    ).astype(np.int64)
    _, controls = np.unique(key, axis=0, return_index=True)
    center, coeff = vertices[controls], displacement[controls]
    k = min(max(1, k), len(center))
    dist, ids = cKDTree(center).query(vertices, k=k, workers=-1)
    if k == 1:
        dist, ids = dist[:, None], ids[:, None]
    weight = np.exp(-0.5 * (dist / max(sigma, 1e-12)) ** 2)
    weight /= weight.sum(1, keepdims=True).clip(1e-12)
    return np.einsum("nk,nkd->nd", weight, coeff[ids])


def _limit(displacement, limit):
    length = np.linalg.norm(displacement, axis=1)
    return displacement * np.minimum(1, limit / np.maximum(length, 1e-12))[:, None]


def _deform_level(
    vertices, tree, X, radius, level, iterations,
    rmed, kernel_scale, step, tolerance, support_k,
):
    vertices, residual = vertices.copy(), np.inf
    k = min(max(1, int(support_k)), len(X))
    for _ in range(iterations):
        dist, ids = tree.query(vertices, k=k, workers=-1)
        if k == 1:
            dist, ids = dist[:, None], ids[:, None]
        winner = np.argmin(dist / radius[ids], axis=1)
        row = np.arange(len(vertices))
        dist, ids = dist[row, winner], ids[row, winner]
        residual = float(np.median(np.abs(dist / radius[ids] - level)))
        if residual <= tolerance:
            break
        direction = (vertices - X[ids]) / np.maximum(dist[:, None], 1e-12)
        guide = (level * radius[ids] - dist)[:, None] * direction
        sigma = max(kernel_scale * level * rmed, 0.5 * rmed)
        vertices += _limit(
            _smooth_displacement(vertices, guide, sigma), step * level * rmed
        )
    return vertices, residual


def _shrink_surface(
    vertices, tree, X, iterations, rmed, kernel_scale, step, tolerance,
):
    vertices, residual = vertices.copy(), np.inf
    for i in range(iterations):
        dist, ids = tree.query(vertices, k=1, workers=-1)
        residual = float(np.median(dist / rmed))
        if residual <= tolerance:
            break
        guide = X[ids] - vertices
        sigma = max(0.5, kernel_scale * (1 - 0.6 * i / max(iterations - 1, 1)))
        vertices += _limit(
            _smooth_displacement(vertices, guide, sigma * rmed), step * rmed
        )
    return vertices, residual


def _project(mesh, vertices_metric, X, label, feature, confidence, sigma):
    k = min(8, len(X))
    dist, ids = cKDTree(X).query(vertices_metric, k=k, workers=-1)
    if k == 1:
        dist, ids = dist[:, None], ids[:, None]
    weight = np.exp(-0.5 * (dist / max(sigma, 1e-12)) ** 2) * confidence[ids]
    weight /= weight.sum(1, keepdims=True).clip(1e-12)
    row = np.repeat(np.arange(mesh.n_points), k)
    P = sparse.csr_matrix(
        (weight.ravel(), (row, ids.ravel())), shape=(mesh.n_points, len(X))
    )
    mesh.point_data["confidence"] = np.asarray(P @ confidence).ravel()

    if label is not None:
        levels, code = _factorize(label)
        onehot = sparse.csr_matrix(
            (np.ones(len(label)), (np.arange(len(label)), code)),
            shape=(len(label), len(levels)),
        )
        probability = (P @ onehot).toarray()
        mesh.point_data["label_id"] = probability.argmax(1).astype(np.int32)
        mesh.point_data["label_confidence"] = probability.max(1)
        mesh.field_data["label_values"] = np.asarray([str(x) for x in levels])

    if feature is not None:
        feature = np.asarray(feature, float)
        vector = feature.ndim == 1
        feature = feature[:, None] if vector else feature
        valid = np.isfinite(feature)
        value = (P @ np.where(valid, feature, 0)) / np.maximum(
            P @ valid.astype(float), 1e-12
        )
        mesh.point_data["feature"] = value[:, 0] if vector else value


def _one_mesh(
    X, confidence, point_mode, k, radius_scales, voxel_size,
    voxel_scale, max_voxels, support_k, field_smooth,
    fill_holes, opening_iters, closing_iters,
    topology_levels, iterations, kernel_scale, step, tolerance,
    feature_power, min_component_faces, keep_largest, smooth_iters,
    smooth_pass_band, final_projection_iters, target_faces,
    backend, device, chunk_size,
):
    tree, radius = _local_radii(X, k, confidence, feature_power)
    origin, spacing, shape, rmed = _grid(
        X, radius, radius_scales[0], voxel_size, voxel_scale, max_voxels
    )
    field = _support_field(
        tree, radius, origin, spacing, shape, support_k, chunk_size
    )
    if field_smooth:
        gaussian_filter(field, field_smooth, output=field)

    mesh = _extract(
        field, radius_scales[0], origin, spacing, backend, device,
        fill_holes, opening_iters, closing_iters,
        min_component_faces, keep_largest,
    )
    residual = np.nan
    for level_id, level in enumerate(radius_scales[1:], 1):
        if level_id <= topology_levels or (
            (opening_iters or closing_iters)
            and level_id == len(radius_scales) - 1
        ):
            mesh = _extract(
                field, level, origin, spacing, backend, device,
                fill_holes, opening_iters, closing_iters,
                min_component_faces, keep_largest,
            )
        else:
            mesh.points, residual = _deform_level(
                np.asarray(mesh.points), tree, X, radius, level, iterations,
                rmed, kernel_scale, step, tolerance, support_k,
            )

    if point_mode == "surface":
        mesh.points, residual = _shrink_surface(
            np.asarray(mesh.points), tree, X, iterations,
            rmed, kernel_scale, step, tolerance,
        )
    if smooth_iters:
        mesh = mesh.smooth_taubin(
            n_iter=max(2, smooth_iters), pass_band=smooth_pass_band,
            normalize_coordinates=False,
        )
    if point_mode == "surface" and final_projection_iters:
        mesh.points, residual = _shrink_surface(
            np.asarray(mesh.points), tree, X, final_projection_iters,
            rmed, kernel_scale, step, tolerance,
        )
    elif point_mode == "volume" and final_projection_iters:
        mesh.points, residual = _deform_level(
            np.asarray(mesh.points), tree, X, radius, radius_scales[-1],
            final_projection_iters, rmed, kernel_scale, step, tolerance, support_k,
        )

    if target_faces is not None and mesh.n_cells > target_faces:
        reduction = 1 - int(target_faces) / mesh.n_cells
        mesh = mesh.decimate_pro(reduction, preserve_topology=True)
    mesh = mesh.clean().triangulate().compute_normals(
        point_normals=True, cell_normals=False,
        consistent_normals=True, auto_orient_normals=True,
    )
    mesh.field_data["grid_shape"] = np.asarray(shape)
    mesh.field_data["voxel_size_metric"] = spacing
    mesh.field_data["local_radius_median_metric"] = [rmed]
    mesh.field_data["radius_scales"] = radius_scales
    mesh.field_data["final_normalized_residual"] = [residual]
    return mesh, rmed


def clay_mesh(
    X,
    label=None,
    feature=None,
    *,
    confidence=None,
    point_mode="volume",
    label_mode="project",
    feature_mode="auto",
    axis_scale=1.0,
    outlier_ratio=2.5,
    outlier_k=16,
    outlier_iters=1,
    confidence_quantile=0.0,
    k=16,
    radius_scales=(2.2, 1.25, 0.65),
    voxel_size=None,
    voxel_scale=0.40,
    max_voxels=12_000_000,
    support_k=12,
    field_smooth=0.7,
    fill_holes=True,
    opening_iters=0,
    closing_iters=1,
    topology_levels=1,
    iterations=8,
    kernel_scale=1.5,
    step=0.35,
    tolerance=0.03,
    feature_power=0.25,
    min_component_faces=20,
    keep_largest=False,
    smooth_iters=2,
    smooth_pass_band=0.1,
    final_projection_iters=3,
    target_faces=None,
    backend="auto",
    device="cuda",
    chunk_size=200_000,
    min_label_points=30,
):
    """Build one PolyData, or {label: PolyData} for label_mode='separate'.

    volume: points fill the object (cells/voxels); surface: points lie on a
    closed surface. A positive 1-D feature is used as confidence in auto mode.
    Every supplied feature is also interpolated to mesh vertices.
    """
    if point_mode not in ("volume", "surface"):
        raise ValueError("point_mode must be volume or surface.")
    if label_mode not in ("project", "separate"):
        raise ValueError("label_mode must be project or separate.")
    radius_scales = tuple(np.asarray(radius_scales, float).ravel())
    if not radius_scales or np.any(np.asarray(radius_scales) <= 0):
        raise ValueError("radius_scales must be positive.")
    if np.any(np.diff(radius_scales) >= 0):
        raise ValueError("radius_scales must be strictly decreasing.")
    if voxel_scale <= 0 or max_voxels < 27 or chunk_size < 1:
        raise ValueError("voxel_scale, max_voxels and chunk_size must be positive.")
    if min(
        field_smooth, opening_iters, closing_iters, iterations, smooth_iters,
        final_projection_iters, tolerance, feature_power,
    ) < 0:
        raise ValueError("smoothing, iterations, tolerance and power must be >= 0.")
    if kernel_scale <= 0 or step <= 0 or min_component_faces < 1:
        raise ValueError("kernel_scale/step must be > 0; min faces must be >= 1.")
    if outlier_ratio is not None and outlier_ratio <= 1:
        raise ValueError("outlier_ratio must be > 1 or None.")
    if outlier_iters < 0 or not 0 <= confidence_quantile < 1:
        raise ValueError("Invalid outlier_iters or confidence_quantile.")
    if not 0 < smooth_pass_band <= 2:
        raise ValueError("smooth_pass_band must be in (0, 2].")
    topology_levels = int(np.clip(topology_levels, 0, len(radius_scales) - 1))

    X, label, feature, confidence = _inputs(
        X, label, feature, confidence, feature_mode
    )
    axis_scale = np.asarray(axis_scale, float)
    axis_scale = np.repeat(axis_scale, 3) if axis_scale.size == 1 else axis_scale
    if axis_scale.shape != (3,) or np.any(axis_scale <= 0):
        raise ValueError("axis_scale must be positive and scalar or length 3.")
    Xm = X * axis_scale
    n_input = len(X)
    keep = _outlier_mask(
        Xm, confidence, outlier_k, outlier_ratio,
        int(outlier_iters), confidence_quantile,
    )
    X, Xm = X[keep], Xm[keep]
    label, feature = _rows(label, keep), _rows(feature, keep)
    confidence = confidence[keep]
    if len(X) < 4:
        raise ValueError("Too few points remain after outlier filtering.")
    options = dict(
        point_mode=point_mode, k=k, radius_scales=radius_scales,
        voxel_size=voxel_size, voxel_scale=voxel_scale,
        max_voxels=int(max_voxels), support_k=support_k,
        field_smooth=field_smooth, fill_holes=fill_holes,
        opening_iters=int(opening_iters),
        closing_iters=int(closing_iters),
        topology_levels=topology_levels, iterations=int(iterations),
        kernel_scale=kernel_scale, step=step, tolerance=tolerance,
        feature_power=feature_power, min_component_faces=min_component_faces,
        keep_largest=keep_largest, smooth_iters=int(smooth_iters),
        smooth_pass_band=smooth_pass_band,
        final_projection_iters=int(final_projection_iters),
        target_faces=target_faces, backend=backend, device=device,
        chunk_size=int(chunk_size),
    )

    if label_mode == "separate":
        if label is None:
            raise ValueError("label is required for label_mode='separate'.")
        levels, code = _factorize(label)
        output = {}
        for i, value in enumerate(levels):
            take = code == i
            if take.sum() < max(min_label_points, k + 2):
                warnings.warn(f"Skip label {value!r}: too few points.")
                continue
            mesh, sigma = _one_mesh(Xm[take], confidence[take], **options)
            metric_points = np.asarray(mesh.points).copy()
            mesh.points = metric_points / axis_scale
            mesh.point_data["label_id"] = np.zeros(mesh.n_points, np.int32)
            mesh.field_data["label_values"] = [str(value)]
            mesh.field_data["axis_scale"] = axis_scale
            mesh.field_data["input_points"] = [n_input]
            mesh.field_data["filtered_points"] = [n_input - len(X)]
            _project(
                mesh, metric_points, Xm[take], None, _rows(feature, take),
                confidence[take], sigma,
            )
            output[value] = mesh
        if not output:
            raise ValueError("No label has enough points to build a mesh.")
        return output

    mesh, sigma = _one_mesh(Xm, confidence, **options)
    metric_points = np.asarray(mesh.points).copy()
    mesh.points = metric_points / axis_scale
    mesh.field_data["axis_scale"] = axis_scale
    mesh.field_data["input_points"] = [n_input]
    mesh.field_data["filtered_points"] = [n_input - len(X)]
    _project(mesh, metric_points, Xm, label, feature, confidence, sigma)
    return mesh


def densify_volume_points(
    X,
    *,
    factor=4.0,
    n_points=None,
    confidence=None,
    axis_scale=1.0,
    outlier_ratio=2.5,
    outlier_k=16,
    outlier_iters=1,
    confidence_quantile=0.0,
    k=16,
    support_scale=0.85,
    voxel_size=None,
    voxel_scale=0.50,
    max_voxels=12_000_000,
    support_k=12,
    support_order=1,
    field_smooth=1.5,
    opening_iters=0,
    min_component_voxels=0,
    keep_largest_support=False,
    closing_iters=2,
    fill_holes=True,
    include_original=True,
    jitter=0.45,
    feature_power=0.25,
    random_state=0,
    chunk_size=200_000,
):
    """Resample a repaired volume support into a denser point cloud."""
    X, _, _, confidence = _inputs(X, None, None, confidence, "auto")
    axis_scale = np.asarray(axis_scale, float)
    axis_scale = np.repeat(axis_scale, 3) if axis_scale.size == 1 else axis_scale
    if axis_scale.shape != (3,) or np.any(axis_scale <= 0):
        raise ValueError("axis_scale must be positive and scalar or length 3.")
    if factor < 1 or support_scale <= 0 or not 0 <= jitter < 0.5:
        raise ValueError("factor >= 1, support_scale > 0 and 0 <= jitter < 0.5.")
    if min(field_smooth, opening_iters, closing_iters) < 0:
        raise ValueError("field_smooth/opening_iters/closing_iters must be >= 0.")
    if support_order < 1 or support_k < 1 or min_component_voxels < 0:
        raise ValueError("support order/k must be >= 1; component size must be >= 0.")
    if outlier_ratio is not None and outlier_ratio <= 1:
        raise ValueError("outlier_ratio must be > 1 or None.")
    if outlier_iters < 0 or not 0 <= confidence_quantile < 1:
        raise ValueError("Invalid outlier_iters or confidence_quantile.")

    Xm = X * axis_scale
    keep = _outlier_mask(
        Xm, confidence, outlier_k, outlier_ratio,
        int(outlier_iters), confidence_quantile,
    )
    Xm, confidence = Xm[keep], confidence[keep]
    if len(Xm) < 4:
        raise ValueError("Too few points remain after outlier filtering.")

    tree, radius = _local_radii(Xm, k, confidence, feature_power)
    origin, spacing, shape, _ = _grid(
        Xm, radius, support_scale, voxel_size, voxel_scale, max_voxels
    )
    field = _support_field(
        tree, radius, origin, spacing, shape, support_k, chunk_size,
        support_order=support_order,
    )
    if field_smooth:
        gaussian_filter(field, field_smooth, output=field)

    inside = field <= support_scale
    structure = generate_binary_structure(3, 2)
    if opening_iters:
        inside = binary_opening(
            inside, structure=structure, iterations=int(opening_iters)
        )
    inside = _filter_mask_components(
        inside, int(min_component_voxels), keep_largest_support
    )
    if closing_iters:
        inside = binary_closing(
            inside, structure=structure, iterations=int(closing_iters)
        )
    if fill_holes:
        inside = binary_fill_holes(inside)

    occupied = np.flatnonzero(inside.ravel())
    if not len(occupied):
        raise ValueError("The reconstructed support is empty.")
    point_index = np.rint((Xm - origin) / spacing).astype(int)
    point_index = np.clip(point_index, 0, np.asarray(shape) - 1)
    point_keep = inside[tuple(point_index.T)]
    retained = Xm[point_keep]
    if not len(retained):
        raise ValueError("No input point lies in the retained support.")
    target = (
        int(np.ceil(factor * len(retained)))
        if n_points is None else int(n_points)
    )
    base = len(retained) if include_original else 0
    if target < max(1, base):
        raise ValueError("n_points is smaller than the retained original cloud.")

    n_new = target - base
    if n_new:
        rng = np.random.default_rng(random_state)
        if n_new <= len(occupied):
            flat = rng.choice(occupied, n_new, replace=False)
            n_regular = n_new
        else:
            extra = rng.choice(occupied, n_new - len(occupied), replace=True)
            flat = np.concatenate((occupied, extra))
            n_regular = len(occupied)
        yz = shape[1] * shape[2]
        ix, rem = divmod(flat, yz)
        iy, iz = divmod(rem, shape[2])
        index = np.column_stack((ix, iy, iz)).astype(float)
        if jitter and n_regular < n_new:
            index[n_regular:] += rng.uniform(
                -jitter, jitter, (n_new - n_regular, 3)
            )
        sampled = origin + index * spacing
        dense = np.vstack((retained, sampled)) if include_original else sampled
    else:
        dense = retained.copy()
    return dense / axis_scale


# Cube corners, six tetrahedra around diagonal 0--6, and tetrahedron tables.
_CORNERS = (
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
)
_TETS = (
    (0, 5, 1, 6), (0, 1, 2, 6), (0, 2, 3, 6),
    (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6),
)
_EDGES = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
_TRI_TABLE = (
    (-1, -1, -1, -1, -1, -1),
    (0, 3, 2, -1, -1, -1),
    (0, 1, 4, -1, -1, -1),
    (1, 4, 2, 2, 4, 3),
    (1, 2, 5, -1, -1, -1),
    (0, 3, 5, 0, 5, 1),
    (0, 2, 5, 0, 5, 4),
    (5, 4, 3, -1, -1, -1),
    (3, 4, 5, -1, -1, -1),
    (4, 5, 0, 5, 2, 0),
    (1, 5, 0, 5, 3, 0),
    (5, 2, 1, -1, -1, -1),
    (3, 4, 2, 2, 4, 1),
    (4, 1, 0, -1, -1, -1),
    (2, 3, 0, -1, -1, -1),
    (-1, -1, -1, -1, -1, -1),
)


def _point_spacing(
    X, slice_id, slice_count, max_slices=5, query_samples=512, ref_chunk=8192
):
    """Median of exact sampled nearest-neighbour spacings on large slices."""
    candidates = torch.nonzero(slice_count >= 2).flatten()
    if not len(candidates):
        raise ValueError("At least one z slice must contain two distinct points.")
    chosen = candidates[torch.argsort(slice_count[candidates], descending=True)]
    generator = torch.Generator(device=X.device).manual_seed(0)
    estimates = []

    for sid in chosen:
        points = X[slice_id == sid, :2]
        points = points - points.mean(0)
        eps = 10 * torch.finfo(X.dtype).eps * points.abs().max().clamp_min(1)
        query = points[
            torch.randint(
                len(points),
                (min(len(points), int(query_samples)),),
                generator=generator,
                device=X.device,
            )
        ]
        nearest = torch.full(
            (len(query),), torch.inf, device=X.device, dtype=X.dtype
        )
        for start in range(0, len(points), int(ref_chunk)):
            distance = torch.cdist(query, points[start:start + int(ref_chunk)])
            distance.masked_fill_(distance <= eps, torch.inf)
            nearest = torch.minimum(nearest, distance.amin(1))
        nearest = nearest[torch.isfinite(nearest)]
        if len(nearest):
            estimates.append(nearest.median())
            if len(estimates) >= int(max_slices):
                break

    if not estimates:
        raise ValueError("Cannot estimate a positive point spacing.")
    spacing = torch.stack(estimates).median()
    if not torch.isfinite(spacing) or spacing <= 0:
        raise ValueError("Cannot estimate a positive point spacing.")
    return float(spacing)


def _remove_outliers(X, slice_id, spacing, radius, min_neighbors):
    """Coordinate-compressed spatial hash for local density filtering."""
    if radius is None or min_neighbors <= 0:
        return torch.ones(len(X), dtype=torch.bool, device=X.device)

    # One-spacing cells make repeated coordinates count once instead of letting
    # a duplicated isolated artifact satisfy the density threshold by itself.
    cell = spacing
    center = X[:, :2].median(0).values
    scaled = (X[:, :2].double() - center.double()) / cell
    if scaled.abs().max() >= 2**62:
        raise ValueError("Coordinate range is too large for outlier hashing.")
    q = torch.floor(scaled).long()
    ns = int(slice_id.max()) + 1
    ix, iy = q[:, 0], q[:, 1]
    ux, rx = torch.unique(ix, sorted=True, return_inverse=True)
    uy, ry = torch.unique(iy, sorted=True, return_inverse=True)
    nx, ny = len(ux), len(uy)
    if nx * ny * ns >= torch.iinfo(torch.int64).max:
        raise ValueError("Coordinate range is too large for outlier hashing.")

    code = (slice_id * nx + rx) * ny + ry
    keys, counts = torch.unique(code, sorted=True, return_counts=True)
    counts = counts.clamp_max(2)
    local = torch.zeros_like(slice_id)

    reach = math.ceil(float(radius)) + 1
    for ox in range(-reach, reach + 1):
        for oy in range(-reach, reach + 1):
            cell_distance = math.hypot(
                max(abs(ox) - 1, 0), max(abs(oy) - 1, 0)
            )
            if cell_distance > float(radius):
                continue
            jx, jy = ix + ox, iy + oy
            px = torch.searchsorted(ux, jx).clamp_max(nx - 1)
            py = torch.searchsorted(uy, jy).clamp_max(ny - 1)
            valid = (ux[px] == jx) & (uy[py] == jy)
            query = (slice_id * nx + px) * ny + py
            pos = torch.searchsorted(keys, query).clamp_max(len(keys) - 1)
            hit = valid & (keys[pos] == query)
            local += torch.where(hit, counts[pos], 0)

    return local >= int(min_neighbors) + 1


def _interpolate_z(values, z_obs, z_new, method):
    if len(z_obs) == 1:
        return values
    hi = torch.searchsorted(z_obs, z_new).clamp(1, len(z_obs) - 1)
    lo = hi - 1
    alpha = ((z_new - z_obs[lo]) / (z_obs[hi] - z_obs[lo])).clamp(0, 1)
    tail = (1,) * (values.ndim - 1)
    alpha_view = alpha.reshape((-1,) + tail)
    if method == "linear" or len(z_obs) == 2:
        return torch.lerp(values[lo], values[hi], alpha_view)
    if method != "pchip":
        raise ValueError("z_method must be 'linear' or 'pchip'.")

    h = torch.diff(z_obs)
    delta = torch.diff(values, dim=0) / h.reshape((-1,) + tail)
    deriv = torch.zeros_like(values)
    left, right = delta[:-1], delta[1:]
    same = left * right > 0
    h0 = h[:-1].reshape((-1,) + tail)
    h1 = h[1:].reshape((-1,) + tail)
    w1, w2 = 2 * h1 + h0, h1 + 2 * h0
    safe_left = torch.where(same, left, torch.ones_like(left))
    safe_right = torch.where(same, right, torch.ones_like(right))
    harmonic = (w1 + w2) / (w1 / safe_left + w2 / safe_right)
    deriv[1:-1] = torch.where(same, harmonic, 0)

    first = ((2 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    first = torch.where(first * delta[0] <= 0, 0, first)
    first = torch.where(
        (delta[0] * delta[1] < 0) & (first.abs() > 3 * delta[0].abs()),
        3 * delta[0], first,
    )
    last = ((2 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (
        h[-1] + h[-2]
    )
    last = torch.where(last * delta[-1] <= 0, 0, last)
    last = torch.where(
        (delta[-1] * delta[-2] < 0) & (last.abs() > 3 * delta[-1].abs()),
        3 * delta[-1], last,
    )
    deriv[0], deriv[-1] = first, last

    t = alpha_view
    t2, t3 = t.square(), t.square() * t
    width = (z_obs[hi] - z_obs[lo]).reshape((-1,) + tail)
    return (
        (2 * t3 - 3 * t2 + 1) * values[lo]
        + (t3 - 2 * t2 + t) * width * deriv[lo]
        + (-2 * t3 + 3 * t2) * values[hi]
        + (t3 - t2) * width * deriv[hi]
    )


def _blur_slices(values, sigma_y, sigma_x):
    """Depthwise separable Gaussian blur for (S,Y,X[,C]) tensors."""
    scalar = values.ndim == 3
    out = values[:, None] if scalar else values.permute(0, 3, 1, 2)
    channels = out.shape[1]
    for sigma, axis in ((sigma_y, 2), (sigma_x, 3)):
        if sigma <= 0:
            continue
        radius = max(1, math.ceil(3 * sigma))
        x = torch.arange(
            -radius, radius + 1, device=values.device, dtype=values.dtype
        )
        kernel = torch.exp(-0.5 * (x / sigma).square())
        kernel /= kernel.sum()
        if axis == 2:
            weight = kernel.reshape(1, 1, -1, 1)
            pad = (0, 0, radius, radius)
        else:
            weight = kernel.reshape(1, 1, 1, -1)
            pad = (radius, radius, 0, 0)
        weight = weight.expand(channels, 1, *weight.shape[2:]).contiguous()
        out = nnf.conv2d(
            nnf.pad(out, pad, mode="replicate"), weight, groups=channels
        )
    return out[:, 0] if scalar else out.permute(0, 2, 3, 1)


def _kde_slices(seed, occupied, sigma_y, sigma_x, level, scale, truncation):
    """Turn weighted point splats into a smooth, normalized implicit field."""
    density = _blur_slices(seed, sigma_y, sigma_x)
    reference = []
    for value, mask in zip(density, occupied):
        sample = value[mask]
        reference.append(sample.median() if len(sample) else value.new_tensor(1))
    reference = torch.stack(reference).clamp_min(torch.finfo(seed.dtype).tiny)
    threshold = float(level) * reference[:, None, None]
    field = float(scale) * torch.log(
        density.clamp_min(torch.finfo(seed.dtype).tiny) / threshold
    )
    return field.clamp(-float(truncation), float(truncation))


def _smooth_slices(field, sigma_y, sigma_x):
    """Smooth each section while preserving its positive area."""
    target_area = (field > 0).sum((1, 2))
    out = _blur_slices(field, sigma_y, sigma_x)

    # Gaussian smoothing moves a zero level set inward. Re-centering the level
    # per slice removes that shrinkage while retaining the smoother boundary.
    n_pixels = out.shape[1] * out.shape[2]
    flat = out.flatten(1)
    for i in range(len(out)):
        area = int(target_area[i])
        if area <= 0:
            out[i] -= flat[i].max() + torch.finfo(out.dtype).eps
        elif area < n_pixels:
            k = n_pixels - area
            lower = flat[i].kthvalue(k).values if k else flat[i].min()
            upper = flat[i].kthvalue(k + 1).values
            out[i] -= 0.5 * (lower + upper)
    return out


def _blur_z(values, sigma):
    """Depthwise Gaussian blur along z for (Z,Y,X[,C]) tensors."""
    if sigma <= 0:
        return values
    scalar = values.ndim == 3
    out = values[None, None] if scalar else values.permute(3, 0, 1, 2)[None]
    channels = out.shape[1]
    radius = max(1, math.ceil(3 * sigma))
    x = torch.arange(
        -radius, radius + 1, device=values.device, dtype=values.dtype
    )
    kernel = torch.exp(-0.5 * (x / sigma).square())
    kernel /= kernel.sum()
    weight = kernel.reshape(1, 1, -1, 1, 1).expand(
        channels, 1, -1, 1, 1
    ).contiguous()
    out = nnf.conv3d(
        nnf.pad(out, (0, 0, 0, 0, radius, radius), mode="replicate"),
        weight,
        groups=channels,
    )
    return out[0, 0] if scalar else out[0].permute(1, 2, 3, 0)


def _constrained_smooth3d(field, inside, spacing, iterations, band_width):
    """Higher-order smoothing without changing the inside/outside mask.

    This is a dense CUDA-friendly Jacobi form of the bending-energy idea used
    by PyMCubes constrained smoothing.  Only a narrow band is optimized; the
    original sign is projected back after every iteration, so smoothing cannot
    create a hole or erase a thin component already present in ``inside``.
    """
    if iterations <= 0:
        return field

    base = field
    band = base.abs() <= float(band_width)
    eps = 32 * torch.finfo(base.dtype).eps * base.abs().max().clamp_min(1)
    out = torch.where(inside, base.clamp_min(eps), base.clamp_max(-eps))

    # Squared second derivatives give fourth-order weights.  Normalization
    # avoids overflow for very small physical voxel sizes.
    h = base.new_tensor((spacing[2], spacing[1], spacing[0])).clamp_min(eps)
    weight = h.reciprocal().pow(4)
    weight /= weight.max()
    denominator = 6 * weight.sum()

    for _ in range(int(iterations)):
        p = nnf.pad(out[None, None], (2, 2, 2, 2, 2, 2), mode="replicate")[0, 0]
        z1 = p[1:-3, 2:-2, 2:-2] + p[3:-1, 2:-2, 2:-2]
        y1 = p[2:-2, 1:-3, 2:-2] + p[2:-2, 3:-1, 2:-2]
        x1 = p[2:-2, 2:-2, 1:-3] + p[2:-2, 2:-2, 3:-1]
        z2 = p[:-4, 2:-2, 2:-2] + p[4:, 2:-2, 2:-2]
        y2 = p[2:-2, :-4, 2:-2] + p[2:-2, 4:, 2:-2]
        x2 = p[2:-2, 2:-2, :-4] + p[2:-2, 2:-2, 4:]
        proposal = (
            weight[0] * (4 * z1 - z2)
            + weight[1] * (4 * y1 - y2)
            + weight[2] * (4 * x1 - x2)
        ) / denominator
        candidate = torch.lerp(out, proposal, 0.5)
        candidate = torch.where(
            inside, candidate.clamp_min(eps), candidate.clamp_max(-eps)
        )
        out = torch.where(band, candidate, base)

    return out


def _marching_tetrahedra(volume, origin, spacing, level=0.0):
    """Extract a welded triangle mesh, fully on the current torch device."""
    device, dtype = volume.device, volume.dtype
    nz, ny, nx = volume.shape
    if min(nz, ny, nx) < 2 or not (volume.min() < level < volume.max()):
        raise ValueError("The implicit field has no extractable zero surface.")
    # Avoid the ambiguous case in which the isosurface hits a grid vertex exactly.
    eps = 16 * torch.finfo(dtype).eps * volume.abs().max().clamp_min(1)
    volume = torch.where(volume == level, volume + eps, volume)

    corner_id = torch.tensor(_CORNERS, device=device, dtype=torch.long)
    corners = corner_id.to(dtype)
    tets = torch.tensor(_TETS, device=device, dtype=torch.long)
    edges = torch.tensor(_EDGES, device=device, dtype=torch.long)
    table = torch.tensor(_TRI_TABLE, device=device, dtype=torch.long)
    bits = torch.tensor((1, 2, 4, 8), device=device)
    origin = torch.as_tensor(origin, device=device, dtype=dtype)
    spacing = torch.as_tensor(spacing, device=device, dtype=dtype)
    area = max(1, (ny - 1) * (nx - 1))
    depth = max(1, min(nz - 1, 2_000_000 // area))
    triangles, triangle_keys = [], []
    n_grid_points = nz * ny * nx

    for z0 in range(0, nz - 1, depth):
        z1 = min(z0 + depth, nz - 1)
        c = (
            volume[z0:z1, :-1, :-1], volume[z0:z1, :-1, 1:],
            volume[z0:z1, 1:, 1:], volume[z0:z1, 1:, :-1],
            volume[z0 + 1:z1 + 1, :-1, :-1],
            volume[z0 + 1:z1 + 1, :-1, 1:],
            volume[z0 + 1:z1 + 1, 1:, 1:],
            volume[z0 + 1:z1 + 1, 1:, :-1],
        )
        low, high = c[0], c[0]
        for value in c[1:]:
            low, high = torch.minimum(low, value), torch.maximum(high, value)
        ids = ((low < level) & (high >= level)).nonzero()
        if not len(ids):
            continue

        for begin in range(0, len(ids), 250_000):
            iz, iy, ix = ids[begin:begin + 250_000].unbind(1)
            cube_values = torch.stack([v[iz, iy, ix] for v in c], 1)
            base_id = torch.stack((ix, iy, iz + z0), 1)
            grid_id = base_id[:, None] + corner_id[None]
            grid_id = (
                (grid_id[:, :, 2] * ny + grid_id[:, :, 1]) * nx
                + grid_id[:, :, 0]
            )
            cube_points = origin + (
                base_id[:, None].to(dtype) + corners[None]
            ) * spacing

            tet_values = cube_values[:, tets]
            cases = ((tet_values >= level).long() * bits).sum(2)
            active = ((cases != 0) & (cases != 15)).nonzero()
            if not len(active):
                continue
            cube_index, tet_index = active.unbind(1)
            tet = tets[tet_index]
            value = cube_values[cube_index[:, None], tet]
            point = cube_points[cube_index[:, None], tet]
            point_id = grid_id[cube_index[:, None], tet]
            case = cases[cube_index, tet_index]
            va, vb = value[:, edges[:, 0]], value[:, edges[:, 1]]
            pa, pb = point[:, edges[:, 0]], point[:, edges[:, 1]]
            den = vb - va
            ratio = torch.where(
                den.abs() > torch.finfo(dtype).eps,
                (level - va) / den,
                torch.full_like(den, 0.5),
            ).clamp(0, 1)
            edge_point = pa + ratio[..., None] * (pb - pa)
            ida, idb = point_id[:, edges[:, 0]], point_id[:, edges[:, 1]]
            edge_key = (
                torch.minimum(ida, idb) * n_grid_points
                + torch.maximum(ida, idb)
            )
            row = table[case]

            for start in (0, 3):
                use = row[:, start] >= 0
                if use.any():
                    edge_id = row[use, start:start + 3]
                    tri = edge_point[use].gather(
                        1, edge_id[..., None].expand(-1, -1, 3)
                    )[:, (0, 2, 1)]
                    key = edge_key[use].gather(1, edge_id)[:, (0, 2, 1)]
                    triangles.append(tri)
                    triangle_keys.append(key)

    if not triangles:
        raise ValueError("No triangles were generated.")

    raw = torch.cat(triangles).reshape(-1, 3)
    raw_key = torch.cat(triangle_keys).reshape(-1)
    _, inverse = torch.unique(raw_key, sorted=True, return_inverse=True)
    n_vertices = int(inverse.max()) + 1
    vertices = torch.zeros((n_vertices, 3), device=device, dtype=dtype)
    vertices.index_add_(0, inverse, raw)
    count = torch.bincount(inverse, minlength=n_vertices).to(dtype)
    vertices /= count[:, None]
    faces = inverse.reshape(-1, 3)
    faces = faces[
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 2] != faces[:, 0])
    ]

    return vertices, faces


def _filter_mesh_components(vertices, faces, threshold):
    """Remove small islands by relative (<1) or absolute face threshold."""
    if threshold <= 0 or len(faces) == 0:
        return vertices, faces, 0

    parent = torch.arange(len(vertices), device=faces.device)
    block = 2_000_000
    for _ in range(64):
        parent = parent[parent]
        updated = parent.clone()
        for a, b in ((0, 1), (1, 2), (2, 0)):
            for start in range(0, len(faces), block):
                edge = faces[start:start + block]
                ra, rb = parent[edge[:, a]], parent[edge[:, b]]
                high, low = torch.maximum(ra, rb), torch.minimum(ra, rb)
                updated.scatter_reduce_(
                    0, high, low, reduce="amin", include_self=True
                )
        updated = updated[updated]
        if torch.equal(updated, parent):
            parent = updated
            break
        parent = updated

    for _ in range(4):
        parent = parent[parent]
    face_root = parent[faces[:, 0]]
    face_count = torch.bincount(face_root, minlength=len(vertices))
    largest = face_count.argmax()
    min_faces = (
        math.ceil(float(face_count[largest]) * float(threshold))
        if threshold < 1
        else math.ceil(float(threshold))
    )
    keep = (face_count[face_root] >= min_faces) | (face_root == largest)
    component_count = int((face_count > 0).sum())
    if keep.all():
        return vertices, faces, 0
    vertex_id = torch.arange(len(vertices), device=faces.device)
    kept_component = (face_count >= min_faces) | (vertex_id == largest)
    kept_count = int((kept_component & (face_count > 0)).sum())

    faces = faces[keep]
    used_mask = torch.zeros(len(vertices), dtype=torch.bool, device=faces.device)
    used_mask[faces] = True
    used = torch.nonzero(used_mask).flatten()
    remap = torch.full(
        (len(vertices),), -1, dtype=torch.long, device=faces.device
    )
    remap[used] = torch.arange(len(used), device=faces.device)
    return vertices[used], remap[faces], component_count - kept_count


def _smooth_mesh(vertices, faces, iterations, strength):
    """Taubin smoothing: smooth without the shrinkage of Laplacian smoothing."""
    if iterations <= 0:
        return vertices
    edge = torch.cat(
        (faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]), dim=0
    )
    lo, hi = edge.min(1).values, edge.max(1).values
    key = torch.unique(lo * len(vertices) + hi, sorted=False)
    edge = torch.stack((key // len(vertices), key % len(vertices)), 1)
    edge = torch.cat((edge, edge.flip(1)), dim=0)
    src, dst = edge[:, 0], edge[:, 1]
    degree = torch.bincount(src, minlength=len(vertices)).clamp_min(1)

    out = vertices
    for _ in range(int(iterations)):
        for step in (float(strength), -1.06 * float(strength)):
            mean = torch.zeros_like(out)
            mean.index_add_(0, src, out[dst])
            mean /= degree[:, None]
            out = out + step * (mean - out)
    return out


def _sample_volume(volume, vertices, origin, spacing, chunk=262_144):
    """Clamped, channel-aware trilinear sampling through grid_sample."""
    scalar = volume.ndim == 3
    if scalar:
        volume = volume[..., None]
    nz, ny, nx, _ = volume.shape
    data = volume.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    q = (vertices - vertices.new_tensor(origin)) / vertices.new_tensor(spacing)
    grid = torch.stack(
        (
            2 * q[:, 0] / (nx - 1) - 1,
            2 * q[:, 1] / (ny - 1) - 1,
            2 * q[:, 2] / (nz - 1) - 1,
        ),
        1,
    ).clamp(-1, 1)
    sampled = []
    for start in range(0, len(grid), int(chunk)):
        part = grid[start:start + int(chunk)].reshape(1, 1, 1, -1, 3)
        value = nnf.grid_sample(
            data, part, mode="bilinear", padding_mode="border",
            align_corners=True,
        )
        sampled.append(value[0, :, 0, 0].T)
    result = torch.cat(sampled)
    return result[:, 0] if scalar else result


def _project_surface(vertices, volume, origin, spacing, iterations):
    """Project smoothed vertices back to the implicit zero level-set."""
    if iterations <= 0:
        return vertices
    dz, dy, dx = float(spacing[2]), float(spacing[1]), float(spacing[0])
    p = nnf.pad(volume[None, None], (1, 1, 1, 1, 1, 1), mode="replicate")[0, 0]
    gradient = torch.stack(
        (
            (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, :-2]) / (2 * dx),
            (p[1:-1, 2:, 1:-1] - p[1:-1, :-2, 1:-1]) / (2 * dy),
            (p[2:, 1:-1, 1:-1] - p[:-2, 1:-1, 1:-1]) / (2 * dz),
        ),
        -1,
    )
    out = vertices
    max_step = 0.5 * min(dx, dy, dz)
    eps = torch.finfo(vertices.dtype).eps
    for _ in range(int(iterations)):
        value = _sample_volume(volume, out, origin, spacing)
        grad = _sample_volume(gradient, out, origin, spacing)
        step = value[:, None] * grad / grad.square().sum(1, keepdim=True).clamp_min(eps)
        length = torch.linalg.vector_norm(step, dim=1, keepdim=True)
        step *= (max_step / length.clamp_min(eps)).clamp_max(1)
        out = out - step
    return out


def _build_one(
    X,
    features,
    feature_was_1d,
    feature_mode,
    voxel_size,
    z_step,
    z_method,
    grid_scale,
    radius_scale,
    weight_power,
    weight_floor,
    outlier_radius,
    min_neighbors,
    opening,
    closing,
    truncation,
    field_smooth,
    density_level,
    surface_smooth,
    component_threshold,
    mesh_smooth,
    smooth_strength,
    project_iterations,
    margin,
    max_voxels,
):
    z_obs, slice_id, slice_count = torch.unique(
        X[:, 2], sorted=True, return_inverse=True, return_counts=True
    )
    point_spacing = _point_spacing(X, slice_id, slice_count)
    keep = _remove_outliers(
        X, slice_id, point_spacing, outlier_radius, min_neighbors
    )
    X = X[keep]
    features = None if features is None else features[keep]
    if len(X) < 4:
        raise ValueError("Too few points remain after outlier filtering.")

    z_obs, slice_id = torch.unique(X[:, 2], sorted=True, return_inverse=True)
    if features is None or feature_mode == "attribute":
        weight = torch.ones(len(X), device=X.device, dtype=X.dtype)
    elif feature_mode == "probability":
        if features[:, 0].max() > 1:
            raise ValueError("probability features must lie in [0,1].")
        weight = float(weight_floor) + (1 - float(weight_floor)) * features[:, 0]
    else:
        value = torch.log1p(features[:, 0])
        lo, hi = torch.quantile(value, value.new_tensor((0.01, 0.99)))
        weight = torch.ones_like(value) if hi <= lo else (value - lo) / (hi - lo)
        weight = float(weight_floor) + (1 - float(weight_floor)) * weight.clamp(0, 1)

    if voxel_size is None:
        dx = dy = float(grid_scale) * point_spacing
        dz = None
    else:
        size = (
            torch.as_tensor(voxel_size, dtype=torch.float64)
            .flatten().detach().cpu().tolist()
        )
        if len(size) == 1:
            dx = dy = dz = float(size[0])
        elif len(size) == 3:
            dx, dy, dz = map(float, size)
        else:
            raise ValueError("voxel_size must be a scalar or (dx,dy,dz).")
    if z_step is not None:
        dz = float(z_step)
    elif dz is None:
        dz = max(dx, dy) if len(z_obs) == 1 else min(
            float(torch.diff(z_obs).median()) / 4, max(dx, dy)
        )
    if min(dx, dy, dz) <= 0:
        raise ValueError("voxel sizes must be positive.")

    smooth = (
        torch.as_tensor(field_smooth, dtype=torch.float64)
        .flatten().detach().cpu()
    )
    if smooth.numel() == 1:
        smooth = smooth.repeat(3)
    if smooth.numel() != 3 or torch.any(smooth < 0):
        raise ValueError("field_smooth must be a non-negative scalar or (z,y,x).")
    ro = math.ceil(float(opening) * point_spacing / min(dx, dy))
    rc = math.ceil(float(closing) * point_spacing / min(dx, dy))
    sigma_y = float(smooth[1]) * point_spacing / dy
    sigma_x = float(smooth[2]) * point_spacing / dx
    kde_y = float(radius_scale) * point_spacing / dy
    kde_x = float(radius_scale) * point_spacing / dx
    blur_pad = 3 * (
        float(radius_scale) + max(float(smooth[1]), float(smooth[2]))
    ) * point_spacing
    pad = (
        float(margin) * point_spacing
        + (ro + rc) * max(dx, dy)
        + blur_pad
    )
    xmin, ymin = (X[:, :2].amin(0) - pad).tolist()
    xmax, ymax = (X[:, :2].amax(0) + pad).tolist()
    zmin, zmax = float(z_obs[0]), float(z_obs[-1])
    nx = math.ceil((xmax - xmin) / dx) + 1
    ny = math.ceil((ymax - ymin) / dy) + 1
    nz = 1 if len(z_obs) == 1 else math.ceil((zmax - zmin) / dz) + 1

    z_grid = torch.linspace(zmin, zmax, nz, device=X.device, dtype=X.dtype)
    if nz > 1:
        dz = float(z_grid[1] - z_grid[0])
    z_scale = dz if len(z_obs) == 1 else float(torch.diff(z_obs).median())
    sigma_z = float(smooth[0]) * z_scale / dz
    cap_layers = max(1, math.ceil(3 * sigma_z) + 1)
    slice_voxels = nx * ny * len(z_obs)
    volume_voxels = nx * ny * (nz + 2 * cap_layers)
    if max(slice_voxels, volume_voxels) > int(max_voxels):
        raise MemoryError("Grid exceeds max_voxels; increase voxel_size or z_step.")
    ix = torch.round((X[:, 0] - xmin) / dx).long().clamp(0, nx - 1)
    iy = torch.round((X[:, 1] - ymin) / dy).long().clamp(0, ny - 1)
    flat_id = (slice_id * ny + iy) * nx + ix
    shape = (len(z_obs), ny, nx)
    count = torch.zeros(math.prod(shape), device=X.device, dtype=X.dtype)
    count.scatter_add_(0, flat_id, torch.ones_like(weight))
    wsum = torch.zeros_like(count)
    wsum.scatter_add_(0, flat_id, weight)
    occupied_mask = count.reshape(shape) > 0
    seed = torch.where(
        count > 0,
        (wsum / count.clamp_min(1)).pow(float(weight_power)),
        0,
    ).reshape(shape)

    occupied = occupied_feature = None
    if features is not None:
        channels = features.shape[1]
        occupied, inverse = torch.unique(
            flat_id, sorted=True, return_inverse=True
        )
        fsum = torch.zeros(
            (len(occupied), channels), device=X.device, dtype=X.dtype
        )
        fsum.index_add_(0, inverse, features)
        occupied_count = torch.bincount(
            inverse, minlength=len(occupied)
        ).to(X.dtype)
        occupied_feature = fsum / occupied_count[:, None]

    trunc = float(truncation) * point_spacing
    field = _kde_slices(
        seed,
        occupied_mask,
        kde_y,
        kde_x,
        density_level,
        point_spacing,
        trunc,
    )

    if ro:
        field = -nnf.max_pool2d(
            -field[:, None], 2 * ro + 1, 1, ro
        )
        field = nnf.max_pool2d(field, 2 * ro + 1, 1, ro)[:, 0]
    if rc:
        # Closing intentionally fills gaps in the unsmoothed occupancy field.
        field = nnf.max_pool2d(
            field[:, None], 2 * rc + 1, 1, rc
        )
        field = -nnf.max_pool2d(-field, 2 * rc + 1, 1, rc)[:, 0]
    slice_inside = field > 0
    field = _smooth_slices(field, sigma_y, sigma_x)
    sign_eps = 32 * torch.finfo(field.dtype).eps * field.abs().max().clamp_min(1)
    field = torch.where(
        slice_inside, field.clamp_min(sign_eps), field.clamp_max(-sign_eps)
    )
    core = _interpolate_z(field, z_obs, z_grid, z_method)
    cap = torch.full(
        (cap_layers, ny, nx), -trunc, device=X.device, dtype=X.dtype
    )
    base_volume = torch.cat((cap, core, cap))
    inside = base_volume > 0
    volume = _blur_z(base_volume, sigma_z)
    volume = _constrained_smooth3d(
        volume,
        inside,
        (dx, dy, dz),
        surface_smooth,
        float(truncation) * point_spacing,
    )
    volume[0], volume[-1] = -trunc, -trunc
    origin = (xmin, ymin, zmin - cap_layers * dz)
    spacing = (dx, dy, dz)
    vertices, faces = _marching_tetrahedra(volume, origin, spacing)
    vertices, faces, removed_components = _filter_mesh_components(
        vertices, faces, component_threshold
    )
    vertices = _smooth_mesh(vertices, faces, mesh_smooth, smooth_strength)
    vertices = _project_surface(
        vertices, volume, origin, spacing, project_iterations
    )

    mesh = {
        "vertices": vertices,
        "faces": faces,
        "removed_components": removed_components,
    }
    if occupied_feature is not None:
        denominator = _blur_slices(
            occupied_mask.to(X.dtype), kde_y, kde_x
        ).clamp_min(torch.finfo(X.dtype).eps)
        parts = []
        for first in range(0, occupied_feature.shape[1], 8):
            channels = min(8, occupied_feature.shape[1] - first)
            feature_slices = torch.zeros(
                (*shape, channels), device=X.device, dtype=X.dtype
            )
            feature_slices.reshape(-1, channels).index_copy_(
                0, occupied, occupied_feature[:, first:first + channels]
            )
            feature_slices = _blur_slices(
                feature_slices, kde_y, kde_x
            ) / denominator[..., None]
            feature_slices = _blur_slices(
                feature_slices, sigma_y, sigma_x
            )
            feature_core = _interpolate_z(
                feature_slices, z_obs, z_grid, z_method
            )
            feature_volume = torch.cat(
                (
                    feature_core[:1].expand(cap_layers, -1, -1, -1),
                    feature_core,
                    feature_core[-1:].expand(cap_layers, -1, -1, -1),
                )
            )
            feature_volume = _blur_z(feature_volume, sigma_z)
            parts.append(_sample_volume(
                feature_volume, vertices, origin, spacing
            ))
        sampled = torch.cat(parts, 1)
        mesh["features"] = sampled[:, 0] if feature_was_1d else sampled
    mesh["removed_points"] = int((~keep).sum())
    return mesh


@torch.no_grad()
def pc_mesh(
    X,
    labels=None,
    features=None,
    feature_mode="weight",
    device=None,
    voxel_size=None,
    z_step=None,
    z_method="pchip",
    grid_scale=1.0,
    radius_scale=1.8,
    weight_power=0.5,
    weight_floor=0.25,
    outlier_radius=2.5,
    min_neighbors=4,
    opening=0.0,
    closing=0.5,
    truncation=4.0,
    field_smooth=(0.5, 1.5, 1.5),
    mesh_smooth=5,
    smooth_strength=0.5,
    margin=3.0,
    min_label_points=20,
    max_voxels=50_000_000,
    component_threshold=0.001,
    density_level=0.3,
    surface_smooth=30,
    project_iterations=2,
):
    """Reconstruct one mesh, or one independent mesh per label.

    ``X`` is (N,3). A weight/probability feature must be (N,) or (N,1);
    ``feature_mode='attribute'`` also accepts a general (N,C) tensor.
    ``radius_scale`` is the KDE bandwidth divided by median point spacing;
    ``density_level`` is its normalized isosurface threshold.
    ``field_smooth=(z,y,x)`` controls the additional field smoothing.
    ``surface_smooth`` is the number of sign-constrained higher-order smoothing
    iterations; it smooths the zero surface without changing its voxel mask.
    ``weight_power`` controls how strongly probability/weight changes the KDE.
    ``project_iterations`` projects the lightly smoothed mesh back to the zero
    level-set. Features are returned as (V,) or (V,C).
    ``component_threshold`` removes disconnected islands. A value below 1 is
    relative to the largest component's face count; a value >=1 is an absolute
    face count. The largest component is always retained; 0 disables it.
    All returned tensors stay on ``device``; CUDA is selected automatically.
    """
    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    if X.ndim != 2 or X.shape[1] != 3 or len(X) < 4:
        raise ValueError("X must have shape (N,3), N >= 4.")
    if not torch.isfinite(X).all():
        raise ValueError("X contains NaN or Inf.")

    if feature_mode not in {"weight", "probability", "attribute"}:
        raise ValueError("feature_mode must be weight, probability, or attribute.")
    feature_was_1d = False
    if features is not None:
        features = torch.as_tensor(features, dtype=X.dtype, device=device)
        feature_was_1d = features.ndim == 1
        if feature_was_1d:
            features = features[:, None]
        if features.ndim != 2 or len(features) != len(X):
            raise ValueError("features must have shape (N,) or (N,C).")
        if not torch.isfinite(features).all():
            raise ValueError("features contain NaN or Inf.")
        if feature_mode != "attribute":
            if features.shape[1] != 1:
                raise ValueError("weight/probability mode requires one feature column.")
            if features.min() < 0:
                raise ValueError("weight/probability features must be non-negative.")
    if not 0 <= weight_floor <= 1:
        raise ValueError("weight_floor must lie in [0,1].")
    if not 0 <= smooth_strength <= 0.7:
        raise ValueError(
            "smooth_strength must lie in [0,0.7]; larger values can fold faces."
        )
    if z_method not in {"linear", "pchip"}:
        raise ValueError("z_method must be linear or pchip.")
    if grid_scale <= 0 or radius_scale <= 0 or weight_power <= 0 or truncation <= 0:
        raise ValueError(
            "grid_scale, radius_scale, weight_power, and truncation must be positive."
        )
    if margin < 0 or opening < 0 or closing < 0:
        raise ValueError("margin, opening, and closing must be non-negative.")
    if outlier_radius is not None and outlier_radius <= 0:
        raise ValueError("outlier_radius must be positive or None.")
    if (
        min_neighbors < 0
        or component_threshold < 0
        or surface_smooth < 0
        or project_iterations < 0
        or mesh_smooth < 0
        or min_label_points < 1
    ):
        raise ValueError("count and iteration parameters must be non-negative.")
    if max_voxels < 8:
        raise ValueError("max_voxels is too small.")
    if not 0 < density_level < 1:
        raise ValueError("density_level must lie in (0,1).")

    args = (
        feature_was_1d, feature_mode, voxel_size, z_step, z_method, grid_scale,
        radius_scale, weight_power, weight_floor, outlier_radius, min_neighbors,
        opening, closing, truncation, field_smooth, density_level,
        surface_smooth,
        component_threshold,
        mesh_smooth, smooth_strength, project_iterations, margin, max_voxels,
    )
    if labels is None:
        return _build_one(X, features, *args)

    if len(labels) != len(X):
        raise ValueError("labels must have shape (N,).")
    try:
        label_tensor = torch.as_tensor(labels, device=device)
        if label_tensor.ndim != 1:
            raise ValueError("labels must have shape (N,).")
        values = torch.unique(label_tensor)
        groups = ((v.item(), label_tensor == v) for v in values)
    except (TypeError, ValueError):
        values = labels.tolist() if hasattr(labels, "tolist") else list(labels)
        lookup, keys, codes = {}, [], []
        for value in values:
            if value not in lookup:
                lookup[value] = len(keys)
                keys.append(value)
            codes.append(lookup[value])
        codes = torch.tensor(codes, device=device)
        groups = ((key, codes == i) for i, key in enumerate(keys))

    meshes = {}
    for label, select in groups:
        if int(select.sum()) < int(min_label_points):
            continue
        meshes[label] = _build_one(
            X[select], None if features is None else features[select], *args
        )
    if not meshes:
        raise ValueError("No label contains enough points to form a mesh.")
    return meshes


def save_meshes(meshes, output, fmt="vtp"):
    """Save one mesh or a label->mesh dictionary."""
    output = Path(output)
    if isinstance(meshes, pv.PolyData):
        path = output if output.suffix else output.with_suffix(f".{fmt}")
        path.parent.mkdir(parents=True, exist_ok=True)
        meshes.save(path)
        return path
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    for i, (label, mesh) in enumerate(meshes.items()):
        name = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(label)).strip("_")
        path = output / f"{name or f'label_{i}'}.{fmt}"
        mesh.save(path)
        paths[label] = path
    return paths


def save_mesh(mesh, filename):
    """Save one mesh as OBJ, or preserve tensors/features as a PT file."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".pt":
        torch.save(
            {k: v.detach().cpu() if torch.is_tensor(v) else v
             for k, v in mesh.items()},
            path,
        )
        return path
    if path.suffix.lower() != ".obj":
        raise ValueError("filename must end with .obj or .pt.")

    vertices = mesh["vertices"].detach().cpu().tolist()
    faces = (mesh["faces"].detach().cpu() + 1).tolist()
    with path.open("w", encoding="utf-8") as f:
        f.writelines(f"v {x:.8g} {y:.8g} {z:.8g}\n" for x, y, z in vertices)
        f.writelines(f"f {a} {b} {c}\n" for a, b, c in faces)
    return path
