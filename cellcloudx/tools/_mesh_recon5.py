"""Occupied-volume reconstruction from XYZ volume samples.

Public APIs: clay_mesh, clay_slice_mesh, grid_point_cloud, grid_to_mesh,
points_to_image, image_to_nrrd. Geometry and label evidence are independent.
Required: NumPy, SciPy >=1.6. Mesh/surface processing: PyVista and VTK.
CUDA is optional for Gaussian filtering and torchmcubes; other stages use CPU.
XYZ coordinates retain their input units. Labels are categories, not probabilities.
Shared interfaces use one geometry before splitting; do not smooth split meshes.
Volume preservation is approximate at tied grid levels and excludes mesh motion.
No coordinate-only method guarantees anatomical topology or recovers every cavity."""

from pathlib import Path
import warnings
import math
import gzip

import numpy as np
from scipy import ndimage as ndi, sparse
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components

class _SurfaceUpdateError(RuntimeError):
    """A geometric surface update failed a local or rasterization check."""


_GRID_UNSET = object()

__version__ = "2026.09.14"
# __all__ = ["clay_slice_mesh", "clay_mesh", "grid_point_cloud", "grid_to_mesh",
#            "points_to_image", "image_to_nrrd"]


def _alias(value, alias, default, name, old_name):
    """Resolve one legacy keyword without silently overriding either value."""
    if value is not None and alias is not None:
        raise ValueError(f"Specify only {name} or its legacy alias {old_name}.")
    return default if value is None and alias is None else (alias if value is None else value)


def _gaussian(field, sigma, device="cpu"):
    """Separable Gaussian, with the same nearest boundary rule on CPU/CUDA."""
    field = np.asarray(field)
    if not np.issubdtype(field.dtype, np.floating):
        field = field.astype(np.float32)
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (field.ndim,))
    if not np.isfinite(sigma).all() or np.any(sigma < 0):
        raise ValueError("Gaussian sigma must be finite and nonnegative.")
    if str(device).split(":")[0] not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or a torch CUDA device.")
    if str(device).split(":")[0] == "cpu":
        return ndi.gaussian_filter(field, sigma, mode="nearest", truncate=3.0)
    if field.ndim != 3:
        raise ValueError("CUDA Gaussian filtering requires a 3D field.")
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        v = torch.as_tensor(np.ascontiguousarray(field), device=device)[None, None]
        for axis, s in enumerate(sigma):
            r = int(3.0 * s + 0.5)
            if r == 0:
                continue
            t = torch.arange(-r, r + 1, device=device, dtype=v.dtype)
            kernel = torch.exp(-0.5 * (t / s).square())
            kernel /= kernel.sum()
            shape = [1, 1, 1, 1, 1]
            shape[axis + 2] = 2 * r + 1
            pad = [0] * 6
            pad[2 * (2 - axis):2 * (2 - axis) + 2] = [r, r]
            v = F.conv3d(F.pad(v, pad, mode="replicate"), kernel.reshape(shape))
        return v[0, 0].cpu().numpy()


def _gaussian_halo(sigma, scales, spacing):
    """Physical reach of the actual truncate=3 discrete Gaussian kernels."""
    radii = np.floor(3.0 * np.asarray(scales)[:, None] * sigma / spacing + 0.5)
    return radii.sum(axis=0) * spacing


def _denoise(points, groups, levels, axis, outlier, max_gap, k, base_spacing=None, *, workers=-1):
    """Estimate spacing, then simultaneously reject local/cross-section outliers."""
    plane = [a for a in range(3) if a != axis] if axis is not None else [0, 1, 2]
    order = np.argsort(groups, kind="stable")
    blocks = np.split(order, np.flatnonzero(np.diff(groups[order])) + 1)
    keep = np.ones(len(points), dtype=bool)
    estimates = []
    for ids in blocks:
        if len(ids) < 4:
            continue
        ki = min(int(k), len(ids) - 1)
        distances = cKDTree(points[ids][:, plane]).query(
            points[ids][:, plane], k=[ki + 1], workers=workers
        )[0][:, 0]
        med = np.median(distances)
        estimates.append(med / ki ** (1.0 / len(plane)))
        if outlier > 0:
            spread = max(1.4826 * np.median(abs(distances - med)), 0.1 * med)
            keep[ids] = distances <= med + outlier * spread
    if not estimates or np.median(estimates) <= 0:
        raise ValueError("Not enough distinct points to estimate sampling spacing.")
    d = float(np.median(estimates) if base_spacing is None else base_spacing)
    if axis is None or outlier <= 0:
        return keep, d

    # All queries use the same pre-rejection trees: no sequential slice bias.
    ids_by_slice = {int(groups[b[0]]): b[keep[b]] for b in blocks if keep[b].any()}
    trees = {g: cKDTree(points[ids][:, plane]) for g, ids in ids_by_slice.items()}
    keys = sorted(trees)
    for j, g in enumerate(keys):
        ids = ids_by_slice[g]
        support = np.full(len(ids), np.inf)
        used = False
        for neighbor in keys[max(0, j - 1):j] + keys[j + 1:j + 2]:
            dz = abs(levels[g] - levels[neighbor])
            if dz > max_gap:
                continue
            distance = trees[neighbor].query(points[ids][:, plane], workers=workers)[0]
            support = np.minimum(support, distance / np.hypot(d, 0.35 * dz))
            used = True
        if used:
            keep[ids] &= support <= outlier
    return keep, d


def _axial_weights(z, levels, sigma):
    """Sparse Gaussian regression; sigma=0 gives physical-distance linear interpolation."""
    rows, cols, values = [], [], []
    n = len(levels)
    widths = np.ones(n) if n == 1 else np.r_[
        levels[1] - levels[0], (levels[2:] - levels[:-2]) / 2,
        levels[-1] - levels[-2]
    ]
    for i, value in enumerate(z):
        right = np.searchsorted(levels, value)
        pair = np.unique(np.clip([right - 1, right], 0, n - 1))
        if sigma == 0:
            ids = pair
            if len(ids) == 1:
                w = np.ones(1)
            else:
                a = np.clip((value - levels[ids[0]]) / np.diff(levels[ids])[0], 0, 1)
                w = np.array([1 - a, a])
        else:
            lo, hi = np.searchsorted(levels, [value - 3 * sigma, value + 3 * sigma])
            ids = np.union1d(np.arange(lo, hi), pair)
            logits = -0.5 * ((value - levels[ids]) / sigma) ** 2
            w = np.exp(logits - logits.max()) * widths[ids]
            w /= w.sum()
        rows.extend([i] * len(ids))
        cols.extend(ids)
        values.extend(w)
    return sparse.csr_matrix((np.asarray(values, np.float32), (rows, cols)),
                             shape=(len(z), n))


def _marching_cubes(field, origin, spacing, backend="auto", device="cpu"):
    """Extract a positive-inside XYZ field, with outward winding per component."""
    field = np.asarray(field, dtype=np.float32)
    origin = np.broadcast_to(np.asarray(origin, dtype=float), (3,))
    spacing = np.broadcast_to(np.asarray(spacing, dtype=float), (3,))
    if field.ndim != 3 or min(field.shape) < 2 or not np.isfinite(field).all():
        raise ValueError("marching cubes requires a finite 3D field, shape >=2 on every axis.")
    if not np.isfinite(origin).all() or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("origin must be finite and spacing finite and positive.")
    lo, hi = float(field.min()), float(field.max())
    if not (lo < 0 < hi):
        raise ValueError(
            f"Marching cubes requires both inside (positive) and outside (negative) samples; "
            f"field range is [{lo:g}, {hi:g}]. Zero samples alone do not define a closed boundary.")
    if backend not in {"auto", "pymcubes", "skimage", "torchmcubes", "lewiner", "lorensen", "vtk"}:
        raise ValueError("Unknown marching-cubes backend.")
    if backend == "auto":
        from importlib import import_module
        backend = "vtk"
        for package, candidate in (("mcubes", "pymcubes"), ("skimage.measure", "skimage")):
            try:
                import_module(package)
            except ImportError:
                continue
            backend = candidate
            break
    if backend == "pymcubes":
        import mcubes
        vertices, faces = mcubes.marching_cubes(np.ascontiguousarray(field), 0.0)
    elif backend in {"skimage", "lewiner", "lorensen"}:
        from skimage.measure import marching_cubes
        vertices, faces, _, _ = marching_cubes(
            field, level=0.0, gradient_direction="descent", allow_degenerate=False,
            method="lorensen" if backend == "lorensen" else "lewiner")
    elif backend == "vtk":
        import pyvista as pv
        grid = pv.ImageData(dimensions=field.shape)
        # VTK's x index varies fastest; this XYZ array must use Fortran order.
        grid.point_data["field"] = field.ravel(order="F")
        surface = grid.contour([0.0], scalars="field", method="flying_edges").triangulate()
        vertices = np.asarray(surface.points, float)
        faces = np.asarray(surface.faces).reshape(-1, 4)[:, 1:]
    else:
        import torch
        from torchmcubes import marching_cubes
        # torchmcubes uses tensor[D,H,W] and returns coordinates in (W,H,D).
        volume = torch.as_tensor(np.ascontiguousarray(field.transpose(2, 1, 0)),
                                 dtype=torch.float32, device=device)
        vertices, faces = marching_cubes(volume, 0.0)
        vertices, faces = vertices.cpu().numpy(), faces.cpu().numpy()
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise RuntimeError("Marching-cubes backend returned invalid vertex coordinates.")
    if faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer):
        raise RuntimeError("Marching-cubes backend returned invalid triangle indices.")
    faces = faces.astype(np.int64, copy=False)
    if not len(faces):
        raise RuntimeError("Marching cubes produced no triangles despite a sign-changing field.")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise RuntimeError("Marching-cubes triangle index is outside the vertex array.")

    def valid_triangles(v, f):
        # Relative roundoff tolerance in index space, not a physical-distance
        # weld. Legitimate short edges and thin walls retain their own vertices.
        first = v[f[:, 1]] - v[f[:, 0]]
        second = v[f[:, 2]] - v[f[:, 0]]
        area = np.linalg.norm(np.cross(first, second), axis=1)
        scale = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
        return area > 32 * np.finfo(float).eps * scale

    valid = valid_triangles(vertices, faces)
    if not valid.all():
        # A zero-valued grid node may be emitted once per incident grid edge.
        # Only exact duplicate locations are joined; no positive weld tolerance
        # can accidentally merge two nearby sheets.
        vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
        faces = inverse[faces[valid]]
        if len(faces):
            faces = faces[valid_triangles(vertices, faces)]
    if not len(faces):
        raise RuntimeError("Marching cubes produced only degenerate triangles.")
    used, inverse = np.unique(faces, return_inverse=True)
    vertices, faces = vertices[used], inverse.reshape(-1, 3)
    faces = _mc_orient_faces(vertices, faces, field, spacing)
    vertices = vertices * spacing + origin
    if not np.isfinite(vertices).all():
        raise RuntimeError("Marching-cubes world coordinates overflowed; rescale the input coordinates.")
    return vertices, faces


def _stack_field(points, confidence, groups, levels, known_empty, axis,
                 origin, step, shape, d, sigma, radius, outlier, max_gap, method, *, workers=-1):
    """Shared physical-z interpolation; observations are occupancy or 2D SDF."""
    levels = np.asarray(levels, dtype=float)
    if levels.ndim != 1 or len(levels) < 2 or not np.isfinite(levels).all() or np.any(np.diff(levels) <= 0):
        raise ValueError("Slice levels must contain at least two finite, strictly increasing positions.")
    plane = [a for a in range(3) if a != axis]
    xy_shape = tuple(shape[plane])
    zz = origin[axis] + np.arange(shape[axis]) * step[axis]
    h = float(np.median(np.diff(levels)))
    radius_xyz = np.broadcast_to(np.asarray(radius, float), (3,))
    band = max(4 * float(radius_xyz.max()) * d, 3 * float(sigma.max()), 4 * float(step.max()))
    sdf_mode = method == "sdf"
    output = np.full((len(zz), int(np.prod(xy_shape))), -band if sdf_mode else 0,
                     np.float32)
    xy = None
    if sdf_mode:
        xy = np.stack(np.meshgrid(*[
            origin[a] + np.arange(shape[a]) * step[a] for a in plane
        ], indexing="ij"), axis=-1).reshape(-1, 2)
    order = np.argsort(groups, kind="stable")
    blocks = np.split(order, np.flatnonzero(np.diff(groups[order])) + 1)
    indices = {int(groups[b[0]]): b for b in blocks}
    active = np.array(sorted(indices))
    empties = np.cumsum(known_empty)
    split = (np.diff(levels[active]) > max_gap) | (np.diff(empties[active]) > 0)
    segments = np.split(active, np.flatnonzero(split) + 1)
    bounds = []
    for s, segment in enumerate(segments):
        zobs = levels[segment]
        values = np.zeros((len(segment), output.shape[1]), np.float32)
        weight = np.ones(values.shape, np.float64) if sdf_mode else None
        for j, g in enumerate(segment):
            ids = indices[g]
            if sdf_mode:
                if np.ndim(radius) == 0:
                    distance, nearest = cKDTree(points[ids][:, plane]).query(xy, workers=workers)
                    support_limit = float(radius) * d
                else:
                    stretch = radius_xyz[plane]
                    distance, nearest = cKDTree(points[ids][:, plane] / stretch).query(
                        xy / stretch, workers=workers)
                    support_limit = d
                mask = (distance <= support_limit).reshape(xy_shape)
                field = ndi.distance_transform_edt(mask, sampling=step[plane])
                field -= ndi.distance_transform_edt(~mask, sampling=step[plane])
                values[j] = np.clip(field.ravel(), -band, band)
                weight[j] = np.where(distance <= 2 * support_limit,
                                     confidence[ids[nearest]], 1.0)
            else:
                ij = np.rint(_grid_coordinates(
                    points[ids][:, plane], origin[plane], step[plane], xy_shape)).astype(np.int64)
                flat = np.ravel_multi_index(ij.T, xy_shape)
                np.maximum.at(values[j], flat, confidence[ids])
        if sdf_mode and outlier > 0:
            for j in range(1, len(segment) - 1):
                a = (zobs[j] - zobs[j - 1]) / (zobs[j + 1] - zobs[j - 1])
                prediction = (1 - a) * values[j - 1] + a * values[j + 1]
                scale = outlier * (d + 0.5 * abs(values[j + 1] - values[j - 1]))
                weight[j] /= 1 + ((values[j] - prediction) / scale) ** 2
        lo, hi = zobs[0] - 0.5 * h, zobs[-1] + 0.5 * h
        first, last = segment[0], segment[-1]
        if first > 0 and known_empty[first - 1]:
            lo = 0.5 * (levels[first - 1] + zobs[0])
        if last + 1 < len(levels) and known_empty[last + 1]:
            hi = 0.5 * (zobs[-1] + levels[last + 1])
        # Caps must remain disjoint even when max_gap is smaller than h.
        if s > 0:
            dz = zobs[0] - levels[segments[s - 1][-1]]
            if dz > max_gap:
                lo = max(lo, zobs[0] - 0.25 * dz)
        if s + 1 < len(segments):
            dz = levels[segments[s + 1][0]] - zobs[-1]
            if dz > max_gap:
                hi = min(hi, zobs[-1] + 0.25 * dz)
        pad = band if sdf_mode else 0.0
        iz = np.flatnonzero((zz >= lo - pad) & (zz <= hi + pad))
        kernel = _axial_weights(zz[iz], zobs, sigma[axis])
        if sdf_mode:
            interpolated = kernel @ (weight * values)
            denominator = kernel @ weight
            np.divide(interpolated, denominator, out=interpolated, where=denominator > 0)
            interpolated[denominator <= 0] = -band
            cap = np.minimum(zz[iz] - lo, hi - zz[iz])
            interpolated = np.minimum(interpolated, cap[:, None])
        else:
            interpolated = kernel @ values
        output[iz] = np.maximum(output[iz], interpolated)
        bounds.append((lo, hi))
    output = output.reshape((len(zz),) + xy_shape)
    return np.transpose(output, np.argsort([axis] + plane)), bounds


def _volume_field(points, confidence, origin, step, shape, d, radius, method, *, chunk_size=100_000, workers=-1):
    """Non-planar data: weighted occupancy or the SDF of local point support."""
    if method == "density":
        grid = np.zeros(tuple(shape), np.float32)
        ijk = np.rint(_grid_coordinates(points, origin, step, shape)).astype(np.int64)
        flat = np.ravel_multi_index(ijk.T, tuple(shape))
        np.maximum.at(grid.ravel(), flat, confidence)
        return grid
    mask = _support_mask_3(points, confidence ** (1.0 / 3), np.broadcast_to(np.asarray(radius, float) * d, (3,)),
                           origin, step, tuple(shape), chunk_size, workers)
    field = ndi.distance_transform_edt(mask, sampling=step)
    field -= ndi.distance_transform_edt(~mask, sampling=step)
    return field.astype(np.float32)


def clay_slice_mesh(points=None, label=None, feature=None, *, method="sdf", confidence=None,
              label_mode="project", slice_axis=2, spacing=None, spacing_scale=0.6,
              base_spacing=None, base_spacing_scale=1.0, smooth_scale=None, smooth_sigma=None,
              outlier_thresh=None, outlier_k=16, max_gap=None, support_scale=None,
              density_level=0.30, min_component=None, mesh_smooth_iters=None,
              backend="auto", device="cpu", max_voxels=20_000_000_000_000,
              outlier_method="slice", outlier_iters=1, outlier_radius=None,
              min_points=0.0, graph_scale=2.0, support_k=None,
              smooth_scales=None, field_smooth_iters=1, preserve_volume=True,
              max_shift_scale=None, opening_scale=0.0, closing_scale=0.0,
              hole_scale=0.0, offset_scale=0.0, pre_min_component=0.0,
              min_volume_component=None, component_diameter_scale=0.0, mesh_smooth_pass_band=0.1,
              label_confidence=None, label_outlier_action="relabel", label_outlier_k=24,
              label_outlier_thresh=0.8, label_outlier_iters=1, label_min_points=3,
              label_smooth=None, label_smooth_weight=None, label_axis_weights=None,
              label_lock_seeds=False, label_max_iter=1000, label_tol=1e-5,
              label_project_k=1, separate_protect_labels=True, label_guard_scale=0.6,
              interface_backend="auto", interface_smooth_iters=None,
              interface_max_shift_scale=None, interface_smooth_pass_band=None,
              split_normal_patches=False, score_k=8, score_scale=2.0,
              chunk_size=100_000, workers=-1, return_outer=False, return_info=False,
              X=None, voxel_size=None, smooth=None, outlier=None, radius=None,
              smooth_iters=None, field_smooth_mode="reembed",
              label_guard_mode="hard", label_post_smooth=0.0):
    """Reconstruct exact planar stacks with physical section interpolation.
    
    Nonplanar data use a general 3-D field. slice_axis selects the exact stack
    axis; max_gap and known-empty sections cap interpolation, including after
    label refinement. sdf/density use distinct section observations.
    Return forms match clay_mesh; separate skips classes with fewer than 8 sites
    in its legacy path. Soft guards/post smoothing enable shared class competition.
    Intrinsic sigma uses point spacing in-plane and observed interval axially.
    smooth_scales=None adds no extra post stage. Explicit stages use
    field_smooth_mode='reembed'/'continuous'; intrinsic interpolation is unchanged.
    Mesh and interface smoothing use the shared geometry before splitting.
    Opening/closing are diameters*d; hole_scale is cavity sphere diameter/d.
    pre_min_component/min_volume_component are occupied-component fractions.
    Only automatic grids may coarsen to meet max_voxels; explicit spacing raises.
    Aliases: X, voxel_size, smooth, outlier, radius, smooth_iters. See signature
    for defaults and clay_mesh for shared label options and return_info arrays."""
    post_smooth = _label_refinement_options(label_guard_mode, label_post_smooth)
    shared_separate = label_mode == "separate" and (label_guard_mode == "soft" or np.any(post_smooth))
    if field_smooth_mode not in {"continuous", "reembed"}:
        raise ValueError("field_smooth_mode must be continuous or reembed.")
    if not isinstance(split_normal_patches, (bool, np.bool_)):
        raise ValueError("split_normal_patches must be boolean.")
    if interface_smooth_pass_band is not None and (np.ndim(interface_smooth_pass_band)
            or not np.isfinite(interface_smooth_pass_band)
            or not 0 < interface_smooth_pass_band <= 2):
        raise ValueError("interface_smooth_pass_band must be None or a finite scalar in (0,2].")
    import pyvista as pv

    points = _alias(points, X, None, "points", "X")
    spacing = _alias(spacing, voxel_size, None, "spacing", "voxel_size")
    smooth_scale = _alias(smooth_scale, smooth, (1., 1., 1.5), "smooth_scale", "smooth")
    outlier_thresh = _alias(outlier_thresh, outlier, 3., "outlier_thresh", "outlier")
    support_scale = _alias(support_scale, radius, 2., "support_scale", "radius")
    mesh_smooth_iters = _alias(mesh_smooth_iters, smooth_iters, 10,
                               "mesh_smooth_iters", "smooth_iters")
    data = _cloud_inputs(points, label, confidence, label_confidence, feature, min_points=8)
    raw, geom = data["points"], data["confidence"]
    keys, label_present = data["keys"], data["label"] is not None
    if method not in {"sdf", "density"}:
        raise ValueError("method must be 'sdf' or 'density'.")
    if label_mode not in {"project", "separate", "joint"}:
        raise ValueError("label_mode must be 'project', 'separate' or 'joint'.")
    if label_mode == "separate" and not label_present:
        raise ValueError("label_mode='separate' requires labels.")
    if slice_axis not in (None, 0, 1, 2):
        raise ValueError("slice_axis must be 0, 1, 2 or None.")
    outlier_method = None if outlier_method in {None, "none"} else str(outlier_method).lower()
    if outlier_method == "none":
        outlier_method = None
    if outlier_method not in {None, "slice", "lof", "ratio", "sor", "radius"}:
        raise ValueError("outlier_method must be slice, lof, ratio, sor, radius or None.")
    support_k = outlier_k if support_k is None else support_k
    for name, value, minimum in (("outlier_k", outlier_k, 2), ("support_k", support_k, 2),
            ("outlier_iters", outlier_iters, 0), ("mesh_smooth_iters", mesh_smooth_iters, 0),
            ("field_smooth_iters", field_smooth_iters, 0), ("score_k", score_k, 0),
            ("chunk_size", chunk_size, 1), ("label_project_k", label_project_k, 1),
            ("max_voxels", max_voxels, 1000)):
        if np.ndim(value) or not np.isfinite(value) or int(value) != value or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    outlier_k, support_k, outlier_iters = int(outlier_k), int(support_k), int(outlier_iters)
    mesh_smooth_iters, score_k, chunk_size = int(mesh_smooth_iters), int(score_k), int(chunk_size)
    max_voxels = int(max_voxels)
    if mesh_smooth_iters == 1:
        raise ValueError("mesh_smooth_iters must be 0 or >=2 for Windowed Sinc.")
    if not isinstance(workers, (int, np.integer)) or workers == 0 or workers < -1:
        raise ValueError("workers must be -1 or a positive integer.")
    for name, value, positive in (("outlier_thresh", outlier_thresh, False),
            ("min_points", min_points, False), ("graph_scale", graph_scale, True),
            ("base_spacing_scale", base_spacing_scale, True),
            ("label_guard_scale", label_guard_scale, False)):
        if np.ndim(value) or not np.isfinite(value) or (value <= 0 if positive else value < 0):
            raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}.")
    if outlier_radius is not None and (np.ndim(outlier_radius) or
            not np.isfinite(outlier_radius) or outlier_radius <= 0):
        raise ValueError("outlier_radius must be positive and finite.")
    if base_spacing is not None and (np.ndim(base_spacing) or
            not np.isfinite(base_spacing) or base_spacing <= 0):
        raise ValueError("base_spacing must be positive and finite.")
    support_xyz = _xyz(support_scale, "support_scale", True)
    if not np.isfinite(density_level) or not 0 < density_level < 1:
        raise ValueError("density_level must lie in (0,1).")
    volume_cutoff = _alias(min_volume_component, min_component, 0.,
                           "min_volume_component", "min_component")
    for name, value in (("min_volume_component", volume_cutoff), ("pre_min_component", pre_min_component)):
        if np.ndim(value) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must lie in [0,1].")
    if not np.isfinite(mesh_smooth_pass_band) or not 0 < mesh_smooth_pass_band <= 2:
        raise ValueError("mesh_smooth_pass_band must lie in (0,2].")
    stages = np.array([1.]) if smooth_scales is None else np.atleast_1d(np.asarray(smooth_scales, float))
    if stages.ndim != 1 or not len(stages) or not np.isfinite(stages).all() or np.any(stages <= 0):
        raise ValueError("smooth_scales must be None or a nonempty positive finite sequence.")
    multipliers = _xyz(smooth_scale, "smooth_scale")
    opening_scale, closing_scale, score_scale = (_xyz(opening_scale, "opening_scale"),
        _xyz(closing_scale, "closing_scale"), _xyz(score_scale, "score_scale", True))
    if label_smooth is not None and (label_smooth_weight is not None or label_axis_weights is not None):
        raise ValueError("Specify label_smooth or legacy label_smooth_weight/label_axis_weights.")
    label_smooth = _xyz(.5 if label_smooth is None else label_smooth, "label_smooth")

    unique = data["unique_source"]
    offset = np.median(raw[unique], axis=0)
    p = raw[unique] - offset
    axis = slice_axis
    levels, groups = np.array([0.]), np.zeros(len(p), int)
    if axis is not None:
        levels, groups = np.unique(p[:, axis], return_inverse=True)
        if not 2 <= len(levels) <= len(p) // 8:
            warnings.warn("No exact planar stack detected; using general 3D. "
                          "All smooth multipliers then use point spacing d.", stacklevel=2)
            axis, levels, groups = None, np.array([0.]), np.zeros(len(p), int)
    h = float(np.median(np.diff(levels))) if axis is not None else 0.
    gap = 3*h if max_gap is None else float(max_gap)
    if axis is not None and (not np.isfinite(gap) or gap <= 0):
        raise ValueError("max_gap must be finite and positive.")
    kept = np.arange(len(p))
    _, d_ref = _denoise(p, groups, levels, axis, 0., gap, support_k, base_spacing,
                         workers=workers)
    enabled = outlier_method is not None and outlier_thresh > 0 and outlier_iters > 0
    if outlier_method == "slice":
        for _ in range(outlier_iters if enabled else 0):
            keep, _ = _denoise(p[kept], groups[kept], levels, axis,
                outlier_thresh, gap, outlier_k, d_ref, workers=workers)
            kept = kept[keep]
            if len(kept) < 8:
                raise ValueError("Fewer than eight points remain after slice outlier filtering.")
            if keep.all():
                break
    if (enabled and outlier_method != "slice") or min_points > 0:
        local, _, _ = _clean_cloud(p[kept], geom[unique[kept]], max(support_k, 3),
            outlier_thresh, 0., workers, outlier_method=outlier_method if enabled and
            outlier_method != "slice" else None, outlier_k=outlier_k,
            outlier_iters=outlier_iters, outlier_radius=outlier_radius,
            min_points=min_points, graph_scale=graph_scale)
        kept = kept[local]
    if len(kept) < 8:
        raise ValueError("Fewer than eight distinct finite reliable points remain.")
    state = _cloud_label_observations(data, kept, offset=offset,
        label_outlier_action=label_outlier_action, label_outlier_k=label_outlier_k,
        label_outlier_thresh=label_outlier_thresh, label_outlier_iters=label_outlier_iters,
        label_min_points=label_min_points, graph_scale=graph_scale, workers=workers)
    site_feature = _site_features(data, state)
    source_ids = state["unique_source"]
    p, groups = state["points"] - offset, groups[kept]
    w = geom[source_ids]
    obs_points = state["observation_points"] - offset
    obs_codes, obs_weight, obs_site = state["codes"], state["weights"], state["obs_site"]
    d = float(d_ref * base_spacing_scale)
    scale = np.full(3, d)
    if axis is not None:
        scale[axis] = h * base_spacing_scale
    sigma = multipliers * scale if smooth_sigma is None else _xyz(smooth_sigma, "smooth_sigma")
    sigma = sigma * np.sqrt(field_smooth_iters)
    automatic_grid = spacing is None
    requested_step = _xyz(np.asarray(spacing_scale)*d if automatic_grid else spacing,
                          "spacing", True)
    if automatic_grid and axis is not None:
        requested_step[axis] = min(requested_step[axis], .25*h*base_spacing_scale)
    post_sigma = sigma if smooth_scales is not None else np.zeros(3)
    support = (float(support_scale) if np.ndim(support_scale) == 0 else support_xyz) \
        if method == "sdf" else 2.
    support_max = float(np.max(support))
    post_active = (np.any(post_sigma > 0) or np.any(opening_scale > 0) or
                   np.any(closing_scale > 0) or hole_scale > 0 or offset_scale != 0)
    residual_outlier = outlier_thresh if enabled and outlier_method == "slice" else 0.
    if label_present and not np.any(obs_weight > 0):
        raise ValueError("No positive-confidence label evidence remains.")
    info = dict(inlier_mask=state["inlier_mask"], retained_indices=source_ids.copy(),
        label_corrected_id=state["corrected_codes"] if label_present else None,
        label_corrected=_category_values(keys, state["corrected_codes"], for_vtk=False)
                        if label_present else None,
        label_reliability=state["label_reliability"],
        label_outlier_mask=state["label_outlier_mask"])
    shared_meta = dict(method=[method], base_spacing=[d], base_spacing_reference=[d_ref],
        base_spacing_scale=[base_spacing_scale], sampling_spacing=scale[None],
        smooth_sigma=sigma[None], post_smooth_sigma=post_sigma[None],
        smooth_scales=stages if smooth_scales is not None else np.empty(0),
        field_smooth_iters=[int(field_smooth_iters)], preserve_volume=[bool(preserve_volume)],
        post_field_smoothing=[smooth_scales is not None],
        reconstruction_mode=["sections" if axis is not None else "volume"],
        outlier_method=[str(outlier_method)], outlier_k=[outlier_k], outlier_iters=[outlier_iters],
        label_mode=[label_mode if label_present else "unlabelled"],
        mesh_smooth_iters=[mesh_smooth_iters], mesh_smooth_pass_band=[mesh_smooth_pass_band],
        label_changed_points=[int(np.count_nonzero(state["label_changed_mask"]))],
        label_outlier_points=[int(np.count_nonzero(state["label_outlier_mask"]))])

    dominant = np.full(len(p), -1, np.int64)
    trusted = np.zeros(len(p), bool)

    def volume(ids, code=None, fixed_grid=None, allow_empty=False):
        points_i, weight_i = p[ids], w[ids]
        step = requested_step.copy()
        if axis is not None and 64*len(levels) > max_voxels:
            raise MemoryError("max_voxels cannot hold the observed-section arrays, even after coarsening.")
        extra = .5*d*(opening_scale+closing_scale) + abs(d*offset_scale)
        extra += 3*post_sigma*stages.sum()
        if fixed_grid is None:
            for attempt in range(40):
                margin = max(2., support_max)*d + 3*sigma + 3*step + np.ceil(extra/step)*step
                if axis is not None:
                    margin[axis] += h
                origin, upper = points_i.min(axis=0)-margin, points_i.max(axis=0)+margin
                if axis is not None:
                    origin[axis], upper[axis] = levels[0]-margin[axis], levels[-1]+margin[axis]
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    extent = (upper-origin)/step
                if (not np.isfinite(extent).all() or np.any(extent < 0) or
                        np.any(extent > np.iinfo(np.intp).max//2)):
                    raise MemoryError("Grid dimensions overflow; increase spacing or rescale coordinates.")
                shape = np.ceil(extent).astype(np.intp)+1
                storage = math.prod(shape.tolist())
                if axis is not None:
                    storage = max(storage, math.prod(np.delete(shape, axis).tolist())*len(levels))
                if storage <= max_voxels:
                    break
                if not automatic_grid:
                    raise MemoryError("Requested grid exceeds max_voxels; increase spacing or max_voxels.")
                step *= max(1.05, (storage/max_voxels)**(1/3))
            else:
                raise MemoryError("Auto coarsening cannot meet max_voxels; increase the limit.")
            if np.any(step > requested_step*1.001):
                warnings.warn(f"Auto grid coarsened to spacing={tuple(step)}.", stacklevel=3)
        else:
            origin, step, shape = fixed_grid
            origin, step, shape = np.asarray(origin), np.asarray(step), np.asarray(shape)
        field_weights = weight_i/weight_i.max() if method == "density" else weight_i
        slabs = []
        if axis is None:
            field = _volume_field(points_i, field_weights, origin, step, shape, d,
                                  support, method, chunk_size=chunk_size, workers=workers)
        else:
            known_empty = np.zeros(len(levels), bool)
            if code is not None:
                valid = (obs_codes == code) & (obs_weight > 0)
                known_empty = np.bincount(groups[obs_site[valid]], minlength=len(levels)) == 0
            field, slabs = _stack_field(points_i, field_weights, groups[ids], levels,
                known_empty, axis, origin, step, shape, d, sigma, support,
                residual_outlier, gap, method, workers=workers)
        blur = sigma/step
        if axis is not None:
            blur[axis] = 0
        field = _gaussian(field, blur, device)
        reference = None
        if method == "density":
            sample = ndi.map_coordinates(field, ((points_i-origin)/step).T,
                                         order=1, mode="nearest", prefilter=False)
            reference = float(np.median(sample))
            if not np.isfinite(reference) or reference <= 0:
                raise ValueError("Empty density field; use a finer grid or larger smooth_sigma.")
            occupied = field >= density_level*reference
            field = (ndi.distance_transform_edt(occupied, sampling=step) -
                     ndi.distance_transform_edt(~occupied, sampling=step)).astype(np.float32)
        field = _gaussian(field, np.where(sigma > 0, .5, 0.), device)
        forbidden = np.zeros(tuple(shape), bool)
        cap = None
        if slabs:
            zz = origin[axis] + np.arange(shape[axis])*step[axis]
            cap = np.full(len(zz), -np.inf)
            for lo, hi in slabs:
                cap = np.maximum(cap, np.minimum(zz-lo, hi-zz))
            view = [1, 1, 1]; view[axis] = len(zz)
            cap = cap.reshape(view)
            np.minimum(field, cap, out=field)
            if post_active:
                forbidden |= np.broadcast_to(cap <= 0, tuple(shape))
        if code is not None and not shared_separate and separate_protect_labels and label_guard_scale > 0:
            foreign = p[trusted & (dominant != code)]
            if len(foreign):
                tree = cKDTree(foreign)
                for start in range(0, forbidden.size, chunk_size):
                    stop = min(start+chunk_size, forbidden.size)
                    ijk = np.column_stack(np.unravel_index(np.arange(start,stop), tuple(shape)))
                    distance = tree.query(origin+ijk*step,
                        distance_upper_bound=d*label_guard_scale, workers=workers)[0]
                    forbidden.ravel()[start:stop] |= np.isfinite(distance)
        for a in range(3):
            sl = [slice(None)]*3; sl[a] = [0,-1]
            field[tuple(sl)] = -max(d, float(step.max()))
            if post_active:
                forbidden[tuple(sl)] = True
        mask, field, metadata = _process_volume(field > 0, step, d,
            smooth_sigma=post_sigma, smooth_scales=stages, smooth_method="sdf",
            preserve_volume=preserve_volume, max_shift_scale=max_shift_scale,
            opening_scale=opening_scale, closing_scale=closing_scale,
            hole_scale=hole_scale, offset_scale=offset_scale,
            pre_min_component=pre_min_component, min_volume_component=volume_cutoff,
            component_diameter_scale=component_diameter_scale,
            keep_largest=volume_cutoff == 1, forbidden=forbidden if forbidden.any() else None,
            initial_field=field, field_smooth_mode=field_smooth_mode,
            device=device)
        if cap is not None:
            np.minimum(field, cap, out=field)
            mask = field > 0
        if not allow_empty and not (field.min() < 0 < field.max()):
            raise ValueError("No surface remains; reduce smoothing/opening or adjust support_scale.")
        metadata.update(spacing=step[None], voxel_size=step[None], retained_points=[len(ids)],
            grid_shape=np.asarray(shape)[None])
        if reference is not None:
            metadata.update(density_reference=[reference*weight_i.max()],
                density_threshold=[density_level*reference*weight_i.max()])
        if allow_empty and cap is not None:
            metadata["label_allowed_mask"] = np.broadcast_to(cap > 0, tuple(shape))
        return mask, field, origin, step, metadata

    def annotate(mesh, ids, metadata, constant=None):
        for name, value in {**metadata, **shared_meta}.items():
            if np.asarray(value).size:
                mesh.field_data[name] = np.atleast_1d(value)
        if not mesh.n_points:
            return mesh
        _annotate_clay_mesh(mesh, p[ids], source_ids[ids], geom,
                           None if site_feature is None else site_feature[ids],
                           d, score_k, score_scale, chunk_size, workers)
        if label_present and label_mode != "joint" and not shared_separate:
            winner, certainty = _project_label_evidence(mesh.points, obs_points,
                obs_codes, obs_weight, label_project_k, d, workers, chunk_size,
                target=constant)
            if constant is not None:
                winner[:] = constant
            mesh.point_data["label_id"] = winner
            mesh.point_data["label"] = _category_values(keys, winner)
            mesh.point_data["label_confidence"] = certainty
            cell_ids, _ = _project_label_evidence(mesh.cell_centers().points, obs_points,
                obs_codes, obs_weight, label_project_k, d, workers, chunk_size)
            if constant is not None:
                cell_ids[:] = constant
            mesh.cell_data["label_id"] = cell_ids
            mesh.cell_data["label"] = _category_values(keys, cell_ids)
        mesh.points = np.asarray(mesh.points, float) + offset
        return mesh

    def outer_mesh(ids, code=None):
        mask, field, origin, step, metadata = volume(ids, code)
        vertices, faces = _marching_cubes(field, origin, step, backend, device)
        mesh = pv.PolyData(vertices, np.column_stack((np.full(len(faces),3),faces))).clean()
        if mesh_smooth_iters:
            mesh = mesh.smooth_taubin(n_iter=mesh_smooth_iters,
                pass_band=mesh_smooth_pass_band, normalize_coordinates=True)
        mesh = mesh.compute_normals(consistent_normals=True, auto_orient_normals=False)
        mesh = annotate(mesh, ids, metadata, constant=code)
        if label_present and label_mode == "project" and np.any(post_smooth):
            local_obs = dict(state, observation_points=obs_points)
            assigned = _grid_label_data(mask, origin, step, local_obs, mode="project", d=d,
                k=label_project_k, workers=workers, chunk_size=chunk_size,
                label_post_smooth=post_smooth, label_lock_seeds=label_lock_seeds,
                label_max_iter=label_max_iter, label_tol=label_tol)
            _mesh_partition_labels(mesh, assigned["label_mask"], assigned["label_score_image"],
                origin + offset, step, keys, workers, chunk_size)
            info["label_info"] = assigned["label_info"]
        return mesh

    all_ids = np.arange(len(p))
    if not label_present or label_mode == "project":
        result = outer_mesh(all_ids)
    elif label_mode == "separate" and not shared_separate:
        dominant, trusted = _label_guard_sites(p, obs_site, obs_codes, obs_weight,
            k=label_outlier_k, graph_scale=graph_scale, workers=workers)
        result = {}
        for code, key in enumerate(keys):
            ids = np.unique(obs_site[(obs_codes == code) & (obs_weight > 0)])
            if len(ids) < 8:
                warnings.warn(f"Skip label {key!r}: fewer than eight retained points.", stacklevel=2)
                continue
            result[key] = outer_mesh(ids, code)
        if not result:
            raise ValueError("No label has enough retained points.")
        if return_outer:
            result = (result, outer_mesh(all_ids))
    else:
        mask, field, origin, step, metadata = volume(all_ids)
        if shared_separate:
            dominant, trusted = _label_guard_sites(p, obs_site, obs_codes, obs_weight,
                k=label_outlier_k, graph_scale=graph_scale, workers=workers)
            shape, padded_origin = _label_grid_halo(mask.shape, origin, step,
                                                   4*d*label_smooth, max_voxels)
            def class_fields():
                for code, key in enumerate(keys):
                    ids = np.unique(obs_site[(obs_codes == code) & (obs_weight > 0)])
                    if not len(ids):
                        continue
                    private, field_i, origin_i, step_i, report_i = volume(ids, code,
                        fixed_grid=(padded_origin, step, shape), allow_empty=True)
                    yield code, field_i, bool(private.any()), p[ids], report_i, origin_i, step_i
            local_obs = dict(state, points=p, observation_points=obs_points)
            assigned = _separate_label_volume(mask, origin, step, d,
                origin + np.argwhere(mask)*step, local_obs, class_fields,
                label_smooth=label_smooth, label_smooth_weight=label_smooth_weight,
                label_axis_weights=label_axis_weights, label_lock_seeds=label_lock_seeds,
                site_codes=dominant, site_trusted=trusted,
                guard_radius=d*label_guard_scale if separate_protect_labels else 0.,
                workers=workers, chunk_size=chunk_size, device=device,
                guard_mode=label_guard_mode, post_smooth=d*post_smooth,
                label_max_iter=label_max_iter, label_tol=label_tol)
            labels, assignment, label_meta = (assigned["label_mask"],
                assigned["label_score_image"], assigned["label_info"])
        else:
            labels, assignment, label_meta = _label_volume(mask, origin, step, obs_points,
                obs_codes, obs_weight, d, label_smooth, support_k, workers,
                label_max_iter=label_max_iter, label_tol=label_tol,
                label_lock_seeds=label_lock_seeds, label_smooth_weight=label_smooth_weight,
                label_axis_weights=label_axis_weights, chunk_size=chunk_size,
                label_post_smooth=post_smooth)
        if shared_separate or np.any(post_smooth):
            info["label_info"] = label_meta
        meshes, outer = _mesh_from_labels(labels, field, origin, step, keys,
            mesh_smooth_iters, mesh_smooth_pass_band, interface_backend=interface_backend,
            interface_smooth_iters=interface_smooth_iters,
            interface_max_shift_scale=interface_max_shift_scale, d=d,
            interface_smooth_pass_band=interface_smooth_pass_band,
            split_normal_patches=split_normal_patches)
        metadata.update(_numeric_label_metadata(label_meta) if shared_separate or np.any(post_smooth) else label_meta)
        counts = np.bincount(labels.ravel(), minlength=len(keys)+1)[1:]
        metadata.update(label_voxel_counts=counts, label_volumes=counts*np.prod(step),
                        label_smooth=(d*label_smooth)[None])
        mask_float = mask.astype(np.float32)
        for mesh in [outer, *meshes.values()]:
            if mesh.n_points:
                coordinates = ((mesh.points-origin)/step).T
                mass = ndi.map_coordinates(mask_float, coordinates, order=1, mode="constant", cval=0)
                certainty = ndi.map_coordinates(assignment, coordinates, order=1, mode="constant", cval=0)
                mesh.point_data["label_confidence"] = np.clip(np.divide(certainty,mass,
                    out=np.zeros_like(certainty),where=mass>1e-8),0,1)
            annotate(mesh, all_ids, metadata)
        result = (meshes, outer) if return_outer else meshes
    return (result, info) if return_info else result


def _label_codes(labels):
    """Factorize hashable, nonmissing categories in first-occurrence order."""
    keys, lookup = [], {}
    inverse = np.empty(len(labels), dtype=np.int64)
    for i, key in enumerate(labels):
        key = key.item() if isinstance(key, np.generic) else key
        try:
            hash(key)
            missing = key is None or bool(key != key)
        except (TypeError, ValueError) as error:
            raise ValueError("label must contain hashable, nonmissing scalar categories.") from error
        if missing:
            raise ValueError("label must not contain missing categories (None, NaN or NA).")
        code = lookup.get(key)
        if code is None:
            code = lookup[key] = len(keys)
            keys.append(key)
        inverse[i] = code
    return keys, inverse


def _blocks(points, reach, merge_bounds=False):
    """Group nearby cells; optionally unite overlapping boxes before cavity filling."""
    cells, inverse = np.unique(np.floor((points - points.min(0)) / reach).astype(np.int64),
                               axis=0, return_inverse=True)
    edges = cKDTree(cells).query_pairs(1.0, p=np.inf, output_type="ndarray")
    graph = sparse.coo_matrix((np.ones(len(edges), np.uint8), (edges[:, 0], edges[:, 1])),
                              shape=(len(cells), len(cells))).tocsr()
    _, groups = sparse.csgraph.connected_components(graph, directed=False)
    groups = groups[inverse]
    order = np.argsort(groups, kind="stable")
    blocks = np.split(order, np.flatnonzero(np.diff(groups[order])) + 1)
    if not merge_bounds or len(blocks) < 2:
        return blocks
    lower = np.array([points[ids].min(0) for ids in blocks]) - reach / 2
    upper = np.array([points[ids].max(0) for ids in blocks]) + reach / 2
    parent = np.arange(len(blocks))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    active = []
    for i in np.argsort(lower[:, 0]):
        active = [j for j in active if upper[j, 0] >= lower[i, 0]]
        for j in active:
            if np.all(upper[j, 1:] >= lower[i, 1:]) and np.all(upper[i, 1:] >= lower[j, 1:]):
                parent[root(j)] = root(i)
        active.append(i)
    merged = {}
    for i, ids in enumerate(blocks):
        merged.setdefault(root(i), []).append(ids)
    return [np.concatenate(ids) for ids in merged.values()]


def _support_mask(points, precision, axes, origin, step, shape, chunk, workers=-1):
    """Exact ellipsoid union at grid nodes, with bounded batches of candidates."""
    full_shape = tuple(shape)
    longest = float(axes.max())
    lower = np.clip(np.floor((points.min(0) - longest - origin) / step), 0, shape).astype(int)
    upper = np.clip(np.ceil((points.max(0) + longest - origin) / step) + 1, 0, shape).astype(int)
    if np.any(upper <= lower):
        return np.zeros(full_shape, bool)
    crop = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
    shape = tuple(upper - lower)
    origin = origin + lower * step
    tree = cKDTree(points)
    shortest, longest = axes.min(1), float(axes.max())
    upper = np.nextafter(longest, np.inf)
    mask = np.zeros(int(np.prod(shape)), bool)
    for start in range(0, len(mask), chunk):
        stop = min(start + chunk, len(mask))
        grid = np.column_stack(np.unravel_index(np.arange(start, stop), shape))
        q = origin + grid * step
        dist, ids = tree.query(q, workers=workers)
        inside = dist <= shortest[ids]
        pending = np.flatnonzero(~inside & (dist <= longest))
        count = min(8, len(points))
        while len(pending):
            remaining = []
            batch = max(1, min(chunk, 1_000_000 // count))
            for j in range(0, len(pending), batch):
                at = pending[j:j + batch]
                dist, ids = tree.query(q[at], k=list(range(1, count + 1)),
                                       distance_upper_bound=upper, workers=workers)
                valid = ids < len(points)
                ids = np.minimum(ids, len(points) - 1)
                delta = q[at, None, :] - points[ids]
                metric = np.einsum("nki,nkij,nkj->nk", delta, precision[ids], delta)
                found = np.any(valid & (metric <= 1.0), axis=1)
                inside[at[found]] = True
                if count < len(points):
                    remaining.append(at[~found & np.isfinite(dist[:, -1])])
            pending = np.concatenate(remaining) if remaining else np.empty(0, int)
            count = min(2 * count, len(points))
        mask[start:stop] = inside
    result = np.zeros(full_shape, bool)
    result[crop] = mask.reshape(shape)
    return result


def _reconstruct(points, precision, axes, d, smooth, gap_size, voxel_size,
                 min_component, smooth_iters, backend, device, max_voxels, chunk,
                 voxel_scale=0.5, method="sdf", density_level=0.5,
                 smooth_sigma=None, *, smooth_scales=(1.0,), preserve_volume=True,
                 max_shift_scale=2.0, closing_scale=0.0, opening_scale=0.0,
                 hole_scale=0.0, offset_scale=0.0, mesh_smooth_pass_band=0.1,
                 workers=-1, return_volumes=False, return_fields=False,
                 pre_min_component=None, protect_points=None, protect_radius=0.0,
                 component_diameter_scale=0.0, field_smooth_mode="reembed"):
    """Blocked reconstruction using the shared volume-processing pipeline."""
    import pyvista as pv

    step = _xyz(d * np.asarray(voxel_scale) if voxel_size is None else voxel_size,
                "spacing", True)
    sigma = _xyz(d * np.asarray(smooth) if smooth_sigma is None else smooth_sigma,
                 "smooth_sigma")
    scales = np.atleast_1d(np.asarray(smooth_scales, float))
    closing = 0.5 * _xyz(d * np.asarray(closing_scale) if gap_size is None else gap_size,
                          "closing diameter")
    opening = 0.5 * d * _xyz(opening_scale, "opening_scale")
    shift = None if max_shift_scale is None else d * max_shift_scale
    pre = min_component if pre_min_component is None else pre_min_component
    if not np.isfinite(pre) or not 0 <= pre <= 1:
        raise ValueError("pre_min_component must lie in [0,1] or be None.")
    if not np.isfinite(protect_radius) or protect_radius < 0:
        raise ValueError("protect_radius must be a finite nonnegative physical radius.")
    if (np.ndim(component_diameter_scale) or not np.isfinite(component_diameter_scale)
            or component_diameter_scale < 0):
        raise ValueError("component_diameter_scale must be finite and nonnegative.")
    protection = None
    if protect_points is not None:
        pp = np.asarray(protect_points, float)
        if pp.ndim != 2 or pp.shape[1] != 3 or not np.isfinite(pp).all():
            raise ValueError("protect_points must contain finite XYZ rows.")
        if len(pp) and protect_radius > 0:
            protection = cKDTree(pp)
    method = "density" if method == "occupancy" else method
    active = method is not None and np.any(sigma) and shift != 0
    kernel_margin = _gaussian_halo(sigma, scales, step) if active else np.zeros(3)
    change_band = shift if active and shift is not None else 0.0
    movement = 2 * (closing + opening) + abs(offset_scale) * d
    margin = axes.max() + movement + kernel_margin + change_band + 3 * step
    blocks = _blocks(points, 2 * float(margin.max()), merge_bounds=hole_scale > 0)
    grids, peak_voxels = [], 0
    for ids in blocks:
        origin = np.floor((points[ids].min(0) - margin) / step) * step
        extent = np.ceil((points[ids].max(0) + margin - origin) / step) + 1
        if not np.isfinite(extent).all() or np.any(extent > min(max_voxels, np.iinfo(np.intp).max)):
            raise MemoryError("Grid dimensions exceed the limit; increase spacing.")
        shape = extent.astype(np.intp)
        size = math.prod(shape.tolist())
        if size > max_voxels:
            raise MemoryError(f"One spatial block needs {size:,} voxels; increase spacing "
                              "or max_voxels. The requested resolution was preserved.")
        peak_voxels = max(peak_voxels, size)
        grids.append((ids, origin, shape, size))

    def make_grid(grid):
        ids, origin, shape, size = grid
        mask = _support_mask(points[ids], precision[ids], axes[ids], origin,
                             step, shape, chunk, workers)
        raw = int(mask.sum())
        forbidden = None
        if protection is not None:
            forbidden = np.zeros(size, bool)
            for start in range(0, size, chunk):
                stop = min(start + chunk, size)
                ijk = np.column_stack(np.unravel_index(np.arange(start, stop), shape))
                distance = protection.query(origin + ijk * step, workers=workers,
                    distance_upper_bound=np.nextafter(protect_radius, np.inf))[0]
                forbidden[start:stop] = distance <= protect_radius
            forbidden = forbidden.reshape(shape)
            mask &= ~forbidden
        return mask, forbidden, raw

    # Two passes avoid retaining every padded grid. One block reuses its mask.
    # The reference is GLOBAL, so a tiny distant block cannot define itself as
    # the largest reference component and survive only because it is isolated.
    largest_raw, winning_raw, cached = 0, None, None
    if pre > 0:
        for block_id, grid in enumerate(grids):
            entry = make_grid(grid)
            if len(grids) == 1:
                cached = entry
            _, components = _component_mask(entry[0], step, d)
            if components["largest_component_voxels"] > largest_raw:
                largest_raw = components["largest_component_voxels"]
                winning_raw = block_id
            del entry
    parts, largest, n_components = [], 0, 0
    raw_count = pre_removed = post_morph_count = smoothed_count = 0
    voxel_volume = float(np.prod(step))
    diameter_cutoff = np.pi / 6 * (d * component_diameter_scale) ** 3 / voxel_volume
    for block_id, grid in enumerate(grids):
        ids, origin, shape, _ = grid
        mask, forbidden, raw = cached if cached is not None else make_grid(grid)
        raw_count += raw
        if pre > 0 and mask.any():
            if pre == 1 and block_id != winning_raw:
                pre_removed += int(mask.sum())
                continue
            mask, before = _component_mask(mask, step, d,
                min_volume_component=pre, reference_count=largest_raw,
                keep_largest=pre == 1)
            pre_removed += before["removed_voxels"]
        if not mask.any():
            continue
        # Block-local filtering is disabled here: prefiltering already used the
        # global reference, and final component sizes are compared across blocks
        # below. All morphology/smoothing/offset logic is shared by the public
        # mesh, slice, point-grid and image reconstruction paths.
        mask, field, volume = _process_volume(mask, step, d,
            smooth_sigma=sigma, smooth_scales=scales, smooth_method=method,
            density_level=density_level, preserve_volume=preserve_volume,
            max_shift=shift, opening_scale=2 * opening / d,
            closing_scale=2 * closing / d, hole_scale=hole_scale,
            offset_scale=offset_scale, min_volume_component=0,
            pre_min_component=0, forbidden=forbidden, device=device,
            field_smooth_mode=field_smooth_mode)
        post_morph_count += int(round(volume["post_morph_volume"] / voxel_volume))
        smoothed_count += int(round(volume["smoothing_volume"] / voxel_volume))
        if not mask.any():
            continue
        for axis in range(3):
            for edge in (0, -1):
                sl = [slice(None)] * 3
                sl[axis] = edge
                np.minimum(field[tuple(sl)], -step.min(), out=field[tuple(sl)])
        mask = field > 0
        regions, count = ndi.label(mask)
        n_components += count
        if not count:
            continue
        sizes = np.bincount(regions.ravel())
        largest = max(largest, int(sizes[1:].max()))
        for region, box in enumerate(ndi.find_objects(regions), 1):
            volume = int(sizes[region])
            if volume < max(min_component * largest, diameter_cutoff):
                continue
            lo = np.array([max(0, sl.start - 1) for sl in box])
            hi = np.array([min(n, sl.stop + 1) for sl, n in zip(box, shape)])
            crop = tuple(slice(l, h) for l, h in zip(lo, hi))
            local_mask = regions[crop] == region
            local = field[crop].copy()
            local[~local_mask & (local > 0)] = -float(step.min())
            if return_volumes or return_fields:
                part = (volume, local_mask, origin + lo * step)
                parts.append((*part, local) if return_fields else part)
            else:
                vertices, faces = _marching_cubes(local, origin + lo * step,
                                                 step, backend, device)
                parts.append((volume, vertices, faces))
        del field, mask, regions
    parts = [part for part in parts if part[0] >= max(min_component * largest, diameter_cutoff)]
    if min_component == 1 and parts:
        parts = [max(parts, key=lambda part: part[0])]
    if not parts:
        raise ValueError("No occupied region remains; reduce spacing, opening or smoothing, "
                         "increase support, or relax filtering/protection.")
    output_volume = sum(part[0] for part in parts) * voxel_volume
    metadata = {
        "spacing": step[None], "spacing_scale": (step / d)[None], "base_spacing": [d],
        "voxel_size": step[None], "voxel_scale": (step / d)[None], "sampling_spacing": [d],
        "smooth_sigma": sigma[None], "smooth_scales": scales,
        "field_smooth_mode": [field_smooth_mode],
        "preserve_volume": [int(preserve_volume)], "method": ["none" if method is None else method],
        "gap_size": (2 * closing)[None], "opening_size": (2 * opening)[None],
        "hole_scale": [hole_scale], "offset_scale": [offset_scale],
        "volume_initial": [post_morph_count * voxel_volume],
        "volume_final": [smoothed_count * voxel_volume], "volume_output": [output_volume],
        "raw_volume": [raw_count * voxel_volume],
        "post_morph_volume": [post_morph_count * voxel_volume],
        "smoothing_volume": [smoothed_count * voxel_volume], "final_volume": [output_volume],
        "pre_removed_voxels": [pre_removed], "pre_min_component": [pre],
        "component_diameter_scale": [component_diameter_scale],
        "volume_components": [len(parts)], "removed_components": [n_components - len(parts)],
        "spatial_blocks": [len(blocks)], "peak_grid_voxels": [peak_voxels],
    }
    if method == "density":
        metadata["density_level"] = [density_level]
    if return_volumes or return_fields:
        return parts, metadata
    vertices, faces, offset = [], [], 0
    for _, v, f in parts:
        vertices.append(v); faces.append(f + offset); offset += len(v)
    faces = np.concatenate(faces)
    mesh = pv.PolyData(np.concatenate(vertices), np.c_[np.full(len(faces), 3), faces]).clean()
    if smooth_iters:
        mesh = mesh.smooth_taubin(n_iter=smooth_iters, pass_band=mesh_smooth_pass_band,
                                  normalize_coordinates=True)
    mesh = mesh.compute_normals(consistent_normals=True, auto_orient_normals=False)
    for key, value in metadata.items():
        mesh.field_data[key] = value
    return mesh


def clay_mesh(
    points=None, label=None, confidence=None, *, feature=None, label_mode="joint",
    k=None, support_k=None, outlier_method="lof", outlier_thresh=None,
    outlier_k=None, outlier_iters=1, outlier_radius=None, min_points=None,
    graph_scale=2.0, support_scale=None, adaptive=True, radius_cap=1.6,
    anisotropy=1.0, base_spacing=None, base_spacing_scale=1.0,
    spacing=None, spacing_scale=None, smooth_scale=None, smooth_sigma=None,
    smooth_scales=(1.0,), field_smooth_iters=1, preserve_volume=True, max_shift_scale=None,
    method="sdf", density_level=0.5, gap_size=None, opening_scale=0.0,
    closing_scale=0.0, hole_scale=0.0, offset_scale=0.0,
    min_component=None, pre_min_component=None, min_volume_component=None,
    component_diameter_scale=0.0,
    label_confidence=None, label_outlier_action="relabel", label_outlier_k=24,
    label_outlier_thresh=0.8, label_outlier_iters=1, label_min_points=3,
    label_smooth=None, label_smooth_weight=None, label_axis_weights=None,
    label_lock_seeds=False, label_max_iter=1000, label_tol=1e-5,
    label_project_k=1, separate_geometry="shared", separate_protect_labels=True,
    label_guard_scale=0.6, mesh_smooth_iters=None, mesh_smooth_pass_band=0.05,
    interface_backend="auto", interface_smooth_iters=None,
    interface_max_shift_scale=None, interface_smooth_pass_band=None,
    split_normal_patches=False, score_k=8, score_scale=2.0,
    backend="auto", device="cpu", return_outer=False, return_info=False,
    max_voxels=20_000_000_000_000, chunk_size=100_000, workers=-1,
    pc=None, radius=None, smooth=None, voxel_size=None, voxel_scale=None,
    reference_spacing=None, outlier=None, smooth_iters=None,
    field_smooth_mode="reembed", support_smooth=0.0,
    label_guard_mode="hard", label_post_smooth=0.0,
):
    """Reconstruct occupied volumes; retain legacy aliases and all label modes.
    
    Unlabelled/project returns PolyData. Joint returns {label: PolyData} sharing
    interfaces and an exact common shell. Legacy separate returns independent
    meshes; soft guards or positive label_post_smooth select class competition
    and shared extraction. return_outer adds (dict, outer); return_info wraps
    (result, info). Project/unlabelled always returns one mesh before info wrapping.
    
    d = estimated/supplied base_spacing * base_spacing_scale. support_scale is
    XYZ support stretching; confidence shrinks radii, label_confidence does not
    shrink common geometry. Duplicate sites keep their label evidence and averaged
    attributes. separate_geometry chooses global supports or local re-estimation.
    support_smooth regularizes adaptive radii; anisotropy caps PCA shape ratio.
    Explicit spacing/smooth_sigma are physical lengths; otherwise use d times
    spacing_scale/smooth_scale. field_smooth_iters multiplies sigma by sqrt(time).
    smooth_scales defines stages; field_smooth_mode selects reembed/continuous.
    
    Opening/closing scales are XYZ diameters*d; hole_scale is maximum enclosed
    cavity sphere diameter/d; offset_scale is signed displacement/d. Pre/final
    component cutoffs are fractions of the global largest component; pre=None
    inherits min_component. preserve_volume precedes mesh smoothing and does not
    preserve each class. max_shift_scale constrains the field, not mesh motion.
    Label cleanup never deletes geometry. label_smooth/label_post_smooth are
    lengths/d; hard seed locks remain optional. Label scores are not calibrated.
    Mesh iterations use Windowed Sinc; internal smoothing operates before split,
    with shared IDs preserved. Do not independently smooth split joint meshes.
    
    Aliases: pc->points, k/support_k, radius->support_scale, smooth->smooth_scale,
    voxel_size->spacing, voxel_scale->spacing_scale, reference_spacing->base_spacing,
    outlier->outlier_thresh, smooth_iters->mesh_smooth_iters,
    min_volume_component/min_component. Conflicting aliases raise.
    Features may be sparse; source_id, score and label arrays are retained in VTP.
    See the function signature for defaults and points_to_image for shared units."""
    post_smooth = _label_refinement_options(label_guard_mode, label_post_smooth)
    shared_separate = label_mode == "separate" and (label_guard_mode == "soft" or np.any(post_smooth))
    if field_smooth_mode not in {"continuous", "reembed"}:
        raise ValueError("field_smooth_mode must be continuous or reembed.")
    if not isinstance(split_normal_patches, (bool, np.bool_)):
        raise ValueError("split_normal_patches must be boolean.")
    if interface_smooth_pass_band is not None and (np.ndim(interface_smooth_pass_band)
            or not np.isfinite(interface_smooth_pass_band)
            or not 0 < interface_smooth_pass_band <= 2):
        raise ValueError("interface_smooth_pass_band must be None or a finite scalar in (0,2].")
    import pyvista as pv

    raw = _alias(points, pc, None, "points", "pc")
    support_scale = _xyz(_alias(support_scale, radius, 1.5, "support_scale", "radius"), "support_scale", True)
    smooth_scale = _xyz(_alias(smooth_scale, smooth, 1., "smooth_scale", "smooth"), "smooth_scale")
    spacing = _alias(spacing, voxel_size, None, "spacing", "voxel_size")
    spacing_scale = _xyz(_alias(spacing_scale, voxel_scale, .4, "spacing_scale", "voxel_scale"), "spacing_scale", True)
    base_spacing = _alias(base_spacing, reference_spacing, None, "base_spacing", "reference_spacing")
    outlier_thresh = _alias(outlier_thresh, outlier, 1.5, "outlier_thresh", "outlier")
    mesh_smooth_iters = _alias(mesh_smooth_iters, smooth_iters, 40, "mesh_smooth_iters", "smooth_iters")
    min_component = _alias(min_volume_component, min_component, .01,
                           "min_volume_component", "min_component")
    k = _alias(k, support_k, 24, "k", "support_k")
    outlier_k = k if outlier_k is None else outlier_k
    data = _cloud_inputs(raw, label, confidence, label_confidence, feature, min_points=4)
    x, label, feature = data["points"], data["label"], data["feature"]
    c = data["confidence"]
    keys = [0] if label is None else data["keys"]
    codes = np.zeros(len(x), np.int64) if label is None else data["codes"]
    if label_mode not in {"joint", "project", "separate"}:
        raise ValueError("label_mode must be joint, project or separate.")
    if separate_geometry not in {"shared", "local"}:
        raise ValueError("separate_geometry must be shared or local.")
    if method not in {"sdf", "density"} or not np.isfinite(density_level) or not 0 < density_level < 1:
        raise ValueError("method must be sdf/density and density_level in (0,1).")
    if backend not in {"auto", "pymcubes", "skimage", "lewiner", "lorensen", "torchmcubes", "vtk"}:
        raise ValueError("Unknown marching cubes backend.")
    if interface_backend not in {"auto", "surface_nets", "voxel_faces"}:
        raise ValueError("Unknown interface_backend.")
    if str(device).split(":")[0] not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or a torch CUDA device.")
    for name, v in (("preserve_volume", preserve_volume), ("adaptive", adaptive),
                    ("label_lock_seeds", label_lock_seeds), ("separate_protect_labels", separate_protect_labels)):
        if not isinstance(v, (bool, np.bool_)):
            raise ValueError(f"{name} must be a boolean.")
    for name, value, lower in (("k", k, 3), ("outlier_k", outlier_k, 3),
            ("outlier_iters", outlier_iters, 0), ("field_smooth_iters", field_smooth_iters, 0),
            ("mesh_smooth_iters", mesh_smooth_iters, 0),
            ("score_k", score_k, 0), ("label_project_k", label_project_k, 1),
            ("label_max_iter", label_max_iter, 1), ("max_voxels", max_voxels, 1000),
            ("chunk_size", chunk_size, 1)):
        if np.ndim(value) or not np.isfinite(value) or int(value) != value or value < lower:
            raise ValueError(f"{name} must be an integer >= {lower}.")
    k, outlier_k, outlier_iters, mesh_smooth_iters, score_k, chunk_size = map(int,
        (k, outlier_k, outlier_iters, mesh_smooth_iters, score_k, chunk_size))
    if mesh_smooth_iters == 1:
        raise ValueError("Use mesh_smooth_iters=0 or >=2.")
    if interface_smooth_iters is not None and (np.ndim(interface_smooth_iters) or
            not np.isfinite(interface_smooth_iters) or int(interface_smooth_iters) != interface_smooth_iters
            or interface_smooth_iters < 0 or interface_smooth_iters == 1):
        raise ValueError("interface_smooth_iters must be None, 0 or an integer >=2.")
    if not isinstance(workers, (int, np.integer)) or workers == 0 or workers < -1:
        raise ValueError("workers must be -1 or a positive integer.")
    for name, value in (("max_shift_scale", max_shift_scale),
            ("interface_max_shift_scale", interface_max_shift_scale),
            ("label_guard_scale", label_guard_scale),
            ("component_diameter_scale", component_diameter_scale)):
        if value is not None and (np.ndim(value) or not np.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite nonnegative or None.")
    if label_guard_scale is None:
        raise ValueError("label_guard_scale must be a finite nonnegative scalar.")
    for name, value in (("min_component", min_component), ("pre_min_component", pre_min_component)):
        if (value is None and name == "min_component") or (value is not None and (np.ndim(value) or not np.isfinite(value) or not 0 <= value <= 1)):
            raise ValueError(f"{name} must lie in [0,1].")
    if np.ndim(hole_scale) or np.isnan(hole_scale) or hole_scale < 0:
        raise ValueError("hole_scale must be nonnegative or inf.")
    if np.ndim(offset_scale) or not np.isfinite(offset_scale):
        raise ValueError("offset_scale must be finite.")
    if not np.isfinite(mesh_smooth_pass_band) or not 0 < mesh_smooth_pass_band <= 2:
        raise ValueError("mesh_smooth_pass_band must lie in (0,2].")
    if not np.isfinite(label_tol) or label_tol <= 0:
        raise ValueError("label_tol must be finite and positive.")
    stages = np.atleast_1d(np.asarray(smooth_scales, float))
    if stages.ndim != 1 or not len(stages) or not np.isfinite(stages).all() or np.any(stages <= 0):
        raise ValueError("smooth_scales must be a nonempty positive finite sequence.")
    opening_scale, closing_scale, score_scale = (_xyz(opening_scale, "opening_scale"),
        _xyz(closing_scale, "closing_scale"), _xyz(score_scale, "score_scale", True))
    spacing = None if spacing is None else _xyz(spacing, "spacing", True)
    smooth_sigma = None if smooth_sigma is None else _xyz(smooth_sigma, "smooth_sigma")
    if component_diameter_scale is None:
        raise ValueError("component_diameter_scale must be nonnegative.")
    smooth_scale = smooth_scale * math.sqrt(field_smooth_iters)
    if smooth_sigma is not None:
        smooth_sigma = smooth_sigma * math.sqrt(field_smooth_iters)
    gap_size = None if gap_size is None else _xyz(gap_size, "gap_size")
    if label_smooth is not None and (label_smooth_weight is not None or label_axis_weights is not None):
        raise ValueError("Use label_smooth or legacy label_smooth_weight/label_axis_weights, not both.")
    label_smooth = _xyz(.5 if label_smooth is None else label_smooth, "label_smooth")
    if label_smooth_weight is not None and (not np.isfinite(label_smooth_weight) or label_smooth_weight < 0):
        raise ValueError("label_smooth_weight must be finite nonnegative.")
    if label_axis_weights is not None:
        label_axis_weights = _xyz(label_axis_weights, "label_axis_weights", True)

    unique_source = data["unique_source"]
    offset = np.median(x[unique_source], axis=0)
    geometry = dict(k=k, outlier_method=outlier_method, outlier_thresh=outlier_thresh,
        outlier_k=outlier_k, outlier_iters=outlier_iters, outlier_radius=outlier_radius,
        min_points=min_points, min_component=min_component, graph_scale=graph_scale,
        radius_cap=radius_cap, base_spacing=base_spacing, base_spacing_scale=base_spacing_scale,
        adaptive=adaptive, support_scale=support_scale, anisotropy=anisotropy,
        chunk_size=chunk_size, workers=workers, support_smooth=support_smooth)
    retained, d, precision, axes, radii = _prepare_geometry(
        x[unique_source] - offset, c[unique_source], **geometry)
    observation_data = _cloud_label_observations(data, retained, offset=offset,
        label_outlier_action=label_outlier_action, label_outlier_k=label_outlier_k,
        label_outlier_thresh=label_outlier_thresh, label_outlier_iters=label_outlier_iters,
        label_min_points=label_min_points, graph_scale=graph_scale, workers=workers)
    site_feature = _site_features(data, observation_data)
    unique_source = observation_data["unique_source"]
    observations, obs_site = observation_data["observations"], observation_data["obs_site"]
    p = x[unique_source] - offset
    corrected = codes if label is None else observation_data["corrected_codes"]
    label_weights = observation_data["label_reliability"][observations]
    label_report = observation_data["label_report"]
    if label is not None and not np.any(label_weights > 0):
        raise ValueError("No positive label evidence remains; check label_confidence.")
    observed_codes = corrected[observations]
    observed_weight = observation_data["weights"]
    info = dict(inlier_mask=observation_data["inlier_mask"], retained_indices=unique_source.copy(),
        label_corrected_id=corrected.copy() if label is not None else None,
        label_reliability=observation_data["label_reliability"],
        label_outlier_mask=observation_data["label_outlier_mask"],
        label_corrected=_category_values(keys, corrected, for_vtk=False) if label is not None else None)
    common_meta = dict(base_spacing_reference=[d / base_spacing_scale],
        base_spacing_scale=[base_spacing_scale], support_scale=support_scale[None],
        support_radius=(d * support_scale)[None], support_max_semiaxis=[float(axes.max())],
        anisotropy=[float(anisotropy)], outlier_method=[str(outlier_method)],
        outlier_k=[outlier_k], outlier_iters=[outlier_iters], retained_points=[len(p)],
        global_retained_points=[len(p)],
        label_mode=[label_mode if label is not None else "unlabelled"],
        label_changed_points=[int(label_report["changed_mask"].sum())],
        label_outlier_points=[int(label_report["outlier_mask"].sum())],
        mesh_smooth_iters=[mesh_smooth_iters], mesh_smooth_pass_band=[mesh_smooth_pass_band],
        field_smooth_iters=[int(field_smooth_iters)], min_volume_component=[min_component],
        component_diameter_scale=[component_diameter_scale],
        support_smooth=[float(support_smooth)], field_smooth_mode=[field_smooth_mode])
    reconstruction = dict(smooth_scales=stages, preserve_volume=preserve_volume,
        max_shift_scale=max_shift_scale, closing_scale=closing_scale,
        opening_scale=opening_scale, hole_scale=hole_scale, offset_scale=offset_scale,
        mesh_smooth_pass_band=mesh_smooth_pass_band, workers=workers,
        pre_min_component=pre_min_component, component_diameter_scale=component_diameter_scale,
        field_smooth_mode=field_smooth_mode)

    def reconstruct(ids, local_d=d, local_q=None, local_a=None, guards=None, fields=False):
        return _reconstruct(p[ids], precision[ids] if local_q is None else local_q,
            axes[ids] if local_a is None else local_a, local_d, smooth_scale,
            gap_size, spacing, min_component, mesh_smooth_iters, backend, device,
            int(max_voxels), chunk_size, spacing_scale, method, density_level,
            smooth_sigma, return_fields=fields, protect_points=guards,
            protect_radius=local_d * label_guard_scale, **reconstruction)

    def annotate(mesh, ids, local_d=d, constant=None):
        if not mesh.n_points:
            return mesh
        _annotate_clay_mesh(mesh, p[ids], unique_source[ids], c,
            None if site_feature is None else site_feature[ids],
            local_d, score_k, score_scale, chunk_size, workers)
        if label is not None and constant is None and label_mode != "joint" and not shared_separate:
            winner, certainty = _project_label_evidence(mesh.points,
                x[observations] - offset, observed_codes, observed_weight,
                label_project_k, local_d, workers, chunk_size)
            mesh.point_data["label_id"] = winner
            mesh.point_data["label"] = _category_values(keys, winner)
            mesh.point_data["label_confidence"] = certainty
            # Face labels are projected at face centres, rather than majority
            # voting already interpolated vertex labels across tissue boundaries.
            ci, _ = _project_label_evidence(mesh.cell_centers().points,
                x[observations] - offset, observed_codes, observed_weight,
                label_project_k, local_d, workers, chunk_size)
            mesh.cell_data["label_id"] = ci
            mesh.cell_data["label"] = _category_values(keys, ci)
        elif constant is not None:
            mesh.point_data["label_id"] = np.full(mesh.n_points, constant, np.int32)
            mesh.point_data["label"] = _category_values(keys, np.full(mesh.n_points, constant))
            _, certainty = _project_label_evidence(mesh.points,
                x[observations] - offset, observed_codes, observed_weight,
                label_project_k, local_d, workers, chunk_size, target=constant)
            mesh.point_data["label_confidence"] = certainty
            mesh.cell_data["label_id"] = np.full(mesh.n_cells, constant, np.int32)
            mesh.cell_data["label"] = _category_values(keys, np.full(mesh.n_cells, constant))
        mesh.points = np.asarray(mesh.points, float) + offset
        for name, value in common_meta.items():
            mesh.field_data[name] = np.atleast_1d(value)
        mesh.field_data["base_spacing"] = [local_d]
        mesh.field_data["base_spacing_reference"] = [local_d / base_spacing_scale]
        mesh.field_data["retained_points"] = [len(ids)]
        mesh.field_data["support_radius"] = (local_d * support_scale)[None]
        return mesh

    def class_fields_on(mask, origin, step, selected):
        # Private continuous fields retain their support/morphology model. Only
        # their final categorical ownership is shared with neighbouring tissues.
        for code, key in enumerate(keys):
            use = selected[(observed_codes[selected] == code) & (observed_weight[selected] > 0)]
            ids = np.unique(obs_site[use])
            if not len(ids):
                continue
            counts = np.bincount(obs_site[use], minlength=len(p))
            reliability = np.bincount(obs_site[use], weights=label_weights[use], minlength=len(p))
            reliability /= np.maximum(counts, 1)
            factor = np.cbrt(reliability[ids])
            local_d = d
            q, semiaxes = precision[ids] / factor[:, None, None]**2, axes[ids] * factor[:, None]
            if separate_geometry == "local" and len(ids) >= 4:
                geom = c[unique_source[ids]] * reliability[ids]
                kept, local_d, q, semiaxes, local_radii = _prepare_geometry(p[ids], geom, **geometry)
                ids = ids[kept]
            sigma = local_d * smooth_scale if smooth_sigma is None else smooth_sigma
            close = closing_scale if gap_size is None else gap_size / local_d
            margin = (semiaxes.max() + 4*sigma*stages.sum() + 4*d*label_smooth
                      + local_d*(close+opening_scale+abs(offset_scale)))
            shape, local_origin = _label_grid_halo(mask.shape, origin, step, margin, max_voxels)
            private = _support_mask(p[ids], q, semiaxes, local_origin, step, shape, chunk_size, workers)
            private, field, report = _process_volume(private, step, local_d,
                smooth_sigma=sigma, smooth_scales=stages, smooth_method=method,
                density_level=density_level, preserve_volume=preserve_volume,
                max_shift_scale=max_shift_scale, opening_scale=opening_scale,
                closing_scale=close, hole_scale=hole_scale, offset_scale=offset_scale,
                pre_min_component=min_component if pre_min_component is None else pre_min_component,
                min_volume_component=min_component, component_diameter_scale=component_diameter_scale,
                keep_largest=min_component == 1, field_smooth_mode=field_smooth_mode, device=device)
            yield code, field, bool(private.any()), p[ids], report, local_origin, step

    all_ids = np.arange(len(p))
    if label is None or label_mode == "project":
        if label is not None and np.any(post_smooth):
            parts, metadata = reconstruct(all_ids, fields=True)
            step = np.asarray(metadata["spacing"]).reshape(3)
            eligible = np.flatnonzero(observed_weight > 0)
            local_obs_points = x[observations] - offset
            owner, _, _, _ = _assign_component_observations(parts, local_obs_points[eligible],
                step, d, workers, chunk_size)
            labelled_parts, reports = [], []
            for part_id, (_, mask, origin, _) in enumerate(parts):
                rows = eligible[owner == part_id]
                obs = dict(keys=keys, codes=observed_codes[rows], weights=observed_weight[rows],
                           observation_points=local_obs_points[rows])
                assigned = _grid_label_data(mask, origin, step, obs, mode="project", d=d,
                    k=label_project_k, workers=workers, chunk_size=chunk_size,
                    label_post_smooth=post_smooth, label_lock_seeds=label_lock_seeds,
                    label_max_iter=label_max_iter, label_tol=label_tol)
                labelled_parts.append((assigned["label_mask"], assigned["label_score_image"], origin + offset))
                reports.append(assigned["label_info"])
            mesh = _mesh_field_parts(parts, metadata, step, backend, device,
                                    mesh_smooth_iters, mesh_smooth_pass_band)
            result = annotate(mesh, all_ids)
            _mesh_partition_labels(result, labelled_parts, None, None, step, keys, workers, chunk_size)
            info["label_info"] = _summarize_label_parts(reports, d*post_smooth)
        else:
            result = annotate(reconstruct(all_ids), all_ids)
    elif label_mode == "separate" and not shared_separate:
        # Only uncontested, reliable foreign sites become morphology guards.
        dominant, trusted = _label_guard_sites(p, obs_site, observed_codes, observed_weight,
            k=label_outlier_k, graph_scale=graph_scale, workers=workers)
        result = {}
        for code, key in enumerate(keys):
            selected = (observed_codes == code) & (observed_weight > 0)
            ids = np.unique(obs_site[selected])
            if len(ids) < 4:
                warnings.warn(f"Label {key!r} has fewer than four retained sites; returning an empty mesh.", stacklevel=2)
                result[key] = pv.PolyData()
                continue
            counts_for_label = np.bincount(obs_site[selected], minlength=len(p))
            reliability_for_label = np.bincount(obs_site[selected],
                weights=label_weights[selected], minlength=len(p))
            reliability_for_label = np.divide(reliability_for_label, counts_for_label,
                out=np.zeros_like(reliability_for_label), where=counts_for_label > 0)
            factor = np.cbrt(reliability_for_label[ids])
            local_d = d
            local_q = precision[ids] / factor[:, None, None]**2
            local_a = axes[ids] * factor[:, None]
            if separate_geometry == "local":
                ix, local_d, local_q, local_a, _ = _prepare_geometry(p[ids],
                    c[unique_source[ids]] * reliability_for_label[ids], **geometry)
                ids = ids[ix]
            guards = (p[trusted & (dominant != code)]
                      if separate_protect_labels and label_guard_scale > 0 else None)
            try:
                mesh = reconstruct(ids, local_d, local_q, local_a, guards)
            except ValueError as error:
                raise ValueError(f"Cannot reconstruct label {key!r}: {error}") from error
            result[key] = annotate(mesh, ids, local_d, code)
        if return_outer:
            result = (result, annotate(reconstruct(all_ids), all_ids))
    else:
        parts, metadata = reconstruct(all_ids, fields=True)
        partition_reports = []
        if shared_separate:
            dominant, trusted = _label_guard_sites(p, obs_site, observed_codes, observed_weight,
                k=label_outlier_k, graph_scale=graph_scale, workers=workers)
        group_meshes = {key: [] for key in keys}
        outer_meshes = []
        point_offset = face_offset = 0
        total_counts = np.zeros(len(keys), np.int64)
        iterations, locked_voxels, residual = 0, 0, 0.0
        seed_voxels = conflict_voxels = projected_points = 0
        surface_counts = {name: 0 for name in ("interface_free_vertices",
            "interface_fixed_vertices", "interface_junction_vertices",
            "mesh_limited_vertices", "interface_limited_vertices")}
        step = np.asarray(metadata["spacing"]).reshape(3)
        eligible = np.flatnonzero(observed_weight > 0)
        obs_points = x[observations] - offset
        owner, assigned, projected, ownership_meta = _assign_component_observations(
            parts, obs_points[eligible], step, d, workers, chunk_size)
        for part_id, (_, mask, origin, field) in enumerate(parts):
            owned = owner == part_id
            selected = eligible[owned]
            if not len(selected):
                raise ValueError(f"Occupied component {part_id + 1} has no owned label observations; "
                                 "inspect support, filtering and label confidence.")
            if shared_separate:
                part_obs = dict(observation_data, points=p, observation_points=obs_points[selected],
                    codes=observed_codes[selected], weights=observed_weight[selected], obs_site=obs_site[selected])
                labelled = _separate_label_volume(mask, origin, step, d,
                    origin + np.argwhere(mask)*step, part_obs,
                    lambda: class_fields_on(mask, origin, step, selected),
                    label_smooth=label_smooth, label_smooth_weight=label_smooth_weight,
                    label_axis_weights=label_axis_weights, label_lock_seeds=label_lock_seeds,
                    site_codes=dominant, site_trusted=trusted,
                    guard_radius=d*label_guard_scale if separate_protect_labels else 0.,
                    workers=workers, chunk_size=chunk_size, device=device,
                    guard_mode=label_guard_mode, post_smooth=d*post_smooth,
                    label_max_iter=label_max_iter, label_tol=label_tol)
                labels, assignment, label_meta = (labelled["label_mask"],
                    labelled["label_score_image"], labelled["label_info"])
            else:
                labels, assignment, label_meta = _label_volume(mask, origin, step,
                    obs_points[selected], observed_codes[selected], observed_weight[selected], d,
                    label_smooth, k, workers, label_max_iter=int(label_max_iter),
                    label_tol=label_tol, label_lock_seeds=label_lock_seeds,
                    label_smooth_weight=label_smooth_weight, label_axis_weights=label_axis_weights,
                    chunk_size=chunk_size, observation_voxel=assigned[owned],
                    observation_projected=projected[owned], label_post_smooth=post_smooth)
            if shared_separate or np.any(post_smooth):
                partition_reports.append(label_meta)
            total_counts += np.bincount(labels.ravel(), minlength=len(keys)+1)[1:]
            iterations = max(iterations, int(label_meta.get("label_iterations", 0)))
            locked_voxels += int(label_meta.get("label_locked_voxels", 0))
            residual = max(residual, float(label_meta.get("label_residual", 0)))
            seed_voxels += int(label_meta.get("label_seed_voxels", 0))
            conflict_voxels += int(label_meta.get("label_conflict_voxels", 0))
            projected_points += int(label_meta.get("label_projected_points", 0))
            meshes, outer = _mesh_from_labels(labels, field, origin, step, keys,
                mesh_smooth_iters, mesh_smooth_pass_band, interface_backend=interface_backend,
                interface_smooth_iters=interface_smooth_iters,
                interface_max_shift_scale=interface_max_shift_scale, d=d,
                interface_smooth_pass_band=interface_smooth_pass_band,
                split_normal_patches=split_normal_patches)
            for name in surface_counts:
                surface_counts[name] += int(outer.field_data[name][0])
            models = [m for m in [outer, *meshes.values()] if m.n_points and m.n_cells]
            if not models:
                raise RuntimeError(f"Occupied component {part_id + 1} produced no surface cells.")
            npoints = 1 + max(int(m.point_data["global_point_id"].max()) for m in models)
            nfaces = 1 + max(int(m.cell_data["global_face_id"].max()) for m in models)
            for mesh in models:
                ijk = ((mesh.points - origin) / step).T
                mass = ndi.map_coordinates(mask.astype(np.float32), ijk, order=1,
                                            mode="constant", cval=0, prefilter=False)
                score = ndi.map_coordinates(assignment, ijk, order=1,
                                             mode="constant", cval=0, prefilter=False)
                mesh.point_data["label_confidence"] = np.clip(np.divide(score, mass,
                    out=np.zeros_like(score), where=mass > 1e-8), 0, 1)
                mesh.point_data["global_point_id"] += point_offset
                mesh.cell_data["global_face_id"] += face_offset
            for key, mesh in meshes.items():
                if mesh.n_cells:
                    group_meshes[key].append(mesh)
            if outer.n_cells:
                outer_meshes.append(outer)
            point_offset += npoints
            face_offset += nfaces
        def merge_parts(items):
            return (pv.merge(items, merge_points=False) if len(items) > 1
                    else items[0] if items else pv.PolyData())
        outer = merge_parts(outer_meshes)
        result = {key: merge_parts(items) for key, items in group_meshes.items()}
        common_meta.update(metadata)
        info.update(surface_counts)
        common_meta.update({name: [value] for name, value in surface_counts.items()})
        unused = ownership_meta["label_unused_points"] + len(observed_weight) - len(eligible)
        info.update(label_unused_points=int(unused),
                    label_owned_observations=int(ownership_meta["label_owned_observations"]),
                    label_projected_points=int(projected_points))
        common_meta.update(label_unused_points=[int(unused)],
            label_owned_observations=[int(ownership_meta["label_owned_observations"])],
            label_projection_reach=[float(ownership_meta["label_projection_reach"])])
        common_meta.update(label_iterations=[iterations], label_voxel_counts=total_counts,
            label_volumes=total_counts * np.prod(np.asarray(metadata["spacing"]).reshape(3)),
            label_smooth=(d * label_smooth)[None], label_residual=[residual],
            label_locked_voxels=[locked_voxels], label_hard_seed_voxels=[locked_voxels],
            label_seed_voxels=[seed_voxels], label_conflict_voxels=[conflict_voxels],
            label_projected_points=[projected_points], label_converged=[True],
            label_graph_weights=np.asarray(label_meta.get("label_graph_weights", np.zeros(3)))[None])
        if label_smooth_weight is not None or label_axis_weights is not None:
            common_meta["label_smooth_weight"] = [1.0 if label_smooth_weight is None else label_smooth_weight]
            common_meta["label_axis_weights"] = np.atleast_2d(np.ones(3) if label_axis_weights is None else label_axis_weights)
        if partition_reports:
            info["label_info"] = _summarize_label_parts(partition_reports, d*post_smooth)
            info["separate_partition"] = "shared" if shared_separate else "joint"
            common_meta.update(_numeric_label_metadata(info["label_info"]))
        for mesh in [outer, *result.values()]:
            annotate(mesh, all_ids)
        absent = [str(key) for key, mesh in result.items() if not mesh.n_cells]
        if absent:
            warnings.warn("No reconstructed region for label(s): " + ", ".join(absent)
                          + "; inspect label cleanup/smoothing and sampling.", stacklevel=2)
        if return_outer:
            result = (result, outer)
    return (result, info) if return_info else result


def _xyz(value, name, positive=False):
    a = np.broadcast_to(np.asarray(value, float), (3,)).copy()
    if not np.isfinite(a).all() or np.any(a <= 0 if positive else a < 0):
        raise ValueError(f"{name} must be finite, scalar or XYZ, and "
                         + ("positive." if positive else "nonnegative."))
    return a


def _morph(mask, radius, spacing, operation):
    """EDT ellipsoid morphology; zero-radius axes never exchange voxels."""
    radius = _xyz(radius, "radius")
    spacing = _xyz(spacing, "spacing", True)
    if operation not in {"dilate", "erode"}:
        raise ValueError("operation must be 'dilate' or 'erode'.")
    mask = np.asarray(mask, dtype=bool)
    active = np.flatnonzero(radius > 0)
    if not len(active) or not mask.any():
        return mask.copy()
    fixed = np.flatnonzero(radius == 0)
    sampling = spacing[active] / radius[active]
    output = np.empty_like(mask)
    for index in np.ndindex(*(mask.shape[a] for a in fixed)):
        sl = [slice(None)] * 3
        for a, i in zip(fixed, index):
            sl[a] = i
        sl = tuple(sl)
        part = mask[sl]
        if not part.any():
            output[sl] = False
        elif operation == "dilate":
            output[sl] = ndi.distance_transform_edt(~part, sampling=sampling) <= 1.0
        else:
            # Treat outside the array as background, including an all-True input.
            distance = ndi.distance_transform_edt(np.pad(part, 1), sampling=sampling)
            core = (slice(1, -1),) * part.ndim
            output[sl] = distance[core] > 1.0
    return output


def _support_mask_3(x, radii, axes, origin, spacing, shape, chunk, workers):
    """Exact union of variable-radius ellipsoids at queried voxel centers."""
    anchor = x.min(0)
    tree = cKDTree((x - anchor) / axes)
    limit = np.nextafter(float(radii.max()), np.inf)
    mask = np.zeros(math.prod(shape), bool)
    for start in range(0, len(mask), chunk):
        stop = min(start + chunk, len(mask))
        ijk = np.column_stack(np.unravel_index(np.arange(start, stop), shape))
        q = ((origin - anchor) + ijk * spacing) / axes
        dd, ids = tree.query(q, workers=workers)
        inside = dd <= radii[ids]
        pending = np.flatnonzero(~inside & (dd <= limit))
        count = min(8, len(x))
        while len(pending):
            remaining = []
            batch = max(1, min(chunk, 1_000_000 // count))
            for lo in range(0, len(pending), batch):
                at = pending[lo:lo + batch]
                dd, ids = tree.query(q[at], k=list(range(1, count + 1)),
                                     distance_upper_bound=limit, workers=workers)
                valid = ids < len(x)
                found = np.any(valid & (dd <= radii[np.minimum(ids, len(x) - 1)]), axis=1)
                inside[at[found]] = True
                if count < len(x):
                    remaining.append(at[~found & np.isfinite(dd[:, -1])])
            pending = np.concatenate(remaining) if remaining else np.empty(0, int)
            count = min(2 * count, len(x))
        mask[start:stop] = inside
    return mask.reshape(shape)


def points_to_image(
    points, *, spacing_scale=0.5, spacing=None, base_spacing=None,
    base_spacing_scale=1.0, method="sdf", support_scale=1.5, adaptive=True,
    support_k=8, radius_cap=2.0, smooth_scale=1.0, smooth_sigma=None,
    smooth_iters=None, preserve_volume=True, max_shift_scale=2.0,
    density_level=0.5, closing_scale=0.0, opening_scale=0.0,
    hole_scale=0.0, offset_scale=0.0, outlier_k=24, outlier_thresh=1.8,
    outlier_iters=1, min_points=0.01, min_voxels=0, component_scale=None,
    keep_largest=False, score_k=8, score_scale=2.0, padding=2, bounds=None,
    max_voxels=20_000_000_000_000, chunk_size=100_000, workers=-1,
    label=None, confidence=None, label_confidence=None, feature=None,
    label_mode="joint", outlier_method="lof", outlier_radius=None,
    graph_scale=2.0, anisotropy=1.0, field_smooth_iters=None,
    smooth_scales=(1.0,), min_volume_component=0.0, pre_min_component=0.0,
    component_diameter_scale=None, label_outlier_action="relabel",
    label_outlier_k=24, label_outlier_thresh=0.8, label_outlier_iters=1,
    label_min_points=3, label_smooth=None, label_smooth_weight=None,
    label_axis_weights=None, label_lock_seeds=False, label_max_iter=1000,
    label_tol=1e-5, label_project_k=1, separate_geometry="shared",
    separate_protect_labels=True, label_guard_scale=0.6,
    return_info=False, device="cpu", separate_output="merged",
    field_smooth_mode="continuous", support_smooth=0.0, return_field=False,
    label_guard_mode="hard", label_post_smooth=0.0,
    mesh_smooth_iters=0, mesh_smooth_pass_band=0.1,
    interface_backend="auto", interface_smooth_iters=None,
    interface_smooth_pass_band=None, interface_max_shift_scale=None,
    surface_smooth_scope="all",
):
    """Reconstruct one scored XYZ image with project/separate/joint labels.
    
    Returns mask:uint8, score_image:float32, spacing/origin:XYZ, occupied points,
    and score in np.argwhere(mask) order. Labels add label_mask:int32 (background
    0, class index+1), label_map and label_score_image. Input label 0 is a class.
    return_info=True returns (result, info); return_field also puts a positive-
    inside field in info. Labels/features and source IDs are aligned in info.
    
    Geometry is cleaned once using confidence; label_confidence/label_outlier_*
    affect class evidence. Dense/sparse features are attributes. Duplicate sites
    share support; their label evidence and confidence-weighted features remain.
    Effective d = estimated/supplied base_spacing * base_spacing_scale.
    spacing = d*spacing_scale unless explicitly supplied. support_scale stretches
    adaptive/PCA supports; support_smooth in [0,1] regularizes sampling radii only.
    
    sdf/density smooth distance/occupancy; support skips smoothing; occupancy
    deposits observed sites. sigma = d*smooth_scale or physical smooth_sigma,
    multiplied by sqrt(field_smooth_iters). smooth_iters is its IMAGE alias.
    smooth_scales are successive stages; continuous carries the field forward,
    reembed rebuilds it from changed masks. With no nonlinear constraints,
    sigma_total = sigma*sqrt(sum(smooth_scales**2)).
    Order: prefilter -> opening -> closing -> field smoothing -> offset -> holes
    -> final components. Opening/closing are XYZ diameters*d; hole_scale bounds
    26-connected enclosed cavity volume by sphere diameter*d. Positive offset
    expands. preserve_volume targets the post-morphology mask before smoothing;
    max_shift_scale bounds field changes in units of d, not mesh displacement.
    
    project votes at occupied nodes; joint additionally diffuses inside the mask;
    separate competes independent class fields on the common mask. Soft guards
    add a finite prior; hard guards and optional reliable seed locks constrain
    ownership. label_smooth and label_post_smooth are scalar/XYZ lengths divided
    by d. Post smoothing uses assignment-weighted diffusion, never integer-ID blur.
    separate_output='per_label' returns dictionaries: old independent grids when
    hard guards/no refinement are used, otherwise splits the shared partition.
    
    Optional mesh_smooth_iters/interface_smooth_iters (0 or >=2) smooth one shared
    surface and rasterize onto the same lattice; PyVista/VTK are then required.
    surface_smooth_scope='interfaces' fixes the common mask; 'all' permits shell
    motion. Pass bands lie in (0,2], smaller generally smooths more. Iteration
    counts are filter orders. interface_max_shift_scale caps internal motion/d.
    Local folds first reduce individual moves, then the common step if needed.
    Invalid raster updates additionally halve shared displacement. Persistent
    extraction or raster failures retain the pre-surface partition with a warning
    and surface_status='reverted'. See info['surface_info'] for the failure stage,
    raster move fraction (relative to the limited mesh), class losses and restored
    locks. After successful surface processing, label scores measure support for
    the selected class among observed neighbours; geometry scores stay separate.
    Outer changes resample source attributes and replace a requested field by
    binary EDT (field_source='post_surface_binary_edt'). Surface motion can change
    volumes/topology and is outside preserve_volume. Earlier info describes
    upstream stages. Explicit resolution is never silently coarsened."""

    time = _alias(field_smooth_iters, smooth_iters, 1, "field_smooth_iters", "smooth_iters")
    diameter = _alias(component_diameter_scale, component_scale, 0.0,
                      "component_diameter_scale", "component_scale")
    support = _xyz(support_scale, "support_scale", True)
    spacing_factor = _xyz(spacing_scale, "spacing_scale", True)
    smooth_factor = _xyz(smooth_scale, "smooth_scale")
    requested_spacing = None if spacing is None else _xyz(spacing, "spacing", True)
    requested_sigma = None if smooth_sigma is None else _xyz(smooth_sigma, "smooth_sigma")
    opening, closing = _xyz(opening_scale, "opening_scale"), _xyz(closing_scale, "closing_scale")
    score_axes, pad = _xyz(score_scale, "score_scale", True), _xyz(padding, "padding")
    if np.any(pad != np.floor(pad)):
        raise ValueError("padding must contain nonnegative integers.")
    if method not in {"sdf", "density", "support", "occupancy"}:
        raise ValueError("method must be sdf, density, support or occupancy.")
    if label_mode not in {"project", "joint", "separate"}:
        raise ValueError("label_mode must be project, joint or separate.")
    if separate_geometry not in {"shared", "local"}:
        raise ValueError("separate_geometry must be shared or local.")
    if separate_output not in {"merged", "per_label"}:
        raise ValueError("separate_output must be 'merged' or 'per_label'.")
    if label_guard_mode not in {"hard", "soft"}:
        raise ValueError("label_guard_mode must be 'hard' or 'soft'.")
    if surface_smooth_scope not in {"all", "interfaces"}:
        raise ValueError("surface_smooth_scope must be 'all' or 'interfaces'.")
    for name, value in (("mesh_smooth_iters", mesh_smooth_iters),
                        ("interface_smooth_iters", interface_smooth_iters)):
        if value is None and name == "interface_smooth_iters":
            continue
        if (np.ndim(value) or not np.isfinite(value) or int(value) != value
                or value < 0 or value == 1):
            raise ValueError(f"{name} must be 0 or an integer >=2 (interface also accepts None).")
    for name, value in (("mesh_smooth_pass_band", mesh_smooth_pass_band),
                        ("interface_smooth_pass_band", interface_smooth_pass_band)):
        if value is None and name == "interface_smooth_pass_band":
            continue
        if np.ndim(value) or not np.isfinite(value) or not 0 < value <= 2:
            raise ValueError(f"{name} must be finite in (0,2].")
    if interface_backend not in {"auto", "surface_nets", "voxel_faces"}:
        raise ValueError("interface_backend must be 'auto', 'surface_nets' or 'voxel_faces'.")
    if interface_max_shift_scale is not None and (np.ndim(interface_max_shift_scale)
            or not np.isfinite(interface_max_shift_scale) or interface_max_shift_scale < 0):
        raise ValueError("interface_max_shift_scale must be nonnegative finite or None.")
    surface_active = mesh_smooth_iters > 0 or (interface_smooth_iters or 0) > 0
    post_smooth = _xyz(label_post_smooth, "label_post_smooth")
    if field_smooth_mode not in {"continuous", "reembed"}:
        raise ValueError("field_smooth_mode must be 'continuous' or 'reembed'.")
    if np.ndim(support_smooth) or not np.isfinite(support_smooth) or not 0 <= support_smooth <= 1:
        raise ValueError("support_smooth must be finite in [0,1].")
    if not isinstance(return_field, (bool, np.bool_)):
        raise ValueError("return_field must be a bool.")
    if return_field and not return_info:
        raise ValueError("return_field=True requires return_info=True; the field is returned in info.")
    for name, value in (("label_lock_seeds", label_lock_seeds),
                        ("separate_protect_labels", separate_protect_labels)):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a bool.")
    if label_smooth is not None and (label_smooth_weight is not None or label_axis_weights is not None):
        raise ValueError("Use label_smooth or legacy label_smooth_weight/label_axis_weights, not both.")
    label_smooth = _xyz(0.5 if label_smooth is None else label_smooth, "label_smooth")
    for name, value, minimum in (("support_k", support_k, 1), ("outlier_k", outlier_k, 2),
            ("outlier_iters", outlier_iters, 0), ("field_smooth_iters", time, 0),
            ("min_voxels", min_voxels, 0), ("score_k", score_k, 1),
            ("chunk_size", chunk_size, 1), ("max_voxels", max_voxels, 1),
            ("label_project_k", label_project_k, 1)):
        if np.ndim(value) or not np.isfinite(value) or int(value) != value or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    support_k, outlier_k, time = int(support_k), int(outlier_k), int(time)
    chunk_size, score_k = int(chunk_size), int(score_k)
    for name, value in (("component_diameter_scale", diameter), ("label_guard_scale", label_guard_scale)):
        if np.ndim(value) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    stages = np.atleast_1d(np.asarray(smooth_scales, float))
    if stages.ndim != 1 or not len(stages) or not np.isfinite(stages).all() or np.any(stages <= 0):
        raise ValueError("smooth_scales must be a nonempty positive finite sequence.")
    if not isinstance(workers, (int, np.integer)) or workers == 0 or workers < -1:
        raise ValueError("workers must be -1 or a positive integer.")
    data = _cloud_inputs(points, label, confidence, label_confidence, feature, min_points=2)
    original_sites = data["unique_source"]
    center = np.median(data["points"][original_sites], axis=0)
    geometry_options = dict(k=support_k, outlier_method=outlier_method,
        outlier_thresh=0 if outlier_thresh is None else outlier_thresh,
        outlier_k=outlier_k, outlier_iters=outlier_iters, outlier_radius=outlier_radius,
        min_points=min_points, min_component=0, graph_scale=graph_scale,
        radius_cap=radius_cap, base_spacing=base_spacing, base_spacing_scale=base_spacing_scale,
        adaptive=adaptive, support_scale=support, anisotropy=anisotropy,
        chunk_size=chunk_size, workers=workers, min_survivors=2, support_smooth=support_smooth)
    retained, d, precision, axes, radii = _prepare_geometry(
        data["points"][original_sites] - center, data["confidence"][original_sites],
        **geometry_options)
    observations = _cloud_label_observations(data, retained, offset=center,
        label_outlier_action=label_outlier_action, label_outlier_k=label_outlier_k,
        label_outlier_thresh=label_outlier_thresh, label_outlier_iters=label_outlier_iters,
        label_min_points=label_min_points, graph_scale=graph_scale, workers=workers)
    site_feature = _site_features(data, observations)
    sources = observations["unique_source"]
    x, geom_conf = data["points"][sources], data["confidence"][sources]
    diagnostics = {name: observations[name] for name in
        ("inlier_mask", "label_reliability", "label_outlier_mask", "label_changed_mask")}
    if data["keys"] is not None:
        names = np.empty(len(data["keys"]), object)
        names[:] = data["keys"]
        diagnostics["label_corrected"] = names[observations["corrected_codes"]]
    site_codes, site_trusted = np.full(len(x), -1, np.int64), np.zeros(len(x), bool)
    if data["keys"] is not None and label_mode == "separate":
        site_codes, site_trusted = _label_guard_sites(x, observations["obs_site"],
            observations["codes"], observations["weights"], k=label_outlier_k,
            graph_scale=graph_scale, workers=workers)

    def build(ids, code=None, *, fixed_grid=None, field_only=False, attach_labels=True):
        p, q, semiaxes, local_d = x[ids], precision[ids], axes[ids], d
        local_sources, local_conf = sources[ids], geom_conf[ids]
        local_feature = None if site_feature is None else site_feature[ids]
        local_obs = observations
        class_reliability = np.ones(len(ids))
        if code is not None:
            use = (observations["codes"] == code) & (observations["weights"] > 0)
            local_obs = dict(observations)
            for name in ("observation_points", "codes", "weights"):
                local_obs[name] = observations[name][use]
            mass = np.bincount(observations["obs_site"][use],
                weights=observations["weights"][use], minlength=len(x))
            geometric = np.bincount(observations["obs_site"][use],
                weights=data["confidence"][observations["observations"][use]], minlength=len(x))
            class_reliability = np.clip(mass[ids] / geometric[ids], 0, 1)
            factor = np.cbrt(class_reliability)
            q, semiaxes = q / factor[:, None, None] ** 2, semiaxes * factor[:, None]
        if code is not None and separate_geometry == "local" and len(ids) >= 2:
            local_options = dict(geometry_options, outlier_iters=0, min_points=0)
            if fixed_grid is not None:
                local_options["base_spacing"] = d / base_spacing_scale
            _, local_d, q, semiaxes, local_radii = _prepare_geometry(
                p - center, local_conf * class_reliability, **local_options)
        step = local_d * spacing_factor if requested_spacing is None else requested_spacing.copy()
        sigma = local_d * smooth_factor if requested_sigma is None else requested_sigma.copy()
        effective_sigma = sigma * np.sqrt(time)
        field_method = method if method in {"sdf", "density"} and time > 0 else None
        if field_method is None or max_shift_scale == 0:
            effective_sigma = np.zeros(3)
        if fixed_grid is None:
            lower, upper = p.min(axis=0), p.max(axis=0)
            if bounds is not None:
                box = np.asarray(bounds, float)
                if box.shape != (2, 3) or not np.isfinite(box).all() or np.any(box[1] < box[0]):
                    raise ValueError("bounds must be finite (2,3), lower <= upper.")
                if np.any(lower < box[0]) or np.any(upper > box[1]):
                    raise ValueError("bounds must enclose all retained points.")
                lower, upper = box
            support_extent = np.zeros(3) if method == "occupancy" else np.sqrt(
                np.max(np.linalg.inv(q).diagonal(axis1=1, axis2=2), axis=0))
            support_extent = np.maximum(support_extent,
                local_d * max(1.0, radius_cap) * support * anisotropy ** (2.0 / 3.0)) if method != "occupancy" else support_extent
            margin = (support_extent + local_d * (opening + closing) + abs(offset_scale) * local_d
                      + _gaussian_halo(effective_sigma, stages, step) + 2 * step)
            if np.any(effective_sigma) and max_shift_scale is not None:
                margin += local_d * max_shift_scale
            layers = np.ceil(margin / step) + pad
            dimensions = np.ceil((upper - lower) / step) + 1 + 2 * layers
            if not np.isfinite(dimensions).all() or np.any(dimensions > max_voxels):
                raise MemoryError("Grid dimensions exceed max_voxels; increase spacing or reduce the domain.")
            shape = tuple(int(v) for v in dimensions)
            if math.prod(shape) > max_voxels:
                raise MemoryError(f"Grid {shape} exceeds max_voxels={max_voxels}; increase spacing.")
            origin = lower - layers * step
        else:
            origin, step, shape = fixed_grid
        mask = (np.zeros(shape, bool) if method == "occupancy" else
                _support_mask(p - center, q, semiaxes, origin - center, step,
                              shape, chunk_size, workers))
        voxel = np.floor((p - origin) / step + 0.5).astype(np.int64)
        mask[tuple(voxel.T)] = True
        forbidden = None
        if code is not None and not field_only and separate_protect_labels and label_guard_scale > 0:
            foreign = site_trusted & (site_codes != code)
            if np.any(foreign):
                tree = cKDTree(x[foreign] - center)
                forbidden = np.zeros(shape, bool)
                for start in range(0, mask.size, chunk_size):
                    stop = min(start + chunk_size, mask.size)
                    at = np.column_stack(np.unravel_index(np.arange(start, stop), shape))
                    distance = tree.query(origin - center + at * step, workers=workers)[0]
                    forbidden.ravel()[start:stop] = distance <= d * label_guard_scale
        mask, field, volume_info = _process_volume(mask, step, local_d,
            smooth_sigma=effective_sigma, smooth_scales=stages, smooth_method=field_method,
            density_level=density_level, preserve_volume=preserve_volume,
            max_shift_scale=max_shift_scale, opening_scale=opening,
            closing_scale=closing, hole_scale=hole_scale, offset_scale=offset_scale,
            min_volume_component=min_volume_component, pre_min_component=pre_min_component,
            min_voxels=min_voxels, component_diameter_scale=diameter,
            keep_largest=keep_largest, forbidden=forbidden, device=device,
            component_connectivity=26, hole_stage="after_offset", offset_method="morphology",
            field_smooth_mode=field_smooth_mode)
        if field_only:
            return field, bool(mask.any()), volume_info
        result, sampled = _image_point_data(mask, origin, step, local_d,
            p, local_conf, local_sources, local_feature, score_axes,
            score_k, chunk_size, workers)
        indices = np.flatnonzero(mask)
        info = dict(diagnostics, base_spacing=local_d,
                    base_spacing_reference=local_d / base_spacing_scale,
                    base_spacing_scale=base_spacing_scale, smooth_sigma=sigma,
                    effective_smooth_sigma=effective_sigma, **sampled,
                    source_indices=local_sources.copy(), volume_info=volume_info,
                    method=method, label_mode=label_mode, support_smooth=float(support_smooth),
                    field_smooth_mode=field_smooth_mode,
                    continuous_sigma_budget=effective_sigma * np.sqrt(np.sum(stages ** 2)))
        if return_field or surface_active:
            info.update(field=field, field_inside="positive")
        if data["keys"] is not None and attach_labels:
            assigned = _grid_label_data(mask, origin, step, local_obs,
                mode="project" if code is not None else label_mode, d=local_d,
                label_smooth=label_smooth, k=label_project_k if code is not None or label_mode == "project" else support_k,
                workers=workers, point_indices=indices, label_max_iter=label_max_iter,
                label_tol=label_tol, label_lock_seeds=label_lock_seeds,
                label_smooth_weight=label_smooth_weight, label_axis_weights=label_axis_weights,
                chunk_size=chunk_size, label_post_smooth=post_smooth)
            for name in ("label_mask", "label_map", "label_score_image"):
                result[name] = assigned[name]
            info.update(label=assigned["point_label"],
                        label_id=assigned["point_label_id"],
                        label_confidence=assigned["point_label_score"],
                        label_info=assigned["label_info"])
        return result, info

    def finish_surface(result, info):
        if not surface_active:
            return result, info
        mask = result["mask"].astype(bool)
        origin, step = result["origin"], result["spacing"]
        keys = data["keys"]
        volume = mask.astype(np.int32) if keys is None else result["label_mask"]
        locked = np.zeros(mask.shape, np.int32)
        if keys is not None and label_mode == "separate" and label_guard_mode == "hard" \
                and separate_protect_labels and label_guard_scale > 0 and np.any(site_trusted):
            tree = cKDTree(x[site_trusted] - center)
            active = np.flatnonzero(mask)
            for start in range(0, len(active), chunk_size):
                sl = slice(start, start + chunk_size)
                distance, nearest = tree.query(result["points"][sl] - center, workers=workers)
                owner = site_codes[site_trusted][nearest] + 1
                take = (distance <= d * label_guard_scale) & (volume.ravel()[active[sl]] == owner)
                locked.ravel()[active[sl][take]] = owner[take]
        if keys is not None and label_lock_seeds:
            seeds = _label_seed_locks(mask, origin, step, d, observations,
                                      workers, chunk_size)
            locked[seeds > 0] = seeds[seeds > 0]
        refined, surface_info = _image_surface_partition(volume, info["field"], step,
            ["occupied"] if keys is None else keys, d,
            mesh_smooth_iters=mesh_smooth_iters, mesh_smooth_pass_band=mesh_smooth_pass_band,
            interface_backend=interface_backend, interface_smooth_iters=interface_smooth_iters,
            interface_smooth_pass_band=interface_smooth_pass_band,
            interface_max_shift_scale=interface_max_shift_scale,
            scope=surface_smooth_scope, locked=locked, chunk_size=chunk_size)
        if surface_info["surface_status"] != "completed":
            info["surface_info"] = surface_info
            if not return_field:
                info.pop("field", None); info.pop("field_inside", None)
            return result, info
        current = refined > 0
        if not np.array_equal(current, mask):
            geometry, sampled = _image_point_data(current, origin, step, d,
                x, geom_conf, sources, site_feature, score_axes,
                score_k, chunk_size, workers)
            result.update(geometry); info.update(sampled)
            if return_field:
                info.update(field=-_sdf(current, step), field_inside="positive",
                            field_source="post_surface_binary_edt")
        if keys is not None:
            codes = refined[current]
            _, certainty = _project_label_evidence(result["points"],
                observations["observation_points"], observations["codes"],
                observations["weights"], label_project_k, d, workers, chunk_size,
                target=codes - 1)
            label_score = np.zeros(mask.shape, np.float32)
            label_score[current] = certainty
            result.update(label_mask=refined, label_score_image=label_score)
            info.update(label=_category_values(keys, codes - 1, for_vtk=False),
                        label_id=codes, label_confidence=certainty)
            info["label_info"] = dict(info["label_info"], label_final_stage="surface_rasterization")
            surface_info["label_score_kind"] = "selected_label_observation_support"
        info["surface_info"] = surface_info
        if not return_field:
            info.pop("field", None); info.pop("field_inside", None)
        return result, info

    if data["keys"] is not None and label_mode == "separate" and (
            separate_output == "merged" or label_guard_mode == "soft" or np.any(post_smooth)
            or surface_active):
        result, info = build(np.arange(len(x)), attach_labels=False)
        mask = result["mask"].astype(bool)
        active = np.flatnonzero(mask)
        fixed_grid = (result["origin"], result["spacing"], mask.shape)
        # Stream one independent class field at a time; never allocate V*C.
        def class_fields():
            for code, key in enumerate(data["keys"]):
                use = (observations["codes"] == code) & (observations["weights"] > 0)
                ids = np.unique(observations["obs_site"][use])
                if not len(ids):
                    continue
                field, nonempty, report = build(ids, code, fixed_grid=fixed_grid, field_only=True)
                yield code, field, nonempty, x[ids], report
        assigned = _separate_label_volume(mask, result["origin"], result["spacing"], d,
            result["points"], observations, class_fields, label_smooth=label_smooth,
            label_smooth_weight=label_smooth_weight, label_axis_weights=label_axis_weights,
            label_lock_seeds=label_lock_seeds, site_codes=site_codes, site_trusted=site_trusted,
            guard_radius=d * label_guard_scale if separate_protect_labels else 0.0,
            workers=workers, chunk_size=chunk_size, device=device,
            guard_mode=label_guard_mode, post_smooth=d * post_smooth,
            label_max_iter=label_max_iter, label_tol=label_tol)
        result.update({name: assigned[name] for name in ("label_mask", "label_map", "label_score_image")})
        ids = assigned["label_mask"].ravel()[active]
        info.update(label=_category_values(data["keys"], ids - 1, for_vtk=False),
            label_id=ids, label_confidence=assigned["label_score_image"].ravel()[active],
            label_info=assigned["label_info"], separate_output="merged")
        result, info = finish_surface(result, info)
        if separate_output == "per_label":
            return _split_image_partition(result, info, return_info, return_field)
        return (result, info) if return_info else result
    if data["keys"] is not None and label_mode == "separate":
        results, infos = {}, {}
        for code, key in enumerate(data["keys"]):
            use = (observations["codes"] == code) & (observations["weights"] > 0)
            ids = np.unique(observations["obs_site"][use])
            if not len(ids):
                empty = np.zeros((1, 1, 1), np.uint8)
                results[key] = dict(mask=empty, score_image=empty.astype(np.float32),
                    spacing=(d * spacing_factor if requested_spacing is None else requested_spacing.copy()),
                    origin=np.zeros(3), points=np.empty((0, 3)), score=np.empty(0, np.float32),
                    label_mask=empty.astype(np.int32), label_score_image=empty.astype(np.float32),
                    label_map={0: None, **{i + 1: value for i, value in enumerate(data["keys"])}})
                infos[key] = dict(diagnostics, label=np.empty(0, object), label_id=np.empty(0, np.int32),
                                  label_confidence=np.empty(0, np.float32), source_id=np.empty(0, np.int64))
                if return_field:
                    infos[key].update(field=np.full(empty.shape, -float(np.min(results[key]["spacing"])),
                                                   np.float32), field_inside="positive")
            else:
                results[key], infos[key] = build(ids, code)
        return (results, infos) if return_info else results
    result, info = build(np.arange(len(x)))
    result, info = finish_surface(result, info)
    return (result, info) if return_info else result



def _sample_cloud(query, points, confidence, sources, feature, d, score_k,
                  score_axes, chunk, workers):
    """Map original-source IDs, geometric scores and per-site attributes."""
    center = np.median(points, axis=0)
    points = points - center
    tree = cKDTree(points)
    bandwidth = d * np.asarray(score_axes)
    score_tree = cKDTree(points / bandwidth) if score_k else None
    nk = min(score_k, len(points))
    nf = min(8, len(points)) if feature is not None else 1
    output = dict(source_id=np.empty(len(query), np.int64))
    if score_k:
        output["score"] = np.empty(len(query), np.float32)
    if feature is not None:
        output["feature"] = np.empty((len(query),) + feature.shape[1:], float)
    width = feature.shape[1] if feature is not None and feature.ndim == 2 else 1
    batch = min(chunk, max(1, 1_000_000 // max(nk, nf * width)))
    for start in range(0, len(query), batch):
        sl = slice(start, start + batch)
        q = query[sl] - center
        if score_k:
            distance, near = score_tree.query(q / bandwidth,
                k=list(range(1, nk+1)), workers=workers)
            output["score"][sl] = np.mean(confidence[near] * np.exp(-.5 * distance**2), axis=1)
        distance, near = tree.query(q, k=list(range(1, nf+1)), workers=workers)
        output["source_id"][sl] = sources[near[:, 0]]
        if feature is not None:
            safe = np.maximum(distance, np.finfo(float).tiny)
            weight = confidence[near] * (safe[:, :1] / safe)**2
            exact = distance[:, 0] == 0
            weight[exact] = 0
            weight[exact, 0] = 1
            weight /= weight.sum(axis=1, keepdims=True)
            if sparse.issparse(feature):
                interpolation = sparse.csr_matrix((weight.ravel(),
                    (np.repeat(np.arange(len(near)), nf), near.ravel())),
                    shape=(len(near), len(points)))
                output["feature"][sl] = (interpolation @ feature).toarray()
            else:
                output["feature"][sl] = np.einsum("nk,nk...->n...", weight, feature[near])
    return output


def _image_point_data(mask, origin, step, d, p, local_conf, local_sources, feature,
                      score_axes, score_k, chunk_size, workers):
    """Use one source sampler before and after optional surface processing."""
    points = origin + np.argwhere(mask) * step
    sampled = _sample_cloud(points, p, local_conf, local_sources, feature, d,
                            score_k, score_axes, chunk_size, workers)
    score = sampled.pop("score")
    score_image = np.zeros(mask.shape, np.float32)
    score_image[mask] = score
    result = dict(mask=mask.astype(np.uint8), score_image=score_image,
                  spacing=step.copy(), origin=origin.copy(), points=points, score=score)
    return result, sampled


def _surface_image_mask(mesh, spacing, shape):
    """Rasterize a closed surface at the existing grid sample positions."""
    from vtkmodules.vtkImagingStencil import vtkPolyDataToImageStencil, vtkImageStencilToImage
    from vtkmodules.util.numpy_support import vtk_to_numpy

    if not mesh.n_cells:
        return (slice(0, 0),) * 3, np.empty((0, 0, 0), bool)
    bounds = np.asarray(mesh.bounds).reshape(3, 2) / spacing[:, None]
    shape = np.asarray(shape, np.int64)
    if np.any(bounds[:, 0] < -0.5 - 1e-6) or np.any(bounds[:, 1] > shape - 0.5 + 1e-6):
        raise _SurfaceUpdateError("Smoothed surface leaves the output grid; increase padding or bounds.")
    lo = np.maximum(np.ceil(bounds[:, 0] - 1e-7).astype(np.int64), 0)
    hi = np.minimum(np.floor(bounds[:, 1] + 1e-7).astype(np.int64), shape - 1)
    if np.any(hi < lo):
        return (slice(0, 0),) * 3, np.empty((0, 0, 0), bool)
    stencil = vtkPolyDataToImageStencil()
    stencil.SetInputData(mesh)
    stencil.SetOutputOrigin(0., 0., 0.)
    stencil.SetOutputSpacing(*spacing)
    stencil.SetOutputWholeExtent(*(int(v) for pair in zip(lo, hi) for v in pair))
    stencil.SetTolerance(0.)
    image = vtkImageStencilToImage()
    image.SetInputConnection(stencil.GetOutputPort())
    image.SetInsideValue(1); image.SetOutsideValue(0)
    image.SetOutputScalarTypeToUnsignedChar(); image.Update()
    value = vtk_to_numpy(image.GetOutput().GetPointData().GetScalars())
    value = value.reshape(tuple(hi - lo + 1), order="F").astype(bool)
    return tuple(slice(int(a), int(b) + 1) for a, b in zip(lo, hi)), value


def _rasterize_label_surfaces(labels, spacing, keys, meshes, outer, outer_active,
                              locked, chunk_size):
    """Rasterize one shared surface; reject ambiguity away from its boundary."""
    before = labels > 0
    report = {}
    target = before.copy()
    if outer_active:
        region, membership = _surface_image_mask(outer, spacing, labels.shape)
        target.fill(False); target[region] = membership
    if locked is None:
        locked = np.zeros(labels.shape, np.int32)
    fixed = locked > 0
    target |= fixed
    result = np.zeros(labels.shape, np.int32)
    count = np.zeros(labels.shape, np.uint32)
    for code, key in enumerate(keys, start=1):
        region, inside = _surface_image_mask(meshes[key], spacing, labels.shape)
        take = inside & target[region]
        result[region][take] = code
        count[region][take] += 1
    ambiguous = target & (count != 1) & ~fixed
    report.update(surface_gap_voxels=int(np.count_nonzero(target & (count == 0))),
                  surface_overlap_voxels=int(np.count_nonzero(target & (count > 1))),
                  surface_ambiguous_voxels=int(ambiguous.sum()))
    if ambiguous.any():
        from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance
        from vtkmodules.vtkCommonCore import vtkDoubleArray
        from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

        nodes = np.flatnonzero(ambiguous)
        best = np.full(len(nodes), np.inf)
        chosen = np.zeros(len(nodes), np.int32)
        original = labels.ravel()[nodes]
        tolerance = float(np.min(spacing) * 1e-7)
        for code, key in enumerate(keys, start=1):
            if not meshes[key].n_cells:
                continue
            distance = vtkImplicitPolyDataDistance(); distance.SetInput(meshes[key])
            for start in range(0, len(nodes), chunk_size):
                sl = slice(start, start + chunk_size)
                xyz = np.column_stack(np.unravel_index(nodes[sl], labels.shape)) * spacing
                value = vtkDoubleArray()
                distance.FunctionValue(numpy_to_vtk(np.ascontiguousarray(xyz)), value)
                delta = vtk_to_numpy(value)
                if not np.isfinite(delta).all():
                    raise _SurfaceUpdateError("Nonfinite signed distance during surface rasterization.")
                take = (delta < best[sl] - tolerance) | (
                    (np.abs(delta - best[sl]) <= tolerance) & (original[sl] == code))
                best[sl][take] = delta[take]
                chosen[sl][take] = code
        invalid = (chosen == 0) | ~np.isfinite(best) | (np.abs(best) > np.linalg.norm(spacing) + tolerance)
        if invalid.any():
            sample = np.column_stack(np.unravel_index(nodes[invalid][:5], labels.shape)).tolist()
            raise _SurfaceUpdateError(f"Surface rasterization has {int(invalid.sum())} nonlocal "
                f"gaps/overlaps (sample XYZ indices {sample}); reduce surface smoothing "
                "or inspect shared-surface topology. No background filling was applied.")
        result.ravel()[nodes] = chosen
    report["surface_lock_restored_voxels"] = int(np.count_nonzero(fixed & (result != locked)))
    result[fixed] = locked[fixed]
    if not np.array_equal(result > 0, target):
        raise _SurfaceUpdateError("Surface rasterization left an occupied voxel without a label.")
    return result, report

def _image_surface_partition(labels, field, spacing, keys, d, *, mesh_smooth_iters,
        mesh_smooth_pass_band, interface_backend, interface_smooth_iters,
        interface_smooth_pass_band, interface_max_shift_scale, scope, locked=None,
        chunk_size=100_000):
    """Shared surface smoothing and simultaneous, exclusive label rasterization."""
    labels = np.asarray(labels, np.int32)
    before = labels > 0
    counts_before = np.bincount(labels.ravel(), minlength=len(keys) + 1)
    report = dict(surface_smooth_scope=scope, surface_added_voxels=0,
        surface_removed_voxels=0, surface_relabelled_voxels=0,
        surface_gap_voxels=0, surface_overlap_voxels=0,
        surface_ambiguous_voxels=0, surface_lock_restored_voxels=0,
        surface_counts_before=counts_before[1:].copy(),
        surface_counts_after=counts_before[1:].copy(), surface_lost_ids=[])
    if not before.any():
        report["surface_status"] = "empty"
        return labels.copy(), report
    if scope == "interfaces" and interface_max_shift_scale == 0:
        report["surface_status"] = "disabled_by_interface_limit"
        return labels.copy(), report

    def revert(reason, stage, backtracks=0):
        warnings.warn("Surface update was rejected; returning the pre-surface partition. "
                      + reason, RuntimeWarning, stacklevel=3)
        report.update(surface_status="reverted", surface_move_fraction=0.,
            surface_backtracks=backtracks, surface_retry_reason=reason,
            surface_failure_stage=stage)
        return labels.copy(), report

    ijk = np.argwhere(before)
    lo = np.maximum(ijk.min(axis=0) - 2, 0)
    hi = np.minimum(ijk.max(axis=0) + 3, labels.shape)
    crop = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    outer_active = scope == "all" and mesh_smooth_iters > 0
    extraction = dict(interface_backend=interface_backend,
        interface_smooth_iters=interface_smooth_iters,
        interface_smooth_pass_band=interface_smooth_pass_band,
        interface_max_shift_scale=interface_max_shift_scale, d=d,
        split_normal_patches=False, smooth_outer=outer_active, _return_reference=True)
    backends = ("auto", "voxel_faces") if interface_backend == "auto" else (interface_backend,)
    errors = []
    for backend in backends:
        extraction["interface_backend"] = backend
        try:
            meshes, outer, reference = _mesh_from_labels(labels[crop], field[crop],
                lo * spacing, spacing, keys, mesh_smooth_iters, mesh_smooth_pass_band, **extraction)
        except _SurfaceUpdateError as error:
            errors.append(f"{backend}: {error}")
            continue
        if errors:
            report["surface_backend_fallback"] = backend
        break
    else:
        report["surface_extraction_errors"] = errors
        report["surface_extraction_attempts"] = len(backends)
        return revert("; ".join(errors), "extraction")
    report["surface_extraction_attempts"] = len(errors) + 1
    if errors:
        report["surface_extraction_errors"] = errors
    report.update({name: np.asarray(value).copy() for name, value in outer.field_data.items()})
    extracted, smoothed, triangles = reference
    reason = None
    for attempt in range(7):
        fraction = 0.5 ** attempt
        try:
            if attempt:
                candidate, _ = _bounded_surface_move(extracted,
                    extracted + fraction * (smoothed - extracted), triangles, spacing)
                for model in [outer, *meshes.values()]:
                    if model.n_points:
                        model.points = candidate[np.asarray(model.point_data["global_point_id"])]
            result, raster_info = _rasterize_label_surfaces(labels, spacing, keys,
                meshes, outer, outer_active, locked, chunk_size)
        except _SurfaceUpdateError as error:
            reason = str(error)
            continue
        report.update(raster_info, surface_backtracks=attempt,
                      surface_move_fraction=fraction)
        break
    else:
        return revert(str(reason), "rasterization", backtracks=7)
    if reason is not None:
        report["surface_retry_reason"] = reason
        warnings.warn(f"Surface displacement reduced to {fraction:g} after rasterization checks.",
                      RuntimeWarning, stacklevel=2)
    target = result > 0
    counts_after = np.bincount(result.ravel(), minlength=len(keys) + 1)
    lost = np.flatnonzero((counts_before[1:] > 0) & (counts_after[1:] == 0)) + 1
    report.update(surface_status="completed", surface_outer_active=bool(outer_active),
        surface_added_voxels=int(np.count_nonzero(target & ~before)),
        surface_removed_voxels=int(np.count_nonzero(before & ~target)),
        surface_relabelled_voxels=int(np.count_nonzero(before & target & (labels != result))),
        surface_counts_after=counts_after[1:].copy(), surface_lost_ids=lost.tolist())
    if len(lost):
        warnings.warn("Surface rasterization removed entire label IDs " + str(lost.tolist())
                      + "; use finer spacing or reduce surface smoothing.", stacklevel=2)
    return result, report

def _split_image_partition(result, info, return_info, return_field):
    """Optional class dictionaries cut from one refined image, on one lattice."""
    results, infos = {}, {}
    occupied = result["mask"].astype(bool)
    point_codes = result["label_mask"][occupied]
    for code, key in result["label_map"].items():
        if code == 0:
            continue
        mask = result["label_mask"] == code
        rows = point_codes == code
        part = dict(result, mask=mask.astype(np.uint8),
            points=result["points"][rows], score=result["score"][rows],
            score_image=np.where(mask, result["score_image"], 0),
            label_mask=np.where(mask, code, 0).astype(np.int32),
            label_score_image=np.where(mask, result["label_score_image"], 0))
        results[key] = part
        if return_info:
            current = dict(info, separate_output="per_label", separate_partition="shared")
            for name in ("label", "label_id", "label_confidence", "source_id", "feature"):
                if name in info:
                    current[name] = info[name][rows]
            if return_field:
                current.update(field=-_sdf(mask, result["spacing"]).astype(np.float32),
                               field_inside="positive")
            infos[key] = current
    return (results, infos) if return_info else results


def _smooth_image_support(points, precision, axes, radii, confidence, k, strength,
                          workers, chunk_size):
    """Regularize physical sampling lengths before restoring confidence."""
    if strength == 0 or len(points) < 3:
        return precision, axes
    raw = np.asarray(radii) / np.cbrt(confidence)
    factor = _support_length_factor(points, raw, k, strength, workers, chunk_size)
    return precision / factor[:, None, None] ** 2, axes * factor[:, None]


def _support_length_factor(points, lengths, k, strength, workers, chunk_size):
    """One bounded log-median regularizer for adaptive support lengths."""
    if not strength or len(points) < 3:
        return np.ones(len(points))
    tree = cKDTree(points)
    count = min(int(k), len(points))
    log_length = np.log(lengths)
    relaxed = np.empty(len(points))
    batch = min(int(chunk_size), max(1, 1_000_000 // count))
    for start in range(0, len(points), batch):
        sl = slice(start, start + batch)
        near = tree.query(points[sl], k=list(range(1, count + 1)), workers=workers)[1]
        relaxed[sl] = np.median(log_length[near], axis=1)
    return np.exp(strength * (relaxed - log_length))


def _separate_label_volume(mask, origin, spacing, d, grid_points, observations,
                           class_fields, *, label_smooth, label_smooth_weight,
                           label_axis_weights, label_lock_seeds, site_codes,
                           site_trusted, guard_radius, workers, chunk_size, device,
                           guard_mode="hard", post_smooth=0.0,
                           label_max_iter=1000, label_tol=1e-5):
    """Compete independent continuous fields inside one fixed occupied domain."""
    active = np.flatnonzero(mask)
    keys, codes, weight = observations["keys"], observations["codes"], observations["weights"]
    labels = np.zeros(mask.shape, np.int32)
    score = np.zeros(mask.shape, np.float32)
    report = dict(label_algorithm="independent_field_competition", label_components=0,
        label_guard_voxels=0, label_hard_seed_voxels=0, label_conflict_voxels=0,
        label_extrapolated_voxels=0, label_fallback_classes=[], class_volume_info={})
    output = dict(label_mask=labels, label_score_image=score,
        label_map={0: None, **{i + 1: key for i, key in enumerate(keys)}}, label_info=report)
    if not len(active):
        return output
    usable = np.flatnonzero(weight > 0)
    if not len(usable):
        raise ValueError("No positive label evidence remains for the occupied grid.")
    component, count = ndi.label(mask)
    component_ids = component.ravel()[active]
    _, node, _, assigned_info = _assign_component_observations(
        [(len(active), mask, origin)], observations["observation_points"][usable],
        spacing, d, workers=workers, chunk_size=chunk_size)
    reached = node >= 0
    source = usable[reached]
    membership = component.ravel()[node[reached]]
    present = sparse.coo_matrix((np.ones(len(source)), (membership, codes[source])),
                                 shape=(count + 1, len(keys))).tocsc()
    observed = np.asarray(present.sum(axis=1)).ravel() > 0
    missing = np.flatnonzero(~observed[1:]) + 1
    if len(missing):
        raise ValueError(f"Occupied components {missing[:10].tolist()} have no label observations; "
                         "reduce smoothing/support or filter these unsupported components.")
    report.update(assigned_info, label_components=int(count))
    del component

    hard = np.zeros(len(active), np.int32)
    guard_labels = np.zeros(len(active), np.int32) if guard_mode == "soft" else hard
    guard_bonus = np.zeros(len(active), np.float32) if guard_mode == "soft" else None
    if guard_radius > 0 and np.any(site_trusted):
        sites = observations["points"][site_trusted]
        tree = cKDTree(sites - origin)
        trusted_codes = site_codes[site_trusted]
        if guard_mode == "soft":
            rows = observations["obs_site"]
            reliability = np.bincount(rows, weights=weight, minlength=len(site_codes))
            reliability /= np.maximum(np.bincount(rows, minlength=len(site_codes)), 1)
            reliability = np.clip(reliability[site_trusted], 0, 1)
        for start in range(0, len(active), chunk_size):
            sl = slice(start, start + chunk_size)
            distance, nearest = tree.query(grid_points[sl] - origin, workers=workers)
            owner = trusted_codes[nearest]
            eligible = np.asarray(present[component_ids[sl], owner]).ravel() > 0
            take = (distance <= guard_radius) & eligible
            guard_labels[sl] = np.where(take, owner + 1, 0)
            if guard_mode == "soft":
                guard_bonus[sl] = np.where(take,
                    reliability[nearest] * np.maximum(0., 1. - (distance / guard_radius)**2), 0.)
        report["label_guard_voxels"] = int(np.count_nonzero(guard_labels))
    if guard_mode == "soft":
        report.update(label_guard_mode="soft", label_soft_guard_voxels=report["label_guard_voxels"])
    if label_lock_seeds and len(source):
        seed_nodes, inverse = np.unique(node[reached], return_inverse=True)
        lo = np.full(len(seed_nodes), len(keys), np.int64)
        hi = np.full(len(seed_nodes), -1, np.int64)
        confidence = np.zeros(len(seed_nodes))
        np.minimum.at(lo, inverse, codes[source])
        np.maximum.at(hi, inverse, codes[source])
        np.maximum.at(confidence, inverse, weight[source])
        locked = (lo == hi) & (confidence >= .9)
        hard[np.searchsorted(active, seed_nodes[locked])] = lo[locked] + 1
        report["label_hard_seed_voxels"] = int(locked.sum())
        report["label_conflict_voxels"] = int(np.count_nonzero(lo != hi))

    blur = d * np.asarray(label_smooth) / spacing
    if label_smooth_weight is not None or label_axis_weights is not None:
        strength = 1.0 if label_smooth_weight is None else label_smooth_weight
        weights = np.ones(3) if label_axis_weights is None else np.asarray(label_axis_weights, float)
        weights = np.broadcast_to(weights, (3,))
        if (np.ndim(strength) or not np.isfinite(strength) or strength < 0
                or not np.isfinite(weights).all() or np.any(weights < 0)):
            raise ValueError("Legacy label weight and axis weights must be finite nonnegative.")
        physical = weights / spacing ** 2
        blur = np.sqrt(strength * physical / (2 * physical.sum())) if physical.sum() else np.zeros(3)
    report["label_smooth_sigma"] = blur * spacing
    best = np.full(len(active), -np.inf)
    log_total = np.full(len(active), -np.inf)
    winner = np.zeros(len(active), np.int32)
    winner_supported = np.zeros(len(active), bool)
    class_constraints = {}
    for entry in class_fields():
        code, field, nonempty, sites, volume_info = entry[:5]
        lattice = entry[5:]
        field_origin, field_spacing = (origin, spacing) if not lattice else lattice[:2]
        allowed = volume_info.get("label_allowed_mask")
        report["class_volume_info"][keys[code]] = {
            name: value for name, value in volume_info.items() if name != "label_allowed_mask"}
        if nonempty:
            if np.any(blur):
                local_blur = blur if not lattice else blur * spacing / field_spacing
                field = _gaussian(field, local_blur, device)
            if not lattice:
                value = np.asarray(field).ravel()[active].astype(np.float64) / d
            else:
                value = np.empty(len(active), np.float64)
                for start in range(0, len(active), chunk_size):
                    sl = slice(start, start + chunk_size)
                    coordinates = ((grid_points[sl] - field_origin) / field_spacing).T
                    value[sl] = ndi.map_coordinates(field, coordinates, order=1,
                        mode="nearest", prefilter=False) / d
        else:
            # A thin class can disappear from its PRIVATE mask while the global
            # domain still has tissue. Keep its observed location as a soft
            # distance prior; do not restore its private geometry or carve holes.
            tree = cKDTree(sites - origin)
            value = np.empty(len(active))
            for start in range(0, len(active), chunk_size):
                sl = slice(start, start + chunk_size)
                value[sl] = -tree.query(grid_points[sl] - origin, workers=workers)[0] / d
            report["label_fallback_classes"].append(keys[code])
        supported = value > 0
        available = np.asarray(present[:, code].toarray()).ravel() > 0
        if guard_mode == "soft":
            value += np.where(guard_labels == code + 1, guard_bonus, 0.)
        eligible = available[component_ids] & ((hard == 0) | (hard == code + 1))
        if allowed is not None:
            allowed = np.asarray(allowed, np.uint8)
            permitted = np.ones(len(active), bool)
            for start in range(0, len(active), chunk_size):
                sl = slice(start, start + chunk_size)
                coordinates = ((grid_points[sl] - field_origin) / field_spacing).T
                permitted[sl] = ndi.map_coordinates(allowed, coordinates,
                    order=0, mode="constant", cval=0, prefilter=False).astype(bool)
            eligible &= permitted
            if np.any((hard == code + 1) & ~permitted):
                raise ValueError("Hard label protection conflicts with a known-empty slice cap; "
                                 "use label_guard_mode='soft' or reduce label_guard_scale.")
            class_constraints[code + 1] = np.packbits(permitted)
        value[~eligible] = -np.inf
        better = value > best
        winner[better] = code + 1
        best[better] = value[better]
        winner_supported[better] = supported[better]
        np.logaddexp(log_total, value, out=log_total)
    invalid = (winner == 0) | ~np.isfinite(best) | ~np.isfinite(log_total)
    if invalid.any():
        raise RuntimeError(f"Separate field competition left {int(invalid.sum())} occupied voxels "
                           "without finite label evidence.")
    labels.ravel()[active] = winner
    score.ravel()[active] = np.clip(np.exp(best - log_total), 0, 1).astype(np.float32)
    report["label_extrapolated_voxels"] = int(np.count_nonzero(~winner_supported))
    if np.any(np.asarray(post_smooth)):
        locked = np.zeros(mask.shape, np.int32)
        locked.ravel()[active] = hard
        refined, certainty, post_report = _smooth_label_partition(labels, spacing,
            post_smooth, confidence=score, locked=locked, max_iter=label_max_iter, tol=label_tol,
            allowed=class_constraints)
        output.update(label_mask=refined, label_score_image=certainty)
        report.update(post_report)
        report["label_extrapolated_voxels_stage"] = "before_label_post_smooth"
    return output


def _label_refinement_options(guard_mode, post_smooth):
    """Canonical options used by every reconstruction entry point."""
    if guard_mode not in {"hard", "soft"}:
        raise ValueError("label_guard_mode must be 'hard' or 'soft'.")
    return _xyz(post_smooth, "label_post_smooth")


def _label_seed_locks(mask, origin, spacing, d, observations, workers, chunk_size,
                      observation_voxel=None):
    """All volume adapters use the same unambiguous >=.9 seed convention."""
    locked = np.zeros(mask.shape, np.int32)
    use = np.flatnonzero(observations["weights"] > 0)
    if not mask.any() or not len(use):
        return locked
    if observation_voxel is None:
        _, nodes, _, _ = _assign_component_observations(
            [(int(mask.sum()), mask, origin)], observations["observation_points"][use],
            spacing, d, workers, chunk_size)
    else:
        nodes = np.asarray(observation_voxel, np.int64)[use]
    reached = nodes >= 0
    nodes, inverse = np.unique(nodes[reached], return_inverse=True)
    codes = observations["codes"][use[reached]]
    lo = np.full(len(nodes), len(observations["keys"]), np.int64)
    hi = np.full(len(nodes), -1, np.int64)
    confidence = np.zeros(len(nodes))
    np.minimum.at(lo, inverse, codes)
    np.maximum.at(hi, inverse, codes)
    np.maximum.at(confidence, inverse, observations["weights"][use[reached]])
    take = (lo == hi) & (confidence >= .9)
    locked.ravel()[nodes[take]] = lo[take] + 1
    return locked


def _post_label_volume(labels, score, origin, spacing, d, observations,
                       post_smooth, lock_seeds, max_iter, tol, workers, chunk_size,
                       observation_voxel=None):
    """Common final partition stage; zero length is an exact no-op."""
    length = _xyz(post_smooth, "label_post_smooth")
    if not np.any(length):
        return labels, score, {}
    locked = (_label_seed_locks(labels > 0, origin, spacing, d, observations,
              workers, chunk_size, observation_voxel) if lock_seeds else None)
    return _smooth_label_partition(labels, spacing, d * length,
        confidence=score, locked=locked, max_iter=max_iter, tol=tol)


def _mesh_partition_labels(mesh, labels, certainty, origin, spacing, keys, workers,
                           chunk_size):
    """Transfer a volume partition to an existing surface without moving it."""
    parts = labels if isinstance(labels, list) else [(labels, certainty, origin)]
    coordinates, ids_all, scores_all = [], [], []
    for part, score, local_origin in parts:
        active = np.argwhere(part > 0)
        coordinates.append(local_origin + active * spacing)
        ids_all.append(part[tuple(active.T)] - 1)
        scores_all.append(score[tuple(active.T)])
    if not mesh.n_points or not any(len(a) for a in coordinates):
        return
    tree = cKDTree(np.concatenate(coordinates))
    ids_all, scores_all = np.concatenate(ids_all), np.concatenate(scores_all)
    for association, query in ((mesh.point_data, mesh.points),
                               (mesh.cell_data, mesh.cell_centers().points)):
        ids = np.empty(len(query), np.int32)
        score = np.empty(len(query), np.float32)
        for start in range(0, len(query), chunk_size):
            sl = slice(start, start + chunk_size)
            near = tree.query(query[sl], workers=workers)[1]
            ids[sl] = ids_all[near]
            score[sl] = scores_all[near]
        association["label_id"] = ids
        association["label"] = _category_values(keys, ids)
        association["label_confidence"] = score


def _numeric_label_metadata(report):
    """Only VTK-compatible diagnostics; full nested reports remain in info."""
    result = {}
    for name, value in report.items():
        if isinstance(value, dict):
            continue
        try:
            value = np.atleast_1d(value)
        except ValueError:
            continue
        if value.size and value.dtype.kind in "biufUS":
            result[name] = value
    return result


def _summarize_label_parts(reports, length):
    """Aggregate occupied-block diagnostics without treating a local loss as global."""
    result = dict(label_post_smooth_length=np.asarray(length).copy(), parts=reports)
    for key in ("label_post_changed_voxels", "label_post_fixed_voxels", "label_guard_voxels",
                "label_soft_guard_voxels"):
        result[key] = sum(int(r.get(key, 0)) for r in reports)
    for key in ("label_post_iterations", "label_post_residual"):
        result[key] = max((r.get(key, 0) for r in reports), default=0)
    counts = {}
    for report in reports:
        for code, before, after in zip(report.get("label_post_classes", []),
                report.get("label_post_counts_before", []), report.get("label_post_counts_after", [])):
            values = counts.setdefault(int(code), [0, 0])
            values[0] += int(before); values[1] += int(after)
    classes = sorted(counts)
    result.update(label_post_classes=np.asarray(classes, np.int32),
        label_post_counts_before=np.array([counts[c][0] for c in classes], np.int64),
        label_post_counts_after=np.array([counts[c][1] for c in classes], np.int64),
        label_post_lost_ids=[c for c in classes if counts[c][1] == 0])
    if any("label_score_kind" in r for r in reports):
        result["label_score_kind"] = "refined_local_membership"
    return result


def _mesh_field_parts(parts, metadata, spacing, backend, device, iterations, pass_band):
    """Extract retained continuous field parts with the ordinary mesh backend."""
    import pyvista as pv
    vertices, faces, offset = [], [], 0
    for _, _, origin, field in parts:
        v, f = _marching_cubes(field, origin, spacing, backend, device)
        vertices.append(v); faces.append(f + offset); offset += len(v)
    faces = np.concatenate(faces)
    mesh = pv.PolyData(np.concatenate(vertices), np.c_[np.full(len(faces), 3), faces]).clean()
    if iterations:
        mesh = mesh.smooth_taubin(n_iter=iterations, pass_band=pass_band, normalize_coordinates=True)
    mesh = mesh.compute_normals(consistent_normals=True, auto_orient_normals=False)
    for name, value in metadata.items():
        mesh.field_data[name] = value
    return mesh


def _label_grid_halo(shape, origin, spacing, margin, max_voxels):
    """Pad class fields independently of the cropped common occupied domain."""
    pad = np.ceil(_xyz(margin, "label field margin") / spacing).astype(np.int64) + 2
    shape = np.asarray(shape, np.int64) + 2 * pad
    if math.prod(shape.tolist()) > max_voxels:
        raise MemoryError("Padded class field exceeds max_voxels; increase spacing or the limit.")
    return tuple(shape), origin - pad * spacing


def _solve_label_system(system, rhs, initial, preconditioner, max_iter, tol, context):
    """CG with explicit residual validation, compatible with old/new SciPy."""
    from scipy.sparse.linalg import cg
    steps = 0
    def count_iteration(_):
        nonlocal steps
        steps += 1
    options = dict(x0=initial, atol=0., maxiter=int(max_iter), M=preconditioner,
                   callback=count_iteration)
    try:
        solution, status = cg(system, rhs, rtol=float(tol), **options)
    except TypeError:
        solution, status = cg(system, rhs, tol=float(tol), **options)
    residual = float(np.linalg.norm(system @ solution - rhs) /
                     max(np.linalg.norm(rhs), np.finfo(float).tiny))
    if (status != 0 or not np.isfinite(solution).all() or not np.isfinite(residual)
            or residual > max(10*tol, 100*np.finfo(float).eps)
            or np.any(solution < -max(10*tol, 1e-8))):
        raise RuntimeError(f"{context}: CG status={status}, residual={residual:.3g}; "
                           "increase label_max_iter or reduce label smoothing.")
    return np.maximum(solution, 0.), steps, residual


def _smooth_label_partition(labels, spacing, length, *, confidence=None, locked=None,
                            max_iter=1000, tol=1e-5, allowed=None):
    """Regularize categorical membership on the fixed occupied voxel graph."""
    from scipy.sparse.linalg import LinearOperator

    a = np.asarray(labels)
    spacing, length = _xyz(spacing, "spacing", True), _xyz(length, "label_post_smooth")
    if a.ndim != 3 or a.dtype.kind not in "iu" or np.any(a < 0):
        raise ValueError("labels must be a nonnegative integer XYZ image.")
    if np.ndim(max_iter) or not np.isfinite(max_iter) or int(max_iter) != max_iter or max_iter < 1:
        raise ValueError("label_max_iter must be a positive integer.")
    if np.ndim(tol) or not np.isfinite(tol) or tol <= 0:
        raise ValueError("label_tol must be finite and positive.")
    mask = a > 0
    evidence = np.ones(a.shape) if confidence is None else np.asarray(confidence)
    if (evidence.shape != a.shape or not np.isfinite(evidence).all()
            or np.any((evidence < 0) | (evidence > 1))):
        raise ValueError("Label assignment confidence must match labels and lie in [0,1].")
    lock = np.zeros_like(a) if locked is None else np.asarray(locked)
    if (lock.shape != a.shape or lock.dtype.kind not in "iu" or np.any(lock < 0)
            or np.any((lock > 0) & ~mask)):
        raise ValueError("locked must contain nonnegative label IDs inside the occupied mask.")
    active = np.flatnonzero(mask)
    output, score = a.astype(np.int32, copy=True), np.zeros(a.shape, np.float32)
    prior = output.ravel()[active].copy()
    hard = lock.ravel()[active]
    prior[hard > 0] = hard[hard > 0]
    classes, counts_before = np.unique(prior, return_counts=True)
    report = dict(label_post_smooth_length=length.copy(), label_post_changed_voxels=0,
        label_post_iterations=0, label_post_residual=0., label_post_lost_ids=[],
        label_post_classes=classes.copy(), label_post_counts_before=counts_before,
        label_post_fixed_voxels=int(np.count_nonzero(hard)),
        label_score_kind="refined_local_membership")
    if not len(active):
        report["label_post_counts_after"] = counts_before.copy()
        return output, score, report
    n = len(active)
    fidelity = np.clip(evidence.ravel()[active].astype(np.float64), 1e-3, 1.)
    index = np.full(mask.size, -1, np.int64)
    index[active] = np.arange(n)
    strides = np.array([a.shape[1] * a.shape[2], a.shape[2], 1])
    rows, cols, values = [], [], []
    for axis, conductance in enumerate(.5 * (length / spacing)**2):
        if conductance == 0:
            continue
        row = np.flatnonzero((active // strides[axis]) % a.shape[axis] < a.shape[axis] - 1)
        col = index[active[row] + strides[axis]]
        take = col >= 0
        row, col = row[take], col[take]
        rows.extend((row, col)); cols.extend((col, row))
        values.extend((np.full(len(row), conductance), np.full(len(row), conductance)))
    del index
    adjacency = (sparse.coo_matrix((np.concatenate(values),
        (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
        if rows else sparse.csr_matrix((n, n)))
    diagonal = fidelity + np.asarray(adjacency.sum(axis=1)).ravel()
    system = sparse.diags(diagonal) - adjacency
    del adjacency, rows, cols, values
    fixed, free = np.flatnonzero(hard > 0), np.flatnonzero(hard == 0)
    coupling = system[free][:, fixed].tocsr() if len(fixed) else None
    if len(fixed):
        system = system[free][:, free].tocsr()
    diag_free = diagonal[free]
    preconditioner = LinearOperator(system.shape, matvec=lambda v: v / diag_free,
                                   dtype=np.float64)
    best, total = np.full(n, -1.), np.zeros(n)
    winner = prior.copy()
    tie = max(1e-12, float(tol) * .1)
    for code in classes:
        initial = (prior == code).astype(np.float64)
        probability = initial.copy()
        if len(free):
            rhs = fidelity[free] * initial[free]
            if len(fixed):
                rhs -= coupling @ initial[fixed]
            solution, steps, residual = _solve_label_system(system, rhs, initial[free],
                preconditioner, max_iter, tol, f"Label refinement, class {int(code)}")
            probability[free] = np.maximum(solution, 0.)
            report["label_post_iterations"] = max(report["label_post_iterations"], steps)
            report["label_post_residual"] = max(report["label_post_residual"], residual)
        candidate = probability
        if allowed and int(code) in allowed:
            # Hard slice caps constrain the final decision even after optional
            # diffusion. Packed foreground masks avoid retaining float V*C arrays.
            permitted = np.unpackbits(allowed[int(code)], count=n).astype(bool)
            if np.any((hard == code) & ~permitted):
                raise ValueError("A locked label seed conflicts with its known-empty slice cap.")
            probability[~permitted] = 0.
            candidate = np.where(permitted, probability, -np.inf)
        take = (candidate > best + tie) | ((np.abs(candidate-best) <= tie) & (prior == code))
        winner[take], best[take] = code, candidate[take]
        total += probability
    mass_error = np.max(np.maximum(total-1., 0.)) if allowed else np.max(np.abs(total-1.))
    if np.any(total <= 0) or not np.isfinite(total).all() or mass_error > max(1e-3, 50*tol):
        raise RuntimeError("Label partition refinement produced inconsistent membership mass.")
    if allowed:
        report["label_post_constrained_classes"] = np.asarray(sorted(allowed), np.int32)
    output.ravel()[active] = winner
    score.ravel()[active] = np.clip(best / total, 0, 1).astype(np.float32)
    counts_after = np.array([np.count_nonzero(winner == code) for code in classes])
    report.update(label_post_changed_voxels=int(np.count_nonzero(output != a)),
        label_post_counts_after=counts_after, label_post_lost_ids=classes[counts_after == 0].tolist())
    if report["label_post_lost_ids"]:
        warnings.warn("Label partition refinement removed label IDs " + str(report["label_post_lost_ids"])
                      + "; reduce label_post_smooth to preserve thin/small classes.", stacklevel=2)
    return output, score, report


def image_to_nrrd(result, filename, *, key="mask", space="3D-right-handed", units=None,
                  compression_level=1):
    """Write mask, score_image, label_mask or label_score_image to .nrrd.
    
    Binary masks use uint8; labels use little-endian int32, preserving IDs >255;
    scores use little-endian float32. XYZ axis 0 varies fastest in the gzip payload.
    Arrays share shape/spacing/origin; optional direction columns are world grid
    axes. Metadata never rescales/reorients coordinates. space describes INPUT
    coordinates (default generic 3D-right-handed); units may be scalar or XYZ.
    compression_level is 0..9. Returns Path; no reconstruction or input mutation."""
    if key not in {"mask", "score_image", "label_mask", "label_score_image"}:
        raise ValueError("key must be 'mask', 'score_image', 'label_mask' or 'label_score_image'.")
    image = np.asarray(result[key])
    if image.ndim != 3 or 0 in image.shape:
        raise ValueError(f"result[{key!r}] must have a nonempty 3-D shape.")
    if key == "mask":
        if not np.isin(image, (0, 1)).all():
            raise ValueError("mask must contain only 0 and 1.")
        dtype, nrrd_type = np.dtype("uint8"), "uint8"
    elif key == "label_mask":
        if image.shape != np.shape(result["mask"]):
            raise ValueError("label_mask and mask must have the same shape.")
        if (image.dtype.kind not in "biuf" or not np.isfinite(image).all()
                or np.any(image < 0) or np.any(image > np.iinfo(np.int32).max)
                or (image.dtype.kind == "f" and np.any(image != np.floor(image)))):
            raise ValueError("label_mask must contain integer codes in [0, 2147483647].")
        dtype, nrrd_type = np.dtype("<i4"), "int32"
    else:
        if image.shape != np.shape(result["mask"]):
            raise ValueError(f"{key} and mask must have the same shape.")
        if np.iscomplexobj(image) or not np.isfinite(image).all() or np.any((image < 0) | (image > 1)):
            raise ValueError(f"{key} must contain finite real values in [0,1].")
        dtype, nrrd_type = np.dtype("<f4"), "float"
    spacing = _xyz(result["spacing"], "spacing", True)
    origin = np.asarray(result["origin"], float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("origin must be finite XYZ.")
    space = {"RAS": "right-anterior-superior", "LPS": "left-posterior-superior"}.get(space, space)
    if space not in {"3D-right-handed", "3D-left-handed", "right-anterior-superior", "left-posterior-superior"}:
        raise ValueError("Unsupported coordinate convention.")
    if not isinstance(compression_level, (int, np.integer)) or not 0 <= compression_level <= 9:
        raise ValueError("compression_level must be an integer from 0 to 9.")
    path = Path(filename).expanduser()
    if path.suffix.lower() != ".nrrd":
        raise ValueError("filename must end in .nrrd.")
    direction = np.asarray(result.get("direction", np.eye(3)), dtype=float)
    if (direction.shape != (3, 3) or not np.isfinite(direction).all()
            or not np.allclose(direction.T @ direction, np.eye(3), atol=1e-7, rtol=0)):
        raise ValueError("direction must have orthonormal world-space columns.")
    vector = lambda v: "(" + ",".join(format(float(t), ".17g") for t in v) + ")"
    header = ["NRRD0005", "type: " + nrrd_type, "dimension: 3",
              "sizes: " + " ".join(map(str, image.shape)), "space: " + space,
              "space directions: " + " ".join(vector(v) for v in (direction * spacing).T),
              "space origin: " + vector(origin), "kinds: domain domain domain",
              "centers: cell cell cell", "encoding: gzip"]
    if dtype.itemsize > 1:
        header.append("endian: little")
    if units is not None:
        unit_list = [units] * 3 if isinstance(units, str) else list(units)
        if (len(unit_list) != 3 or any(not isinstance(u, str) or not u or
                any(ord(c) < 32 or ord(c) > 126 or c in '\\"' for c in u)
                for u in unit_list)):
            raise ValueError("units must be one or three printable ASCII strings without quotes/backslashes.")
        header.append('space units: ' + ' '.join('"' + u + '"' for u in unit_list))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(("\n".join(header) + "\n\n").encode("ascii"))
        with gzip.GzipFile(fileobj=stream, mode="wb", compresslevel=int(compression_level),
                           mtime=0, filename="") as compressed:
            # Plane batches avoid copying the complete XYZ array to Fortran order.
            for z in range(image.shape[2]):
                compressed.write(image[:, :, z].astype(dtype, copy=False).tobytes(order="F"))
    return str(path)


def _sdf(mask, spacing):
    phi = ndi.distance_transform_edt(~mask, sampling=spacing).astype(np.float32)
    phi -= ndi.distance_transform_edt(mask, sampling=spacing).astype(np.float32)
    return phi


def grid_to_mesh(info, mc_method=None, *, backend=None, device="cpu"):
    """Extract an outer mesh (vertices, faces) from an occupied grid info dict.
    
    Uses mask, spacing, origin and optional orthonormal direction. Optional
    field must agree with mask and field_inside='positive'/'negative'; absent
    field uses binary signed EDT. Reflection reverses triangle winding.
    backend and mc_method are aliases; auto tries PyMCubes, scikit-image, VTK."""
    backend = _alias(backend, mc_method, "auto", "backend", "mc_method")
    if backend not in {"auto", "pymcubes", "skimage", "torchmcubes", "lewiner", "lorensen", "vtk"}:
        raise ValueError("Unknown marching-cubes backend.")
    mask = np.asarray(info["mask"], dtype=bool)
    if mask.ndim != 3 or min(mask.shape) < 1:
        raise ValueError("mask must be a nonempty-shape XYZ array.")
    spacing = _xyz(info["spacing"], "spacing", True)
    origin = np.asarray(info["origin"], dtype=float)
    direction = np.asarray(info.get("direction", np.eye(3)), dtype=float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("origin must be a finite XYZ vector.")
    if (direction.shape != (3, 3) or not np.isfinite(direction).all()
            or not np.allclose(direction.T @ direction, np.eye(3), atol=1e-7, rtol=0)):
        raise ValueError("direction columns must be orthonormal.")
    inside = info.get("field_inside", "negative")
    if inside not in {"negative", "positive"}:
        raise ValueError("field_inside must be 'negative' or 'positive'.")
    if not mask.any():
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int64)
    field = info.get("field")
    if field is not None:
        field = np.asarray(field, dtype=np.float32)
        if field.shape != mask.shape or not np.isfinite(field).all():
            raise ValueError("field must be finite and have the same XYZ shape as mask.")
        field = field if inside == "positive" else -field
    # Cleanup or user edits to mask must not be undone by a stale cached field.
    if field is None or not np.array_equal(field > 0, mask):
        field = -_sdf(np.pad(mask, 1), spacing)
    else:
        field = np.pad(field, 1, constant_values=-max(float(np.max(np.abs(field))),
                                                       float(spacing.min())))
    vertices, faces = _marching_cubes(field, np.zeros(3), spacing, backend, device)
    vertices = origin + (vertices - spacing) @ direction.T
    if np.linalg.det(direction) < 0:
        faces = faces[:, [0, 2, 1]]
    return vertices, faces


def grid_point_cloud(points, spacing=None, *,
    spacing_scale=0.5, base_spacing=None, base_spacing_scale=1.0, axis_scale=1.0,
    direction=None, method='knn', support_scale=1.25, support_k=8, support_radius=None,
    radius_cap=2.0, kernel_scale=2.0, density_level=0.2, smooth_sigma=None, smooth=None,
    smooth_scale=1.0, smooth_scales=(1.0,), smooth_method='sdf', max_shift=None,
    preserve_volume=True, outlier_method='lof', outlier_k=24, outlier_thresh=1.8,
    outlier_iters=1, outlier_radius=None, min_points=None, min_component=None,
    keep_largest=False, graph_scale=None, component_scale=None, closing=0.0, opening=0.0,
    fill_holes=False, output='volume', backend=None, mc_method=None, device='cpu',
    max_voxels=20_000_000_000_000, chunk_size=100000, workers=-1, return_info=False,
    return_field=False, confidence=None, geometry_confidence=None, feature=None, label=None,
    label_confidence=None, label_mode='project', label_smooth=None, label_smooth_weight=None,
    label_axis_weights=None, label_max_iter=1000, label_tol=1e-05, label_lock_seeds=False,
    label_project_k=1, label_outlier_k=24, label_outlier_thresh=0.8, label_outlier_iters=1,
    label_min_points=3, label_outlier_action='relabel', separate_protect_labels=True,
    label_guard_scale=0.6, closing_scale=None, opening_scale=None, hole_scale=None,
    offset_scale=0.0, max_shift_scale=_GRID_UNSET, field_smooth_iters=1,
    pre_min_component=0.0, min_volume_component=0.0, component_diameter_scale=0.0, score_k=8,
    score_scale=2.0, field_smooth_mode="reembed",
    label_guard_mode="hard", label_post_smooth=0.0, support_smooth=0.0,
    separate_geometry="local", separate_output="per_label", interface_backend="auto",
):
    """Regular grid reconstruction; full parameter documentation below."""
    return _grid_point_cloud_impl(**locals())


def _clay_smooth_field(mask, spacing, sigma, scales, method, density_level,
                       preserve_volume, max_shift, device, *, forbidden=None,
                       smooth_mode="reembed", initial_field=None):
    """Return a continuous, positive-inside physical implicit field and mask."""
    if smooth_mode not in {"continuous", "reembed"}:
        raise ValueError("smooth_mode must be 'continuous' or 'reembed'.")
    current = np.asarray(mask, bool).copy()
    spacing = _xyz(spacing, "spacing", True)
    sigma = _xyz(sigma, "smooth_sigma")
    forbidden = (np.zeros_like(current) if forbidden is None else
                 np.asarray(forbidden, bool))
    if forbidden.shape != current.shape:
        raise ValueError("forbidden must match the occupied grid.")
    current &= ~forbidden
    for axis in range(3):
        for edge in (0, -1):
            sl = [slice(None)] * 3
            sl[axis] = edge
            current[tuple(sl)] = False
    if not current.any():
        return np.full(current.shape, -float(spacing.min()), np.float32), current
    phi0 = np.asarray(-_sdf(current, spacing), np.float32)
    constraint = (_sdf(forbidden, spacing).astype(np.float32)
                  if forbidden.any() else None)
    active = np.any(sigma > 0) and max_shift != 0
    if not active:
        field = (phi0 if method == "sdf" else
                 (current.astype(np.float32) - np.float32(density_level)) *
                 np.float32(2.0 * spacing.min()))
        if constraint is not None:
            np.minimum(field, constraint, out=field)
        return field, current
    band = (~forbidden if max_shift is None else
            (phi0 > -max_shift) & (phi0 <= max_shift) & ~forbidden)
    for axis in range(3):
        for edge in (0, -1):
            sl = [slice(None)] * 3
            sl[axis] = edge
            band[tuple(sl)] = False
    target = int(np.count_nonzero(current & band))
    base = phi0 if method == "sdf" else current.astype(np.float32)
    if initial_field is not None and method == "sdf" and smooth_mode == "continuous":
        base = np.asarray(initial_field, np.float32)
        if base.shape != current.shape or not np.isfinite(base).all():
            raise ValueError("initial_field must be finite and match the occupied grid.")
    field = phi0.copy()
    for stage, scale in enumerate(scales):
        field = np.asarray(_gaussian(base, sigma * scale / spacing, device),
                           np.float32)
        # Keep density in occupancy units; the physical field below is shifted
        # and gradient-normalized and must never be fed back as a probability.
        next_density = field.copy() if smooth_mode == "continuous" and method == "density" else None
        level = (_clay_volume_level(field[band], target) if preserve_volume else
                 (np.float32(density_level) if method == "density" else np.float32(0)))
        field -= level
        if method == "density":
            # One positive stage-wide scale preserves the exact interpolated
            # density zero-set; varying scales at neighbouring nodes would not.
            grad2 = np.zeros_like(field)
            for axis in range(3):
                gradient = np.gradient(field, spacing[axis], axis=axis)
                grad2 += gradient * gradient
            floor = 1.0 / (8.0 * max(float(np.linalg.norm(sigma * scale)),
                                     float(spacing.max())))
            np.sqrt(grad2, out=grad2)
            near = band & (np.abs(field) <= 0.2) & (grad2 > floor)
            gradient_scale = float(np.median(grad2[near])) if near.any() else floor
            field /= np.float32(max(gradient_scale, floor))
        if max_shift is not None:
            np.clip(field, phi0 - max_shift, phi0 + max_shift, out=field)
        if constraint is not None:
            np.minimum(field, constraint, out=field)
        for axis in range(3):
            for edge in (0, -1):
                sl = [slice(None)] * 3
                sl[axis] = edge
                np.minimum(field[tuple(sl)], -spacing.min(), out=field[tuple(sl)])
        updated = field > 0
        if not updated.any():
            return field, updated
        if stage + 1 < len(scales):
            if smooth_mode == "continuous":
                base = field if method == "sdf" else next_density
            elif not np.array_equal(updated, current):
                base = (-_sdf(updated, spacing).astype(np.float32) if method == "sdf"
                        else updated.astype(np.float32))
        current = updated
    return field, current


# Shared reconstruction and label-noise helpers.

def _clean_cloud(points, confidence, k, outlier_thresh, min_component, workers,
                 *, outlier_method="lof", outlier_k=None, outlier_iters=1,
                 outlier_radius=None, min_points=None, graph_scale=2.0,
                 radius_cap=1.6, min_survivors=4, keep_largest=False):
    """Return retained input-row indices, reference length d, and support radii."""
    points = np.asarray(points, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    method = "none" if outlier_method is None else str(outlier_method).lower()
    if method not in {"none", "lof", "ratio", "sor", "radius"}:
        raise ValueError("outlier_method must be None, 'lof', 'ratio', 'sor' or 'radius'.")
    nk = k if outlier_k is None else outlier_k
    for name, value, minimum in (("k", k, 1), ("outlier_k", nk, 1),
                                  ("outlier_iters", outlier_iters, 0)):
        if not np.isfinite(value) or int(value) != value or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    nk, k, outlier_iters = int(nk), int(k), int(outlier_iters)
    if not np.isfinite(outlier_thresh) or outlier_thresh < 0:
        raise ValueError("outlier_thresh must be finite and nonnegative.")
    enabled = method != "none" and outlier_iters > 0 and outlier_thresh > 0
    if enabled and method in {"lof", "ratio"} and outlier_thresh <= 1:
        raise ValueError("LOF and ratio require outlier_thresh > 1, or 0 to disable.")
    if outlier_radius is not None and (not np.isfinite(outlier_radius) or outlier_radius <= 0):
        raise ValueError("outlier_radius must be None or a positive physical length.")
    min_points = min_component if min_points is None else min_points
    for name, value, minimum in (("min_points", min_points, 0),
                                  ("graph_scale", graph_scale, 0),
                                  ("radius_cap", radius_cap, 1)):
        if not np.isfinite(value) or value < minimum or (name == "graph_scale" and value == 0):
            raise ValueError(f"Invalid {name}: expected " + (">0." if name == "graph_scale" else f">={minimum}."))
    keep = np.arange(len(points))
    tiny = np.finfo(float).tiny

    def neighbors(ids, count):
        if len(ids) < min_survivors:
            raise ValueError(f"At least {min_survivors} distinct points must survive cleaning.")
        kk = min(count, len(ids) - 1)
        distance, index = cKDTree(points[ids]).query(points[ids], k=kk + 1, workers=workers)
        return distance[:, 1:], index[:, 1:]

    for _ in range(outlier_iters if enabled else 0):
        distance, index = neighbors(keep, nk)
        if method == "lof":
            reach = np.maximum(distance, distance[index, -1])
            mean_reach = np.maximum(reach.mean(axis=1), tiny)
            good = (mean_reach[:, None] / mean_reach[index]).mean(axis=1) <= outlier_thresh
        elif method == "ratio":
            ratio = distance[:, -1] / np.maximum(np.median(distance[index, -1], axis=1), tiny)
            good = ratio <= outlier_thresh
        elif method == "sor":
            mean = distance.mean(axis=1)
            center = np.median(mean)
            spread = 1.4826 * np.median(np.abs(mean - center))
            floor = 64 * np.finfo(float).eps * max(float(center), np.finfo(float).tiny)
            if spread <= floor:
                spread = max(float(np.std(mean)), floor)
            good = mean <= center + outlier_thresh * spread
        else:
            if nk >= len(keep):
                raise ValueError("Radius filtering needs more retained points than outlier_k.")
            radius = outlier_radius
            if radius is None:
                radius = outlier_thresh * np.median(distance[:, -1])
            good = distance[:, -1] <= radius
        if good.all():
            break
        keep = keep[good]

    distance, index = neighbors(keep, k)
    if min_points > 0 or keep_largest:
        local = np.median(distance[:, :min(6, distance.shape[1])], axis=1)
        local = np.minimum(local, radius_cap * np.median(local))
        connected = distance <= graph_scale * np.minimum(local[:, None], local[index])
        row, column = np.nonzero(connected)
        graph = sparse.coo_matrix((np.ones(len(row), dtype=np.uint8),
            (row, index[row, column])), shape=(len(keep), len(keep))).tocsr()
        _, component = sparse.csgraph.connected_components(graph, directed=False)
        counts = np.bincount(component)
        threshold = min_points * counts.max() if min_points < 1 else min_points
        good = counts[component] >= threshold
        if keep_largest:
            good &= component == int(counts.argmax())
        if not good.all():
            keep = keep[good]
            distance, index = neighbors(keep, k)

    lengths = distance[:, -1] / np.cbrt(distance.shape[1])
    d = float(np.median(lengths))
    if not np.isfinite(d) or d <= 0:
        raise ValueError("Cannot estimate a positive point sampling length.")
    radii = np.clip(lengths, 0.6 * d, radius_cap * d) * np.cbrt(confidence[keep])
    return keep, d, radii


def _fill_volume_holes(mask, spacing, diameter, forbidden=None):
    """Fill bounded enclosed 26-connected voids, excluding protected cavities."""
    mask = np.asarray(mask, bool)
    if diameter <= 0 or not mask.any():
        return mask.copy()
    regions, count = ndi.label(~mask, structure=np.ones((3, 3, 3), bool))
    sizes = np.bincount(regions.ravel(), minlength=count + 1)
    limit = np.inf if np.isinf(diameter) else np.pi / 6 * float(diameter) ** 3
    fill = sizes * float(np.prod(spacing)) <= limit
    fill[0] = False
    for axis in range(3):
        for edge in (0, -1):
            fill[np.unique(np.take(regions, edge, axis=axis))] = False
    if forbidden is not None:
        fill[np.unique(regions[np.asarray(forbidden, bool)])] = False
    return mask | fill[regions]


def _label_volume(mask, origin, spacing, points, codes, confidence, d,
                  label_smooth, k, workers, *, label_max_iter=1000, label_tol=1e-5,
                  label_lock_seeds=False, label_smooth_weight=None,
                  label_axis_weights=None, chunk_size=50_000,
                  observation_voxel=None, observation_projected=None, label_post_smooth=0.0):
    """Soft joint partition of one occupied XYZ grid; background remains zero."""
    from scipy.sparse.linalg import LinearOperator

    mask = np.asarray(mask, dtype=bool)
    origin = np.asarray(origin, dtype=float)
    spacing = np.broadcast_to(np.asarray(spacing, dtype=float), (3,))
    smooth = np.broadcast_to(np.asarray(label_smooth, dtype=float), (3,))
    points = np.asarray(points, dtype=float)
    codes = np.asarray(codes, dtype=np.int64)
    confidence = np.asarray(confidence, dtype=float)
    if (mask.ndim != 3 or origin.shape != (3,) or not np.isfinite(origin).all() or
            np.any(spacing <= 0) or not np.isfinite(spacing).all() or
            np.any(smooth < 0) or not np.isfinite(smooth).all() or not np.isfinite(d) or d <= 0):
        raise ValueError("Require a 3-D mask, positive spacing/d and nonnegative label_smooth.")
    if (points.ndim != 2 or points.shape[1] != 3 or
            codes.shape != (len(points),) or confidence.shape != (len(points),) or
            np.any(codes < 0) or np.any(codes >= np.iinfo(np.int32).max) or
            not np.isfinite(points).all() or
            not np.isfinite(confidence).all() or np.any(confidence < 0)):
        raise ValueError("Invalid label observations or confidence.")
    for name, value in (("label_max_iter", label_max_iter), ("chunk_size", chunk_size)):
        if not np.isscalar(value) or not np.isfinite(value) or int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not np.isscalar(label_tol) or not np.isfinite(label_tol) or label_tol <= 0:
        raise ValueError("label_tol must be positive and finite.")
    if not isinstance(label_lock_seeds, (bool, np.bool_)):
        raise ValueError("label_lock_seeds must be a bool.")
    lengths2 = (float(d) * smooth / spacing) ** 2
    if label_smooth_weight is not None or label_axis_weights is not None:
        weight = 1.0 if label_smooth_weight is None else label_smooth_weight
        axis_weights = np.ones(3) if label_axis_weights is None else np.broadcast_to(
            np.asarray(label_axis_weights, float), (3,))
        if (not np.isscalar(weight) or not np.isfinite(weight) or weight < 0 or
                not np.isfinite(axis_weights).all() or np.any(axis_weights < 0)):
            raise ValueError("Legacy label weight and axis weights must be finite and nonnegative.")
        physical = axis_weights / spacing ** 2
        lengths2 = weight * physical / (2 * physical.sum()) if physical.sum() else np.zeros(3)
    output = np.zeros(mask.shape, dtype=np.int32)
    score = np.zeros(mask.shape, dtype=np.float32)
    active = np.flatnonzero(mask)
    if not len(active):
        return output, score, {"label_observations": 0, "label_iterations": 0,
            "label_residual": 0.0, "label_probability_mass_error": 0.0,
            "label_locked_voxels": 0, "label_hard_seed_voxels": 0,
            "label_conflict_voxels": 0, "label_seed_voxels": 0,
            "label_projected_points": 0, "label_unused_points": len(points),
            "label_ignored_observations": len(points), "label_components": 0,
            "label_nclasses": 0, "label_classes": np.empty(0, dtype=np.int64),
            "label_graph_weights": lengths2.copy()}

    components, n_components = ndi.label(mask)
    if observation_voxel is None:
        if observation_projected is not None:
            raise ValueError("observation_projected requires observation_voxel.")
        eligible = np.flatnonzero(confidence > 0)
        _, assigned, projected, _ = _assign_component_observations(
            [(len(active), mask, origin)], points[eligible], spacing, d,
            workers=workers, chunk_size=chunk_size)
        observation_voxel = np.full(len(points), -1, dtype=np.int64)
        observation_voxel[eligible] = assigned
        projected_count = int(projected.sum())
    else:
        assigned = np.asarray(observation_voxel)
        if (assigned.shape != (len(points),) or assigned.dtype.kind not in "iu" or
                np.any(assigned < -1) or np.any(assigned >= mask.size)):
            raise ValueError("observation_voxel must contain local flat node IDs or -1.")
        observation_voxel = assigned.astype(np.int64, copy=True)
        reached = observation_voxel >= 0
        if not mask.ravel()[observation_voxel[reached]].all():
            raise ValueError("Preassigned label observations must refer to occupied nodes.")
        if observation_projected is None:
            projected_count = 0
        else:
            projected = np.asarray(observation_projected)
            if projected.shape != (len(points),) or projected.dtype.kind != "b":
                raise ValueError("observation_projected must be a per-observation boolean array.")
            if np.any(projected & ~reached):
                raise ValueError("Unused label observations cannot be marked projected.")
            projected_count = int(np.count_nonzero(projected & (confidence > 0)))
    membership = np.zeros(len(points), dtype=np.int32)
    reached = observation_voxel >= 0
    membership[reached] = components.ravel()[observation_voxel[reached]]
    usable = (membership > 0) & (confidence > 0)
    if not usable.any():
        raise ValueError("No positive-confidence label observation reaches the occupied domain.")

    # Diagnostic seeds mean distinct occupied nodes receiving nearest-voxel
    # observations, including soft conflicts; they are not kNN-initialized nodes.
    seed_voxels, seed_inverse = np.unique(observation_voxel[usable], return_inverse=True)
    seed_min = np.full(len(seed_voxels), codes[usable].max() + 1, dtype=np.int64)
    seed_max = np.full(len(seed_voxels), -1, dtype=np.int64)
    seed_confidence = np.zeros(len(seed_voxels))
    np.minimum.at(seed_min, seed_inverse, codes[usable])
    np.maximum.at(seed_max, seed_inverse, codes[usable])
    np.maximum.at(seed_confidence, seed_inverse, np.clip(confidence[usable], 0.0, 1.0))
    conflict_count = int(np.count_nonzero(seed_min != seed_max))
    locked_count = int(np.count_nonzero((seed_min == seed_max) & (seed_confidence >= 0.9)))         if label_lock_seeds else 0
    seed_count = len(seed_voxels)
    del seed_voxels, seed_inverse, seed_min, seed_max, seed_confidence

    component_ids = components.ravel()[active]
    order = np.argsort(component_ids, kind="stable")
    groups = np.split(active[order], np.cumsum(np.bincount(
        component_ids, minlength=n_components + 1)[1:])[:-1])
    del active, components, component_ids, order
    local_index = np.full(mask.size, -1, dtype=np.int32)
    strides = np.asarray((mask.shape[1] * mask.shape[2], mask.shape[2], 1))
    max_iterations, max_residual, max_mass_error = 0, 0.0, 0.0
    for component, voxels in enumerate(groups, 1):
        selected = np.flatnonzero(usable & (membership == component))
        if not len(selected):
            raise ValueError(f"Occupied component {component} has no label observations; "
                             "reduce support/smoothing or filter this unsupported component.")
        class_ids, local_codes = np.unique(codes[selected], return_inverse=True)
        if len(class_ids) == 1:
            output.ravel()[voxels] = int(class_ids[0]) + 1
            score.ravel()[voxels] = 1.0
            continue
        x, inverse, counts = np.unique(points[selected], axis=0,
                                       return_inverse=True, return_counts=True)
        c = np.clip(confidence[selected], 0.0, 1.0)
        evidence = sparse.coo_matrix((c, (inverse, local_codes)),
                                     shape=(len(x), len(class_ids))).tocsr()
        total = np.asarray(evidence.sum(axis=1)).ravel()
        evidence = sparse.diags(1.0 / total) @ evidence
        site_confidence = total / counts
        tree = cKDTree(x)
        neighbours = min(max(int(k), 1), 16, len(x))
        observations, fidelity = [], np.empty(len(voxels), dtype=np.float64)
        # Form sparse class observations directly; no full V*k or V*C array.
        for start in range(0, len(voxels), int(chunk_size)):
            stop = min(start + int(chunk_size), len(voxels))
            xyz = origin + np.column_stack(np.unravel_index(
                voxels[start:stop], mask.shape)) * spacing
            distances, ids = tree.query(xyz, k=neighbours, workers=workers)
            distances, ids = distances.reshape(-1, neighbours), ids.reshape(-1, neighbours)
            kernel = 1.0 / (distances ** 2 + (0.05 * float(d)) ** 2)
            mean_confidence = (kernel * site_confidence[ids]).sum(axis=1) / kernel.sum(axis=1)
            kernel *= site_confidence[ids]
            kernel /= kernel.sum(axis=1, keepdims=True)
            rows = np.repeat(np.arange(stop - start), neighbours)
            weights = sparse.coo_matrix((kernel.ravel(), (rows, ids.ravel())),
                                        shape=(stop - start, len(x))).tocsr()
            observations.append(weights @ evidence)
            fidelity[start:stop] = np.maximum(
                1e-3, mean_confidence * np.exp(-0.5 * (distances[:, 0] / d) ** 2))
        q = sparse.vstack(observations, format="csc")
        del observations, tree, evidence

        n = len(voxels)
        local_index[voxels] = np.arange(n, dtype=np.int32)
        locked_class = np.full(n, -1, dtype=np.int32)
        if label_lock_seeds:
            observation_rows = local_index[observation_voxel[selected]]
            minimum = np.full(n, len(class_ids), dtype=np.int32)
            maximum = np.full(n, -1, dtype=np.int32)
            reliability = np.zeros(n)
            np.minimum.at(minimum, observation_rows, local_codes)
            np.maximum.at(maximum, observation_rows, local_codes)
            np.maximum.at(reliability, observation_rows, c)
            locked = (maximum >= 0) & (minimum == maximum) & (reliability >= 0.9)
            locked_class[locked] = minimum[locked]
        fixed = np.flatnonzero(locked_class >= 0)
        free = np.flatnonzero(locked_class < 0)
        rows, cols, weights = [], [], []
        for axis, weight in enumerate(lengths2):
            if weight == 0:
                continue
            inside = (voxels // strides[axis]) % mask.shape[axis] < mask.shape[axis] - 1
            row = np.flatnonzero(inside)
            col = local_index[voxels[row] + strides[axis]]
            connected = col >= 0
            row, col = row[connected], col[connected]
            rows.extend((row, col))
            cols.extend((col, row))
            weights.extend((np.full(len(row), weight), np.full(len(row), weight)))
        local_index[voxels] = -1
        if rows:
            adjacency = sparse.coo_matrix((np.concatenate(weights),
                                           (np.concatenate(rows), np.concatenate(cols))),
                                          shape=(n, n)).tocsr()
            degree = np.asarray(adjacency.sum(axis=1)).ravel()
            diagonal = fidelity + degree
            system = sparse.diags(diagonal) - adjacency
            del adjacency, degree
            if len(fixed):
                coupling = system[free][:, fixed].tocsr()
                system = system[free][:, free].tocsr()
                diagonal = diagonal[free]
            else:
                coupling = None
            preconditioner = LinearOperator(system.shape, matvec=lambda v: v / diagonal,
                                             dtype=np.float64)
        else:
            system = None
        del rows, cols, weights
        best = np.full(n, -1.0)
        normalizer = np.zeros(n)
        winner = np.zeros(n, dtype=np.int32)
        for column, class_id in enumerate(class_ids):
            initial = q[:, column].toarray().ravel()
            probability = initial.copy()
            if len(fixed):
                probability[fixed] = (locked_class[fixed] == column).astype(float)
            if system is not None and len(free):
                rhs = fidelity[free] * initial[free]
                if len(fixed):
                    rhs -= coupling @ probability[fixed]
                solution, steps, residual = _solve_label_system(system, rhs, initial[free],
                    preconditioner, label_max_iter, label_tol,
                    f"Label diffusion, component {component}, class {int(class_id)}")
                probability[free] = solution
                max_iterations = max(max_iterations, steps)
                max_residual = max(max_residual, residual)
            negative_limit = max(10 * float(label_tol), 1e-8)
            if (not np.isfinite(probability).all() or
                    np.min(probability) < -negative_limit):
                bad = np.flatnonzero(~np.isfinite(probability) | (probability < -negative_limit))
                sample = np.column_stack(np.unravel_index(voxels[bad[:3]], mask.shape)).tolist()
                raise RuntimeError(f"Invalid label probabilities in component {component}, "
                    f"class {int(class_id)} at {len(bad)} voxels, sample indices {sample}.")
            probability = np.maximum(probability, 0.0)
            improve = probability > best
            winner[improve] = int(class_id) + 1
            best[improve] = probability[improve]
            normalizer += probability
        invalid = ~np.isfinite(normalizer) | (normalizer <= 0) | (winner <= 0)
        if invalid.any():
            sample = np.column_stack(np.unravel_index(voxels[invalid][:3], mask.shape)).tolist()
            raise RuntimeError(f"Joint partition has {int(invalid.sum())} invalid occupied "
                f"voxels in component {component}; zero-mass={int((normalizer <= 0).sum())}, "
                f"unassigned={int((winner <= 0).sum())}, sample indices {sample}.")
        mass_error = float(np.max(np.abs(normalizer - 1.0)))
        max_mass_error = max(max_mass_error, mass_error)
        if mass_error > max(1e-3, 50 * float(label_tol)):
            raise RuntimeError(f"Joint probabilities in component {component} do not sum "
                f"to one (maximum error {mass_error:.3g}); tighten label_tol or inspect "
                "the diffusion solution before extracting a surface.")
        output.ravel()[voxels] = winner
        score.ravel()[voxels] = np.clip(best / normalizer, 0.0, 1.0)
    report = {"label_observations": int(usable.sum()),
                           "label_ignored_observations": int((~usable).sum()),
                           "label_iterations": int(max_iterations),
                           "label_residual": float(max_residual),
                           "label_probability_mass_error": float(max_mass_error),
                           "label_locked_voxels": int(locked_count),
                           "label_hard_seed_voxels": int(locked_count),
                           "label_conflict_voxels": conflict_count,
                           "label_seed_voxels": int(seed_count),
                           "label_projected_points": int(projected_count),
                           "label_unused_points": int((~usable).sum()),
                           "label_components": int(n_components),
                           "label_nclasses": int(len(np.unique(codes[usable]))),
                           "label_classes": np.unique(codes[usable]),
                           "label_graph_weights": lengths2.copy()}

    if np.any(_xyz(label_post_smooth, "label_post_smooth")):
        observations = dict(keys=list(range(int(codes.max()) + 1)), codes=codes,
                            weights=confidence, observation_points=points)
        output, score, post = _post_label_volume(output, score, origin, spacing, d,
            observations, label_post_smooth, label_lock_seeds, label_max_iter,
            label_tol, workers, chunk_size, observation_voxel)
        report.update(post)
    return output, score, report


def _sinc_points(points, faces, iterations, pass_band, *, fixed=None):
    """Windowed-sinc low-pass filtering with stable IDs and optional fixed nodes."""
    points = np.asarray(points, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    if iterations == 0 or not len(faces):
        return points.copy()
    if fixed is not None:
        fixed = np.asarray(fixed, dtype=bool)
        if fixed.shape != (len(points),):
            raise ValueError("fixed must be a Boolean mask over the shared points.")
        row, col = faces.ravel(), np.roll(faces, -1, axis=1).ravel()
        graph = sparse.coo_matrix((np.ones(2 * len(row)),
            (np.r_[row, col], np.r_[col, row])), shape=(len(points), len(points))).tocsr()
        graph.setdiag(0); graph.eliminate_zeros(); graph.data.fill(1.0)
        degree = np.asarray(graph.sum(axis=1)).ravel()
        free = (degree > 0) & ~fixed
        transition = sparse.diags(np.divide(free.astype(float), degree,
            out=np.zeros(len(points)), where=degree > 0)) @ graph
        transition += sparse.diags((~free).astype(float))
        n = int(iterations)
        order = np.arange(1, n + 1, dtype=float)
        theta = 2.0 * np.arcsin(np.sqrt(float(pass_band) / 2.0))
        coefficients = np.r_[theta / np.pi,
            2.0 * np.sin(order * theta) / (np.pi * order) *
            (0.54 + 0.46 * np.cos(np.pi * order / (n + 1)))]
        coefficients /= coefficients.sum()  # exactly preserve translations
        center = points.mean(axis=0)
        previous = points - center
        current = transition @ previous
        filtered = coefficients[0] * previous + coefficients[1] * current
        for coefficient in coefficients[2:]:
            following = 2.0 * (transition @ current) - previous
            filtered += coefficient * following
            previous, current = current, following
        result = filtered + center
        result[~free] = points[~free]
        return result
    import pyvista as pv
    from vtkmodules.vtkFiltersCore import vtkWindowedSincPolyDataFilter
    model = pv.PolyData(points, np.column_stack(
        (np.full(len(faces), faces.shape[1]), faces)).ravel())
    model.point_data['_input_point_id'] = np.arange(len(points), dtype=np.int64)
    filt = vtkWindowedSincPolyDataFilter()
    filt.SetInputData(model)
    filt.SetNumberOfIterations(int(iterations))
    filt.SetPassBand(float(pass_band))
    filt.NormalizeCoordinatesOn(); filt.NonManifoldSmoothingOn()
    filt.FeatureEdgeSmoothingOff(); filt.BoundarySmoothingOn(); filt.Update()
    output = pv.wrap(filt.GetOutput())
    ids = np.asarray(output.point_data['_input_point_id'], dtype=np.int64)
    if (len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= len(points))
            or not np.isin(np.unique(faces), ids).all()):
        raise RuntimeError("Surface smoothing did not preserve referenced point IDs.")
    result = points.copy()
    result[ids] = output.points
    return result


def _patch_mesh(points, triangles, pairs, selected, outer_normals, keys, code=None,
                *, split_normal_patches=False):
    """Extract ownership with closed index topology, or optional display seams."""
    import pyvista as pv
    if not len(selected):
        return pv.PolyData()
    faces, sides = triangles[selected].copy(), pairs[selected]
    if code is not None:
        reverse = sides[:, 1] == code + 1
        faces[reverse] = faces[reverse, ::-1]
    external = sides[:, 1] == 0
    patch = np.where(external, 0, sides[:, 0] * (len(keys) + 1) + sides[:, 1])
    if split_normal_patches:
        vertex_patch, inverse = np.unique(np.column_stack(
            (faces.ravel(), np.repeat(patch, 3))), axis=0, return_inverse=True)
        global_ids = vertex_patch[:, 0]
        ext_points = vertex_patch[:, 1] == 0
    else:
        global_ids, inverse = np.unique(faces.ravel(), return_inverse=True)
        ext_points = np.isin(global_ids, np.unique(faces[external]))
    f = inverse.reshape(-1, 3)
    p = points[global_ids]
    mesh = pv.PolyData(p.copy(), np.column_stack((np.full(len(f), 3), f)).ravel())
    normals = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
    point_normals = np.zeros_like(p)
    for j in range(3):
        np.add.at(point_normals, f[:, j], normals)
    point_normals /= np.maximum(np.linalg.norm(point_normals, axis=1, keepdims=True), 1e-30)
    point_normals[ext_points] = outer_normals[global_ids[ext_points]]
    mesh.point_data['Normals'] = point_normals
    mesh.cell_data['Normals'] = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
    mesh.GetPointData().SetActiveNormals('Normals'); mesh.GetCellData().SetActiveNormals('Normals')
    mesh.point_data['global_point_id'] = global_ids
    mesh.cell_data['global_face_id'] = np.asarray(selected, dtype=np.int64)
    mesh.cell_data['BoundaryLabels'] = sides.copy()
    mesh.field_data['split_normal_patches'] = [int(split_normal_patches)]
    if code is None:
        mesh.cell_data['label_id'] = sides[:, 0].astype(np.int32) - 1
        mesh.cell_data['label'] = np.asarray([str(keys[c - 1]) for c in sides[:, 0]])
    else:
        value = keys[code]
        value = str(value) if np.asarray(value).ndim or np.asarray(value).dtype.kind == 'O' else value
        mesh.point_data['label'] = np.full(mesh.n_points, value)
        mesh.point_data['label_id'] = np.full(mesh.n_points, code, dtype=np.int32)
        mesh.cell_data['label_id'] = np.full(mesh.n_cells, code, dtype=np.int32)
        mesh.cell_data['label'] = _category_values(keys, np.full(mesh.n_cells, code))
    return mesh


def _mesh_from_labels(labels, field, origin, spacing, keys,
                      mesh_smooth_iters=30, mesh_smooth_pass_band=0.1, *,
                      interface_backend="auto", interface_smooth_iters=None,
                      interface_max_shift_scale=None, d=1.0,
                      interface_smooth_pass_band=None, split_normal_patches=False,
                      smooth_outer=True, _return_reference=False):
    """Shared label surfaces with independent, constrained internal smoothing."""
    import pyvista as pv
    a = np.asarray(labels)
    field = np.asarray(field)
    origin = np.asarray(origin, dtype=float)
    spacing = np.broadcast_to(np.asarray(spacing, dtype=float), (3,))
    if a.ndim != 3 or a.shape != field.shape or min(a.shape) < 2:
        raise ValueError('labels and field must be equal-sized 3-D arrays, each axis >=2.')
    if a.dtype.kind not in 'biu' or a.min() < 0 or a.max() > len(keys):
        raise ValueError('labels must contain integer IDs 0..len(keys).')
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError('origin must contain three finite coordinates.')
    if not np.isfinite(spacing).all() or np.any(spacing <= 0) or not np.isfinite(field).all():
        raise ValueError('spacing must be positive and field finite.')
    if (int(mesh_smooth_iters) != mesh_smooth_iters or mesh_smooth_iters < 0
            or not 0 < mesh_smooth_pass_band <= 2):
        raise ValueError('Require integer mesh_smooth_iters >=0 and 0<pass_band<=2.')
    if mesh_smooth_iters == 1:
        raise ValueError('Use mesh_smooth_iters=0 to disable, or >=2 for Windowed Sinc.')
    if interface_backend not in {"auto", "surface_nets", "voxel_faces"}:
        raise ValueError("interface_backend must be 'auto', 'surface_nets' or 'voxel_faces'.")
    if interface_smooth_iters is not None and (
            not np.isscalar(interface_smooth_iters) or not np.isfinite(interface_smooth_iters)
            or int(interface_smooth_iters) != interface_smooth_iters
            or interface_smooth_iters < 0 or interface_smooth_iters == 1):
        raise ValueError("interface_smooth_iters must be None, 0, or an integer >=2.")
    if not np.isscalar(d) or not np.isfinite(d) or d <= 0:
        raise ValueError("d must be a positive finite sampling length.")
    if interface_max_shift_scale is not None and (
            not np.isscalar(interface_max_shift_scale)
            or not np.isfinite(interface_max_shift_scale) or interface_max_shift_scale < 0):
        raise ValueError("interface_max_shift_scale must be a nonnegative finite scalar or None.")
    if not isinstance(split_normal_patches, (bool, np.bool_)):
        raise ValueError("split_normal_patches must be boolean.")
    if not isinstance(smooth_outer, (bool, np.bool_)):
        raise ValueError("smooth_outer must be boolean.")
    internal_band = mesh_smooth_pass_band if interface_smooth_pass_band is None else interface_smooth_pass_band
    if (np.ndim(internal_band) or not np.isfinite(internal_band)
            or not 0 < internal_band <= 2):
        raise ValueError("interface_smooth_pass_band must be None or a finite scalar in (0,2].")
    if not np.array_equal(a > 0, field > 0):
        raise ValueError("labels foreground must agree exactly with field > 0.")
    a = np.pad(a.astype(np.int32, copy=False), 1)
    ids = np.unique(a)
    ids = ids[ids > 0]
    if not len(ids):
        return {key: pv.PolyData() for key in keys}, pv.PolyData()
    vtkSurfaceNets3D = None
    if interface_backend != "voxel_faces":
        try:
            from vtkmodules.vtkFiltersCore import vtkSurfaceNets3D
        except ImportError as error:
            if interface_backend == "surface_nets":
                raise ImportError("interface_backend='surface_nets' requires VTK with "
                                  "vtkSurfaceNets3D; use 'voxel_faces' or update VTK.") from error
            warnings.warn('VTK Surface Nets unavailable; using shared voxel faces.', stacklevel=2)
    actual_backend = "surface_nets" if vtkSurfaceNets3D is not None else "voxel_faces"
    if vtkSurfaceNets3D is None:
        corners, sides = [], []
        for axis in range(3):
            low, high = [slice(None)] * 3, [slice(None)] * 3
            low[axis], high[axis] = slice(None, -1), slice(1, None)
            al, ah = a[tuple(low)], a[tuple(high)]
            ij = np.argwhere(al != ah)
            if not len(ij):
                continue
            u, v = (axis + 1) % 3, (axis + 2) % 3
            offsets = np.zeros((4, 3), dtype=np.int64)
            offsets[:, axis] = 1
            offsets[:, u], offsets[:, v] = (-1, 1, 1, -1), (-1, -1, 1, 1)
            corners.append(2 * ij[:, None, :] + offsets)
            sides.append(np.column_stack((al[tuple(ij.T)], ah[tuple(ij.T)])))
        points, inverse = np.unique(np.concatenate(corners).reshape(-1, 3),
                                    axis=0, return_inverse=True)
        points, quads, pairs = points.astype(float) * 0.5, inverse.reshape(-1, 4), np.concatenate(sides)
    else:
        from vtkmodules.vtkCommonDataModel import vtkImageData
        from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
        grid = vtkImageData()
        grid.SetDimensions(a.shape)
        grid.GetPointData().SetScalars(numpy_to_vtk(a.ravel(order='F'), deep=True))
        nets = vtkSurfaceNets3D()
        nets.SetInputData(grid)
        nets.SetBackgroundLabel(0)
        nets.SetNumberOfLabels(len(ids))
        for i, label_id in enumerate(ids):
            nets.SetLabel(i, int(label_id))
        nets.SmoothingOff()
        nets.SetOutputMeshTypeToQuads()
        nets.Update()
        raw = nets.GetOutput()
        if not raw.GetNumberOfCells():
            return {key: pv.PolyData() for key in keys}, pv.PolyData()
        points = vtk_to_numpy(raw.GetPoints().GetData()).astype(float)
        quads = vtk_to_numpy(raw.GetPolys().GetData()).reshape(-1, 5)[:, 1:].copy()
        boundary = raw.GetCellData().GetArray('BoundaryLabels')
        if boundary is None:
            raise RuntimeError("Surface Nets did not return BoundaryLabels.")
        pairs = vtk_to_numpy(boundary).astype(np.int32, copy=True)
        if pairs.shape != (len(quads), 2) or np.any(pairs[:, 0] == pairs[:, 1]):
            raise RuntimeError("Surface Nets returned invalid two-sided label metadata.")
        # VTK returns oriented BoundaryLabels (normal from first to second).
        # Validate this convention where local grid samples are unambiguous;
        # samples never replace the authoritative class pair.
        center = points[quads].mean(axis=1)
        normal = np.cross(points[quads[:, 1]] - points[quads[:, 0]],
                          points[quads[:, 2]] - points[quads[:, 0]])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
        before = np.clip(np.floor(center - 0.25 * normal + 0.5).astype(np.int64), 0, np.asarray(a.shape)-1)
        after = np.clip(np.floor(center + 0.25 * normal + 0.5).astype(np.int64), 0, np.asarray(a.shape)-1)
        sampled = np.column_stack((a[tuple(before.T)], a[tuple(after.T)]))
        reverse = np.all(sampled == pairs[:, ::-1], axis=1)
        quads[reverse] = quads[reverse, ::-1]
    flip = (pairs[:, 0] == 0) | ((pairs[:, 1] != 0) & (pairs[:, 0] > pairs[:, 1]))
    quads[flip], pairs[flip] = quads[flip, ::-1], pairs[flip, ::-1]
    # Preserve backend topology: coincident Surface Nets vertices may be intentional.
    points = origin + (points - 1) * spacing
    extracted_points = points.copy()
    triangles = np.concatenate((quads[:, (0, 1, 2)], quads[:, (0, 2, 3)]))
    external_quads = quads[pairs[:, 1] == 0]
    internal_quads = quads[pairs[:, 1] != 0]
    exterior_ids, exterior_inverse = np.unique(external_quads.ravel(), return_inverse=True)
    fixed = np.zeros(len(points), bool); fixed[exterior_ids] = True
    interior_ids = np.flatnonzero(~fixed)
    # Extra interface smoothing must not move multi-tissue junctions. Count
    # DISTINCT incident tissue IDs; duplicate faces never create a junction.
    junction = np.zeros(len(points), bool)
    if len(internal_quads):
        incidence = np.unique(np.column_stack((np.repeat(internal_quads.ravel(), 2),
            np.repeat(pairs[pairs[:, 1] != 0], 4, axis=0).ravel())), axis=0)
        junction = np.bincount(incidence[:, 0], minlength=len(points)) >= 3
    interface_fixed = fixed | junction
    candidate = points.copy()
    if smooth_outer:
        padded_field = np.pad(field, 1, constant_values=-float(spacing.min()))
        projected = _project_zero_surface(points[exterior_ids], padded_field, origin-spacing, spacing)
        candidate[exterior_ids] = _sinc_points(projected, exterior_inverse.reshape(-1, 4),
                                              mesh_smooth_iters, mesh_smooth_pass_band)
    if mesh_smooth_iters and len(internal_quads):
        candidate = _sinc_points(candidate, internal_quads, mesh_smooth_iters,
                                  mesh_smooth_pass_band, fixed=fixed)
    def cap_internal(value):
        if interface_max_shift_scale is not None and len(interior_ids):
            delta = value[interior_ids] - extracted_points[interior_ids]
            factor = np.minimum(1.0, float(d * interface_max_shift_scale) /
                                np.maximum(np.linalg.norm(delta, axis=1), 1e-30))
            value[interior_ids] = extracted_points[interior_ids] + delta * factor[:, None]
        return value
    smoothed, limited_base = _bounded_surface_move(points, cap_internal(candidate), triangles, spacing)
    limited_internal = 0
    if interface_smooth_iters and len(internal_quads) and len(interior_ids):
        candidate = _sinc_points(smoothed, internal_quads, interface_smooth_iters,
                                  internal_band, fixed=interface_fixed)
        smoothed, limited_internal = _bounded_surface_move(smoothed, cap_internal(candidate),
                                                           triangles, spacing, fixed=interface_fixed)
    pairs = np.concatenate((pairs, pairs)).astype(np.int32)
    external = pairs[:, 1] == 0
    outer_faces = triangles[external]
    normals = np.cross(smoothed[outer_faces[:, 1]] - smoothed[outer_faces[:, 0]],
                       smoothed[outer_faces[:, 2]] - smoothed[outer_faces[:, 0]])
    outer_normals = np.zeros_like(smoothed)
    for j in range(3):
        np.add.at(outer_normals, outer_faces[:, j], normals)
    outer_normals /= np.maximum(np.linalg.norm(outer_normals, axis=1, keepdims=True), 1e-30)
    outer = _patch_mesh(smoothed, triangles, pairs, np.flatnonzero(external), outer_normals, keys,
                        split_normal_patches=split_normal_patches)
    meshes = {key: _patch_mesh(smoothed, triangles, pairs,
               np.flatnonzero(np.any(pairs == code + 1, axis=1)), outer_normals, keys, code,
               split_normal_patches=split_normal_patches)
              for code, key in enumerate(keys)}
    for mesh in [outer, *meshes.values()]:
        mesh.field_data['mesh_smooth_iters'] = [int(mesh_smooth_iters)]
        mesh.field_data['mesh_smooth_pass_band'] = [float(mesh_smooth_pass_band)]
        mesh.field_data['interface_backend'] = [actual_backend]
        mesh.field_data['interface_smooth_iters'] = [int(interface_smooth_iters or 0)]
        mesh.field_data['interface_smooth_pass_band'] = [float(internal_band)]
        mesh.field_data['mesh_limited_vertices'] = [int(limited_base)]
        mesh.field_data['interface_limited_vertices'] = [int(limited_internal)]
        mesh.field_data['interface_free_vertices'] = [int(np.count_nonzero(~interface_fixed))]
        mesh.field_data['interface_fixed_vertices'] = [int(np.count_nonzero(interface_fixed[np.unique(internal_quads)]))]
        mesh.field_data['interface_junction_vertices'] = [int(np.count_nonzero(junction & ~fixed))]
        if interface_max_shift_scale is not None:
            mesh.field_data['interface_max_shift_scale'] = [float(interface_max_shift_scale)]
    if _return_reference:
        return meshes, outer, (extracted_points, smoothed, triangles)
    return meshes, outer


def _prepare_geometry(points, confidence, *, k=24,
                      outlier_method="lof", outlier_thresh=1.5,
                      outlier_k=None, outlier_iters=1, outlier_radius=None,
                      min_points=None, min_component=0.01, graph_scale=2.0,
                      radius_cap=1.6, base_spacing=None, base_spacing_scale=1.0,
                      adaptive=True, support_scale=1.5, anisotropy=1.0,
                      chunk_size=100_000, workers=-1, min_survivors=4, keep_largest=False, support_smooth=0.0):
    """Clean once and return ``retained, d, precision, axes, radii``."""

    x, c = np.asarray(points, dtype=float), np.asarray(confidence, dtype=float)
    if (x.ndim != 2 or x.shape[1] != 3 or len(x) < min_survivors or
            c.shape != (len(x),) or not np.isfinite(x).all() or
            not np.isfinite(c).all() or np.any((c <= 0) | (c > 1))):
        raise ValueError(f"Geometry requires at least {min_survivors} finite unique XYZ points and confidence in (0,1].")
    support = np.broadcast_to(np.asarray(support_scale, dtype=float), (3,)).copy()
    if not np.isfinite(support).all() or np.any(support <= 0):
        raise ValueError("support_scale must be a positive finite scalar or XYZ.")
    for name, value, minimum in (("k", k, 1), ("chunk_size", chunk_size, 1)):
        if np.ndim(value) or not np.isfinite(value) or int(value) != value or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    for name, value in (("base_spacing", base_spacing),
                        ("base_spacing_scale", base_spacing_scale)):
        if value is None and name == "base_spacing":
            continue
        if np.ndim(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite scalar.")
    if np.ndim(anisotropy) or not np.isfinite(anisotropy) or anisotropy < 1:
        raise ValueError("anisotropy must be a finite scalar >=1.")
    if not isinstance(adaptive, (bool, np.bool_)):
        raise ValueError("adaptive must be True or False.")
    if np.ndim(support_smooth) or not np.isfinite(support_smooth) or not 0 <= support_smooth <= 1:
        raise ValueError("support_smooth must be finite in [0,1].")
    k, chunk_size = int(k), int(chunk_size)
    scale = float(base_spacing_scale)

    retained, estimated, compact_radii = _clean_cloud(
        x, c, k, outlier_thresh, min_component, workers,
        outlier_method=outlier_method, outlier_k=outlier_k,
        outlier_iters=outlier_iters, outlier_radius=outlier_radius,
        min_points=min_points, graph_scale=graph_scale, radius_cap=radius_cap,
        min_survivors=min_survivors, keep_largest=keep_largest)
    x, c = x[retained], c[retained]
    d = float(estimated if base_spacing is None else base_spacing) * scale
    if not np.isfinite(d) or d <= 0:
        raise ValueError("Effective base spacing is outside floating-point range.")

    nk = min(k, len(x) - 1)
    batch = min(chunk_size, max(1, 1_000_000 // (nk + 1)))
    needs_tree = anisotropy != 1 or (adaptive and base_spacing is not None)
    tree = cKDTree(x) if needs_tree else None
    kth = None
    if adaptive and base_spacing is None:
        # _clean_cloud already clipped around estimated d; scaling both its
        # lengths and reference is exactly equivalent and avoids another query.
        radii = np.asarray(compact_radii, dtype=float) * scale
    elif not adaptive:
        radii = d * np.cbrt(c)
    else:
        kth = np.empty(len(x), dtype=float)
        for start in range(0, len(x), batch):
            sl = slice(start, start + batch)
            kth[sl] = tree.query(x[sl], k=[nk + 1], workers=workers)[0][:, 0]
        local = kth * (scale / np.cbrt(nk))
        radii = np.clip(local, 0.6 * d, radius_cap * d) * np.cbrt(c)

    precision = np.empty((len(x), 3, 3), dtype=float)
    axes = np.empty((len(x), 3), dtype=float)
    eye = np.eye(3)
    if anisotropy == 1:
        axes[:] = radii[:, None] * support
        precision[:] = eye / axes[:, :, None] ** 2
    else:
        # Positive covariance regularization and ratio floor also support
        # collinear/planar neighborhoods; no zero eigenvalue enters log/sqrt.
        ratio_floor = max((1.0 / float(anisotropy)) ** 2, np.finfo(float).eps)
        shrink = min(0.2, ratio_floor)
        for start in range(0, len(x), batch):
            sl = slice(start, start + batch)
            distance, near = tree.query(x[sl], k=nk + 1, workers=workers)
            distance, near = distance[:, 1:], near[:, 1:]
            length = distance[:, -1]
            if np.any(length <= 0) or not np.isfinite(length).all():
                raise ValueError("PCA requires finite positive distances between distinct points.")
            delta = (x[near] - x[sl, None, :]) / length[:, None, None]
            weights = c[near] / c[near].max(axis=1, keepdims=True)
            weights *= np.exp(-0.5 * (distance / length[:, None]) ** 2)
            weights /= weights.sum(axis=1, keepdims=True)
            delta -= np.einsum("nk,nkj->nj", weights, delta)[:, None, :]
            covariance = np.einsum("nk,nki,nkj->nij", weights, delta, delta)
            trace = np.trace(covariance, axis1=1, axis2=2)
            covariance = (1 - shrink) * covariance + (shrink / 3) * trace[:, None, None] * eye
            eigen, basis = np.linalg.eigh(covariance)
            floor = np.maximum(eigen[:, -1:] * ratio_floor, np.finfo(float).tiny)
            eigen = np.maximum(eigen, floor)
            relative = np.exp(0.5 * (np.log(eigen) - np.log(eigen).mean(axis=1, keepdims=True)))
            # A = D_world B D_relative maps the unit ball to the local shape.
            # Q = A^{-T} A^{-1}/r^2 remains correct when world/PCA axes differ.
            transform = support[None, :, None] * basis * relative[:, None, :]
            axes[sl] = np.linalg.svd(transform, compute_uv=False) * radii[sl, None]
            inverse = (basis.transpose(0, 2, 1) / support[None, None, :]
                       / relative[:, :, None] / radii[sl, None, None])
            precision[sl] = inverse.transpose(0, 2, 1) @ inverse
    if (not np.isfinite(precision).all() or not np.isfinite(axes).all() or
            not np.isfinite(radii).all() or np.any(axes <= 0) or np.any(radii <= 0)):
        raise ValueError("Support geometry is outside floating-point range; rescale coordinates or confidence.")
    if adaptive and support_smooth:
        previous = axes[:, 0].copy()
        precision, axes = _smooth_image_support(x, precision, axes, radii, c,
            k, support_smooth, workers, chunk_size)
        radii = radii * (axes[:, 0] / previous)
    return retained, d, precision, axes, radii


def _clean_labels(points, codes, confidence, label_confidence=None, *,
                  label_outlier_k=24, label_outlier_thresh=0.8,
                  label_outlier_iters=1, label_min_points=3,
                  label_outlier_action="relabel", graph_scale=2.0, workers=-1):
    """Correct unsupported label islands without removing any geometry."""
    x = np.asarray(points, dtype=np.float64)
    original = np.asarray(codes)
    geom = np.asarray(confidence, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError("points must have shape (N, 3) and be finite.")
    n = len(x)
    if (original.shape != (n,) or not np.issubdtype(original.dtype, np.integer)
            or np.any(original < 0)):
        raise ValueError("codes must be a length-N nonnegative integer array.")
    if geom.shape != (n,) or not np.isfinite(geom).all() or np.any(geom < 0):
        raise ValueError("confidence must be a finite nonnegative length-N array.")
    supplied = (np.ones(n, dtype=np.float64) if label_confidence is None
                else np.asarray(label_confidence, dtype=np.float64))
    if (supplied.shape != (n,) or not np.isfinite(supplied).all()
            or np.any((supplied < 0) | (supplied > 1))):
        raise ValueError("label_confidence must be a length-N array in [0, 1].")
    for name, value, minimum in (("label_outlier_k", label_outlier_k, 1),
                                 ("label_outlier_iters", label_outlier_iters, 0)):
        if np.ndim(value) or not np.isfinite(value) or int(value) != value or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if (np.ndim(label_outlier_thresh) or not np.isfinite(label_outlier_thresh)
            or not 0.5 < label_outlier_thresh <= 1):
        raise ValueError("label_outlier_thresh must be in (0.5, 1].")
    if (np.ndim(label_min_points) or not np.isfinite(label_min_points)
            or label_min_points < 0):
        raise ValueError("label_min_points must be finite and nonnegative.")
    if np.ndim(graph_scale) or not np.isfinite(graph_scale) or graph_scale <= 0:
        raise ValueError("graph_scale must be finite and positive.")
    if label_outlier_action not in {"relabel", "downweight", "none"}:
        raise ValueError("label_outlier_action must be 'relabel', 'downweight' or 'none'.")
    corrected = original.astype(np.int64, copy=True)
    reliability = supplied.copy()
    flagged = np.zeros(n, dtype=bool)
    support = np.full(n, np.nan, dtype=np.float64)
    report = dict(changed_mask=np.zeros(n, dtype=bool), outlier_mask=flagged,
                  support=support, iterations=0, n_changed=0, n_outliers=0)
    if (n < 4 or label_outlier_action == "none" or label_outlier_iters == 0
            or label_min_points == 0 or np.unique(corrected).size < 2):
        return corrected, reliability, report

    # np.unique sorts coordinates, making kNN tie handling independent of row order.
    sites, row_site = np.unique(x, axis=0, return_inverse=True)
    m = len(sites)
    if m < 4:
        return corrected, reliability, report
    nk = min(int(label_outlier_k), m - 1)
    distance, neighbors = cKDTree(sites).query(sites, k=nk + 1, workers=workers)
    distance, neighbors = distance[:, 1:], neighbors[:, 1:]
    local = np.minimum(np.median(distance[:, :min(6, nk)], axis=1),
                       2.0 * distance[:, 0])
    width = np.minimum(local[:, None], local[neighbors])
    valid = distance <= graph_scale * width
    row, column = np.nonzero(valid)
    other = neighbors[row, column]
    kernel = np.exp(-0.5 * (distance[row, column] / width[row, column]) ** 2)
    graph = sparse.csr_matrix((np.ones(len(row)), (row, other)), shape=(m, m))
    geometry_weights = sparse.csr_matrix((kernel, (row, other)), shape=(m, m))
    graph = graph.maximum(graph.T)
    # Symmetry is used for component membership; evidence uses each site's kNN.
    classes = np.unique(corrected)
    class_count = len(classes)

    for iteration in range(int(label_outlier_iters)):
        dense_code = np.searchsorted(classes, corrected)
        row_weight = geom * reliability
        seeds = sparse.coo_matrix((row_weight, (row_site, dense_code)),
                                  shape=(m, class_count)).tocsr()
        seeds.eliminate_zeros()
        mass = np.asarray(seeds.sum(axis=1)).ravel()
        site_conf = np.zeros(m)
        np.maximum.at(site_conf, row_site, row_weight)
        seeds = sparse.diags(np.divide(1.0, mass, out=np.zeros(m), where=mass > 0)) @ seeds
        weights = geometry_weights @ sparse.diags(site_conf)
        totals = np.asarray(weights.sum(axis=1)).ravel()
        votes = (sparse.diags(np.divide(1.0, totals, out=np.zeros(m), where=totals > 0))
                 @ weights @ seeds).tocsr()
        winner = np.asarray(votes.argmax(axis=1)).ravel()
        best = votes.max(axis=1).toarray().ravel()
        vote_rows = np.repeat(np.arange(m), np.diff(votes.indptr))
        runner_up = votes.copy()
        runner_up.data[runner_up.indices == winner[vote_rows]] = 0
        second = runner_up.max(axis=1).toarray().ravel()
        own = np.asarray(votes[row_site, dense_code]).ravel()
        if iteration == 0:
            support[:] = own

        # Count neighbour COORDINATES carrying the winning label, not duplicate rows.
        present = seeds.copy()
        present.data.fill(1.0)
        neighbor_count = geometry_weights.copy()
        neighbor_count.data.fill(1.0)
        vote_count = (neighbor_count @ present).tocsr()
        winner_count = np.asarray(vote_count[np.arange(m), winner]).ravel()
        evidence = ((best >= label_outlier_thresh) & (best - second >= 0.25)
                    & (winner_count >= 3))

        small = np.zeros(n, dtype=bool)
        order = np.argsort(dense_code, kind="stable")
        boundaries = np.r_[0, np.flatnonzero(np.diff(dense_code[order])) + 1, n]
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            ids = order[start:stop]
            active, mapping = np.unique(row_site[ids], return_inverse=True)
            threshold = (label_min_points * len(active) if label_min_points < 1
                         else label_min_points)
            if threshold <= 1:
                continue
            _, component = sparse.csgraph.connected_components(graph[active][:, active],
                                                               directed=False)
            component_size = np.bincount(component)
            small[ids] = component_size[component[mapping]] < threshold
        bad = small & evidence[row_site] & (winner[row_site] != dense_code)
        report["iterations"] = iteration + 1
        if not bad.any():
            break
        flagged |= bad
        if label_outlier_action == "relabel":
            corrected[bad] = classes[winner[row_site[bad]]]
            reliability[bad] = supplied[bad] * best[row_site[bad]]
        else:
            updated = supplied[bad] * own[bad]
            changed = np.any(np.abs(updated - reliability[bad]) > 1e-12)
            reliability[bad] = updated
            if not changed:
                break

    report["changed_mask"] = corrected != original
    report["n_changed"] = int(np.count_nonzero(report["changed_mask"]))
    report["n_outliers"] = int(np.count_nonzero(flagged))
    return corrected, reliability, report


def _category_values(keys, codes, for_vtk=True):
    """Keep categorical identity in numeric IDs; strings only for VTK objects."""
    # Construct object first: numpy otherwise coerces a mixed int/string list,
    # obscuring the original identity in return_info (mesh IDs remain distinct).
    values = np.empty(len(keys), dtype=object)
    values[:] = keys
    array = values[np.asarray(codes, dtype=np.int64)]
    if not for_vtk:
        return array
    if all(isinstance(k, (str, np.str_)) for k in keys):
        return array.astype(str)
    if all(isinstance(k, (int, float, np.integer, np.floating)) for k in keys):
        return np.asarray(array.tolist())
    return array.astype(str)


def _project_label_evidence(query, points, codes, weights, k, d, workers, chunk, target=None):
    """Confidence-weighted site voting; duplicate sites get one total weight."""
    usable = weights > 0
    points, codes, weights = points[usable], codes[usable], weights[usable]
    sites, inverse, counts = np.unique(points, axis=0, return_inverse=True, return_counts=True)
    classes, local = np.unique(codes, return_inverse=True)
    evidence = sparse.coo_matrix((weights, (inverse, local)), shape=(len(sites), len(classes))).tocsr()
    totals = np.asarray(evidence.sum(axis=1)).ravel()
    distribution = sparse.diags(1 / totals) @ evidence
    reliability = totals / counts
    tree = cKDTree(sites)
    k = min(int(k), len(sites))
    winner = np.empty(len(query), np.int32)
    certainty = np.empty(len(query), np.float32)
    targets = None if target is None else np.broadcast_to(np.asarray(target), (len(query),))
    columns = None if targets is None else np.searchsorted(classes, targets)
    for start in range(0, len(query), chunk):
        end = min(start + chunk, len(query))
        distance, ids = tree.query(query[start:end], k=list(range(1, k + 1)), workers=workers)
        w = reliability[ids] / (distance**2 + (0.05*d)**2)
        w /= w.sum(axis=1, keepdims=True)
        matrix = sparse.coo_matrix((w.ravel(), (np.repeat(np.arange(end-start), k), ids.ravel())),
                                   shape=(end-start, len(sites))).tocsr() @ distribution
        selected = np.asarray(matrix.argmax(axis=1)).ravel()
        winner[start:end] = classes[selected]
        if target is None:
            certainty[start:end] = matrix.max(axis=1).toarray().ravel()
        else:
            column = columns[start:end]
            valid = column < len(classes)
            valid &= classes[np.minimum(column, len(classes)-1)] == targets[start:end]
            certainty[start:end] = 0
            rows = np.flatnonzero(valid)
            certainty[start+rows] = np.asarray(matrix[rows, column[valid]]).ravel()
    return winner, certainty


def _annotate_clay_mesh(mesh, points, source_ids, confidence, feature,
                        d, score_k, score_scale, chunk, workers):
    """Attach shared source, feature and geometric-score mappings."""
    sampled = _sample_cloud(np.asarray(mesh.points), points, confidence, source_ids,
                            feature, d, score_k, score_scale, chunk, workers)
    for name, value in sampled.items():
        mesh.point_data[name] = value


def _clay_volume_level(values, target):
    """Fit one strict isolevel without arbitrarily splitting tied plateaus."""
    values = np.asarray(values, np.float32).ravel()
    if not len(values):
        return np.float32(0)
    if target <= 0:
        return values.max()
    if target >= len(values):
        return np.nextafter(values.min(), np.float32(-np.inf))
    cut = len(values) - int(target)
    ordered = np.partition(values, (cut - 1, cut))
    lo, hi = ordered[cut - 1], ordered[cut]
    if lo < hi:
        level = np.float32(float(lo) + (float(hi) - float(lo)) * 0.5)
        return lo if level >= hi else level
    above, tied = np.count_nonzero(values > hi), np.count_nonzero(values == hi)
    return (np.nextafter(hi, np.float32(-np.inf))
            if abs(above + tied - target) <= abs(above - target) else hi)


def _cloud_inputs(points, label=None, confidence=None, label_confidence=None,
                  feature=None, *, min_points=2):
    """Normalize cloud observations once and preserve original-row/class IDs."""
    raw = points
    if isinstance(raw, (str, Path)):
        import pyvista as pv
        raw = pv.read(raw)
    dataset = raw if hasattr(raw, "points") else None
    values = dataset.points if dataset is not None else raw
    x = np.asarray(values.detach().cpu().numpy() if hasattr(values, "detach") else values,
                   dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("points must be an (N,3) array or a point-cloud dataset/path.")
    n = len(x)

    def read(value, name, allow_sparse=False, column=False):
        if value is None:
            return None
        if isinstance(value, str):
            if dataset is None:
                raise ValueError(f"Named {name} requires a point-cloud dataset/path.")
            if value not in dataset.point_data:
                raise ValueError(f"Point-data array {value!r} was not found for {name}.")
            value = dataset.point_data[value]
        if sparse.issparse(value):
            if column:
                if value.shape != (n, 1):
                    raise ValueError(f"Sparse {name} must have shape (N,1).")
                result = value.toarray().ravel()
            elif allow_sparse:
                result = value.tocsr(copy=True)
            else:
                raise ValueError(f"{name} requires one category per point, not sparse scores.")
        else:
            result = np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach")
                                else value)
            if column and result.shape == (n, 1):
                result = result[:, 0]
        if result.ndim == 0 or result.shape[0] != n:
            raise ValueError(f"{name} must align with the N original points.")
        return result

    labels = read(label, "label")
    feature = read(feature, "feature", allow_sparse=True)
    c = read(confidence, "confidence", column=True)
    lc = read(label_confidence, "label_confidence", column=True)
    c = np.ones(n) if c is None else np.asarray(c, dtype=np.float64).copy()
    lc = np.ones(n) if lc is None else np.asarray(lc, dtype=np.float64).copy()
    for name, value in (("confidence", c), ("label_confidence", lc)):
        if (value.shape != (n,) or not np.isfinite(value).all()
                or np.any((value < 0) | (value > 1))):
            raise ValueError(f"{name} must contain N finite values in [0,1].")
    if feature is not None:
        feature_data = feature.data if sparse.issparse(feature) else feature
        if (feature.ndim not in (1, 2) or (feature.ndim == 2 and feature.shape[1] == 0)
                or not np.issubdtype(feature.dtype, np.number) or np.iscomplexobj(feature_data)
                or not np.isfinite(feature_data).all()):
            raise ValueError("feature must contain finite real values with shape (N,) or (N,C).")

    keys, codes = None, None
    if labels is not None:
        if labels.shape != (n,):
            raise ValueError("label must have shape (N,).")
        keys, codes = _label_codes(labels)
    if (np.ndim(min_points) or not np.isfinite(min_points)
            or int(min_points) != min_points or min_points < 1):
        raise ValueError("min_points must be a positive integer.")
    finite = np.isfinite(x).all(axis=1)
    source = np.flatnonzero(finite & (c > 0))
    _, inverse = np.unique(x[source], axis=0, return_inverse=True)
    order = np.lexsort((source, -c[source], inverse))
    first = np.r_[True, np.diff(inverse[order]) != 0] if len(order) else np.empty(0, bool)
    unique_source = source[order[first]]
    if len(unique_source) < min_points:
        raise ValueError(f"At least {int(min_points)} distinct finite positive-confidence points are required.")
    return dict(points=x, label=labels, keys=keys, codes=codes, confidence=c,
                label_confidence=lc, feature=feature, source=source,
                unique_source=unique_source, inverse=inverse, finite_mask=finite)

def _cloud_label_observations(data, retained=None, *, offset=None,
                              label_outlier_k=24, label_outlier_thresh=0.8,
                              label_outlier_iters=1, label_min_points=3,
                              label_outlier_action="relabel", graph_scale=2.0,
                              workers=-1):
    """Select geometry survivors, then clean only their label evidence."""
    unique_source = data["unique_source"]
    m = len(unique_source)
    if retained is None:
        retained = np.arange(m)
    retained = np.asarray(retained)
    if not retained.size:
        retained = retained.astype(np.int64)
    if retained.dtype == bool:
        if retained.shape != (m,):
            raise ValueError("A retained Boolean mask must align with unique geometry sites.")
        retained = np.flatnonzero(retained)
    if (retained.ndim != 1 or not np.issubdtype(retained.dtype, np.integer)
            or np.any((retained < 0) | (retained >= m))
            or np.unique(retained).size != len(retained)):
        raise ValueError("retained must contain distinct valid unique-site indices.")
    lookup = np.full(m, -1, dtype=np.int64)
    lookup[retained] = np.arange(len(retained))
    keep = lookup[data["inverse"]] >= 0
    observations = data["source"][keep]
    obs_site = lookup[data["inverse"][keep]]
    unique_source = unique_source[retained]
    x = data["points"]
    observation_points = x[observations]
    n = len(x)
    corrected = None if data["codes"] is None else data["codes"].copy()
    weights = data["label_confidence"][observations].copy()
    report = dict(changed_mask=np.zeros(len(observations), bool),
                  outlier_mask=np.zeros(len(observations), bool),
                  support=np.full(len(observations), np.nan), iterations=0,
                  n_changed=0, n_outliers=0)
    if corrected is not None and len(observations):
        shift = (np.median(x[unique_source], axis=0) if offset is None
                 else np.asarray(offset, dtype=float))
        if shift.shape != (3,) or not np.isfinite(shift).all():
            raise ValueError("offset must have three finite coordinates.")
        corrected[observations], weights, report = _clean_labels(
            observation_points - shift, corrected[observations],
            data["confidence"][observations], weights,
            label_outlier_k=label_outlier_k, label_outlier_thresh=label_outlier_thresh,
            label_outlier_iters=label_outlier_iters, label_min_points=label_min_points,
            label_outlier_action=label_outlier_action, graph_scale=graph_scale, workers=workers)
    inlier = np.zeros(n, bool)
    inlier[observations] = True
    reliability = np.zeros(n)
    reliability[observations] = weights
    outlier = np.zeros(n, bool)
    outlier[observations] = report["outlier_mask"]
    changed = np.zeros(n, bool)
    changed[observations] = report["changed_mask"]
    return dict(points=x[unique_source], unique_source=unique_source,
                observations=observations, obs_site=obs_site,
                observation_points=observation_points, keys=data["keys"],
                codes=None if corrected is None else corrected[observations],
                weights=data["confidence"][observations] * weights,
                corrected_codes=corrected, label_reliability=reliability,
                inlier_mask=inlier, label_outlier_mask=outlier,
                label_changed_mask=changed, label_report=report)

def _grid_label_data(mask, origin, spacing, observations, *, direction=None,
                     mode="joint", d=1.0, label_smooth=0.5, k=8, workers=-1,
                     score=True, point_points=None, point_indices=None,
                     label_max_iter=1000, label_tol=1e-5, label_lock_seeds=False,
                     label_smooth_weight=None, label_axis_weights=None,
                     chunk_size=50_000, label_post_smooth=0.0):
    """Attach globally indexed labels to an existing mask without changing it."""
    mask = np.asarray(mask, bool)
    origin = np.asarray(origin, dtype=float)
    try:
        spacing = np.broadcast_to(np.asarray(spacing, dtype=float), (3,))
    except ValueError as error:
        raise ValueError("spacing must be a positive scalar or XYZ.") from error
    direction = np.eye(3) if direction is None else np.asarray(direction, dtype=float)
    if (mask.ndim != 3 or origin.shape != (3,) or not np.isfinite(origin).all()
            or not np.isfinite(spacing).all() or np.any(spacing <= 0)):
        raise ValueError("Require a 3-D mask, finite XYZ origin and positive spacing.")
    if (direction.shape != (3, 3) or not np.isfinite(direction).all()
            or not np.allclose(direction.T @ direction, np.eye(3), atol=1e-7, rtol=0)):
        raise ValueError("direction columns must be orthonormal grid axes in world coordinates.")
    if mode not in {"joint", "project"}:
        raise ValueError("Grid label mode must be joint or project; separate geometry belongs to the caller.")
    if np.ndim(d) or not np.isfinite(d) or d <= 0:
        raise ValueError("d must be a positive finite reference length.")
    for name, value in (("k", k), ("chunk_size", chunk_size)):
        if np.ndim(value) or not np.isfinite(value) or int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not isinstance(score, (bool, np.bool_)):
        raise ValueError("score must be a Boolean.")
    keys = observations["keys"]
    if keys is None or observations["codes"] is None:
        raise ValueError("Label attachment requires categorical label observations.")
    if len(keys) >= np.iinfo(np.int32).max:
        raise ValueError("Too many categories for an int32 label image.")
    points = np.asarray(observations["observation_points"], float)
    codes = np.asarray(observations["codes"], np.int64)
    weights = np.asarray(observations["weights"], float)
    if (points.ndim != 2 or points.shape[1] != 3 or codes.shape != (len(points),)
            or weights.shape != (len(points),) or not np.isfinite(points).all()
            or not np.isfinite(weights).all() or np.any(weights < 0)
            or np.any((codes < 0) | (codes >= len(keys)))):
        raise ValueError("Invalid observation coordinates, global class IDs or weights.")
    local_points = (points - origin) @ direction
    active = np.flatnonzero(mask)
    if len(active) and not np.any(weights > 0):
        raise ValueError("No positive label evidence remains for the occupied grid.")
    label_mask = np.zeros(mask.shape, dtype=np.int32)
    label_score = np.zeros(mask.shape, dtype=np.float32) if score else None
    info = dict(label_mode=mode)
    if len(active) and mode == "joint":
        label_mask, certainty, info_joint = _label_volume(
            mask, np.zeros(3), spacing, local_points, codes, weights, d,
            label_smooth, int(k), workers, label_max_iter=label_max_iter,
            label_tol=label_tol, label_lock_seeds=label_lock_seeds,
            label_smooth_weight=label_smooth_weight, label_axis_weights=label_axis_weights,
            chunk_size=int(chunk_size), label_post_smooth=label_post_smooth)
        if score:
            label_score = certainty
        info.update(info_joint)
    elif len(active):
        # One projector preparation; its vote matrix is processed in chunks.
        # Store O(V*3) coordinates, never a dense O(V*C) probability array.
        query = np.empty((len(active), 3), dtype=np.float64)
        query[:, 0] = (active // (mask.shape[1] * mask.shape[2])) * spacing[0]
        query[:, 1] = ((active // mask.shape[2]) % mask.shape[1]) * spacing[1]
        query[:, 2] = (active % mask.shape[2]) * spacing[2]
        chosen, certainty = _project_label_evidence(
            query, local_points, codes, weights, int(k), d, workers, int(chunk_size))
        label_mask.ravel()[active] = chosen + 1
        if score:
            label_score.ravel()[active] = certainty
        del query

    if mode == "project" and np.any(_xyz(label_post_smooth, "label_post_smooth")):
        local_obs = dict(observations, observation_points=local_points)
        prior_score = np.zeros(mask.shape, np.float32)
        if len(active):
            prior_score.ravel()[active] = certainty
        label_mask, refined_score, post = _post_label_volume(label_mask, prior_score,
            np.zeros(3), spacing, d, local_obs, label_post_smooth, label_lock_seeds,
            label_max_iter, label_tol, workers, chunk_size)
        if score:
            label_score = refined_score
        info.update(post)

    return _label_array_result(label_mask, label_score, keys, info, origin, spacing,
                               direction, point_points, point_indices)


def _label_array_result(label_mask, label_score, keys, info, origin, spacing,
                        direction, point_points=None, point_indices=None):
    """Keep image/grid category IDs, names and sampled scores aligned."""
    mask = label_mask > 0
    score = label_score is not None
    result = dict(label_mask=label_mask, label_map={0: None, **{
        i + 1: key for i, key in enumerate(keys)}}, label_info=info)
    if score:
        result["label_score_image"] = label_score
    if point_points is None and point_indices is None:
        return result
    if point_points is not None and point_indices is not None:
        raise ValueError("Use point_points or point_indices, not both.")
    valid = None
    if point_points is not None:
        query = np.asarray(point_points, dtype=float)
        if query.ndim != 2 or query.shape[1] != 3 or not np.isfinite(query).all():
            raise ValueError("point_points must be finite world XYZ with shape (M,3).")
        index = np.rint(((query - origin) @ direction) / spacing).astype(np.int64)
        valid = np.all((index >= 0) & (index < np.asarray(mask.shape)), axis=1)
        flat = np.zeros(len(index), np.int64)
        flat[valid] = np.ravel_multi_index(index[valid].T, mask.shape)
    elif point_indices is not None:
        index = np.asarray(point_indices)
        if not index.size:
            index = index.astype(np.int64)
        if not np.issubdtype(index.dtype, np.integer):
            raise ValueError("point_indices must contain integers.")
        if index.ndim == 1:
            flat = index.astype(np.int64, copy=False)
            valid = (flat >= 0) & (flat < mask.size)
            flat = np.where(valid, flat, 0)
        elif index.ndim == 2 and index.shape[1] == 3:
            valid = np.all((index >= 0) & (index < np.asarray(mask.shape)), axis=1)
            flat = np.zeros(len(index), np.int64)
            flat[valid] = np.ravel_multi_index(index[valid].T, mask.shape)
        else:
            raise ValueError("point_indices must be flat indices or an (M,3) index array.")
    point_ids = np.zeros(len(flat), np.int32)
    point_ids[valid] = label_mask.ravel()[flat[valid]]
    names = np.empty(len(keys) + 1, dtype=object)
    names[0] = None
    names[1:] = keys
    result.update(point_label_id=point_ids, point_label=names[point_ids])
    if score:
        point_score = np.zeros(len(flat), np.float32)
        point_score[valid] = label_score.ravel()[flat[valid]]
        result["point_label_score"] = point_score
    return result


def _component_mask(mask, spacing=1.0, d=1.0, *, min_volume_component=0.0,
                    min_voxels=0, component_diameter_scale=0.0,
                    keep_largest=False, component_connectivity=6,
                    reference_count=None):
    """Filter occupied components by relative, voxel-count and physical volume."""
    mask = np.asarray(mask, bool)
    if mask.ndim != 3:
        raise ValueError("mask must be a three-dimensional XYZ array.")
    spacing = _xyz(spacing, "spacing", True)
    if np.ndim(d) or not np.isfinite(d) or d <= 0:
        raise ValueError("d must be a positive finite reference length.")
    if (np.ndim(min_volume_component) or not np.isfinite(min_volume_component)
            or not 0 <= min_volume_component <= 1):
        raise ValueError("min_volume_component must lie in [0,1].")
    if (np.ndim(min_voxels) or not np.isfinite(min_voxels)
            or int(min_voxels) != min_voxels or min_voxels < 0):
        raise ValueError("min_voxels must be a nonnegative integer.")
    if (np.ndim(component_diameter_scale) or not np.isfinite(component_diameter_scale)
            or component_diameter_scale < 0):
        raise ValueError("component_diameter_scale must be finite and nonnegative.")
    if component_connectivity not in (6, 18, 26):
        raise ValueError("component_connectivity must be 6, 18 or 26.")
    if not isinstance(keep_largest, (bool, np.bool_)):
        raise ValueError("keep_largest must be boolean.")
    if reference_count is not None and (np.ndim(reference_count)
            or not np.isfinite(reference_count) or reference_count < 0):
        raise ValueError("reference_count must be a finite nonnegative scalar or None.")
    structure = ndi.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[component_connectivity])
    regions, count = ndi.label(mask, structure=structure)
    sizes = np.bincount(regions.ravel(), minlength=count + 1)
    sizes[0] = 0
    largest = int(sizes.max())
    reference = largest if reference_count is None else float(reference_count)
    voxel_volume = float(np.prod(spacing))
    physical_cutoff = np.pi / 6 * (float(d) * component_diameter_scale) ** 3
    cutoff = max(float(min_voxels), float(min_volume_component) * reference,
                 physical_cutoff / voxel_volume)
    retained = sizes >= cutoff
    retained[0] = False
    if keep_largest and count:
        largest_id = int(sizes.argmax())
        retained &= np.arange(len(sizes)) == largest_id
    cleaned = retained[regions]
    kept = int(np.count_nonzero(retained))
    info = dict(component_count=int(count), kept_components=kept,
                removed_components=int(count) - kept,
                removed_voxels=int(np.count_nonzero(mask & ~cleaned)),
                largest_component_voxels=largest,
                min_component_voxels=float(cutoff),
                component_connectivity=int(component_connectivity))
    return cleaned, info

def _process_volume(mask, spacing, d, *, smooth_sigma=0.0, smooth_scales=(1.0,),
                    smooth_method="sdf", density_level=0.5, preserve_volume=True,
                    max_shift_scale=None, max_shift=None, opening_scale=0.0,
                    closing_scale=0.0, hole_scale=0.0, offset_scale=0.0,
                    min_volume_component=0.0, pre_min_component=0.0,
                    min_voxels=0, component_diameter_scale=0.0,
                    keep_largest=False, forbidden=None, device="cpu",
                    morph_order=("opening", "closing"), initial_field=None,
                    component_connectivity=6, hole_stage="before_smoothing",
                    offset_method="field", field_smooth_mode="reembed"):
    """Apply shared morphology, smoothing and component controls to an XYZ mask."""
    current = np.asarray(mask, bool).copy()
    if current.ndim != 3 or any(n == 0 for n in current.shape):
        raise ValueError("mask must be a nonempty three-dimensional XYZ array.")
    spacing = _xyz(spacing, "spacing", True)
    sigma = _xyz(smooth_sigma, "smooth_sigma")
    opening = _xyz(opening_scale, "opening_scale")
    closing = _xyz(closing_scale, "closing_scale")
    if np.ndim(d) or not np.isfinite(d) or d <= 0:
        raise ValueError("d must be a positive finite reference length.")
    d = float(d)
    method = None if smooth_method is None else str(smooth_method).lower()
    if method == "occupancy":
        method = "density"
    if method not in (None, "sdf", "density"):
        raise ValueError("smooth_method must be 'sdf', 'density', 'occupancy' or None.")
    if np.ndim(density_level) or not np.isfinite(density_level) or not 0 < density_level < 1:
        raise ValueError("density_level must lie strictly between 0 and 1.")
    stages = np.atleast_1d(np.asarray(smooth_scales, float))
    if stages.ndim != 1 or not len(stages) or not np.isfinite(stages).all() or np.any(stages <= 0):
        raise ValueError("smooth_scales must be a nonempty positive finite sequence.")
    if not isinstance(preserve_volume, (bool, np.bool_)):
        raise ValueError("preserve_volume must be boolean.")
    if max_shift is not None and max_shift_scale is not None:
        raise ValueError("Pass max_shift or max_shift_scale, not both.")
    for name, value in (("max_shift", max_shift), ("max_shift_scale", max_shift_scale)):
        if value is not None and (np.ndim(value) or not np.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be a finite nonnegative scalar or None.")
    shift = max_shift if max_shift_scale is None else d * max_shift_scale
    if np.ndim(hole_scale) or np.isnan(hole_scale) or hole_scale < 0:
        raise ValueError("hole_scale must be nonnegative or np.inf.")
    if np.ndim(offset_scale) or not np.isfinite(offset_scale):
        raise ValueError("offset_scale must be a finite signed scalar.")
    pre = min_volume_component if pre_min_component is None else pre_min_component
    if np.ndim(pre) or not np.isfinite(pre) or not 0 <= pre <= 1:
        raise ValueError("pre_min_component must lie in [0,1] or be None.")
    order = tuple(morph_order)
    if len(order) != 2 or set(order) != {"opening", "closing"}:
        raise ValueError("morph_order must contain 'opening' and 'closing' exactly once.")
    if hole_stage not in ("before_smoothing", "after_offset"):
        raise ValueError("hole_stage must be 'before_smoothing' or 'after_offset'.")
    if offset_method not in ("field", "morphology"):
        raise ValueError("offset_method must be 'field' or 'morphology'.")
    if field_smooth_mode not in {"continuous", "reembed"}:
        raise ValueError("field_smooth_mode must be 'continuous' or 'reembed'.")
    field = None
    if initial_field is not None:
        field = np.asarray(initial_field)
        if field.shape != current.shape or not np.isfinite(field).all():
            raise ValueError("initial_field must be finite and have the same shape as mask.")
        if not np.array_equal(field > 0, current):
            raise ValueError("initial_field must be positive exactly on the supplied mask.")
        field = field.copy() if np.issubdtype(field.dtype, np.floating) else field.astype(float)
    blocked = None if forbidden is None else np.asarray(forbidden, bool)
    if blocked is not None and blocked.shape != current.shape:
        raise ValueError("forbidden must have the same shape as mask.")
    if blocked is not None and not blocked.any():
        blocked = None
    raw_count = int(current.sum())
    protected_removed = 0
    allowed_field = None
    if blocked is not None:
        protected_removed = int(np.count_nonzero(current & blocked))
        current &= ~blocked
        allowed_field = _sdf(blocked, spacing).astype(np.float32)
        if field is not None:
            np.minimum(field, allowed_field, out=field)
    pre_removed = 0
    if pre > 0:
        cleaned, before = _component_mask(current, spacing, d,
            min_volume_component=pre, keep_largest=keep_largest,
            component_connectivity=component_connectivity)
        pre_removed = before["removed_voxels"]
        if field is not None and pre_removed:
            removed = current & ~cleaned
            field[removed] = -np.maximum(np.abs(field[removed]), spacing.min() / 2)
        current = cleaned
    for operation in order:
        radius = 0.5 * d * (opening if operation == "opening" else closing)
        if not np.any(radius) or not current.any():
            continue
        original = current
        first, second = (("erode", "dilate") if operation == "opening" else
                         ("dilate", "erode"))
        current = _morph(current, radius, spacing, first)
        if blocked is not None:
            current &= ~blocked
        current = _morph(current, radius, spacing, second)
        if blocked is not None:
            current &= ~blocked
        if operation == "closing":
            current |= original
        if not np.array_equal(current, original):
            field = None
    if hole_stage == "before_smoothing" and hole_scale > 0 and current.any():
        updated = _fill_volume_holes(current, spacing, d * hole_scale, blocked)
        if not np.array_equal(updated, current):
            field = None
        current = updated
    morph_count = int(current.sum())
    active = method is not None and np.any(sigma > 0) and shift != 0
    if current.any() and (active or field is None):
        if method is None:
            field = -_sdf(current, spacing).astype(np.float32)
        else:
            field, current = _clay_smooth_field(current, spacing, sigma, stages,
                method, density_level, preserve_volume, shift, device, forbidden=blocked,
                smooth_mode=field_smooth_mode, initial_field=field)
    if field is None:
        field = np.full(current.shape, -float(spacing.min()), np.float32)
    smoothed_count = int(current.sum())
    if offset_scale and current.any():
        if offset_method == "morphology":
            current = _morph(current, np.full(3, abs(d * offset_scale)), spacing,
                             "dilate" if offset_scale > 0 else "erode")
            if blocked is not None:
                current &= ~blocked
            field = -_sdf(current, spacing).astype(np.float32)
        else:
            field = (-_sdf(current, spacing).astype(np.float32)
                     + np.float32(d * offset_scale))
        if allowed_field is not None:
            np.minimum(field, allowed_field, out=field)
        current = field > 0
    if hole_stage == "after_offset" and hole_scale > 0 and current.any():
        updated = _fill_volume_holes(current, spacing, d * hole_scale, blocked)
        if not np.array_equal(updated, current):
            if field_smooth_mode == "continuous":
                # Filling an enclosed cavity must not quantize the already
                # smoothed OUTER zero-set by rebuilding a global binary EDT.
                added = updated & ~current
                field[added] = np.maximum(np.abs(field[added]), spacing.min() / 2)
            else:
                field = -_sdf(updated, spacing).astype(np.float32)
        current = updated
    cleaned, components = _component_mask(current, spacing, d,
        min_volume_component=min_volume_component, min_voxels=min_voxels,
        component_diameter_scale=component_diameter_scale, keep_largest=keep_largest,
        component_connectivity=component_connectivity)
    if components["removed_voxels"]:
        removed = current & ~cleaned
        field[removed] = -np.maximum(np.abs(field[removed]), spacing.min() / 2)
    current = cleaned
    # Projection/mesh consumers rely on this exact mask/field sign agreement.
    if blocked is not None:
        np.minimum(field, allowed_field, out=field)
        current = field > 0
    voxel_volume = float(np.prod(spacing))
    metadata = dict(raw_volume=raw_count * voxel_volume,
        post_morph_volume=morph_count * voxel_volume,
        smoothing_volume=smoothed_count * voxel_volume,
        final_volume=int(current.sum()) * voxel_volume,
        volume_initial=morph_count * voxel_volume,
        volume_final=smoothed_count * voxel_volume,
        volume_output=int(current.sum()) * voxel_volume,
        protected_removed_voxels=protected_removed, pre_removed_voxels=pre_removed,
        pre_min_component=float(pre), min_volume_component=float(min_volume_component),
        smooth_sigma=sigma.copy(), smooth_scales=stages.copy(),
        field_smooth_mode=field_smooth_mode,
        smooth_method="none" if method is None else method,
        max_shift=np.inf if shift is None else float(shift),
        **components)
    if not np.array_equal(current, field > 0):
        raise RuntimeError("Volume processing produced inconsistent mask/field signs.")
    if blocked is not None and np.any(current & blocked):
        raise RuntimeError("Volume processing entered a protected background region.")
    return current, field, metadata


def _grid_support(points, confidence, tree, radii, origin, spacing, shape,
                  axis_scale, method, k, bandwidth, density_level, chunk, workers):
    """Original kNN/KDE supports plus confidence-aware exact nearest supports."""
    if method == "kde":
        density = np.zeros(shape, dtype=np.float32)
        for start in range(0, len(points), chunk):
            sl = slice(start, start + chunk)
            q = _grid_coordinates(points[sl] * axis_scale, origin, spacing, shape)
            ijk = np.floor(q).astype(np.int64)
            frac = q - ijk
            for offset in np.ndindex(2, 2, 2):
                offset = np.asarray(offset)
                weights = confidence[sl] * np.prod(np.where(offset, frac, 1 - frac), axis=1)
                # A point on the final node has zero weight for its +1 corner.
                # Exclude that corner before indexing instead of clipping it inward.
                active = weights > 0
                if np.any(active):
                    np.add.at(density, tuple((ijk[active] + offset).T), weights[active])
        ndi.gaussian_filter(density, bandwidth / spacing, output=density,
                            mode="constant", truncate=3.0)
        sampled = np.empty(len(points), np.float32)
        for start in range(0, len(points), chunk):
            sl = slice(start, start + chunk)
            q = _grid_coordinates(points[sl] * axis_scale, origin, spacing, shape)
            sampled[sl] = ndi.map_coordinates(density, q.T, order=1,
                                              mode="constant", prefilter=False)
        return density > density_level * float(np.median(sampled))
    total = int(np.prod(shape))
    mask = np.zeros(total, bool)
    lookup = np.r_[radii, -1.0]
    maximum = float(np.max(radii))
    for start in range(0, total, chunk):
        stop = min(total, start + chunk)
        ijk = np.column_stack(np.unravel_index(np.arange(start, stop), shape))
        query = (origin + ijk * spacing) / axis_scale
        if method == "knn":
            distance, ids = tree.query(query, k=list(range(1, k + 1)), workers=workers)
            mask[start:stop] = np.mean((distance / radii[ids]) ** 2, axis=1) <= 1
            continue
        # Without confidence all radii are equal, reproducing fixed nearest.
        unresolved, kk = np.arange(len(query)), min(8, len(points))
        while len(unresolved):
            pending = []
            batch = max(1, min(chunk, 1_000_000 // kk))
            for offset in range(0, len(unresolved), batch):
                rows = unresolved[offset:offset + batch]
                distance, ids = tree.query(query[rows], k=list(range(1, kk + 1)),
                    distance_upper_bound=np.nextafter(maximum, np.inf), workers=workers)
                inside = np.any(distance <= lookup[ids], axis=1)
                mask[start + rows[inside]] = True
                revisit = ~inside & (distance[:, -1] <= maximum)
                if kk < len(points) and revisit.any():
                    pending.append(rows[revisit])
            if not pending:
                break
            unresolved, kk = np.concatenate(pending), min(len(points), 2 * kk)
    return mask.reshape(shape)


def _grid_point_cloud_impl(
    points, spacing=None, *,
    spacing_scale=0.5, base_spacing=None, base_spacing_scale=1.0,
    axis_scale=1.0, direction=None,
    method="knn", support_scale=1.25, support_k=8,
    support_radius=None, radius_cap=2.0,
    kernel_scale=2.0, density_level=0.2,
    smooth_sigma=None, smooth=None, smooth_scale=1.0, smooth_scales=(1.0,),
    smooth_method="sdf", max_shift=None, preserve_volume=True,
    outlier_method="lof", outlier_k=24, outlier_thresh=1.8,
    outlier_iters=1, outlier_radius=None,
    min_points=None, min_component=None, keep_largest=False,
    graph_scale=None, component_scale=None,
    closing=0.0, opening=0.0, fill_holes=False,
    output="volume", backend=None, mc_method=None, device="cpu",
    max_voxels=20_000_000_000_000, chunk_size=100_000, workers=-1,
    return_info=False, return_field=False,
    confidence=None, geometry_confidence=None, feature=None,
    label=None, label_confidence=None, label_mode="project",
    label_smooth=None, label_smooth_weight=None, label_axis_weights=None,
    label_max_iter=1000, label_tol=1e-5, label_lock_seeds=False, label_project_k=1,
    label_outlier_k=24, label_outlier_thresh=0.8, label_outlier_iters=1,
    label_min_points=3, label_outlier_action="relabel",
    separate_protect_labels=True, label_guard_scale=0.6,
    closing_scale=None, opening_scale=None, hole_scale=None, offset_scale=0.0,
    max_shift_scale=_GRID_UNSET, field_smooth_iters=1,
    pre_min_component=0.0, min_volume_component=0.0,
    component_diameter_scale=0.0, score_k=8, score_scale=2.0, field_smooth_mode="reembed",
    label_guard_mode="hard", label_post_smooth=0.0, support_smooth=0.0,
    separate_geometry="local", separate_output="per_label", interface_backend="auto",
    _foreign_points=None, _support_confidence=None,
):
    """Reconstruct a regular XYZ grid with an optional orthonormal world basis.
    
    method: knn adaptive RMS support, nearest fixed support, kde point deposition.
    axis_scale changes the metric; direction columns are world grid axes.
    output volume/boundary -> world XYZ points, mask -> bool XYZ array, mesh ->
    (vertices, faces). return_info wraps the output; return_field adds a NEGATIVE-
    inside field to info. Joint mesh output is the outer shell only; label_mask
    in info retains the shared partition. Image/grid label IDs start at 1.
    
    project votes, joint diffuses, separate defaults to independent class outputs.
    Soft guards, positive label_post_smooth, separate_output='merged' or
    separate_geometry='shared' enable competition on a common occupied grid.
    Per-label shared mesh output retains exact common interface coordinates/IDs.
    Label confidence never rescales geometric score or feature interpolation.
    
    Canonical opening/closing are XYZ diameters*d, applied open -> close; legacy
    opening/closing are metric radii with close -> open order. Do not mix them.
    hole_scale bounds enclosed 26-connected cavities; legacy fill_holes=True
    means all eligible cavities. Positive offset expands. min_component aliases
    POINT min_points; component_scale aliases graph_scale. Volume cutoffs use
    min_volume_component/component_diameter_scale instead.
    smooth/smooth_sigma are physical; smooth_scale is relative to d. Sigma is
    multiplied by sqrt(field_smooth_iters). field_smooth_mode selects reembed or
    continuous. max_shift=None keeps the legacy automatic band; explicit
    max_shift_scale=None removes the field clamp. Zero disables smoothing.
    support_smooth regularizes support radii and is unavailable for KDE.
    confidence/geometry_confidence and backend/mc_method are aliases."""
    post_smooth = _label_refinement_options(label_guard_mode, label_post_smooth)
    if np.ndim(support_smooth) or not np.isfinite(support_smooth) or not 0 <= support_smooth <= 1:
        raise ValueError("support_smooth must lie in [0,1].")
    if support_smooth and method == "kde":
        raise ValueError("support_smooth is unavailable for KDE's global bandwidth.")
    if separate_geometry not in {"shared", "local"} or separate_output not in {"merged", "per_label"}:
        raise ValueError("Invalid separate_geometry or separate_output.")
    shared_separate = label_mode == "separate" and (label_guard_mode == "soft" or
        np.any(post_smooth) or separate_output == "merged" or separate_geometry == "shared")
    if field_smooth_mode not in {"continuous", "reembed"}:
        raise ValueError("field_smooth_mode must be continuous or reembed.")
    smooth_sigma = _alias(smooth_sigma, smooth, None, "smooth_sigma", "smooth")
    min_points = _alias(min_points, min_component, 0.01, "min_points", "min_component")
    graph_scale = _alias(graph_scale, component_scale, 2.0, "graph_scale", "component_scale")
    backend = _alias(backend, mc_method, "auto", "backend", "mc_method")
    confidence = _alias(confidence, geometry_confidence, None, "confidence", "geometry_confidence")
    # Snapshot only canonical public options. Per-class data and private support
    # weights are set explicitly below; transient locals can never leak in.
    input_parameters = dict(
        spacing=spacing, spacing_scale=spacing_scale, base_spacing=base_spacing,
        base_spacing_scale=base_spacing_scale, axis_scale=axis_scale, direction=direction,
        method=method, support_scale=support_scale, support_k=support_k,
        support_radius=support_radius, radius_cap=radius_cap, kernel_scale=kernel_scale,
        density_level=density_level, smooth_sigma=smooth_sigma, smooth_scale=smooth_scale,
        smooth_scales=smooth_scales, smooth_method=smooth_method, max_shift=max_shift,
        preserve_volume=preserve_volume, outlier_method=outlier_method, outlier_k=outlier_k,
        outlier_thresh=outlier_thresh, outlier_iters=outlier_iters, outlier_radius=outlier_radius,
        min_points=min_points, keep_largest=keep_largest, graph_scale=graph_scale,
        closing=closing, opening=opening, fill_holes=fill_holes,
        output=output, backend=backend, device=device,
        max_voxels=max_voxels, chunk_size=chunk_size, workers=workers,
        return_info=return_info, return_field=return_field, confidence=confidence,
        feature=feature, label=label, label_confidence=label_confidence,
        label_mode=label_mode, label_smooth=label_smooth, label_smooth_weight=label_smooth_weight,
        label_axis_weights=label_axis_weights, label_max_iter=label_max_iter, label_tol=label_tol,
        label_lock_seeds=label_lock_seeds, label_project_k=label_project_k, label_outlier_k=label_outlier_k,
        label_outlier_thresh=label_outlier_thresh, label_outlier_iters=label_outlier_iters, label_min_points=label_min_points,
        label_outlier_action=label_outlier_action, separate_protect_labels=separate_protect_labels, label_guard_scale=label_guard_scale,
        closing_scale=closing_scale, opening_scale=opening_scale, hole_scale=hole_scale,
        offset_scale=offset_scale, max_shift_scale=max_shift_scale, field_smooth_iters=field_smooth_iters,
        pre_min_component=pre_min_component, min_volume_component=min_volume_component, component_diameter_scale=component_diameter_scale,
        score_k=score_k, score_scale=score_scale, field_smooth_mode=field_smooth_mode,
        label_guard_mode=label_guard_mode, label_post_smooth=label_post_smooth,
        support_smooth=support_smooth, separate_geometry=separate_geometry,
        separate_output=separate_output, interface_backend=interface_backend,
    )
    if method not in {"knn", "nearest", "kde"} or output not in {"volume", "boundary", "mask", "mesh"}:
        raise ValueError("Unknown reconstruction method or output.")
    if smooth_method not in {"sdf", "occupancy", None}:
        raise ValueError("smooth_method must be 'sdf', 'occupancy' or None.")
    if label_mode not in {"project", "joint", "separate"}:
        raise ValueError("label_mode must be 'project', 'joint' or 'separate'.")
    if backend not in {"auto", "pymcubes", "skimage", "torchmcubes", "lewiner", "lorensen", "vtk"}:
        raise ValueError("Unknown marching-cubes backend.")
    for name, value, minimum in (("support_k", support_k, 1), ("outlier_k", outlier_k, 1),
            ("outlier_iters", outlier_iters, 0), ("chunk_size", chunk_size, 1),
            ("max_voxels", max_voxels, 1), ("score_k", score_k, 1),
            ("field_smooth_iters", field_smooth_iters, 0), ("label_project_k", label_project_k, 1)):
        if np.ndim(value) or not np.isfinite(value) or value != int(value) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    support_k, outlier_k, chunk_size = int(support_k), int(outlier_k), int(chunk_size)
    for name, value in (("support_scale", support_scale), ("graph_scale", graph_scale),
                        ("score_scale", score_scale), ("base_spacing_scale", base_spacing_scale)):
        if np.ndim(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite scalar.")
    for name, value in (("support_radius", support_radius), ("base_spacing", base_spacing)):
        if value is not None and (np.ndim(value) or not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be None or a positive finite scalar.")
    if not np.isfinite(radius_cap) or radius_cap < 1 or not 0 < density_level < 1:
        raise ValueError("radius_cap must be >=1 and density_level must be in (0,1).")
    if not np.isfinite([closing, opening]).all() or min(closing, opening) < 0:
        raise ValueError("Legacy closing/opening must be finite nonnegative metric radii.")
    if closing and closing_scale is not None or opening and opening_scale is not None:
        raise ValueError("Do not provide legacy and canonical morphology controls together.")
    if (closing or opening) and (closing_scale is not None or opening_scale is not None):
        raise ValueError("Do not mix legacy and canonical opening/closing ordering.")
    if fill_holes and hole_scale is not None:
        raise ValueError("Do not provide both fill_holes and hole_scale.")
    if max_shift_scale is not _GRID_UNSET and max_shift is not None:
        raise ValueError("Do not provide both max_shift and max_shift_scale.")
    direction = np.eye(3) if direction is None else np.asarray(direction, float)
    if (direction.shape != (3, 3) or not np.isfinite(direction).all() or
            not np.allclose(direction.T @ direction, np.eye(3), atol=1e-7, rtol=0)):
        raise ValueError("direction columns must be orthonormal.")
    axis_scale = _xyz(axis_scale, "axis_scale", True)
    scales = np.atleast_1d(np.asarray(smooth_scales, float))
    if scales.ndim != 1 or not len(scales) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("smooth_scales must be a nonempty positive finite sequence.")
    if (np.ndim(label_guard_scale) or not np.isfinite(label_guard_scale)
            or label_guard_scale < 0):
        raise ValueError("label_guard_scale must be finite and nonnegative.")
    if not isinstance(separate_protect_labels, (bool, np.bool_)):
        raise ValueError("separate_protect_labels must be boolean.")
    if label_smooth is not None and (label_smooth_weight is not None or label_axis_weights is not None):
        raise ValueError("Use label_smooth or legacy label_smooth_weight/label_axis_weights, not both.")
    label_smooth = 0.5 if label_smooth is None else label_smooth
    data = _cloud_inputs(points, label, confidence, label_confidence, feature, min_points=2)
    if _support_confidence is not None:
        _support_confidence = np.asarray(_support_confidence, dtype=float)
        if (_support_confidence.shape != (len(data["points"]),) or
                not np.isfinite(_support_confidence).all() or
                np.any((_support_confidence < 0) | (_support_confidence > 1))):
            raise ValueError("Internal support confidence must align with points and lie in [0,1].")
    raw_sites = data["unique_source"]
    metric = (data["points"][raw_sites] @ direction) / axis_scale
    kept, _, _ = _clean_cloud(metric, data["confidence"][raw_sites],
        max(support_k, outlier_k), 0.0 if outlier_thresh is None else outlier_thresh,
        0.0, workers, outlier_method=outlier_method, outlier_k=outlier_k,
        outlier_iters=outlier_iters, outlier_radius=outlier_radius,
        min_points=min_points, graph_scale=graph_scale, radius_cap=radius_cap,
        min_survivors=2, keep_largest=keep_largest)
    observations = _cloud_label_observations(data, kept,
        label_outlier_k=label_outlier_k, label_outlier_thresh=label_outlier_thresh,
        label_outlier_iters=label_outlier_iters, label_min_points=label_min_points,
        label_outlier_action=label_outlier_action, graph_scale=graph_scale, workers=workers)
    metric = metric[kept]
    geom_conf = data["confidence"][observations["unique_source"]]
    support_conf = geom_conf
    if _support_confidence is not None:
        # Repeated records at one coordinate must not inflate support. Geometry
        # and attributes still use the original confidence, independently.
        support_conf = np.zeros(len(metric), dtype=float)
        np.maximum.at(support_conf, observations["obs_site"],
                      _support_confidence[observations["observations"]])
        if np.any(support_conf <= 0):
            raise ValueError("Every retained site requires positive support confidence.")
    tree = cKDTree(metric)
    support_k = min(support_k, len(metric) - 1)
    distance = tree.query(metric, k=list(range(2, support_k + 2)), workers=workers)[0]
    estimated_step = float(np.median(distance[:, -1]) / np.cbrt(support_k))
    if not np.isfinite(estimated_step) or estimated_step <= 0:
        raise ValueError("Points must define a finite positive reference scale.")
    reference_step = estimated_step if base_spacing is None else float(base_spacing)
    step = reference_step * base_spacing_scale
    spacing = _xyz(step * axis_scale * _xyz(spacing_scale, "spacing_scale", True)
                   if spacing is None else spacing, "spacing", True)
    sigma = _xyz(step * axis_scale * _xyz(smooth_scale, "smooth_scale")
                 if smooth_sigma is None else smooth_sigma, "smooth_sigma")
    effective_sigma = sigma * np.sqrt(field_smooth_iters)
    bandwidth = step * axis_scale * _xyz(kernel_scale, "kernel_scale", True)
    if max_shift_scale is _GRID_UNSET:
        shift = float(2 * effective_sigma.max() * scales.max() if max_shift is None else max_shift)
    else:
        shift = None if max_shift_scale is None else float(step * max_shift_scale)
    if shift is not None and (not np.isfinite(shift) or shift < 0):
        raise ValueError("max_shift/max_shift_scale must be nonnegative finite or explicit None.")
    if closing or opening:
        open_scale, close_scale = 2 * opening * axis_scale / step, 2 * closing * axis_scale / step
        morph_order = ("closing", "opening")
    else:
        open_scale = _xyz(0 if opening_scale is None else opening_scale, "opening_scale")
        close_scale = _xyz(0 if closing_scale is None else closing_scale, "closing_scale")
        morph_order = ("opening", "closing")
    holes = np.inf if fill_holes else (0.0 if hole_scale is None else hole_scale)
    if np.ndim(holes) or np.isnan(holes) or holes < 0:
        raise ValueError("hole_scale must be nonnegative or np.inf.")
    if np.ndim(offset_scale) or not np.isfinite(offset_scale):
        raise ValueError("offset_scale must be a finite signed scalar.")
    # Independent class volumes preserve the globally corrected class coding.
    if data["keys"] is not None and label_mode == "separate" and not shared_separate:
        # Only spatially corroborated class observations form hard exclusions.
        # Site aggregation prevents duplicate records from acting as neighbors.
        guard_codes = guard_trusted = None
        if separate_protect_labels and label_guard_scale > 0:
            guard_codes, guard_trusted = _label_guard_sites(
                observations["points"], observations["obs_site"],
                observations["codes"], observations["weights"],
                k=label_outlier_k, graph_scale=graph_scale, workers=workers)
        results, infos = {}, {}
        for code, key in enumerate(data["keys"]):
            rows = observations["observations"][(observations["codes"] == code) & (observations["weights"] > 0)]
            if len(np.unique(data["points"][rows], axis=0)) < 2:
                empty = np.zeros((1, 1, 1), bool)
                result = (empty if output == "mask" else
                    (np.empty((0, 3)), np.empty((0, 3), np.int64)) if output == "mesh"
                    else np.empty((0, 3)))
                info = dict(mask=empty, origin=np.zeros(3), spacing=spacing.copy(),
                    direction=direction.copy(), label_mask=np.zeros_like(empty, np.int32),
                    label_score_image=np.zeros_like(empty, np.float32),
                    score_image=np.zeros_like(empty, np.float32), score=np.empty(0, np.float32),
                    label=np.empty(0, object), label_id=np.empty(0, np.int32),
                    label_values=data["keys"], label_map={0: None, **{i+1:v for i,v in enumerate(data["keys"])}})
                if return_field:
                    info.update(field=np.full(empty.shape, float(spacing.min()), np.float32),
                                field_inside="negative")
            else:
                parameters = input_parameters.copy()
                class_labels = np.empty(len(rows), dtype=object)
                class_labels.fill(key)  # A tuple category is one object, not an array axis.
                reliability = observations["label_reliability"][rows]
                parameters.update(label=class_labels, label_mode="project",
                    confidence=data["confidence"][rows], label_confidence=reliability,
                    _support_confidence=data["confidence"][rows] * reliability,
                    feature=None if data["feature"] is None else data["feature"][rows],
                    label_outlier_action="none", outlier_iters=0,
                    min_points=0, keep_largest=False, return_info=True)
                if guard_trusted is not None:
                    foreign = guard_trusted & (guard_codes != code)
                    parameters["_foreign_points"] = (
                        observations["points"][foreign], step * label_guard_scale)
                result, info = _grid_point_cloud_impl(data["points"][rows], **parameters)
                info["label_mask"][info["label_mask"] > 0] = code + 1
                if "point_label_id" in info:
                    info["point_label_id"][info["point_label_id"] > 0] = code + 1
                if "label_id" in info:
                    info["label_id"][info["label_id"] > 0] = code + 1
                info["label_values"] = data["keys"]
                info["label_map"] = {0: None, **{i+1:v for i,v in enumerate(data["keys"])}}
                info["source_indices"] = rows.copy()
            # Diagnostics always align with the ORIGINAL input, even for
            # independently constructed class grids.
            names = np.empty(len(data["keys"]), object)
            names[:] = data["keys"]
            info.update(inlier_mask=observations["inlier_mask"].copy(),
                label_corrected=names[observations["corrected_codes"]],
                label_corrected_id=observations["corrected_codes"].copy(),
                label_reliability=observations["label_reliability"].copy(),
                label_changed_mask=observations["label_changed_mask"].copy(),
                label_outlier_mask=observations["label_outlier_mask"].copy())
            results[key], infos[key] = result, info
        return (results, infos) if return_info else results

    local = np.sqrt(np.mean(distance ** 2, axis=1)) * step / estimated_step
    radii = np.minimum(local, radius_cap * np.median(local))
    if support_radius is not None:
        radii.fill(support_radius)
    if method == "nearest":
        radii.fill(float(np.median(radii)))
    radii *= _support_length_factor(metric, radii, support_k, support_smooth, workers, chunk_size)
    radii *= support_scale * np.cbrt(support_conf)
    del distance
    support_pad = 3 * bandwidth if method == "kde" else radii.max() * axis_scale
    active_sigma = effective_sigma if smooth_method is not None and shift != 0 else np.zeros(3)
    padding = (support_pad + step * (open_scale + close_scale) +
               3 * active_sigma * scales.sum() + abs(step * offset_scale) + 2 * spacing)
    if shift is not None and np.any(active_sigma):
        padding += shift
    physical = metric * axis_scale
    origin_local = np.floor((physical.min(axis=0) - padding) / spacing) * spacing
    dims = np.ceil((physical.max(axis=0) + padding - origin_local) / spacing) + 1
    size = float(np.prod(dims))
    if not np.isfinite(size) or size > max_voxels:
        raise MemoryError(f"Grid needs {size:,.0f} nodes; max_voxels={max_voxels:,}. Increase spacing or crop points.")
    shape = tuple(dims.astype(np.int64))
    mask = _grid_support(metric, support_conf, tree, radii, origin_local, spacing, shape,
                         axis_scale, method, support_k, bandwidth, density_level, chunk_size, workers)
    if not mask.any():
        raise ValueError("Empty support mask; reduce spacing or increase support.")
    forbidden = None
    if _foreign_points is not None and len(_foreign_points[0]):
        foreign_points, guard_radius = _foreign_points
        foreign_tree = cKDTree((np.asarray(foreign_points) @ direction) / axis_scale)
        forbidden = np.zeros(mask.size, bool)
        for start in range(0, mask.size, chunk_size):
            stop = min(mask.size, start + chunk_size)
            ijk = np.column_stack(np.unravel_index(np.arange(start, stop), shape))
            query = (origin_local + ijk * spacing) / axis_scale
            other = foreign_tree.query(query, k=1, workers=workers)[0]
            own = tree.query(query, k=1, workers=workers)[0]
            forbidden[start:stop] = (other <= guard_radius) & (other < own)
        forbidden = forbidden.reshape(shape)
    mask, positive_field, volume_info = _process_volume(mask, spacing, step,
        smooth_sigma=effective_sigma, smooth_scales=scales, smooth_method=smooth_method,
        density_level=0.5, preserve_volume=preserve_volume, max_shift=shift,
        opening_scale=open_scale, closing_scale=close_scale, hole_scale=holes,
        offset_scale=offset_scale, min_volume_component=min_volume_component,
        pre_min_component=pre_min_component, component_diameter_scale=component_diameter_scale,
        morph_order=morph_order, component_connectivity=6, forbidden=forbidden, device=device,
        field_smooth_mode=field_smooth_mode)
    if not mask.any():
        raise ValueError("No occupied nodes remain after morphology/smoothing/filtering.")
    field = -positive_field
    info = dict(mask=mask, origin=origin_local @ direction.T, spacing=spacing.copy(),
                direction=direction.copy(), base_spacing=step,
                base_spacing_reference=reference_step, base_spacing_scale=base_spacing_scale,
                smooth_sigma=sigma, smooth=sigma, effective_smooth_sigma=effective_sigma,
                volume_info=volume_info, inlier_mask=observations["inlier_mask"],
                points_clean=observations["points"], method=method)
    indices = None
    if output == "mesh":
        result = grid_to_mesh(dict(info, field=field, field_inside="negative"), backend=backend, device=device)
        output_points = result[0]
    elif output == "mask":
        result, output_points = mask, None
    else:
        selected = mask if output == "volume" else mask & ~ndi.binary_erosion(mask)
        indices = np.column_stack(np.nonzero(selected))
        result = info["origin"] + (indices * spacing) @ direction.T
        output_points = result
    if data["keys"] is not None and (return_info or label_mode == "joint" or shared_separate):
        if shared_separate:
            local_obs = dict(observations, points=observations["points"] @ direction,
                            observation_points=observations["observation_points"] @ direction)
            guard_codes, guard_trusted = _label_guard_sites(local_obs["points"],
                local_obs["obs_site"], local_obs["codes"], local_obs["weights"],
                k=label_outlier_k, graph_scale=graph_scale, workers=workers)
            private_shape, private_origin = _label_grid_halo(mask.shape, origin_local, spacing,
                4*step*_xyz(label_smooth, "label_smooth"), max_voxels)
            def class_fields():
                for code, key in enumerate(data["keys"]):
                    use = (observations["codes"] == code) & (observations["weights"] > 0)
                    ids = np.unique(observations["obs_site"][use])
                    if not len(ids):
                        continue
                    weights = observations["label_reliability"][observations["observations"][use]]
                    counts = np.bincount(observations["obs_site"][use], minlength=len(metric))
                    reliability = np.bincount(observations["obs_site"][use], weights=weights,
                                               minlength=len(metric)) / np.maximum(counts, 1)
                    private_conf = support_conf[ids] * reliability[ids]
                    class_tree = cKDTree(metric[ids])
                    count = max(1, min(support_k, len(ids)-1))
                    private_radii = radii[ids] * np.cbrt(reliability[ids])
                    if separate_geometry == "local" and len(ids) > 1:
                        distances = class_tree.query(metric[ids], k=list(range(2, count+2)), workers=workers)[0]
                        reference = float(np.median(distances[:, -1]) / np.cbrt(count))
                        private_radii = np.sqrt(np.mean(distances**2, axis=1))*step/reference
                        private_radii = np.minimum(private_radii, radius_cap*np.median(private_radii))
                        if support_radius is not None:
                            private_radii.fill(support_radius)
                        if method == "nearest":
                            private_radii.fill(float(np.median(private_radii)))
                        private_radii *= _support_length_factor(metric[ids], private_radii, count,
                                                               support_smooth, workers, chunk_size)
                        private_radii *= support_scale*np.cbrt(private_conf)
                    private = _grid_support(metric[ids], private_conf, class_tree, private_radii,
                        private_origin, spacing, private_shape, axis_scale, method,
                        min(support_k, len(ids)), bandwidth, density_level, chunk_size, workers)
                    private, field_i, report_i = _process_volume(private, spacing, step,
                        smooth_sigma=effective_sigma, smooth_scales=scales, smooth_method=smooth_method,
                        preserve_volume=preserve_volume, max_shift=shift,
                        opening_scale=open_scale, closing_scale=close_scale, hole_scale=holes,
                        offset_scale=offset_scale, min_volume_component=min_volume_component,
                        pre_min_component=pre_min_component, component_diameter_scale=component_diameter_scale,
                        morph_order=morph_order, component_connectivity=6, device=device,
                        field_smooth_mode=field_smooth_mode)
                    yield code, field_i, bool(private.any()), local_obs["points"][ids], report_i, private_origin, spacing
            assigned = _separate_label_volume(mask, origin_local, spacing, step,
                origin_local + np.argwhere(mask)*spacing, local_obs, class_fields,
                label_smooth=_xyz(label_smooth, "label_smooth"), label_smooth_weight=label_smooth_weight,
                label_axis_weights=label_axis_weights, label_lock_seeds=label_lock_seeds,
                site_codes=guard_codes, site_trusted=guard_trusted,
                guard_radius=step*label_guard_scale if separate_protect_labels else 0.,
                workers=workers, chunk_size=chunk_size, device=device,
                guard_mode=label_guard_mode, post_smooth=step*post_smooth,
                label_max_iter=label_max_iter, label_tol=label_tol)
            labeled = _label_array_result(assigned["label_mask"], assigned["label_score_image"],
                data["keys"], assigned["label_info"], info["origin"], spacing, direction,
                point_points=output_points if indices is None else None, point_indices=indices)
            info["separate_partition"] = "shared"
        else:
            labeled = _grid_label_data(mask, info["origin"], spacing, observations,
                direction=direction, mode=label_mode, d=step, label_smooth=label_smooth,
                k=label_project_k if label_mode == "project" else support_k,
                workers=workers, point_indices=indices,
                point_points=output_points if indices is None else None,
                label_max_iter=label_max_iter, label_tol=label_tol, label_lock_seeds=label_lock_seeds,
                label_smooth_weight=label_smooth_weight, label_axis_weights=label_axis_weights,
                chunk_size=chunk_size, label_post_smooth=post_smooth)
        if output == "mesh" and "point_label_id" in labeled:
            missing = labeled["point_label_id"] == 0
            if missing.any():
                edge = np.argwhere(mask & ~ndi.binary_erosion(mask))
                boundary_tree = cKDTree(edge * spacing)
                query = (output_points[missing] - info["origin"]) @ direction
                nearest = boundary_tree.query(query, workers=workers)[1]
                at = tuple(edge[nearest].T)
                ids = labeled["label_mask"][at]
                labeled["point_label_id"][missing] = ids
                labeled["point_label"][missing] = _category_values(data["keys"], ids - 1,
                                                                    for_vtk=False)
                labeled["point_label_score"][missing] = labeled["label_score_image"][at]
        info.update(labeled)
        info["label_values"] = data["keys"]
        for name, source in (("label", "point_label"), ("label_id", "point_label_id"),
                             ("label_confidence", "point_label_score")):
            if source in info:
                info[name] = info[source]
        names = np.empty(len(data["keys"]), object)
        names[:] = data["keys"]
        info["label_corrected"] = names[observations["corrected_codes"]]
        info["label_corrected_id"] = observations["corrected_codes"].copy()
        info["label_reliability"] = observations["label_reliability"].copy()
        info["label_changed_mask"] = observations["label_changed_mask"]
        info["label_outlier_mask"] = observations["label_outlier_mask"]
    if return_info:
        info.update(_grid_scores_features(mask, origin_local, spacing, axis_scale, metric,
            geom_conf, data, observations, step, score_k, score_scale, chunk_size, workers))
        if indices is not None:
            at = tuple(indices.T)
            info["score"] = info["score_image"][at]
            if "feature_image" in info:
                info["feature"] = info["feature_image"][at]
        elif output_points is not None:
            coordinates = ((output_points @ direction) - origin_local) / spacing
            info["score"] = ndi.map_coordinates(info["score_image"], coordinates.T,
                order=1, mode="constant", cval=0, prefilter=False)
            if "feature_image" in info:
                values = info["feature_image"]
                values = values[..., None] if values.ndim == 3 else values
                projected = [ndi.map_coordinates(values[..., i], coordinates.T, order=1,
                    mode="nearest", prefilter=False) for i in range(values.shape[-1])]
                info["feature"] = np.column_stack(projected)
                if data["feature"].ndim == 1:
                    info["feature"] = info["feature"][:, 0]
        if return_field:
            info["field"], info["field_inside"] = field, "negative"
        if shared_separate and separate_output == "per_label":
            return _split_grid_partition(info, positive_field, output, True, return_field,
                                         interface_backend, workers, chunk_size)
        return result, info
    if shared_separate and separate_output == "per_label":
        return _split_grid_partition(info, positive_field, output, False, False,
                                     interface_backend, workers, chunk_size)
    return result


def _split_grid_partition(info, positive_field, output, return_info, return_field,
                          interface_backend, workers, chunk_size):
    """Retain grid separate's dictionary return after a shared volume partition."""
    labels, keys = info["label_mask"], info["label_values"]
    origin, spacing, direction = info["origin"], info["spacing"], info["direction"]
    meshes = None
    if output == "mesh":
        meshes, _ = _mesh_from_labels(labels, positive_field, origin @ direction,
            spacing, keys, mesh_smooth_iters=0, interface_smooth_iters=0,
            interface_backend=interface_backend, d=info["base_spacing"])
    results, infos = {}, {}
    for code, key in enumerate(keys, 1):
        mask = labels == code
        current = dict(info)
        for name in ("point_label_id", "point_label", "point_label_score", "label_id",
                     "label", "label_confidence", "score", "feature", "field"):
            current.pop(name, None)
        current.update(mask=mask, label_mask=np.where(mask, code, 0).astype(np.int32),
            label_score_image=np.where(mask, info["label_score_image"], 0).astype(np.float32))
        indices, query = None, None
        if output == "mesh":
            mesh = meshes[key]
            query = np.asarray(mesh.points) @ direction.T
            faces = mesh.faces.reshape(-1, 4)[:, 1:].copy()
            if np.linalg.det(direction) < 0:
                faces = faces[:, [0, 2, 1]]
            result = (query, faces)
            for name in ("global_point_id", "global_face_id"):
                source = mesh.point_data if name == "global_point_id" else mesh.cell_data
                if name in source:
                    current[name] = np.asarray(source[name]).copy()
        elif output == "mask":
            result = mask
            indices = np.argwhere(mask)
        else:
            selected = mask if output == "volume" else mask & ~ndi.binary_erosion(mask)
            indices = np.argwhere(selected)
            result = origin + (indices * spacing) @ direction.T
        if return_info:
            count = len(query) if query is not None else len(indices)
            ids = np.full(count, code, np.int32)
            names = _category_values(keys, ids - 1, for_vtk=False)
            current.update(label_id=ids, point_label_id=ids.copy(), label=names, point_label=names.copy())
            arrays = {"score_image": "score", "label_score_image": "label_confidence"}
            if "feature_image" in info:
                arrays["feature_image"] = "feature"
            for image_name, point_name in arrays.items():
                image = info[image_name]
                select = mask.reshape(mask.shape + (1,)*(image.ndim-3))
                current[image_name] = np.where(select, image, 0)
                if indices is not None:
                    values = image[tuple(indices.T)]
                else:
                    coordinates = (((query - origin) @ direction) / spacing).T
                    if image.ndim == 3:
                        values = ndi.map_coordinates(image, coordinates, order=1,
                            mode="constant", cval=0, prefilter=False)
                    else:
                        values = np.column_stack([ndi.map_coordinates(image[..., j], coordinates,
                            order=1, mode="nearest", prefilter=False) for j in range(image.shape[-1])])
                current[point_name] = values
            current["point_label_score"] = current["label_confidence"].copy()
            if return_field:
                current.update(field=_sdf(mask, spacing).astype(np.float32), field_inside="negative")
            infos[key] = current
        results[key] = result
    return (results, infos) if return_info else results


def _site_features(data, observations):
    """Geometric-confidence mean at each retained site; sparse features stay sparse."""
    feature = data["feature"]
    if feature is None:
        return None
    rows, sites = observations["observations"], observations["obs_site"]
    weight = data["confidence"][rows]
    count = len(observations["unique_source"])
    if len(rows) == count:
        return feature[rows[np.argsort(sites)]]
    mass = np.bincount(sites, weights=weight, minlength=count)
    average = sparse.csr_matrix((weight / mass[sites], (sites, np.arange(len(rows)))),
                                shape=(count, len(rows)))
    return average @ feature[rows]


def _grid_scores_features(mask, origin, spacing, axis_scale, points, confidence,
                          data, observations, d, k, scale, chunk, workers):
    """Batch score/attribute interpolation; duplicate attributes are averaged."""
    feature = data["feature"]
    averaged = None
    if feature is not None:
        averaged = _site_features(data, observations)
        if sparse.issparse(averaged):
            averaged = averaged.tocsr()
        elif averaged.ndim == 1:
            averaged = averaged[:, None]
    score_image = np.zeros(mask.shape, np.float32)
    values = None if averaged is None else np.zeros(mask.shape + (averaged.shape[1],), np.float32)
    tree = cKDTree(points)
    flat = np.flatnonzero(mask)
    kk = min(int(k), len(points))
    for start in range(0, len(flat), chunk):
        rows = flat[start:start + chunk]
        ijk = np.column_stack(np.unravel_index(rows, mask.shape))
        query = (origin + ijk * spacing) / axis_scale
        distance, near = tree.query(query, k=list(range(1, kk + 1)), workers=workers)
        proximity = np.exp(-0.5 * (distance / (d * scale)) ** 2)
        evidence = proximity * confidence[near]
        score_image.ravel()[rows] = evidence.mean(axis=1)
        if averaged is not None:
            # Stabilize only relative interpolation weights, not absolute scores.
            logw = -0.5 * (distance / (d * scale)) ** 2 + np.log(confidence[near])
            logw -= logw.max(axis=1, keepdims=True)
            weight = np.exp(logw)
            weight /= weight.sum(axis=1, keepdims=True)
            chosen = averaged[near.ravel()]
            if sparse.issparse(chosen):
                chosen = chosen.toarray()
            chosen = np.asarray(chosen).reshape(len(rows), kk, -1)
            values.reshape(-1, values.shape[-1])[rows] = np.einsum("nk,nkf->nf", weight, chosen)
    info = dict(score_image=score_image, score=score_image[mask])
    if values is not None:
        if feature.ndim == 1:
            values = values[..., 0]
        info.update(feature_image=values, feature=values[mask])
    return info


grid_point_cloud.__doc__ = _grid_point_cloud_impl.__doc__


def _project_zero_surface(points, field, origin, spacing):
    """Bounded Newton projection with a residual test and per-point backtracking."""
    p = np.asarray(points, float).copy()
    gradient = np.gradient(np.asarray(field, float), *spacing)
    tolerance = 1e-4 * float(np.min(spacing))
    for _ in range(10):
        coord = ((p - origin) / spacing).T
        value = ndi.map_coordinates(field, coord, order=1, mode='nearest', prefilter=False)
        active = np.abs(value) > tolerance
        if not active.any():
            break
        g = np.column_stack([ndi.map_coordinates(a, coord, order=1,
            mode='nearest', prefilter=False) for a in gradient])
        norm2 = np.einsum('ij,ij->i', g, g)
        active &= norm2 > 1e-16
        delta = value[:, None] * g / np.maximum(norm2[:, None], 1e-16)
        delta *= np.minimum(1.0, 1.0 / np.maximum(
            np.linalg.norm(delta / spacing, axis=1), 1e-30))[:, None]
        delta[~active] = 0
        pending = np.flatnonzero(active)
        improved = 0
        for _ in range(8):
            if not len(pending):
                break
            candidate = p[pending] - delta[pending]
            candidate_value = ndi.map_coordinates(field,
                ((candidate - origin) / spacing).T, order=1, mode='nearest', prefilter=False)
            accept = np.abs(candidate_value) < np.abs(value[pending])
            p[pending[accept]] = candidate[accept]
            improved += int(accept.sum())
            pending = pending[~accept]
            delta[pending] *= 0.5
        if not improved:
            break
    return p


def _bounded_surface_move(reference, candidate, triangles, spacing, *, fixed=None):
    """Limit local folds, then shrink the whole update if local checks keep failing.

    Local reductions can invalidate adjacent faces, so a fixed number of local
    sweeps alone need not converge on a large connected surface. A common step
    reduction approaches the valid reference without moving fixed vertices.
    These checks prevent local flips/collapse, not distant intersections.
    """
    reference = np.asarray(reference, float)
    candidate = np.asarray(candidate, float)
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise _SurfaceUpdateError("Surface coordinates must be finite.")
    triangles = np.asarray(triangles, np.int64)
    delta = candidate - reference
    if fixed is not None:
        delta[np.asarray(fixed, bool)] = 0
    if not np.isfinite(delta).all():
        raise _SurfaceUpdateError("Surface displacement overflowed; rescale coordinates.")
    normal = np.cross(reference[triangles[:, 1]] - reference[triangles[:, 0]],
                      reference[triangles[:, 2]] - reference[triangles[:, 0]])
    area = np.linalg.norm(normal, axis=1)
    epsilon = 1e-12 * float(np.min(spacing)) ** 2
    if not np.isfinite(area).all() or np.any(area <= epsilon):
        raise _SurfaceUpdateError("Extracted label surface contains invalid or degenerate triangles.")
    normal /= area[:, None]
    moving = np.any(delta != 0, axis=1)
    factor = np.ones(len(reference))
    local_steps, global_steps = 32, 64
    for attempt in range(local_steps + global_steps + 1):
        result = reference + delta * factor[:, None]
        updated = np.cross(result[triangles[:, 1]] - result[triangles[:, 0]],
                           result[triangles[:, 2]] - result[triangles[:, 0]])
        invalid = (~np.isfinite(updated).all(axis=1) |
                   (np.linalg.norm(updated, axis=1) <= np.maximum(epsilon, 0.01 * area)) |
                   (np.einsum('ij,ij->i', normal, updated) <= 0.01 * area))
        if not invalid.any():
            return result, int(np.count_nonzero(moving & (factor < 1.0)))
        if attempt < local_steps:
            factor[np.unique(triangles[invalid])] *= 0.5
        else:
            factor *= 0.5
    raise _SurfaceUpdateError("Cannot accept shared-surface update without local folds/collapse; "
                       "reduce smoothing or refine the label grid.")


def _assign_component_observations(parts, points, spacing, d, workers=-1,
                                   chunk_size=50_000):
    """Assign each observation to one retained occupied part before cropping."""
    points = np.asarray(points, dtype=float)
    spacing = np.broadcast_to(np.asarray(spacing, dtype=float), (3,))
    if (points.ndim != 2 or points.shape[1] != 3 or
            not np.isfinite(points).all() or np.any(spacing <= 0) or
            not np.isfinite(spacing).all() or not np.isfinite(d) or d <= 0):
        raise ValueError("Invalid observation coordinates, spacing or base spacing.")
    if (not np.isscalar(chunk_size) or not np.isfinite(chunk_size) or
            chunk_size < 1 or int(chunk_size) != chunk_size):
        raise ValueError("chunk_size must be a positive integer.")
    chunk_size = int(chunk_size)
    owner = np.full(len(points), -1, dtype=np.int32)
    node = np.full(len(points), -1, dtype=np.int64)
    projected = np.zeros(len(points), dtype=bool)
    reach = max(float(d), float(np.linalg.norm(spacing)))
    domains = []
    for component, part in enumerate(parts):
        mask, origin = np.asarray(part[1], bool), np.asarray(part[2], float)
        if mask.ndim != 3 or origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("Each occupied part needs a 3-D mask and finite XYZ origin.")
        domains.append((mask, origin))
        if not mask.any():
            continue
        for start in range(0, len(points), chunk_size):
            ids = np.arange(start, min(start + chunk_size, len(points)))
            ids = ids[owner[ids] < 0]
            if not len(ids):
                continue
            # Half-up rounding is invariant to integer lattice translations;
            # bankers rounding can choose a different seed after cropping.
            ijk_float = np.floor((points[ids] - origin) / spacing + 0.5)
            inside = ((ijk_float >= 0) & (ijk_float < np.asarray(mask.shape))).all(axis=1)
            ids, ijk = ids[inside], ijk_float[inside].astype(np.int64)
            flat = np.ravel_multi_index(ijk.T, mask.shape)
            keep = mask.ravel()[flat]
            owner[ids[keep]], node[ids[keep]] = component, flat[keep]

    pending = np.flatnonzero(owner < 0)
    nearest_distance = np.full(len(points), np.inf)
    # All directly occupied observations were assigned above. Never steal one
    # for a different cropped part simply because its padded box is nearby.
    for component, (mask, origin) in enumerate(domains):
        if not len(pending) or not mask.any():
            continue
        upper = origin + (np.asarray(mask.shape) - 1) * spacing
        candidate = pending[((points[pending] >= origin - reach) &
                             (points[pending] <= upper + reach)).all(axis=1)]
        if not len(candidate):
            continue
        # An off-mask query's closest lattice node is on the foreground
        # boundary. Using only boundary nodes keeps exact physical distances.
        boundary = mask & ~ndi.binary_erosion(mask, border_value=0)
        active = np.flatnonzero(boundary)
        xyz = origin + np.column_stack(np.unravel_index(active, mask.shape)) * spacing
        tree = cKDTree(xyz)
        query_k = min(2, len(active))
        for start in range(0, len(candidate), chunk_size):
            ids = candidate[start:start + chunk_size]
            distances, nodes = tree.query(points[ids], k=query_k,
                distance_upper_bound=np.nextafter(reach, np.inf), workers=workers)
            if query_k == 1:
                best, nearest = np.asarray(distances), np.asarray(nodes)
            else:
                best, nearest = distances[:, 0], nodes[:, 0].copy()
                # KD-tree tie ordering is unspecified; keep deterministic
                # lowest flat index for exact equidistant lattice nodes.
                ties = np.flatnonzero(np.isfinite(best) & (distances[:, 1] == best))
                for at in ties:
                    candidates = tree.query_ball_point(points[ids[at]],
                        np.nextafter(best[at], np.inf))
                    candidates = np.asarray(candidates, dtype=np.int64)
                    distance2 = np.sum((xyz[candidates] - points[ids[at]]) ** 2, axis=1)
                    nearest[at] = candidates[distance2 == distance2.min()].min()
            take = np.isfinite(best) & (best <= reach) & (best < nearest_distance[ids])
            chosen = ids[take]
            owner[chosen] = component
            node[chosen] = active[nearest[take]]
            nearest_distance[chosen] = best[take]
            projected[chosen] = True
    return owner, node, projected, {
        "label_owned_observations": int(np.count_nonzero(owner >= 0)),
        "label_projected_points": int(projected.sum()),
        "label_unused_points": int(np.count_nonzero(owner < 0)),
        "label_projection_reach": float(reach),
    }


def _label_guard_sites(points, obs_site, codes, weights, *, k=24,
                       graph_scale=2.0, workers=-1):
    """Return dominant IDs and reliable, locally supported morphology guards."""
    x = np.asarray(points, dtype=np.float64)
    site, code = np.asarray(obs_site), np.asarray(codes)
    weight = np.asarray(weights, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError("Guard points must have finite shape (N, 3).")
    n = len(x)
    if (site.ndim != 1 or code.shape != site.shape or weight.shape != site.shape
            or not np.issubdtype(site.dtype, np.integer)
            or not np.issubdtype(code.dtype, np.integer)
            or np.any((site < 0) | (site >= n)) or np.any(code < 0)
            or not np.isfinite(weight).all() or np.any(weight < 0)):
        raise ValueError("Guard observations need valid site IDs, label IDs and nonnegative weights.")
    if np.ndim(k) or not np.isfinite(k) or k < 1 or int(k) != k:
        raise ValueError("Guard k must be a positive integer.")
    if np.ndim(graph_scale) or not np.isfinite(graph_scale) or graph_scale <= 0:
        raise ValueError("Guard graph_scale must be finite and positive.")
    dominant = np.full(n, -1, dtype=np.int64)
    trusted = np.zeros(n, dtype=bool)
    if not len(site):
        return dominant, trusted
    classes, dense = np.unique(code, return_inverse=True)
    evidence = sparse.coo_matrix((weight, (site, dense)),
                                 shape=(n, len(classes))).tocsr()
    evidence.eliminate_zeros()
    total = np.asarray(evidence.sum(axis=1)).ravel()
    maximum = evidence.max(axis=1).toarray().ravel()
    selected = np.asarray(evidence.argmax(axis=1)).ravel()
    positive = total > 0
    dominant[positive] = classes[selected[positive]]
    fraction = np.divide(maximum, total, out=np.zeros(n), where=positive)
    count = np.bincount(site, minlength=n)
    candidate = (fraction >= .8) & (total / np.maximum(count, 1) >= .5)
    if n < 3 or int(k) < 2 or not candidate.any():
        return dominant, trusted

    # Sorted coordinates give the tree a stable ordering even if site IDs change.
    order = np.lexsort(x.T[::-1])
    nk = min(int(k), n - 1)
    distance, neighbors = cKDTree(x[order]).query(
        x[order], k=nk + 1, workers=workers)
    distance, neighbors = distance[:, 1:], order[neighbors[:, 1:]]
    if np.any(distance[:, 0] <= 0):
        raise ValueError("Guard points must contain distinct geometry coordinates.")
    local = np.empty(n)
    local[order] = np.minimum(np.median(distance[:, :min(6, nk)], axis=1),
                              2.0 * distance[:, 0])
    limit = graph_scale * np.minimum(local[order, None], local[neighbors])
    supported = ((distance <= limit) & candidate[neighbors]
                 & (dominant[neighbors] == dominant[order, None]))
    trusted[order] = candidate[order] & (supported.sum(axis=1) >= 2)
    return dominant, trusted


def _mc_orient_faces(vertices, faces, field, spacing):
    """Make shared-edge winding consistent, then orient each surface into background."""
    n = len(faces)
    edges = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    owners = np.repeat(np.arange(n, dtype=np.int64), 3)
    forward = edges[:, 0] < edges[:, 1]
    ordered = np.sort(edges, axis=1)
    order = np.lexsort((ordered[:, 1], ordered[:, 0]))
    ordered = ordered[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.any(ordered[1:] != ordered[:-1], axis=1))]
    counts = np.diff(np.r_[starts, len(ordered)])
    paired = starts[counts == 2]
    left, right = order[paired], order[paired + 1]
    a, b = owners[left], owners[right]
    # Equal directed edges require opposite face flips; opposite edges require
    # the same flip. Nonmanifold edges have no unique two-face relation.
    opposite = forward[left] == forward[right]
    graph = sparse.csr_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(n, n))
    _, components = connected_components(graph, directed=False)
    if np.any(opposite):
        from collections import deque
        relations = sparse.csr_matrix(
            (np.r_[np.where(opposite, -1, 1), np.where(opposite, -1, 1)],
             (np.r_[a, b], np.r_[b, a])), shape=(n, n))
        flips = np.zeros(n, dtype=np.int8)
        for root in np.flatnonzero(np.bincount(components) > 0):
            candidates = np.flatnonzero(components == root)
            if not np.any(opposite[components[a] == root]):
                flips[candidates] = 1
                continue
            first = int(candidates[0])
            flips[first] = 1
            queue = deque([first])
            while queue:
                i = queue.popleft()
                sl = slice(relations.indptr[i], relations.indptr[i + 1])
                for j, relation in zip(relations.indices[sl], relations.data[sl]):
                    expected = flips[i] * int(np.sign(relation))
                    if flips[j] == 0:
                        flips[j] = expected
                        queue.append(int(j))
                    elif flips[j] != expected:
                        raise RuntimeError("Marching cubes produced inconsistent, non-orientable face adjacency.")
        faces = faces.copy()
        reverse = flips < 0
        faces[reverse] = faces[reverse][:, [0, 2, 1]]
    if np.any(counts > 2):
        warnings.warn(
            f"Marching cubes produced {int(np.count_nonzero(counts > 2))} nonmanifold edges; "
            "winding is determined on manifold-connected surface patches.",
            RuntimeWarning, stacklevel=2)

    # Sample separately in each surface component; a large outer shell cannot
    # outvote the orientation of an inner cavity or a small detached component.
    grouped = np.argsort(components, kind="stable")
    sections = np.split(grouped, np.flatnonzero(np.diff(components[grouped])) + 1)
    relative_step = spacing / np.max(spacing)
    faces = faces.copy()
    for ids in sections:
        chosen = ids[np.linspace(0, len(ids) - 1, min(4096, len(ids)), dtype=np.int64)]
        triangles = vertices[faces[chosen]]
        normals = np.cross((triangles[:, 1] - triangles[:, 0]) * relative_step,
                           (triangles[:, 2] - triangles[:, 0]) * relative_step)
        area = np.linalg.norm(normals, axis=1)
        unit = normals / area[:, None]
        ray = unit / relative_step
        ray /= np.max(np.abs(ray), axis=1)[:, None]
        center = triangles.mean(axis=1)
        score = 0.0
        magnitude = 0.0
        # The second radius is used only when the first gives numerically
        # ambiguous evidence, rather than creating a full gradient volume.
        for radius in (0.25, 0.5):
            plus = ndi.map_coordinates(field, (center + radius * ray).T,
                                       order=1, mode="nearest", prefilter=False)
            minus = ndi.map_coordinates(field, (center - radius * ray).T,
                                        order=1, mode="nearest", prefilter=False)
            contribution = (plus.astype(float) - minus.astype(float)) * (area / area.max())
            score = float(contribution.sum())
            magnitude = float(np.abs(contribution).sum())
            if magnitude > 0 and abs(score) > 64 * np.finfo(float).eps * magnitude:
                break
        if magnitude == 0 or abs(score) <= 64 * np.finfo(float).eps * magnitude:
            raise RuntimeError(
                f"Cannot orient marching-cubes surface component with {len(ids)} triangles: "
                "field samples on its two sides are numerically indistinguishable.")
        if score > 0:
            faces[ids] = faces[ids][:, [0, 2, 1]]
    return faces


def _grid_coordinates(points, origin, spacing, shape):
    """Validate node coordinates, snapping roundoff alone at the grid boundary."""
    points = np.asarray(points, dtype=float)
    origin, spacing = np.asarray(origin, dtype=float), np.asarray(spacing, dtype=float)
    shape = np.asarray(shape, dtype=np.int64)
    if points.ndim != 2 or shape.shape != (points.shape[1],) or np.any(shape < 1):
        raise ValueError("Raster coordinates and grid shape have incompatible dimensions.")
    origin = np.broadcast_to(origin, shape.shape)
    spacing = np.broadcast_to(spacing, shape.shape)
    if (not np.isfinite(points).all() or not np.isfinite(origin).all()
            or not np.isfinite(spacing).all() or np.any(spacing <= 0)):
        raise ValueError("Raster points/origin must be finite and spacing must be positive.")
    coordinates = (points - origin) / spacing
    scale = np.maximum(1.0, np.maximum(np.abs(origin) / spacing, shape - 1))
    tolerance = np.minimum(32 * np.finfo(float).eps * scale, 1e-6)
    outside = (coordinates < -tolerance) | (coordinates > shape - 1 + tolerance)
    if np.any(outside):
        row, axis = np.argwhere(outside)[0]
        raise ValueError(
            f"{int(np.count_nonzero(np.any(outside, axis=1)))} raster points lie outside the grid; "
            f"first is row {row}, axis {axis}, index {coordinates[row, axis]:.9g}, "
            f"valid range [0, {shape[axis] - 1}]. Increase grid coverage rather than clipping points.")
    return np.clip(coordinates, 0, shape - 1)

# Previous private entry points remain available for compatibility.
_separate_image_labels = _separate_label_volume
_image_seed_locks = _label_seed_locks
_smooth_image_partition = _smooth_label_partition
