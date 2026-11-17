"""Occupied-volume reconstruction with project, separate and joint labels.

This complete standalone module preserves clay_mesh, clay_slice_mesh,
points_to_image, image_to_nrrd, grid_point_cloud and grid_to_mesh.
All four reconstruction functions support project/separate/joint labels.
Shared adapters keep geometry and class evidence independent.
points_to_image adds int32 label_mask without replacing mask or score_image.
clay_mesh retains both legacy keyword controls and the later compact controls.
Read its function docstring for current defaults, units, returns and aliases.

Required: NumPy, SciPy >=1.6. Mesh output: PyVista/VTK.
Optional MC engines: PyMCubes, scikit-image, torchmcubes; auto falls back to VTK.
Joint extraction uses Surface Nets or shared voxel faces, not independent MC.
Optional torch enables CUDA Gaussian filtering/torchmcubes, not CPU KD-trees,
EDT, morphology or label graph solves. CPU is the default.

Geometry outliers are filtered globally. Label outliers have a separate,
conservative relabel/downweight/disable control and never delete common support.
Tiny voxel components can be removed BEFORE closing. Cavities use 26-connected
background; guarded separate reconstruction avoids filling foreign tissue.
The continuous field reaches joint outer-surface projection. Shared smoothing
and outer-shell synchronization happen before splitting; mesh_smooth_* acts.

Points must sample occupied VOLUME, rather than surface scans alone. No
coordinate-only method distinguishes all missing measurements from real cavities.
A corrected label score is not a calibrated biological probability. Geometry
and label cleanup can remove rare true structures; controls can be disabled.
Independent separate meshes may overlap or leave gaps. Shared joint interfaces
have identical coordinates but arbitrary inputs can still yield nonmanifold or
self-intersecting meshes. Do not independently smooth/decimate split joint meshes.
STL stores geometry; VTP also stores labels, confidence, features and shared IDs.
"""

from pathlib import Path
import warnings
import math
import gzip

import numpy as np
from scipy import ndimage as ndi, sparse
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

_GRID_UNSET = object()

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
    """Extract a positive-inside XYZ field with outward right-hand winding."""
    field = np.asarray(field, dtype=np.float32)
    origin = np.broadcast_to(np.asarray(origin, dtype=float), (3,))
    spacing = np.broadcast_to(np.asarray(spacing, dtype=float), (3,))
    if field.ndim != 3 or min(field.shape) < 2 or not np.isfinite(field).all():
        raise ValueError("marching cubes requires a finite 3D field, shape >=2 on every axis.")
    if not np.isfinite(origin).all() or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("origin must be finite and spacing finite and positive.")
    if not (field.min() < 0 < field.max()):
        raise ValueError("The field must contain both negative and positive values.")
    if backend not in {"auto", "pymcubes", "skimage", "torchmcubes", "lewiner", "lorensen", "vtk"}:
        raise ValueError("Unknown marching-cubes backend.")
    if backend == "auto":
        try:
            import mcubes
        except ImportError:
            try:
                import skimage.measure
            except ImportError:
                backend = "vtk"
            else:
                backend = "skimage"
        else:
            backend = "pymcubes"
    if backend == "pymcubes":
        import mcubes
        vertices, faces = mcubes.marching_cubes(np.ascontiguousarray(field), 0.0)
    elif backend in {"skimage", "lewiner", "lorensen"}:
        from skimage.measure import marching_cubes
        vertices, faces, _, _ = marching_cubes(
            field, level=0.0, gradient_direction="descent", allow_degenerate=False,
            method="lorensen" if backend == "lorensen" else "lewiner"
        )
    elif backend == "vtk":
        # Keep the compact version's NumPy/SciPy/PyVista-only installation usable.
        import pyvista as pv
        grid = pv.ImageData(dimensions=field.shape)
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
    vertices = np.asarray(vertices, dtype=np.float64) * spacing + origin
    faces = np.asarray(faces, dtype=np.int64)
    if not len(faces):
        raise ValueError("Marching cubes produced no triangles.")
    normal = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                      vertices[faces[:, 2]] - vertices[faces[:, 0]])
    # Sample a bounded number of faces, avoiding three full gradient volumes.
    ids = np.arange(0, len(faces), max(1, len(faces) // 10000))
    ijk = ((vertices[faces[ids]].mean(axis=1) - origin) / spacing).T
    gradient = np.empty((len(ids), 3))
    for a in range(3):
        delta = np.zeros((3, 1))
        delta[a] = 0.5
        gradient[:, a] = (ndi.map_coordinates(field, ijk + delta, order=1,
            mode="nearest", prefilter=False) - ndi.map_coordinates(field,
            ijk - delta, order=1, mode="nearest", prefilter=False)) / spacing[a]
    if np.einsum("ij,ij->", normal[ids], gradient) > 0:
        faces = faces[:, [0, 2, 1]]
    nonzero = np.any(normal != 0, axis=1)
    if not nonzero.all():
        # Exact zero-valued grid nodes can produce duplicate vertices and
        # collapsed triangles in classic marching cubes.
        vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
        faces = inverse[faces[nonzero]]
        used, inverse = np.unique(faces, return_inverse=True)
        vertices, faces = vertices[used], inverse.reshape(-1, 3)
    return vertices, faces



def _stack_field(points, confidence, groups, levels, known_empty, axis,
                 origin, step, shape, d, sigma, radius, outlier, max_gap, method, *, workers=-1):
    """Shared physical-z interpolation; observations are occupancy or 2D SDF."""
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
                ij = np.rint((points[ids][:, plane] - origin[plane]) / step[plane]).astype(int)
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
        ijk = np.rint((points - origin) / step).astype(int)
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
              backend="auto", device="cpu", max_voxels=2_000_000_000,
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
              interface_max_shift_scale=None, score_k=8, score_scale=2.0,
              chunk_size=100_000, workers=-1, return_outer=False, return_info=False,
              X=None, voxel_size=None, smooth=None, outlier=None, radius=None,
              smooth_iters=None):
    """Reconstruct exact planar stacks, with a general 3-D fallback.

    Existing ``sdf``/``density`` section interpolation, arbitrary slice_axis,
    max_gap segmentation, missing-label section caps, auto grid coarsening,
    CUDA Gaussian filters and every legacy alias are retained. Nonplanar points
    use general 3-D fields rather than flattening transformed coordinates.

    ``project`` returns one outer mesh with labels. ``separate`` independently
    reconstructs each corrected label, skipping labels with fewer than 8 sites;
    meshes can overlap or leave gaps. ``joint`` reconstructs one occupied field,
    then partitions it and extracts shared interfaces. With return_outer=True,
    joint/separate return (dict, outer); project still returns one mesh.
    return_info=True wraps any result as (result, info).

    Confidence weights geometry; label_confidence and label_outlier_* affect
    labels only. Label cleanup precedes separate support/morphology. Contradictory
    duplicate observations remain available to joint diffusion and projection.
    Dense/sparse feature attributes, source_id, score and label_confidence are
    mapped to output vertices. return_info retains original-row cleanup arrays.

    Legacy defaults remain: spacing_scale=.6, smooth_scale=(1,1,1.5), support_scale=2,
    outlier_thresh=3, outlier_k=16, mesh_smooth_iters=10. ``outlier_method='slice'``
    applies simultaneous in-plane MAD plus cross-section support filtering;
    outlier_iters repeats it. Other methods are 'lof','ratio','sor','radius',None
    and run on global XYZ. None/0 iterations/0 threshold disables point filtering.
    support_k independently estimates the in-plane/reference scale; None uses
    outlier_k. min_points/graph_scale remove global small point components.

    Intrinsic physical sigma is smooth_scale times sampling spacing (slice-axis
    uses the observed median interval), or explicit smooth_sigma. Multiplying by
    sqrt(field_smooth_iters) controls intrinsic smoothing; 0 disables it. This
    scalar is applied once, not again to post-processing stages.

    smooth_scales=None preserves legacy intrinsic smoothing with NO extra field
    stages. Supplying e.g. (1,.5) adds shared SDF stages with the same physical
    sigma. preserve_volume/max_shift_scale constrain ONLY these explicit post
    stages, not intrinsic section interpolation or later mesh smoothing.
    Opening/closing/hole/offset scales use reference length d; opening/closing
    are XYZ DIAMETERS, hole_scale is maximum cavity equivalent-sphere diameter.
    All default to 0. pre_min_component and min_volume_component are component
    fractions; min_component remains the legacy post-component alias (default 0).
    component_diameter_scale adds an absolute occupied-component cutoff equal to
    the volume of a sphere of diameter d*value; 0 disables this cutoff. True
    max_gap/known-empty section caps remain enforced after post-processing.

    separate_protect_labels prevents filling protected reliable foreign-label
    sites (radius=d*label_guard_scale) during morphology. It cannot guarantee
    globally matching independently generated surfaces; joint provides them.
    mesh_smooth_iters and mesh_smooth_pass_band affect actual output geometry.
    Optional interface settings add internal-only smoothing/caps in joint.
    support_scale accepts a scalar or XYZ: section support uses the two in-plane
    axes, while its slice-axis extent follows the observed section caps. General
    3-D support uses all three axes. Density mode retains its deposition model.
    Smaller spacing increases resolution; lower pass band increases smoothing.
    """
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

    def volume(ids, code=None):
        points_i, weight_i = p[ids], w[ids]
        step = requested_step.copy()
        if axis is not None and 64*len(levels) > max_voxels:
            raise MemoryError("max_voxels cannot hold the observed-section arrays, even after coarsening.")
        extra = .5*d*(opening_scale+closing_scale) + abs(d*offset_scale)
        extra += 3*post_sigma*stages.sum()
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
        if code is not None and separate_protect_labels and label_guard_scale > 0:
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
            initial_field=field,
            device=device)
        if cap is not None:
            np.minimum(field, cap, out=field)
            mask = field > 0
        if not (field.min() < 0 < field.max()):
            raise ValueError("No surface remains; reduce smoothing/opening or adjust support_scale.")
        metadata.update(spacing=step[None], voxel_size=step[None], retained_points=[len(ids)],
            grid_shape=np.asarray(shape)[None])
        if reference is not None:
            metadata.update(density_reference=[reference*weight_i.max()],
                density_threshold=[density_level*reference*weight_i.max()])
        return mask, field, origin, step, metadata

    def annotate(mesh, ids, metadata, constant=None):
        for name, value in {**metadata, **shared_meta}.items():
            if np.asarray(value).size:
                mesh.field_data[name] = np.atleast_1d(value)
        if not mesh.n_points:
            return mesh
        _annotate_clay_mesh(mesh, p[ids], source_ids[ids], raw, geom, data["feature"],
                           d, score_k, score_scale, chunk_size, workers)
        if label_present and label_mode != "joint":
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
        return annotate(mesh, ids, metadata, constant=code)

    all_ids = np.arange(len(p))
    if not label_present or label_mode == "project":
        result = outer_mesh(all_ids)
    elif label_mode == "separate":
        evidence = sparse.coo_matrix((obs_weight,(obs_site,obs_codes)),
                                     shape=(len(p),len(keys))).tocsr()
        totals = np.asarray(evidence.sum(axis=1)).ravel()
        dominant = np.asarray(evidence.argmax(axis=1)).ravel()
        maximum = evidence.max(axis=1).toarray().ravel()
        dominance = np.divide(maximum,totals,out=np.zeros_like(totals),where=totals>0)
        counts = np.bincount(obs_site,minlength=len(p))
        trusted = (dominance >= .8) & (totals/np.maximum(counts,1) >= .5)
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
        labels, assignment, label_meta = _label_volume(mask, origin, step, obs_points,
            obs_codes, obs_weight, d, label_smooth, support_k, workers,
            label_max_iter=label_max_iter, label_tol=label_tol,
            label_lock_seeds=label_lock_seeds, label_smooth_weight=label_smooth_weight,
            label_axis_weights=label_axis_weights, chunk_size=chunk_size)
        meshes, outer = _mesh_from_labels(labels, field, origin, step, keys,
            mesh_smooth_iters, mesh_smooth_pass_band, interface_backend=interface_backend,
            interface_smooth_iters=interface_smooth_iters,
            interface_max_shift_scale=interface_max_shift_scale, d=d)
        metadata.update(label_meta)
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



def _geometry(points, confidence, k, outlier, radius, anisotropy, chunk,
              reference_spacing=None, *, outlier_method="ratio", outlier_iters=1,
              outlier_radius=None, min_points=0.0, graph_scale=2.0,
              adaptive=True, radius_cap=2.0, support_k=None, workers=-1,
              base_spacing_scale=1.0):
    """Batched point filtering and volume-normalized local PCA ellipsoids.

    ``retained`` always indexes the input rows. Outlier distances and component
    connections use physical Euclidean XYZ distances; ``radius`` stretches the
    final supports along WORLD XYZ, after local PCA. ``axes`` contains their
    true principal semiaxes, for conservative sphere queries in _support_mask.
    """
    points, confidence = np.asarray(points, float), np.asarray(confidence, float)
    retained = np.arange(len(points))
    support_k = k if support_k is None else int(support_k)
    radius = np.broadcast_to(np.asarray(radius, float), (3,))
    chunk = max(1, min(int(chunk), 1_000_000 // max(k, support_k)))
    enabled = outlier_method is not None and outlier is not None and outlier > 0
    if outlier_method not in {None, "ratio", "lof", "sor", "radius"}:
        raise ValueError("outlier_method must be ratio, lof, sor, radius or None.")

    tree = rk = None
    query_k = None
    for _ in range(int(outlier_iters) if enabled else 0):
        n = len(points)
        nk = min(k, n - 1)
        tree = cKDTree(points)
        rk = tree.query(points, k=[nk + 1], workers=workers)[0][:, 0]
        query_k = nk
        if not np.isfinite(rk).all() or np.any(rk <= 0):
            raise ValueError("Point distances must be finite and positive.")
        keep = np.empty(n, bool)
        if outlier_method in {"lof", "sor"}:
            average = np.empty(n)
            for start in range(0, n, chunk):
                sl = slice(start, start + chunk)
                dd, ids = tree.query(points[sl], k=nk + 1, workers=workers)
                dd, ids = dd[:, 1:], ids[:, 1:]
                average[sl] = (np.maximum(dd, rk[ids]).mean(1)
                               if outlier_method == "lof" else dd.mean(1))
            if outlier_method == "sor":
                med = np.median(average)
                mad = 1.4826 * np.median(np.abs(average - med))
                floor = 64 * np.finfo(float).eps * max(float(med), np.finfo(float).tiny)
                if mad <= floor:
                    mad = max(float(np.std(average)), floor)
                keep = average <= med + outlier * mad
        if outlier_method in {"ratio", "lof"}:
            for start in range(0, n, chunk):
                sl = slice(start, start + chunk)
                ids = tree.query(points[sl], k=nk + 1, workers=workers)[1][:, 1:]
                # LOF = mean(lrd_j/lrd_i); use reachability means directly to
                # avoid reciprocals of very small physical distances.
                score = (rk[sl] / np.median(rk[ids], axis=1)
                         if outlier_method == "ratio" else
                         (average[sl, None] / average[ids]).mean(1))
                keep[sl] = score <= outlier
        elif outlier_method == "radius":
            limit = outlier * np.median(rk) if outlier_radius is None else outlier_radius
            keep = rk <= limit
        if keep.all():
            break
        retained = retained[keep]
        points, confidence = points[keep], confidence[keep]
        tree = rk = None
        if len(points) < 4:
            raise ValueError("Fewer than four points remain after outlier removal.")

    if min_points > 0:
        tree = cKDTree(points)
        nk = min(max(k, support_k), len(points) - 1)
        local = np.empty(len(points))
        for start in range(0, len(points), chunk):
            sl = slice(start, start + chunk)
            dd = tree.query(points[sl], k=min(6, nk) + 1, workers=workers)[0][:, 1:]
            local[sl] = np.median(dd, axis=1)
        local = np.minimum(local, radius_cap * np.median(local))
        rows, cols = [], []
        for start in range(0, len(points), chunk):
            sl = slice(start, start + chunk)
            dd, ids = tree.query(points[sl], k=nk + 1, workers=workers)
            dd, ids = dd[:, 1:], ids[:, 1:]
            edge = dd <= graph_scale * np.minimum(local[sl, None], local[ids])
            r, c = np.nonzero(edge)
            rows.append(r + start)
            cols.append(ids[r, c])
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        graph = sparse.csr_matrix((np.ones(len(rows), np.uint8), (rows, cols)),
                                  shape=(len(points), len(points)))
        _, groups = sparse.csgraph.connected_components(graph, directed=False)
        sizes = np.bincount(groups)
        cutoff = min_points * sizes.max() if min_points < 1 else min_points
        keep = sizes[groups] >= cutoff
        retained = retained[keep]
        points, confidence = points[keep], confidence[keep]
        tree = rk = None
        del rows, cols, graph, local, groups, sizes
        if len(points) < 4:
            raise ValueError("Fewer than four points remain; decrease min_points.")

    # Estimate the reconstruction scale only from the final retained cloud.
    if tree is None:
        tree = cKDTree(points)
    nk = min(support_k, len(points) - 1)
    if rk is None or query_k != nk:
        rk = tree.query(points, k=[nk + 1], workers=workers)[0][:, 0]
    d = float(np.median(rk) / np.cbrt(nk) if reference_spacing is None else reference_spacing)
    d *= base_spacing_scale
    if not np.isfinite(rk).all() or np.any(rk <= 0) or not np.isfinite(d) or d <= 0:
        raise ValueError("Point distances must give a finite positive sampling scale.")
    precision = np.empty((len(points), 3, 3), np.float64)
    axes = np.empty((len(points), 3), np.float64)
    eye = np.eye(3)
    ratio_floor = max((1.0 / anisotropy) ** 2, np.finfo(float).eps)
    shrink = min(0.2, ratio_floor)
    uniform = bool(np.all(radius == radius[0]))
    for start in range(0, len(points), chunk):
        sl = slice(start, start + chunk)
        if adaptive or anisotropy != 1:
            dist, ids = tree.query(points[sl], k=nk + 1, workers=workers)
            dist, ids = dist[:, 1:], ids[:, 1:]
        local = (base_spacing_scale * np.median(rk[ids], axis=1) / np.cbrt(nk)
                 if adaptive else np.full(len(points[sl]), d))
        r = np.clip(local, d / radius_cap, radius_cap * d) * np.cbrt(confidence[sl])
        if anisotropy == 1:
            lengths = r[:, None] * radius
            axes[sl] = lengths
            precision[sl] = eye / lengths[:, :, None] ** 2
            continue
        # Dimensionless covariance avoids squaring physical length units.
        delta = (points[ids] - points[sl, None, :]) / rk[sl, None, None]
        weights = confidence[ids] / confidence[ids].max(axis=1, keepdims=True)
        weights *= np.exp(-0.5 * (dist / rk[sl, None]) ** 2)
        weights /= weights.sum(axis=1, keepdims=True)
        delta -= np.einsum("nk,nkj->nj", weights, delta)[:, None, :]
        cov = np.einsum("nk,nki,nkj->nij", weights, delta, delta)
        cov = ((1 - shrink) * cov + (shrink / 3) *
               np.trace(cov, axis1=1, axis2=2)[:, None, None] * eye)
        eigen, basis = np.linalg.eigh(cov)
        eigen = np.maximum(eigen, eigen[:, -1:] * ratio_floor)
        relative = np.sqrt(eigen / np.exp(np.log(eigen).mean(axis=1))[:, None])
        if uniform:
            lengths = (radius[0] * r)[:, None] * relative
            axes[sl] = lengths
            precision[sl] = (basis / lengths[:, None, :] ** 2) @ basis.transpose(0, 2, 1)
        else:
            # A = D_radius B D_relative maps a unit ball to the scaled support.
            # Its singular values are true semiaxes; Q=A^-T A^-1 / r^2.
            transform = radius[None, :, None] * basis * relative[:, None, :]
            axes[sl] = np.linalg.svd(transform, compute_uv=False) * r[:, None]
            inverse = (basis.transpose(0, 2, 1) / radius[None, None, :]
                       / relative[:, :, None] / r[:, None, None])
            precision[sl] = inverse.transpose(0, 2, 1) @ inverse
    if (not np.isfinite(precision).all() or not np.isfinite(axes).all()
            or np.any(axes <= 0)):
        raise ValueError("Support geometry is outside floating-point range; rescale coordinates or confidence.")
    return points, confidence, retained, precision, axes, d




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
    """Exact ellipsoid union at grid nodes, with bounded batches of candidates.

    A fixed Euclidean kNN shortlist can miss a larger/elongated ellipsoid.
    Increase the shortlist until a containing ellipsoid is found or every
    center within the largest semiaxis has been checked.
    """
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
    return mask.reshape(tuple(shape))




def _reconstruct(points, precision, axes, d, smooth, gap_size, voxel_size,
                 min_component, smooth_iters, backend, device, max_voxels, chunk,
                 voxel_scale=0.5, method="sdf", density_level=0.5,
                 smooth_sigma=None, *, smooth_scales=(1.0,), preserve_volume=True,
                 max_shift_scale=2.0, closing_scale=0.0, opening_scale=0.0,
                 hole_scale=0.0, offset_scale=0.0, mesh_smooth_pass_band=0.1,
                 workers=-1, return_volumes=False, return_fields=False,
                 pre_min_component=None, protect_points=None, protect_radius=0.0,
                 component_diameter_scale=0.0):
    """Blocked reconstruction using the shared volume-processing pipeline.

    pre_min_component=None inherits min_component; 0 disables prefiltering.
    Thresholds use the largest component across ALL blocks, before and after
    morphology. A value of 1 retains one largest component, including ties.
    protect_points/protect_radius describe reliable foreign-label support cores
    in the same coordinates and physical units as points. They are optional;
    protection excludes these cores throughout reconstruction and vetoes whole
    cavity filling. Separate mode remains independent and may still overlap away
    from the protected cores. return_fields implies return_volumes and appends a
    cropped continuous physical field to each (volume, mask, origin) tuple.
    component_diameter_scale additionally removes final components smaller than
    a physical sphere of diameter d*component_diameter_scale, across all blocks.
    """
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
    kernel_margin = 4 * sigma * scales.sum() if active else np.zeros(3)
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
            pre_min_component=0, forbidden=forbidden, device=device)
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





def _joint_mesh(parts, metadata, points, codes, confidence, d, n_labels, *,
                label_smooth_weight, label_axis_weights, label_lock_seeds,
                label_max_iter, label_tol, interface_backend, interface_smooth_iters,
                interface_max_shift_scale, chunk, workers):
    """Partition the retained volume components, then assemble shared interfaces."""
    import pyvista as pv

    step = np.asarray(metadata["spacing"])[0]
    projection = max(float(np.linalg.norm(step)), 0.5 * d)
    # Assign each source to ONE final occupied component. Cropped boxes can
    # overlap around cavities; passing every source to every box duplicates seeds.
    owner = np.full(len(points), -1, np.int32)
    distance = np.full(len(points), np.inf)
    for component, (_, mask, origin) in enumerate(parts):
        upper = origin + (np.asarray(mask.shape) - 1) * step
        candidate = np.flatnonzero(((points >= origin - projection) &
                                    (points <= upper + projection)).all(axis=1))
        if not len(candidate):
            continue
        ijk = np.rint((points[candidate] - origin) / step).astype(np.int64)
        ijk = np.clip(ijk, 0, np.asarray(mask.shape) - 1)
        outside = ~mask[tuple(ijk.T)]
        dist = np.linalg.norm(origin + ijk * step - points[candidate], axis=1)
        if outside.any():
            # Exact bounded projection for off-grid observations. An EDT lookup
            # at the rounded grid index need not be nearest to the actual point.
            targets = points[candidate[outside]]
            tree = cKDTree(targets)
            occupied = np.flatnonzero(mask)
            nodes = []
            for start in range(0, len(occupied), chunk):
                indices = np.column_stack(np.unravel_index(occupied[start:start + chunk], mask.shape))
                coords = origin + indices * step
                near = tree.query(coords, distance_upper_bound=np.nextafter(projection, np.inf),
                                  workers=workers)[0]
                if np.isfinite(near).any():
                    nodes.append(coords[np.isfinite(near)])
            dist[outside] = (cKDTree(np.concatenate(nodes)).query(targets, workers=workers)[0]
                             if nodes else np.inf)
        keep = (dist <= projection) & (dist < distance[candidate])
        selected = candidate[keep]
        owner[selected], distance[selected] = component, dist[keep]
    unused = int(np.count_nonzero(owner < 0))
    if unused:
        warnings.warn(f"joint excluded {unused} retained label observation(s) outside the final "
                      "occupied domain and its bounded seed projection distance.", stacklevel=2)

    vertices, faces, pairs, scores, offsets = [], [], [], [], 0
    voxel_counts = np.zeros(n_labels, np.int64)
    statistics = {key: 0 for key in ("label_seed_voxels", "label_hard_seed_voxels",
                  "label_conflict_voxels", "label_projected_points", "label_unused_points")}
    statistics["label_unused_points"] = unused
    max_iterations, max_residual, used_backends = 0, 0.0, set()
    for component, (_, mask, origin) in enumerate(parts):
        ids = np.flatnonzero(owner == component)
        if not len(ids):
            raise ValueError(f"Occupied component {component} has no nearby retained label observations; "
                             "reduce field smoothing/opening or use finer spacing.")
        labels, info = _joint_partition(mask, origin, step, points[ids], codes[ids],
            confidence[ids], d, label_smooth_weight=label_smooth_weight,
            label_axis_weights=label_axis_weights, label_lock_seeds=label_lock_seeds,
            label_max_iter=label_max_iter, label_tol=label_tol, chunk=chunk, workers=workers)
        if not np.array_equal(labels > 0, mask):
            raise RuntimeError("Joint partition must cover exactly the occupied domain.")
        voxel_counts += np.bincount(labels.ravel(), minlength=n_labels + 1)[1:]
        surface = _joint_surfaces(labels, origin, step, interface_backend=interface_backend,
            interface_smooth_iters=interface_smooth_iters,
            interface_max_shift_scale=interface_max_shift_scale)
        if not surface.n_cells:
            raise RuntimeError("Joint interface extraction returned an empty surface.")
        vertices.append(np.asarray(surface.points).copy())
        faces.append(np.asarray(surface.faces).reshape(-1, 4)[:, 1:] + offsets)
        pairs.append(np.asarray(surface.cell_data["BoundaryLabels"]).copy())
        offsets += surface.n_points
        # Normalize interpolation by occupied support, so exterior padding does
        # not halve assignment confidence on the outer surface.
        coordinates = ((surface.points - origin) / step).T
        support = ndi.map_coordinates(mask.astype(np.float32), coordinates, order=1,
                                      mode="constant", cval=0, prefilter=False)
        score = ndi.map_coordinates(info["label_confidence"], coordinates, order=1,
                                   mode="constant", cval=0, prefilter=False)
        scores.append(np.divide(score, support, out=np.zeros_like(score), where=support > 1e-8))
        for key in statistics:
            statistics[key] += info[key]
        max_iterations = max(max_iterations, max(info["label_iterations"], default=0))
        max_residual = max(max_residual, max(info["label_residuals"], default=0.0))
        used_backends.add(str(surface.field_data["interface_backend"][0]))
    faces = np.concatenate(faces)
    mesh = pv.PolyData(np.concatenate(vertices), np.c_[np.full(len(faces), 3), faces])
    mesh.cell_data["BoundaryLabels"] = np.concatenate(pairs)
    mesh.cell_data["global_face_id"] = np.arange(mesh.n_cells, dtype=np.int64)
    mesh.point_data["global_point_id"] = np.arange(mesh.n_points, dtype=np.int64)
    mesh.point_data["label_confidence"] = np.clip(np.concatenate(scores), 0, 1)
    for key, value in metadata.items():
        mesh.field_data[key] = value
    for key, value in statistics.items():
        mesh.field_data[key] = [value]
    mesh.field_data["label_mode"] = ["joint"]
    mesh.field_data["label_voxel_counts"] = voxel_counts
    mesh.field_data["label_volumes"] = voxel_counts * np.prod(step)
    mesh.field_data["label_smooth_weight"] = [label_smooth_weight]
    mesh.field_data["label_axis_weights"] = np.asarray(label_axis_weights)[None]
    mesh.field_data["label_lock_seeds"] = [int(label_lock_seeds)]
    mesh.field_data["label_iterations"] = [max_iterations]
    mesh.field_data["label_residual"] = [max_residual]
    mesh.field_data["label_converged"] = [1]
    mesh.field_data["interface_backend"] = sorted(used_backends)
    mesh.field_data["interface_smooth_iters"] = [interface_smooth_iters]
    mesh.field_data["interface_max_shift"] = [interface_max_shift_scale * step.min()]
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
    interface_max_shift_scale=None, score_k=8, score_scale=2.0,
    backend="auto", device="cpu", return_outer=False, return_info=False,
    max_voxels=2_000_000_000, chunk_size=100_000, workers=-1,
    pc=None, radius=None, smooth=None, voxel_size=None, voxel_scale=None,
    reference_spacing=None, outlier=None, smooth_iters=None,
):
    """Reconstruct labelled occupied volumes without discarding earlier controls.

    Parameters
    ----------
    points, label, confidence
        XYZ array/tensor, PyVista dataset or readable point-cloud path; aligned
        categories and geometric reliability in [0,1]. Named label/confidence
        arrays can reference point_data. The third positional argument keeps
        the compact API's confidence meaning; pass attributes as feature=.
        Nonfinite coordinates and zero-confidence observations are excluded.
        Duplicates share geometric support but retain their label evidence.
    feature : dense/sparse (N,) or (N,C), or point_data name, optional
        Finite real attributes interpolated from retained original observations.
        Attributes never modify support. Sparse interpolation uses W @ feature.
    label_mode : {'joint', 'project', 'separate'}, default 'joint'
        joint: one global occupied domain, soft joint partition, shared internal
        interfaces and exact common outer shell. Returns {label: PolyData}.
        project: one global mesh, corrected labels projected onto its exterior;
        no internal interfaces or independent tissue volumes. Returns PolyData.
        separate: independently reconstruct each corrected label's support.
        Returns {label: PolyData}; gaps/overlaps may remain. Label guard protects
        reliable foreign observations but does not make interfaces shared.
        Without labels, all modes produce one whole-domain PolyData.
    k, support_k : int >=3, optional
        Synonyms for geometry neighbours, default24; supplying both raises.
        Independent outlier_k defaults to the resolved geometry neighbour count.
    outlier_method : {'lof','ratio','sor','radius',None}, default 'lof'
        Geometry filter on ALL labels together before any morphology. LOF is
        reachability-density ratio; ratio compares kth distances to neighbours;
        SOR uses robust mean-neighbour distance deviation with an SD fallback
        for degenerate MAD; radius requires outlier_k other points in range.
    outlier_thresh, outlier_k, outlier_iters, outlier_radius
        Default1.5, geometry k, 1, None. Higher positive cutoff rejects fewer;
        LOF/ratio require >1. Threshold0, iterations0 or methodNone disables.
        Radius uses explicit physical radius or cutoff*median(kth distance).
    min_points, graph_scale
        Global point-component filtering before support. min_points=None follows
        min_component; 0 disables; (0,1) relative to largest; >=1 absolute count.
        Distance-limited kNN edges use graph_scale*min(capped local lengths).
    support_scale, adaptive, radius_cap, anisotropy
        Defaults1.5,True,1.6,1. Support XYZ stretches world axes. Adaptive radius
        clips measured local spacing to [.6*d,radius_cap*d], then confidence**1/3.
        adaptive=False uses d. anisotropy>1 enables volume-normalized local PCA;
        it caps PCA axis ratio BEFORE world XYZ stretching. Geometry is shared
        across labels in joint/project and separate_geometry='shared'.
    base_spacing, base_spacing_scale
        Fixed reference length or estimate, then multiplier(default1).
        Effective d controls dimensionless settings. Explicit physical overrides
        retain their coordinate-unit values. Input coordinates never rescale.
    spacing, spacing_scale
        Explicit positive scalar/XYZ grid spacing, or d*spacing_scale(default.4).
        Resolution changes do not change physical Gaussian sigma. Grid limits
        raise instead of silently coarsening. Disconnected regions use blocks.
    smooth_scale, smooth_sigma, smooth_scales
        Base Gaussian sigma=d*smooth_scale(default1), or explicit physical sigma.
        Stage multipliers(default(1,)); (1.5,1,.5) means three successive stages.
        Final continuous field is retained for subvoxel exterior fitting.
    field_smooth_iters
        Nonnegative Gaussian diffusion time, default1. Each stage sigma is
        multiplied by sqrt(field_smooth_iters); zero disables field smoothing.
        The old smooth_iters alias still controls mesh_smooth_iters.
    min_volume_component, component_diameter_scale
        min_volume_component is the canonical alias for min_component; specify
        only one. Physical component cutoff additionally uses sphere volume
        pi/6*(d*component_diameter_scale)**3 after smoothing.
    method, density_level
        'sdf'(default) or 'density': smooth signed distance or binary occupancy.
        density_level(default.5) sets occupancy level if preserve_volume=False.
        Density is not a raw point-count KDE.
    preserve_volume, max_shift_scale
        DefaultTrue,None. Preserve the occupied volume AFTER morphology and
        cavity filling, before field smoothing. Equal-valued ties remain intact.
        Optional continuous field clamp phi0 +/- d*max_shift_scale restricts
        sign changes, not final mesh displacement. Zero disables field smoothing.
        Neither option prevents later offsets, component removal or mesh movement.
    opening_scale, closing_scale, gap_size
        Opening removes thin protrusions/bridges; closing joins narrow gaps.
        Scales(default0) are ellipsoid DIAMETERS times d; zero axis does not mix
        along it. Physical gap_size overrides closing_scale, including zero.
    hole_scale
        Default0. Fill enclosed 26-connected background components with physical
        volume <=pi/6*(d*hole_scale)**3; inf fills all eligible enclosed cavities.
        Exterior-connected and diagonally escaping voids are not filled.
        In guarded separate mode a cavity containing foreign tissue is excluded
        AS A WHOLE. A label island is not the same thing as an empty cavity.
    offset_scale
        Default0; positive expands and negative erodes the post-smoothing mask
        by d*abs(value). Explicit offset re-distances and has voxel uncertainty.
    min_component, pre_min_component
        Volume-component cutoffs relative to global largest occupied component.
        min_component(default.01) applies after smoothing. pre_min_component=None
        inherits it and filters BEFORE morphology, so closing cannot rescue a
        small noise island. Set pre_min_component=0 to retain old post-only order.
        Components use six-connected foreground; value1 retains one largest.
    label_confidence
        Optional independent [0,1] label reliability. Does NOT shrink geometric
        support of the COMMON domain or remove its points. It multiplies label
        evidence; in separate mode it also scales each class's support radius
        by the cube root of mean label reliability at that site. Named arrays
        and tensors accepted.
    label_outlier_action, label_outlier_k, label_outlier_thresh,
    label_outlier_iters, label_min_points
        Defaults'relabel',24,.8,1,3. On a distance-bounded graph, only tiny
        same-label components are candidates. Correct only with >=threshold
        alternative neighbour support, >=.25 lead and >=3 supporting distinct
        sites. Components smaller than label_min_points are eligible (>=1 count;
        (0,1) fraction of that class). Thin connected layers are protected.
        action'downweight' keeps IDs but reduces label evidence; 'none' disables.
        Zero iterations/min_points also disable. Never deletes geometric points.
        Isolated real biological cells cannot be distinguished from mislabels
        without extra evidence; action='none' preserves all supplied labels.
    label_smooth, label_smooth_weight, label_axis_weights
        label_smooth(default.5) is physical regularization length/d, scalar/XYZ.
        Solve (diag(a)+sum((d*label_smooth/h)**2 L_axis))P=aQ in occupied voxels.
        Legacy weight/axis controls are retained and use their original normalized
        graph conductances when supplied. Do not combine with label_smooth.
    label_lock_seeds, label_max_iter, label_tol
        DefaultFalse,1000,1e-5. Optional hard lock for nonconflicting reliable
        corrected observations. Contradictory duplicate evidence stays soft.
        Iterative solve reports nonconvergence instead of returning bad labels.
    label_project_k
        Default1; neighbours for projecting corrected labels in project mode.
        More neighbours perform confidence-weighted voting at the surface.
    separate_geometry : {'shared','local'}, default 'shared'
        shared retains globally estimated per-point supports/d and does not
        repeat cleaning per tissue. local re-estimates/filters per label like
        the older independent method; it can remove sparse boundary tissue.
    separate_protect_labels, label_guard_scale
        DefaultTrue,.6. Guard reliable foreign sites with balls of radius d*scale
        during separate support/morphology/smoothing/offset. Hole filling skips
        their entire cavities. False/0 restores unrestricted separate morphology.
        Guards preserve observed cores, not complete true class volume/topology.
    mesh_smooth_iters, mesh_smooth_pass_band
        Default40,.05. Actual windowed-sinc/Taubin smoothing of returned meshes.
        Zero iterations disables; otherwise >=2. Lower pass band is stronger;
        iterations set filter degree and are not a monotonic strength control.
        In joint, exterior is smoothed without internal-face pull, then its
        coordinates/normals are synchronized before splitting label meshes.
    interface_backend, interface_smooth_iters, interface_max_shift_scale
        Joint only. 'auto'/'surface_nets'/'voxel_faces'; auto falls back if needed.
        Optional extra internal shared relaxation(defaultNone) AFTER the mesh
        smoothing stage; optional internal displacement limit in units of d.
        These never disable or clamp the whole exterior's mesh smoothing.
    score_k, score_scale
        Defaults8,2. Geometric score=mean(confidence*exp(-metric_distance**2/2)),
        using bandwidth d*score_scale. score_k=0 omits score. Geometry unchanged.
    backend, device
        MC backend for unlabelled/project/separate: auto,pymcubes,skimage,lewiner,
        lorensen,torchmcubes,vtk. Joint uses interface_backend. device controls CPU
        or optional torch CUDA Gaussian/torchmcubes; KD-trees/EDT remain CPU.
    return_outer, return_info
        Joint return_outer=True -> (dict, exact_common_outer).
        Separate return_outer=True -> (dict, global_reference_outer), which is
        NOT the exact union of independently reconstructed labels.
        Project/unlabelled always return PolyData. return_info=True wraps ANY
        result as (result, info); info has original-row inlier_mask,
        label_corrected, label_reliability and label_outlier_mask.
    max_voxels, chunk_size, workers
        Defaults20million,100000,-1. Per-block grid cap, bounded query batch,
        KD-tree workers. Voxel cap is not a total-memory promise.
    pc, radius, smooth, voxel_size, voxel_scale, reference_spacing, outlier,
    smooth_iters
        Legacy aliases retained; conflicting canonical/alias arguments raise.

    Returns preserve source_id, label/label_id, feature, score and field scales.
    Joint adds label_confidence, shared global_point_id/global_face_id and
    BoundaryLabels. STL discards these arrays; VTP retains them. Do not smooth
    split joint meshes independently. Render the returned common outer mesh for
    the whole organ: merging closed tissues duplicates opposing internal faces.
    """
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
    c, lc = data["confidence"], data["label_confidence"]
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

    source, unique_source = data["source"], data["unique_source"]
    offset = np.median(x[unique_source], axis=0)
    geometry = dict(k=k, outlier_method=outlier_method, outlier_thresh=outlier_thresh,
        outlier_k=outlier_k, outlier_iters=outlier_iters, outlier_radius=outlier_radius,
        min_points=min_points, min_component=min_component, graph_scale=graph_scale,
        radius_cap=radius_cap, base_spacing=base_spacing, base_spacing_scale=base_spacing_scale,
        adaptive=adaptive, support_scale=support_scale, anisotropy=anisotropy,
        chunk_size=chunk_size, workers=workers)
    retained, d, precision, axes, radii = _prepare_geometry(
        x[unique_source] - offset, c[unique_source], **geometry)
    observation_data = _cloud_label_observations(data, retained, offset=offset,
        label_outlier_action=label_outlier_action, label_outlier_k=label_outlier_k,
        label_outlier_thresh=label_outlier_thresh, label_outlier_iters=label_outlier_iters,
        label_min_points=label_min_points, graph_scale=graph_scale, workers=workers)
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
        component_diameter_scale=[component_diameter_scale])
    reconstruction = dict(smooth_scales=stages, preserve_volume=preserve_volume,
        max_shift_scale=max_shift_scale, closing_scale=closing_scale,
        opening_scale=opening_scale, hole_scale=hole_scale, offset_scale=offset_scale,
        mesh_smooth_pass_band=mesh_smooth_pass_band, workers=workers,
        pre_min_component=pre_min_component, component_diameter_scale=component_diameter_scale)

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
        _annotate_clay_mesh(mesh, p[ids], unique_source[ids], x, c, feature,
            local_d, score_k, score_scale, chunk_size, workers)
        if label is not None and constant is None and label_mode != "joint":
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

    all_ids = np.arange(len(p))
    if label is None or label_mode == "project":
        result = annotate(reconstruct(all_ids), all_ids)
    elif label_mode == "separate":
        # Only uncontested, reliable foreign sites become morphology guards.
        evidence = sparse.coo_matrix((observed_weight, (obs_site, observed_codes)),
                                      shape=(len(p), len(keys))).tocsr()
        totals = np.asarray(evidence.sum(axis=1)).ravel()
        dominant = np.asarray(evidence.argmax(axis=1)).ravel()
        maximum = evidence.max(axis=1).toarray().ravel()
        dominance = np.divide(maximum, totals, out=np.zeros_like(maximum), where=totals > 0)
        counts = np.bincount(obs_site, minlength=len(p))
        mean_reliability = totals / np.maximum(counts, 1)
        trusted = (dominance >= .8) & (mean_reliability >= .5)
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
        group_meshes = {key: [] for key in keys}
        outer_meshes = []
        point_offset = face_offset = 0
        total_counts = np.zeros(len(keys), np.int64)
        iterations, locked_voxels, residual = 0, 0, 0.0
        seed_voxels = conflict_voxels = projected_points = 0
        for _, mask, origin, field in parts:
            step = np.asarray(metadata["spacing"]).reshape(3)
            labels, assignment, label_meta = _label_volume(mask, origin, step,
                x[observations] - offset, observed_codes, observed_weight, d,
                label_smooth, k, workers, label_max_iter=int(label_max_iter),
                label_tol=label_tol, label_lock_seeds=label_lock_seeds,
                label_smooth_weight=label_smooth_weight, label_axis_weights=label_axis_weights,
                chunk_size=chunk_size)
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
                interface_max_shift_scale=interface_max_shift_scale, d=d)
            models = [m for m in [outer, *meshes.values()] if m.n_points]
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
        common_meta.update(label_iterations=[iterations], label_voxel_counts=total_counts,
            label_volumes=total_counts * np.prod(np.asarray(metadata["spacing"]).reshape(3)),
            label_smooth=(d * label_smooth)[None], label_residual=[residual],
            label_locked_voxels=[locked_voxels], label_hard_seed_voxels=[locked_voxels],
            label_seed_voxels=[seed_voxels], label_conflict_voxels=[conflict_voxels],
            label_projected_points=[projected_points], label_converged=[True],
            label_graph_weights=np.asarray(label_meta["label_graph_weights"])[None])
        if label_smooth_weight is not None or label_axis_weights is not None:
            common_meta["label_smooth_weight"] = [1.0 if label_smooth_weight is None else label_smooth_weight]
            common_meta["label_axis_weights"] = np.atleast_2d(np.ones(3) if label_axis_weights is None else label_axis_weights)
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



def _geometry_3(x, support, adaptive, k, cap, outlier_k, threshold,
              iterations, min_points, chunk, workers, base_spacing=None):
    """Batched fixed-k LOF, then components of a bounded spatial kNN graph."""
    for _ in range(iterations if threshold is not None else 0):
        tree = cKDTree(x)
        nk = min(outlier_k, len(x) - 1)
        kd = tree.query(x, k=[nk + 1], workers=workers)[0][:, 0]
        lrd = np.empty(len(x))
        for start in range(0, len(x), chunk):
            sl = slice(start, start + chunk)
            dd, ids = tree.query(x[sl], k=nk + 1, workers=workers)
            reach = np.maximum(dd[:, 1:], kd[ids[:, 1:]])
            lrd[sl] = 1.0 / np.maximum(reach.mean(1), np.finfo(float).tiny)
        keep = np.empty(len(x), bool)
        for start in range(0, len(x), chunk):
            sl = slice(start, start + chunk)
            ids = tree.query(x[sl], k=nk + 1, workers=workers)[1][:, 1:]
            keep[sl] = lrd[ids].mean(1) / lrd[sl] <= threshold
        if keep.all():
            break
        x = x[keep]
        if len(x) < 2:
            raise ValueError("Fewer than two distinct points remain after outlier removal.")

    # Re-estimate scale after component filtering, never from discarded outliers.
    for stage in range(2):
        nk = min(k, len(x) - 1)
        rk = cKDTree(x).query(x, k=[nk + 1], workers=workers)[0][:, 0]
        median = float(np.median(rk))
        s0 = median / np.cbrt(nk) if base_spacing is None else float(base_spacing)
        if not np.isfinite(s0) or s0 <= 0:
            raise ValueError("Point coordinates do not define a finite positive scale.")
        radii = np.clip(rk / median, 1.0 / cap, cap) if adaptive else np.ones(len(x))
        if stage == 1 or min_points == 0:
            return x, s0, radii
        metric = (x - x.min(0)) / (s0 * support)
        tree = cKDTree(metric)
        nk = min(max(k, outlier_k), len(x) - 1)
        rows, cols = [], []
        for start in range(0, len(x), chunk):
            stop = min(start + chunk, len(x))
            dd, ids = tree.query(metric[start:stop], k=nk + 1, workers=workers)
            dd, ids = dd[:, 1:], ids[:, 1:]
            edge = dd <= radii[start:stop, None] + radii[ids]
            r, c = np.nonzero(edge)
            rows.append(r + start)
            cols.append(ids[r, c])
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        graph = csr_matrix((np.ones(len(rows), np.uint8), (rows, cols)),
                           shape=(len(x), len(x)))
        _, ids = connected_components(graph, directed=False)
        sizes = np.bincount(ids)
        cutoff = min_points * sizes.max() if min_points < 1 else min_points
        keep = sizes[ids] >= cutoff
        if keep.all():
            return x, s0, radii
        x = x[keep]
        if len(x) < 2:
            raise ValueError("Fewer than two points remain; reduce min_points.")



def _support_mask_3(x, radii, axes, origin, spacing, shape, chunk, workers):
    """Exact union of variable-radius ellipsoids at queried voxel centers.

    Common XYZ axis lengths are 'axes'; each point multiplies them by radii[i].
    Adaptive candidate expansion avoids missing a farther, larger ellipsoid.
    No full grid coordinate array or grid-by-point distance matrix is allocated.
    """
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



def _smooth_mask_3(mask, spacing, sigma, method, iterations, level, shift, preserve):
    """Diffuse a continuous field, then threshold once in a fixed SDF band."""
    if not mask.any() or not np.any(sigma) or iterations == 0 or shift == 0:
        return mask
    phi = None
    if method == "sdf" or shift is not None:
        phi = ndi.distance_transform_edt(~mask, sampling=spacing).astype(np.float32)
        phi -= ndi.distance_transform_edt(mask, sampling=spacing)
    band = np.ones(mask.shape, bool) if shift is None else np.abs(phi) <= shift
    target = int(np.count_nonzero(mask[band]))
    if not band.any():
        return mask
    # Gaussian variances add. One effective kernel avoids quantizing the shape
    # between passes and reduces EDT/filter work; no iterative binary masks.
    voxel_sigma = sigma * math.sqrt(iterations) / spacing
    if method == "sdf":
        field = ndi.gaussian_filter(phi, voxel_sigma, mode="nearest", truncate=3.0)
        cut = 0.0
    else:
        field = -ndi.gaussian_filter(mask.astype(np.float32), voxel_sigma,
                                     mode="constant", cval=0.0, truncate=3.0)
        cut = -level
    del phi
    inclusive = False
    if preserve:
        values = field[band]
        if target == 0:
            cut = float(values.min())
        elif target == len(values):
            cut, inclusive = float(values.max()), True
        else:
            values.partition((target - 1, target))
            cut = float(values[target - 1])
            inclusive = cut < values[target]
            if not inclusive:
                # Do not split a tied plateau by arbitrary voxel index. Choose
                # inclusion/exclusion with the smaller volume error.
                below = np.count_nonzero(values < cut)
                through = np.count_nonzero(values <= cut)
                inclusive = through - target <= target - below
        del values
    inside = field <= cut if inclusive else field < cut
    return np.where(band, inside, mask)



def points_to_image(
    points, *, spacing_scale=0.5, spacing=None, base_spacing=None,
    base_spacing_scale=1.0, method="sdf", support_scale=1.5, adaptive=True,
    support_k=8, radius_cap=2.0, smooth_scale=1.0, smooth_sigma=None,
    smooth_iters=None, preserve_volume=True, max_shift_scale=2.0,
    density_level=0.5, closing_scale=0.0, opening_scale=0.0,
    hole_scale=0.0, offset_scale=0.0, outlier_k=24, outlier_thresh=1.8,
    outlier_iters=1, min_points=0.01, min_voxels=0, component_scale=None,
    keep_largest=False, score_k=8, score_scale=2.0, padding=2, bounds=None,
    max_voxels=2_000_000_000, chunk_size=100_000, workers=-1,
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
    return_info=False, device="cpu",
):
    """Reconstruct a scored XYZ image, optionally with a tissue segmentation.

    The unlabelled result keeps exactly six keys: uint8 binary ``mask``,
    float32 ``score_image``, physical ``spacing``/``origin``, occupied voxel
    center ``points``, and corresponding float32 ``score``. Points are in
    np.argwhere(mask) order and score_image[mask.astype(bool)] equals score.

    Methods remain sdf/density/support/occupancy. sdf and density filter the
    ellipsoid union's distance/occupancy field; support skips field smoothing;
    occupancy marks observed voxels and skips support interpolation/smoothing.
    Opening, closing, offset, bounded cavity fill and mask cleanup apply to all.
    Support geometry now uses the same cleaned adaptive/PCA model as clay_mesh;
    anisotropy=1 is axis aligned. Numeric support settings can differ from the
    older image-only radius estimator after this explicit geometry unification.

    smooth_iters retains its IMAGE meaning and aliases field_smooth_iters:
    effective Gaussian sigma is sigma*sqrt(iters), default iters=1. It does not
    control mesh smoothing. Explicit smooth_scales add successive re-embedding
    stages. Physical smooth_sigma overrides d*smooth_scale. Zero time or an
    all-zero sigma disables filtering; support/occupancy always skip it.

    Morphology order preserves this image API: opening -> closing -> field
    smoothing -> offset -> hole filling -> final 26-connected components.
    opening_scale/closing_scale are XYZ DIAMETERS*d; hole_scale bounds cavity
    volume by an equivalent sphere diameter*d (0 off, inf all enclosed).
    offset_scale is signed physical displacement/d. Hole background uses
    26-connectivity and excludes exterior-connected tunnels.

    min_points filters point components; min_volume_component is a relative
    mask-volume cutoff. min_voxels is absolute. component_diameter_scale is a
    physical sphere-diameter cutoff, retaining component_scale as its alias.
    keep_largest still refers to MASK components. New pre_min_component filters
    occupied components before morphology; it defaults to zero for compatibility.

    Labels are cleaned independently of geometric confidence. project assigns
    voted labels to all occupied voxels (including interior voxels); joint adds
    occupied-domain diffusion; separate returns {original_label: image_dict}.
    joint/project add only label_mask, label_map and label_score_image to the
    six result keys. label_mask is int32: 0 background, original class index+1.
    label_map includes {0:None}; input label zero remains an ordinary class.
    Separate shared geometry is filtered once; local optionally re-estimates
    supports within corrected classes. Reliable foreign cores prevent filling
    another class's enclosed region, without guaranteeing exact class topology.

    confidence is geometric reliability; label_confidence affects only label
    evidence. feature is an attribute, dense/sparse (N,) or (N,C), never an
    implicit confidence. Optional return_info returns (result,info), or paired
    dicts in separate mode. info carries original-row masks/corrected labels,
    point label/label_id (same positive codes as label_mask)/label_confidence,
    source_id and feature. Legacy mesh label_id remains 0-based separately.
    Scoring uses retained original observations, never generated points.

    Bounds must enclose all retained points and padding is extra voxel layers.
    Explicit spacing is never coarsened. Gaussian device may use optional CUDA;
    neighbors, support, EDT, morphology and label solve remain CPU.
    """
    import math
    import numpy as np
    from scipy import sparse
    from scipy.spatial import cKDTree

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
        chunk_size=chunk_size, workers=workers, min_survivors=2)
    retained, d, precision, axes, radii = _prepare_geometry(
        data["points"][original_sites] - center, data["confidence"][original_sites],
        **geometry_options)
    observations = _cloud_label_observations(data, retained, offset=center,
        label_outlier_action=label_outlier_action, label_outlier_k=label_outlier_k,
        label_outlier_thresh=label_outlier_thresh, label_outlier_iters=label_outlier_iters,
        label_min_points=label_min_points, graph_scale=graph_scale, workers=workers)
    sources = observations["unique_source"]
    x, geom_conf = data["points"][sources], data["confidence"][sources]
    diagnostics = {name: observations[name] for name in
        ("inlier_mask", "label_reliability", "label_outlier_mask", "label_changed_mask")}
    if data["keys"] is not None:
        names = np.empty(len(data["keys"]), object)
        names[:] = data["keys"]
        diagnostics["label_corrected"] = names[observations["corrected_codes"]]
    site_codes = site_certainty = site_evidence = None
    if data["keys"] is not None and label_mode == "separate":
        if np.any(observations["weights"] > 0):
            site_codes, site_certainty = _project_label_evidence(
                x - center, observations["observation_points"] - center,
                observations["codes"], observations["weights"], 1, d, workers, chunk_size)
        else:
            site_codes, site_certainty = np.full(len(x), -1), np.zeros(len(x))
        site_evidence = np.zeros(len(x))
        np.maximum.at(site_evidence, observations["obs_site"], observations["weights"])

    def build(ids, code=None):
        p, q, semiaxes, local_d = x[ids], precision[ids], axes[ids], d
        local_sources, local_conf = sources[ids], geom_conf[ids]
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
            _, local_d, q, semiaxes, _ = _prepare_geometry(
                p - center, local_conf * class_reliability, **local_options)
        step = local_d * spacing_factor if requested_spacing is None else requested_spacing.copy()
        sigma = local_d * smooth_factor if requested_sigma is None else requested_sigma.copy()
        effective_sigma = sigma * np.sqrt(time)
        field_method = method if method in {"sdf", "density"} and time > 0 else None
        if field_method is None or max_shift_scale == 0:
            effective_sigma = np.zeros(3)
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
        margin = (support_extent + local_d * (opening + closing) + abs(offset_scale) * local_d
                  + 4 * effective_sigma * stages.sum() + 2 * step)
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
        mask = (np.zeros(shape, bool) if method == "occupancy" else
                _support_mask(p - center, q, semiaxes, origin - center, step,
                              shape, chunk_size, workers))
        voxel = np.floor((p - origin) / step + 0.5).astype(np.int64)
        mask[tuple(voxel.T)] = True
        forbidden = None
        if code is not None and separate_protect_labels and label_guard_scale > 0:
            foreign = (site_codes != code) & (site_certainty >= 0.8) & (site_evidence >= 0.5)
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
            component_connectivity=26, hole_stage="after_offset", offset_method="morphology")
        indices = np.flatnonzero(mask)
        grid_points = np.empty((len(indices), 3), np.float64)
        score = np.empty(len(indices), np.float32)
        source_ids = np.empty(len(indices), np.int64)
        score_image = np.zeros(shape, np.float32)
        feature_values = None
        if data["feature"] is not None:
            feature_values = np.empty((len(indices),) + data["feature"].shape[1:], float)
        physical_score = local_d * score_axes
        score_tree = cKDTree((p - center) / physical_score)
        point_tree = cKDTree(p - center)
        nk, nf = min(score_k, len(p)), min(8, len(p))
        width = 1 if feature_values is None or feature_values.ndim == 1 else feature_values.shape[1]
        batch = min(chunk_size, max(1, 1_000_000 // max(nk, nf * width)))
        for start in range(0, len(indices), batch):
            sl = slice(start, start + batch)
            at = np.column_stack(np.unravel_index(indices[sl], shape))
            query = origin - center + at * step
            grid_points[sl] = query + center
            dist, near = score_tree.query(query / physical_score,
                k=list(range(1, nk + 1)), workers=workers)
            score[sl] = np.mean(local_conf[near] * np.exp(-0.5 * dist ** 2), axis=1)
            distance, near = point_tree.query(query, k=list(range(1, nf + 1)), workers=workers)
            source_ids[sl] = local_sources[near[:, 0]]
            if feature_values is not None:
                safe = np.maximum(distance, np.finfo(float).tiny)
                weights = local_conf[near] * (safe[:, :1] / safe) ** 2
                exact = distance[:, 0] == 0
                weights[exact] = 0
                weights[exact, 0] = 1
                weights /= weights.sum(axis=1, keepdims=True)
                rows = local_sources[near]
                if sparse.issparse(data["feature"]):
                    interpolation = sparse.csr_matrix((weights.ravel(),
                        (np.repeat(np.arange(len(near)), nf), rows.ravel())),
                        shape=(len(near), len(data["points"])))
                    feature_values[sl] = (interpolation @ data["feature"]).toarray()
                else:
                    feature_values[sl] = np.einsum("nk,nk...->n...", weights, data["feature"][rows])
        score_image.ravel()[indices] = score
        result = dict(mask=mask.astype(np.uint8), score_image=score_image,
                      spacing=step.copy(), origin=origin, points=grid_points, score=score)
        info = dict(diagnostics, base_spacing=local_d,
                    base_spacing_reference=local_d / base_spacing_scale,
                    base_spacing_scale=base_spacing_scale, smooth_sigma=sigma,
                    effective_smooth_sigma=effective_sigma, source_id=source_ids,
                    source_indices=local_sources.copy(), volume_info=volume_info,
                    method=method, label_mode=label_mode)
        if feature_values is not None:
            info["feature"] = feature_values
        if data["keys"] is not None:
            assigned = _grid_label_data(mask, origin, step, local_obs,
                mode="project" if code is not None else label_mode, d=local_d,
                label_smooth=label_smooth, k=label_project_k if code is not None or label_mode == "project" else support_k,
                workers=workers, point_indices=indices, label_max_iter=label_max_iter,
                label_tol=label_tol, label_lock_seeds=label_lock_seeds,
                label_smooth_weight=label_smooth_weight, label_axis_weights=label_axis_weights,
                chunk_size=chunk_size)
            for name in ("label_mask", "label_map", "label_score_image"):
                result[name] = assigned[name]
            info.update(label=assigned["point_label"],
                        label_id=assigned["point_label_id"],
                        label_confidence=assigned["point_label_score"],
                        label_info=assigned["label_info"])
        return result, info

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
            else:
                results[key], infos[key] = build(ids, code)
        return (results, infos) if return_info else results
    result, info = build(np.arange(len(x)))
    return (result, info) if return_info else result



def image_to_nrrd(result, filename, *, key="mask", space="3D-right-handed", units=None,
                  compression_level=1):
    """Write one selected image and its spatial metadata; return the output path.

    key='mask' writes a uint8 binary mask. key='label_mask' writes int32 tissue
    codes, with 0 for background and positive integers for labels; IDs above 255
    are preserved. key='score_image' or 'label_score_image' writes float32 values
    in [0,1], retaining fractional values. All selected images share mask's shape.
    Call separately for each image. Multi-byte data are explicitly little-endian.
    No reconstruction is performed and result is not modified. Array axis 0 is fastest.
    Optional result["direction"] has grid basis vectors as world-space COLUMNS;
    NRRD writes their spacing-scaled vectors in array-axis order. Identity is
    assumed when direction is absent (the points_to_image XYZ convention).
    Read with pynrrd: nrrd.read(path, index_order='F'). The writer itself has
    no pynrrd dependency: it writes a standard NRRD0005 header and gzip payload.

    space describes the INPUT coordinate convention: '3D-right-handed' (generic
    default), '3D-left-handed', 'RAS' or 'LPS' (or their long NRRD names).
    units is an optional string, e.g. 'um' or 'mm', or a three-string XYZ sequence.
    Metadata does not rescale/reorient coordinates. For mm coordinates, convert
    the input points yourself before reconstruction if conversion is needed.
    compression_level is a gzip level from 0 to 9. Filename must end in .nrrd.
    """
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



def _clean_points(x, k, support_k, method, threshold, iters, outlier_radius,
                  min_points, keep_largest, component_scale, radius_cap, workers):
    """Filter in the same anisotropic metric used for support construction."""
    # Cache queries when a stage retains every point.
    tree = dist = ids = None
    for stage in ["outlier"] * iters + ["components", "final"]:
        if len(x) < 2:
            raise ValueError("Fewer than two unique points remain; reduce filtering.")
        if tree is None:
            tree = cKDTree(x)
            kn = min(max(k, support_k), len(x) - 1)
            dist, ids = tree.query(x, k=kn + 1, workers=workers)
            dist, ids = dist[:, 1:], ids[:, 1:]
        if stage == "final":
            return x, tree, dist
        if stage == "outlier":
            if method is None or threshold is None or threshold == 0:
                continue
            kk = min(k, dist.shape[1])
            dd, nn = dist[:, :kk], ids[:, :kk]
            if method == "lof":
                reach = np.maximum(dd, dd[:, -1][nn])
                lrd = 1.0 / np.maximum(reach.mean(axis=1), np.finfo(float).tiny)
                keep = (lrd[nn] / lrd[:, None]).mean(axis=1) <= threshold
            elif method == "sor":
                score = dd.mean(axis=1)
                med = np.median(score)
                mad = 1.4826 * np.median(np.abs(score - med))
                floor = 64 * np.finfo(float).eps * max(float(med), np.finfo(float).tiny)
                if mad <= floor:
                    mad = max(float(np.std(score)), floor)
                keep = score <= med + threshold * mad
            else:  # radius: require k neighbours within the chosen metric radius
                radius = threshold * np.median(dd[:, -1]) if outlier_radius is None else outlier_radius
                keep = dd[:, -1] <= radius
        else:
            if not (min_points or keep_largest):
                continue
            local = np.median(dist[:, :min(6, dist.shape[1])], axis=1)
            local = np.minimum(local, radius_cap * np.median(local))
            edge = dist <= component_scale * np.minimum(local[:, None], local[ids])
            rows, cols = np.nonzero(edge)
            graph = csr_matrix((np.ones(len(rows), dtype=np.uint8),
                                (rows, ids[rows, cols])), shape=(len(x), len(x)))
            _, labels = connected_components(graph, directed=False)
            sizes = np.bincount(labels)
            cutoff = min_points * sizes.max() if 0 < min_points < 1 else min_points
            valid = sizes >= max(1, cutoff)
            if keep_largest:
                valid &= np.arange(len(sizes)) == sizes.argmax()
            keep = valid[labels]
        if not keep.all():
            x = x[keep]
            tree = None



def _support_mask_2(x, tree, radii, origin, spacing, shape, axis_scale, method,
                  k, bandwidth, density_level, chunk_size, workers):
    """kNN/nearest in batches; KDE via trilinear deposition and convolution."""
    if method == "kde":
        density = np.zeros(shape, dtype=np.float32)
        # x and tree coordinates are metric coordinates; grid is in physical units.
        for start in range(0, len(x), chunk_size):
            q = (x[start:start + chunk_size] * axis_scale - origin) / spacing
            ijk = np.floor(q).astype(np.int64)
            frac = q - ijk
            for offset in np.ndindex(2, 2, 2):
                offset = np.asarray(offset)
                weight = np.prod(np.where(offset, frac, 1 - frac), axis=1)
                np.add.at(density, tuple((ijk + offset).T), weight)
        ndi.gaussian_filter(density, bandwidth / spacing, output=density,
                            mode="constant", truncate=3.0)
        sampled = np.empty(len(x), dtype=np.float32)
        for start in range(0, len(x), chunk_size):
            q = (x[start:start + chunk_size] * axis_scale - origin) / spacing
            sampled[start:start + chunk_size] = ndi.map_coordinates(
                density, q.T, order=1, mode="constant", prefilter=False)
        level = density_level * float(np.median(sampled))
        return density > level

    total = int(np.prod(shape))
    mask = np.empty(total, dtype=bool)
    neighbors = list(range(1, k + 1)) if method == "knn" else [1]
    radius = float(np.median(radii))
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        ijk = np.column_stack(np.unravel_index(np.arange(start, stop), shape))
        q = (origin + ijk * spacing) / axis_scale
        dist, ids = tree.query(q, k=neighbors, workers=workers)
        if method == "nearest":
            mask[start:stop] = dist[:, 0] <= radius
        else:
            dist /= radii[ids]
            np.square(dist, out=dist)
            mask[start:stop] = dist.mean(axis=1) <= 1.0
    return mask.reshape(shape)



def _sdf(mask, spacing):
    phi = ndi.distance_transform_edt(~mask, sampling=spacing).astype(np.float32)
    phi -= ndi.distance_transform_edt(mask, sampling=spacing).astype(np.float32)
    return phi



def _smooth_field(mask, spacing, smooth, scales, method, max_shift, preserve_volume):
    """Each scale re-embeds the previous mask; all stages use one original band."""
    phi0 = _sdf(mask, spacing)
    if method is None or not np.any(smooth) or max_shift == 0:
        return phi0, mask
    # Freeze BOTH endpoints explicitly; do not count immutable boundary ties.
    band = np.abs(phi0) < max_shift
    if not band.any():
        return phi0, mask
    target = int(mask[band].sum())
    frozen_inside = (phi0 < 0) & ~band
    frozen_outside = (phi0 > 0) & ~band
    current = mask
    field = phi0
    for scale in scales:
        if method == "sdf":
            base = phi0 if current is mask else _sdf(current, spacing)
            field = ndi.gaussian_filter(base, smooth * scale / spacing,
                                        mode="nearest", truncate=3.0)
        else:  # occupancy: a dimensionless implicit field, negative inside
            field = ndi.gaussian_filter(current.astype(np.float32),
                                        smooth * scale / spacing,
                                        mode="constant", truncate=3.0)
            field *= -1
            field += 0.5
        if preserve_volume:
            values = field[band]
            if target == 0:
                level = values.min()  # field < 0: exclude even the smallest value
            elif target == len(values):
                level = np.nextafter(values.max(), np.float32(np.inf))
            else:
                values.partition(target)
                level = values[target]  # correct count when threshold values are distinct
                below = np.count_nonzero(values < level)
                through = np.count_nonzero(values <= level)
                if target - below > through - target:
                    level = np.nextafter(level, np.float32(np.inf))
            field -= level
        if method == "sdf":
            np.maximum(field, phi0 - max_shift, out=field)
            np.minimum(field, phi0 + max_shift, out=field)
            field[~band] = phi0[~band]
        else:
            field[frozen_inside] = -0.5
            field[frozen_outside] = 0.5
        current = field < 0
    return field, current



def grid_to_mesh(info, mc_method=None, *, backend=None, device="cpu"):
    """Return (world vertices, triangular faces) from an XYZ reconstruction.

    Accepts grid_point_cloud info or points_to_image output. Missing direction
    means identity. Reuses a finite implicit field when its sign agrees with
    mask; otherwise rebuilds a negative-inside SDF from the current mask.
    field_inside is "negative" by default, or "positive" when explicitly set.
    A one-node exterior layer closes masks that touch the image bounds; it
    assumes space outside the supplied image is empty. Reflections preserve
    outward winding. Vertices lie on an interpolated surface, not grid nodes.

    backend: auto, pymcubes, skimage, torchmcubes, lewiner or lorensen.
    mc_method is the legacy alias; do not specify both names.
    """
    backend = _alias(backend, mc_method, "auto", "backend", "mc_method")
    if backend not in {"auto", "pymcubes", "skimage", "torchmcubes", "lewiner", "lorensen"}:
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
    spacing_scale=0.5,
    base_spacing=None,
    base_spacing_scale=1.0,
    axis_scale=1.0,
    direction=None,
    method='knn',
    support_scale=1.25,
    support_k=8,
    support_radius=None,
    radius_cap=2.0,
    kernel_scale=2.0,
    density_level=0.2,
    smooth_sigma=None,
    smooth=None,
    smooth_scale=1.0,
    smooth_scales=(1.0,),
    smooth_method='sdf',
    max_shift=None,
    preserve_volume=True,
    outlier_method='lof',
    outlier_k=24,
    outlier_thresh=1.8,
    outlier_iters=1,
    outlier_radius=None,
    min_points=None,
    min_component=None,
    keep_largest=False,
    graph_scale=None,
    component_scale=None,
    closing=0.0,
    opening=0.0,
    fill_holes=False,
    output='volume',
    backend=None,
    mc_method=None,
    device='cpu',
    max_voxels=2_000_000_000,
    chunk_size=100000,
    workers=-1,
    return_info=False,
    return_field=False,
    confidence=None,
    geometry_confidence=None,
    feature=None,
    label=None,
    label_confidence=None,
    label_mode='project',
    label_smooth=None,
    label_smooth_weight=None,
    label_axis_weights=None,
    label_max_iter=1000,
    label_tol=1e-05,
    label_lock_seeds=False,
    label_project_k=1,
    label_outlier_k=24,
    label_outlier_thresh=0.8,
    label_outlier_iters=1,
    label_min_points=3,
    label_outlier_action='relabel',
    separate_protect_labels=True,
    label_guard_scale=0.6,
    closing_scale=None,
    opening_scale=None,
    hole_scale=None,
    offset_scale=0.0,
    max_shift_scale=_GRID_UNSET,
    field_smooth_iters=1,
    pre_min_component=0.0,
    min_volume_component=0.0,
    component_diameter_scale=0.0,
    score_k=8,
    score_scale=2.0,
):
    """Regular grid reconstruction; see the shared implementation docstring below."""
    return _grid_point_cloud_impl(points, spacing,
        spacing_scale=spacing_scale,
        base_spacing=base_spacing,
        base_spacing_scale=base_spacing_scale,
        axis_scale=axis_scale,
        direction=direction,
        method=method,
        support_scale=support_scale,
        support_k=support_k,
        support_radius=support_radius,
        radius_cap=radius_cap,
        kernel_scale=kernel_scale,
        density_level=density_level,
        smooth_sigma=smooth_sigma,
        smooth=smooth,
        smooth_scale=smooth_scale,
        smooth_scales=smooth_scales,
        smooth_method=smooth_method,
        max_shift=max_shift,
        preserve_volume=preserve_volume,
        outlier_method=outlier_method,
        outlier_k=outlier_k,
        outlier_thresh=outlier_thresh,
        outlier_iters=outlier_iters,
        outlier_radius=outlier_radius,
        min_points=min_points,
        min_component=min_component,
        keep_largest=keep_largest,
        graph_scale=graph_scale,
        component_scale=component_scale,
        closing=closing,
        opening=opening,
        fill_holes=fill_holes,
        output=output,
        backend=backend,
        mc_method=mc_method,
        device=device,
        max_voxels=max_voxels,
        chunk_size=chunk_size,
        workers=workers,
        return_info=return_info,
        return_field=return_field,
        confidence=confidence,
        geometry_confidence=geometry_confidence,
        feature=feature,
        label=label,
        label_confidence=label_confidence,
        label_mode=label_mode,
        label_smooth=label_smooth,
        label_smooth_weight=label_smooth_weight,
        label_axis_weights=label_axis_weights,
        label_max_iter=label_max_iter,
        label_tol=label_tol,
        label_lock_seeds=label_lock_seeds,
        label_project_k=label_project_k,
        label_outlier_k=label_outlier_k,
        label_outlier_thresh=label_outlier_thresh,
        label_outlier_iters=label_outlier_iters,
        label_min_points=label_min_points,
        label_outlier_action=label_outlier_action,
        separate_protect_labels=separate_protect_labels,
        label_guard_scale=label_guard_scale,
        closing_scale=closing_scale,
        opening_scale=opening_scale,
        hole_scale=hole_scale,
        offset_scale=offset_scale,
        max_shift_scale=max_shift_scale,
        field_smooth_iters=field_smooth_iters,
        pre_min_component=pre_min_component,
        min_volume_component=min_volume_component,
        component_diameter_scale=component_diameter_scale,
        score_k=score_k,
        score_scale=score_scale,
    )



def _clay_smooth_field(mask, spacing, sigma, scales, method, density_level,
                       preserve_volume, max_shift, device, *, forbidden=None):
    """Return a continuous, positive-inside physical implicit field and mask.

    Stages re-embed changed masks; final continuous values are retained. Volume
    matching targets the POST-morphology mask. The physical field constraint is
    phi0-max_shift <= phi <= phi0+max_shift, not a discontinuous node sign reset.
    Density uses its requested occupancy level (or a fitted volume level), then
    divides by a bounded typical physical gradient; this preserves its zero set while
    making the same physical change constraint meaningful. Neither constraint
    claims a strict final-mesh displacement or topology bound.
    """
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
    field = phi0.copy()
    for stage, scale in enumerate(scales):
        field = np.asarray(_gaussian(base, sigma * scale / spacing, device),
                           np.float32)
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
        if stage + 1 < len(scales) and not np.array_equal(updated, current):
            base = (-_sdf(updated, spacing).astype(np.float32) if method == "sdf"
                    else updated.astype(np.float32))
        current = updated
    return field, current



def _joint_surfaces(label_volume, origin, spacing, *,
                    interface_backend="surface_nets", interface_smooth_iters=15,
                    interface_max_shift_scale=1.0):
    """Extract all interfaces once; labels are 0=background and 1..C=tissues.

    The input array is XYZ, with ``origin`` at array index (0, 0, 0).
    BoundaryLabels records (inside_label, outside_label): normals point from
    component 0 to component 1. Positive IDs are sorted; background is last.
    Built-in Surface Nets smoothing is disabled: coincident points are welded
    FIRST, then every interface shares the same constrained vertex update.
    """
    import pyvista as pv

    a = np.asarray(label_volume)
    origin = np.asarray(origin, dtype=float)
    spacing = np.broadcast_to(np.asarray(spacing, dtype=float), (3,))
    if a.ndim != 3 or min(a.shape) < 1 or a.dtype.kind not in "biu":
        raise ValueError("label_volume must be a nonempty 3-D integer XYZ array.")
    if np.any(a < 0) or (a.size and a.max() > np.iinfo(np.int32).max):
        raise ValueError("Joint label IDs must be nonnegative int32 values.")
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("origin must contain three finite values.")
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("spacing must contain positive finite values.")
    if (not np.isfinite(interface_smooth_iters) or interface_smooth_iters < 0
            or int(interface_smooth_iters) != interface_smooth_iters):
        raise ValueError("interface_smooth_iters must be a nonnegative integer.")
    if not np.isfinite(interface_max_shift_scale) or interface_max_shift_scale < 0:
        raise ValueError("interface_max_shift_scale must be finite and nonnegative.")
    if interface_backend not in {"surface_nets", "voxel_faces", "auto"}:
        raise ValueError("interface_backend must be 'surface_nets', 'voxel_faces', or 'auto'.")
    a = np.pad(a.astype(np.int32, copy=False), 1)
    ids = np.unique(a)
    ids = ids[ids > 0]
    if not len(ids):
        return pv.PolyData()

    if interface_backend in {"surface_nets", "auto"}:
        try:
            from vtkmodules.vtkFiltersCore import vtkSurfaceNets3D
        except ImportError as exc:
            if interface_backend == "surface_nets":
                raise ImportError("surface_nets requires VTK >= 9.3; install a compatible "
                                  "VTK/PyVista pair or set interface_backend='voxel_faces'.") from exc
            interface_backend = "voxel_faces"
        else:
            interface_backend = "surface_nets"
            from vtkmodules.vtkCommonDataModel import vtkImageData
            from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
            grid = vtkImageData()
            grid.SetDimensions(a.shape)
            grid.GetPointData().SetScalars(numpy_to_vtk(a.ravel(order="F"), deep=True))
            filt = vtkSurfaceNets3D()
            filt.SetInputData(grid)
            filt.SetBackgroundLabel(0)
            filt.SetNumberOfLabels(len(ids))
            for i, key in enumerate(ids):
                filt.SetLabel(i, int(key))
            filt.SmoothingOff()
            filt.SetOutputMeshTypeToQuads()
            filt.SetOutputStyleToDefault()
            filt.Update()
            raw = filt.GetOutput()
            if not raw.GetNumberOfCells():
                return pv.PolyData()
            pts = vtk_to_numpy(raw.GetPoints().GetData()).astype(float)
            quads = vtk_to_numpy(raw.GetPolys().GetData()).reshape(-1, 5)[:, 1:].copy()
            # VTK 9.3 does not orient every exterior face consistently with its
            # BoundaryLabels. Sample the two sides before smoothing instead.
            center = pts[quads].mean(axis=1)
            normals = np.cross(pts[quads[:, 1]] - pts[quads[:, 0]],
                               pts[quads[:, 2]] - pts[quads[:, 0]])
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)
            ij0 = np.floor(center - 0.25 * normals + 0.5).astype(np.int64)
            ij1 = np.floor(center + 0.25 * normals + 0.5).astype(np.int64)
            pairs = np.column_stack((a[tuple(ij0.T)], a[tuple(ij1.T)]))
            if np.any(pairs[:, 0] == pairs[:, 1]):
                raise RuntimeError("Surface Nets side labels could not be resolved.")

    if interface_backend == "voxel_faces":
        # Face-centered cubical interfaces, each differing neighbour pair once.
        # Integer corner coordinates permit exact welding across axes/labels.
        all_corners, all_pairs = [], []
        for axis in range(3):
            lo, hi = [slice(None)] * 3, [slice(None)] * 3
            lo[axis], hi[axis] = slice(None, -1), slice(1, None)
            al, ah = a[tuple(lo)], a[tuple(hi)]
            ij = np.argwhere(al != ah)
            if not len(ij):
                continue
            # Corners are in doubled grid coordinates; winding is +axis.
            u, v = (axis + 1) % 3, (axis + 2) % 3
            offsets = np.zeros((4, 3), dtype=np.int64)
            offsets[:, axis] = 1
            offsets[:, u] = (-1, 1, 1, -1)
            offsets[:, v] = (-1, -1, 1, 1)
            all_corners.append(2 * ij[:, None, :] + offsets[None, :, :])
            all_pairs.append(np.column_stack((al[tuple(ij.T)], ah[tuple(ij.T)])))
        corners = np.concatenate(all_corners)
        pts, inverse = np.unique(corners.reshape(-1, 3), axis=0, return_inverse=True)
        pts = pts.astype(float) * 0.5
        quads = inverse.reshape(-1, 4)
        pairs = np.concatenate(all_pairs)

    # Standardize label order while preserving the sampled face orientation.
    flip = (pairs[:, 0] == 0) | ((pairs[:, 1] != 0) & (pairs[:, 0] > pairs[:, 1]))
    quads[flip] = quads[flip, ::-1]
    pairs[flip] = pairs[flip, ::-1]
    # Weld before ANY displacement, including Surface Nets' duplicated points.
    pts, inverse = np.unique(pts, axis=0, return_inverse=True)
    quads = inverse[quads]
    pts = origin + (pts - 1.0) * spacing
    triangles = np.concatenate((quads[:, (0, 1, 2)], quads[:, (0, 2, 3)]))
    triangle_pairs = np.concatenate((pairs, pairs))

    limit = float(interface_max_shift_scale) * spacing.min()
    if interface_smooth_iters and limit > 0:
        # Fixed quad-edge stencil avoids directional bias from triangulation.
        edges = np.sort(np.concatenate((quads[:, (0, 1)], quads[:, (1, 2)],
                                        quads[:, (2, 3)], quads[:, (3, 0)])), axis=1)
        edges, counts = np.unique(edges, axis=0, return_counts=True)
        # Triple junctions move along their shared junction graph; all other
        # surface vertices use their full edge neighbourhood.
        junction_edges = edges[counts >= 3]
        junction = np.zeros(len(pts), dtype=bool)
        junction[junction_edges.ravel()] = True
        row, col = np.concatenate((edges, edges[:, ::-1])).T
        keep = ~junction[row] | (np.tile(counts, 2) >= 3)
        row, col = row[keep], col[keep]
        graph = sparse.csr_matrix((np.ones(len(row)), (row, col)), shape=(len(pts), len(pts)))
        degree = np.asarray(graph.sum(axis=1)).ravel()
        graph = sparse.diags(1.0 / np.maximum(degree, 1)) @ graph
        initial = pts.copy()
        ref = np.cross(initial[triangles[:, 1]] - initial[triangles[:, 0]],
                       initial[triangles[:, 2]] - initial[triangles[:, 0]])
        ref2 = np.einsum("ij,ij->i", ref, ref)
        for _ in range(int(interface_smooth_iters)):
            candidate = pts + 0.25 * (graph @ pts - pts)
            candidate[degree == 0] = pts[degree == 0]
            delta = candidate - initial
            length = np.linalg.norm(delta, axis=1)
            candidate = initial + delta * np.minimum(1.0, limit / np.maximum(length, 1e-30))[:, None]
            # Keep all triangles' orientation relative to the original faces.
            # Joint halving preserves the common vertex table and displacement cap.
            for _ in range(12):
                normal = np.cross(candidate[triangles[:, 1]] - candidate[triangles[:, 0]],
                                  candidate[triangles[:, 2]] - candidate[triangles[:, 0]])
                if np.all(np.einsum("ij,ij->i", normal, ref) > 1e-8 * ref2):
                    break
                candidate = 0.5 * (candidate + pts)
            else:
                break
            pts = candidate
    surface = pv.PolyData(pts, np.column_stack((np.full(len(triangles), 3), triangles)).ravel())
    surface.cell_data["BoundaryLabels"] = triangle_pairs.astype(np.int32)
    surface.point_data["global_point_id"] = np.arange(surface.n_points, dtype=np.int64)
    surface.cell_data["global_face_id"] = np.arange(surface.n_cells, dtype=np.int64)
    surface.field_data["interface_backend"] = np.asarray([interface_backend])
    return surface


def _split_joint_surface(surface, keys):
    """Split shared interfaces; ID ``code + 1`` corresponds to ``keys[code]``.

    For every A/B triangle the two outputs use exactly the same coordinates and
    reversed winding. Never independently smooth, decimate, or auto-orient these
    outputs if the shared-interface property must be retained.
    """
    import pyvista as pv

    result = {}
    if surface.n_cells:
        triangles = np.asarray(surface.faces).reshape(-1, 4)[:, 1:]
        pairs = np.asarray(surface.cell_data["BoundaryLabels"])
    for code, key in enumerate(keys):
        lid = code + 1
        selected = np.flatnonzero(np.any(pairs == lid, axis=1)) if surface.n_cells else np.empty(0, dtype=int)
        if not len(selected):
            warnings.warn(f"Label {key!r} has no occupied voxels or extractable interface; returning an empty mesh.", stacklevel=2)
            result[key] = pv.PolyData()
            continue
        faces = triangles[selected].copy()
        reverse = pairs[selected, 1] == lid
        faces[reverse] = faces[reverse, ::-1]
        point_ids, inverse = np.unique(faces.ravel(), return_inverse=True)
        mesh = pv.PolyData(np.asarray(surface.points)[point_ids].copy(),
                           np.column_stack((np.full(len(faces), 3), inverse.reshape(-1, 3))).ravel())
        for name in surface.point_data.keys():
            mesh.point_data[name] = np.asarray(surface.point_data[name])[point_ids].copy()
        for name in surface.cell_data.keys():
            mesh.cell_data[name] = np.asarray(surface.cell_data[name])[selected].copy()
        value = str(key) if np.asarray(key).ndim or np.asarray(key).dtype.kind == "O" else key
        mesh.point_data["label"] = np.full(mesh.n_points, value)
        mesh.point_data["label_id"] = np.full(mesh.n_points, code, dtype=np.int32)
        mesh.cell_data["BoundaryLabels"] = pairs[selected].copy()
        # Compute normals directly from prescribed winding: cavity components
        # keep their inward-to-cavity direction, and internal A/B normals oppose.
        f = inverse.reshape(-1, 3)
        p = np.asarray(mesh.points)
        normal = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
        point_normal = np.zeros_like(p)
        for j in range(3):
            np.add.at(point_normal, f[:, j], normal)
        mesh.cell_data["Normals"] = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
        mesh.point_data["Normals"] = point_normal / np.maximum(np.linalg.norm(point_normal, axis=1, keepdims=True), 1e-30)
        for name in surface.field_data.keys():
            mesh.field_data[name] = surface.field_data[name].copy()
        result[key] = mesh
    return result


def _joint_outer_surface(surface):
    """Return the exact joint external/cavity surface with outward winding.

    Select background-facing triangles after joint smoothing; this avoids a
    separate reconstruction whose exterior could differ from the tissue meshes.
    Cell label_id identifies the tissue adjacent to each external triangle.
    """
    import pyvista as pv

    if not surface.n_cells:
        return pv.PolyData()
    pairs = np.asarray(surface.cell_data["BoundaryLabels"])
    selected = np.flatnonzero(np.any(pairs == 0, axis=1))
    if not len(selected):
        return pv.PolyData()
    faces = np.asarray(surface.faces).reshape(-1, 4)[selected, 1:].copy()
    reverse = pairs[selected, 0] == 0
    faces[reverse] = faces[reverse, ::-1]
    point_ids, inverse = np.unique(faces.ravel(), return_inverse=True)
    mesh = pv.PolyData(np.asarray(surface.points)[point_ids].copy(),
                       np.column_stack((np.full(len(faces), 3), inverse.reshape(-1, 3))).ravel())
    for name in surface.point_data.keys():
        mesh.point_data[name] = np.asarray(surface.point_data[name])[point_ids].copy()
    for name in surface.cell_data.keys():
        mesh.cell_data[name] = np.asarray(surface.cell_data[name])[selected].copy()
    for name in surface.field_data.keys():
        mesh.field_data[name] = surface.field_data[name].copy()
    mesh.cell_data["label_id"] = pairs[selected].max(axis=1).astype(np.int32) - 1
    f, p = inverse.reshape(-1, 3), np.asarray(mesh.points)
    normal = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
    point_normal = np.zeros_like(p)
    for j in range(3):
        np.add.at(point_normal, f[:, j], normal)
    mesh.cell_data["Normals"] = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
    mesh.point_data["Normals"] = point_normal / np.maximum(np.linalg.norm(point_normal, axis=1, keepdims=True), 1e-30)
    return mesh


def _joint_partition(mask, origin, spacing, points, codes, confidence, d, *,
                     label_smooth_weight=0.3, label_axis_weights=(1, 1, 1.5),
                     label_lock_seeds=True, label_max_iter=200, label_tol=1e-5,
                     chunk=100000, workers=-1):
    """Partition occupied XYZ voxels jointly; return (labels, info).

    Labels are original integer codes + 1, with 0 reserved for background.
    Observations are confidence-weighted trilinear splats. The soft data weight
    is min(splat mass / median positive mass, 1); each observed site's class
    distribution supplies its data term. Six-neighbour physical conductances
    axis_weights / spacing**2 are normalized to interior weighted degree 1.
    Thus label_smooth_weight compares data fidelity with normalized diffusion.
    This is a dimensionless local regularizer; normalization cancels a common
    spacing factor, so it does not preserve a fixed physical diffusion length
    when grid resolution changes. Axis ratios still reflect physical spacing.
    Only conflict-free sites with splat-weighted mean confidence >= 0.9 are
    locked. Contradictory duplicate points remain soft mixed observations.
    With lock_seeds=True and only hard observations, positive diffusion weight
    cancels from the harmonic equation: changing it cannot move that boundary.

    Positive weights solve (diag(a) + weight * L) P = a*Q, eliminating hard
    seeds before CG. Classes are solved sequentially, never allocating V*C.
    At weight=0 observed distributions are extended from their graph-nearest
    observed site, without traversing background. Non-convergence raises an
    error rather than returning an unconverged segmentation.

    Points with no occupied splat corner may project to the nearest occupied
    node within max(norm(spacing), 0.5*d); more distant observations are unused.
    Any occupied component with no observation then raises ValueError. The
    output always covers exactly mask. info['label_confidence'] is a float32
    XYZ array of max(P)/sum(P), a relative assignment score, not a calibrated
    anatomical probability. workers is accepted for interface consistency.
    """
    from scipy.sparse import csgraph
    from scipy.sparse.linalg import cg, LinearOperator

    mask = np.asarray(mask, dtype=bool)
    origin = np.asarray(origin, dtype=float)
    spacing = np.broadcast_to(np.asarray(spacing, dtype=float), (3,)).copy()
    points = np.asarray(points, dtype=float)
    codes = np.asarray(codes)
    confidence = np.asarray(confidence, dtype=float)
    axis = np.asarray(label_axis_weights, dtype=float)
    weight = float(label_smooth_weight)
    if (mask.ndim != 3 or origin.shape != (3,) or axis.shape != (3,) or
            not np.isfinite(origin).all() or not np.isfinite(spacing).all() or
            np.any(spacing <= 0) or not np.isfinite(axis).all() or np.any(axis <= 0)):
        raise ValueError('mask must be XYZ; origin finite; spacing/label_axis_weights positive XYZ.')
    if (not np.isfinite(weight) or weight < 0 or not np.isfinite(d) or d <= 0 or
            not np.isfinite(label_tol) or label_tol <= 0 or
            int(label_max_iter) != label_max_iter or label_max_iter < 1 or chunk < 1):
        raise ValueError('Joint diffusion weight must be >=0; d, tolerance and iteration/chunk counts positive.')
    if (points.ndim != 2 or points.shape[1] != 3 or codes.shape != (len(points),) or
            confidence.shape != (len(points),) or not np.isfinite(points).all() or
            not np.issubdtype(codes.dtype, np.integer) or np.any(codes < 0) or
            not np.isfinite(confidence).all() or np.any((confidence <= 0) | (confidence > 1))):
        raise ValueError('Joint points require aligned nonnegative integer codes and confidence in (0,1].')
    occupied = np.flatnonzero(mask)
    n = len(occupied)
    if not n or not len(points):
        raise ValueError('Joint reconstruction needs occupied voxels and labelled observations.')
    if n >= np.iinfo(np.int32).max:
        raise ValueError('Joint grid exceeds the supported int32 occupied-voxel count.')
    classes = np.unique(codes)
    class_codes = np.searchsorted(classes, codes)
    nclass = len(classes)
    lookup = np.full(mask.shape, -1, dtype=np.int32)
    lookup.ravel()[occupied] = np.arange(n, dtype=np.int32)
    shape = np.asarray(mask.shape)

    # Sparse V*C observation table: only local splat contributions are stored.
    rows, cols, masses, reliabilities = [], [], [], []
    unused = []
    chunk = int(chunk)
    for start in range(0, len(points), chunk):
        stop = min(start + chunk, len(points))
        xyz = (points[start:stop] - origin) / spacing
        lower = np.floor(xyz).astype(np.int64)
        frac = xyz - lower
        used = np.zeros(stop - start, dtype=bool)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    corner = np.asarray((dx, dy, dz))
                    ijk = lower + corner
                    w = np.prod(np.where(corner, frac, 1 - frac), axis=1)
                    valid = ((ijk >= 0) & (ijk < shape)).all(1) & (w > 1e-12)
                    ids = np.flatnonzero(valid)
                    if not len(ids):
                        continue
                    rr = lookup[tuple(ijk[ids].T)]
                    ids, rr = ids[rr >= 0], rr[rr >= 0]
                    if not len(ids):
                        continue
                    used[ids] = True
                    ids_global = ids + start
                    m = w[ids] * confidence[ids_global]
                    rows.append(rr)
                    cols.append(class_codes[ids_global])
                    masses.append(m)
                    reliabilities.append(w[ids])
        unused.extend((np.flatnonzero(~used) + start).tolist())

    # Bounded exact projection. Shortlist occupied nodes with a nearby unused
    # point, in chunks; anisotropic spacing never expands a cubic grid stencil.
    projected = 0
    unused = np.asarray(unused, dtype=np.int64)
    max_projection = max(float(np.linalg.norm(spacing)), 0.5 * float(d))
    if len(unused):
        eligible = unused
        if len(eligible):
            point_tree = cKDTree(points[eligible])
            candidate_coordinates, candidate_rows = [], []
            limit = np.nextafter(max_projection, np.inf)
            for start in range(0, n, chunk):
                stop = min(start + chunk, n)
                ijk = np.column_stack(np.unravel_index(occupied[start:stop], mask.shape))
                coords = origin + ijk * spacing
                distances = point_tree.query(coords, k=1, distance_upper_bound=limit,
                                             workers=workers)[0]
                close = np.isfinite(distances)
                if np.any(close):
                    candidate_coordinates.append(coords[close])
                    candidate_rows.append(np.flatnonzero(close) + start)
            if candidate_rows:
                coords = np.concatenate(candidate_coordinates)
                candidate_rows = np.concatenate(candidate_rows)
                candidate_tree = cKDTree(coords)
                distances, neighbours = candidate_tree.query(points[eligible], k=1,
                                        distance_upper_bound=limit, workers=workers)
                good = np.isfinite(distances)
                selected = eligible[good]
                projected = len(selected)
                if projected:
                    rows.append(candidate_rows[neighbours[good]])
                    cols.append(class_codes[selected])
                    masses.append(confidence[selected])
                    reliabilities.append(np.ones(projected))
                    unused = unused[~np.isin(unused, selected)]
                del coords, candidate_coordinates, candidate_rows, candidate_tree
            del point_tree
    if not rows:
        raise ValueError('No label observations reach the occupied mask; reduce smoothing/opening or use finer spacing.')
    rr, cc = np.concatenate(rows), np.concatenate(cols)
    mm, geometric_mass = np.concatenate(masses), np.concatenate(reliabilities)
    observation = sparse.coo_matrix((mm, (rr, cc)), shape=(n, nclass)).tocsr()
    observation.eliminate_zeros()
    mass = np.asarray(observation.sum(axis=1)).ravel()
    denominator = np.bincount(rr, weights=geometric_mass, minlength=n)
    mean_reliability = np.divide(mass, denominator, out=np.zeros(n), where=denominator > 0)
    observed = mass > 0
    a = np.minimum(mass / np.median(mass[observed]), 1)
    observation = sparse.diags(np.divide(a, mass, out=np.zeros(n), where=observed)) @ observation
    del rows, cols, masses, reliabilities, rr, cc, mm, geometric_mass, denominator

    components, count = ndi.label(mask)
    component_ids = components.ravel()[occupied]
    component_observed = np.bincount(component_ids, weights=observed, minlength=count + 1)
    missing = np.flatnonzero(component_observed[1:] == 0) + 1
    if len(missing):
        raise ValueError(f'{len(missing)} occupied component(s) have no nearby label observations; '
                         'reduce field smoothing/opening or use finer spacing.')
    del components, component_ids
    hard = observed & (np.diff(observation.indptr) == 1) & (mean_reliability >= 0.9)
    hard &= bool(label_lock_seeds)
    hard_class = np.full(n, -1, dtype=np.int32)
    hard_class[hard] = observation.indices[observation.indptr[:-1][hard]]
    active_seeds = np.flatnonzero(observed)
    label_indices = np.full(n, -1, dtype=np.int32)
    score = np.zeros(n, dtype=float)
    total = np.zeros(n, dtype=float)

    if nclass == 1:
        labels = np.zeros(mask.shape, dtype=np.int32)
        labels[mask] = int(classes[0]) + 1
        if len(unused):
            warnings.warn(f'Joint partition ignored {len(unused)} label observation(s) outside the final '
                          'mask and conservative projection distance.', RuntimeWarning, stacklevel=2)
        return labels, dict(label_confidence=mask.astype(np.float32), label_converged=True,
            label_iterations=[0], label_residuals=[0.0], label_seed_voxels=int(observed.sum()),
            label_hard_seed_voxels=int(hard.sum()), label_conflict_voxels=0,
            label_projected_points=int(projected), label_unused_points=int(len(unused)),
            label_components=int(count), label_classes=classes.copy(),
            label_smooth_weight=weight, label_axis_weights=axis.copy())

    # Store each undirected interior edge once. Conductance normalization keeps
    # lambda comparable across grid resolution; anisotropic ratios stay physical.
    conductance = axis / spacing**2
    conductance /= 2 * conductance.sum()
    edge_left, edge_right, edge_weights = [], [], []
    for dimension in range(3):
        low, high = [slice(None)] * 3, [slice(None)] * 3
        low[dimension], high[dimension] = slice(None, -1), slice(1, None)
        u, v = lookup[tuple(low)].ravel(), lookup[tuple(high)].ravel()
        keep = (u >= 0) & (v >= 0)
        edge_left.append(u[keep])
        edge_right.append(v[keep])
        edge_weights.append(np.full(np.count_nonzero(keep), conductance[dimension]))
    u, v, w = np.concatenate(edge_left), np.concatenate(edge_right), np.concatenate(edge_weights)
    del edge_left, edge_right, edge_weights, lookup
    iterations, residuals = [], []
    if weight == 0:
        # Observed voxels retain their mixed-label MAP assignment. Unobserved
        # voxels inherit the nearest observed site by physical graph distance.
        for c in range(nclass):
            p = observation.getcol(c).toarray().ravel()
            improve = p > score
            label_indices[improve], score[improve] = c, p[improve]
        np.divide(score, a, out=score, where=observed)
        if np.any(~observed):
            lengths = 1 / np.sqrt(w)
            graph = sparse.coo_matrix((np.r_[lengths, lengths], (np.r_[u, v], np.r_[v, u])),
                                      shape=(n, n)).tocsr()
            distance, predecessor, source = csgraph.dijkstra(
                graph, directed=False, indices=active_seeds, min_only=True,
                return_predecessors=True)
            fill = ~observed
            if np.any(source[fill] < 0):
                raise RuntimeError('Occupied graph contains an unreachable unobserved voxel.')
            label_indices[fill] = label_indices[source[fill]]
            score[fill] = score[source[fill]]
            del graph, distance, predecessor, source
        total.fill(1)
    else:
        free = ~hard
        free_ids = np.flatnonzero(free)
        mapping = np.full(n, -1, dtype=np.int32)
        mapping[free] = np.arange(len(free_ids), dtype=np.int32)
        degree = np.bincount(u, weights=w, minlength=n) + np.bincount(v, weights=w, minlength=n)
        diagonal = (a + weight * degree)[free]
        both_free = free[u] & free[v]
        off_u, off_v = mapping[u[both_free]], mapping[v[both_free]]
        off_w = -weight * w[both_free]
        matrix = sparse.coo_matrix((np.r_[off_w, off_w], (np.r_[off_u, off_v], np.r_[off_v, off_u])),
                                   shape=(len(free_ids), len(free_ids))).tocsr()
        matrix += sparse.diags(diagonal)
        rhs = observation[free].tocsr()
        # Add Dirichlet neighbours to the RHS, never overwriting only matrix rows.
        for source, target in ((u, v), (v, u)):
            boundary = free[source] & hard[target]
            if np.any(boundary):
                rhs += sparse.coo_matrix((weight * w[boundary],
                                          (mapping[source[boundary]], hard_class[target[boundary]])),
                                         shape=(len(free_ids), nclass)).tocsr()
        rhs = rhs.tocsc()
        label_indices[hard], score[hard], total[hard] = hard_class[hard], 1, 1
        if len(free_ids):
            inv_diagonal = 1 / diagonal
            preconditioner = LinearOperator(matrix.shape, matvec=lambda x: inv_diagonal * x)
            for c in range(nclass):
                b = rhs.getcol(c).toarray().ravel()
                if not np.any(b):
                    iterations.append(0)
                    residuals.append(0.0)
                    continue
                calls = [0]
                def callback(_):
                    calls[0] += 1
                # scipy <1.12 uses tol; newer scipy uses rtol and atol.
                try:
                    p, status = cg(matrix, b, M=preconditioner, rtol=label_tol, atol=0,
                                   maxiter=int(label_max_iter), callback=callback)
                except TypeError:
                    p, status = cg(matrix, b, M=preconditioner, tol=label_tol,
                                   maxiter=int(label_max_iter), callback=callback)
                residual = float(np.linalg.norm(matrix @ p - b) / np.linalg.norm(b))
                if status != 0 or not np.isfinite(p).all():
                    raise RuntimeError(f'Joint label diffusion did not converge for code {classes[c]} '
                                       f'(CG status={status}, relative residual={residual:.3g}); '
                                       'increase label_max_iter or reduce grid size.')
                p = np.clip(p, 0, 1)
                improve = p > score[free_ids]
                targets = free_ids[improve]
                label_indices[targets], score[targets] = c, p[improve]
                total[free_ids] += p
                iterations.append(calls[0])
                residuals.append(residual)
        np.divide(score, total, out=score, where=total > 0)
    if np.any(label_indices < 0) or np.any(total <= 0):
        raise RuntimeError('Joint diffusion left an occupied voxel without label evidence.')
    labels = np.zeros(mask.shape, dtype=np.int32)
    labels.ravel()[occupied] = classes[label_indices].astype(np.int32) + 1
    confidence_image = np.zeros(mask.shape, dtype=np.float32)
    confidence_image.ravel()[occupied] = np.clip(score, 0, 1)
    if len(unused):
        warnings.warn(f'Joint partition ignored {len(unused)} label observation(s) outside the final occupied '
                      'mask and conservative projection distance.', RuntimeWarning, stacklevel=2)
    info = dict(label_confidence=confidence_image, label_converged=True,
                label_iterations=iterations, label_residuals=residuals,
                label_seed_voxels=int(observed.sum()), label_hard_seed_voxels=int(hard.sum()),
                label_conflict_voxels=int(np.count_nonzero(np.diff(observation.indptr) > 1)),
                label_projected_points=int(projected), label_unused_points=int(len(unused)),
                label_components=int(count), label_classes=classes.copy(),
                label_smooth_weight=weight, label_axis_weights=axis.copy())
    return labels, info


# Shared reconstruction and label-noise helpers.

def _clean_cloud(points, confidence, k, outlier_thresh, min_component, workers,
                 *, outlier_method="lof", outlier_k=None, outlier_iters=1,
                 outlier_radius=None, min_points=None, graph_scale=2.0,
                 radius_cap=1.6, min_survivors=4, keep_largest=False):
    """Return retained input-row indices, reference length d, and support radii.

    Inputs are finite unique XYZ rows and positive confidence. All labels are
    cleaned together. k estimates geometry; outlier_k (None -> k) is independent.
    outlier_method is lof, ratio, sor, radius, or None/'none'. Zero threshold or
    zero iterations disables outlier rejection. LOF/ratio require a cutoff >1;
    SOR/radius require a positive cutoff. Each round recomputes the neighbourhood.
    SOR uses median + threshold * 1.4826*MAD, with standard-deviation
    fallback when MAD is numerically degenerate (e.g. regular lattice interiors).
    Radius filtering requires outlier_k neighbours within outlier_radius (physical
    units), or threshold * median(kth neighbour distance) when radius is None.
    LOF/ratio/SOR neighbour counts are capped at N-1; radius requires the requested
    count. min_points=None inherits min_component; 0 disables point-group removal,
    (0,1) is relative to the largest point group, >=1 an absolute point count.
    graph_scale limits edges by the smaller local median of up to six distances.
    radius_cap>=1 caps local support expansion: clip(d_i, 0.6*d, radius_cap*d).
    Confidence then scales radii by its cube root, exactly as in the compact API.
    """
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




def _volume_sdf(mask, spacing):
    """Positive inside, in physical units; keep a continuous final field."""
    field = ndi.distance_transform_edt(mask, sampling=spacing).astype(np.float32)
    field -= ndi.distance_transform_edt(~mask, sampling=spacing)
    return field



def _morph_volume(mask, radius, spacing, operation):
    """Ellipsoid morphology; a zero-radius axis never exchanges voxels."""
    radius, spacing = np.asarray(radius), np.asarray(spacing)
    active, fixed = np.flatnonzero(radius > 0), np.flatnonzero(radius == 0)
    if not len(active) or not mask.any():
        return mask.copy()
    result = np.empty_like(mask)
    sampling = spacing[active] / radius[active]
    for index in np.ndindex(*(mask.shape[a] for a in fixed)):
        sl = [slice(None)] * 3
        for axis, value in zip(fixed, index):
            sl[axis] = value
        sl = tuple(sl)
        part = mask[sl]
        if not part.any():
            result[sl] = False
        elif operation == "dilate":
            result[sl] = ndi.distance_transform_edt(~part, sampling=sampling) <= 1
        else:
            distance = ndi.distance_transform_edt(np.pad(part, 1), sampling=sampling)
            result[sl] = distance[(slice(1, -1),) * part.ndim] > 1
    return result



def _fill_volume_holes(mask, spacing, diameter, forbidden=None):
    """Fill bounded enclosed 26-connected voids, excluding protected cavities.

    A single protected voxel vetoes filling its whole background component;
    reliable foreign tissue therefore cannot be mistaken for an empty cavity.
    """
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



def _volume_level(values, target):
    """Threshold for field>level; equal-valued voxels are never split by index."""
    values = np.asarray(values, dtype=np.float32).ravel()
    if not len(values):
        return np.float32(0)
    if target <= 0:
        return values.max()
    if target >= len(values):
        return np.nextafter(values.min(), np.float32(-np.inf))
    cut = len(values) - int(target)
    bracket = np.partition(values, (cut - 1, cut))
    low, high = bracket[cut - 1], bracket[cut]
    if low < high:
        level = np.float32(float(low) + (float(high) - float(low)) * 0.5)
        return low if level >= high else level
    above = np.count_nonzero(values > high)
    tied = np.count_nonzero(values == high)
    return (np.nextafter(high, np.float32(-np.inf))
            if abs(above + tied - target) <= abs(above - target) else high)



def _occupied_volume(points, radii, d, support_scale, spacing_scale,
                     smooth_scale, min_component, max_voxels, workers, *,
                     smooth_scales=(1.0,), preserve_volume=True,
                     max_shift_scale=None, opening_scale=0.0,
                     closing_scale=0.0, hole_scale=0.0, offset_scale=0.0):
    """Shared support -> opening/closing -> bounded holes -> staged SDF -> offset.

    XYZ opening/closing scales are DIAMETERS times d, with zero axes inactive.
    hole_scale bounds enclosed background by its equivalent-sphere diameter;
    np.inf fills all enclosed holes, never exterior-connected tunnels. Background
    uses 26-connectivity so a diagonal escape also prevents explicit hole filling.
    Each smoothing stage uses sigma=d*smooth_scale*stage and re-distances changed
    masks. Volume matching targets the POST-morphology mask, not raw support.
    max_shift_scale bounds smoothing FIELD change from that mask's original SDF;
    it is not a strict surface-displacement or topology guarantee. Zero disables
    field smoothing. Explicit nonzero offset first re-distances the final mask,
    then adds d*offset_scale (positive expands); this introduces voxel uncertainty.
    min_component filtering follows offset. No tissue is independently processed.
    """
    import math

    points, radii = np.asarray(points, float), np.asarray(radii, float)
    support = np.broadcast_to(np.asarray(support_scale, float), (3,))
    spacing = d * np.broadcast_to(np.asarray(spacing_scale, float), (3,))
    sigma = d * np.broadcast_to(np.asarray(smooth_scale, float), (3,))
    scales = np.asarray(smooth_scales, float).reshape(-1)
    opening = 0.5 * d * np.broadcast_to(np.asarray(opening_scale, float), (3,))
    closing = 0.5 * d * np.broadcast_to(np.asarray(closing_scale, float), (3,))
    delta = None if max_shift_scale is None else float(d * max_shift_scale)
    radius_max = float(radii.max())
    # Extend the original single-stage grid by WHOLE voxels. Otherwise merely
    # changing a morphology/stage parameter changes the sampling phase of support.
    margin = radius_max * support + 3.0 * sigma + 3.0 * spacing
    extra = (3.0 * sigma * max(float(scales.sum()) - 1.0, 0.0)
             + opening + closing + abs(d * offset_scale))
    margin += np.ceil(extra / spacing) * spacing
    origin = points.min(axis=0) - margin
    extent = points.max(axis=0) + margin - origin
    shape = tuple(int(v) + 1 for v in np.ceil(extent / spacing))
    size = math.prod(shape)
    if size > int(max_voxels):
        raise ValueError(f"Grid {shape} needs {size:,} voxels, exceeding max_voxels="
                         f"{int(max_voxels):,}; increase spacing_scale or max_voxels.")

    # Exact union of adaptive ellipsoids, via expanding KD-tree candidate lists.
    tree = cKDTree(points / support)
    lookup_radius = np.r_[radii, -1.0]
    mask = np.zeros(size, dtype=bool)
    query_bound = np.nextafter(radius_max, np.inf)
    for start in range(0, size, 32768):
        stop = min(start + 32768, size)
        ijk = np.column_stack(np.unravel_index(np.arange(start, stop), shape))
        query = (origin + ijk * spacing) / support
        unresolved, kk = np.arange(len(query)), min(8, len(points))
        while len(unresolved):
            next_unresolved = []
            query_chunk = max(1, min(32768, 1_000_000 // kk))
            for offset in range(0, len(unresolved), query_chunk):
                rows = unresolved[offset:offset + query_chunk]
                distance, index = tree.query(query[rows], k=kk,
                    distance_upper_bound=query_bound, workers=workers)
                if kk == 1:
                    distance, index = distance[:, None], index[:, None]
                supported = np.any(distance <= lookup_radius[index], axis=1)
                mask[start + rows[supported]] = True
                if kk < len(points):
                    revisit = ~supported & (distance[:, -1] <= radius_max)
                    if revisit.any():
                        next_unresolved.append(rows[revisit])
            if not next_unresolved:
                break
            unresolved, kk = np.concatenate(next_unresolved), min(2 * kk, len(points))

    mask = mask.reshape(shape)
    raw_count = int(mask.sum())
    if not raw_count:
        raise ValueError("No grid node has point support; decrease spacing_scale.")
    if np.any(opening):
        mask = _morph_volume(_morph_volume(mask, opening, spacing, "erode"),
                             opening, spacing, "dilate")
    if np.any(closing):
        mask = _morph_volume(_morph_volume(mask, closing, spacing, "dilate"),
                             closing, spacing, "erode")
    mask = _fill_volume_holes(mask, spacing, d * hole_scale)
    target = int(mask.sum())
    if not target:
        raise ValueError("Morphology removed all occupied voxels; reduce opening_scale.")
    phi0 = _volume_sdf(mask, spacing)
    field, base, level = phi0.copy(), phi0, np.float32(0)
    band = None if delta is None else (phi0 > -delta) & (phi0 <= delta)
    # Outside the eligible band, clamping fixes the sign. Match only its count.
    band_target = target if band is None else int(np.count_nonzero(mask & band))
    if np.any(sigma > 0) and delta != 0:
        for stage_index, stage in enumerate(scales):
            field = ndi.gaussian_filter(base, sigma * stage / spacing, mode="nearest")
            if preserve_volume:
                level = _volume_level(field if band is None else field[band], band_target)
                field -= level
            if delta is not None:
                np.clip(field, phi0 - delta, phi0 + delta, out=field)
            new_mask = field > 0
            if not new_mask.any():
                raise ValueError("Field smoothing removed the entire occupied domain.")
            if stage_index + 1 < len(scales) and not np.array_equal(new_mask, mask):
                base = _volume_sdf(new_mask, spacing)
            mask = new_mask
    smoothed_count = int(mask.sum())
    if offset_scale != 0:
        field = _volume_sdf(mask, spacing)
        field += d * offset_scale
    exterior = np.float32(-spacing.min())
    for axis in range(3):
        for edge in (0, -1):
            index = [slice(None)] * 3
            index[axis] = edge
            np.minimum(field[tuple(index)], exterior, out=field[tuple(index)])
    mask, removed = field > 0, 0
    if min_component > 0:
        component, count = ndi.label(mask)
        if count:
            sizes = np.bincount(component.ravel())
            sizes[0] = 0
            retained = sizes >= min_component * sizes.max()
            retained[0] = False
            cleaned = retained[component]
            discarded = mask & ~cleaned
            removed = int(discarded.sum())
            field[discarded] = -np.maximum(field[discarded], spacing.min() / 2)
            mask = cleaned
    if not mask.any():
        raise ValueError("No occupied voxels remain after smoothing/offset/filtering.")
    voxel_volume = float(np.prod(spacing))
    metadata = dict(base_spacing=d, voxel_count=size,
                    raw_volume=raw_count * voxel_volume,
                    post_morph_volume=target * voxel_volume,
                    smoothing_volume=smoothed_count * voxel_volume,
                    final_volume=int(mask.sum()) * voxel_volume,
                    level_shift=float(level), removed_voxels=removed,
                    smooth_scales=scales.copy())
    return mask, field, origin, spacing.copy(), metadata



def _label_volume(mask, origin, spacing, points, codes, confidence, d,
                  label_smooth, k, workers, *, label_max_iter=1000, label_tol=1e-5,
                  label_lock_seeds=False, label_smooth_weight=None,
                  label_axis_weights=None, chunk_size=50_000):
    """Soft joint partition of one occupied XYZ grid; background remains zero.

    Nearest-neighbour observations initialize class probabilities within each
    occupied component. The screened diffusion graph has only six-neighbour
    foreground edges, with physical regularization lengths d*label_smooth.
    Its positive fidelity is soft by default. Optional seed locks apply only to
    conflict-free occupied voxels with observations of confidence >=0.9.
    Symmetric Dirichlet elimination preserves the conjugate-gradient assumptions. Initial
    kNN evidence uses Euclidean distance and may cross a gap within a connected
    component; the subsequent graph diffusion never traverses background.
    Legacy weight/axis coefficients, when requested, replace physical lengths
    with weight*(axis/h**2)/(2*sum(axis/h**2)). Hard locks require prior cleaning
    of mistaken labels; conflicting observations at one voxel always stay soft.
    The score is relative assignment certainty, not geometric reliability.
    """
    from scipy.sparse.linalg import cg, LinearOperator

    mask = np.asarray(mask, dtype=bool)
    origin = np.asarray(origin, dtype=float)
    spacing = np.broadcast_to(np.asarray(spacing, dtype=float), (3,))
    smooth = np.broadcast_to(np.asarray(label_smooth, dtype=float), (3,))
    points = np.asarray(points, dtype=float)
    codes = np.asarray(codes, dtype=np.int64)
    confidence = np.asarray(confidence, dtype=float)
    if (mask.ndim != 3 or origin.shape != (3,) or
            np.any(spacing <= 0) or not np.isfinite(spacing).all() or
            np.any(smooth < 0) or not np.isfinite(smooth).all() or d <= 0):
        raise ValueError("Require a 3-D mask, positive spacing/d and nonnegative label_smooth.")
    if (points.ndim != 2 or points.shape[1] != 3 or
            codes.shape != (len(points),) or confidence.shape != (len(points),) or
            np.any(codes < 0) or not np.isfinite(points).all() or
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
            "label_residual": 0.0, "label_locked_voxels": 0, "label_hard_seed_voxels": 0,
            "label_conflict_voxels": 0, "label_seed_voxels": 0,
            "label_projected_points": 0, "label_unused_points": len(points),
            "label_ignored_observations": len(points), "label_components": 0,
            "label_nclasses": 0, "label_classes": np.empty(0, dtype=np.int64),
            "label_graph_weights": lengths2.copy()}

    components, n_components = ndi.label(mask)
    index = np.rint((points - origin) / spacing).astype(np.int64)
    in_bounds = np.all((index >= 0) & (index < np.asarray(mask.shape)), axis=1)
    membership = np.zeros(len(points), dtype=np.int32)
    membership[in_bounds] = components[tuple(index[in_bounds].T)]
    observation_voxel = np.full(len(points), -1, dtype=np.int64)
    observation_voxel[in_bounds] = np.ravel_multi_index(index[in_bounds].T, mask.shape)
    # Assign an out-of-mask observation once, within a bounded physical reach.
    missing = np.flatnonzero((membership == 0) & (confidence > 0))
    projected_count = 0
    if len(missing):
        occupied_xyz = np.column_stack(np.unravel_index(active, mask.shape))
        occupied_xyz = origin + occupied_xyz * spacing
        distance, nearest = cKDTree(occupied_xyz).query(
            points[missing], distance_upper_bound=max(float(d), np.linalg.norm(spacing)),
            workers=workers)
        reached = np.isfinite(distance)
        projected_count = int(reached.sum())
        observation_voxel[missing[reached]] = active[nearest[reached]]
        membership[missing[reached]] = components.ravel()[observation_voxel[missing[reached]]]
        del occupied_xyz
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
    max_iterations, max_residual = 0, 0.0
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
                steps = [0]
                def count_iteration(_):
                    steps[0] += 1
                options = dict(x0=initial[free], atol=0.0, maxiter=int(label_max_iter),
                               M=preconditioner, callback=count_iteration)
                try:
                    solution, status = cg(system, rhs, rtol=float(label_tol), **options)
                except TypeError:  # SciPy versions using tol instead of rtol.
                    solution, status = cg(system, rhs, tol=float(label_tol), **options)
                residual = float(np.linalg.norm(system @ solution - rhs) /
                                 max(np.linalg.norm(rhs), np.finfo(float).tiny))
                if status != 0:
                    raise RuntimeError(f"Label diffusion did not converge (CG status {status}, "
                                       f"relative residual {residual:.3g}); increase label_max_iter, "
                                       "reduce label_smooth, or coarsen spacing_scale.")
                probability[free] = solution
                max_iterations = max(max_iterations, steps[0])
                max_residual = max(max_residual, residual)
            probability = np.maximum(probability, 0.0)
            improve = probability > best
            winner[improve] = int(class_id) + 1
            best[improve] = probability[improve]
            normalizer += probability
        output.ravel()[voxels] = winner
        score.ravel()[voxels] = np.clip(best / np.maximum(normalizer, 1e-30), 0.0, 1.0)
    return output, score, {"label_observations": int(usable.sum()),
                           "label_ignored_observations": int((~usable).sum()),
                           "label_iterations": int(max_iterations),
                           "label_residual": float(max_residual),
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




def _sinc_points(points, faces, iterations, pass_band):
    """Filter a shared vertex table; caller supplies quads or triangles."""
    if iterations == 0 or not len(faces):
        return points.copy()
    import pyvista as pv
    from vtkmodules.vtkFiltersCore import vtkWindowedSincPolyDataFilter
    filt = vtkWindowedSincPolyDataFilter()
    filt.SetInputData(pv.PolyData(points, np.column_stack(
        (np.full(len(faces), faces.shape[1]), faces)).ravel()))
    filt.SetNumberOfIterations(int(iterations))
    filt.SetPassBand(float(pass_band))
    filt.NormalizeCoordinatesOn()
    filt.NonManifoldSmoothingOn()
    filt.FeatureEdgeSmoothingOff()
    filt.BoundarySmoothingOn()
    filt.Update()
    return np.asarray(pv.wrap(filt.GetOutput()).points).copy()



def _patch_mesh(points, triangles, pairs, selected, outer_normals, keys, code=None):
    """Split for ownership and shading; coordinates remain globally identical."""
    import pyvista as pv
    if not len(selected):
        return pv.PolyData()
    faces, sides = triangles[selected].copy(), pairs[selected]
    if code is not None:
        reverse = sides[:, 1] == code + 1
        faces[reverse] = faces[reverse, ::-1]
    external = sides[:, 1] == 0
    patch = np.where(external, 0, sides[:, 0] * (len(keys) + 1) + sides[:, 1])
    # Separate normal patches, including distinct internal interfaces meeting
    # at a junction. Repeated coordinates always keep the SAME global point ID.
    pairs_of_ids = np.column_stack((faces.ravel(), np.repeat(patch, 3)))
    vertex_patch, inverse = np.unique(pairs_of_ids, axis=0, return_inverse=True)
    global_ids = vertex_patch[:, 0]
    f = inverse.reshape(-1, 3)
    p = points[global_ids]
    mesh = pv.PolyData(p.copy(), np.column_stack((np.full(len(f), 3), f)).ravel())
    normals = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
    point_normals = np.zeros_like(p)
    for j in range(3):
        np.add.at(point_normals, f[:, j], normals)
    point_normals /= np.maximum(np.linalg.norm(point_normals, axis=1, keepdims=True), 1e-30)
    ext_points = vertex_patch[:, 1] == 0
    point_normals[ext_points] = outer_normals[global_ids[ext_points]]
    mesh.point_data['Normals'] = point_normals
    mesh.cell_data['Normals'] = normals / np.maximum(
        np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
    mesh.GetPointData().SetActiveNormals('Normals')
    mesh.GetCellData().SetActiveNormals('Normals')
    mesh.point_data['global_point_id'] = global_ids
    mesh.cell_data['global_face_id'] = np.asarray(selected, dtype=np.int64)
    mesh.cell_data['BoundaryLabels'] = sides.copy()
    if code is None:
        mesh.cell_data['label_id'] = sides[:, 0].astype(np.int32) - 1
        mesh.cell_data['label'] = np.asarray([str(keys[c - 1]) for c in sides[:, 0]])
    else:
        value = keys[code]
        value = str(value) if np.asarray(value).ndim or np.asarray(value).dtype.kind == 'O' else value
        mesh.point_data['label'] = np.full(mesh.n_points, value)
        mesh.point_data['label_id'] = np.full(mesh.n_points, code, dtype=np.int32)
        mesh.cell_data['label_id'] = np.full(mesh.n_cells, code, dtype=np.int32)
    return mesh



def _mesh_from_labels(labels, field, origin, spacing, keys,
                      mesh_smooth_iters=30, mesh_smooth_pass_band=0.1, *,
                      interface_backend="auto", interface_smooth_iters=None,
                      interface_max_shift_scale=None, d=1.0):
    """Return ``({original_label: PolyData}, common_outer_PolyData)``.

    Labels and signed field are XYZ arrays on the same grid. Internal IDs are
    0=background, 1..C=keys. All interfaces are extracted once, exact coincident
    vertices are welded, and exterior vertices first recover the continuous
    field's zero surface. Windowed Sinc smooths the global table and the complete
    exterior; exterior positions are synchronized before splitting. The second
    operation prevents internal junctions pulling grooves into the outer shell.
    interface_smooth_iters adds OPTIONAL internal-only smoothing after the global
    mesh filter; None/0 disables this extra step, never the mesh_smooth_* filter.
    interface_max_shift_scale*d limits ONLY fully internal vertex movement from
    extracted positions; exterior and outer-interface junction vertices are not
    capped. Every update is made before splitting, preserving identical interfaces.

    Split meshes duplicate vertices ONLY for independent normal patches. Equal
    global_point_id always means exactly equal coordinates. A/B triangles have
    identical geometry and reversed winding. No topology or thickness guarantee
    is implied for arbitrary label arrangements or excessive smoothing.
    """
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
        center = points[quads].mean(axis=1)
        normal = np.cross(points[quads[:, 1]] - points[quads[:, 0]],
                          points[quads[:, 2]] - points[quads[:, 0]])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
        before = np.floor(center - 0.25 * normal + 0.5).astype(np.int64)
        after = np.floor(center + 0.25 * normal + 0.5).astype(np.int64)
        pairs = np.column_stack((a[tuple(before.T)], a[tuple(after.T)]))
        if np.any(pairs[:, 0] == pairs[:, 1]):
            raise RuntimeError('Cannot resolve the two sides of a Surface Nets face.')
    flip = (pairs[:, 0] == 0) | ((pairs[:, 1] != 0) & (pairs[:, 0] > pairs[:, 1]))
    quads[flip], pairs[flip] = quads[flip, ::-1], pairs[flip, ::-1]
    points, inverse = np.unique(points, axis=0, return_inverse=True)
    quads = inverse[quads]
    points = origin + (points - 1) * spacing
    extracted_points = points.copy()
    external_quads = quads[pairs[:, 1] == 0]
    exterior_ids, exterior_inverse = np.unique(external_quads.ravel(), return_inverse=True)
    # Recover subvoxel geometry lost when the common smooth field was binarized.
    gradient = np.gradient(field.astype(np.float32, copy=False), *spacing)
    p = points[exterior_ids].copy()
    for _ in range(3):
        coord = ((p - origin) / spacing).T
        value = ndi.map_coordinates(field, coord, order=1, mode='nearest')
        grad = np.column_stack([ndi.map_coordinates(g, coord, order=1, mode='nearest')
                                for g in gradient])
        delta = value[:, None] * grad / np.maximum(np.sum(grad * grad, axis=1)[:, None], 1e-20)
        delta *= np.minimum(1, np.linalg.norm(spacing) /
                            np.maximum(np.linalg.norm(delta, axis=1), 1e-20))[:, None]
        p -= delta
    points[exterior_ids] = p
    smoothed = _sinc_points(points, quads, mesh_smooth_iters, mesh_smooth_pass_band)
    interior_ids = np.setdiff1d(np.unique(quads[pairs[:, 1] != 0]), exterior_ids)
    if interface_smooth_iters and len(interior_ids):
        relaxed = _sinc_points(smoothed, quads[pairs[:, 1] != 0],
                               interface_smooth_iters, mesh_smooth_pass_band)
        smoothed[interior_ids] = relaxed[interior_ids]
    if interface_max_shift_scale is not None and len(interior_ids):
        delta = smoothed[interior_ids] - extracted_points[interior_ids]
        multiplier = np.minimum(1.0, float(d * interface_max_shift_scale) /
                                np.maximum(np.linalg.norm(delta, axis=1), 1e-30))
        smoothed[interior_ids] = extracted_points[interior_ids] + multiplier[:, None] * delta
    smoothed[exterior_ids] = _sinc_points(p, exterior_inverse.reshape(-1, 4),
                                        mesh_smooth_iters, mesh_smooth_pass_band)
    triangles = np.concatenate((quads[:, (0, 1, 2)], quads[:, (0, 2, 3)]))
    pairs = np.concatenate((pairs, pairs)).astype(np.int32)
    if not np.isfinite(smoothed).all():
        raise RuntimeError('Shared surface smoothing produced nonfinite coordinates.')
    # Repair only genuinely collapsed triangles, never globally reject rotations
    # relative to the original staircase normals. IDs and faces stay shared.
    repaired = 0
    for _ in range(4):
        cross = np.cross(smoothed[triangles[:, 1]] - smoothed[triangles[:, 0]],
                         smoothed[triangles[:, 2]] - smoothed[triangles[:, 0]])
        bad = np.linalg.norm(cross, axis=1) <= 1e-12 * spacing.min()**2
        if not np.any(bad):
            break
        vertices = np.unique(triangles[bad])
        smoothed[vertices] = extracted_points[vertices]
        repaired += len(vertices)
    else:
        raise RuntimeError('Label extraction contains unresolved degenerate triangles.')
    if repaired:
        warnings.warn(f'Restored {repaired} vertex updates around collapsed triangles.', stacklevel=2)
    external = pairs[:, 1] == 0
    outer_faces = triangles[external]
    normals = np.cross(smoothed[outer_faces[:, 1]] - smoothed[outer_faces[:, 0]],
                       smoothed[outer_faces[:, 2]] - smoothed[outer_faces[:, 0]])
    outer_normals = np.zeros_like(smoothed)
    for j in range(3):
        np.add.at(outer_normals, outer_faces[:, j], normals)
    outer_normals /= np.maximum(np.linalg.norm(outer_normals, axis=1, keepdims=True), 1e-30)
    outer = _patch_mesh(smoothed, triangles, pairs, np.flatnonzero(external), outer_normals, keys)
    meshes = {key: _patch_mesh(smoothed, triangles, pairs,
               np.flatnonzero(np.any(pairs == code + 1, axis=1)), outer_normals, keys, code)
              for code, key in enumerate(keys)}
    for mesh in [outer, *meshes.values()]:
        mesh.field_data['mesh_smooth_iters'] = [int(mesh_smooth_iters)]
        mesh.field_data['mesh_smooth_pass_band'] = [float(mesh_smooth_pass_band)]
        mesh.field_data['interface_backend'] = [actual_backend]
        mesh.field_data['interface_smooth_iters'] = [int(interface_smooth_iters or 0)]
        if interface_max_shift_scale is not None:
            mesh.field_data['interface_max_shift_scale'] = [float(interface_max_shift_scale)]
    return meshes, outer



def _prepare_geometry(points, confidence, *, k=24,
                      outlier_method="lof", outlier_thresh=1.5,
                      outlier_k=None, outlier_iters=1, outlier_radius=None,
                      min_points=None, min_component=0.01, graph_scale=2.0,
                      radius_cap=1.6, base_spacing=None, base_spacing_scale=1.0,
                      adaptive=True, support_scale=1.5, anisotropy=1.0,
                      chunk_size=100_000, workers=-1, min_survivors=4, keep_largest=False):
    """Clean once and return ``retained, d, precision, axes, radii``.

    ``retained`` indexes the supplied finite unique XYZ coordinates. ``d`` is
    the supplied or estimated reference spacing times ``base_spacing_scale``.
    ``radii`` are scalar sampling radii AFTER confidence scaling and BEFORE
    WORLD XYZ support stretching or PCA shape changes. ``precision`` defines
    the support inequality ``(q-x).T @ precision @ (q-x) <= 1``. ``axes`` gives
    its true principal semiaxes, suitable for conservative sphere queries.

    Adaptive lengths use the retained cloud's kth nonself distance/cuberoot(k),
    multiplied by base_spacing_scale, then clipped to [0.6*d, radius_cap*d].
    An explicit base_spacing changes the reference and clipping bounds, not the
    measured physical local lengths. Adaptive=False uses the same d everywhere.
    Confidence multiplies every support semiaxis by c**(1/3).

    PCA shapes have unit determinant, and anisotropy caps their largest/smallest
    axis ratio BEFORE WORLD XYZ stretching by support_scale. The latter need
    not align with the local PCA axes. anisotropy=1 skips PCA entirely and, with
    default base spacing, reproduces the compact support geometry exactly.
    """
    import numpy as np
    from scipy.spatial import cKDTree

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
    return retained, d, precision, axes, radii




def _clean_labels(points, codes, confidence, label_confidence=None, *,
                  label_outlier_k=24, label_outlier_thresh=0.8,
                  label_outlier_iters=1, label_min_points=3,
                  label_outlier_action="relabel", graph_scale=2.0, workers=-1):
    """Correct unsupported label islands without removing any geometry.

    Parameters
    ----------
    points : (N, 3) finite array
        Already retained geometric observations; repeated coordinates are allowed.
    codes : (N,) nonnegative integer array
        Integer class IDs. IDs need not be contiguous.
    confidence : (N,) nonnegative finite array
        Geometric confidence, used to weight neighbours but never modified.
    label_confidence : (N,) array in [0, 1], optional
        Independent label evidence weights, default one. These weights do not
        alter the common occupied geometry or the point outlier filter.
    label_outlier_k : int, default 24
        Maximum number of distinct neighbouring coordinates used in voting.
    label_outlier_thresh : float in (0.5, 1], default 0.8
        Required weighted support for an alternative class. Its lead over the
        second class must additionally be at least 0.25. At least three distinct
        neighbour coordinates must contain evidence for the winning class.
    label_outlier_iters : int, default 1
        Synchronous rounds; zero disables cleanup.
    label_min_points : float, default 3
        Only same-class graph components SMALLER than this size are candidates.
        >=1 is an absolute number of distinct coordinates; (0,1) is a fraction
        of that class's total distinct coordinates. Zero disables cleanup.
        Connected thin sheets and small disconnected but locally coherent
        tissues are protected by this component test, regardless of local
        majority. An isolated true observation cannot be distinguished from a
        mislabel by geometry alone; use 1 or action='none' to preserve all.
    label_outlier_action : {'relabel', 'downweight', 'none'}, default 'relabel'
        Relabel only supported candidates, or retain their IDs and reduce their
        evidence by own-class support, or disable all cleanup. No point is ever
        deleted. Relabelled observations receive their winning support times
        their supplied label confidence.
    graph_scale : positive float, default 2
        Limit edges to graph_scale * min(local_spacing_i, local_spacing_j).
        local_spacing is median of up to six closest distinct coordinates,
        capped at twice the nearest-neighbour distance. This avoids long votes
        across gaps around small disconnected point groups.

    Returns
    -------
    corrected : (N,) int64 array
        Updated IDs aligned with the input observations.
    reliability : (N,) float64 array
        Label-only reliability, to multiply with geometric confidence when
        constructing label evidence. Non-candidates keep supplied reliability.
    report : dict
        ``changed_mask`` compares final IDs with input IDs; ``outlier_mask``
        records candidates encountered; ``support`` is leave-one-coordinate-out
        support for the original class in the first round. No input is mutated.

    Notes
    -----
    Voting excludes the entire query coordinate, including duplicate rows.
    Within each neighbouring coordinate, duplicate/conflicting observations are
    normalized to one class distribution; multiplicity cannot increase that
    coordinate's overall voting weight. Sparse matrices store class evidence:
    an N-by-number_of_classes dense array is never allocated. Repeated rounds
    reuse the fixed geometric graph and apply changes simultaneously.
    """
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
    target_column = np.flatnonzero(classes == target) if target is not None else None
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
            certainty[start:end] = matrix[:, target_column[0]].toarray().ravel() if len(target_column) else 0
    return winner, certainty



def _annotate_clay_mesh(mesh, points, source_ids, original_points, confidence, feature,
                        d, score_k, score_scale, chunk, workers):
    """Attach original-source mapping, sparse/dense features and geometry scores."""
    tree = cKDTree(points)
    nearest = tree.query(mesh.points, workers=workers)[1]
    mesh.point_data["source_id"] = source_ids[nearest]
    if score_k:
        k = min(score_k, len(points))
        bandwidth = d * score_scale
        score_tree = cKDTree(points / bandwidth)
        score = np.empty(mesh.n_points, np.float32)
        for start in range(0, mesh.n_points, chunk):
            sl = slice(start, start+chunk)
            distance, ids = score_tree.query(mesh.points[sl] / bandwidth,
                k=list(range(1,k+1)), workers=workers)
            score[sl] = np.mean(confidence[source_ids[ids]] * np.exp(-.5*distance**2), axis=1)
        mesh.point_data["score"] = score
    if feature is not None:
        k = min(8, len(points))
        width = 1 if feature.ndim == 1 else feature.shape[1]
        values = np.empty((mesh.n_points,) + feature.shape[1:], np.float64)
        batch = min(chunk, max(1, 1_000_000 // (k*width)))
        for start in range(0, mesh.n_points, batch):
            sl = slice(start, start+batch)
            dist, ids = tree.query(mesh.points[sl], k=list(range(1,k+1)), workers=workers)
            safe = np.maximum(dist, np.finfo(float).tiny)
            w = confidence[source_ids[ids]] * (safe[:, :1] / safe)**2
            exact = dist[:,0] == 0
            w[exact] = 0
            w[exact,0] = 1
            w /= w.sum(axis=1,keepdims=True)
            if sparse.issparse(feature):
                interpolation = sparse.csr_matrix((w.ravel(),
                    (np.repeat(np.arange(len(ids)),k),source_ids[ids].ravel())),
                    shape=(len(ids),len(original_points)))
                values[sl] = (interpolation @ feature).toarray()
            else:
                values[sl] = np.einsum('nk,nk...->n...',w,feature[source_ids[ids]])
        mesh.point_data["feature"] = values



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
    """Normalize cloud observations once and preserve original-row/class IDs.

    Arrays/tensors, PyVista datasets and paths readable by PyVista are accepted.
    String data arguments name dataset point_data arrays. Label is a 1-D array
    of hashable nonmissing categories. Confidence columns may also be sparse
    (N,1); features may be sparse (N,C) and are never globally densified.
    Confidence and label confidence are independent numbers in [0,1].

    Features remain attributes and never implicitly change confidence.
    Nonfinite XYZ and zero geometric confidence are excluded from geometry.
    Labels are factorized BEFORE exclusion in first-occurrence order, exactly
    once for all outputs and sub-label reconstructions.

    ``source`` contains all usable ORIGINAL row indices, including duplicate
    coordinates. ``unique_source`` represents unique geometry coordinates using
    their greatest geometric confidence, with first input row breaking ties.
    ``inverse`` maps rows of source to these unique sites; it preserves all
    duplicate label observations rather than choosing a single arbitrary label.
    No caller input is modified.
    """
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
    """Select geometry survivors, then clean only their label evidence.

    retained indexes data['unique_source']; a Boolean mask is also accepted.
    No geometry is constructed/deleted here. Every duplicate observation at a
    surviving coordinate remains represented. Returned point coordinates are
    WORLD coordinates; offset only recenters the temporary label-cleanup solve.
    Corrected codes/reliability/inlier/outlier arrays retain original row length.
    """
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
                     chunk_size=50_000):
    """Attach globally indexed labels to an existing mask without changing it.

    direction columns are orthonormal grid basis vectors in world coordinates;
    origin is world position of grid node (0,0,0). Points are transformed as
    (world-origin)@direction, and the joint graph uses physical grid-axis spacing.
    mode='joint' performs occupied-domain diffusion; 'project' uses independent
    weighted kNN evidence at each occupied node. This helper does not implement
    separate geometry: a caller reconstructing separate masks must keep the same
    observations['keys']/integer IDs and invoke it with that mask's observations.

    label_mask has dtype int32, background=0, foreground=global code+1. label_map
    contains {0: None, code+1: original_category}, including classes absent from
    this particular mask. label_score_image is the winning probability, float32
    and zero outside, when score=True. No dense voxel-by-class tensor is created.

    Optional point arrays are returned only when point_indices or point_points
    is supplied. Use flat indices or (M,3) grid indices, or point_points as
    world XYZ, to align them with another sampled output. World samples use the
    nearest grid node; outside/background samples have ID=0, label=None, score=0.
    """
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
            chunk_size=int(chunk_size))
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
    """Filter occupied components by relative, voxel-count and physical volume.

    All three cutoffs apply together. ``min_volume_component`` is a fraction
    of the largest component; ``reference_count`` can provide that largest size
    across several spatial blocks. A fraction of 1 retains equally large ties;
    ``keep_largest=True`` deterministically retains only one. Diameter means the
    volume of a sphere of diameter ``d*component_diameter_scale``. Connectivity
    is 6, 18 or 26 and applies only to foreground, never to cavity detection.
    Returns the filtered mask and integer/physical component diagnostics.
    """
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
                    offset_method="field"):
    """Apply shared morphology, smoothing and component controls to an XYZ mask.

    Returns ``(mask, positive_inside_field, metadata)``. Lengths in smooth_sigma
    and max_shift are physical; XYZ opening/closing scales are DIAMETERS times d,
    hole_scale is an equivalent-sphere cavity diameter times d, and offset_scale
    is a signed distance times d. Explicit physical and relative shift controls
    conflict if both are supplied. ``occupancy`` is an alias of ``density``;
    smooth_method=None disables field smoothing. Foreground connectivity can be
    6/18/26, while holes always use 26-connected background.

    initial_field must have positive sign exactly on the supplied mask. When
    smoothing is inactive and morphology does not change occupancy, it is kept
    exactly. Component removal only updates removed positive nodes, preserving
    retained continuous surfaces. Changed morphology or an explicit nonzero
    offset re-embeds its mask to obtain a consistent physical distance field.
    The grid must be padded by the caller for requested morphology/smoothing.

    forbidden excludes reliable foreign-label support throughout processing;
    its presence vetoes filling the whole enclosing background component. The
    volume-preserving field target is measured AFTER prefiltering/morphology.
    Empty results are returned normally for the caller to handle explicitly.
    Legacy adapters may request hole_stage='after_offset' or binary-morphology
    offset_method='morphology'; the canonical defaults use early cavity filling
    and an explicit offset of the physical field.
    """
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
                method, density_level, preserve_volume, shift, device, forbidden=blocked)
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
        smooth_method="none" if method is None else method,
        max_shift=np.inf if shift is None else float(shift),
        **components)
    return current, field, metadata














def _grid_support(points, confidence, tree, radii, origin, spacing, shape,
                  axis_scale, method, k, bandwidth, density_level, chunk, workers):
    """Original kNN/KDE supports plus confidence-aware exact nearest supports."""
    if method == "kde":
        density = np.zeros(shape, dtype=np.float32)
        for start in range(0, len(points), chunk):
            sl = slice(start, start + chunk)
            q = (points[sl] * axis_scale - origin) / spacing
            ijk = np.floor(q).astype(np.int64)
            frac = q - ijk
            for offset in np.ndindex(2, 2, 2):
                offset = np.asarray(offset)
                weights = confidence[sl] * np.prod(np.where(offset, frac, 1 - frac), axis=1)
                np.add.at(density, tuple((ijk + offset).T), weights)
        ndi.gaussian_filter(density, bandwidth / spacing, output=density,
                            mode="constant", truncate=3.0)
        sampled = np.empty(len(points), np.float32)
        for start in range(0, len(points), chunk):
            sl = slice(start, start + chunk)
            q = (points[sl] * axis_scale - origin) / spacing
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
    max_voxels=2_000_000_000, chunk_size=100_000, workers=-1,
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
    component_diameter_scale=0.0, score_k=8, score_scale=2.0,
    _foreign_points=None,
):
    """Reconstruct a regular occupied grid while retaining all legacy methods.

    Geometry: ``method='knn'`` uses adaptive RMS-neighbour support, ``nearest``
    uses fixed supports (scaled per point by confidence), and ``kde`` deposits
    confidence-weighted points before Gaussian filtering. ``axis_scale`` changes
    the metric; columns of ``direction`` are orthonormal world grid axes.
    ``confidence``/``geometry_confidence`` are aliases and affect geometry;
    ``label_confidence`` controls class evidence only. Attributes in ``feature``
    are interpolated to occupied nodes when return_info=True.

    Canonical morphology scales are physical DIAMETERS in reference-length units
    along the grid axes. Zero XYZ entries forbid exchange along those axes.
    Canonical order is opening -> closing -> enclosed-hole filling -> smoothing
    -> signed offset -> component filtering. ``hole_scale=0`` disables explicit
    filling; inf fills all enclosed cavities, never exterior-connected tunnels.
    Legacy ``closing/opening`` retain metric RADII and close -> open order;
    their respective canonical aliases cannot be supplied together. A legacy
    fill_holes=True is equivalent to hole_scale=inf. min_component continues to
    alias POINT min_points; min_volume_component filters final occupied voxels.
    component_scale continues to alias graph_scale, not voxel component size.

    ``smooth_scales`` are successive re-embedded stages, multiplied by the
    resolved sigma and sqrt(field_smooth_iters). ``smooth_method`` remains sdf,
    occupancy, or None. preserve_volume targets post-morphology occupancy.
    Legacy max_shift=None selects the old automatic physical band. Explicit
    max_shift_scale=None selects an unbounded field; a finite scale multiplies
    reference length. Zero disables field smoothing. This constrains the field,
    not exact surface displacement/topology. Positive offset expands geometry.

    Label modes share class codes in first input occurrence order. project
    projects corrected evidence to common occupied nodes; joint diffuses labels
    only within the common domain. Both keep the original output type and put
    label_mask (background0, class+1), label_values, label_id/label/assignment
    confidence in info (point_label_id also uses background0/class+1). separate returns {label: legacy_output} after independent
    reconstruction; with return_info also {label: info}. Joint mesh output is
    the ordinary outer (vertices,faces), not internal per-label closed meshes;
    label_mask provides the shared internal partition for subsequent extraction.

    Without labels, output types are unchanged: volume/boundary -> (M,3) world
    points, mask -> bool XYZ image, mesh -> (vertices,faces). return_info wraps
    (result,info); return_field adds a NEGATIVE-inside implicit field to info.
    score_image/score are geometric proximity scores, not calibrated probability;
    label_score_image is class assignment certainty, independent of score.
    """
    input_parameters = locals().copy()
    smooth_sigma = _alias(smooth_sigma, smooth, None, "smooth_sigma", "smooth")
    min_points = _alias(min_points, min_component, 0.01, "min_points", "min_component")
    graph_scale = _alias(graph_scale, component_scale, 2.0, "graph_scale", "component_scale")
    backend = _alias(backend, mc_method, "auto", "backend", "mc_method")
    confidence = _alias(confidence, geometry_confidence, None, "confidence", "geometry_confidence")
    if method not in {"knn", "nearest", "kde"} or output not in {"volume", "boundary", "mask", "mesh"}:
        raise ValueError("Unknown reconstruction method or output.")
    if smooth_method not in {"sdf", "occupancy", None}:
        raise ValueError("smooth_method must be 'sdf', 'occupancy' or None.")
    if label_mode not in {"project", "joint", "separate"}:
        raise ValueError("label_mode must be 'project', 'joint' or 'separate'.")
    if backend not in {"auto", "pymcubes", "skimage", "torchmcubes", "lewiner", "lorensen"}:
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
    if data["keys"] is not None and label_mode == "separate":
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
            else:
                parameters = input_parameters.copy()
                parameters.pop("points", None)
                parameters.update(label=np.full(len(rows), key, dtype=object),
                    label_mode="project", confidence=data["confidence"][rows] * observations["label_reliability"][rows], geometry_confidence=None,
                    label_confidence=observations["label_reliability"][rows],
                    feature=None if data["feature"] is None else data["feature"][rows],
                    label_outlier_action="none", outlier_iters=0,
                    min_points=0, min_component=None, keep_largest=False, return_info=True)
                if separate_protect_labels and label_guard_scale > 0:
                    foreign = ((observations["codes"] != code) &
                               (observations["weights"] >= 0.5))
                    parameters["_foreign_points"] = (
                        observations["observation_points"][foreign], step * label_guard_scale)
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
    radii *= support_scale * np.cbrt(geom_conf)
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
    mask = _grid_support(metric, geom_conf, tree, radii, origin_local, spacing, shape,
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
        morph_order=morph_order, component_connectivity=6, forbidden=forbidden, device=device)
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
    if data["keys"] is not None and (return_info or label_mode == "joint"):
        labeled = _grid_label_data(mask, info["origin"], spacing, observations,
            direction=direction, mode=label_mode, d=step, label_smooth=label_smooth,
            k=label_project_k if label_mode == "project" else support_k,
            workers=workers, point_indices=indices,
            point_points=output_points if indices is None else None,
            label_max_iter=label_max_iter, label_tol=label_tol, label_lock_seeds=label_lock_seeds,
            label_smooth_weight=label_smooth_weight, label_axis_weights=label_axis_weights,
            chunk_size=chunk_size)
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
        return result, info
    return result

def _grid_scores_features(mask, origin, spacing, axis_scale, points, confidence,
                          data, observations, d, k, scale, chunk, workers):
    """Batch score/attribute interpolation; duplicate attributes are averaged."""
    feature = data["feature"]
    averaged = None
    if feature is not None:
        rows, sites = observations["observations"], observations["obs_site"]
        weights = data["confidence"][rows]
        total = np.bincount(sites, weights=weights, minlength=len(points))
        average = sparse.csr_matrix((weights / total[sites], (sites, np.arange(len(rows)))),
                                    shape=(len(points), len(rows)))
        averaged = average @ feature[rows]
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



