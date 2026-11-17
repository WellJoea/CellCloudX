"""Occupied-volume point reconstruction, slice interpolation, image and mesh I/O.

Required: numpy, scipy >= 1.6.
Mesh extraction: PyMCubes or scikit-image; backend="auto" prefers PyMCubes.
clay_mesh/clay_slice_mesh also require PyVista (imported only when called).
Optional: torch for CUDA Gaussian filtering; torchmcubes for CUDA extraction.

Input points sample occupied tissue VOLUME, not just its outer surface.
No method can infer from coordinates alone whether an empty region is missing
observations or a true anatomical cavity. Scores measure geometric observation
support, not calibrated probabilities. No implicit global cavity filling.

Common scales
-------------
spacing: physical grid intervals; overrides spacing_scale.
base_spacing: reference sampling length; fixes automatic scale parameters.
support_scale: dimensionless support expansion; exact support model varies.
smooth_scale: dimensionless Gaussian width; smooth_sigma overrides it physically.
mesh_smooth_iters: final Taubin iterations (two clay functions only).
points_to_image smooth_iters is field diffusion time, a different operation.
outlier_thresh=0 disables point outlier rejection in all reconstruction functions.
Threshold values are specific to each outlier algorithm, not interchangeable.
backend: auto, pymcubes, skimage, lewiner, lorensen, torchmcubes.
All image arrays are XYZ; direction COLUMNS give world-space grid axes.
Legacy keyword aliases are accepted; supplying both names raises ValueError.
Mesh label_id preserves category identity in first-occurrence input order,
including object categories with identical display strings but different types.
Missing labels (None/NaN/NA) are rejected; no background category is invented.

Examples
--------
    image = points_to_image(points, spacing_scale=0.5,
                            support_scale=(1.5,1.5,2),
                            smooth_scale=(1,1,2), outlier_thresh=1.8)
    mask, score_image = image["mask"], image["score_image"]
    dense, score = image["points"], image["score"]
    image_to_nrrd(image, "mask.nrrd")
    image_to_nrrd(image, "score.nrrd", key="score_image")
    vertices, faces = grid_to_mesh(image, backend="auto")

    grid, info = grid_point_cloud(points, spacing_scale=0.5,
                                  smooth_scale=(1,1,2),
                                  return_info=True, return_field=True)
    vertices, faces = grid_to_mesh(info)  # reuse field, avoid reconstructing grid
    dense = grid_point_cloud(points, spacing=info["spacing"]/2,
                             base_spacing=info["base_spacing"],
                             smooth_sigma=info["smooth_sigma"])

    mesh = clay_mesh(points, spacing_scale=0.5, support_scale=2,
                     smooth_scale=(1,1,2), mesh_smooth_iters=10)
    stack_mesh = clay_slice_mesh(points, slice_axis=2,
                                spacing_scale=0.6, smooth_scale=(1,1,1.5))
    # slice_axis applies only to exact parallel planes in current coordinates.
    # Warped/combined clouds can use clay_mesh without slice assumptions.

Notes
-----
Explicit spacing is never silently coarsened. clay_slice_mesh alone can coarsen
an automatically selected grid, with a warning. Default max_voxels=2_000_000_000
limits grid nodes, not total RAM: EDT, fields and generated points need more.
Grid-only and image-only workflows do not require PyVista. CPU algorithms retain
SciPy trees/EDT; device="cuda" does not make the full pipeline GPU-native.

Algorithm and format references
-------------------------------
https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.marching_cubes
https://github.com/pmneila/PyMCubes
https://teem.sourceforge.net/nrrd/format.html
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



def _denoise(points, groups, levels, axis, outlier, max_gap, k, base_spacing=None):
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
            points[ids][:, plane], k=[ki + 1], workers=-1
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
            distance = trees[neighbor].query(points[ids][:, plane], workers=-1)[0]
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
    if backend not in {"auto", "pymcubes", "skimage", "torchmcubes", "lewiner", "lorensen"}:
        raise ValueError("Unknown marching-cubes backend.")
    if backend == "auto":
        try:
            import mcubes
        except ImportError:
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
                 origin, step, shape, d, sigma, radius, outlier, max_gap, method):
    """Shared physical-z interpolation; observations are occupancy or 2D SDF."""
    plane = [a for a in range(3) if a != axis]
    xy_shape = tuple(shape[plane])
    zz = origin[axis] + np.arange(shape[axis]) * step[axis]
    h = float(np.median(np.diff(levels)))
    band = max(4 * radius * d, 3 * float(sigma.max()), 4 * float(step.max()))
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
                distance, nearest = cKDTree(points[ids][:, plane]).query(xy, workers=-1)
                mask = (distance <= radius * d).reshape(xy_shape)
                field = ndi.distance_transform_edt(mask, sampling=step[plane])
                field -= ndi.distance_transform_edt(~mask, sampling=step[plane])
                values[j] = np.clip(field.ravel(), -band, band)
                weight[j] = np.where(distance <= 2 * radius * d,
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



def _volume_field(points, confidence, origin, step, shape, d, radius, method):
    """Non-planar data: weighted occupancy or the SDF of local point support."""
    if method == "density":
        grid = np.zeros(tuple(shape), np.float32)
        ijk = np.rint((points - origin) / step).astype(int)
        flat = np.ravel_multi_index(ijk.T, tuple(shape))
        np.maximum.at(grid.ravel(), flat, confidence)
        return grid
    mask = _support_mask_3(points, confidence ** (1.0 / 3), np.full(3, radius * d),
                           origin, step, tuple(shape), 100_000, -1)
    field = ndi.distance_transform_edt(mask, sampling=step)
    field -= ndi.distance_transform_edt(~mask, sampling=step)
    return field.astype(np.float32)



def clay_slice_mesh(points=None, label=None, feature=None, *, method="sdf", confidence=None,
              label_mode="project", slice_axis=2, spacing=None, spacing_scale=0.6,
              base_spacing=None, smooth_scale=None, smooth_sigma=None,
              outlier_thresh=None, outlier_k=16, max_gap=None, support_scale=None,
              density_level=0.30, min_component=0.0, mesh_smooth_iters=None,
              backend="auto", device="cpu", max_voxels=2_000_000_000,
              X=None, voxel_size=None, smooth=None, outlier=None, radius=None,
              smooth_iters=None):
    """Build one PolyData, or {label: PolyData}, from points inside tissue.

    points : (N,3) array/tensor, PyVista dataset, or readable point-cloud path
        Coordinates must use the same physical unit on all axes. This is a
        VOLUME-sampling model, not a reconstruction of boundary-only samples.
    label, feature, confidence : arrays or points.point_data names, optional
        label: hashable, nonmissing (N,) categories; None/NaN/NA are rejected.
        feature: finite real (N,) or (N,C), mapped to mesh
        vertices by confidence-weighted inverse squared distance, k<=8.
        confidence: reliability in [0,1]; zero removes a point. Density mode
        uses it as occupancy amplitude; SDF sections use it in regression.
    method : 'sdf' or 'density'
        sdf: point support -> SDF -> shape smoothing -> zero surface.
        density: smooth occupancy -> relative density threshold -> SDF.
        Both share denoising, physical scales, gaps, meshing and data mapping.
    label_mode : 'project' or 'separate'
        Default project returns one mesh with nearest-point labels. Separate
        returns a dict. A label absent from a globally observed section ends
        that label's interpolation segment. Labels with <8 retained points
        are skipped with a warning. Independently built label meshes may overlap.
    slice_axis : 0, 1, 2 or None
        Default 2. Exact planes along this axis enable section-aware processing.
        If fewer than two planes, or <8 points per plane on average, use the
        general 3D branch with a warning. None explicitly selects general 3D.
        Non-planar coordinates are not flattened; no local slice directions
        are inferred after nonlinear warping or combining differently oriented stacks.
    smooth_scale : scalar or (sx,sy,sz), default (1,1,1.5)
        Dimensionless XYZ multipliers. Physical sigma = smooth * axis_spacing.
        axis_spacing=(d,d,h) for z sections, with h on the selected slice axis;
        general 3D uses (d,d,d). d uses the shared kNN estimate, h the median
        OBSERVED interval. Zero disables main smoothing on that axis; sections
        then use linear interpolation in z. A final half-voxel Gaussian removes
        grid artifacts on enabled axes. Refining the grid preserves main sigma.
    smooth_sigma : scalar or (sx,sy,sz), optional
        Physical XYZ Gaussian sigma, overriding smooth_scale on all axes.
    spacing : scalar or (dx,dy,dz), optional
        Physical grid spacing. Auto uses spacing_scale*d, limited to h/4 on the section
        axis, and coarsens to max_voxels. An explicit grid exceeding the limit
        raises MemoryError instead of silently changing requested resolution.
    spacing_scale : positive scalar or XYZ vector, default 0.6
        Dimensionless auto-grid spacing / d; ignored when spacing is supplied.
    base_spacing : positive float, optional
        Fixed physical point-sampling scale d, useful after point densification.
        Slice-axis smoothing still uses observed section spacing h.
    outlier_thresh : float, default 3
        Shared kNN median/MAD cutoff and cross-section support cutoff. Smaller
        positive values reject more; zero disables point filtering and SDF
        residual downweighting. Not numerically equivalent to old outlier_ratio.
    outlier_k : int, default 16
        Number of neighbors for sampling-scale estimation and local denoising.
    max_gap : float, optional
        Maximum PHYSICAL separation between observed sections to bridge;
        default 3*h. Larger gaps split interpolation. Ignored in general 3D.
        Gap/features smaller than grid resolution cannot be reliably resolved.
    support_scale : float, default 2
        SDF ONLY: initial point-support radius / d. Larger values close small
        sampling gaps, thicken tissue and narrow holes. Ignored by density.
    density_level : float in (0,1), default .30
        DENSITY ONLY: threshold / median smoothed density at retained points.
        Smaller values expand tissue; larger values contract it. Ignored by sdf.
    min_component : float in [0,1], default 0
        Minimum occupied-voxel component size / largest component size.
        0 retains all; 1 keeps only the largest occupied VOLUME, preserving its
        inner cavity surface. Absolute voxel counts are not used in this API.
    mesh_smooth_iters : int, default 10
        Final Taubin iterations, pass_band=.1. Main smoothing is in the field.
    backend : 'auto', 'pymcubes', 'skimage', 'lewiner', 'lorensen' or 'torchmcubes'
        Auto prefers PyMCubes. All backends use the same physical field, origin
        and spacing. No extra mcubes.smooth is applied.
    device : str, default 'cpu'
        CUDA uses torch Gaussian filtering, and torchmcubes when explicitly
        selected. Trees, EDT, axial regression, PyMCubes, skimage and PyVista
        remain on CPU; this is not an end-to-end GPU pipeline.
    max_voxels : int, default 2_000_000_000
        Limit for the output grid AND observed-section arrays. This is not a
        byte-level memory limit: several temporary arrays are also required.

    Legacy X, voxel_size, smooth, outlier, radius and smooth_iters keywords are
    accepted; supplying both an alias and its canonical keyword raises an error.
    Mesh point_data stores 'label'/'feature'; field_data records method, actual
    grid spacing, axis sampling spacing, physical smoothing sigma and mode.
    No global hole filling is used. Small holes may still disappear under
    support expansion or smoothing. Dense coherent outliers remain ambiguous.
    """
    import pyvista as pv

    points = _alias(points, X, None, "points", "X")
    spacing = _alias(spacing, voxel_size, None, "spacing", "voxel_size")
    smooth_scale = _alias(smooth_scale, smooth, (1.0, 1.0, 1.5), "smooth_scale", "smooth")
    outlier_thresh = _alias(outlier_thresh, outlier, 3.0, "outlier_thresh", "outlier")
    support_scale = _alias(support_scale, radius, 2.0, "support_scale", "radius")
    mesh_smooth_iters = _alias(mesh_smooth_iters, smooth_iters, 10, "mesh_smooth_iters", "smooth_iters")
    if isinstance(points, (str, Path)):
        points = pv.read(points)
    source = points if hasattr(points, "points") else None
    arrays = [label, feature, confidence]
    if source is None and any(isinstance(a, str) for a in arrays):
        raise ValueError("Named label/feature/confidence arrays require a PyVista input dataset.")
    arrays = [np.asarray(source.point_data[a] if isinstance(a, str) else
                         a.detach().cpu().numpy() if hasattr(a, "detach") else a)
              if a is not None else None for a in arrays]
    label, feature, confidence = arrays
    raw = source.points if source is not None else points
    points = np.asarray(raw.detach().cpu().numpy() if hasattr(raw, "detach") else raw,
                        dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 8:
        raise ValueError("points must contain at least eight 3D points.")
    if method not in {"sdf", "density"}:
        raise ValueError("method must be 'sdf' or 'density'.")
    if label_mode not in {"project", "separate"} or (label_mode == "separate" and label is None):
        raise ValueError("Use label_mode='project', or 'separate' with label supplied.")
    if slice_axis not in (None, 0, 1, 2):
        raise ValueError("slice_axis must be 0, 1, 2, or None.")
    for a in arrays:
        if a is not None and (a.ndim == 0 or len(a) != len(points)):
            raise ValueError("label, feature and confidence must align with points.")
    if label is not None and label.ndim != 1:
        raise ValueError("label must be one-dimensional.")
    label_keys, label_codes = ([], None) if label is None else _label_codes(label)
    if feature is not None and (feature.ndim not in (1, 2) or
                                (feature.ndim == 2 and feature.shape[1] == 0) or
                                not np.issubdtype(feature.dtype, np.number)
                                or np.iscomplexobj(feature) or not np.isfinite(feature).all()):
        raise ValueError("feature must be a finite real (N,) or (N,C), C>=1, array.")
    confidence = np.ones(len(points)) if confidence is None else np.asarray(confidence, float)
    if confidence.ndim != 1 or not np.isfinite(confidence).all() or np.any(
            (confidence < 0) | (confidence > 1)):
        raise ValueError("confidence must be finite and in [0,1].")
    multipliers = np.broadcast_to(np.asarray(smooth_scale, float), (3,)).copy()
    outlier, radius, smooth_iters = outlier_thresh, support_scale, mesh_smooth_iters
    if not np.isfinite(multipliers).all() or np.any(multipliers < 0) or not (
            np.isfinite(outlier) and outlier >= 0 and np.isfinite(outlier_k) and outlier_k >= 2):
        raise ValueError("smooth_scale/outlier_thresh must be nonnegative and outlier_k >=2.")
    if method == "sdf" and not (np.isfinite(radius) and radius > 0):
        raise ValueError("support_scale must be finite and positive for method='sdf'.")
    if method == "density" and not 0 < density_level < 1:
        raise ValueError("density_level must be in (0,1) for method='density'.")
    if (not 0 <= min_component <= 1 or not np.isfinite(max_voxels) or max_voxels < 1000
            or not np.isfinite(smooth_iters) or smooth_iters < 0
            or int(smooth_iters) != smooth_iters or int(outlier_k) != outlier_k
            or int(max_voxels) != max_voxels):
        raise ValueError("Require 0<=min_component<=1, max_voxels>=1000 and nonnegative integer mesh_smooth_iters.")
    retained = np.flatnonzero(np.isfinite(points).all(axis=1) & (confidence > 0))
    retained = retained[np.argsort(-confidence[retained], kind="stable")]
    _, unique = np.unique(points[retained], axis=0, return_index=True)
    retained = retained[np.sort(unique)]
    points = points[retained]
    if len(points) < 8:
        raise ValueError("Fewer than eight distinct finite reliable points remain.")
    offset = np.median(points, axis=0)
    points = points - offset
    axis = slice_axis
    levels, groups = np.array([0.0]), np.zeros(len(points), dtype=int)
    if axis is not None:
        levels, groups = np.unique(points[:, axis], return_inverse=True)
        if not (2 <= len(levels) <= len(points) // 8):
            warnings.warn("No exact planar stack detected; using general 3D. "
                          "All smooth multipliers then use point spacing d.", stacklevel=2)
            axis = None
            levels, groups = np.array([0.0]), np.zeros(len(points), dtype=int)
    h = float(np.median(np.diff(levels))) if axis is not None else 0.0
    gap = 3 * h if max_gap is None else float(max_gap)
    if axis is not None and not (np.isfinite(gap) and gap > 0):
        raise ValueError("max_gap must be finite and positive.")
    original_groups = groups.copy()
    original_label_codes = label_codes[retained] if label is not None else None
    if base_spacing is not None:
        base_spacing = float(base_spacing)
        if not np.isfinite(base_spacing) or base_spacing <= 0:
            raise ValueError("base_spacing must be finite and positive.")
    keep, d = _denoise(points, groups, levels, axis, outlier, gap, outlier_k, base_spacing)
    points, groups, retained = points[keep], groups[keep], retained[keep]
    if len(points) < 8:
        raise ValueError("Fewer than eight points remain; increase outlier_thresh or set it to zero.")
    confidence = confidence[retained]
    label_codes = label_codes[retained] if label is not None else None
    label = label[retained] if label is not None else None
    feature = feature[retained] if feature is not None else None
    scale = np.full(3, d)
    if axis is not None:
        scale[axis] = h
    sigma = multipliers * scale if smooth_sigma is None else np.broadcast_to(
        np.asarray(smooth_sigma, float), (3,)).copy()
    if not np.isfinite(sigma).all() or np.any(sigma < 0):
        raise ValueError("smooth_sigma must be finite and nonnegative.")
    automatic_grid = spacing is None
    requested_step = np.broadcast_to(np.asarray(np.asarray(spacing_scale) * d if automatic_grid else
                                                spacing, float), (3,)).copy()
    if automatic_grid and axis is not None:
        requested_step[axis] = min(requested_step[axis], 0.25 * h)
    if not np.isfinite(requested_step).all() or np.any(requested_step <= 0):
        raise ValueError("spacing and spacing_scale must give finite positive grid spacing.")
    support = radius if method == "sdf" else 2.0

    def build(ids, key=None, code=None):
        p, w = points[ids], confidence[ids]
        step = requested_step.copy()
        # Three padding cells per side leave at least 8*8 nodes per plane.
        if axis is not None and 64 * len(levels) > max_voxels:
            raise MemoryError("max_voxels cannot hold the observed-section arrays, even after coarsening.")
        for attempt in range(40):
            margin = max(2.0, support) * d + 3 * sigma + 3 * step
            if axis is not None:
                margin[axis] += h
            origin, upper = p.min(axis=0) - margin, p.max(axis=0) + margin
            if axis is not None:
                origin[axis], upper[axis] = levels[0] - margin[axis], levels[-1] + margin[axis]
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                extent = (upper - origin) / step
            if (not np.isfinite(extent).all() or np.any(extent < 0)
                    or np.any(extent > np.iinfo(np.intp).max // 2)):
                raise MemoryError("Grid dimensions overflow; increase spacing or rescale coordinates.")
            shape = np.ceil(extent).astype(np.intp) + 1
            storage = math.prod(shape.tolist())
            if axis is not None:
                storage = max(storage, math.prod(np.delete(shape, axis).tolist()) * len(levels))
            if storage <= max_voxels:
                break
            if not automatic_grid:
                raise MemoryError("Requested grid exceeds max_voxels; increase spacing or max_voxels.")
            step *= max(1.05, (storage / max_voxels) ** (1 / 3))
        else:
            raise MemoryError("Auto coarsening cannot meet max_voxels; increase the limit or use fewer sections.")
        if np.any(step > requested_step * 1.001):
            warnings.warn(f"Auto grid coarsened to spacing={tuple(step)}.", stacklevel=3)
        slabs = []
        # Density thresholds are relative; normalize amplitudes before float32 deposition.
        field_confidence = w / w.max() if method == "density" else w
        if axis is None:
            field = _volume_field(p, field_confidence, origin, step, shape, d, support, method)
        else:
            known_empty = np.zeros(len(levels), dtype=bool)
            if label_mode == "separate":
                known_empty = np.bincount(original_groups[original_label_codes == code],
                                          minlength=len(levels)) == 0
            field, slabs = _stack_field(
                p, field_confidence, groups[ids], levels, known_empty, axis, origin, step, shape,
                d, sigma, support, outlier, gap, method
            )
        blur = sigma / step
        if axis is not None:
            blur[axis] = 0  # Physical-z smoothing was already done by the sparse kernel.
        field = _gaussian(field, blur, device)
        reference = None
        if method == "density":
            sample = ndi.map_coordinates(field, ((p - origin) / step).T,
                                         order=1, mode="nearest", prefilter=False)
            reference = float(np.median(sample))
            if not np.isfinite(reference) or reference <= 0:
                raise ValueError("Empty density field; use a finer grid or larger smooth_sigma.")
            mask = field >= density_level * reference
            field = ndi.distance_transform_edt(mask, sampling=step)
            field -= ndi.distance_transform_edt(~mask, sampling=step)
            field = field.astype(np.float32)
        field = _gaussian(field, np.where(sigma > 0, 0.5, 0.0), device)
        if slabs:
            zz = origin[axis] + np.arange(shape[axis]) * step[axis]
            cap = np.full(len(zz), -np.inf)
            for lo, hi in slabs:
                cap = np.maximum(cap, np.minimum(zz - lo, hi - zz))
            view = [1, 1, 1]
            view[axis] = len(zz)
            np.minimum(field, cap.reshape(view), out=field)
        # Filter occupied VOLUMES, so a cavity's inner surface is retained.
        removed_components = 0
        if min_component > 0:
            regions, number = ndi.label(field > 0)
            if number:
                counts = np.bincount(regions.ravel())
                largest = 1 + np.argmax(counts[1:])
                good = counts >= min_component * counts[largest]
                good[0] = False
                if min_component == 1:
                    good[:] = False
                    good[largest] = True
                removed_components = int(number - good.sum())
                remove = (regions > 0) & ~good[regions]
                field[remove] = -max(d, float(step.max()))
        for a in range(3):
            border = [slice(None)] * 3
            border[a] = [0, -1]
            field[tuple(border)] = -max(d, float(step.max()))
        if not (field.min() < 0 < field.max()):
            raise ValueError(f"No surface for label {key!r}; reduce smooth_sigma or adjust support_scale.")
        vertices, faces = _marching_cubes(field, origin, step, backend, device)
        mesh = pv.PolyData(vertices, np.c_[np.full(len(faces), 3), faces]).clean()
        if smooth_iters:
            mesh = mesh.smooth_taubin(n_iter=int(smooth_iters), pass_band=0.1,
                                      normalize_coordinates=True)
        mesh = mesh.compute_normals(consistent_normals=True, auto_orient_normals=False)
        if feature is not None or label is not None:
            k = min(8, len(p)) if feature is not None else 1
            distances, neighbors = cKDTree(p).query(
                mesh.points, k=list(range(1, k + 1)), workers=-1
            )
            if label is not None:
                projected = label[ids][neighbors[:, 0]]
                mesh.point_data["label_id"] = label_codes[ids][neighbors[:, 0]]
                mesh.point_data["label"] = projected.astype(str) if projected.dtype.kind == "O" else projected
            if feature is not None:
                f = feature[ids]
                safe = np.maximum(distances, np.finfo(float).tiny)
                factors = w[neighbors] * (safe[:, :1] / safe) ** 2
                exact = distances[:, 0] == 0
                factors[exact] = 0
                factors[exact, 0] = 1
                factors /= factors.sum(axis=1, keepdims=True)
                values = np.empty((mesh.n_points,) + f.shape[1:], dtype=np.float64)
                chunk = max(1, 2_000_000 // (k * (1 if f.ndim == 1 else f.shape[1])))
                for start in range(0, mesh.n_points, chunk):
                    sl = slice(start, start + chunk)
                    values[sl] = np.einsum("nk,nk...->n...", factors[sl], f[neighbors[sl]])
                mesh.point_data["feature"] = values
        mesh.points = np.asarray(mesh.points, np.float64) + offset
        mesh.field_data["method"] = [method]
        mesh.field_data["spacing"] = step[None, :]
        mesh.field_data["voxel_size"] = step[None, :]
        mesh.field_data["base_spacing"] = [d]
        mesh.field_data["sampling_spacing"] = scale[None, :]
        mesh.field_data["smooth_sigma"] = sigma[None, :]
        mesh.field_data["reconstruction_mode"] = ["sections" if axis is not None else "volume"]
        mesh.field_data["retained_points"] = [len(p)]
        mesh.field_data["removed_components"] = [removed_components]
        if reference is not None:
            mesh.field_data["density_reference"] = [reference * w.max()]
            mesh.field_data["density_threshold"] = [density_level * reference * w.max()]
        return mesh

    if label_mode == "project":
        return build(np.arange(len(points)))
    meshes = {}
    for code, key in enumerate(label_keys):
        ids = np.flatnonzero(label_codes == code)
        if len(ids) < 8:
            warnings.warn(f"Skip label {key!r}: fewer than eight retained points.", stacklevel=2)
            continue
        meshes[key] = build(ids, key, code)
    if not meshes:
        raise ValueError("No label has enough retained points.")
    return meshes



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
              reference_spacing=None):
    """Local-density rejection, bounded local scales and regularized PCA."""
    tree = cKDTree(points)
    k = min(k, len(points) - 1)
    rk = tree.query(points, k=[k + 1], workers=-1)[0][:, 0]
    keep = np.ones(len(points), bool)
    if outlier > 0:
        for start in range(0, len(points), chunk):
            sl = slice(start, start + chunk)
            ids = tree.query(points[sl], k=k + 1, workers=-1)[1][:, 1:]
            score = rk[sl] / np.maximum(np.median(rk[ids], axis=1), 1e-30)
            keep[sl] = score <= outlier
    retained = np.flatnonzero(keep)
    points, confidence = points[keep], confidence[keep]
    if len(points) < 4:
        raise ValueError("Fewer than four points remain after local outlier removal.")
    if not keep.all():
        tree = cKDTree(points)
        k = min(k, len(points) - 1)
        rk = tree.query(points, k=[k + 1], workers=-1)[0][:, 0]
    d = float(np.median(rk) / np.cbrt(k) if reference_spacing is None else reference_spacing)
    if not np.isfinite(rk).all() or np.any(rk <= 0) or not np.isfinite(d) or d <= 0:
        raise ValueError("Point distances must give a finite positive sampling scale.")
    precision = np.empty((len(points), 3, 3), np.float64)
    axes = np.empty((len(points), 3), np.float64)
    eye = np.eye(3)
    # Weaken regularization when a larger axis ratio is explicitly requested.
    shrink = min(0.2, 1.0 / anisotropy ** 2)
    for start in range(0, len(points), chunk):
        sl = slice(start, start + chunk)
        dist, ids = tree.query(points[sl], k=k + 1, workers=-1)
        dist, ids = dist[:, 1:], ids[:, 1:]
        local = np.median(rk[ids], axis=1) / np.cbrt(k)
        r = radius * np.clip(local, 0.5 * d, 2.0 * d) * np.cbrt(confidence[sl])
        if anisotropy == 1:
            axes[sl] = r[:, None]
            precision[sl] = eye / r[:, None, None] ** 2
            continue
        # Normalize before forming covariance: orientation/ratios do not depend
        # on this scalar, and small/large physical units remain well conditioned.
        delta = (points[ids] - points[sl, None, :]) / rk[sl, None, None]
        w = np.exp(-0.5 * (dist / rk[sl, None]) ** 2) * confidence[ids]
        w /= w.sum(axis=1, keepdims=True)
        delta -= np.einsum("nk,nkj->nj", w, delta)[:, None, :]
        cov = np.einsum("nk,nki,nkj->nij", w, delta, delta)
        cov = ((1 - shrink) * cov + (shrink / 3) *
               np.trace(cov, axis1=1, axis2=2)[:, None, None] * eye)
        eigen, basis = np.linalg.eigh(cov)
        eigen = np.maximum(eigen, eigen[:, -1:] / anisotropy ** 2)
        relative = np.sqrt(eigen / np.exp(np.log(eigen).mean(axis=1))[:, None])
        lengths = r[:, None] * relative
        axes[sl] = lengths
        precision[sl] = (basis / lengths[:, None, :] ** 2) @ basis.transpose(0, 2, 1)
    return points, confidence, retained, precision, axes, d



def _blocks(points, reach):
    """Conservative spatial blocks; sparse cell graph avoids distant empty grids."""
    cells, inverse = np.unique(np.floor((points - points.min(0)) / reach).astype(np.int64),
                               axis=0, return_inverse=True)
    edges = cKDTree(cells).query_pairs(1.0, p=np.inf, output_type="ndarray")
    graph = sparse.coo_matrix((np.ones(len(edges), np.uint8), (edges[:, 0], edges[:, 1])),
                              shape=(len(cells), len(cells))).tocsr()
    _, groups = sparse.csgraph.connected_components(graph, directed=False)
    groups = groups[inverse]
    order = np.argsort(groups, kind="stable")
    return np.split(order, np.flatnonzero(np.diff(groups[order])) + 1)



def _support_mask(points, precision, axes, origin, step, shape, chunk):
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
        dist, ids = tree.query(q, workers=-1)
        inside = dist <= shortest[ids]
        pending = np.flatnonzero(~inside & (dist <= longest))
        count = min(8, len(points))
        while len(pending):
            remaining = []
            batch = max(1, min(chunk, 1_000_000 // count))
            for j in range(0, len(pending), batch):
                at = pending[j:j + batch]
                dist, ids = tree.query(q[at], k=list(range(1, count + 1)),
                                       distance_upper_bound=upper, workers=-1)
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
                 smooth_sigma=None):
    import pyvista as pv

    step = np.broadcast_to(np.asarray(d * np.asarray(voxel_scale) if voxel_size is None else voxel_size,
                                      dtype=float), (3,)).copy()
    if not np.isfinite(step).all() or np.any(step <= 0):
        raise ValueError("spacing_scale/spacing must be positive and scalar or length three.")
    sigma = np.broadcast_to(np.asarray(d * smooth if smooth_sigma is None else
                                        smooth_sigma, float), (3,)).copy()
    closing = gap_size / 2
    reach = 2 * axes.max() + gap_size + 6 * sigma.max() + 4 * step.max()
    blocks = _blocks(points, reach)
    parts, largest, n_components, peak_voxels = [], 0, 0, 0
    for ids in blocks:
        p, a = points[ids], axes[ids]
        margin = a.max() + gap_size + 3 * sigma + 3 * step.max()
        origin = p.min(0) - margin
        extent = np.ceil((p.max(0) + margin - origin) / step) + 1
        if not np.isfinite(extent).all() or np.any(extent > max_voxels):
            raise MemoryError("One grid axis exceeds max_voxels; increase spacing.")
        shape = extent.astype(np.int64)
        size = int(np.prod(shape.astype(object)))
        if size > max_voxels:
            raise MemoryError(f"One spatial block needs {size:,} voxels; increase "
                              "spacing_scale, spacing or max_voxels. Resolution was not changed.")
        peak_voxels = max(peak_voxels, size)
        mask = _support_mask(p, precision[ids], a, origin, step, shape, chunk)
        if not mask.any():
            continue
        if closing > 0:
            dilated = ndi.distance_transform_edt(~mask, sampling=step) <= closing
            mask |= ndi.distance_transform_edt(dilated, sampling=step) > closing
            del dilated
        if method == "sdf":
            field = ndi.distance_transform_edt(mask, sampling=step).astype(np.float32)
            field -= ndi.distance_transform_edt(~mask, sampling=step).astype(np.float32)
            band = max(a.max() + gap_size, 3 * sigma.max(), 2 * step.max())
            np.clip(field, -band, band, out=field)
        else:
            # A smoothed occupied-volume fraction, independent of point counts.
            field = mask.astype(np.float32)
            band = float(density_level)
        del mask
        if np.any(sigma > 0):
            field = _gaussian(field, sigma / step, device)
        if method == "density":
            field -= density_level
        for axis in range(3):
            border = [slice(None)] * 3
            border[axis] = [0, -1]
            field[tuple(border)] = -band
        regions, count = ndi.label(field > 0)
        n_components += count
        if count == 0:
            continue
        sizes = np.bincount(regions.ravel())
        largest = max(largest, int(sizes[1:].max()))
        boxes = ndi.find_objects(regions)
        for region, box in enumerate(boxes, 1):
            volume = int(sizes[region])
            if volume < min_component * largest:
                continue
            lo = np.array([max(0, s.start - 1) for s in box])
            hi = np.array([min(n, s.stop + 1) for s, n in zip(box, shape)])
            crop = tuple(slice(l, h) for l, h in zip(lo, hi))
            local = field[crop].copy()
            # Keep every air voxel, including cavities, of this VOLUME region.
            local[(regions[crop] != region) & (local > 0)] = (
                -step.min() if method == "sdf" else -density_level)
            vertices, faces = _marching_cubes(local, origin + lo * step, step, backend, device)
            mesh = pv.PolyData(vertices, np.c_[np.full(len(faces), 3), faces]).clean()
            parts.append((volume, mesh))
        del field, regions
    parts = [(v, m) for v, m in parts if v >= min_component * largest]
    if min_component == 1 and parts:
        parts = [max(parts, key=lambda item: item[0])]
    if not parts:
        raise ValueError("No occupied region remains; decrease spacing_scale/spacing "
                         "or smooth_scale, increase support_scale, or decrease density_level in density mode.")
    # Concatenate triangles without reorienting cavity surfaces or merging labels.
    vertices, faces, offset = [], [], 0
    for _, mesh in parts:
        vertices.append(mesh.points)
        faces.append(mesh.faces.reshape(-1, 4)[:, 1:] + offset)
        offset += mesh.n_points
    faces = np.concatenate(faces)
    mesh = pv.PolyData(np.concatenate(vertices), np.c_[np.full(len(faces), 3), faces])
    if smooth_iters:
        mesh = mesh.smooth_taubin(n_iter=smooth_iters, pass_band=0.1,
                                  normalize_coordinates=True)
    mesh = mesh.compute_normals(consistent_normals=True, auto_orient_normals=False)
    mesh.field_data["spacing"] = step[None, :]
    mesh.field_data["spacing_scale"] = (step / d)[None, :]
    mesh.field_data["base_spacing"] = [d]
    mesh.field_data["voxel_size"] = step[None, :]
    mesh.field_data["voxel_scale"] = (step / d)[None, :]
    mesh.field_data["sampling_spacing"] = [d]
    mesh.field_data["smooth_sigma"] = sigma[None, :]
    mesh.field_data["method"] = [method]
    if method == "density":
        mesh.field_data["density_level"] = [density_level]
    mesh.field_data["gap_size"] = [gap_size]
    mesh.field_data["volume_components"] = [len(parts)]
    mesh.field_data["removed_components"] = [n_components - len(parts)]
    mesh.field_data["spatial_blocks"] = [len(blocks)]
    mesh.field_data["peak_grid_voxels"] = [peak_voxels]
    return mesh



def clay_mesh(
    points=None, label=None, feature=None, confidence=None, *, label_mode="project",
    method="sdf", density_level=0.5,
    support_scale=None, smooth_scale=None, smooth_sigma=None,
    gap_size=0.0, anisotropy=2.0, spacing=None, spacing_scale=None,
    base_spacing=None, outlier_thresh=None, outlier_k=24,
    min_component=0.01, mesh_smooth_iters=None,
    backend="auto", device="cpu", max_voxels=2_000_000_000, chunk_size=100_000,
    pc=None, radius=None, smooth=None, voxel_size=None, voxel_scale=None,
    reference_spacing=None, outlier=None, smooth_iters=None,
):
    """Reconstruct a smooth occupied volume from arbitrarily arranged 3D points.

    Parameters
    ----------
    points : (N,3) array, tensor, PyVista dataset or readable file path
        Volume samples in current physical XYZ coordinates. Surface-only input
        produces supported neighborhoods; a solid interior is not inferred.
    label, feature, confidence : arrays or names in points.point_data, optional
        label: (N,) categories. feature: finite real (N,) or (N,C), projected
        with confidence-weighted inverse squared distance to at most 8 points.
        confidence: (N,) in [0,1]; zero excludes a point. Ellipsoid volume is
        proportional to confidence through semiaxes multiplied by c**(1/3).
        Duplicate coordinates keep the highest-confidence observation and its
        label/feature; the first input wins confidence ties.
    label_mode : 'project' or 'separate', default 'project'
        Project returns one mesh with nearest labels. Separate reconstructs
        each label independently and returns {label: mesh}; meshes may overlap.
    method : 'sdf' or 'density', default 'sdf'
        Both use ellipsoid support, optional closing and the same smoothing.
        sdf: Gaussian smoothing of a clipped signed Euclidean distance field.
        density: Gaussian smoothing of binary occupancy, then density_level.
        Density here is OCCUPIED VOLUME FRACTION, not point-count KDE.
    density_level : float in (0,1), default 0.5
        Density only: iso-level. Smaller values expand/connect the region;
        larger values contract it. Ignored for sdf.
    support_scale : positive float, default 2
        Multiplier of bounded local sampling spacing: local estimates are
        clipped to [0.5*d, 2*d], where d=base_spacing. Increasing this fills
        sampling gaps, thickens tissue and narrows holes.
    smooth_scale : nonnegative scalar or XYZ sequence, default 1
        Dimensionless Gaussian widths; physical sigma = d*smooth_scale.
        XYZ refers to current global axes, independent of local PCA axes.
    smooth_sigma : nonnegative scalar or XYZ sequence, optional
        Physical Gaussian standard deviations; overrides smooth_scale.
        Zero disables field smoothing on that axis; final mesh smoothing may
        still move vertices along all axes.
    gap_size : nonnegative float, default 0
        PHYSICAL diameter of the Euclidean closing ball; 0 disables closing.
        Fills small gaps and may also close genuine narrow holes. It is not
        a maximum slice gap. No global hole-filling step is applied.
    anisotropy : float >=1, default 2
        Maximum local support semiaxis ratio. 1 gives balls and skips PCA.
        Shrinkage=min(0.2,1/anisotropy**2); eigenvalues are floored to
        lambda_max/anisotropy**2, then axes are volume-normalized. Local PCA
        follows observed support directions; it does not infer missing axes.
    spacing : positive scalar or XYZ sequence, optional
        Absolute physical grid spacing; overrides spacing_scale.
    spacing_scale : positive scalar or XYZ sequence, default 0.5
        Relative grid spacing: spacing = d*spacing_scale if spacing is absent.
        Decrease to refine the grid, without changing physical smoothing.
    base_spacing : positive scalar, optional
        Fix d instead of estimating median(r_k)/k**(1/3) for each input/label.
        This is an effective volume-sampling scale, not original slice spacing.
        It freezes the relative grid and smoothing scales when changing input
        density. Local support still adapts to the actual neighborhood geometry.
    outlier_thresh : float, default 2.5
        Reject when r_k / median(neighbors' r_k) exceeds this ratio; 0 disables
        rejection, otherwise require >1. Dense coherent noise can survive.
        This local density ratio differs from the slice method's MAD cutoff.
    outlier_k : int >=3, default 24
        Shared neighbor count for denoising, local scale and PCA.
    min_component : float in [0,1], default 0.01
        Minimum occupied VOLUME / largest volume, separately for each label.
        0 retains all; 1 retains one largest volume and its cavity boundaries.
    mesh_smooth_iters : nonnegative int, default 10
        Final Taubin smoothing iterations, pass_band=0.1. 0 disables.
    backend : str, default 'auto'
        'auto', 'pymcubes', 'skimage', 'lewiner', 'lorensen', or 'torchmcubes'.
        Auto prefers PyMCubes. CUDA extraction requires torchmcubes explicitly.
    device : str, default 'cpu'
        CUDA supports torch Gaussian filtering and explicit torchmcubes only.
        Neighborhoods, support, EDT and PyVista remain on CPU.
    max_voxels : int, default 2_000_000_000
        Grid-node limit per spatial block; overflow raises without coarsening.
        This is not a byte limit: EDT and component labels need temporaries.
    chunk_size : positive int, default 100_000
        Query batch size; ellipsoid and feature batches are additionally bounded.

    Returns
    -------
    pyvista.PolyData or dict
        Optional point arrays 'label', 'feature', and 'source_id' (nearest
        retained input row). Field data records physical scales and grid use.
        Reconstruction is a geometric prior, not a topology/boundary guarantee.

    Notes
    -----
    Legacy aliases remain accepted: pc, radius, smooth, voxel_size, voxel_scale,
    reference_spacing, outlier and smooth_iters respectively name points,
    support_scale, smooth_scale, spacing, spacing_scale, base_spacing,
    outlier_thresh and mesh_smooth_iters. Specifying both names is an error.
    """
    import pyvista as pv

    pc = _alias(points, pc, None, "points", "pc")
    radius = _alias(support_scale, radius, 2.0, "support_scale", "radius")
    smooth = _alias(smooth_scale, smooth, 1.0, "smooth_scale", "smooth")
    voxel_size = _alias(spacing, voxel_size, None, "spacing", "voxel_size")
    voxel_scale = _alias(spacing_scale, voxel_scale, 0.5, "spacing_scale", "voxel_scale")
    reference_spacing = _alias(base_spacing, reference_spacing, None,
                               "base_spacing", "reference_spacing")
    outlier = _alias(outlier_thresh, outlier, 2.5, "outlier_thresh", "outlier")
    smooth_iters = _alias(mesh_smooth_iters, smooth_iters, 10,
                          "mesh_smooth_iters", "smooth_iters")
    if isinstance(pc, (str, Path)):
        pc = pv.read(pc)
    source = pc if hasattr(pc, "points") else None
    raw = source.points if source is not None else pc
    points = np.asarray(raw.detach().cpu().numpy() if hasattr(raw, "detach") else raw,
                        dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError("points must contain at least four 3D points.")
    arrays = []
    for value in (label, feature, confidence):
        if isinstance(value, str):
            if source is None:
                raise ValueError("Named arrays require a PyVista input.")
            value = source.point_data[value]
        if value is not None:
            value = np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value)
            if value.ndim == 0 or len(value) != len(points):
                raise ValueError("label, feature and confidence must align with points.")
        arrays.append(value)
    label, feature, confidence = arrays
    if label is not None and label.ndim != 1:
        raise ValueError("label must have shape (N,).")
    if label is not None:
        keys, label_codes = _label_codes(label)
    if feature is not None and (feature.ndim not in (1, 2) or
            (feature.ndim == 2 and feature.shape[1] == 0) or
            not np.issubdtype(feature.dtype, np.number) or np.iscomplexobj(feature) or
            not np.isfinite(feature).all()):
        raise ValueError("feature must be a finite real (N,) or (N,C), C>=1, array.")
    confidence = np.ones(len(points)) if confidence is None else np.asarray(confidence, float)
    if confidence.ndim != 1 or not np.isfinite(confidence).all() or np.any(
            (confidence < 0) | (confidence > 1)):
        raise ValueError("confidence must have shape (N,) and lie in [0,1].")
    if label_mode not in {"project", "separate"} or (label_mode == "separate" and label is None):
        raise ValueError("Use label_mode='project', or 'separate' with label supplied.")
    if backend not in {"auto", "pymcubes", "skimage", "lewiner", "lorensen", "torchmcubes"}:
        raise ValueError("Unknown marching-cubes backend.")
    smooth = np.broadcast_to(np.asarray(smooth, float), (3,)).copy()
    if not np.isfinite(smooth).all() or np.any(smooth < 0):
        raise ValueError("smooth_scale must be nonnegative, scalar or length three.")
    if smooth_sigma is not None:
        smooth_sigma = np.broadcast_to(np.asarray(smooth_sigma, float), (3,)).copy()
        if not np.isfinite(smooth_sigma).all() or np.any(smooth_sigma < 0):
            raise ValueError("smooth_sigma must be finite, nonnegative and scalar or length three.")
    if method not in {"sdf", "density"}:
        raise ValueError("method must be 'sdf' or 'density'.")
    if method == "density" and not 0 < density_level < 1:
        raise ValueError("density_level must lie in (0,1).")
    if reference_spacing is not None and not (np.ndim(reference_spacing) == 0 and
            np.isfinite(reference_spacing) and reference_spacing > 0):
        raise ValueError("base_spacing must be a positive scalar.")
    if not np.isfinite([radius, gap_size, anisotropy, outlier]).all() or not (
            radius > 0 and gap_size >= 0 and anisotropy >= 1 and
            (outlier == 0 or outlier > 1) and 0 <= min_component <= 1):
        raise ValueError("Invalid support, smoothing, outlier or component parameters.")
    if any(int(v) != v for v in (outlier_k, smooth_iters, max_voxels, chunk_size)) or not (
            outlier_k >= 3 and smooth_iters >= 0 and max_voxels >= 1000 and chunk_size > 0):
        raise ValueError("Require integer k>=3, iterations>=0, max_voxels>=1000, chunk_size>0.")
    outlier_k, smooth_iters, max_voxels, chunk_size = map(
        int, (outlier_k, smooth_iters, max_voxels, chunk_size))

    def build(ids):
        ids = ids[np.isfinite(points[ids]).all(axis=1) & (confidence[ids] > 0)]
        ids = ids[np.argsort(-confidence[ids], kind="stable")]
        _, unique = np.unique(points[ids], axis=0, return_index=True)
        ids = ids[unique]
        if len(ids) < 4:
            raise ValueError("Each reconstruction needs four distinct finite reliable points.")
        offset = np.median(points[ids], axis=0)
        p, c, retained, precision, axes, d = _geometry(
            points[ids] - offset, confidence[ids], outlier_k, outlier, radius,
            anisotropy, min(chunk_size, max(1, 1_000_000 // outlier_k)), reference_spacing)
        ids = ids[retained]
        mesh = _reconstruct(p, precision, axes, d, smooth, gap_size, voxel_size,
                            min_component, smooth_iters, backend, device, max_voxels, chunk_size,
                            voxel_scale, method, density_level, smooth_sigma)
        tree = cKDTree(p)
        nearest = tree.query(mesh.points, workers=-1)[1]
        mesh.point_data["source_id"] = ids[nearest]
        if label is not None:
            values = label[ids[nearest]]
            mesh.point_data["label_id"] = label_codes[ids[nearest]]
            mesh.point_data["label"] = values.astype(str) if values.dtype.kind == "O" else values
        if feature is not None:
            k = min(8, len(p))
            values = np.empty((mesh.n_points,) + feature.shape[1:], np.float64)
            width = 1 if feature.ndim == 1 else max(1, feature.shape[1])
            batch = min(chunk_size, max(1, 1_000_000 // (k * width)))
            for start in range(0, mesh.n_points, batch):
                sl = slice(start, start + batch)
                dist, near = tree.query(mesh.points[sl], k=list(range(1, k + 1)), workers=-1)
                # Divide distances by their row minimum before squaring so
                # the same interpolation works in any physical length unit.
                safe = np.maximum(dist, np.finfo(float).tiny)
                w = c[near] * (safe[:, :1] / safe) ** 2
                exact = dist[:, 0] == 0
                w[exact] = 0
                w[exact, 0] = 1
                w /= w.sum(axis=1, keepdims=True)
                values[sl] = np.einsum("nk,nk...->n...", w, feature[ids[near]])
            mesh.point_data["feature"] = values
        mesh.points = np.asarray(mesh.points, np.float64) + offset
        mesh.field_data["retained_points"] = [len(p)]
        mesh.field_data["support_radius"] = [radius * d]
        mesh.field_data["support_max_semiaxis"] = [float(axes.max())]
        mesh.field_data["anisotropy"] = [anisotropy]
        return mesh

    if label_mode == "project":
        return build(np.arange(len(points)))
    result = {}
    for i, key in enumerate(keys):
        try:
            result[key] = build(np.flatnonzero(label_codes == i))
        except ValueError as error:
            raise ValueError(f"Cannot reconstruct label {key!r}: {error}") from error
    return result



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
    points, *,
    spacing_scale=0.5,
    spacing=None,
    base_spacing=None,
    method="sdf",
    support_scale=1.5,
    adaptive=True,
    support_k=8,
    radius_cap=2.0,
    smooth_scale=1.0,
    smooth_sigma=None,
    smooth_iters=1,
    preserve_volume=True,
    max_shift_scale=2.0,
    density_level=0.5,
    closing_scale=0.0,
    opening_scale=0.0,
    hole_scale=0.0,
    offset_scale=0.0,
    outlier_k=24,
    outlier_thresh=1.8,
    outlier_iters=1,
    min_points=0.01,
    min_voxels=0,
    component_scale=0.0,
    keep_largest=False,
    score_k=8,
    score_scale=2.0,
    padding=2,
    bounds=None,
    max_voxels=2_000_000_000,
    chunk_size=100_000,
    workers=-1,
):
    """Return mask, score_image, spacing, origin, points and score; no file I/O.

    Parameters
    ----------
    points : (N,3) array/tensor or object with .points
        XYZ coordinates in consistent units, with at least two distinct finite
        rows. Duplicates and nonfinite rows are removed. No slice axis is assumed.
    spacing_scale : positive scalar or XYZ, default 0.5
        Voxel spacing = s0 * spacing_scale. The automatic reference scale is
        s0 = median(distance to kth other point) / k**(1/3), k=support_k capped
        by N-1, after filtering. This is a VOLUME-sampling scale estimate, not an
        estimate of missing slice spacing. k=1 uses median nearest-neighbor distance.
        Decrease it to increase grid resolution; halving all three spacings
        requires about 8 times as many voxels for a fixed physical domain.
    spacing : positive scalar or XYZ, optional
        Actual voxel spacing in coordinate units. Overrides spacing_scale.
        Changing spacing alone does not change support or physical smoothing.
    base_spacing : positive float, optional
        Fixed reference scale s0, overriding the automatic estimate throughout
        component filtering and reconstruction. Useful to keep scale parameters
        comparable across clouds or different sampling densities. Does not disable
        local adaptive radii. Keep it fixed when comparing resolutions/settings.
    method : {'sdf', 'density', 'support', 'occupancy'}, default 'sdf'
        sdf: local ellipsoid support -> physical SDF -> Gaussian -> threshold.
        density: same support -> Gaussian-smoothed binary occupancy -> threshold;
                 it is not a raw-point KDE and does not sum point counts.
        support: local ellipsoid union, with no field smoothing.
        occupancy: mark retained input-point voxels, with no field smoothing.
        Optional morphology and cleanup below apply to ALL methods.
    support_scale : positive scalar or XYZ, default 1.5
        Ellipsoid semiaxes = s0 * support_scale * local_radius_factor.
        Larger z support bridges along WORLD z, independently of z voxel spacing.
        All support ellipsoids are axis-aligned; no local PCA or normals are used.
    adaptive, support_k, radius_cap : default True, 8, 2.0
        Factor = clip(local kth distance / its median, 1/radius_cap, radius_cap).
        adaptive=False uses a fixed factor of 1. radius_cap must be >=1. The cap
        limits excessive expansion in sparse regions. support_k also defines s0.
    smooth_scale : nonnegative scalar or XYZ, default 1.0
        Per-unit-time physical Gaussian sigma = s0 * smooth_scale. Zero disables
        filtering on an axis. SDF/density only. Sigma is a standard deviation,
        not a support radius or voxel count. Independent of voxel resolution.
    smooth_sigma : nonnegative scalar or XYZ, optional
        Physical Gaussian sigma in coordinate units; overrides smooth_scale.
        Effective filter sigma in voxels = smooth_sigma * sqrt(smooth_iters)
        / spacing. Without this override, use s0*smooth_scale in the numerator.
    smooth_iters : nonnegative int, default 1
        Gaussian diffusion-time multiplier: sigma_effective = sigma*sqrt(iters).
        The continuous SDF/occupancy field is filtered with ONE effective kernel,
        then thresholded once. 4 gives twice the linear smoothing scale of 1;
        0 disables smoothing. This replaces the old reinitialize/binarize-per-pass
        behavior for iters>1, which could stagnate on the voxel grid. Small
        structures can disappear; no topology or fine-detail guarantee is made.
    preserve_volume : bool, default True
        Adjust smoothing threshold to approximately retain the support volume
        AFTER opening/closing. For tied field values, include or exclude the
        complete plateau according to which is closer to the target volume.
        Exact volume is not always possible without splitting symmetric ties.
        Does not preserve unknown true volume, topology, or the later hole/offset
        changes. Applies only to smoothing, not to every reconstruction step.
    max_shift_scale : nonnegative scalar or None, default 2.0
        Smoothing can change signs only where the INITIAL voxel SDF has magnitude
        <= s0*this value. None removes the band limit; 0 disables smoothing.
        This is a voxel-SDF constraint, not an exact mesh Hausdorff bound.
        A small band can limit the effect even when smooth_scale is large.
    density_level : float in (0,1), default 0.5
        Threshold of the smoothed support occupancy for method='density', used
        only when preserve_volume=False. Lower values expand the foreground.
    closing_scale, opening_scale : nonnegative scalar or XYZ, default 0
        Ellipsoid DIAMETERS in multiples of s0. Closing bridges small gaps;
        opening removes narrow protrusions. Zero on an axis prevents movement
        along that axis. Both are disabled by default to avoid closing real gaps.
    hole_scale : nonnegative float, default 0
        After smoothing/offset, fill only ENCLOSED cavities whose volume is no
        larger than a sphere of diameter s0*hole_scale. 0 disables; np.inf fills
        all enclosed cavities. Through-holes are not enclosed cavities.
    offset_scale : finite float, default 0
        Final isotropic expansion (>0) or erosion (<0), by abs(offset_scale)*s0.
    outlier_k, outlier_thresh, outlier_iters : default 24, 1.8, 1
        Batched fixed-k LOF; reject scores above outlier_thresh (>1). 0/None or
        outlier_iters=0 disables rejection. Compact distant noise groups may
        survive LOF; min_points provides an additional component filter.
    min_points : nonnegative float, default 0.01
        Filter POINT components before grid allocation. A distance-limited kNN
        graph in the support metric is used; k=max(support_k,outlier_k).
        (0,1): fraction of largest component; >=1: minimum point count; 0: off.
    min_voxels, keep_largest : default 0, False
        Final 26-connected MASK component filtering: absolute voxel count, and
        optionally retain only the largest mask component. min_voxels is NOT a
        physical size; use component_scale for consistent filtering across grids.
    component_scale : nonnegative float, default 0
        Minimum retained mask-component volume equals the volume of a sphere
        of diameter s0*component_scale. Unlike min_voxels, this physical threshold
        does not change when spacing changes. If both are positive, apply the
        stricter threshold. 0 disables this additional filter. This is a volume
        criterion, not a test of maximum width or component diameter.
    score_k : positive int, default 8
        Number of distinct retained observations used to score each generated
        point, capped at the retained point count. Larger k requires support
        from more observations. A score is not forced to 1 at a source point.
    score_scale : positive scalar or XYZ, default 2.0
        Scoring Gaussian bandwidth = s0 * score_scale. In this normalized metric,
        score(q) = mean(exp(-0.5 * distance(q, neighbor)**2)) over score_k nearest
        observations. Output lies in [0,1]: closer/more nearby observations give
        stronger support; unsupported gap-filling or expansion gives weak support.
        No division by the sum of Gaussian weights is applied. Fixed bandwidth
        avoids assigning high scores merely because adaptive support radii grew.
        Independent of the reconstruction support/smoothing settings except via
        their effect on point filtering and s0. Changing scoring parameters does
        not change the mask, grid or filtering. This is a geometric observation-
        support measure, not a calibrated probability or proof of correct anatomy.
        This changes the score definition, not shape smoothing; do not increase
        it merely to make reconstructed areas appear more reliable.
    padding : nonnegative integer or XYZ integers, default 2
        Extra empty voxel layers in addition to automatically computed support,
        morphology and smoothing margins. Automatically sized margins prevent
        normal boundary clipping; the volume includes background voxels.
    bounds : (2,3) array, optional
        Explicit lower/upper XYZ bounds BEFORE padding. Must enclose all retained
        points; outside points raise an error, never get silently cropped.
        Default is the filtered point bounding box.
    max_voxels, chunk_size, workers : default 2_000_000_000, 100_000, -1
        Padded grid size limit, KDTree query batch size, and query threads.
        Oversized grids raise instead of silently coarsening. Temporary EDT
        arrays need additional memory; mask + score_image require 5 bytes per grid
        voxel; float64 points + float32 score require 28 bytes per foreground
        voxel. Runtime is CPU; no CUDA dependency.

    Returns
    -------
    dict with exactly six keys:
        mask    : (nx,ny,nz) uint8, binary foreground 1 / background 0.
        score_image : (nx,ny,nz) float32, foreground scores in [0,1], background 0.
                  Exactly score_image[mask.astype(bool)] == score. A foreground
                  score can also be 0: use mask to distinguish background.
        spacing : (3,) float64, actual XYZ voxel spacing, in input units.
        origin  : (3,) float64, world coordinate of voxel center (0,0,0).
        points  : (M,3) float64, foreground voxel centers in np.argwhere order.
                  points = origin + np.argwhere(mask) * spacing.
        score   : (M,) float32 in [0,1], matching points row for row. Computed
                  against retained unique ORIGINAL observations, after all mask
                  operations; generated points never provide their own support.

    Grid axes align with the input Cartesian XYZ axes, so no direction matrix
    or transform is needed in the result. Image shapes/dtypes come from the arrays.
    The previous 'image' key is now 'mask'; no duplicate alias is returned.
    Coordinates alone cannot distinguish a missing observation from a real
    anatomical gap. Support, closing and smoothing can merge nearby structures.
    """
    scale = _xyz(spacing_scale, "spacing_scale", True)
    physical_spacing = None if spacing is None else _xyz(spacing, "spacing", True)
    support = _xyz(support_scale, "support_scale", True)
    smooth = _xyz(smooth_scale, "smooth_scale")
    physical_sigma = None if smooth_sigma is None else _xyz(smooth_sigma, "smooth_sigma")
    closing = _xyz(closing_scale, "closing_scale")
    opening = _xyz(opening_scale, "opening_scale")
    score_axes = _xyz(score_scale, "score_scale", True)
    pad = _xyz(padding, "padding")
    if np.any(pad != np.floor(pad)):
        raise ValueError("padding must contain integers.")
    if method not in {"sdf", "density", "support", "occupancy"}:
        raise ValueError("method must be sdf, density, support, or occupancy.")
    for name, value, lower in (("support_k", support_k, 1), ("outlier_k", outlier_k, 2),
                               ("outlier_iters", outlier_iters, 0), ("smooth_iters", smooth_iters, 0),
                               ("min_voxels", min_voxels, 0), ("score_k", score_k, 1),
                               ("chunk_size", chunk_size, 1),
                               ("max_voxels", max_voxels, 1)):
        if not np.isfinite(value) or int(value) != value or value < lower:
            raise ValueError(f"{name} must be an integer >= {lower}.")
    support_k, outlier_k, outlier_iters, smooth_iters, chunk_size = map(
        int, (support_k, outlier_k, outlier_iters, smooth_iters, chunk_size))
    if (not np.isfinite([radius_cap, min_points, offset_scale, component_scale]).all()
            or radius_cap < 1 or min_points < 0 or component_scale < 0):
        raise ValueError("Require radius_cap >= 1, nonnegative min_points/component_scale, "
                         "and finite offset_scale.")
    if base_spacing is not None and (not np.isfinite(base_spacing) or base_spacing <= 0):
        raise ValueError("base_spacing must be a finite positive scalar, or None.")
    if np.isnan(hole_scale) or hole_scale < 0:
        raise ValueError("hole_scale must be nonnegative, or np.inf.")
    if not 0 < density_level < 1:
        raise ValueError("density_level must be in (0,1).")
    if outlier_thresh == 0:
        outlier_thresh = None
    if outlier_thresh is not None and (not np.isfinite(outlier_thresh) or outlier_thresh <= 1):
        raise ValueError("outlier_thresh must exceed 1, or be 0/None to disable filtering.")
    if max_shift_scale is not None and (not np.isfinite(max_shift_scale) or max_shift_scale < 0):
        raise ValueError("max_shift_scale must be nonnegative, or None.")
    raw = getattr(points, "points", points)
    x = np.asarray(raw.detach().cpu().numpy() if hasattr(raw, "detach") else raw, float)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("points must have shape (N,3).")
    x = np.unique(x[np.isfinite(x).all(1)], axis=0)
    if len(x) < 2:
        raise ValueError("At least two distinct finite points are required.")
    x, s0, radii = _geometry_3(x, support, adaptive, support_k, radius_cap,
                            outlier_k, outlier_thresh, outlier_iters,
                            min_points, chunk_size, workers, base_spacing)
    spacing = s0 * scale if physical_spacing is None else physical_spacing
    sigma = s0 * smooth if physical_sigma is None else physical_sigma
    axes = s0 * support
    use_smooth = (method in {"sdf", "density"} and smooth_iters > 0
                  and np.any(sigma > 0) and max_shift_scale != 0)
    if not use_smooth:
        sigma = np.zeros(3)
    extent = np.zeros(3) if method == "occupancy" else axes * radii.max()
    # The entire dilation of closing must fit, including its later erosion.
    # One continuous Gaussian kernel, plus the permitted sign-change band.
    smooth_margin = 3 * sigma * math.sqrt(smooth_iters)
    if use_smooth and max_shift_scale is not None:
        smooth_margin += s0 * max_shift_scale
    margin = (extent + s0 * (closing + opening) + abs(offset_scale) * s0
              + smooth_margin + 2 * spacing)
    lower, upper = x.min(0), x.max(0)
    if bounds is not None:
        box = np.asarray(bounds, float)
        if box.shape != (2, 3) or not np.isfinite(box).all() or np.any(box[1] < box[0]):
            raise ValueError("bounds must be finite (2,3), lower <= upper.")
        if np.any(lower < box[0]) or np.any(upper > box[1]):
            raise ValueError("bounds must enclose all retained points.")
        lower, upper = box
    layers = np.ceil(margin / spacing) + pad
    size = np.ceil((upper - lower) / spacing) + 1 + 2 * layers
    if not np.isfinite(size).all() or np.any(size > max_voxels):
        raise MemoryError("Grid exceeds max_voxels; increase spacing/spacing_scale or reduce the domain.")
    shape = tuple(int(v) for v in size)
    if math.prod(shape) > max_voxels:
        raise MemoryError(f"Grid {shape} has {math.prod(shape):,} voxels, exceeding max_voxels; "
                         "increase spacing/spacing_scale or reduce the domain.")
    origin = lower - layers * spacing
    mask = (np.zeros(shape, bool) if method == "occupancy" else
            _support_mask_3(x, radii, axes, origin, spacing, shape, chunk_size, workers))
    voxel = np.floor((x - origin) / spacing + 0.5).astype(np.int64)
    mask[tuple(voxel.T)] = True  # Retain sampled voxels even on a very coarse grid.
    del radii, voxel
    if opening.any():
        radius = 0.5 * s0 * opening
        mask = _morph(_morph(mask, radius, spacing, "erode"), radius, spacing, "dilate")
    if closing.any():
        radius = 0.5 * s0 * closing
        mask = _morph(_morph(mask, radius, spacing, "dilate"), radius, spacing, "erode")
    if use_smooth:
        shift = None if max_shift_scale is None else s0 * max_shift_scale
        mask = _smooth_mask_3(mask, spacing, sigma, method, smooth_iters,
                            density_level, shift, preserve_volume)
    if offset_scale:
        radius = np.full(3, abs(offset_scale) * s0)
        mask = _morph(mask, radius, spacing, "dilate" if offset_scale > 0 else "erode")
    if hole_scale > 0 and mask.any():
        holes = ndi.binary_fill_holes(mask) & ~mask
        if np.isfinite(hole_scale):
            ids, _ = ndi.label(holes)  # 6-connected background / 26-connected foreground.
            sizes = np.bincount(ids.ravel())
            limit = math.pi / 6 * float(s0 * hole_scale) ** 3 / float(np.prod(spacing))
            valid = sizes <= limit
            valid[0] = False
            holes = valid[ids]
            del ids, sizes, valid
        mask |= holes
        del holes
    if (min_voxels or component_scale or keep_largest) and mask.any():
        ids, _ = ndi.label(mask, structure=np.ones((3, 3, 3), bool))
        sizes = np.bincount(ids.ravel())
        sizes[0] = 0
        volume_cutoff = math.pi / 6 * float(s0 * component_scale) ** 3 / float(np.prod(spacing))
        keep = sizes >= max(1, min_voxels, volume_cutoff)
        if keep_largest:
            keep &= np.arange(len(sizes)) == sizes.argmax()
        mask = keep[ids]
        del ids, sizes, keep
    image = mask.astype(np.uint8)
    indices = np.flatnonzero(mask)
    del mask
    grid_points = np.empty((len(indices), 3), np.float64)
    score = np.empty(len(indices), np.float32)
    score_image = np.zeros(shape, np.float32)
    flat_score_image = score_image.ravel()
    score_axes = _xyz(s0 * score_axes, "physical score bandwidth", True)
    anchor = x.min(0)
    tree = cKDTree((x - anchor) / score_axes)
    nk = min(int(score_k), len(x))
    batch = max(1, min(chunk_size, 1_000_000 // nk))
    for start in range(0, len(indices), batch):
        sl = slice(start, start + batch)
        ijk = np.column_stack(np.unravel_index(indices[sl], shape))
        grid_points[sl] = origin + ijk * spacing
        q = ((origin - anchor) + ijk * spacing) / score_axes
        distance = tree.query(q, k=list(range(1, nk + 1)), workers=workers)[0]
        np.square(distance, out=distance)
        distance *= -0.5
        np.exp(distance, out=distance)
        score[sl] = distance.mean(axis=1)
        flat_score_image[indices[sl]] = score[sl]
    return {"mask": image, "score_image": score_image,
            "spacing": spacing, "origin": origin,
            "points": grid_points, "score": score}



def image_to_nrrd(result, filename, *, key="mask", space="3D-right-handed", units=None,
                  compression_level=1):
    """Write one selected image and its spatial metadata; return the output path.

    key='mask' writes a uint8 binary mask; key='score_image' writes float32 scores,
    preserving fractional values and writing an explicit little-endian payload.
    Call twice to save both images, which share exactly the same spatial grid.
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
    if key not in {"mask", "score_image"}:
        raise ValueError("key must be 'mask' or 'score_image'.")
    image = np.asarray(result[key])
    if image.ndim != 3 or 0 in image.shape:
        raise ValueError(f"result[{key!r}] must have a nonempty 3-D shape.")
    if key == "mask":
        if not np.isin(image, (0, 1)).all():
            raise ValueError("mask must contain only 0 and 1.")
        dtype, nrrd_type = np.dtype("uint8"), "uint8"
    else:
        if image.shape != np.shape(result["mask"]):
            raise ValueError("score_image and mask must have the same shape.")
        if np.iscomplexobj(image) or not np.isfinite(image).all() or np.any((image < 0) | (image > 1)):
            raise ValueError("score_image must contain finite real values in [0,1].")
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
                keep = score <= med + threshold * max(mad, np.finfo(float).eps * med)
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



def grid_point_cloud(
    points, spacing=None, *,
    spacing_scale=0.5, base_spacing=None, axis_scale=1.0, direction=None,
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
):
    """Reconstruct an occupied region from arbitrary 3-D volume samples.

    Parameters
    ----------
    points : (N, 3) array or object with .points
        World coordinates in consistent physical units. Remove nonfinite rows
        and exact duplicates. No original slice coordinates are required.
    spacing : positive scalar or (sx, sy, sz), optional
        Physical grid steps along direction columns. Overrides spacing_scale.
        Default: h*axis_scale*spacing_scale. h is base_spacing, or the median
        support_k-th nonself distance divided by cbrt(support_k), estimated
        after filtering in metric q = (points @ direction) / axis_scale.
    spacing_scale : positive scalar or triple, default 0.5
        Automatic grid spacing multiplier. Smaller values give a denser grid;
        explicit spacing is used unchanged. Does not alter support/smoothing.
    base_spacing : positive scalar, optional
        Fixed metric reference h for automatic spacing, smoothing and KDE.
        Also rescales adaptive support radii by h/estimated_h; an explicit
        support_radius overrides them. Physical reference axes=h*axis_scale.
        For axis_scale=1 this estimator agrees with points_to_image when their
        support-neighbour count and retained points agree. Earlier grid code
        used median first-neighbour distance; set this explicitly to retain
        that scale when reproducing a previous reconstruction.
    axis_scale : positive scalar or triple, default 1
        Distance metric axes: d(x,y)=norm(((x-y) @ direction)/axis_scale).
        (1,1,3) compresses the third axis in all kNN filtering/support queries,
        giving support ellipsoids with a longer third physical semi-axis.
        Relative axis ratios are normally useful; this does not move points.
    direction : orthonormal (3,3) array, optional
        Columns are grid basis vectors in world coordinates. Default identity.
        Supports rotated axes; cannot follow a spatially varying nonlinear warp.
    method : {"knn", "nearest", "kde"}, default "knn"
        knn: mean of k squared distances normalised by neighbour support radii
        <= 1; adaptive to local sampling density, as in the original algorithm.
        nearest: union of fixed-radius ellipsoids, using a nearest-point query.
        kde: trilinear point deposition then Gaussian density convolution;
        faster on dense grids, but sensitive to sampling-density differences.
    support_scale : positive float, default 1.25
        Multiplier of adaptive RMS kNN radii (knn) or their capped median
        (nearest). Larger values bridge gaps AND expand boundaries. Ignored by
        KDE. When support_radius is supplied, it is also multiplied by this.
    support_k : positive int, default 8
        Number of neighbours for local RMS scales and knn support evaluation.
        Automatically capped at the retained unique point count minus one.
    support_radius : positive float, optional
        Fixed radius in metric coordinates, replacing adaptive RMS radii.
        Physical ellipsoid semi-axes are support_radius*support_scale*axis_scale.
        Ignored by KDE. Set support_scale=1 for a direct radius specification.
    radius_cap : float >= 1, default 2
        Cap adaptive radii and graph local scales at this multiple of their
        respective medians. Does not cap an explicitly supplied support_radius.
    kernel_scale : positive scalar or triple, default 2
        KDE bandwidth = h * axis_scale * kernel_scale, in physical grid-axis
        units. Controls reconstruction only; independent of final smoothing.
    density_level : float in (0,1), default 0.2
        KDE threshold relative to median density evaluated at retained points.
        Smaller values expand support; values are not occupancy probabilities.
    smooth_sigma : nonnegative scalar or triple, optional
        Physical Gaussian standard deviation along grid axes. Overrides
        smooth_scale. Automatic sigma=h*axis_scale*smooth_scale.
    smooth : nonnegative scalar or triple, optional
        Legacy alias for smooth_sigma; do not specify both. Explicit physical
        smooth is no longer multiplied again by smooth_scale.
    smooth_scale : nonnegative scalar or triple, default 1
        Automatic smoothing multiplier, independent of output spacing.
        Resolved sigma is info['smooth_sigma']; info['smooth'] is its legacy
        alias. smooth_scales below applies separate reconstruction stages.
    smooth_scales : positive float sequence, default (1,)
        Multipliers applied successively to resolved sigma, usually decreasing,
        e.g. (2,1). Each stage thresholds and rebuilds SDF/occupancy before the
        next stage. This is not repeated Gaussian filtering of a fixed field.
        Use (1,) for one stage; more stages do not guarantee a better result.
    smooth_method : {"sdf", "occupancy", None}, default "sdf"
        sdf: Gaussian smoothing of physical-distance embedding (recommended).
        occupancy: Gaussian smoothing of binary occupancy; useful for strong
        suppression of small protrusions, but may remove thin regions faster.
        None disables smoothing. Returned field is negative inside in both
        methods; occupancy field is dimensionless, SDF field is only implicit
        after filtering/clipping, not an exact signed distance.
    max_shift : nonnegative float, optional
        Physical displacement BAND around the initial post-morphology voxel
        SDF, shared by every scale. Default: 2*max(smooth)*max(smooth_scales).
        Nodes with abs(initial SDF) >= max_shift keep their original sign.
        For sdf, field changes are also clipped to +/-max_shift. A bound smaller
        than every spacing freezes the voxel mask. Zero disables smoothing.
        This is not a strict geometric/topological guarantee for the mesh.
    preserve_volume : bool, default True
        Fit the threshold inside the eligible band to approximately retain
        the initial post-morphology voxel count at every stage. Threshold ties
        can prevent an exact count. Does not preserve the unknown true volume,
        topology, or volumes before filtering/closing/opening/hole filling.
    outlier_method : {"lof", "sor", "radius", None}, default "lof"
        lof: local density-ratio rejection, suited to nonuniform sampling.
        sor: reject mean-kNN distances > median + outlier_thresh*1.4826*MAD.
        radius: require outlier_k neighbours within outlier_radius; if absent,
        radius=outlier_thresh*median(kth-neighbour distance). These all use the
        axis_scale metric. None disables point outlier filtering only.
    outlier_k : int >= 2, default 24
        Outlier neighbours; also controls the component graph's candidate edges.
        Automatically capped at N-1 for outliers/graph.
    outlier_thresh : nonnegative float or None, default 1.8
        Method-specific cutoff; LOF normally > 1. Smaller values reject more.
        0 or None disables outlier filtering, including the radius method.
        Otherwise ignored for radius when outlier_radius is supplied.
    outlier_iters : nonnegative int, default 1
        Recompute neighbours after each removal. 1-2 is usually sufficient;
        many iterations may peel away valid boundaries.
    outlier_radius : positive float, optional
        Radius-filter distance in metric coordinates; ellipsoidal physically.
    min_points : nonnegative number, default 0.01
        Drop POINT graph components smaller than this fraction of the largest
        (0<value<1), or this absolute point count (value>=1). 0 disables cutoff.
        Real disconnected/sparse slices can also be removed; use 0 if needed.
    min_component : nonnegative number, optional
        Legacy alias for min_points; do not specify both. This cutoff counts
        points and is distinct from any mesh-face component threshold.
    keep_largest : bool, default False
        Keep only the largest point graph component, also subject to cutoff.
    graph_scale : positive float, default 2
        Edge length <= graph_scale*min(local_i,local_j). Local scales are
        capped medians of up to six neighbour distances. Larger values permit
        longer edges, provided they are among the kNN candidates. Disabling
        outlier filtering does not disable this independent component stage.
    component_scale : positive float, optional
        Legacy alias for graph_scale; do not specify both. This graph edge
        multiplier differs from points_to_image's voxel component-size scale.
    closing, opening : nonnegative floats, default 0
        Morphological radii in the axis_scale metric. Closing then opening,
        using EDT with sampling=spacing/axis_scale. Closing joins small gaps;
        opening removes thin protrusions. Physical axis radii = value*axis_scale.
        These are metric RADII, unlike clay_mesh gap_size (physical diameter).
        Larger real cavities/thin parts cannot be distinguished from defects.
    fill_holes : bool, default False
        Explicitly fill every enclosed 3-D cavity (6-connected exterior).
    output : {"volume", "boundary", "mask", "mesh"}, default "volume"
        volume/boundary: (M,3) regular world grid points. Boundary uses six
        neighbours and includes cavity walls; it retains grid stair steps.
        mask: bool (nx,ny,nz), avoids materialising M coordinates.
        mesh: (vertices, faces), both (*,3); vertices lie on an interpolated
        isosurface, not on grid nodes. No PyVista object is required.
    backend : {"auto", "pymcubes", "skimage", "torchmcubes", "lewiner", "lorensen"}
        Used only for output='mesh'. auto selects an available CPU extractor.
    mc_method : str, optional
        Legacy alias for backend; do not specify both.
    device : str, default "cpu"
        Mesh extraction device for torchmcubes. Grid construction and Gaussian
        smoothing remain CPU calculations.
    max_voxels : positive int, default 2_000_000_000
        Limit padded grid nodes; raise instead of silently changing spacing.
        Not a total RAM limit: EDT and filtering need multiple dense arrays.
    chunk_size : positive int, default 100_000
        Grid-query and KDE-deposition batch size.
    workers : int, default -1
        cKDTree query threads; -1 means all available CPUs.
    return_info : bool, default False
        Return (result, info). info contains mask, origin, spacing, direction,
        base_spacing and resolved smooth_sigma (also the legacy smooth key);
        shape is already available as mask.shape.
    return_field : bool, default False
        With return_info=True, also retain the final implicit field to reuse
        with grid_to_mesh, together with field_inside='negative'.

    Notes
    -----
    Arbitrary orientation/nonlinear-transformed coordinates are accepted, but
    one global axis_scale/direction cannot recover spatially varying former
    slice directions. Geometric priors cannot identify whether a large missing
    region is a genuine cavity. KDE is an approximate density estimator, knn
    support can jump when the adaptive kth-neighbour set changes, and all
    methods remain resolution/scale dependent. Only torchmcubes extraction
    can use a GPU; the point queries and fields remain on CPU.

    Memory: O(N*max(outlier_k,support_k) + V + chunk_size*support_k + M).
    Grid and smoothing are dense; only the point-component graph is sparse.
    """
    smooth_sigma = _alias(smooth_sigma, smooth, None, "smooth_sigma", "smooth")
    min_points = _alias(min_points, min_component, 0.01, "min_points", "min_component")
    graph_scale = _alias(graph_scale, component_scale, 2.0, "graph_scale", "component_scale")
    backend = _alias(backend, mc_method, "auto", "backend", "mc_method")
    raw = np.asarray(getattr(points, "points", points), dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError("points must have shape (N,3).")
    if method not in ("knn", "nearest", "kde") or output not in ("volume", "boundary", "mask", "mesh"):
        raise ValueError("Unknown reconstruction method or output.")
    if outlier_method not in ("lof", "sor", "radius", None) or smooth_method not in ("sdf", "occupancy", None):
        raise ValueError("Unknown outlier_method or smooth_method.")
    if backend not in {"auto", "pymcubes", "skimage", "torchmcubes", "lewiner", "lorensen"}:
        raise ValueError("Unknown marching-cubes backend.")
    for name, value, minimum in [("support_k", support_k, 1), ("outlier_k", outlier_k, 2),
                                  ("outlier_iters", outlier_iters, 0),
                                  ("chunk_size", chunk_size, 1), ("max_voxels", max_voxels, 1)]:
        if not np.isfinite(value) or value != int(value) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    support_k, outlier_k, outlier_iters, chunk_size = map(int, (support_k, outlier_k, outlier_iters, chunk_size))
    positive = [support_scale, graph_scale, radius_cap]
    positive += [v for v in (support_radius, outlier_radius) if v is not None]
    if (not np.isfinite(positive).all() or min(positive) <= 0 or radius_cap < 1
            or not np.isfinite([min_points, closing, opening]).all()
            or min(min_points, closing, opening) < 0 or not 0 < density_level < 1):
        raise ValueError("Invalid support, outlier, component, morphology, or density parameter.")
    if outlier_thresh is not None and (not np.isfinite(outlier_thresh) or outlier_thresh < 0):
        raise ValueError("outlier_thresh must be finite and nonnegative, or None.")
    if base_spacing is not None and (not np.isfinite(base_spacing) or base_spacing <= 0):
        raise ValueError("base_spacing must be a finite positive scalar.")
    direction = np.eye(3) if direction is None else np.asarray(direction, dtype=float)
    if direction.shape != (3, 3) or not np.allclose(direction.T @ direction, np.eye(3), atol=1e-7, rtol=0):
        raise ValueError("direction columns must be orthonormal.")
    axis_scale = _xyz(axis_scale, "axis_scale", positive=True)
    scales = np.atleast_1d(np.asarray(smooth_scales, dtype=float))
    if scales.ndim != 1 or not len(scales) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("smooth_scales must be a nonempty sequence of positive values.")

    x = np.unique(raw[np.isfinite(raw).all(axis=1)], axis=0)
    x = (x @ direction) / axis_scale
    x, tree, dist = _clean_points(
        x, outlier_k, support_k, outlier_method, outlier_thresh, outlier_iters,
        outlier_radius, min_points, keep_largest, graph_scale, radius_cap, workers)
    support_k = min(support_k, len(x) - 1)
    estimated_step = float(np.median(dist[:, support_k - 1]) / np.cbrt(support_k))
    if not np.isfinite(estimated_step) or estimated_step <= 0:
        raise ValueError("Point coordinates do not define a finite positive scale.")
    step = estimated_step if base_spacing is None else float(base_spacing)
    scale = _xyz(spacing_scale, "spacing_scale", True)
    spacing = _xyz(step * axis_scale * scale if spacing is None else spacing, "spacing", True)
    scale = _xyz(smooth_scale, "smooth_scale")
    smooth = _xyz(step * axis_scale * scale if smooth_sigma is None else smooth_sigma,
                   "smooth_sigma")
    bandwidth = step * axis_scale * _xyz(kernel_scale, "kernel_scale", True)
    max_shift = float(2 * smooth.max() * scales.max() if max_shift is None else max_shift)
    if not np.isfinite(max_shift) or max_shift < 0:
        raise ValueError("max_shift must be finite and nonnegative.")
    local = np.sqrt(np.mean(dist[:, :support_k] ** 2, axis=1))
    local *= step / estimated_step
    radii = np.minimum(local, radius_cap * np.median(local))
    if support_radius is not None:
        radii.fill(support_radius)
    radii *= support_scale
    del dist

    support_pad = 3 * bandwidth if method == "kde" else radii.max() * axis_scale
    active_smooth = smooth if smooth_method is not None and max_shift > 0 else np.zeros(3)
    padding = support_pad + 2 * closing * axis_scale + 3 * active_smooth * scales.max()
    padding += (max_shift if np.any(active_smooth) else 0) + 2 * spacing
    physical = x * axis_scale
    origin = np.floor((physical.min(axis=0) - padding) / spacing) * spacing
    dims = np.ceil((physical.max(axis=0) + padding - origin) / spacing) + 1
    size = float(np.prod(dims))
    if not np.isfinite(size) or size > max_voxels:
        raise MemoryError(f"Grid needs {size:,.0f} nodes; max_voxels={max_voxels:,}. Increase spacing or crop/filter points.")
    shape = tuple(dims.astype(np.int64))
    mask = _support_mask_2(x, tree, radii, origin, spacing, shape, axis_scale, method,
                         support_k, bandwidth, density_level, chunk_size, workers)
    del physical, x, tree, radii
    if not mask.any():
        raise ValueError("Empty support mask; reduce spacing or increase support.")

    # Exact EDT-based binary morphology in the chosen ellipsoidal metric.
    sampling = spacing / axis_scale
    if closing:
        mask = ndi.distance_transform_edt(~mask, sampling=sampling) <= closing
        mask = ndi.distance_transform_edt(mask, sampling=sampling) > closing
    if opening:
        mask = ndi.distance_transform_edt(mask, sampling=sampling) > opening
        if mask.any():
            mask = ndi.distance_transform_edt(~mask, sampling=sampling) <= opening
    if fill_holes:
        mask = ndi.binary_fill_holes(mask)
    if not mask.any():
        raise ValueError("Empty support mask; reduce spacing/opening or increase support.")
    field, mask = _smooth_field(mask, spacing, smooth, scales, smooth_method, max_shift, preserve_volume)
    info = dict(mask=mask, origin=origin @ direction.T, spacing=spacing,
                direction=direction.copy(), base_spacing=step,
                smooth_sigma=smooth, smooth=smooth)
    if output == "mesh":
        result = grid_to_mesh(dict(info, field=field, field_inside="negative"),
                              backend=backend, device=device)
    elif output == "mask":
        result = mask
    else:
        selected = mask if output == "volume" else mask & ~ndi.binary_erosion(mask)
        ijk = np.column_stack(np.nonzero(selected))
        result = info["origin"] + (ijk * spacing) @ direction.T
    if return_info:
        if return_field:
            info["field"] = field
            info["field_inside"] = "negative"
        return result, info
    return result

